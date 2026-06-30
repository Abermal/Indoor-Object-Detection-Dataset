"""Train the indoor object detector and record the run in MLflow."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from indoor_object_detection import (
    DatasetSplit,
    ensure_dataset,
    make_splits,
    parse_dlib_annotations,
    write_yolo_dataset,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Validated command-line configuration for one training run."""

    model: str
    epochs: int
    image_size: int
    batch_size: int
    device: str
    workers: int
    seed: int
    dataset_root: Path
    yolo_dir: Path
    project: Path
    run_name: str
    experiment_name: str
    tracking_uri: str
    force_prepare: bool


def parse_args() -> TrainingConfig:
    """Parse command-line arguments into a statically typed configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt", help="model weights")
    parser.add_argument("--epochs", type=int, default=30, help="training epochs")
    parser.add_argument("--image-size", type=int, default=640, help="input size")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--device", default="cpu", help="CPU, GPU, or device ID")
    parser.add_argument("--workers", type=int, default=0, help="data workers")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("."), help="dataset search root"
    )
    parser.add_argument(
        "--yolo-dir",
        type=Path,
        default=Path("data/indoor_yolo"),
        help="prepared YOLO dataset directory",
    )
    parser.add_argument(
        "--project", type=Path, default=Path("runs/detect"), help="output root"
    )
    parser.add_argument("--run-name", default="indoor_yolo_mlflow", help="run name")
    parser.add_argument(
        "--experiment-name",
        default="indoor-object-detection",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--tracking-uri", default="sqlite:///mlflow.db", help="MLflow tracking URI"
    )
    parser.add_argument(
        "--force-prepare", action="store_true", help="rebuild prepared data"
    )
    return TrainingConfig(**vars(parser.parse_args()))


def prepare_dataset(args: TrainingConfig) -> Path:
    """Locate or prepare a YOLO dataset and return its YAML manifest."""
    data_yaml = args.yolo_dir / "data.yaml"
    if data_yaml.exists() and not args.force_prepare:
        LOGGER.info("Using prepared dataset at %s", data_yaml)
        return data_yaml

    dataset_root = ensure_dataset(args.dataset_root)
    LOGGER.info("Parsing annotations from %s", dataset_root)
    records = parse_dlib_annotations(dataset_root)
    splits = make_splits(records, seed=args.seed)
    for split in DatasetSplit:
        LOGGER.info("%s images: %d", split, len(splits[split]))
    return write_yolo_dataset(splits, args.yolo_dir) / "data.yaml"


def main() -> None:
    """Train the detector and log parameters, metrics, and artifacts to MLflow."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    from ultralytics import YOLO, settings

    data_yaml = prepare_dataset(args)
    os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment_name
    os.environ["MLFLOW_RUN"] = args.run_name
    settings.update({"mlflow": True})

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(args.project),
        name=args.run_name,
        exist_ok=True,
    )

    save_dir = Path(model.trainer.save_dir)
    LOGGER.info("Training completed; artifacts: %s", save_dir.resolve())


if __name__ == "__main__":
    main()
