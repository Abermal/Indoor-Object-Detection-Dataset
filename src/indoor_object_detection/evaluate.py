"""Evaluate a trained Ultralytics detector on the held-out test partition.

The module performs two complementary evaluations:

1. ``YOLO.val`` calculates dataset-level and per-class precision, recall, and
   COCO-style mAP metrics. It also creates Ultralytics' diagnostic plots,
   including the confusion matrix and precision-recall curves.
2. ``YOLO.predict`` scores each test image using greedy, class-aware box
   matching. Those scores are intended for ranking qualitative examples; they
   are not a replacement for dataset-level average precision.

The resulting CSV files and plots are written beneath ``--output-dir``. Run
``evaluate-indoor-detector --help`` for command-line usage.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from ultralytics import YOLO

from indoor_object_detection import DatasetSplit
from indoor_object_detection.plotting import plot_ranked_examples

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_path(value: str) -> Path:
    """Convert a filesystem path or ``file:`` URI to a local path.

    Both ``file:D:/models/best.pt`` and ``file:///D:/models/best.pt`` are
    accepted on Windows. Percent-encoded characters and UNC authorities are
    decoded. The returned path is not required to exist; callers decide
    whether existence is mandatory.

    Args:
        value: Native filesystem path or local ``file:`` URI.

    Returns:
        An expanded local path. A leading home-directory marker is expanded.

    Raises:
        ValueError: If ``value`` uses a URI scheme other than ``file``.
    """
    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme.lower() != "file":
        raise ValueError(f"Expected a local path or file URI, got {value!r}")
    if parsed.scheme.lower() != "file":
        return Path(value).expanduser()

    uri_path = unquote(parsed.path)
    if parsed.netloc:
        uri_path = f"//{parsed.netloc}{uri_path}"
    if os.name == "nt" and len(uri_path) >= 3 and uri_path[0] == "/" and uri_path[2] == ":":
        uri_path = uri_path[1:]
    return Path(uri_path).expanduser()


def yolo_label_to_xyxy(
    label_row: Sequence[float], image_width: int, image_height: int
) -> tuple[int, NDArray[np.float64]]:
    """Convert a normalized YOLO label to pixel corner coordinates.

    Args:
        label_row: Five values in ``class_id x_center y_center width height``
            order. Coordinates and dimensions must be normalized to ``[0, 1]``.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        The integer class ID and a float array in ``x1, y1, x2, y2`` order.

    Raises:
        ValueError: If ``label_row`` does not contain exactly five values.
    """
    class_id, x_center, y_center, width, height = label_row
    return int(class_id), np.array(
        [
            (x_center - width / 2) * image_width,
            (y_center - height / 2) * image_height,
            (x_center + width / 2) * image_width,
            (y_center + height / 2) * image_height,
        ],
        dtype=float,
    )


def box_iou(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Calculate intersection over union for two corner-format boxes.

    Args:
        a: First box as ``[x1, y1, x2, y2]`` pixel coordinates.
        b: Second box as ``[x1, y1, x2, y2]`` pixel coordinates.

    Returns:
        Intersection over union in ``[0, 1]``. Degenerate boxes whose union has
        zero area return ``0.0``.
    """
    intersection_width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection_height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def read_yolo_labels(
    label_path: Path, image_width: int, image_height: int
) -> list[tuple[int, NDArray[np.float64]]]:
    """Read a YOLO label file and convert its boxes to pixel coordinates.

    Missing and empty label files represent images without annotated objects
    and therefore return an empty list.

    Args:
        label_path: YOLO text file containing one object per line.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        ``(class_id, xyxy_box)`` pairs in file order.

    Raises:
        ValueError: If a non-empty row does not contain five numeric values.
    """
    if not label_path.exists():
        return []
    contents = label_path.read_text(encoding="utf-8").strip()
    if not contents:
        return []
    labels: list[tuple[int, NDArray[np.float64]]] = []
    for line in contents.splitlines():
        values = [float(value) for value in line.split()]
        if len(values) != 5:
            raise ValueError(f"Expected 5 values in {label_path}, got {len(values)}")
        labels.append(yolo_label_to_xyxy(values, image_width, image_height))
    return labels


def score_prediction(
    result: Any, label_path: Path, iou_threshold: float = 0.5
) -> dict[str, Any]:
    """Calculate class-aware detection metrics for one image.

    Each ground-truth box is greedily paired with the unused prediction of the
    same class that has the highest IoU. A pair is a true positive only when its
    IoU reaches ``iou_threshold``. Unmatched predictions are false positives;
    unmatched ground-truth boxes are false negatives.

    The ranking score mirrors the notebook and is defined as
    ``0.4 * precision + 0.4 * recall + 0.2 * mean_matched_iou``. It is a
    qualitative ranking heuristic, not a standard detector metric. For images
    without ground truth, recall is defined as ``1.0``.

    Args:
        result: One Ultralytics prediction result. It must expose ``orig_shape``
            and ``boxes``.
        label_path: YOLO ground-truth label file corresponding to the image.
        iou_threshold: Minimum IoU for a same-class pair to count as a match.

    Returns:
        A mapping containing the ranking score, precision, recall, F1,
        mean matched IoU, TP/FP/FN counts, object counts, and decoded
        ground-truth boxes used by the plotting code.
    """
    image_height, image_width = result.orig_shape
    ground_truth = read_yolo_labels(label_path, image_width, image_height)
    boxes = result.boxes
    pred_boxes = boxes.xyxy.cpu().numpy() if boxes is not None else np.empty((0, 4))
    pred_classes = (
        boxes.cls.cpu().numpy().astype(int)
        if boxes is not None
        else np.empty((0,), dtype=int)
    )
    used_predictions: set[int] = set()
    matched_ious: list[float] = []

    for gt_class, gt_box in ground_truth:
        candidates = [
            (index, box_iou(gt_box, pred_box))
            for index, pred_box in enumerate(pred_boxes)
            if index not in used_predictions and pred_classes[index] == gt_class
        ]
        if candidates:
            index, iou = max(candidates, key=lambda item: item[1])
            if iou >= iou_threshold:
                used_predictions.add(index)
                matched_ious.append(iou)

    true_positives = len(matched_ious)
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision = true_positives / len(pred_boxes) if len(pred_boxes) else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    score = 0.4 * precision + 0.4 * recall + 0.2 * mean_iou
    return {
        "score": score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "ground_truth_count": len(ground_truth),
        "prediction_count": len(pred_boxes),
        "ground_truth": ground_truth,
    }


def per_class_table(metrics: Any) -> pd.DataFrame:
    """Convert Ultralytics class-level box metrics into a table.

    Args:
        metrics: Return value from ``YOLO.val`` with ``names`` and ``box``
            attributes.

    Returns:
        A table sorted by class ID with precision, recall, mAP50, and
        mAP50-95 columns.
    """
    rows: list[dict[str, Any]] = []
    names = metrics.names
    for class_id, class_name in names.items():
        precision, recall, map50, map5095 = metrics.box.class_result(int(class_id))
        rows.append(
            {
                "class_id": int(class_id),
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "mAP50": map50,
                "mAP50-95": map5095,
            }
        )
    return pd.DataFrame(rows).sort_values("class_id")


def evaluate(
    weights: Path,
    data_yaml: Path,
    output_dir: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    device: str | None,
    confidence: float,
    nms_iou: float,
    match_iou: float,
    examples: int,
) -> None:
    """Evaluate one checkpoint on the prepared test split and write artifacts.

    The prepared dataset must use the layout produced by
    :func:`indoor_object_detection.write_yolo_dataset`: ``data_yaml`` beside
    ``images/test`` and ``labels/test``. Existing files in ``output_dir`` may be
    overwritten because the evaluation is run with ``exist_ok=True``.

    Written artifacts include:

    - ``overall_metrics.csv`` with dataset-level precision, recall, and mAP;
    - ``per_class_metrics.csv`` with one row per class;
    - ``per_image_metrics.csv`` with heuristic TP/FP/FN and IoU scores;
    - ``best_examples.png`` and ``worst_examples.png``; and
    - native Ultralytics plots such as confusion matrices and PR curves.

    Args:
        weights: Trained Ultralytics checkpoint.
        data_yaml: Manifest for the prepared YOLO dataset.
        output_dir: Destination for CSV files and plots.
        image_size: Square inference size passed to Ultralytics.
        batch_size: Validation batch size. Per-image prediction remains
            streamed to limit memory usage.
        workers: Number of validation data-loader workers. Use zero to disable
            multiprocessing, which is the safest setting on Windows.
        device: Ultralytics device selector, such as ``"cpu"``, ``"0"``, or
            ``None`` for automatic selection.
        confidence: Minimum confidence for a predicted box to be included in
            per-image metrics and plots. Does not affect ``YOLO.val`` metrics.
        nms_iou: Overlap threshold used to remove duplicate boxes during the
            separate per-image prediction pass. Higher values retain more
            overlapping boxes. This does not alter ``YOLO.val`` metrics.
        match_iou: A predicted box counts as a per-image true positive only if
            it has the correct class and at least this IoU with an unmatched
            ground-truth box.
        examples: Maximum number of best and worst images in each montage.

    Raises:
        FileNotFoundError: If the checkpoint, manifest, prepared test folders,
            or test images do not exist.
        ValueError: If ``examples`` is less than one.
    """
    weights = weights.resolve(strict=True)
    data_yaml = data_yaml.resolve(strict=True)
    if examples < 1:
        raise ValueError(f"examples must be at least 1, got {examples}")
    images_dir = data_yaml.parent / "images" / DatasetSplit.TEST
    labels_dir = data_yaml.parent / "labels" / DatasetSplit.TEST
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected prepared test images and labels under {data_yaml.parent}"
        )
    image_paths = sorted(
        path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {images_dir}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    validation_args: dict[str, Any] = {
        "data": str(data_yaml),
        "split": DatasetSplit.TEST,
        "imgsz": image_size,
        "batch": batch_size,
        "workers": workers,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": True,
        "plots": True,
    }
    if device is not None:
        validation_args["device"] = device
    LOGGER.info("Evaluating %s on the test split", weights)
    metrics = model.val(**validation_args)

    overall = pd.DataFrame(
        [
            {
                "precision": metrics.box.mp,
                "recall": metrics.box.mr,
                "mAP50": metrics.box.map50,
                "mAP75": metrics.box.map75,
                "mAP50-95": metrics.box.map,
            }
        ]
    )
    overall.to_csv(output_dir / "overall_metrics.csv", index=False)
    per_class_table(metrics).to_csv(output_dir / "per_class_metrics.csv", index=False)

    prediction_args: dict[str, Any] = {
        "source": [str(path) for path in image_paths],
        "imgsz": image_size,
        "conf": confidence,
        "iou": nms_iou,
        "stream": True,
        "verbose": False,
    }
    if device is not None:
        prediction_args["device"] = device
    ranked: list[dict[str, Any]] = []
    for image_path, result in zip(
        image_paths, model.predict(**prediction_args), strict=True
    ):
        relative_path = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_path.with_suffix(".txt")
        item = score_prediction(result, label_path, iou_threshold=match_iou)
        item.update(
            {
                "image": str(relative_path),
                "image_path": image_path,
                "label_path": label_path,
                "result": result,
            }
        )
        ranked.append(item)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    excluded_columns = {"ground_truth", "image_path", "label_path", "result"}
    per_image = pd.DataFrame(
        [
            {key: value for key, value in item.items() if key not in excluded_columns}
            for item in ranked
        ]
    )
    per_image.to_csv(output_dir / "per_image_metrics.csv", index=False)
    example_count = min(examples, len(ranked))
    best_figure = plot_ranked_examples(ranked[:example_count], "Best test examples")
    best_figure.savefig(
        output_dir / "best_examples.png", dpi=160, bbox_inches="tight"
    )
    best_figure.clear()
    worst_figure = plot_ranked_examples(ranked[-example_count:], "Worst test examples")
    worst_figure.savefig(
        output_dir / "worst_examples.png", dpi=160, bbox_inches="tight"
    )
    worst_figure.clear()
    LOGGER.info("Wrote evaluation artifacts to %s", output_dir)
    print(overall.to_string(index=False))


def main() -> None:
    """Parse command-line arguments and run test-set evaluation."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: evaluate-indoor-detector "
            "mlruns/1/<run-id>/artifacts/weights/best.pt --device 0"
        ),
    )
    inputs = parser.add_argument_group("model and dataset")
    inputs.add_argument(
        "weights",
        help=(
            "trained Ultralytics .pt checkpoint to evaluate; accepts a normal "
            "path or a local file: URI"
        ),
    )
    inputs.add_argument(
        "--data",
        default="data/indoor_yolo/data.yaml",
        help=(
            "YOLO data.yaml manifest; its test entry selects the images used "
            "for aggregate metrics and per-image scoring"
        ),
    )

    runtime = parser.add_argument_group("evaluation runtime")
    runtime.add_argument(
        "--output-dir",
        default="runs/detect/indoor_yolo_test",
        help=(
            "destination for CSV files, confusion matrices, Ultralytics plots, "
            "and best/worst montages; same-named files are overwritten"
        ),
    )
    runtime.add_argument(
        "--image-size",
        type=int,
        default=1280,
        help=(
            "image size passed to Ultralytics; each input is resized and "
            "letterboxed to this square inference size"
        ),
    )
    runtime.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "images processed together by YOLO.val; reduce this if aggregate "
            "validation runs out of GPU memory"
        ),
    )
    runtime.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "processes used to load YOLO.val batches; 0 loads in the main "
            "process and avoids multiprocessing issues on Windows"
        ),
    )
    runtime.add_argument(
        "--device",
        help=(
            "hardware for validation and prediction: cpu, a CUDA index such as "
            "0, or a GPU list such as 0,1; omit for automatic selection"
        ),
    )

    ranking = parser.add_argument_group("per-image ranking and plots")
    ranking.add_argument(
        "--prediction-confidence",
        "--confidence",
        dest="confidence",
        type=float,
        default=0.25,
        metavar="0..1",
        help=(
            "YOLO gives every candidate box a confidence score from 0 (weak "
            "evidence) to 1 (strong evidence). Treat only candidates at or "
            "above this value as detections in per-image metrics and plots. "
            "This prevents weak proposals from becoming false positives and "
            "cluttering the montages. It does not change YOLO.val metrics"
        ),
    )
    ranking.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        metavar="0..1",
        help=(
            "in the per-image prediction pass, suppress a lower-confidence "
            "duplicate when overlap exceeds this IoU; higher values retain "
            "more overlapping boxes; does not change YOLO.val metrics"
        ),
    )
    ranking.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        metavar="0..1",
        help=(
            "count a box as a per-image true positive only when its class is "
            "correct and its IoU with an unmatched ground-truth box reaches "
            "this value; unmatched predictions/labels become FP/FN"
        ),
    )
    ranking.add_argument(
        "--examples",
        type=int,
        default=9,
        help=(
            "images in each montage: the N highest-ranked images go to "
            "best_examples.png and the N lowest to worst_examples.png"
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    evaluate(
        weights=resolve_path(args.weights),
        data_yaml=resolve_path(args.data),
        output_dir=resolve_path(args.output_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        match_iou=args.match_iou,
        examples=args.examples,
    )


if __name__ == "__main__":
    main()
