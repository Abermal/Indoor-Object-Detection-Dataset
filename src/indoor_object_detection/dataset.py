"""Download, parse, split, summarize, and export the indoor dataset."""

from __future__ import annotations

import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, Iterable, Mapping
from xml.etree import ElementTree as ET

if TYPE_CHECKING:
    from pandas import DataFrame


CLASSES: Final[tuple[str, ...]] = (
    "chair",
    "clock",
    "exit",
    "fireextinguisher",
    "printer",
    "screen",
    "trashbin",
)

ZENODO_DOWNLOAD_URL: Final[str] = (
    "https://zenodo.org/api/records/2654485/files/"
    "Indoor%20Object%20Detection%20Dataset.zip/content"
)


class DatasetSplit(StrEnum):
    """Supported dataset partitions and their serialized directory names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


DATASET_SPLITS: Final[tuple[DatasetSplit, ...]] = tuple(DatasetSplit)


@dataclass(frozen=True)
class ObjectBox:
    """A labelled object bounding box in absolute pixel coordinates."""

    label: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class ImageRecord:
    """An annotated image and its source sequence."""

    image_path: Path
    sequence: str
    boxes: tuple[ObjectBox, ...]

    @property
    def labels(self) -> set[str]:
        """Return the distinct class labels attached to the image."""
        return {box.label for box in self.boxes}


def find_dataset_root(search_root: str | Path = ".") -> Path | None:
    """Return the unpacked dataset directory when it exists under search_root."""
    search_root = Path(search_root)
    candidates = [
        search_root / "Indoor Object Detection Dataset",
        search_root / "data" / "Indoor Object Detection Dataset",
        search_root,
    ]
    for candidate in candidates:
        if (candidate / "annotation").is_dir() and any(candidate.glob("sequence_*")):
            return candidate.resolve()
    return None


def download_dataset(destination: str | Path = ".") -> Path:
    """Download and unpack the Zenodo dataset when it is not already present."""
    destination = Path(destination)
    existing = find_dataset_root(destination)
    if existing is not None:
        return existing

    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / "Indoor Object Detection Dataset.zip"
    if not zip_path.exists():
        print(f"Downloading dataset to {zip_path} ...")
        urllib.request.urlretrieve(ZENODO_DOWNLOAD_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)

    dataset_root = find_dataset_root(destination)
    if dataset_root is None:
        raise FileNotFoundError(
            "Dataset archive extracted, but the dataset root was not found."
        )
    return dataset_root


def ensure_dataset(search_root: str | Path = ".") -> Path:
    """Find the local dataset or download it into search_root."""
    dataset_root = find_dataset_root(search_root)
    return dataset_root if dataset_root is not None else download_dataset(search_root)


def parse_dlib_annotations(dataset_root: str | Path) -> list[ImageRecord]:
    """Parse dlib XML annotations into image records with absolute image paths."""
    dataset_root = Path(dataset_root)
    records: list[ImageRecord] = []

    for xml_path in sorted((dataset_root / "annotation").glob("annotation_s*.xml")):
        sequence_id = xml_path.stem.split("_s")[-1]
        image_dir = dataset_root / f"sequence_{sequence_id}"
        tree = ET.parse(xml_path)

        for image_node in tree.findall(".//image"):
            filename = image_node.attrib["file"]
            boxes: list[ObjectBox] = []
            for box_node in image_node.findall("box"):
                label = (box_node.findtext("label") or "").strip()
                if label not in CLASSES:
                    raise ValueError(f"Unexpected class {label!r} in {xml_path}")
                left = float(box_node.attrib["left"])
                top = float(box_node.attrib["top"])
                width = float(box_node.attrib["width"])
                height = float(box_node.attrib["height"])
                boxes.append(ObjectBox(label, left, top, left + width, top + height))
            records.append(ImageRecord(image_dir / filename, sequence_id, tuple(boxes)))

    missing = [
        record.image_path for record in records if not record.image_path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} image files, first: {missing[0]}"
        )
    return records


def _split_sizes(
    total: int,
    ratios: tuple[float, float, float],
) -> dict[DatasetSplit, int]:
    """Convert split ratios into image counts that sum to ``total``."""
    train = round(total * ratios[0])
    val = round(total * ratios[1])
    test = total - train - val
    return {
        DatasetSplit.TRAIN: train,
        DatasetSplit.VAL: val,
        DatasetSplit.TEST: test,
    }


def _split_has_all_classes(split_records: Iterable[ImageRecord]) -> bool:
    """Return whether records collectively contain every supported class."""
    labels = set()
    for record in split_records:
        labels.update(record.labels)
    return set(CLASSES).issubset(labels)


def make_splits(
    records: list[ImageRecord],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    max_attempts: int = 2000,
    temporal_block_size: int = 5,
) -> dict[DatasetSplit, list[ImageRecord]]:
    """Divide annotated images into training, validation, and test sets.

    Consecutive images from the same recorded sequence are often nearly
    identical. Putting them in different sets could make test results look
    better than they really are, so this function keeps each group of
    consecutive images in one set. It tries several random group assignments
    and returns the one closest to the requested proportions that also includes
    every supported object class in every set. Among assignments with equally
    accurate set sizes, it prefers the one whose bounding-box counts for each
    class are closest to those proportions.

    Args:
        records: Annotated images to divide. Each item contains the image file
            path, the recorded sequence it came from, and its labelled object
            bounding boxes.
        ratios: Fractions of the images requested for the training, validation,
            and test sets, in that order. The three values must sum to ``1.0``.
        seed: Starting value for the random assignment. Using the same value
            with the same inputs produces the same result.
        max_attempts: Maximum number of random assignments to try while looking
            for one that places every object class in every set.
        temporal_block_size: Number of consecutive images from a recorded
            sequence that must stay together in the same set.
    """
    if len(ratios) != len(DATASET_SPLITS) or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must contain three positive values")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios must sum to 1.0")
    if temporal_block_size < 1:
        raise ValueError("temporal_block_size must be at least 1")

    target_sizes = _split_sizes(len(records), ratios)
    if any(size == 0 for size in target_sizes.values()):
        raise ValueError(
            "records and ratios must allocate at least one image to every split"
        )

    # Each class should have approximately the requested fraction of its boxes
    # in each set, not merely be present there at least once.
    total_class_counts = Counter(
        box.label for record in records for box in record.boxes
    )
    target_class_counts = {
        split: {
            label: total_class_counts[label] * ratio for label in CLASSES
        }
        for split, ratio in zip(DATASET_SPLITS, ratios, strict=True)
    }

    # The numeric suffix in each image filename is its position in a recorded
    # sequence. Grouping adjacent numbers keeps near-identical images together.
    grouped_records: dict[tuple[str, int], list[ImageRecord]] = {}
    for record in records:
        frame_number = int(record.image_path.stem.rsplit("_", 1)[1])
        group_key = (record.sequence, (frame_number - 1) // temporal_block_size)
        grouped_records.setdefault(group_key, []).append(record)
    base_groups = list(grouped_records.values())
    best_valid: tuple[
        tuple[int, float], dict[DatasetSplit, list[ImageRecord]]
    ] | None = None

    # Different shuffles can produce different results because whole groups,
    # rather than individual images, must be assigned as indivisible units.
    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        groups = list(base_groups)
        rng.shuffle(groups)
        splits: dict[DatasetSplit, list[ImageRecord]] = {
            split: [] for split in DATASET_SPLITS
        }
        split_class_counts: dict[DatasetSplit, Counter[str]] = {
            split: Counter() for split in DATASET_SPLITS
        }

        for group in groups:
            group_class_counts = Counter(
                box.label for record in group for box in record.boxes
            )

            def assignment_score(split: DatasetSplit) -> float:
                """Return how useful it is to place the current group in a set.

                A set scores higher when it still needs images and boxes of the
                classes found in this group. Exceeding either the requested set
                size or a class's requested box count receives a large penalty.
                A tiny random tie-breaker lets later attempts explore equivalent
                assignments while remaining reproducible from ``seed``.
                """
                remaining = target_sizes[split] - len(splits[split])
                image_need = remaining / target_sizes[split]
                image_overflow = max(0, len(group) - remaining)

                class_need = 0.0
                class_overflow = 0.0
                boxes_in_group = sum(group_class_counts.values())
                for label, count in group_class_counts.items():
                    target = target_class_counts[split][label]
                    if target == 0:
                        continue
                    remaining_for_class = target - split_class_counts[split][label]
                    class_need += count * (remaining_for_class / target)
                    class_overflow += max(0.0, count - remaining_for_class) / target
                if boxes_in_group:
                    class_need /= boxes_in_group

                return (
                    image_need
                    + class_need
                    - (image_overflow * 10)
                    - (class_overflow * 10)
                    + (rng.random() * 1e-6)
                )

            selected_split = max(DATASET_SPLITS, key=assignment_score)
            splits[selected_split].extend(group)
            split_class_counts[selected_split].update(group_class_counts)

        if all(_split_has_all_classes(splits[split]) for split in DATASET_SPLITS):
            size_error = sum(
                abs(len(splits[split]) - target_sizes[split])
                for split in DATASET_SPLITS
            )
            class_error = sum(
                abs(
                    split_class_counts[split][label]
                    - target_class_counts[split][label]
                )
                / total_class_counts[label]
                for split in DATASET_SPLITS
                for label in CLASSES
                if total_class_counts[label]
            )
            quality = (size_error, class_error)

            # Set size is the primary requirement; class balance breaks ties
            # between assignments with the same size error.
            if best_valid is None or quality < best_valid[0]:
                best_valid = (quality, splits)
            if quality == (0, 0.0):
                return splits

    if best_valid is not None:
        return best_valid[1]
    raise RuntimeError(
        "Could not build a temporal split containing every class in train/val/test."
    )


def summarize_splits(
    splits: Mapping[DatasetSplit, list[ImageRecord]], norm : bool = False,
) -> "DataFrame":
    """Return image and object counts per split as a pandas DataFrame."""
    import pandas as pd

    rows = []
    for split, records in splits.items():
        row = {
            "split": split,
            "images": len(records),
            "boxes": sum(len(record.boxes) for record in records),
        }
        class_counts = Counter(box.label for record in records for box in record.boxes)
        row.update({label: class_counts[label] for label in CLASSES})
        rows.append(row)

    distribution = pd.DataFrame(rows).set_index("split")
    if norm:
        class_columns = list(CLASSES)
        distribution[class_columns] = (
            distribution[class_columns]
            .div(distribution["boxes"].replace(0, 1), axis=0)
            .mul(100)
            .round(2)
        )
        distribution = distribution.rename(
            columns={class_name: f"{class_name}_%" for class_name in CLASSES}
        )
    return distribution



def _yolo_line(box: ObjectBox, image_width: int, image_height: int) -> str:
    """Serialize one bounding box in normalized YOLO label format."""
    x1 = max(0.0, min(float(image_width), box.x1))
    y1 = max(0.0, min(float(image_height), box.y1))
    x2 = max(0.0, min(float(image_width), box.x2))
    y2 = max(0.0, min(float(image_height), box.y2))
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = max(0.0, x2 - x1) / image_width
    height = max(0.0, y2 - y1) / image_height
    class_id = CLASSES.index(box.label)
    return f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"


def _jpeg_size(image_path: Path) -> tuple[int, int]:
    """Read JPEG dimensions without importing image libraries."""
    with image_path.open("rb") as file:
        data = file.read(2)
        if data != b"\xff\xd8":
            raise ValueError(f"{image_path} is not a JPEG file")
        while True:
            marker_start = file.read(1)
            while marker_start and marker_start != b"\xff":
                marker_start = file.read(1)
            marker = file.read(1)
            while marker == b"\xff":
                marker = file.read(1)
            if not marker:
                break
            marker_code = marker[0]
            if marker_code in {0xD8, 0xD9}:
                continue
            segment_length = int.from_bytes(file.read(2), "big")
            if marker_code in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                file.read(1)
                height = int.from_bytes(file.read(2), "big")
                width = int.from_bytes(file.read(2), "big")
                return width, height
            file.seek(segment_length - 2, 1)
    raise ValueError(f"Could not read JPEG dimensions from {image_path}")


def write_yolo_dataset(
    splits: Mapping[DatasetSplit, list[ImageRecord]],
    output_dir: str | Path = "data/indoor_yolo",
    overwrite: bool = True,
) -> Path:
    """Convert split records to the YOLO directory format used by Ultralytics."""
    output_dir = Path(output_dir)
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for raw_split, records in splits.items():
        split = DatasetSplit(raw_split)
        image_out = output_dir / "images" / split
        label_out = output_dir / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for index, record in enumerate(records, start=1):
            if index == 1 or index == len(records) or index % 250 == 0:
                print(f"Writing {split}: {index}/{len(records)}")
            target_name = f"s{record.sequence}_{record.image_path.name}"
            shutil.copy2(record.image_path, image_out / target_name)
            width, height = _jpeg_size(record.image_path)
            label_lines = [_yolo_line(box, width, height) for box in record.boxes]
            (label_out / Path(target_name).with_suffix(".txt").name).write_text(
                "\n".join(label_lines),
                encoding="utf-8",
            )

    split_paths = {
        split: (Path("images") / split).as_posix() for split in DATASET_SPLITS
    }
    yaml_text = "\n".join(
        [
            f"path: {output_dir.resolve()}",
            *[f"{split}: {split_paths[split]}" for split in DATASET_SPLITS],
            "names:",
            *[f"  {idx}: {name}" for idx, name in enumerate(CLASSES)],
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    return output_dir
