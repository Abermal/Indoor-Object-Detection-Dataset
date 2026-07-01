"""Tests for typed dataset splitting and YOLO serialization."""

from pathlib import Path

import pytest

from indoor_object_detection.dataset import (
    CLASSES,
    DATASET_SPLITS,
    DatasetSplit,
    ImageRecord,
    ObjectBox,
    make_splits,
    write_yolo_dataset,
)


def _record(frame: int) -> ImageRecord:
    boxes = tuple(ObjectBox(label, 0, 0, 10, 10) for label in CLASSES)
    return ImageRecord(Path(f"frame_{frame}.jpg"), "1", boxes)


def _record_with_label(frame: int, label: str) -> ImageRecord:
    return ImageRecord(
        Path(f"frame_{frame}.jpg"),
        "1",
        (ObjectBox(label, 0, 0, 10, 10),),
    )


def test_dataset_splits_are_string_enums() -> None:
    assert DATASET_SPLITS == (
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    )
    assert DatasetSplit.TRAIN == "train"
    assert Path("images") / DatasetSplit.VAL == Path("images/val")


def test_make_splits_returns_enum_keys_and_preserves_blocks() -> None:
    splits = make_splits(
        [_record(frame) for frame in range(1, 21)],
        ratios=(0.6, 0.2, 0.2),
        temporal_block_size=2,
        max_attempts=20,
    )

    assert tuple(splits) == DATASET_SPLITS
    assert {split: len(items) for split, items in splits.items()} == {
        DatasetSplit.TRAIN: 12,
        DatasetSplit.VAL: 4,
        DatasetSplit.TEST: 4,
    }
    assigned_split = {
        record.image_path: split
        for split, split_records in splits.items()
        for record in split_records
    }
    for first_frame in range(1, 21, 2):
        assert assigned_split[Path(f"frame_{first_frame}.jpg")] == assigned_split[
            Path(f"frame_{first_frame + 1}.jpg")
        ]


def test_make_splits_balances_class_distribution() -> None:
    records = [
        _record_with_label(frame, label)
        for frame, label in enumerate(CLASSES * 10, start=1)
    ]

    splits = make_splits(
        records,
        ratios=(0.6, 0.2, 0.2),
        temporal_block_size=1,
        max_attempts=100,
    )

    expected_per_class = {
        DatasetSplit.TRAIN: 6,
        DatasetSplit.VAL: 2,
        DatasetSplit.TEST: 2,
    }
    for split, split_records in splits.items():
        class_counts = {
            label: sum(label in record.labels for record in split_records)
            for label in CLASSES
        }
        assert set(class_counts.values()) == {expected_per_class[split]}


@pytest.mark.parametrize(
    ("ratios", "message"),
    [
        ((0.8, 0.2), "three positive values"),
        ((0.8, 0.2, 0.0), "three positive values"),
        ((0.8, 0.15, 0.1), "sum to 1.0"),
    ],
)
def test_make_splits_rejects_invalid_ratios(
    ratios: tuple[float, ...],
    message: str,
) -> None:
    records = [_record(frame) for frame in range(1, 11)]
    with pytest.raises(ValueError, match=message):
        make_splits(records, ratios=ratios)  # type: ignore[arg-type]


def test_make_splits_rejects_an_empty_partition() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        make_splits([_record(1), _record(2)], ratios=(0.8, 0.1, 0.1))


def test_write_yolo_dataset_serializes_enum_split_names(tmp_path: Path) -> None:
    image_path = tmp_path / "frame_1.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xc0\x00\x08\x08\x00\x0a\x00\x14")
    record = ImageRecord(
        image_path,
        "1",
        (ObjectBox(CLASSES[0], 0, 0, 10, 10),),
    )
    splits = {split: [record] for split in DATASET_SPLITS}

    output = write_yolo_dataset(splits, tmp_path / "yolo")

    yaml_text = (output / "data.yaml").read_text(encoding="utf-8")
    for split in DATASET_SPLITS:
        assert f"{split}: images/{split}" in yaml_text
        assert (output / "images" / split).is_dir()
