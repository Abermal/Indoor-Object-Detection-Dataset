"""Tests for held-out detector evaluation helpers."""

from pathlib import Path

import numpy as np
import pytest

from indoor_object_detection.evaluate import (
    box_iou,
    resolve_path,
    yolo_label_to_xyxy,
)


def test_resolve_path_accepts_file_uri() -> None:
    resolved = resolve_path("file:D:/models/best.pt")

    assert resolved == Path("D:/models/best.pt")


def test_yolo_label_to_xyxy_converts_normalized_coordinates() -> None:
    class_id, box = yolo_label_to_xyxy([2, 0.5, 0.5, 0.4, 0.2], 100, 200)

    assert class_id == 2
    assert box == pytest.approx(np.array([30.0, 80.0, 70.0, 120.0]))


def test_box_iou() -> None:
    first = np.array([0.0, 0.0, 10.0, 10.0])
    second = np.array([5.0, 5.0, 15.0, 15.0])

    assert box_iou(first, second) == pytest.approx(25 / 175)
