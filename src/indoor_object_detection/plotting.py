"""Reusable visualizations for indoor-object detection results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, sqrt
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from indoor_object_detection import CLASSES, ImageRecord


def _add_box_label(
    axis: Axes,
    text: str,
    box: Sequence[float],
    text_color: str,
    background_color: str,
    place_below: bool,
) -> None:
    """Place a label immediately above or below a box."""
    x1, y1, _, y2 = box
    anchor = (x1, y2) if place_below else (x1, y1)
    offset = (0, 5) if place_below else (0, -3)
    axis.annotate(
        text,
        xy=anchor,
        xytext=offset,
        textcoords="offset points",
        ha="left",
        va="top" if place_below else "bottom",
        color=text_color,
        fontsize=8,
        bbox={"facecolor": background_color, "alpha": 0.75, "pad": 1},
    )


def plot_ground_truth(
    records: Sequence[ImageRecord], n: int = 4, seed: int | None = None
) -> Figure:
    """Plot a random sample of images with their ground-truth boxes.

    Args:
        records: Annotated image records from which to sample.
        n: Number of images to include.
        seed: Optional seed for reproducible sampling.

    Returns:
        A Matplotlib figure that can be displayed or saved by the caller.

    Raises:
        ValueError: If ``n`` is not between one and the number of records.
    """
    if not 1 <= n <= len(records):
        raise ValueError(f"n must be between 1 and {len(records)}, got {n}")
    indices = np.random.default_rng(seed).choice(len(records), size=n, replace=False)
    sample = [records[int(index)] for index in indices]
    figure = Figure(figsize=(5 * n, 4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, n)
    axes_list = [axes] if n == 1 else list(axes)
    for axis, record in zip(axes_list, sample, strict=True):
        with Image.open(record.image_path) as source_image:
            image = source_image.convert("RGB")
        axis.imshow(image)
        for box in record.boxes:
            axis.add_patch(
                Rectangle(
                    (box.x1, box.y1),
                    box.x2 - box.x1,
                    box.y2 - box.y1,
                    fill=False,
                    color="lime",
                    linewidth=2,
                )
            )
            _add_box_label(
                axis,
                box.label,
                (box.x1, box.y1, box.x2, box.y2),
                text_color="black",
                background_color="lime",
                place_below=False,
            )
        axis.axis("off")
    figure.tight_layout()
    return figure


def plot_ranked_examples(
    examples: Sequence[Mapping[str, Any]],
    title: str,
    class_names: Sequence[str] = CLASSES,
    plot_width: float = 4.,
) -> Figure:
    """Plot scored predictions with ground-truth and predicted boxes.

    Ground-truth boxes are green and predictions are red. Each example must
    contain the keys produced by :func:`indoor_object_detection.evaluate.score_prediction`,
    plus ``image_path`` and the corresponding Ultralytics ``result``.

    Args:
        examples: Ranked per-image evaluation records to visualize.
        title: Figure title.
        class_names: Class names indexed by integer class ID.

    Returns:
        A Matplotlib figure that can be displayed or saved by the caller.

    Raises:
        ValueError: If no examples are supplied.
    """
    if not examples:
        raise ValueError("At least one example is required")
    columns = ceil(sqrt(len(examples)))
    rows = ceil(len(examples) / columns)
    figure = Figure(figsize=(plot_width * columns, 0.8 * rows * plot_width))
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    axes_list = list(axes.flat)
    for axis, item in zip(axes_list, examples):
        with Image.open(Path(item["image_path"])) as source_image:
            image = source_image.convert("RGB")
        axis.imshow(image)
        for class_id, box in item["ground_truth"]:
            axis.add_patch(
                Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    fill=False,
                    color="lime",
                    linewidth=2,
                )
            )
            _add_box_label(
                axis,
                f"GT {class_names[class_id]}",
                box,
                text_color="black",
                background_color="lime",
                place_below=False,
            )

        boxes = item["result"].boxes
        pred_boxes = boxes.xyxy.cpu().numpy() if boxes is not None else []
        pred_classes = boxes.cls.cpu().numpy().astype(int) if boxes is not None else []
        pred_confidences = boxes.conf.cpu().numpy() if boxes is not None else []
        for box, class_id, confidence in zip(
            pred_boxes, pred_classes, pred_confidences, strict=True
        ):
            axis.add_patch(
                Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    fill=False,
                    color="red",
                    linewidth=2,
                )
            )
            _add_box_label(
                axis,
                f"P {class_names[class_id]} {confidence:.2f}",
                box,
                text_color="white",
                background_color="red",
                place_below=True,
            )
        axis.set_title(
            f"{Path(item['image_path']).stem}\n"
            f"score={item['score']:.2f}, P={item['precision']:.2f}, "
            f"R={item['recall']:.2f}"
        )
        axis.axis("off")
    for axis in axes_list[len(examples) :]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure
