"""Compatibility tests for Ultralytics MLflow logging."""

from typing import ClassVar

import pytest
from ultralytics.utils.callbacks import mlflow as mlflow_callbacks


def test_fit_epoch_callback_logs_validation_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the native callback logs validation values each epoch."""

    class Trainer:
        epoch: int = 2
        metrics: ClassVar[dict[str, float]] = {
            "val/box_loss": 1.5,
            "metrics/mAP50(B)": 0.25,
        }
        _mlflow_active: bool = True

    class Mlflow:
        calls: ClassVar[list[tuple[dict[str, float], int]]] = []

        @classmethod
        def log_metrics(
            cls,
            metrics: dict[str, float],
            step: int,
        ) -> None:
            cls.calls.append((metrics, step))

    monkeypatch.setattr(mlflow_callbacks, "mlflow", Mlflow)
    mlflow_callbacks.on_fit_epoch_end(Trainer())

    assert Mlflow.calls == [
        ({"val/box_loss": 1.5, "metrics/mAP50B": 0.25}, 2)
    ]
