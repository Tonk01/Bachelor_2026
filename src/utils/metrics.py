from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EventMAEResult:
    mae_position: float
    n_valid: int


def validate_1d_array(x: np.ndarray | list[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1d")

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or INF")

    return arr


def target_map_mae(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    true = validate_1d_array(y_true, "y_true")
    pred = validate_1d_array(y_pred, "y_pred")

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    return float(np.mean(np.abs(true - pred)))


def predicted_index_from_target(y_pred: np.ndarray | list[float], min_confidence: float = 0.0) -> int | None:
    pred = validate_1d_array(y_pred, "y_pred")

    confidence = float(np.max(pred))

    if confidence < min_confidence:
        return None

    return int(np.argmax(pred))


def event_position_mae(true_indices: list[int | None], pred_indices: list[int | None]) -> EventMAEResult:
    if len(true_indices) != len(pred_indices):
        raise ValueError("true_indices and pred_indices must have the same length")

    errors: list[float] = []

    for true_idx, pred_idx in zip(true_indices, pred_indices):
        if true_idx is None or pred_idx is None:
            continue

        errors.append(abs(float(pred_idx) - float(true_idx)))

    if not errors:
        return EventMAEResult(
            mae_position=0.0,
            n_valid=0,
        )

    return EventMAEResult(
        mae_position=float(np.mean(errors)),
        n_valid=len(errors),
    )


def event_target_position_mae(true_indices: list[int | None],pred_targets: list[np.ndarray], min_confidence: float = 0.0) -> EventMAEResult:
    pred_indices = [predicted_index_from_target(pred, min_confidence=min_confidence) for pred in pred_targets]

    return event_position_mae(
        true_indices=true_indices,
        pred_indices=pred_indices,
    )