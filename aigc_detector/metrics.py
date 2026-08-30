from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    labels: torch.Tensor, probabilities: torch.Tensor, threshold: float = 0.5
) -> dict[str, float | int | list[list[int]]]:
    y = labels.detach().cpu().numpy().astype(int)
    p = probabilities.detach().cpu().numpy()
    predictions = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    sensitivity = float(tp / (tp + fn)) if fn + tp else float("nan")
    return {
        "sample_count": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float((specificity + sensitivity) / 2),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": sensitivity,
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "specificity": specificity,
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else float("nan"),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def select_threshold(labels: torch.Tensor, probabilities: torch.Tensor, objective: str = "balanced") -> float:
    """Select a validation-only threshold for balanced accuracy or F1."""
    y = labels.detach().cpu().numpy().astype(int)
    p = probabilities.detach().cpu().numpy()
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        predictions = (p >= threshold).astype(int)
        if objective == "f1":
            score = f1_score(y, predictions, zero_division=0)
        elif objective == "balanced":
            tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if tn + fp else 0.0
            sensitivity = tp / (tp + fn) if tp + fn else 0.0
            score = (specificity + sensitivity) / 2
        else:
            raise ValueError(f"Unknown threshold objective {objective!r}")
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Positive scalar temperature optimized on a held-out validation split."""
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 20.0))


def expected_calibration_error(
    labels: torch.Tensor, probabilities: torch.Tensor, bins: int = 15
) -> float:
    """Equal-width expected calibration error for binary probabilities."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    y = labels.detach().cpu().float()
    p = probabilities.detach().cpu().float()
    edges = torch.linspace(0.0, 1.0, bins + 1)
    total = max(1, len(y))
    error = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (p < edges[index + 1])
        if index == bins - 1:
            mask |= p == 1.0
        if mask.any():
            error += float(mask.sum()) / total * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return error


def operational_thresholds(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    recall_target: float = 0.95,
    precision_target: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Choose high-recall and high-precision thresholds on calibration data only."""
    if not 0 < recall_target <= 1 or not 0 < precision_target <= 1:
        raise ValueError("precision and recall targets must be in (0, 1]")
    y = labels.detach().cpu().numpy().astype(bool)
    p = probabilities.detach().cpu().numpy()
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    rows = []
    for threshold in candidates:
        predicted = p >= threshold
        tp = int(np.logical_and(predicted, y).sum())
        fp = int(np.logical_and(predicted, ~y).sum())
        fn = int(np.logical_and(~predicted, y).sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append((float(threshold), precision, recall))
    recall_rows = [row for row in rows if row[2] >= recall_target]
    precision_rows = [row for row in rows if row[1] >= precision_target]
    high_recall = max(recall_rows, key=lambda row: (row[0], row[1]))
    high_precision = max(precision_rows, key=lambda row: (row[2], -row[0]))
    return {
        "high_recall": {
            "threshold": high_recall[0], "precision": high_recall[1], "recall": high_recall[2],
            "target_recall": recall_target,
        },
        "high_precision": {
            "threshold": high_precision[0], "precision": high_precision[1], "recall": high_precision[2],
            "target_precision": precision_target,
        },
    }
