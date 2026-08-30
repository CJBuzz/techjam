from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score


def classification_metrics(labels: torch.Tensor, probabilities: torch.Tensor) -> dict[str, float]:
    y = labels.detach().cpu().numpy().astype(int)
    p = probabilities.detach().cpu().numpy()
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


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
