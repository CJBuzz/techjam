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
