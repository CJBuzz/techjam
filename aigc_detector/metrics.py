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

