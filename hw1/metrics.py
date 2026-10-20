"""Binary-classification metrics computed from raw logits."""

from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def binary_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """Accuracy, precision, recall, F1 and the confusion matrix.

    Computed from the logits directly so that no probability rounding happens
    before thresholding. Precision is defined as 0 when nothing is predicted
    positive, which is the convention scikit-learn uses with zero_division=0.
    """
    probabilities = torch.sigmoid(logits.float())
    predictions = (probabilities >= threshold).float()
    actual = targets.float()

    true_positive = float(((predictions == 1) & (actual == 1)).sum())
    false_positive = float(((predictions == 1) & (actual == 0)).sum())
    true_negative = float(((predictions == 0) & (actual == 0)).sum())
    false_negative = float(((predictions == 0) & (actual == 1)).sum())

    total = true_positive + false_positive + true_negative + false_negative
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "predicted_positive_rate": (true_positive + false_positive) / total if total else 0.0,
    }


@torch.no_grad()
def roc_auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Threshold-free ranking quality, via the rank-sum identity.

    Useful because two runs can share an accuracy at threshold 0.5 while
    ranking the examples very differently.
    """
    scores = logits.float().flatten()
    labels = targets.float().flatten()
    positives = int(labels.sum())
    negatives = int((1 - labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype, device=scores.device)
    # Average the ranks inside groups of equal score, so ties do not bias the result.
    unique_scores, inverse = scores.unique(return_inverse=True)
    if len(unique_scores) < len(scores):
        summed = torch.zeros(len(unique_scores), dtype=ranks.dtype)
        counts = torch.zeros(len(unique_scores), dtype=ranks.dtype)
        summed.index_add_(0, inverse, ranks)
        counts.index_add_(0, inverse, torch.ones_like(ranks))
        ranks = (summed / counts)[inverse]
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
