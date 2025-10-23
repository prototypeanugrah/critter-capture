"""Multi-class classification metrics helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: Dict[str, Dict[str, float]]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
) -> ClassificationMetrics:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    per_class = {}
    for idx, name in enumerate(label_names):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[idx],
            average="binary",
            zero_division=0,
        )
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return ClassificationMetrics(
        accuracy=float(accuracy),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        per_class=per_class,
    )


__all__ = ["ClassificationMetrics", "compute_classification_metrics"]

