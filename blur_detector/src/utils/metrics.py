from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: np.ndarray

    def __str__(self) -> str:
        return (
            f"Acc={self.accuracy:.4f}  P={self.precision:.4f}  "
            f"R={self.recall:.4f}  F1={self.f1:.4f}  AUC={self.roc_auc:.4f}\n"
            f"Confusion:\n{self.confusion}"
        )


def compute_metrics(labels: List[int], preds: List[int], probs_blurred: List[float]) -> EvalMetrics:
    return EvalMetrics(
        accuracy=accuracy_score(labels, preds),
        precision=precision_score(labels, preds, pos_label=1, zero_division=0),
        recall=recall_score(labels, preds, pos_label=1, zero_division=0),
        f1=f1_score(labels, preds, pos_label=1, zero_division=0),
        roc_auc=roc_auc_score(labels, probs_blurred),
        confusion=confusion_matrix(labels, preds),
    )
