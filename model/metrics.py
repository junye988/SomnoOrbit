from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def classification_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(5))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
        "per_class_recall": (
            np.diag(matrix) / np.maximum(1, matrix.sum(1))
        ).tolist(),
    }
