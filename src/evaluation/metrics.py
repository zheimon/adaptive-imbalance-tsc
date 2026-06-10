"""Evaluation metrics for class-imbalanced time-series classification."""
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 score."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy (mean per-class recall)."""
    return float(balanced_accuracy_score(y_true, y_pred))


def g_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Geometric mean of per-class recall.

    G-mean = (prod_c recall_c)^{1/K}, computed in log-space for stability.
    Classes with zero support are excluded.
    """
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        rec = float((y_pred[mask] == c).mean())
        recalls.append(rec)
    if not recalls:
        return 0.0
    log_sum = np.sum(np.log(np.maximum(recalls, 1e-12)))
    return float(np.exp(log_sum / len(recalls)))


def per_class_prf(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-class precision, recall, F1, and support.

    Returns
    -------
    pd.DataFrame with columns: ['class', 'precision', 'recall', 'f1', 'support']
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    prec = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    support = np.array([(y_true == c).sum() for c in classes])

    return pd.DataFrame(
        {
            "class": classes,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }
    )


def confusion_matrix_normalized(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Row-normalized confusion matrix (each row sums to 1)."""
    cm = confusion_matrix(y_true, y_pred).astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return cm / row_sums


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute all evaluation metrics in one call.

    Returns
    -------
    dict with keys:
        'macro_f1': float
        'balanced_accuracy': float
        'g_mean': float
        'per_class_prf': pd.DataFrame
        'confusion_matrix_normalized': np.ndarray
    """
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "g_mean": g_mean(y_true, y_pred),
        "per_class_prf": per_class_prf(y_true, y_pred),
        "confusion_matrix_normalized": confusion_matrix_normalized(y_true, y_pred),
    }
