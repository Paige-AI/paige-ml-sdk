"""
Translators from sklearn metrics into objects that satisfy the sdk's metrics protocols.
"""

from typing import cast

from numpy._typing import NDArray
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def sklearn_accuracy_score(preds: NDArray, targets: NDArray) -> float:
    accuracy = accuracy_score(y_true=targets, y_pred=preds)
    return accuracy


def sklearn_balanced_accuracy_score(preds: NDArray, targets: NDArray) -> float:
    balanced_accuracy = cast(float, balanced_accuracy_score(y_true=targets, y_pred=preds))
    return balanced_accuracy


def sklearn_f1_score(preds: NDArray, targets: NDArray) -> float:
    f1 = cast(float, f1_score(y_true=targets, y_pred=preds))
    return f1


def sklearn_f1_weighted_score(preds: NDArray, targets: NDArray) -> float:
    f1_weighted = cast(float, f1_score(y_true=targets, y_pred=preds, average='weighted'))
    return f1_weighted


def sklearn_roc_auc_score(probs: NDArray, targets: NDArray) -> float:
    auc = cast(float, roc_auc_score(y_true=targets, y_score=probs))
    return auc
