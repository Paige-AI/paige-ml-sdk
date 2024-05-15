from dataclasses import dataclass
from typing import Dict, List, Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import classification_report, multilabel_confusion_matrix, roc_auc_score

from paige.ml_sdk.array_procedures import enforce_1d_shape, enforce_2d_shape
from paige.ml_sdk.enforce_type import enforce_type

MetricName = str
MetricsOutput = Dict[MetricName, float]


class PredsMulticlassTargetsMetric(Protocol):
    def __call__(self, preds: NDArray[np.int_], targets: NDArray[np.int_]) -> float:
        ...


class ProbsMulticlassTargetsMetric(Protocol):
    def __call__(self, probs: NDArray[np.float_], targets: NDArray[np.int_]) -> float:
        ...


class ConfusionMatrixMulticlassTargetsMetric(Protocol):
    def __call__(self, cm: NDArray[np.float_]) -> float:
        ...


class ProbsToPreds(Protocol):
    def __call__(self, probs: NDArray[np.float_]) -> NDArray[np.int_]:
        ...


class PredsAndTargetsToConfusionMatrix(Protocol):
    def __call__(self, preds: NDArray[np.int_], targets: NDArray[np.int_]) -> NDArray[np.float_]:
        ...


@dataclass
class MulticlassClassificationMetricsData:
    """
    Data required for metrics computation.

    Args:
        probs: model probabilities.
        targets: ground truth labels.
    """

    probs: NDArray[np.float_]
    targets: NDArray[np.int_]

    def __post_init__(self) -> None:
        self.probs = enforce_2d_shape(enforce_type(np.ndarray, self.probs))
        self.targets = enforce_1d_shape(enforce_type(np.ndarray, self.targets))


def probs_to_preds_using_argmax(probs: NDArray[np.float_]) -> NDArray[np.int_]:
    return cast(NDArray[np.int_], np.argmax(probs, axis=1))


def preds_and_targets_to_cm(preds: NDArray[np.int_], targets: NDArray) -> NDArray:
    return multilabel_confusion_matrix(y_true=targets, y_pred=preds)


EPSILON = 1e-6


def sensitivity_metric(cm: NDArray) -> float:
    _tn, _fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + EPSILON)
    return sensitivity


def specificity_metric(cm: NDArray) -> float:
    tn, fp, _fn, _tp = cm.ravel()
    specificity = tn / (tn + fp + EPSILON)
    return specificity


def f1_metric(cm: NDArray) -> float:
    _tn, fp, fn, tp = cm.ravel()
    f1 = (2 * tp) / ((2 * tp) + fp + fn + EPSILON)
    return f1


def ppv_metric(cm: NDArray) -> float:
    _tn, fp, _fn, tp = cm.ravel()
    ppv = tp / (tp + fp + EPSILON)
    return ppv


def npv_metric(cm: NDArray) -> float:
    tn, _fp, fn, _tp = cm.ravel()
    npv = tn / (tn + fn + EPSILON)
    return npv


def fnr_metric(cm: NDArray) -> float:
    _tn, _fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp + EPSILON)
    return fnr


def fpr_metric(cm: NDArray) -> float:
    tn, fp, _fn, _tp = cm.ravel()
    fpr = fp / (fp + tn + EPSILON)
    return fpr


def tn_metric(cm: NDArray) -> float:
    tn, _fp, _fn, _tp = cm.ravel()
    return tn


def fp_metric(cm: NDArray) -> float:
    _tn, fp, _fn, _tp = cm.ravel()
    return fp


def fn_metric(cm: NDArray) -> float:
    _tn, _fp, fn, _tp = cm.ravel()
    return fn


def tp_metric(cm: NDArray) -> float:
    _tn, _fp, _fn, tp = cm.ravel()
    return tp


class MulticlassClassificationMetricsComputer:
    """
    Metrics computer for multiclass classification task.

    Args:
        probs_to_preds: A ProbsToPreds callable converts a multiclass output to a prediction.
    """

    def __init__(
        self,
        probs_to_preds: ProbsToPreds = probs_to_preds_using_argmax,
        preds_and_targets_to_cm: PredsAndTargetsToConfusionMatrix = preds_and_targets_to_cm,
        sensitivity: ConfusionMatrixMulticlassTargetsMetric = sensitivity_metric,
        specificity: ConfusionMatrixMulticlassTargetsMetric = specificity_metric,
        f1: ConfusionMatrixMulticlassTargetsMetric = f1_metric,
        ppv: ConfusionMatrixMulticlassTargetsMetric = ppv_metric,
        npv: ConfusionMatrixMulticlassTargetsMetric = npv_metric,
        fnr: ConfusionMatrixMulticlassTargetsMetric = fnr_metric,
        fpr: ConfusionMatrixMulticlassTargetsMetric = fpr_metric,
        tn: ConfusionMatrixMulticlassTargetsMetric = tn_metric,
        fp: ConfusionMatrixMulticlassTargetsMetric = fp_metric,
        fn: ConfusionMatrixMulticlassTargetsMetric = fn_metric,
        tp: ConfusionMatrixMulticlassTargetsMetric = fn_metric,
    ) -> None:
        self.probs_to_preds = probs_to_preds
        self.preds_and_targets_to_cm = preds_and_targets_to_cm
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.f1 = f1
        self.ppv = ppv
        self.npv = npv
        self.fnr = fnr
        self.fpr = fpr
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.tp = fn

    # Probability based metrics
    @staticmethod
    def multi_class_ovr_roc_auc(probs: NDArray, targets: NDArray) -> Dict[str, float]:
        unaveraged_auc = cast(
            List[float],
            roc_auc_score(
                y_true=targets, y_score=probs, average=None, multi_class='ovr'  # type: ignore
            ),
        )
        return {f'ovr_auc_index_{i}': auc_value for i, auc_value in enumerate(unaveraged_auc)}

    @staticmethod
    def macro_average_ovr_roc_auc(probs: NDArray, targets: NDArray) -> Dict[str, float]:
        macro_averaged_auc = cast(
            float, roc_auc_score(y_true=targets, y_score=probs, average='macro', multi_class='ovr')
        )
        return {'macro_average_ovr_auc': macro_averaged_auc}

    # Prediction based metrics
    @staticmethod
    def classification_report(preds: NDArray, targets: NDArray) -> Dict[str, float]:
        report = cast(Dict, classification_report(y_true=targets, y_pred=preds, output_dict=True))
        return cast(
            Dict[str, float], pd.json_normalize(report, sep='_').to_dict(orient='records')[0]
        )

    @staticmethod
    def multi_class_confusion_matrix_metrics(
        metric_name: str, metric: ConfusionMatrixMulticlassTargetsMetric, cm: NDArray
    ) -> Dict[MetricName, float]:
        return {f'{metric_name}_index_{i}': metric(cm[i]) for i in range(cm.shape[0])}

    def __call__(self, data: MulticlassClassificationMetricsData) -> MetricsOutput:
        """
        Compute metrics.

        Args:
            data: will compute metrics on this.
        """
        if not self._contains_all_possible_targets(data.probs, data.targets):
            return {}

        preds = self.probs_to_preds(probs=data.probs)
        cm = self.preds_and_targets_to_cm(preds=preds, targets=data.targets)

        return {
            **self.multi_class_ovr_roc_auc(probs=data.probs, targets=data.targets),
            **self.macro_average_ovr_roc_auc(probs=data.probs, targets=data.targets),
            **self.classification_report(preds=preds, targets=data.targets),
            **self.multi_class_confusion_matrix_metrics('sensitivity', self.sensitivity, cm),
            **self.multi_class_confusion_matrix_metrics('specificity', self.specificity, cm),
            **self.multi_class_confusion_matrix_metrics('f1', self.f1, cm),
            **self.multi_class_confusion_matrix_metrics('ppv', self.ppv, cm),
            **self.multi_class_confusion_matrix_metrics('npv', self.npv, cm),
            **self.multi_class_confusion_matrix_metrics('fnr', self.fnr, cm),
            **self.multi_class_confusion_matrix_metrics('fpr', self.fpr, cm),
            **self.multi_class_confusion_matrix_metrics('tn', self.tn, cm),
            **self.multi_class_confusion_matrix_metrics('fp', self.fp, cm),
            **self.multi_class_confusion_matrix_metrics('fn', self.fn, cm),
            **self.multi_class_confusion_matrix_metrics('tp', self.tp, cm),
        }

    @staticmethod
    def _contains_all_possible_targets(probs: NDArray, targets: NDArray) -> bool:
        """Checks if targets contain all possible values."""
        return len(np.unique(targets)) == probs.shape[1]
