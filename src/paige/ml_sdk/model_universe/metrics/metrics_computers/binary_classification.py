import logging
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from paige.ml_sdk.array_procedures import enforce_1d_shape
from paige.ml_sdk.enforce_type import enforce_type
from paige.ml_sdk.model_universe.metrics.binary_metrics_functions import (
    BinaryConfusionMatrixOutput,
    binary_confusion_matrix,
    delong_interval,
    sensitivity,
    specificity,
    wilsons_interval,
)
from paige.ml_sdk.model_universe.metrics.binary_threshold_selectors import (
    ThresholdSelector,
    YoudenJStatistic,
)
from paige.ml_sdk.model_universe.metrics.sklearn import (
    sklearn_accuracy_score,
    sklearn_balanced_accuracy_score,
    sklearn_f1_score,
    sklearn_f1_weighted_score,
    sklearn_roc_auc_score,
)

logger = logging.getLogger(__name__)


def cast_nan_to_none(*args: float) -> Tuple[Optional[float], ...]:
    return tuple((None if np.isnan(x) else x) for x in args)


def make_predictions_from_positive_class_scores(
    *, probs: NDArray[np.float_], threshold: float
) -> NDArray[np.int_]:
    """
    Makes binary predictions based on the positive class confidence scores.

    Preconditions
    -------------
    - probs and targets are 1D arrays
    """
    if threshold < 0:
        raise ValueError(f'Threshold must be positive but received: {threshold}.')
    if np.logical_or(probs < 0, probs > 1).any():
        raise ValueError(f'Probs must be bounded in [0, 1] but received: {probs}.')
    # >= is consistent with sklearn.metrics.roc_curve
    probs_above_threshold: NDArray[np.bool_] = probs >= threshold
    return probs_above_threshold.astype(int)


MetricName = str
MetricsOutput = Dict[MetricName, Optional[float]]


# TODO: make these ABC and define them in ml.evaluator.metrics_functions.element_wise.binary
class PredsBinaryTargetsMetric(Protocol):
    def __call__(self, preds: np.ndarray, targets: np.ndarray) -> float:
        ...


class ProbsBinaryTargetsMetric(Protocol):
    def __call__(self, probs: np.ndarray, targets: np.ndarray) -> float:
        ...


class ConfusionMatrixMetric(Protocol):
    def __call__(self, preds: NDArray, targets: NDArray) -> BinaryConfusionMatrixOutput:
        ...


@dataclass
class BinaryClassificationMetricsData:
    """
    Data required for metrics computation.

    Args:
        probs: model probabilities.
        targets: ground truth labels.
    """

    probs: NDArray[np.float_]
    targets: NDArray[np.int_]

    def __post_init__(self) -> None:
        self.probs = enforce_1d_shape(enforce_type(np.ndarray, self.probs))
        self.targets = enforce_1d_shape(enforce_type(np.ndarray, self.targets))


class BinaryClassificationMetricsComputer:
    """
        Metrics computer for binary classification task.

        Args:
            accuracy: A user definfed accuracy metric (i.e. correct / all). Defaults to sklearn_accuracy_score.
            auc: A user defined AUC metric. Defaults to sklearn_roc_auc_score.
            f1: A user defined F1 metric. Defaults to sklearn_f1_score.
            sensitivity: A user defined sensitivity metric. Defaults to sensitivity.
            specificity: A user defined specificity metric. Defaults to specificity.
            confusion_matrix: A user defined confusiong matrix metric. Defaults to binary_confusion_matrix.
            threshold_selector: A ThresholdSelector object that selects a threshold from probs and targets. Defaults to YoudenJStatistic().
    .
    """

    METRICS = {
        'accuracy',
        'accuracy_ci_low',
        'accuracy_ci_high',
        'balanced_accuracy',
        'auc',
        'auc_ci_low',
        'auc_ci_high',
        'f1',
        'f1_weighted',
        'sensitivity',
        'sensitivity_ci_low',
        'sensitivity_ci_high',
        'specificity',
        'specificity_ci_low',
        'specificity_ci_high',
        'tn_count',
        'tp_count',
        'fn_count',
        'fp_count',
        'threshold',
    }

    def __init__(
        self,
        *,
        accuracy: PredsBinaryTargetsMetric = sklearn_accuracy_score,
        balanced_accuracy: PredsBinaryTargetsMetric = sklearn_balanced_accuracy_score,
        auc: ProbsBinaryTargetsMetric = sklearn_roc_auc_score,
        f1: PredsBinaryTargetsMetric = sklearn_f1_score,
        f1_weighted: PredsBinaryTargetsMetric = sklearn_f1_weighted_score,
        sensitivity: PredsBinaryTargetsMetric = sensitivity,
        specificity: PredsBinaryTargetsMetric = specificity,
        confusion_matrix: ConfusionMatrixMetric = binary_confusion_matrix,
        threshold_selector: Optional[ThresholdSelector] = None,
    ) -> None:
        # user-defined metrics.
        self._accuracy = accuracy
        self._balanced_accuracy = balanced_accuracy
        self._auc = auc
        self._f1 = f1
        self._f1_weighted = f1_weighted
        self._sensitivity = sensitivity
        self._specificity = specificity
        self._confusion_matrix = confusion_matrix

        self.threshold_selector = threshold_selector or YoudenJStatistic()

        self._default_output: MetricsOutput = {metric_name: None for metric_name in self.METRICS}

    def __call__(self, data: BinaryClassificationMetricsData) -> MetricsOutput:
        """
        Compute metrics.

        Args:
            data: will compute metrics on this.
        """
        if len(data.probs) == 0 and len(data.targets) == 0:
            logger.warning('Empty data, returning default output')
            return self._default_output
        threshold = self.threshold_selector(probs=data.probs, targets=data.targets)
        preds = make_predictions_from_positive_class_scores(probs=data.probs, threshold=threshold)
        cm_output = self._confusion_matrix(preds=preds, targets=data.targets)
        num_positive = int(cm_output.tp_count) + int(cm_output.fn_count)
        has_positive = num_positive > 0
        num_negative = int(cm_output.tn_count) + int(cm_output.fp_count)
        has_negative = num_negative > 0

        accuracy_ci_low, accuracy_ci_high = cast_nan_to_none(
            *wilsons_interval(
                int(cm_output.tp_count) + int(cm_output.tn_count), num_negative + num_positive
            )
        )
        sensitivity_ci_low, sensitivity_ci_high = (
            cast_nan_to_none(*wilsons_interval(int(cm_output.tp_count), num_positive))
            if has_positive
            else (
                self._default_output['sensitivity_ci_low'],
                self._default_output['sensitivity_ci_high'],
            )
        )
        specificity_ci_low, specificity_ci_high = (
            cast_nan_to_none(*wilsons_interval(int(cm_output.tn_count), num_negative))
            if has_negative
            else (
                self._default_output['specificity_ci_low'],
                self._default_output['specificity_ci_high'],
            )
        )
        auc_ci_low, auc_ci_high = (
            cast_nan_to_none(*delong_interval(y_true=data.targets, y_score=data.probs))
            if (has_positive and has_negative)
            else (
                self._default_output['auc_ci_low'],
                self._default_output['auc_ci_high'],
            )
        )

        return {
            'accuracy': self._accuracy(preds=preds, targets=data.targets),
            'accuracy_ci_low': accuracy_ci_low,
            'accuracy_ci_high': accuracy_ci_high,
            'balanced_accuracy': (
                self._balanced_accuracy(preds=preds, targets=data.targets)
                if (has_positive and has_negative)
                else self._default_output['accuracy']
            ),
            'auc': (
                self._auc(probs=data.probs, targets=data.targets)
                if (has_positive and has_negative)
                else self._default_output['auc']
            ),
            'auc_ci_low': auc_ci_low,
            'auc_ci_high': auc_ci_high,
            'f1': (
                self._f1(preds=preds, targets=data.targets)
                if (has_positive and has_negative)
                else self._default_output['f1']
            ),
            'f1_weighted': (
                self._f1_weighted(preds=preds, targets=data.targets)
                if (has_positive and has_negative)
                else self._default_output['f1_weighted']
            ),
            'sensitivity': (
                self._sensitivity(preds=preds, targets=data.targets)
                if has_positive
                else self._default_output['sensitivity']
            ),
            'sensitivity_ci_low': sensitivity_ci_low,
            'sensitivity_ci_high': sensitivity_ci_high,
            'specificity': (
                self._specificity(preds=preds, targets=data.targets)
                if has_negative
                else self._default_output['specificity']
            ),
            'specificity_ci_low': specificity_ci_low,
            'specificity_ci_high': specificity_ci_high,
            'tp_count': cm_output.tp_count,
            'tn_count': cm_output.tn_count,
            'fn_count': cm_output.fn_count,
            'fp_count': cm_output.fp_count,
            'threshold': float(threshold),  # threshold selector returns float32, tests expect float
        }


# Functions useful inside algo data to metrics data mappers.
# TODO (george): find a better place for these.


def extract_positive_class_from_probs(probs: Tensor, pos_cls_idx: int) -> Tensor:
    """
    Takes probabilities for the positive class from 2D tensors.
    Leaves 1D tensors unchanged.
    Returns 1D tensor.
    """
    if pos_cls_idx not in {0, 1}:
        raise ValueError(f'pos_cls_idx must be 0 or 1, got: {pos_cls_idx}')
    if probs.ndim == 1:
        return probs
    if probs.ndim == 2 and probs.shape[1] == 1:
        return probs[:, 0]
    if probs.ndim == 2 and probs.shape[1] == 2:
        return probs[:, pos_cls_idx]
    raise ValueError(f'expected 1D tensor or 2D tensor with 2 columns, got: {probs.shape}.')


def enforce_valid_confidence_scores_values(probs: Tensor, label_name: str) -> Tensor:
    """
    Extracts probabilities for a given label.
    """
    if (probs < 0).any() or (probs > 1).any():
        invalid_probs = probs[torch.logical_or(probs < 0.0, probs > 1.0)]
        raise ValueError(
            f'Confidence scores for `{label_name}` containing valid probabilities '
            f'(bounded in [0, 1]) were expected, but received {invalid_probs}.'
        )
    return probs
