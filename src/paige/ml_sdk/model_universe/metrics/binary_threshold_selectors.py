from abc import ABC
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_curve


class PredsBinaryTargetsMetric(Protocol):
    def __call__(self, preds: np.ndarray, targets: np.ndarray) -> float: ...


class ThresholdSelector(ABC):
    def __call__(self, *, probs: NDArray, targets: NDArray) -> float: ...


class MinimizeExpectedCost(ThresholdSelector):
    def __init__(self, prevalence: float = 0.5, fp_cost: float = 1, fn_cost: float = 1) -> None:
        """Chooses the threshold that minimizes the 'expected cost'.

        It works by calculating the desired slope
            m = (fp_cost / fn_cost) * ((1 - prevalence) / prevalence)

        and finding the point where a line with this slope touches the ROC curve.

        References:
         - Zweig, M. H., and G. Campbell. “Receiver-Operating Characteristic (ROC) Plots: A Fundamental Evaluation Tool in Clinical Medicine.”
         Clinical Chemistry, vol. 39, no. 4, Apr. 1993, pp. 561-77.


        Args:
            prevalence: Prior probability of positive class. Defaults to 0.5.
            fp_cost: Relative cost of false positives. Defaults to 1.
            fn_cost: Relative cost of false negatives. Defaults to 1.
        """
        if not (0 < prevalence < 1):
            raise ValueError(f'prevalence must be in (0,1), not {prevalence}.')
        if fp_cost <= 0:
            raise ValueError(f'fp_cost must be positive, not {fp_cost}.')
        if fn_cost <= 0:
            raise ValueError(f'fn_cost must be positive, not {fn_cost}.')

        # m is the desired slope
        self.m = (fp_cost / fn_cost) * ((1 - prevalence) / prevalence)

    def __call__(self, *, probs: NDArray, targets: NDArray) -> float:
        """Computes an optimal operating point with Youden's J Statistic.

        .. note::
            `thresholds[0]` represents no instances being predicted and is arbitrarily set
            to `max(y_score) + 1`, inherited by the use of `sklearn.metrics.roc_curve`.

        Preconditions
        -------------
        - probs and targets are 1D arrays

        References
        - https://en.wikipedia.org/wiki/Youden%27s_J_statistic
        - https://stats.stackexchange.com/a/386433
        """
        fpr_list, tpr_list, thresholds = roc_curve(y_true=targets, y_score=probs)
        best_idx = np.argmax(tpr_list - self.m * fpr_list)
        threshold: float = thresholds[best_idx]
        return threshold


class YoudenJStatistic(MinimizeExpectedCost):
    def __init__(self) -> None:
        """Computes an optimal operating point with Youden's J Statistic.

        .. note::
            This is a special case of `MinimizeExpectedCost` where m=1.

        Preconditions
        -------------
        - probs and targets are 1D arrays

        References
        - https://en.wikipedia.org/wiki/Youden%27s_J_statistic
        - https://stats.stackexchange.com/a/386433
        """
        self.m = 1


class MaximizeMetric(ThresholdSelector):
    def __init__(self, metric: PredsBinaryTargetsMetric) -> None:
        """Chooses threshold that maximizes the `metric`.

        Args:
            metric: a binary metric (e.g. sensitivity, specificity)
        """
        super().__init__()
        self.metric = metric

    def __call__(self, *, probs: NDArray, targets: NDArray) -> float:
        _, _, thresholds = roc_curve(y_true=targets, y_score=probs, drop_intermediate=False)
        best_metric, best_t = -float('inf'), -float('inf')
        for t in thresholds:
            # TODO: use make_predictions_from_positive_class_scores here
            preds = probs >= t  # consistent with scikit.metrics.roc_curve
            metric = self.metric(preds=preds, targets=targets)
            if metric > best_metric:
                best_metric = metric
                best_t = t
        return best_t


class FixMetric(ThresholdSelector):
    def __init__(self, metric: PredsBinaryTargetsMetric, target_value: float) -> None:
        """Chooses threshold that that gets the `metric` closest to the `target_value`

        Args:
            metric: a binary metric (e.g. sensitivity, specificity)
            target_value: target value
        """
        super().__init__()
        self.metric = metric
        self.target_value = target_value

    def __call__(self, *, probs: NDArray, targets: NDArray) -> float:
        _, _, thresholds = roc_curve(y_true=targets, y_score=probs, drop_intermediate=False)
        best_diff, best_t = float('inf'), -float('inf')
        for t in thresholds:
            # TODO: use make_predictions_from_positive_class_scores here
            preds = probs >= t  # consistent with scikit.metrics.roc_curve
            metric = self.metric(preds=preds, targets=targets)
            diff = abs(metric - self.target_value)
            if diff < best_diff:
                best_diff = diff
                best_t = t
        return best_t
