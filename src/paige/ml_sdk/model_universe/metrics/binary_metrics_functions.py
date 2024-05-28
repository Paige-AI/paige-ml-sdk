"""
Colllection of metrics and statistics for binary labels.
"""

import math
from dataclasses import dataclass
from typing import Any, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats as st
from sklearn.metrics import confusion_matrix, recall_score


def sensitivity(preds: NDArray, targets: NDArray) -> float:
    # in binary classification, recall of the positive class is the same thing as sensitivity.
    sensitivity: float = recall_score(y_true=targets, y_pred=preds, pos_label=1, average='binary')
    return sensitivity


def specificity(preds: NDArray, targets: NDArray) -> float:
    # in binary classification, recall of the negative class is the same thing as specificity.
    specificity: float = recall_score(y_true=targets, y_pred=preds, pos_label=0, average='binary')
    return specificity


@dataclass(frozen=True)
class BinaryConfusionMatrixOutput:
    """Dataclass encapsulating the outputs of a binary confusion matrix."""

    tp_count: float
    fp_count: float
    fn_count: float
    tn_count: float


def binary_confusion_matrix(preds: NDArray, targets: NDArray) -> BinaryConfusionMatrixOutput:
    cm = confusion_matrix(y_true=targets, y_pred=preds).astype('float')

    if cm.shape == (2, 2):
        pass
    elif cm.shape == (1, 1):
        cm = confusion_matrix(y_true=targets, y_pred=preds, labels=[0, 1]).astype('float')
    else:
        raise ValueError(
            'This function is only meant to work with binary preds and targets.',
            f'Expected confusion matrix shape (2, 2), got {cm.shape}.',
        )

    tn_count, fp_count, fn_count, tp_count = cm.ravel()

    return BinaryConfusionMatrixOutput(
        tn_count=tn_count, fp_count=fp_count, fn_count=fn_count, tp_count=tp_count
    )


def wilsons_interval(r: int, n: int, confidence_level: float = 0.95) -> Tuple[float, float]:
    """Calculates confidence interval using Wilson's method, which is the recommended method for binary ratios (acc, sens, spec)
    Reference: https://tbrieder.org/epidata/course_reading/b_altman.pdf pg 46-47

    Args:
        r: observed number of subjects with some feature (e.g. tp in accuracy)
        n: sample size
        confidence_level: Defaults to 0.95.
    """
    z = st.norm.ppf(0.5 + confidence_level / 2).item()
    A = 2 * r + z**2
    B = z * math.sqrt(z**2 + 4 * r * (1 - r / n))
    C = 2 * (n + z**2)
    return (A - B) / C, (A + B) / C


class DeLongInterval:
    """Includes functions that uses DeLong's method to calculate confidence intervals for AUCs.

    Code adapted from https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py
    """

    # AUC comparison adapted from
    # https://github.com/Netflix/vmaf/
    @staticmethod
    def _compute_midrank(x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Computes midranks.
        Args:
        x - a 1D numpy array
        Returns:
        array of midranks
        """
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2

    @staticmethod
    def fastDeLong(
        predictions_sorted_transposed: NDArray[np.float_], label_1_count: int
    ) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        """
        The fast version of DeLong's method for computing the covariance of
        unadjusted AUC.
        Args:
        predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
            sorted such as the examples with label "1" are first
        Returns:
        (AUC value, DeLong covariance)
        Reference:
        @article{sun2014fast,
        title={Fast Implementation of DeLong's Algorithm for
                Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
        author={Xu Sun and Weichao Xu},
        journal={IEEE Signal Processing Letters},
        volume={21},
        number={11},
        pages={1389--1393},
        year={2014},
        publisher={IEEE}
        }
        """
        # Short variables are named as they are in the paper
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)
        for r in range(k):
            tx[r, :] = DeLongInterval._compute_midrank(positive_examples[r, :])
            ty[r, :] = DeLongInterval._compute_midrank(negative_examples[r, :])
            tz[r, :] = DeLongInterval._compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    @staticmethod
    def logpvalue(aucs: NDArray[np.float_], delong_covs: NDArray[np.float_]) -> NDArray[np.float_]:
        """Computes log(10) of p-values.
        Args:
        aucs: 1D array of AUCs
        delong_covs: DeLong covariances (2nd output of fastDeLong)
        Returns:
        log10(pvalue)
        """
        diff = np.array([[1, -1]])
        z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(diff, delong_covs), diff.T))
        return np.log10(2) + st.norm.logsf(z, loc=0, scale=1) / np.log(10)

    @staticmethod
    def _y_true_statistics(y_true: NDArray[np.int_]) -> Tuple[NDArray, int]:
        assert np.array_equal(np.unique(y_true), [0, 1])
        order = (-y_true).argsort()
        label_1_count = int(y_true.sum())
        return order, label_1_count

    @staticmethod
    def auc_variance(
        y_true: NDArray[np.int_], y_score: NDArray[np.float_]
    ) -> Tuple[float, NDArray[np.float_]]:
        """
        Computes ROC AUC variance for a single set of predictions
        Args:
        y_true: np.array of 0 and 1
        y_score: np.array of floats of the probability of being class 1
        """
        order, label_1_count = DeLongInterval._y_true_statistics(y_true)
        predictions_sorted_transposed = y_score[np.newaxis, order]
        aucs, delongcov = DeLongInterval.fastDeLong(predictions_sorted_transposed, label_1_count)
        assert len(aucs) == 1, 'There is a bug in the code, please forward this to the developers'
        return aucs[0], delongcov

    @staticmethod
    def auc_test(
        y_true: NDArray[np.int_],
        y_score1: NDArray[np.float_],
        y_score2: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """
        Computes log(p-value) for hypothesis that two ROC AUCs are different
        Args:
        y_true: np.array of 0 and 1
        y_score1: predictions of the first model,
            np.array of floats of the probability of being class 1
        y_score2: predictions of the second model,
            np.array of floats of the probability of being class 1
        """
        order, label_1_count = DeLongInterval._y_true_statistics(y_true)
        predictions_sorted_transposed = np.vstack((y_score1, y_score2))[:, order]
        aucs, delongcov = DeLongInterval.fastDeLong(predictions_sorted_transposed, label_1_count)
        return DeLongInterval.logpvalue(aucs, delongcov)

    @staticmethod
    def auc_ci(
        y_true: NDArray[np.int_],
        y_score: NDArray[np.float_],
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Computes confidence interval of ROC AUC for a single set of predictions
        Args:
        y_true: np.array of 0 and 1
        y_score: np.array of floats of the probability of being class 1
        confidence_level: Defaults to 0.95
        """
        auc, auc_cov = DeLongInterval.auc_variance(y_true, y_score)

        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - confidence_level) / 2)

        ci = st.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
        ci[ci > 1] = 1
        ci[ci < 0] = 0
        return ci[0].item(), ci[1].item()


delong_interval = DeLongInterval.auc_ci  # alias
