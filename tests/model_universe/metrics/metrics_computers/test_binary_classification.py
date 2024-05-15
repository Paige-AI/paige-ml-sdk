from dataclasses import dataclass
from typing import Type
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import Tensor

from paige.ml_sdk.model_universe.metrics.metrics_computers.binary_classification import (
    BinaryClassificationMetricsComputer,
    BinaryClassificationMetricsData,
    MetricsOutput,
    extract_positive_class_from_probs,
    make_predictions_from_positive_class_scores,
)


class Test_BinaryClassificationMetricsComputer:
    @dataclass
    class Case:
        name: str
        data: BinaryClassificationMetricsData
        expected: MetricsOutput

        def __post_init__(self) -> None:
            self.__name__ = self.name

    @pytest.mark.parametrize(
        'test_case',
        [
            Case(
                name='instance level metrics, two batches, bce loss',
                data=BinaryClassificationMetricsData(
                    probs=np.array(
                        [0.1, 0.2, 0.25, 0.25, 0.3, 0.4, 0.1, 0.2, 0.25, 0.25, 0.25, 0.45]
                    ),
                    targets=np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]),
                ),
                expected={
                    'accuracy': 0.83,
                    'accuracy_ci_low': 0.552,
                    'accuracy_ci_high': 0.953,
                    'balanced_accuracy': 0.83,
                    'auc': 0.91,
                    'auc_ci_low': 0.790,
                    'auc_ci_high': 1.0,
                    'f1': 0.86,
                    'f1_weighted': 0.83,
                    'sensitivity': 1.0,
                    'sensitivity_ci_low': 0.61,
                    'sensitivity_ci_high': 1.0,
                    'specificity': 0.67,
                    'specificity_ci_low': 0.30,
                    'specificity_ci_high': 0.90,
                    'tp_count': 6.0,
                    'tn_count': 4.0,
                    'fn_count': 0.0,
                    'fp_count': 2.0,
                    'threshold': 0.25,
                },
            ),
            Case(
                name='instance level metrics, one class samples only',
                data=BinaryClassificationMetricsData(
                    probs=np.array([0.1, 0.2]),
                    targets=np.array([0, 0]),
                ),
                expected={
                    'accuracy': 1.0,
                    'accuracy_ci_low': 0.342,
                    'accuracy_ci_high': 1.0,
                    'balanced_accuracy': None,
                    'auc': None,
                    'auc_ci_low': None,
                    'auc_ci_high': None,
                    'f1': None,
                    'f1_weighted': None,
                    'sensitivity': None,
                    'sensitivity_ci_low': None,
                    'sensitivity_ci_high': None,
                    'specificity': 1.0,
                    'specificity_ci_low': 0.34,
                    'specificity_ci_high': 1.0,
                    'tp_count': 0.0,
                    'tn_count': 2.0,
                    'fn_count': 0.0,
                    'fp_count': 0.0,
                    'threshold': float('inf'),
                },
            ),
        ],
    )
    def test_should_run_metrics_computation_lifecycle_with_expected_metrics_results(
        self, test_case: Case
    ) -> None:
        # Arrange
        metrics_computer = BinaryClassificationMetricsComputer()

        # Act
        epoch_metrics = metrics_computer(test_case.data)

        # Assert
        assert test_case.expected.keys() == epoch_metrics.keys()
        for key in test_case.expected:
            assert epoch_metrics[key] == pytest.approx(test_case.expected[key], 1e-2)


class Test_make_predictions_from_positive_class_scores:
    def test_should_make_binary_prediction_based_on_threshold(self) -> None:
        preds = make_predictions_from_positive_class_scores(
            probs=np.array([0.1, 0.2, 0.3]), threshold=0.2
        )
        np.testing.assert_array_almost_equal(preds, [0, 1, 1])

    @pytest.mark.parametrize(
        'probs, threshold, msg',
        (
            pytest.param(
                np.array([-0.1, 1.1]), 0.1, r'Probs must be bounded.*', id='out-of-bound-probs'
            ),
            pytest.param(
                np.array([0.5]), -0.1, r'Threshold must be positive.*', id='threshold-negative'
            ),
        ),
    )
    def test_should_raise_if_invalid_inputs_are_given(
        self, probs: np.ndarray, threshold: float, msg: str
    ) -> None:
        with pytest.raises(ValueError, match=msg):
            make_predictions_from_positive_class_scores(probs=probs, threshold=threshold)


class Test_extract_positive_class_from_probs:
    MOCK_INT = MagicMock(spec=int)

    @dataclass
    class Case:
        name: str
        probs: Tensor
        pos_cls_idx: int
        expected: Tensor

        def __post_init__(self) -> None:
            self.__name__ = self.name

    @pytest.mark.parametrize(
        'test_case',
        [
            Case(
                name='1D with no items',
                probs=torch.tensor([]),
                pos_cls_idx=0,
                expected=torch.tensor([]),
            ),
            Case(
                name='1D with items',
                probs=torch.tensor([1, 2]),
                pos_cls_idx=0,
                expected=torch.tensor([1, 2]),
            ),
            Case(
                name='2D with one column',
                probs=torch.tensor([[1], [2]]),
                pos_cls_idx=0,
                expected=torch.tensor([1, 2]),
            ),
            Case(
                name='2D with two columns, pick second',
                probs=torch.tensor([[1, 9], [2, 8]]),
                pos_cls_idx=1,
                expected=torch.tensor([9, 8]),
            ),
        ],
    )
    def test_should_extract_correct_axis(self, test_case: Case) -> None:
        probs = extract_positive_class_from_probs(test_case.probs, test_case.pos_cls_idx)
        torch.testing.assert_close(test_case.expected, probs)

    @dataclass
    class CaseRaises:
        name: str
        probs: Tensor
        pos_cls_idx: int
        expected: Type[Exception]

        def __post_init__(self) -> None:
            self.__name__ = self.name

    @pytest.mark.parametrize(
        'test_case',
        [
            CaseRaises(
                name='3D tensor',
                probs=torch.rand(3, 2, 1),
                pos_cls_idx=0,
                expected=ValueError,
            ),
            CaseRaises(
                name='positive class is not 0 or 1',
                probs=torch.rand(3, 10),
                pos_cls_idx=2,
                expected=ValueError,
            ),
            CaseRaises(
                name='2D with no items',
                probs=torch.tensor([[]]),
                pos_cls_idx=0,
                expected=ValueError,
            ),
        ],
    )
    def test_should_raise_on_bad_input(self, test_case: CaseRaises) -> None:
        with pytest.raises(test_case.expected):
            extract_positive_class_from_probs(test_case.probs, test_case.pos_cls_idx)
