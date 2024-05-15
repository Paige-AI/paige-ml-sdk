from dataclasses import dataclass

import numpy as np
import pytest

from paige.ml_sdk.model_universe.metrics.metrics_computers.multiclass_classification import (
    MetricsOutput,
    MulticlassClassificationMetricsComputer,
    MulticlassClassificationMetricsData,
)


class Test_MulticlassClassificationMetricsComputer:
    @dataclass
    class Case:
        name: str
        data: MulticlassClassificationMetricsData
        expected: MetricsOutput

        def __post_init__(self) -> None:
            self.__name__ = self.name

    @pytest.mark.parametrize(
        'test_case',
        [
            Case(
                name='instance level metrics',
                data=MulticlassClassificationMetricsData(
                    probs=np.array(
                        [
                            # class 0, class 1, class 2
                            [0.2, 0.7, 0.1],
                            [0.9, 0.1, 0.0],
                            [0.7, 0.0, 0.3],
                            [0.7, 0.2, 0.1],
                            [0.9, 0.0, 0.1],
                            [0.4, 0.0, 0.6],
                        ]
                    ),
                    targets=np.array([1, 0, 0, 1, 2, 0]),
                ),
                # Note: This is not nice (or useful) but we do not have better option right now :/
                # If we use mock.ANY could be more future-proof but not testing correctness of the computations.
                expected={
                    'ovr_auc_index_0': 0.5555,
                    'ovr_auc_index_1': 1.0,
                    'ovr_auc_index_2': 0.3999,
                    'macro_average_ovr_auc': 0.6518,
                    'accuracy': 0.5,
                    '0_precision': 0.5,
                    '0_recall': 0.6666,
                    '0_f1-score': 0.5714,
                    '0_support': 3,
                    '1_precision': 1.0,
                    '1_recall': 0.5,
                    '1_f1-score': 0.6666,
                    '1_support': 2,
                    '2_precision': 0.0,
                    '2_recall': 0.0,
                    '2_f1-score': 0.0,
                    '2_support': 1,
                    'macro avg_precision': 0.5,
                    'macro avg_recall': 0.3888,
                    'macro avg_f1-score': 0.4126,
                    'macro avg_support': 6,
                    'weighted avg_precision': 0.5833,
                    'weighted avg_recall': 0.5,
                    'weighted avg_f1-score': 0.5079,
                    'weighted avg_support': 6,
                    'sensitivity_index_0': 0.6666,
                    'sensitivity_index_1': 0.4999,
                    'sensitivity_index_2': 0.0,
                    'specificity_index_0': 0.3333,
                    'specificity_index_1': 0.9999,
                    'specificity_index_2': 0.7999,
                    'f1_index_0': 0.5714,
                    'f1_index_1': 0.66666,
                    'f1_index_2': 0.0,
                    'ppv_index_0': 0.4999,
                    'ppv_index_1': 0.9999,
                    'ppv_index_2': 0.0,
                    'npv_index_0': 0.4999,
                    'npv_index_1': 0.7999,
                    'npv_index_2': 0.7999,
                    'fnr_index_0': 0.3333,
                    'fnr_index_1': 0.4999,
                    'fnr_index_2': 0.9999,
                    'fpr_index_0': 0.6666,
                    'fpr_index_1': 0.0,
                    'fpr_index_2': 0.1999,
                    'tn_index_0': 1,
                    'tn_index_1': 4,
                    'tn_index_2': 4,
                    'fp_index_0': 2,
                    'fp_index_1': 0,
                    'fp_index_2': 1,
                    'fn_index_0': 1,
                    'fn_index_1': 1,
                    'fn_index_2': 1,
                    'tp_index_0': 1,
                    'tp_index_1': 1,
                    'tp_index_2': 1,
                },
            ),
            Case(
                name='instance level metrics, one class samples only',
                data=MulticlassClassificationMetricsData(
                    probs=np.array([[0.1, 0.2], [0.3, 0.4]]),
                    targets=np.array([0, 0]),
                ),
                # Returns empty as keys depend on the classes.
                expected={},
            ),
        ],
    )
    def test_should_run_metrics_computation_lifecycle_with_expected_metrics_results(
        self, test_case: Case
    ) -> None:
        metrics_computer = MulticlassClassificationMetricsComputer()

        epoch_metrics = metrics_computer(test_case.data)

        for key in test_case.expected:
            np.testing.assert_allclose(test_case.expected[key], epoch_metrics[key], atol=1e-2)
