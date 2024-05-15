""" Just test that inferfaces work as expected; unit tests of metrics functions already
    exist in test_binary_classifcation.py and test_multiclass_classifcation.py
"""

from torch import Tensor

from paige.ml_sdk.model_universe.metrics.metrics_computers.aggregator_metrics import (
    AggregatorBinClsGroupMetricsComputer,
    AggregatorMulticlassGroupMetricsComputer,
)


class TestAggregatorBinClsGroupMetricsComputer:
    def test_should_compute_metrics(self):
        metrics_computer = AggregatorBinClsGroupMetricsComputer(label_name='label')
        label_map = {'label': Tensor([1, 0, 1])}
        instance_mask_map = {'label': Tensor([1, 1, 1]).bool()}
        heads_activations = {'label': Tensor([0.9, 0.2, 0.1])}
        metrics_computer(label_map, instance_mask_map, heads_activations)


class TestAggregatorMulticlassGroupMetricsComputer:
    def test_should_compute_metrics(self):
        metrics_computer = AggregatorMulticlassGroupMetricsComputer(label_name='label')
        label_map = {'label': Tensor([2, 0, 1])}
        instance_mask_map = {'label': Tensor([1, 0, 1]).bool()}
        heads_activations = {'label': Tensor([[0.2, 0.2, 0.8], [0.5, 0.4, 0.1], [0.1, 0.6, 0.3]])}
        metrics_computer(label_map, instance_mask_map, heads_activations)
