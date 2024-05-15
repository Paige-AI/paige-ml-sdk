import logging
from dataclasses import dataclass
from typing import Dict, Literal

import torch

from paige.ml_sdk.distributed.collections_map import to_numpy
from paige.ml_sdk.model_universe.aggregator import AggregatorMetricsComputer
from paige.ml_sdk.model_universe.instance_mask import apply_instance_mask
from paige.ml_sdk.model_universe.metrics.metrics_computers.binary_classification import (
    BinaryClassificationMetricsComputer,
    BinaryClassificationMetricsData,
    MetricsOutput,
    enforce_valid_confidence_scores_values,
    extract_positive_class_from_probs,
)
from paige.ml_sdk.model_universe.metrics.metrics_computers.multiclass_classification import (
    MulticlassClassificationMetricsComputer,
    MulticlassClassificationMetricsData,
)

logger = logging.getLogger(__name__)


@dataclass
class AggregatorBinClsGroupMetricsComputer(AggregatorMetricsComputer):
    label_name: str
    pos_cls_idx: int = 1
    metrics_computer: BinaryClassificationMetricsComputer = BinaryClassificationMetricsComputer()

    def __call__(
        self,
        label_map: Dict[str, torch.Tensor],
        instance_mask_map: Dict[str, torch.Tensor],
        heads_activations: Dict[str, torch.Tensor],
    ) -> Dict[str, MetricsOutput]:
        metrics_data = self.map_to_metrics_data(label_map, instance_mask_map, heads_activations)
        metrics = self.metrics_computer(metrics_data)
        return {self.label_name: metrics}

    def map_to_metrics_data(
        self,
        label_map: Dict[str, torch.Tensor],
        instance_mask_map: Dict[str, torch.Tensor],
        heads_activations: Dict[str, torch.Tensor],
    ) -> BinaryClassificationMetricsData:
        """Maps `Aggregator` data to `BinaryClassificationMCMapOutput`."""
        probs = heads_activations[self.label_name]
        probs = enforce_valid_confidence_scores_values(probs, self.label_name)
        probs = extract_positive_class_from_probs(probs, pos_cls_idx=self.pos_cls_idx)
        targets = label_map[self.label_name]

        # apply instance_mask_map
        instance_mask = instance_mask_map[self.label_name]
        probs = apply_instance_mask(probs, instance_mask=instance_mask)
        targets = apply_instance_mask(targets, instance_mask=instance_mask)

        return BinaryClassificationMetricsData(
            probs=to_numpy(probs),
            targets=to_numpy(targets),
        )


@dataclass
class AggregatorMulticlassGroupMetricsComputer(AggregatorMetricsComputer):
    label_name: str
    metrics_computer: MulticlassClassificationMetricsComputer = (
        MulticlassClassificationMetricsComputer()
    )

    def __call__(
        self,
        label_map: Dict[str, torch.Tensor],
        instance_mask_map: Dict[str, torch.Tensor],
        heads_activations: Dict[str, torch.Tensor],
    ) -> Dict[str, MetricsOutput]:
        metrics_data = self.map_to_metrics_data(label_map, instance_mask_map, heads_activations)
        metrics = self.metrics_computer(metrics_data)
        return {self.label_name: metrics}  # type: ignore

    def map_to_metrics_data(
        self,
        label_map: Dict[str, torch.Tensor],
        instance_mask_map: Dict[str, torch.Tensor],
        heads_activations: Dict[str, torch.Tensor],
    ) -> MulticlassClassificationMetricsData:
        """Maps `Aggregator` data to `MulticlassClassificationMetricsData`."""
        probs = heads_activations[self.label_name]
        targets = label_map[self.label_name]

        # apply instance_mask_map
        instance_mask = instance_mask_map[self.label_name]
        probs = apply_instance_mask(probs, instance_mask=instance_mask)
        targets = apply_instance_mask(targets, instance_mask=instance_mask)

        return MulticlassClassificationMetricsData(
            probs=to_numpy(probs),
            targets=to_numpy(targets),
        )
