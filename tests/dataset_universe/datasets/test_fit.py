import torch
from torch import Tensor

from paige.ml_sdk.dataset_universe.datasets.fit import EmbeddingAggregatorFitDatasetItem


class TestEmbeddingAggregatorFitDatasetItem:
    def test_should_have_expected_items(self):
        group_index = 1
        embeddings = Tensor([1])
        label_map = {'label': 0}
        instance_mask_map = {'label': True}
        item = EmbeddingAggregatorFitDatasetItem(
            group_index, embeddings, label_map, instance_mask_map
        )
        assert item.group_index == 1
        assert torch.equal(item.embeddings, embeddings)
        assert item.label_map == label_map
        assert item.instance_mask_map == instance_mask_map
