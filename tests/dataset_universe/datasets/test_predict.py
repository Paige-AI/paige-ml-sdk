import torch
from torch import Tensor

from paige.ml_sdk.dataset_universe.datasets.predict import EmbeddingAggregatorPredictDatasetItem


class TestEmbeddingAggregatorPredictDatasetItem:
    def test_should_have_expected_items(self):
        item = EmbeddingAggregatorPredictDatasetItem(1, Tensor([1]))
        assert item.group_index == 1
        assert torch.equal(item.embeddings, Tensor([1]))
