from typing import NamedTuple, Protocol

from torch import Tensor
from torch.utils.data import Dataset


class EmbeddingAggregatorPredictDatasetItem(NamedTuple):
    group_index: int
    embeddings: Tensor


class EmbeddingAggregatorPredictDatasetProvider(Protocol):
    @property
    def dataset(self) -> Dataset[EmbeddingAggregatorPredictDatasetItem]:
        ...
