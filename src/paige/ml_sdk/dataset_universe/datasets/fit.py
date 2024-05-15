from typing import Mapping, NamedTuple, Protocol, Union

from torch import Tensor
from torch.utils.data import Dataset


class EmbeddingAggregatorFitDatasetItem(NamedTuple):
    group_index: int
    embeddings: Tensor
    label_map: Mapping[str, Union[int, float]]
    instance_mask_map: Mapping[str, bool]


class EmbeddingAggregatorFitDatasetProvider(Protocol):
    """
    Provide map-style pytorch dataset and associated data.
    """

    @property
    def dataset(self) -> Dataset[EmbeddingAggregatorFitDatasetItem]:
        ...
