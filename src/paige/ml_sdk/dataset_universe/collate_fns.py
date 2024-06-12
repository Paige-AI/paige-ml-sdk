import logging
from typing import Dict, NamedTuple, Sequence, Set, Union

import torch
from torch import Tensor

from paige.ml_sdk.convert_to_torch_dtype import convert_to_torch_dtype
from paige.ml_sdk.dataset_universe.datasets.fit import EmbeddingAggregatorFitDatasetItem
from paige.ml_sdk.dataset_universe.datasets.predict import EmbeddingAggregatorPredictDatasetItem

SupportedDatasetOutputTypes = Union[
    EmbeddingAggregatorFitDatasetItem, EmbeddingAggregatorPredictDatasetItem
]


logger = logging.getLogger(__name__)


class EmbeddingAggregatorFitCollatedItems(NamedTuple):
    group_indices: Tensor
    embeddings: Tensor
    label_map: Dict[str, Tensor]
    padding_mask: Tensor
    sequence_lengths: Tensor
    instance_mask_map: Dict[str, Tensor]


class EmbeddingAggregatorPredictCollatedItems(NamedTuple):
    group_indices: Tensor
    embeddings: Tensor
    padding_mask: Tensor
    sequence_lengths: Tensor


class _CollateHelperOutput(NamedTuple):
    group_indices: Tensor
    embeddings: Tensor
    padding_mask: Tensor
    sequence_lengths: Tensor


class _EmbeddingAggregatorVariableSizeSequencesCollateHelper:
    SUPPORTED_ORDERS: Set[str] = {'bsf'}

    def __init__(
        self,
        *,
        order: str = 'bsf',
        valid_indicator: float = 0,
        padding_indicator: float = 1,
        padding_mask_dtype: torch.dtype = torch.uint8,
    ) -> None:
        """Collate functionality to be used as a DataLoader's collate_fn

        Args:
            order: Desired order of group embeddings batch tensor. 'bsf' stands for Batch, Sequence,
                Features which is the required order for input tensors to models that use nn.Linear
                layers. Defaults to 'bsf'.
            valid_indicator: Value to use in padding masks to indicate valid positions. Defaults to
                0.
            padding_indicator: Value to use in padding masks to indicate padding positions. Defaults
                to 1.
            padding_mask_dtype: dtype of padding mask tensor. Defaults to torch.uint8.

        Raises:
            ValueError: Unsupported order string is provided.
        """
        # NOTE: `order` does not do anything right now as only 'bsf' is supported but we anticipate adding more.
        if order not in self.SUPPORTED_ORDERS:
            raise ValueError(
                f'Got unsupported `order` of {order}. Expected one of the following options: {self.SUPPORTED_ORDERS}'
            )
        self._order = order
        self._order_handlers = {'bsf': self._collate_bsf}
        self._valid_indicator = valid_indicator
        self._padding_indicator = padding_indicator
        self._padding_mask_dtype = padding_mask_dtype

    def collate_embeddings_mask_and_lengths(
        self, batch: Sequence[SupportedDatasetOutputTypes]
    ) -> _CollateHelperOutput:
        """Collects embeddings, padding mask, and sequence lenths from batch items into three
        tensors.

        Args:
            batch: List of items from Dataset.

        Returns:
            _CollateHelperOutput: Batch tensor for group indices, embeddings, padding masks,
            and sequence lengths.
        """
        return self._order_handlers[self._order](batch)

    def _collate_bsf(self, batch: Sequence[SupportedDatasetOutputTypes]) -> _CollateHelperOutput:
        """Collates embeddings when order=bsf

        ..note:: Batch elements can have 0 embeddings if the corresponding group had no instance
          images after filtering. In this case, the padding_mask will indicate that the entire
          batch element is empty.

        Args:
            batch: A sequence of length batchsize containing embeddings w/ shape
            (sequence_length, features) a.k.a (n_embeddings, embedding_size).

        Returns:
            Collated embeddings, padding_masks, and sequence_lengths.
        """
        batch_size = len(batch)
        sequence_lengths_batch = torch.zeros((batch_size,), dtype=torch.int64)
        group_indices_batch = torch.zeros((batch_size,), dtype=torch.int64)

        # first loop through `batch` to collect sequence lengths to then find the longest in the
        # batch and collect group indices
        for idx, item in enumerate(batch):
            # one instance of embeddings are expected to have shape (seq_length, features)
            sequence_lengths_batch[idx] = item.embeddings.shape[0]
            group_indices_batch[idx] = item.group_index

        feature_size = batch[-1].embeddings.shape[1]
        max_sequence_length = int(sequence_lengths_batch.max().item())

        tensor_size = (batch_size, max_sequence_length, feature_size)
        mask_size = (batch_size, max_sequence_length)

        embeddings_batch = torch.zeros(tensor_size, dtype=batch[-1].embeddings.dtype)
        mask_batch = torch.zeros(mask_size, dtype=self._padding_mask_dtype)

        # second loop through `batch` to update empty embeddings batch tensor and padding mask batch
        # tensor
        for idx, item in enumerate(batch):
            sequence_length = int(sequence_lengths_batch[idx])
            # sequence length could be 0 if a slide has 0 foreground tiles
            if sequence_length > 0:
                embeddings_batch[idx, :sequence_length] += item.embeddings
            else:
                logger.warning('Batch element has sequence length 0.')
            mask_batch[idx, :sequence_length].fill_(self._valid_indicator)
            mask_batch[idx, sequence_length:].fill_(self._padding_indicator)

        return _CollateHelperOutput(
            group_indices=group_indices_batch,
            embeddings=embeddings_batch,
            padding_mask=mask_batch,
            sequence_lengths=sequence_lengths_batch,
        )


class EmbeddingAggregatorVariableSizeSequencesFitCollate:
    def __init__(
        self,
        *,
        order: str = 'bsf',
        valid_indicator: float = 0,
        padding_indicator: float = 1,
        padding_mask_dtype: torch.dtype = torch.uint8,
    ) -> None:
        """Collate functionality to be used as the a Dataloader's collate_fn during trainer.fit().

        Args:
            order: Desired order of group embeddings batch tensor. 'bsf' stands for Batch, Sequence,
                Features which is the required order for input tensors to models that use nn.Linear
                layers. Defaults to 'bsf'.
            valid_indicator: Value to use in padding masks to indicate valid positions. Defaults to
                0.
            padding_indicator: Value to use in padding masks to indicate padding positions. Defaults
                to 1.
            padding_mask_dtype: dtype of padding mask tensor. Defaults to torch.uint8.
        """
        self._collater = _EmbeddingAggregatorVariableSizeSequencesCollateHelper(
            order=order,
            valid_indicator=valid_indicator,
            padding_indicator=padding_indicator,
            padding_mask_dtype=padding_mask_dtype,
        )

    def _collate_label_map(
        self, batch: Sequence[EmbeddingAggregatorFitDatasetItem]
    ) -> Dict[str, Tensor]:
        """Creates a label map with the same keys as what's provided by the Dataset batch but with
        empty tensors to collect a batch of labels in.
        """

        label_map_keys = list(batch[0].label_map.keys())
        label_map_value_dtypes = [
            convert_to_torch_dtype(type(v)) for v in batch[0].label_map.values()
        ]

        label_map_empty_values = [
            torch.empty((len(batch),), dtype=label_dtype) for label_dtype in label_map_value_dtypes
        ]
        label_map_batch = dict(zip(label_map_keys, label_map_empty_values))

        # third loop through `batch` to collect labels if we are in a `fit` stage.
        # this logic could be combined with loop 1 or 2 but kept separate for now
        # for easy interpretation of logic since it should be low cost as well.
        for idx, item in enumerate(batch):
            for label_name, label_value in item.label_map.items():
                label_map_batch[label_name][idx] = label_value

        return label_map_batch

    def _collate_instance_mask_map(
        self, batch: Sequence[EmbeddingAggregatorFitDatasetItem]
    ) -> Dict[str, Tensor]:
        instance_mask_map_batch = {
            label_name: torch.empty(len(batch), dtype=torch.bool)
            for label_name in batch[0].instance_mask_map
        }

        for idx, item in enumerate(batch):
            for label_name, mask_value in item.instance_mask_map.items():
                instance_mask_map_batch[label_name][idx] = mask_value

        return instance_mask_map_batch

    def __call__(
        self,
        batch: Sequence[EmbeddingAggregatorFitDatasetItem],
    ) -> EmbeddingAggregatorFitCollatedItems:
        """Collates group embeddings into a batch tensor with padding if necessary, creates sequence
        length tensor which describes the true length of each sequence in the batch, creates a
        padding mask to indicate which positions in a sequence are valid and which are padding.

        Args:
            batch: List of items from Dataset. Length of batch size in loader.

        Returns:
            EmbeddingAggregatorFitCollatedItems: Output item with batch of group indices,
            embeddings, label map, padding mask, and sequence lengths.
        """
        helper_output = self._collater.collate_embeddings_mask_and_lengths(batch)

        # we expect there to be labels to deal with during a `fit` routine but not a `predict`
        # routine
        label_map_batch = self._collate_label_map(batch)
        instance_mask_map_batch = self._collate_instance_mask_map(batch)

        return EmbeddingAggregatorFitCollatedItems(
            group_indices=helper_output.group_indices,
            embeddings=helper_output.embeddings,
            label_map=label_map_batch,
            padding_mask=helper_output.padding_mask,
            sequence_lengths=helper_output.sequence_lengths,
            instance_mask_map=instance_mask_map_batch,
        )


class EmbeddingAggregatorVariableSizeSequencesPredictCollate:
    def __init__(
        self,
        *,
        order: str = 'bsf',
        valid_indicator: float = 0,
        padding_indicator: float = 1,
        padding_mask_dtype: torch.dtype = torch.uint8,
    ) -> None:
        """Collate functionality to be used as the a Dataloader's collate_fn during trainer.predict().

        Args:
            order: Desired order of group embeddings batch tensor. 'bsf' stands for Batch, Sequence,
                Features which is the required order for input tensors to models that use nn.Linear
                layers. Defaults to 'bsf'.
            valid_indicator: Value to use in padding masks to indicate valid positions. Defaults to
                0.
            padding_indicator: Value to use in padding masks to indicate padding positions. Defaults
                to 1.
            padding_mask_dtype: dtype of padding mask tensor. Defaults to torch.uint8.
        """
        self._collater = _EmbeddingAggregatorVariableSizeSequencesCollateHelper(
            order=order,
            valid_indicator=valid_indicator,
            padding_indicator=padding_indicator,
            padding_mask_dtype=padding_mask_dtype,
        )

    def __call__(
        self,
        batch: Sequence[EmbeddingAggregatorPredictDatasetItem],
    ) -> EmbeddingAggregatorPredictCollatedItems:
        """Collates group embeddings into a batch tensor with padding if necessary, creates sequence
        length tensor which describes the true length of each sequence in the batch, creates a
        padding mask to indicate which positions in a sequence are valid and which are padding.

        Args:
            batch: List of items from Dataset. Length of batch size in loader.

        Returns:
            EmbeddingAggregatorPredictCollatedItems: Output item with batch of group indices,
            embeddings, padding mask, and sequence lengths.
        """
        helper_output = self._collater.collate_embeddings_mask_and_lengths(batch)

        return EmbeddingAggregatorPredictCollatedItems(
            group_indices=helper_output.group_indices,
            embeddings=helper_output.embeddings,
            padding_mask=helper_output.padding_mask,
            sequence_lengths=helper_output.sequence_lengths,
        )
