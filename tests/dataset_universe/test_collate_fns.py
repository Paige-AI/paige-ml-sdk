from dataclasses import dataclass
from typing import List

import pytest
import torch
from _pytest.fixtures import SubRequest

from paige.ml_sdk.dataset_universe.collate_fns import (
    EmbeddingAggregatorFitCollatedItems,
    EmbeddingAggregatorPredictCollatedItems,
    EmbeddingAggregatorVariableSizeSequencesFitCollate,
    EmbeddingAggregatorVariableSizeSequencesPredictCollate,
)
from paige.ml_sdk.dataset_universe.datasets.fit import EmbeddingAggregatorFitDatasetItem
from paige.ml_sdk.dataset_universe.datasets.predict import EmbeddingAggregatorPredictDatasetItem


@dataclass
class FitTestCase:
    order: str
    batch: List[EmbeddingAggregatorFitDatasetItem]


@pytest.fixture
def f_agg_fit_dataset_item_batch_bsf() -> List[EmbeddingAggregatorFitDatasetItem]:
    # NOTE: changes to this fixture will need to be accounted for in
    # tests that use this fixture since they make strict assumptions
    # about what this fixture produces.
    group_index_1, group_index_2 = 0, 1
    ge1, ge2 = torch.randn(5, 16), torch.randn(3, 16)
    lm1, lm2 = {'label_0': 1, 'label_1': 5}, {'label_0': 0, 'label_1': 2}
    im1, im2 = {'label_0': True, 'label_1': False}, {'label_0': True, 'label_1': True}

    item1 = EmbeddingAggregatorFitDatasetItem(
        group_index=group_index_1, embeddings=ge1, label_map=lm1, instance_mask_map=im1
    )
    item2 = EmbeddingAggregatorFitDatasetItem(
        group_index=group_index_2, embeddings=ge2, label_map=lm2, instance_mask_map=im2
    )

    return [item1, item2]


@pytest.fixture
def fit_test_case(
    request: SubRequest,
    f_agg_fit_dataset_item_batch_bsf: List[EmbeddingAggregatorFitDatasetItem],
) -> FitTestCase:
    return FitTestCase('bsf', f_agg_fit_dataset_item_batch_bsf)


@dataclass
class PredictTestCase:
    order: str
    batch: List[EmbeddingAggregatorPredictDatasetItem]


@pytest.fixture
def f_agg_predict_dataset_item_batch_bsf() -> List[EmbeddingAggregatorPredictDatasetItem]:
    group_index_1, group_index_2 = 0, 1
    ge1, ge2 = torch.randn(5, 16), torch.randn(3, 16)
    item1 = EmbeddingAggregatorPredictDatasetItem(group_index=group_index_1, embeddings=ge1)
    item2 = EmbeddingAggregatorPredictDatasetItem(group_index=group_index_2, embeddings=ge2)

    return [item1, item2]


@pytest.fixture
def predict_test_case(
    request: SubRequest,
    f_agg_predict_dataset_item_batch_bsf: List[EmbeddingAggregatorPredictDatasetItem],
) -> PredictTestCase:
    return PredictTestCase('bsf', f_agg_predict_dataset_item_batch_bsf)


class Test_EmbeddingAggregatorVariableSizeSequenceFitCollate:
    def test_should_raise_error_for_unsupported_order(self) -> None:
        with pytest.raises(
            ValueError,
            match=r'Got unsupported `order` of .*. Expected one of the following options: .*',
        ):
            EmbeddingAggregatorVariableSizeSequencesFitCollate(order='this_should_break')

    def test_should_return_correct_shaped_tensor(self, fit_test_case: FitTestCase) -> None:
        order, batch = fit_test_case.order, fit_test_case.batch
        collate_fn = EmbeddingAggregatorVariableSizeSequencesFitCollate(order=order)
        batched_item = collate_fn(batch)

        assert isinstance(batched_item, EmbeddingAggregatorFitCollatedItems)
        assert torch.equal(batched_item.group_indices, torch.tensor((0, 1)))
        assert batched_item.embeddings.shape == (2, 5, 16)
        for _, label_values in batched_item.label_map.items():
            assert label_values.shape == ((2,) if order == 'bsf' else (1,))
        assert batched_item.padding_mask.shape == (2, 5)
        assert batched_item.sequence_lengths.shape == (2,)

    def test_should_return_correct_sequence_lengths(self, fit_test_case: FitTestCase) -> None:
        order, batch = fit_test_case.order, fit_test_case.batch
        collate_fn = EmbeddingAggregatorVariableSizeSequencesFitCollate(order=order)
        batched_item = collate_fn(batch)

        lengths = [5, 3]
        for expected, actual in zip(lengths, batched_item.sequence_lengths):
            assert actual == expected

    def test_should_return_correct_positions_indicated_as_padding_order_bsf(
        self, fit_test_case: FitTestCase
    ) -> None:
        order, batch = fit_test_case.order, fit_test_case.batch
        collate_fn = EmbeddingAggregatorVariableSizeSequencesFitCollate(
            order=order, valid_indicator=0, padding_indicator=1
        )
        batched_item = collate_fn(batch)

        # index 1 of the batch is what should have padding since index 0
        # is the longest sequence in this case
        assert batched_item.padding_mask[1, :3].sum() == 0
        # should sum to 2 since there are 3 valid idx's and 2 padding
        # idx's since it has to reach the same length as the largest
        # sequence in the batch which is 5 in this case
        assert batched_item.padding_mask[1, 3:].sum() == 2

    def test_should_return_no_padding_indicators_in_mask_of_longest_sequence(
        self, fit_test_case: FitTestCase
    ) -> None:
        order, batch = fit_test_case.order, fit_test_case.batch
        collate_fn = EmbeddingAggregatorVariableSizeSequencesFitCollate(
            order=order, valid_indicator=0, padding_indicator=1
        )
        batched_item = collate_fn(batch)

        # index 0 of the batch is the longest sequence here
        # padding mask should all be 0's then
        assert batched_item.padding_mask[0].sum() == 0

    def test_should_use_correct_padding_mask_values(self, fit_test_case: FitTestCase) -> None:
        order, batch = fit_test_case.order, fit_test_case.batch
        collate_fn = EmbeddingAggregatorVariableSizeSequencesFitCollate(
            order=order, valid_indicator=56, padding_indicator=24
        )
        batched_item = collate_fn(batch)

        assert batched_item.padding_mask[1, 0] == 56
        assert batched_item.padding_mask[1, 4] == 24

    def test_should_return_correct_instance_mask_tensors(self, fit_test_case: FitTestCase) -> None:
        order, batch = fit_test_case.order, fit_test_case.batch
        collate_fn = EmbeddingAggregatorVariableSizeSequencesFitCollate(order=order)
        batched_item = collate_fn(batch)

        for instance_mask_tensor in batched_item.instance_mask_map.values():
            assert len(instance_mask_tensor) == len(batch)
            assert instance_mask_tensor.dtype == torch.bool

        assert torch.equal(
            batched_item.instance_mask_map['label_0'], torch.tensor([1, 1], dtype=torch.bool)
        )
        assert torch.equal(
            batched_item.instance_mask_map['label_1'], torch.tensor([0, 1], dtype=torch.bool)
        )


class Test_EmbeddingAggregatorVariableSizeSequencePredictCollate:
    def test_should_raise_error_unsupported_order(self) -> None:
        with pytest.raises(
            ValueError,
            match=r'Got unsupported `order` of .*. Expected one of the following options: .*',
        ):
            EmbeddingAggregatorVariableSizeSequencesPredictCollate(order='this_should_break')

    def test_should_return_correct_shaped_tensor_order_bsf(
        self, predict_test_case: PredictTestCase
    ) -> None:
        order, batch = predict_test_case.order, predict_test_case.batch
        collate_fn = EmbeddingAggregatorVariableSizeSequencesPredictCollate(order=order)
        batched_item = collate_fn(batch)

        assert isinstance(batched_item, EmbeddingAggregatorPredictCollatedItems)
        assert batched_item.embeddings.shape == (2, 5, 16)
        assert batched_item.padding_mask.shape == (2, 5)
        assert batched_item.sequence_lengths.shape == (2,)

    def test_should_return_correct_sequence_lengths(
        self, predict_test_case: PredictTestCase
    ) -> None:
        order, batch = predict_test_case.order, predict_test_case.batch
        collate_fn = EmbeddingAggregatorVariableSizeSequencesPredictCollate(order=order)
        batched_item = collate_fn(batch)

        lengths = [5, 3]
        for expected, actual in zip(lengths, batched_item.sequence_lengths):
            assert actual == expected

    def test_should_return_correct_positions_indicated_as_padding_order_bsf(
        self, predict_test_case: PredictTestCase
    ) -> None:
        order, batch = predict_test_case.order, predict_test_case.batch
        collate_fn = EmbeddingAggregatorVariableSizeSequencesPredictCollate(
            order=order, valid_indicator=0, padding_indicator=1
        )
        batched_item = collate_fn(batch)

        # index 1 of the batch is what should have padding since index 0
        # is the longest sequence in this case
        assert batched_item.padding_mask[1, :3].sum() == 0
        # should sum to 2 since there are 3 valid idx's and 2 padding
        # idx's since it has to reach the same length as the largest
        # sequence in the batch which is 5 in this case
        assert batched_item.padding_mask[1, 3:].sum() == 2

    def test_should_return_no_padding_indicators_in_mask_of_longest_sequence(
        self, predict_test_case: PredictTestCase
    ) -> None:
        order, batch = predict_test_case.order, predict_test_case.batch
        collate_fn = EmbeddingAggregatorVariableSizeSequencesPredictCollate(
            order=order, valid_indicator=0, padding_indicator=1
        )
        batched_item = collate_fn(batch)

        # index 0 of the batch is the longest sequence here
        # padding mask should all be 0's then
        assert batched_item.padding_mask[0].sum() == 0

    def test_should_use_correct_padding_mask_values(
        self, predict_test_case: PredictTestCase
    ) -> None:
        order, batch = predict_test_case.order, predict_test_case.batch
        collate_fn = EmbeddingAggregatorVariableSizeSequencesPredictCollate(
            order=order, valid_indicator=56, padding_indicator=24
        )
        batched_item = collate_fn(batch)

        assert batched_item.padding_mask[1, 0] == 56
        assert batched_item.padding_mask[1, 4] == 24
