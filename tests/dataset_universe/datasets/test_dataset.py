from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from paige.ml_sdk.dataset_universe.collate_fns import (
    EmbeddingAggregatorFitCollatedItems,
    EmbeddingAggregatorVariableSizeSequencesFitCollate,
)
from paige.ml_sdk.dataset_universe.datasets.dataset import EmbeddingDataset
from paige.ml_sdk.dataset_universe.datasets.fit import EmbeddingAggregatorFitDatasetItem


@pytest.fixture
def f_df(f_path_to_dataset_csv: Path) -> pd.DataFrame:
    return pd.read_csv(f_path_to_dataset_csv, header=0)


@pytest.fixture
def f_path_to_dataset_parquet(f_path_to_dataset_csv: Path) -> Path:
    df = pd.read_csv(f_path_to_dataset_csv)
    out = f_path_to_dataset_csv.parent / 'dataset.parquet'
    df.to_parquet(out)
    return out


@pytest.fixture
def f_dataset(f_path_to_dataset_csv: Path, f_path_to_embeddings_dir: Path) -> EmbeddingDataset:
    return EmbeddingDataset.from_filepath(
        f_path_to_dataset_csv,
        embeddings_dir=f_path_to_embeddings_dir,
        label_columns={'cancer'},
        embeddings_filename_column='image_uri',
        group_column='group',
        label_missing_value=-999,
    )


class TestAggregatorDataset:
    def test_should_load_from_dataframe(
        self, f_path_to_dataset_csv: Path, f_path_to_embeddings_dir: Path
    ) -> None:
        EmbeddingDataset(
            pd.read_csv(f_path_to_dataset_csv),
            embeddings_dir=f_path_to_embeddings_dir,
            label_columns={'cancer'},
            embeddings_filename_column='image_uri',
            label_missing_value=-999,
        )

    def test_should_load_from_parquet(
        self, f_path_to_dataset_parquet: Path, f_path_to_embeddings_dir: Path
    ) -> None:
        EmbeddingDataset.from_filepath(
            f_path_to_dataset_parquet,
            embeddings_dir=f_path_to_embeddings_dir,
            label_columns={'cancer'},
            embeddings_filename_column='image_uri',
            label_missing_value=-999,
            mode='parquet',
        )

    def test_should_raise_AssertionError_if_nonexistant_target_column(
        self, f_df: pd.DataFrame, f_path_to_embeddings_dir: Path
    ) -> None:
        with pytest.raises(AssertionError):
            EmbeddingDataset(
                f_df,
                embeddings_dir=f_path_to_embeddings_dir,
                embeddings_filename_column='image_uri',
                label_columns={'i_dont_exist'},
                group_column='group',  # should cause error
                label_missing_value=-999,
            )

    def test_should_raise_AssertionError_if_slides_not_unique(
        self, f_df: pd.DataFrame, f_path_to_embeddings_dir: Path
    ) -> None:
        f_df['non_unique_slides'] = 'image_uri'
        with pytest.raises(AssertionError):
            EmbeddingDataset(
                f_df,
                embeddings_dir=f_path_to_embeddings_dir,
                embeddings_filename_column='non_unique_slides',  # should cause error
                label_columns={'cancer'},
                group_column='group',
                label_missing_value=300,
            )

    def test_should_give_expected_items(self, f_dataset: EmbeddingDataset) -> None:
        first_item, second_item = f_dataset[0], f_dataset[1]
        assert isinstance(first_item, EmbeddingAggregatorFitDatasetItem)
        assert isinstance(second_item, EmbeddingAggregatorFitDatasetItem)
        assert first_item.group_index == 0
        assert second_item.group_index == 1
        assert first_item.embeddings.shape == (2, 5)
        assert second_item.embeddings.shape == (1, 5)
        assert first_item.label_map == {'cancer': 0}
        assert second_item.label_map == {'cancer': 1}
        assert first_item.instance_mask_map == {'cancer': True}
        assert second_item.instance_mask_map == {'cancer': True}

    def test_should_iterate_with_dataloader(self, f_dataset: EmbeddingDataset) -> None:
        collate_fn = EmbeddingAggregatorVariableSizeSequencesFitCollate(
            order='bsf', valid_indicator=0, padding_indicator=1
        )
        dataloader = DataLoader(
            f_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
        )
        # There will only be one item, since batch_size=2 and there are two groups of embeddings.
        assert len(dataloader) == 1
        for item in dataloader:
            assert isinstance(item, EmbeddingAggregatorFitCollatedItems)
            assert torch.equal(item.group_indices, Tensor([0, 1]))
            assert item.embeddings.shape == (2, 2, 5)
            assert torch.equal(item.label_map['cancer'], Tensor([0, 1]))
            assert torch.equal(item.instance_mask_map['cancer'], Tensor([True, True]))
            assert torch.equal(item.padding_mask, Tensor([[0, 0], [0, 1]]))
            assert torch.equal(item.sequence_lengths, Tensor([2, 1]))
