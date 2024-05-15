from pathlib import Path

import pandas as pd
import pytest

from paige.ml_sdk.dataset_universe.collate_fns import EmbeddingAggregatorFitCollatedItems
from paige.ml_sdk.dataset_universe.datamodule import AggregatorDataModule
from paige.ml_sdk.dataset_universe.datasets.dataset import EmbeddingDataset


@pytest.fixture
def f_dataset(
    f_path_to_dataset_csv: Path,
    f_path_to_embeddings_dir: Path,
):
    return EmbeddingDataset(
        pd.read_csv(f_path_to_dataset_csv, header=0),
        embeddings_dir=f_path_to_embeddings_dir,
        label_columns={'cancer', 'precursor'},
        group_column='image_uri',
        label_missing_value=-999,
        embeddings_filename_column='image_uri',
    )


class TestAggregatorDatamodule:
    def test_should_create_train_and_tune_dataloaders(self, f_dataset: EmbeddingDataset) -> None:
        datamodule = AggregatorDataModule(
            train_dataset=f_dataset, tune_dataset=f_dataset, num_workers=2, batch_size=2
        )
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()

        for item_1, item_2 in zip(train_dataloader, val_dataloader):
            assert isinstance(item_1, EmbeddingAggregatorFitCollatedItems)
            assert isinstance(item_2, EmbeddingAggregatorFitCollatedItems)

        with pytest.raises(RuntimeError):
            datamodule.test_dataloader()

    def test_should_create_train_tune_test_dataloaders(self, f_dataset: EmbeddingDataset) -> None:
        datamodule = AggregatorDataModule(
            train_dataset=f_dataset,
            tune_dataset=f_dataset,
            test_dataset=f_dataset,
            num_workers=3,
            batch_size=1,
        )
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        test_dataloader = datamodule.test_dataloader()

        for item_1, item_2, item_3 in zip(train_dataloader, val_dataloader, test_dataloader):
            assert isinstance(item_1, EmbeddingAggregatorFitCollatedItems)
            assert isinstance(item_2, EmbeddingAggregatorFitCollatedItems)
            assert isinstance(item_3, EmbeddingAggregatorFitCollatedItems)
