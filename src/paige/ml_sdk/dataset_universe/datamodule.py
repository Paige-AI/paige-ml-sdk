from pathlib import Path
from typing import Optional, Set, Union

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from paige.ml_sdk.dataset_universe.collate_fns import (
    EmbeddingAggregatorVariableSizeSequencesFitCollate,
)
from paige.ml_sdk.dataset_universe.datasets.dataset import EmbeddingDataset

PathLike = Union[str, Path]


class AggregatorDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        train_dataset: EmbeddingDataset,
        tune_dataset: EmbeddingDataset,
        num_workers: int,
        batch_size: int,
        test_dataset: Optional[EmbeddingDataset] = None,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.tune_dataset = tune_dataset
        self.test_dataset = test_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size

    def make_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=EmbeddingAggregatorVariableSizeSequencesFitCollate(),
        )

    def train_dataloader(self) -> DataLoader:
        return self.make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader(self.tune_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError('attempting test loop but test dataloader is not defined.')
        return self.make_dataloader(self.test_dataset, shuffle=False)


def init_aggregator_datamodule(
    *,
    train_dataset_path: PathLike,
    tune_dataset_path: PathLike,
    train_embeddings_dir: PathLike,
    label_columns: Set[str],
    embeddings_filename_column: str,
    test_dataset_path: Optional[PathLike] = None,
    tune_embeddings_dir: Optional[PathLike] = None,
    test_embeddings_dir: Optional[PathLike] = None,
    label_missing_value: int = -999,
    group_col: Optional[str] = None,
    filename_extension: str = '.pt',
    num_workers: int = 0,
    batch_size: int = 1,
) -> AggregatorDataModule:
    """Initializes an AggregatorDataModule.

    Args:
        train_dataset_path: Path to train dataset csv.
        tune_dataset_path: Path to tune dataset csv.
        train_embeddings_dir: Path to train embeddings dir.
        label_columns: columns to use as targets. Must be consistent across train, tune, and test csvs.
        embeddings_filename_column: column containing slide name. Used to identify embedding files. Must be
            consistent across train, tune, and test csvs.
        test_dataset_path: Path to train dataset csv. If omitted, testing is skipped.
        tune_embeddings_dir: Path to tune embeddings dir. If omitted, embeddings are assumed to be
            under `train_embeddings_dir`.
        test_embeddings_dir: Path to test embeddings dir. If omitted, embeddings are assumed to be
            under `train_embeddings_dir`.
        label_missing_value: Missing label value. Defaults to -999.
        group_col: column specifying level at which to perform aggregation. If omitted, defaults to
            slide-level.
        num_workers: number of dataloader workers. Defaults to 0.
        batch_size: batch size. Defaults to 1.

    Returns:
        AggregatorDataModule: a datamodule complete with a train dataloader, tune dataloader, and
            optionally a test dataloader.
    """
    train_dataset = EmbeddingDataset.from_csv(
        train_dataset_path,
        train_embeddings_dir,
        label_columns,
        embeddings_filename_column,
        label_missing_value,
        group_col or embeddings_filename_column,  # fallback assumption: slide-level grouping
        filename_extension=filename_extension,
    )

    tune_dataset = EmbeddingDataset.from_csv(
        tune_dataset_path,
        tune_embeddings_dir or train_embeddings_dir,  # fallback assumption: share w/ train dir
        label_columns,
        embeddings_filename_column,
        label_missing_value,
        group_col or embeddings_filename_column,
        filename_extension=filename_extension,
    )

    if test_dataset_path:
        # makes same fallback mode assumptions as above
        test_dataset = EmbeddingDataset.from_csv(
            test_dataset_path,
            test_embeddings_dir or train_embeddings_dir,
            label_columns,
            embeddings_filename_column,
            label_missing_value,
            group_col or embeddings_filename_column,
            filename_extension=filename_extension,
        )
    else:
        test_dataset = None

    return AggregatorDataModule(
        train_dataset=train_dataset,
        tune_dataset=tune_dataset,
        test_dataset=test_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )
