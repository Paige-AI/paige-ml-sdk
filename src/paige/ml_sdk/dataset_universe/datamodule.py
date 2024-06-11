from pathlib import Path
from typing import Literal, Optional, Set, Union

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


def init_datamodule_from_dataset_filepaths(
    *,
    label_columns: Set[str],
    embeddings_filename_column: str,
    train_dataset_path: Optional[PathLike] = None,
    tune_dataset_path: Optional[PathLike] = None,
    test_dataset_path: Optional[PathLike] = None,
    embeddings_dir: Optional[PathLike] = None,
    train_embeddings_dir: Optional[PathLike] = None,
    tune_embeddings_dir: Optional[PathLike] = None,
    test_embeddings_dir: Optional[PathLike] = None,
    label_missing_value: int = -999,
    group_col: Optional[str] = None,
    filename_extension: str = '.pt',
    num_workers: int = 0,
    batch_size: int = 1,
    mode: Literal['csv', 'parquet'] = 'csv',
) -> AggregatorDataModule:
    """Constructs datasets from their filepaths and intializes an AggregatorDataModule instance.

    Args:
        label_columns: columns to use as targets. Must be consistent across
            train, tune, and test dataset csvs.
        embeddings_filename_column: column containing slide name. Used to
            identify embedding files. Must be consistent across train, tune,
            and test dataset csvs.
        train_dataset_path: Path to train dataset csv. If omitted, test
            dataset must be set.
        tune_dataset_path: Path to tune dataset csv. Must be set if
            train_dataset_path is set.
        test_dataset_path: Path to test dataset csv. Required if running
            inference on a pretrained model, but optional during training.
        embeddings_dir: Specifies a single directory to be used to load
            embeddings from. To be used if train, tune, and test embeddings
            all reside in a common directory. If this is not the case, use the
            train/tune/test_embeddings_dir arguments instead.
        train_embeddings_dir: Path to train embeddings dir. May be omitted if
            embeddings_dir is used, or if user wishes only to run inference
            on a test set (skipping training).
        tune_embeddings_dir: Path to tune embeddings dir. May be omitted if
            embeddings_dir is used, or if user wishes only to run inference
            on a test set (skipping training).
        test_embeddings_dir: Path to test embeddings dir. May be omitted if
            embeddings_dir is used, or during training if no test set is
            available.
        label_missing_value: Missing label value. Defaults to -999.
        group_col: column specifying level at which to perform aggregation. If
            omitted, defaults to slide-level.
        filename_extension: Filename extension. Defaults to .pt
        num_workers: number of dataloader workers. Defaults to 0.
        batch_size: batch size. Defaults to 1.
        mode: whether to load datasets from csv or parquet. Defaults to csv.

    Returns:
        AggregatorDataModule: a datamodule complete with a train dataloader, tune dataloader, and
            optionally a test dataloader.
    """
    use_global_embeddings_dir = embeddings_dir is not None
    use_specific_embeddings_dirs = bool(
        train_embeddings_dir or tune_embeddings_dir or test_embeddings_dir
    )
    if use_global_embeddings_dir and use_specific_embeddings_dirs:
        raise RuntimeError(
            'Either `embeddings_dir` should be specified, or'
            '`[train/tune/test]_embeddings_dir`, but not both.'
        )
    elif not use_global_embeddings_dir and not use_specific_embeddings_dirs:
        raise RuntimeError(
            'No embeddings directories were provided'
            'If all embeddings reside in a common directory, use `embeddings_dir`'
            'otherwise, specify [train/tune/test]_embeddings_dir individually.'
        )
    elif use_global_embeddings_dir:
        train_embeddings_dir = embeddings_dir
        tune_embeddings_dir = embeddings_dir
        test_embeddings_dir = embeddings_dir

    if train_dataset_path:
        train_dataset = EmbeddingDataset.from_filepath(
            train_dataset_path,
            train_embeddings_dir,
            label_columns,
            embeddings_filename_column,
            label_missing_value,
            group_col or embeddings_filename_column,  # fallback assumption: slide-level grouping
            filename_extension=filename_extension,
            mode=mode,
        )
    else:
        train_dataset = None

    if tune_dataset_path:
        tune_dataset = EmbeddingDataset.from_filepath(
            tune_dataset_path,
            tune_embeddings_dir,
            label_columns,
            embeddings_filename_column,
            label_missing_value,
            group_col or embeddings_filename_column,  # fallback assumption: slide-level grouping
            filename_extension=filename_extension,
            mode=mode,
        )
    else:
        tune_dataset = None

    if test_dataset_path:
        # makes same fallback mode assumptions as above
        test_dataset = EmbeddingDataset.from_filepath(
            test_dataset_path,
            test_embeddings_dir,
            label_columns,
            embeddings_filename_column,
            label_missing_value,
            group_col or embeddings_filename_column,
            filename_extension=filename_extension,
            mode=mode,
        )
    else:
        test_dataset = None

    if not (train_dataset or test_dataset):
        raise RuntimeError(
            'Neither train_dataset_path nor test_dataset_path were provided.'
            'At least one must be provided.'
        )

    if train_dataset and not tune_dataset:
        raise RuntimeError('Train dataset was provided without a tune dataset.')

    return AggregatorDataModule(
        train_dataset=train_dataset,
        tune_dataset=tune_dataset,
        test_dataset=test_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )
