from pathlib import Path

import pytest
import torch
from torch import Tensor

from paige.ml_sdk.dataset_universe.embedding_loader import (
    EmbeddingNotFoundError,
    FileSystemEmbeddingLoader,
)


@pytest.fixture
def f_embedding_loader(f_path_to_embeddings_dir: Path) -> FileSystemEmbeddingLoader:
    return FileSystemEmbeddingLoader(f_path_to_embeddings_dir)


class TestFileSystemEmbeddingLoader:
    def test_load_should_load_existing_embedding(
        self,
        f_embedding_loader: FileSystemEmbeddingLoader,
        f_path_to_embeddings_dir: Path,
    ) -> None:
        # these are the names of the slides written by the `f_path_to_embeddings_dir` fixture
        # in conftest.py
        for slide in ('slide_0.svs', 'slide_1.svs', 'slide_2.svs'):
            embeddings = f_embedding_loader.load(slide)
            expected_embeddings = torch.load(Path(f_path_to_embeddings_dir / f'{slide}.pt'))[
                'embeddings'
            ]
        assert isinstance(embeddings, Tensor)
        assert torch.equal(embeddings, expected_embeddings)

    def test_load_should_raise_file_not_found_error_for_nonexistent_embedding(
        self,
        f_embedding_loader: FileSystemEmbeddingLoader,
    ) -> None:
        loader = f_embedding_loader
        with pytest.raises(EmbeddingNotFoundError):
            loader.load('batman.svs')

    def test_lookup_embeddings_filepath_should_raise_error_if_no_match(
        self, f_embedding_loader: FileSystemEmbeddingLoader
    ) -> None:
        loader = f_embedding_loader
        with pytest.raises(EmbeddingNotFoundError):
            loader.lookup_embeddings_filepath('robin.svs')
