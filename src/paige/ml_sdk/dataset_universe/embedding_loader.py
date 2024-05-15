import logging
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol, Union

import torch
from torch import Tensor

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


def load_torch(p: Union[PathLike, Iterable[PathLike]]) -> Tensor:
    if isinstance(p, (str, Path)):
        return torch.load(p)['embeddings']  # type: ignore[no-any-return]
    else:
        embeddings = []
        for p_ in p:
            embedder_output = torch.load(p_)
            embeddings.append(embedder_output['embeddings'])
        return torch.cat(embeddings)


class EmbeddingNotFoundError(Exception):
    pass


# In case other Embedding Loader classes must be implemented
class EmbeddingLoader(Protocol):
    def load(self, __identifier: Union[str, Iterable[str]]) -> Tensor: ...

    def lookup_embeddings_filepath(self, embedding_filename: str) -> Optional[Path]: ...


class FileSystemEmbeddingLoader(EmbeddingLoader):
    """Loads embeddings files."""

    def __init__(
        self,
        embeddings_dir: Union[str, Path],
        load_func: Callable[[Union[Path, Iterable[Path]]], Tensor] = load_torch,
        extension: str = '.pt',
    ):
        """Initialize embedding loader.

        Args:
            embeddings_dir: Directory expected to contain all embedding files.
            load_func: Reads one or more embedding files, concatenates them in the latter case.
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.load_func = load_func
        self.extension = extension

    def load(self, embedding_filename_or_names: Union[str, Iterable[str]]) -> Tensor:
        """
        Loads embeddings for a given group name.

        Args:
            group_name: Identifies the name(s) of the embeddings filepath to be loaded.

        ..note:: `group_name` is can be an iterable of group_names, in which case multiple
          embeddings are loaded. A more accurate arg name would be `group_name_or_names`.
        """
        if isinstance(embedding_filename_or_names, str):
            embeddings = self.load_func(
                self.lookup_embeddings_filepath(embedding_filename_or_names)
            )
        else:
            embeddings = self.load_func(
                self.lookup_embeddings_filepath(p) for p in embedding_filename_or_names
            )
        return embeddings

    def lookup_embeddings_filepath(self, embedding_filename: str) -> Path:
        """
        Finds the embedding filepath.

        Args:
            reference: The name of the embeddings file

        Raises:
            EmbeddingNotFoundError: If no embeddings were found.

        Returns:
            The path to the embeddings file.
        """
        embedding_path = self.embeddings_dir / (embedding_filename + self.extension)
        if not embedding_path.exists():
            raise EmbeddingNotFoundError(f'embedding_path {embedding_path} does not exist')

        return embedding_path
