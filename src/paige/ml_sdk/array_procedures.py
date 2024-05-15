from typing import Optional, TypeVar, Union, overload

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

TensorOrNDArray = Union[np.ndarray, Tensor]


@overload
def to_numpy(array: TensorOrNDArray, dtype: None = None) -> NDArray:
    ...


@overload
def to_numpy(array: Tensor, dtype: Optional[torch.dtype] = None) -> NDArray:
    ...


def to_numpy(array: TensorOrNDArray, dtype: Optional[torch.dtype] = None) -> NDArray:
    """Converts tensor to `numpy.ndarray` with data on CPU."""
    if isinstance(array, np.ndarray):
        return array
    t = array.detach().to(device='cpu', dtype=dtype, copy=False)
    # need to cast since `.numpy()` returns Any
    res: NDArray = t.numpy()
    return res


T = TypeVar('T', bound=np.ndarray)
P = TypeVar('P', bound=Tensor)


@overload
def enforce_1d_shape(array: T) -> T:
    ...


@overload
def enforce_1d_shape(array: P) -> P:
    ...


def enforce_1d_shape(array: Union[T, P]) -> Union[Union[T, P], TensorOrNDArray]:
    """
    Converts (N, 1) array to (N,). No-op if the array is already (N,).
    Raises if the array is some other size.

    Args:
        array: A torch `Tensor` or a `np.ndarray`.
    """
    is_1d = array.ndim == 1
    is_n_by_1 = array.ndim == 2 and array.shape[1] == 1
    if not (is_1d or is_n_by_1):
        raise ValueError(
            'The shape of array is expected to be either (N,) or (N, 1), '
            f'received: {array.shape}.'
        )
    if is_n_by_1:
        res = array.squeeze(1)

        return res

    return array


@overload
def enforce_2d_shape(array: T) -> T:
    ...


@overload
def enforce_2d_shape(array: P) -> P:
    ...


def enforce_2d_shape(array: Union[T, P]) -> Union[Union[T, P], TensorOrNDArray]:
    """
    Checks the array is of shape [N, M]

    Args:
        array: A torch `Tensor` or a `np.ndarray`.
    """
    is_2d = array.ndim == 2
    if not is_2d:
        raise ValueError(
            'The shape of array is expected to be either (N, M)' f'received: {array.shape}.'
        )

    return array
