from typing import Dict, Type, Union

import numpy as np
import torch

TorchSupportedDtypes = Union[
    Type[np.bool_],
    Type[np.uint8],
    Type[np.int8],
    Type[np.int16],
    Type[np.int32],
    Type[np.int64],
    Type[np.float16],
    Type[np.float32],
    Type[np.float64],
    Type[np.complex64],
    Type[np.complex128],
    Type[int],
    Type[float],
]

# ref: https://github.com/pytorch/pytorch/blob/32e790997ba4a10503ee5041b257fa8a5e8e5df9/torch/testing/_internal/common_utils.py#L730
NUMPY_TO_TORCH_DTYPE_DICT: Dict[TorchSupportedDtypes, torch.dtype] = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

PYTHON_TO_TORCH_DTYPE_DICT: Dict[TorchSupportedDtypes, torch.dtype] = {
    bool: torch.bool,
    int: torch.int64,
    float: torch.float64,
}

ALL_TO_TORCH_DTYPE_DICT = {**NUMPY_TO_TORCH_DTYPE_DICT, **PYTHON_TO_TORCH_DTYPE_DICT}


def convert_to_torch_dtype(dtype: Union[TorchSupportedDtypes, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype

    resolved_dtype = ALL_TO_TORCH_DTYPE_DICT.get(dtype)
    if resolved_dtype is None:
        raise ValueError(
            f'Expected a Numpy or Python dtype that has a corresponding PyTorch dtype. Got {dtype}.'
        )

    return resolved_dtype
