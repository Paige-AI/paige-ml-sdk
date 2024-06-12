from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import zip_longest
from typing import Any, Dict, Generic, Protocol, Sequence, Set, Tuple, TypeVar, overload

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import Never

T_TENSOR = TypeVar('T_TENSOR', bound=Tensor)
T_NDARRAY = TypeVar('T_NDARRAY', bound=NDArray[Any])
T_TUPLE = TypeVar('T_TUPLE', bound=Tuple[Any, ...])
T_DICT = TypeVar('T_DICT', bound=Dict[Any, Any])
T_REST_contra = TypeVar('T_REST_contra', contravariant=True)
R_REST_co = TypeVar('R_REST_co', covariant=True)


class ReduceCollection(ABC, Generic[T_TENSOR, T_NDARRAY, T_REST_contra, R_REST_co]):
    """
    Reduce a collection of items.

    Arrays are reduced by overwriting abstract methods.
    - Tensor
    - NDArray

    Nested collections of arrays are reduced by distributing the reduction operation across
    their elements.
    - Tuple
    - Dict

    Note a fundamental difference in the treatment of arrays and the rest of the collections,
    especially tuples.
    The use cases for tuples and arrays are fundamentally different - tuples are closer
    to structs and dataclasses than to arrays (e.g. namedtuple, non-homogenious typing).
    """

    @abstractmethod
    def reduce_tensors(self, __tensors: Sequence[T_TENSOR]) -> T_TENSOR:
        ...

    @abstractmethod
    def reduce_ndarrays(self, __ndarrays: Sequence[T_NDARRAY]) -> T_NDARRAY:
        ...

    @abstractmethod
    def reduce_other(self, __objs: Sequence[T_REST_contra]) -> R_REST_co:
        ...

    def reduce_tuples(self, __tuples: Sequence[T_TUPLE]) -> T_TUPLE:
        """
        Reduce tuples by distributing the reduction operation over their items.

        Given
            [ (x1,y1), (x2,y2), ..., (xn,yn) ]
        reduction is
            (reduce([x1, x2, ..., xn]), reduce([y1, y2, ..., yn]))

        Preconditions:
            Assumes all tuples are of exact same size and type.
        """
        tuple_cls = __tuples[0].__class__
        tuple_args = [
            self.reduce(items_grouped_by_position)
            for items_grouped_by_position in zip_longest(*__tuples)
        ]
        return tuple.__new__(tuple_cls, tuple_args)  # type: ignore

    def reduce_dicts(self, __dicts: Sequence[T_DICT]) -> T_DICT:
        """
        Reduce dicts by distributing the reduction operation over the values that share
        the same key.

        Given
            [ {'a':x1,'b':y1}, {'a':x2,'b':y2}, ..., {'a':xn,'b':yn} ]
        reduction is
            {'a':reduce([x1, x2, ..., xn]), 'b': reduce([y1, y2, ..., yn])}

        Preconditions:
            Assumes all dictionaries have the exact same keys with values of the same type.
        """
        dict_cls = __dicts[0].__class__
        dict_keys: Set[Any] = set(__dicts[0])
        merged_dict = defaultdict(list)
        for d in __dicts:
            if dict_keys != set(d):
                raise KeyError(f"keys don't match: {dict_keys} and {set(d)}.")
            for k, v in d.items():
                merged_dict[k].append(v)
        return dict_cls({k: self.reduce(v) for k, v in merged_dict.items()})

    @overload
    def reduce(self, items: Sequence[T_TENSOR]) -> T_TENSOR:
        ...

    @overload
    def reduce(self, items: Sequence[T_NDARRAY]) -> T_NDARRAY:
        ...

    @overload
    def reduce(self, items: Sequence[T_TUPLE]) -> T_TUPLE:
        ...

    @overload
    def reduce(self, items: Sequence[T_DICT]) -> T_DICT:
        ...

    @overload
    def reduce(self, items: Sequence[T_REST_contra]) -> R_REST_co:
        ...

    def reduce(self, items: Sequence) -> Any:
        """
        Reduces items.

        Preconditions:
            Assumes all items are of the exact same type.
        """
        if len(items) == 0:
            raise ValueError('no items to reduce.')
        first_item = items[0]
        if isinstance(first_item, Tensor):
            return self.reduce_tensors(items)
        elif isinstance(first_item, np.ndarray):
            return self.reduce_ndarrays(items)
        elif isinstance(first_item, tuple):
            return self.reduce_tuples(items)
        elif isinstance(first_item, dict):
            return self.reduce_dicts(items)
        else:
            return self.reduce_other(items)


# Specific reducers


class SupportReduceSequence(Protocol[T_REST_contra, R_REST_co]):
    """
    A family of functions that handle reducing a sequence of objects
    into another object.
    Used to parametrise handling of non-Tensor and non-NDARray types
    in the implementaitons of ReduceCollection.
    """

    def __call__(self, __objs: Sequence[T_REST_contra]) -> R_REST_co:
        ...


T = TypeVar('T')


def fail(objs: Sequence[Any]) -> Never:
    raise TypeError(
        f'aggregation of {[type(o) for o in objs]} not supported out of the box. '
        'Instantiate Concat with a handler that implements SupportReduceSequence.'
    )


class Concat(ReduceCollection[Tensor, NDArray[Any], T_REST_contra, R_REST_co]):
    """
    Concatenate respective arrays in collections along 0th dimension and construct
    a new collection of the same type with resulting arrays.

    Preconditions:
        Assumes all items are of the exact same type.
        Assumes inner arrays have the same shape except 0th dimension.
        Assumes concatenated Tensors are on the same device.

    >>> ds = [
    ...     {'a': torch.tensor([1, 2]), 'b': np.array([9, 8])},
    ...     {'a': torch.tensor([3]), 'b': np.array([7])},
    ... ]
    >>> Concat(fail).reduce(ds)
    {'a': tensor([1, 2, 3]), 'b': array([9, 8, 7])}

    Args:
        reduce_other: a handler function that will be called to concatenate non-Tensors
            and non-NDArrays.
    """

    def __init__(self, reduce_other: SupportReduceSequence[T_REST_contra, R_REST_co]) -> None:
        self._reduce_rest = reduce_other

    def reduce_tensors(self, tensors: Sequence[Tensor]) -> Tensor:
        """
        Concatenate tensors along their first dimension.

        >>> Concat(fail).reduce_tensors([ torch.tensor([1, 2]), torch.tensor([3]) ])
        tensor([1, 2, 3])
        """
        tensor_dims = [len(t.shape) for t in tensors]
        unique_tensor_dims = list(np.unique(tensor_dims))
        if len(unique_tensor_dims) == 1:
            tensors = list(tensors)
        elif (len(unique_tensor_dims) == 2) and abs(
            unique_tensor_dims[0] - unique_tensor_dims[1]
        ) == 1:
            max_tensor_dim = max(unique_tensor_dims)
            tensors = [t if len(t.shape) == max_tensor_dim else t.unsqueeze(0) for t in tensors]
        else:
            raise RuntimeError(
                f'Will not be able to concatenate list of tensors with dimensions: {unique_tensor_dims}'
            )

        return torch.cat(tensors)

    def reduce_ndarrays(self, ndarrays: Sequence[NDArray[Any]]) -> NDArray[Any]:
        """
        Concatenate numpy ndarrays along their first dimension.

        Preconditions:
            Assumes all arrays have the same dtype.

        >>> Concat(fail).reduce_ndarrays([ np.array([1, 2]), np.array([3]) ])
        array([1, 2, 3])
        """
        return np.concatenate(ndarrays)

    def reduce_other(self, obj: Sequence[T_REST_contra]) -> R_REST_co:
        return self._reduce_rest(obj)


concat = Concat(fail).reduce
