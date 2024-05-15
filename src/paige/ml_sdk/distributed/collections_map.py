from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, cast, overload

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, from_numpy

from paige.ml_sdk.array_procedures import to_numpy
from paige.ml_sdk.distributed.collective import (
    get_world_main_process_rank,
    is_td_active,
    is_world_main_process,
    tensor_gather,
)

# Map-able types
MAPPABLE = (Tensor, np.ndarray, tuple, dict)

T_TUPLE = TypeVar('T_TUPLE', bound=Tuple[Any, ...])
T_DICT = TypeVar('T_DICT', bound=Dict[Any, Any])

R_TENSOR = TypeVar('R_TENSOR')
R_NDARRAY = TypeVar('R_NDARRAY')


class MapCollection(ABC, Generic[R_TENSOR, R_NDARRAY]):
    """
    Map arrays (Tensor, NDArray) in a collection recursively.

    Arrays are mapped by applying a mapping directly on them.
    - Tensor
    - NDArray

    Nested collections of arrays are mapped by traversing the collection recursively
    and applying the mapping over all of the the arrays.
    - Tuple
    - Dict

    Note a fundamental difference in the treatment of arrays and the rest of the collections,
    especially tuples.
    The use cases for tuples and arrays are fundamentally different - tuples are closer
    to structs and dataclasses than to arrays (e.g. namedtuple, non-homogenious typing).
    """

    @abstractmethod
    def map_tensor(self, __tensor: Tensor) -> R_TENSOR:
        """Map Tensor to another Tensor or something else."""
        ...

    @abstractmethod
    def map_ndarray(self, __ndarray: NDArray[Any]) -> R_NDARRAY:
        """Map NDArray to another NDArray or something else."""
        ...

    def map_tuple(self, __atuple: T_TUPLE) -> T_TUPLE:
        """Apply mapping to each item in the tuple."""
        # TODO:
        #   /app/src/paige/ml/collections_map.py:59:50 - error: Argument of type "list[R_TENSOR@MapCollection]" cannot be assigned to parameter "__iterable" of type "Iterable[_T_co@tuple]" in function "__new__"
        #       TypeVar "_T_co@Iterable" is covariant
        #           Type "R_TENSOR@MapCollection" cannot be assigned to type "_T_co@tuple" (reportGeneralTypeIssues)
        return tuple.__new__(__atuple.__class__, [self.map(v) for v in __atuple])  # type: ignore

    def map_dict(self, __adict: T_DICT) -> T_DICT:
        """Apply mapping over values of each key."""
        return __adict.__class__({k: self.map(v) for k, v in __adict.items()})

    @overload
    def map(self, item: Tensor) -> R_TENSOR: ...

    @overload
    def map(self, item: NDArray[Any]) -> R_NDARRAY: ...

    @overload
    def map(self, item: T_TUPLE) -> T_TUPLE: ...

    @overload
    def map(self, item: T_DICT) -> T_DICT: ...

    def map(self, item: Any) -> Any:
        if isinstance(item, Tensor):
            return self.map_tensor(item)
        elif isinstance(item, np.ndarray):
            return self.map_ndarray(item)
        elif isinstance(item, tuple):
            return self.map_tuple(item)
        elif isinstance(item, dict):
            return self.map_dict(item)
        else:
            raise TypeError(f'unsupported type to map: {type(item)}.')


# Specific mappers


class TensorTicket(Tensor):
    """
    A ticket that poses as torch Tensor.
    Used to temporarily replace real Tensors in a collection.

    .. note::
        The ticket needs to be a Tensor so that it is processed by `map_tensor` method
        when Tensors are withdrawn from the buffer (see `ToCPU.WithdrawTensors` and
        `Gather.WithdrawArrays`).
    """

    ticket_number: int

    def __new__(cls, ticket_number: int) -> 'TensorTicket':
        c = super().__new__(cls)
        c.ticket_number = ticket_number
        return c

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return f'{self.__class__.__name__}({self.ticket_number})'


class NDArrayTicket(np.ndarray):
    """
    A ticket that poses as numpy NDArray.
    Used to temporarily replace real NDArrays in a collection.

    .. note:
        The ticket needs to be a NDArray so that it is processed by `map_ndarray` method
        when NDArrays are withdrawn from the buffer (see `Gather.WithdrawArrays`).
    """

    ticket_number: int

    def __new__(cls, ticket_number: int) -> 'NDArrayTicket':
        c = super().__new__(cls, ())
        c.ticket_number = ticket_number
        return c

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.ticket_number})'


# To CPU


T = TypeVar('T')


class ToCPU(ABC):
    """Move every GPU Tensor in a collection to CPU memory."""

    @abstractmethod
    def to_cpu(self, item: T) -> T: ...


class ToCPUPerTensor(ToCPU, MapCollection[Tensor, NDArray[Any]]):
    """
    Move every GPU Tensor in a collection to CPU memory, one tensor at a time.
    Other data is left untouched.
    """

    def map_tensor(self, tensor_leaf: Tensor) -> Tensor:
        """Move Tensor to CPU; if already on CPU, this is a pass-through."""
        return tensor_leaf.to('cpu')

    def map_ndarray(self, ndarray_leaf: NDArray[Any]) -> NDArray[Any]:
        """Leave ndarray as-is since it's already in CPU memory."""
        return ndarray_leaf

    def to_cpu(self, item: T) -> T:
        """
        Move all GPU Tensors in a collection to CPU memory, one tensor at a time.

        Args:
            item: one of the following:
            - nested Dict or Tuple or combination of both, with leafs as Tensors and/or NDArrays
            - Tensor
            - NDArray
        """
        if not isinstance(item, MAPPABLE):
            raise TypeError(f'moving to cpu unsupported for: {type(item)}, expected MAPPABLE.')
        return cast(T, self.map(item))


to_cpu_per_tensor = ToCPUPerTensor().to_cpu


class ToCPUStackTensors(ToCPU):
    """Move all GPU Tensors in a collection to CPU memory, all in one go."""

    class DepositTensors(MapCollection[Tensor, NDArray[Any]]):
        def __init__(self, buffer: List[Tensor]) -> None:
            """
            Collect all Tensors in the collection into a buffer.
            Replace collected Tensors with a ticket which refers to a position
            in the buffer where the underlying Tensor was placed.

            Args:
                buffer: external container whither to place all Tensors in the collection.
            """
            self.buffer = buffer

        def map_tensor(self, tensor_leaf: Tensor) -> Tensor:
            """
            Add tensor to the buffer pending moving to CPU.
            Return position ticket of the tensor in the buffer.

            If Tensor is already on CPU, it is left in the collection intact.
            """
            if tensor_leaf.device == torch.device('cpu'):
                return tensor_leaf
            next_index = len(self.buffer)
            self.buffer.append(tensor_leaf)
            return TensorTicket(next_index)

        def map_ndarray(self, ndarray_leaf: NDArray[Any]) -> NDArray[Any]:
            """Leave ndarray as-is since it's already in CPU memory."""
            return ndarray_leaf

    class WithdrawTensors(MapCollection[Tensor, NDArray[Any]]):
        def __init__(self, buffer: Tensor) -> None:
            """
            Use tickets in the collection to "withdraw" Tensors from the buffer
            and place them in their original leaf location in the collection.

            Args:
                buffer: buffer with Tensors from the original collection, located along
                    the 0th dimension of the buffer; equivalent to the buffer in DepositTensors.
            """
            self.buffer = buffer

        def map_tensor(self, tensor_leaf_or_ticket: Tensor) -> Tensor:
            """
            Retrieve Tensor from the buffer by the ticket.

            If Tensor was already on the CPU in the original collection, this will be that Tensor
            and thus should be left intact.
            """
            if isinstance(tensor_leaf_or_ticket, TensorTicket):
                # ticket
                return self.buffer[tensor_leaf_or_ticket.ticket_number]
            # leaf (original Tensor)
            return tensor_leaf_or_ticket

        def map_ndarray(self, ndarray_leaf: NDArray[Any]) -> NDArray[Any]:
            """Leave ndarray as-is since it's already in CPU memory."""
            return ndarray_leaf

    def to_cpu(self, item: T) -> T:
        """
        Move all GPU Tensors in a collection to CPU memory, all in one go.

        Preconditions:
            Assumes all GPU Tensors are of the same shape.

        .. note::
            This operation allocates one big tensor, copies individual tensors from the collection
            into it, and moves it with one call to `cpu`.
            It then reconstructs the original collection.

        Args:
            item: one of the following:
            - nested Dict or Tuple or combination of both, with leafs as Tensors and/or NDArrays
            - Tensor
            - NDArray
        """
        if not isinstance(item, MAPPABLE):
            raise TypeError(f'moving to cpu unsupported for: {type(item)}, expected MAPPABLE.')
        with self._tensor_buffer() as buf:
            deposit_tensors = self.DepositTensors(buf)
            item_with_tickets = deposit_tensors.map(item)
            tensor_stack_cpu = self._move_tensors_to_cpu(deposit_tensors.buffer)
            withdraw_tensors = self.WithdrawTensors(tensor_stack_cpu)
            reconstructed_item = withdraw_tensors.map(item_with_tickets)
            return cast(T, reconstructed_item)  # mypy doesn't recognise this is equiv. to 'item'

    @staticmethod
    def _move_tensors_to_cpu(buffer: List[Tensor]) -> Tensor:
        if len(buffer) == 0:
            return torch.empty(0)
        return torch.stack(buffer).cpu()

    @contextmanager
    def _tensor_buffer(self) -> Iterator[List[Tensor]]:
        buffer: List[Tensor] = []
        try:
            yield buffer
        finally:
            buffer.clear()


to_cpu_stack_tensors = ToCPUStackTensors().to_cpu


# Gather


class GatherPerTensor:
    """
    Gather Tensors and NDArrays in a collection distributed across multiple ranks
    on the main rank, one array on each rank at a time.
    """

    class DepositArrays(MapCollection[TensorTicket, NDArrayTicket]):
        def __init__(self, buffer: List[Tensor]) -> None:
            """
            Collect all Tensors and NDArrays (as Tensors) in the collection into a buffer.
            Replace collected Tensors and NDArrays with a respective ticket which refers
            to a position in the buffer where the underlying Tensor or NDArrays was placed.

            Args:
                buffer: external container whither to place all Tensors and NDArrays (as Tensors)
                    in the collection.
            """
            self.buffer = buffer

        def map_tensor(self, tensor_leaf: Tensor) -> 'TensorTicket':
            """Add Tensor to the buffer and leave a ticket on its place."""
            if tensor_leaf.device != torch.device('cpu'):
                raise RuntimeError('found tensors on GPU, can gather only tensors on CPU.')
            next_index = len(self.buffer)
            self.buffer.append(tensor_leaf)
            return TensorTicket(next_index)

        def map_ndarray(self, ndarray_leaf: NDArray[Any]) -> 'NDArrayTicket':
            """Add NDArray to the buffer (as Tensor) and leave a ticket on its place."""
            next_index = len(self.buffer)
            self.buffer.append(from_numpy(ndarray_leaf))
            return NDArrayTicket(next_index)

    class WithdrawArrays(MapCollection[Tensor, NDArray[Any]]):
        def __init__(self, buffer: List[Optional[Tensor]]) -> None:
            """
            Use tickets in the collection to "withdraw" Tensors and NDArrays from the buffer
            and place them in their original leaf location in the collection.

            Args:
                buffer: buffer with Tensors and/or NDArrays (as Tensors) from the original collection,
                    located along the 0th dimension of the buffer; equivalent to the buffer in DepositArrays.
            """
            self.buffer = buffer

        def map_tensor(self, tensor_ticket: Tensor) -> Tensor:
            """Retrieve Tensor from the buffer using the ticket."""
            if not isinstance(tensor_ticket, TensorTicket):
                raise TypeError(
                    'all Tensors should have been replaced with TensorTicket, '
                    f'got: {type(tensor_ticket)}.'
                )
            t = self.buffer[tensor_ticket.ticket_number]
            if t is None:
                raise ValueError('None gathered on main rank, expected Tensor.')
            return t

        def map_ndarray(self, ndarray_ticket: NDArray[Any]) -> NDArray[Any]:
            """
            Retrieve NDarray from the buffer using the ticket.

            Since NDArray is represented as a Tensor in the buffer, convert it back to NDArray.
            """
            if not isinstance(ndarray_ticket, NDArrayTicket):
                raise TypeError(
                    'all NDArrays should have been replaced with NDArrayTicket, '
                    f'got: {type(ndarray_ticket)}.'
                )
            t = self.buffer[ndarray_ticket.ticket_number]
            if t is None:
                raise ValueError('None gathered on main rank, expected Tensor.')
            return to_numpy(t)

    def gather(self, item: T) -> Optional[T]:
        """
        Gather Tensors and NDArrays in a collection distributed across multiple ranks
        on the main rank, one array on each rank at a time.

        Preconditions:
            Tensors in the same position in each collection across the group to have
            the same shape, except 0th dimension which can be different.
            All tensors are on CPU.

        Args:
            item: one of the following:
            - nested Dict or Tuple or combination of both, with leafs as Tensors and/or NDArrays
            - Tensor
            - NDArray
        """
        if not is_td_active():
            return item
        if not isinstance(item, MAPPABLE):
            raise TypeError(f'gather unsupported for: {type(item)}, expected MAPPABLE.')
        with self._tensor_buffer() as buf:
            deposit_arrays = self.DepositArrays(buf)
            item_with_tickets = deposit_arrays.map(item)
            buffer_on_main_rank = self._gather_tensors(deposit_arrays.buffer)
            if not is_world_main_process():
                if any(buffer_on_main_rank):
                    raise ValueError('tensors gathered on non-main rank, expected None.')
                return None
            withdraw_arrays = self.WithdrawArrays(buffer_on_main_rank)
            reconstructed_item = withdraw_arrays.map(item_with_tickets)
            return cast(T, reconstructed_item)  # mypy doesn't recognise this is equiv. to 'item'

    @staticmethod
    def _gather_tensors(buffer: List[Tensor]) -> List[Optional[Tensor]]:
        buffer_on_main_rank: List[Optional[Tensor]] = []
        # NOTE: we use the default backend here. gloo new_process_group got timeout here chronically.
        for t in buffer:
            gathered_tensor = tensor_gather(t, dim=0, dst=get_world_main_process_rank(), group=None)
            buffer_on_main_rank.append(gathered_tensor)
        return buffer_on_main_rank

    @contextmanager
    def _tensor_buffer(self) -> Iterator[List[Tensor]]:
        buffer: List[Tensor] = []
        try:
            yield buffer
        finally:
            buffer.clear()


gather_per_tensor = GatherPerTensor().gather
