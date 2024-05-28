"""A drop-in replacement for `torch.distributed` package.

.. note::

    Motivation: `torch.distributed` collective APIs are still quite low a level for most
    end users and many issues and confusions can arise with subtle mistakes. Through this
    subpackage, we promote best practices towards safer, more efficient usage.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Type, cast

import torch
import torch.distributed as td
from environs import Env
from torch import Tensor

from paige.ml_sdk.distributed.torch import TorchProcessGroup

WORLD_MAIN_PROCESS_RANK = 0
DEFAULT_RAISE_IF_TD_INACTIVE = True
# node instantiation can vary by several minutes from pulling of
# docker containers and package installation so 30min is for safety
DEFAULT_GLOO_IPC_TIMEOUT_SECONDS = 1800

env = Env()
logger = logging.getLogger(__name__)


def is_td_active() -> bool:
    """Checks if torch.distributed environment is active.

    .. note::

        This will check against the default process group only.
    """
    is_active: bool = td.is_available() and td.is_initialized()
    return is_active


def is_world_main_process(
    group: Optional[TorchProcessGroup] = None,
    raise_if_td_inactive: bool = DEFAULT_RAISE_IF_TD_INACTIVE,
) -> bool:
    """Returns whether the current process is the world main process.

    Args:
        group: A descriptor for an active torch distributed process group.
        raise_if_td_inactive: Whether to raise when the default process group is not active.
            Set this to `True` when you need to guard against using this API without an
            active torch distributed process group.
    """
    global_rank: int = get_global_rank(group, raise_if_td_inactive=raise_if_td_inactive)
    return global_rank == WORLD_MAIN_PROCESS_RANK


def get_world_main_process_rank(raise_if_td_inactive: bool = DEFAULT_RAISE_IF_TD_INACTIVE) -> int:
    """Returns the global rank of the world main process.

    Args:
        raise_if_td_inactive: Whether to raise when the default process group is not active.
            Set this to `True` when you need to guard against using this API without an
            active torch distributed process group.
    """
    if raise_if_td_inactive:
        _assert_td_active()

    return WORLD_MAIN_PROCESS_RANK


def get_global_rank(
    group: Optional[TorchProcessGroup] = None,
    raise_if_td_inactive: bool = DEFAULT_RAISE_IF_TD_INACTIVE,
) -> int:
    """
    Returns the global rank of the current process given a Group.

    NOTE on raise_if_td_inactive:
    If the user doesn't want to raise an error in case torch.distributed is not
    initialised, we assume that the user accepts that the code can run outside
    a distributed process group.
    In this case we want to give the correct value of the rank, including in 1 GPU
    non-distributed, multi-GPU interactive, multi-GPU cluster, and multi-node settings.

    There are two known use cases for the user to wish to know the rank outside
    the process group, and we should support both consistently:
    - the code runs with 1 GPU without DDP, which means there's no process group;
      however the code in question was designed to be run in the process group as well.
      Therefore, in this setting, this function should imitate a no-op and return rank 0.
    - this function is called outside the process group in multi-process setting,
      e.g. before the process group is initialised.

    Args:
        group: A descriptor for an active torch distributed process group.
        raise_if_td_inactive: Whether to raise when the default process group is not active.
            Set this to `True` when you need to guard against using this API without an
            active torch distributed process group.

    Returns:
        global rank.

    Raises:
        RuntimeError:
        - if torch.distributed is inactive while raise_if_td_inactive is True.
        - if torch.distributed rank is undetermined (-1).
        - computed rank is not consistent with rank_zero_only.rank.
    """
    if raise_if_td_inactive:
        _assert_td_active()

    # torch.distributed
    if is_td_active():
        td_rank = td.get_rank(group=group)

        if td_rank == -1:
            raise RuntimeError(
                'The current process is not part of a given process group (if not '
                'specified, the default process group).'
            )

        return td_rank

    return WORLD_MAIN_PROCESS_RANK


def get_world_size(
    group: Optional[TorchProcessGroup] = None,
    raise_if_td_inactive: bool = DEFAULT_RAISE_IF_TD_INACTIVE,
) -> int:
    """Returns the global rank of the current process given a Group.

    Args:
        group: A descriptor for an active torch distributed process group.
        raise_if_td_inactive: Whether to raise when the default process group is not active.
            Set this to `True` when you need to guard against using this API without an
            active torch distributed process group.
    """
    if raise_if_td_inactive:
        _assert_td_active()

    if is_td_active():
        world_size = int(td.get_world_size(group=group))

        if world_size == -1:
            raise RuntimeError(
                'The current process is not part of a given process group (if not '
                'specified, the default process group).'
            )
    else:
        world_size = 1

    return world_size


def tensor_gather(
    tensor: Tensor, *, dim: int, dst: int = 0, group: Optional[TorchProcessGroup] = None
) -> Optional[Tensor]:
    """Gathers data from all processes into destination process.

    WARNING: Tensors must have the same dtype!

    Note this API is supported only in gloo backend.

    Args:
        tensor: A Torch tensor.
        dim: dimension to concat the tensors on.
        dst: Rank of the destination rank. Defaults to 0.
        group: Process group handle. Defaults to None.

    Returns:
        Returns tensor on dst rank, None on other ranks.
    """
    cur_rank_tensor = tensor.contiguous()

    original_device = cur_rank_tensor.device
    if is_nccl_backend(group=group):
        # Assumes using NCCL implies (multi-)GPU usage.
        device = torch.device('cuda', torch.cuda.current_device())
        cur_rank_tensor = cur_rank_tensor.to(device=device)

    original_dtype = cur_rank_tensor.dtype

    # TODO (MLF-278): extract this and apply consistently for all collective APIs subject to the dtype constraint.
    #          need to upcast: https://github.com/pytorch/pytorch/issues/24137
    if original_dtype == torch.bool:
        cur_rank_tensor = cur_rank_tensor.to(torch.uint8)

    size_stats_tg = _TensorGroupSizeStats.create_with_all_gather(cur_rank_tensor, group=group)
    size_stats_tg.check_concatenable(dim)

    world_size = td.get_world_size(group)
    rank = get_global_rank(group)
    padded_tg = _PaddedTensorGroup(
        cur_rank_tensor,
        size_stats_tg.max_size,
        rank=rank,
        world_size=world_size,
        size_list=size_stats_tg.size_list,
    )

    gather_list = padded_tg.padded_tensor_list if rank == dst else None
    td.gather(padded_tg.padded_cur_rank_tensor, gather_list, dst=dst, group=group)

    if rank != dst:
        return None
    output_tensor_list = padded_tg.unpad()
    size_stats_tg.check_sizes_match([t.size() for t in output_tensor_list])
    output_tensor_gathered: Tensor = torch.cat(output_tensor_list, dim=dim)

    if original_dtype == torch.bool:
        output_tensor_gathered = output_tensor_gathered.to(original_dtype)

    if original_device != output_tensor_gathered.device:
        output_tensor_gathered = output_tensor_gathered.to(device=original_device)

    return output_tensor_gathered


def is_nccl_backend(group: Optional[TorchProcessGroup] = None) -> bool:
    """
    Check if the backend for this process group is set to 'nccl'.

    Args:
        group: The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        bool: Boolean indication of 'nccl' backend is in use.
    """
    return bool(td.get_backend(group=group) == 'nccl')


@dataclass
class _TensorGroupSizeStats:
    """
    Computes size statistics tensors of a process group.

    Args:
        size_list: A list of sizes of a tensor group.
        max_size: A maximum tensor size of a tensor group.

    .. note::
        Some of the torch distributed collective APIs (esp.  gather/scatter) require
        tensors of each rank to be of the same size. Since we often have situations
        where different ranks have tensors of different sizes we need to pad tensors
        to have the same size (the maximum size of all). The current class is
        responsible for providing size-related information necessary for padding. Also
        see `_PaddedTensorGroup`.

    """

    size_list: Sequence[torch.Size]
    max_size: torch.Size

    @classmethod
    def create_with_all_gather(
        cls: Type['_TensorGroupSizeStats'],
        tensor: Tensor,
        group: Optional[TorchProcessGroup] = None,
    ) -> '_TensorGroupSizeStats':
        """Creates a size stats instance for group of tensors."""
        world_size = td.get_world_size(group)
        tensor_size_list: Sequence[Optional[torch.Size]] = [None] * world_size
        td.all_gather_object(tensor_size_list, tensor.size(), group=group)
        # all_gather replaced all None's
        tensor_size_list = cast(Sequence[torch.Size], tensor_size_list)
        max_size = max(tensor_size_list)
        return cls(size_list=tensor_size_list, max_size=max_size)

    def check_concatenable(self, dim: int) -> None:
        """Raises if sizes are not concatenable along dim."""
        size_bar_dim = [torch.Size([*s[:dim], *s[dim + 1 :]]) for s in self.size_list]
        if not self.do_sizes_match(size_bar_dim[:1], size_bar_dim[1:]):
            raise ValueError(f'tensors cannot be concatenated, dims: {self}')

    def check_sizes_match(self, b: Sequence[torch.Size]) -> None:
        """Raises if sizes of two tensors don't match."""
        if not self.do_sizes_match(self.size_list, b):
            raise ValueError(f'tensor sizes {self} do not match tensor sizes {b}.')

    @staticmethod
    def do_sizes_match(a: Sequence[torch.Size], b: Sequence[torch.Size]) -> bool:
        for a_size, b_size in zip(a, b):
            if a_size != b_size:
                return False
        return True


class _PaddedTensorGroup:
    def __init__(
        self,
        tensor: Tensor,
        padded_output_size: torch.Size,
        *,
        rank: int,
        world_size: int,
        size_list: Sequence[torch.Size],
    ) -> None:
        """Padding tensor groups.

        Args:
            tensor: A tensor contributed by the current rank process.
            padded_output_size: A target size of tensor after padding.
            rank: Rank of the default process group.
            world_size: World size of the default process group.
            size_list: A list of original (pre-pad) sizes of tensors.

        .. note::

            See the note in `_TensorGroupSizeStats` for the motivation.

        """
        self.non_padded_region_slices_list = self._create_non_padded_region_slices_list(
            size_list, world_size
        )
        self.padded_tensor_list = self._initialize_padded_tensor_list(
            tensor, padded_output_size, world_size
        )
        self._fill_current_rank_tensor(tensor, rank)
        self.padded_cur_rank_tensor = self.padded_tensor_list[rank]

    def _initialize_padded_tensor_list(
        self, tensor: Tensor, padded_output_size: torch.Size, world_size: int
    ) -> List[Tensor]:
        """Creates a list of padded tensors."""
        padded_tensor_list = [tensor.new_empty(size=padded_output_size) for _ in range(world_size)]

        return padded_tensor_list

    def _create_non_padded_region_slices_list(
        self, size_list: Sequence[torch.Size], world_size: int
    ) -> List[Tuple[slice, ...]]:
        """Create region slices that should be filled in with actual data."""
        # Not a great way to satisfy mypy cleanly.
        non_padded_region_slices_list: List[Tuple[slice, ...]] = [None] * world_size  # type: ignore

        for rank in range(world_size):
            non_padded_region_slices = tuple((slice(None, size) for size in size_list[rank]))
            non_padded_region_slices_list[rank] = non_padded_region_slices
        return non_padded_region_slices_list

    def _fill_current_rank_tensor(self, tensor: Tensor, rank: int) -> None:
        """Fill actual data."""
        # these are all views and in-place.
        padded_cur_rank_tensor = self.padded_tensor_list[rank]
        non_padded_region_slices = self.non_padded_region_slices_list[rank]
        padded_cur_rank_tensor[non_padded_region_slices] = tensor

    def unpad(self) -> List[Tensor]:
        """Converts list of padded tensors to their respective original sizes."""
        n_tensors = len(self.padded_tensor_list)
        # Not a great way to satisfy mypy cleanly.
        output_tensor_list: List[Tensor] = [None] * n_tensors  # type: ignore
        for rank in range(n_tensors):
            padded_tensor = self.padded_tensor_list[rank]
            non_padded_indices = self.non_padded_region_slices_list[rank]
            output_tensor_list[rank] = padded_tensor[non_padded_indices]

        return output_tensor_list

    def padded_tensor(self, rank: int) -> Tensor:
        """Returns a padded tensor of a given rank."""
        return self.padded_tensor_list[rank]


def _assert_td_active() -> None:
    """Asserts the default torch distributed process group is active."""
    if not is_td_active():
        raise RuntimeError(
            'The default process group has not been initialized. Perhaps forgot to set '
            'up a process group using `torch.distributed.init_process_group`?'
        )
