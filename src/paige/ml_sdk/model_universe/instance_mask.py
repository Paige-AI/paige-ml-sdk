from typing import Optional

from torch import Tensor


def apply_instance_mask(tensor: Tensor, *, instance_mask: Tensor) -> Tensor:
    """Apply `instance_mask` by indexing into the tensor.

    Precondition:

        instance_mask: boolean `Tensor` of size (B, ) where T (<= B) has entries with true.
        tensor: `Tensor` of size (B, ...).

    Postcondition:

        tensor: `Tensor` of size (T, ...).

    """
    _verify_instance_mask_invariants(tensor, instance_mask)

    return tensor[instance_mask]


def multiply_instance_mask_elementwise(tensor: Tensor, *, instance_mask: Tensor) -> Tensor:
    _verify_instance_mask_invariants(tensor, instance_mask)

    # A requirement for elementwise multiplication
    if tensor.shape != instance_mask.shape:
        raise ValueError(f'Tensor shape mismatch: {tensor.shape=} != {instance_mask.shape=}.')

    return tensor * instance_mask


def _verify_instance_mask_invariants(tensor: Tensor, instance_mask: Optional[Tensor]) -> None:
    if instance_mask is None:
        # None as runtime corrupt input is risky because tensor[None] will not raise.
        raise TypeError('instance_mask must be Tensor but received None.')

    if instance_mask.ndim != 1:
        raise ValueError(f'instance_mask should be 1d but received {instance_mask.ndim}d.')

    n_instances = tensor.shape[0]
    if len(instance_mask) != n_instances:
        raise ValueError(
            f'instance_mask size should match the number of instances in a batch but '
            f'got {instance_mask}-sized mask and {n_instances} instances.'
        )
