from torch import Tensor

from paige.ml_sdk.array_procedures import enforce_1d_shape
from paige.ml_sdk.dataset_universe.collate_fns import EmbeddingAggregatorFitCollatedItems
from paige.ml_sdk.model_universe.aggregator import AggregatorOutput
from paige.ml_sdk.model_universe.instance_mask import multiply_instance_mask_elementwise

__all__ = [
    'AggregatorLossComputer',
]


from typing import Optional, Protocol, Tuple

import torch
import torch.nn as nn


class InputTargetLossFamily(Protocol):
    # the contract over reduction data member we inherit from pytorch `_Loss`.
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py#L13
    reduction: str

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...


def _enforce_loss_fn_reduction_none(loss_fn: InputTargetLossFamily) -> InputTargetLossFamily:
    """Make a loss function compute elementwise."""
    # Taking a calculated risk it's better to enforce this blindly rather than asking users to explicitly
    # set the reduction parameter for their losses to avoid breaking changes in existing configs.
    # once this feature stabilizes, we can do global cleanup.
    # TODO (dh): remove this after patching all loss computer usage (in hydra) with reduction=none
    loss_fn.reduction = 'none'
    return loss_fn


def _reduce_loss(unreduced_loss: Tensor, instance_mask: Tensor) -> Tensor:
    """Reduces elementwise loss values to a scalar-valued loss."""
    masked_loss = multiply_instance_mask_elementwise(
        enforce_1d_shape(unreduced_loss), instance_mask=instance_mask
    )
    n_relevant_instances = instance_mask.sum()

    if n_relevant_instances == 0:
        # If all missing, avoid zero division error.
        return masked_loss.sum()
    else:
        return torch.divide(masked_loss.sum(), n_relevant_instances)


class AggregatorLossComputer(nn.Module):
    def __init__(
        self,
        loss_fn: InputTargetLossFamily,
        match_target_dim_to_input_dim: bool,
        *,
        target_to_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """A handler for computing a loss function using head logits.

        ..note:: Only loss functions which operate on logits may be used; if a loss function
            operating on activations is desired, this class should be refactored to handle
            losses operating on either heads_logits or heads_activations.

        Args:
            loss_fn: A loss function whose `__call__` signature includes input and target where
                input is logits (e.g. `BCELossWithLogitsLoss`, `CrossEntropyLoss`).
            match_target_dim_to_input_dim: Whether to match the dimension of targets to that of
                input. See the documentation of `_match_target_dim_to_input_dim_at_most_2d_case`.
            target_to_dtype: A `torch.dtype` to cast targets as. This is necessary when a loss
                function requires targets to be in a certain dtype (e.g. `CrossEntropyLoss` needs
                `torch.long`, `BCELoss` needs `torch.float`). Defaults to None.
        """
        super().__init__()
        self._loss_fn = _enforce_loss_fn_reduction_none(loss_fn)
        self._match_target_dim_to_input_dim = match_target_dim_to_input_dim
        self._target_to_dtype = target_to_dtype

    def __call__(  # type: ignore
        self,
        __batch: EmbeddingAggregatorFitCollatedItems,
        __output: AggregatorOutput,
        __label_name: str,
    ) -> Tensor:
        out: Tensor = super().__call__(__batch, __output, __label_name)
        return out

    @torch.jit.unused
    def forward(  # type: ignore
        self,
        batch: EmbeddingAggregatorFitCollatedItems,
        output: AggregatorOutput,
        label_name: str,
    ) -> Tensor:
        """Computes loss for a label."""
        head_logits = output.heads_logits[label_name]
        targets = batch.label_map[label_name]
        mask = batch.instance_mask_map[label_name]

        input_, target = _preprocess_for_loss_fn_receiving_input_target(
            head_logits,
            targets,
            match_target_dim_to_input_dim=self._match_target_dim_to_input_dim,
            target_to_dtype=self._target_to_dtype,
        )

        # If you get: IndexError: Target <missing_label_val> is out of bounds.
        # hint: do CrossEntropyLoss(ignore_index=<missing_label_val>)
        loss = torch.zeros_like(target, dtype=torch.float)
        loss[mask] = self._loss_fn(input=input_[mask], target=target[mask])
        return _reduce_loss(loss, instance_mask=mask)


def _match_target_dim_to_input_dim_at_most_2d_case(inputs: Tensor, targets: Tensor) -> Tensor:
    """Reshape target dim according to the input dim.

    This is often necessary because different loss functions have different shape requirements for
    targets in relation to inputs. Whether matching should be done or not is a decision to be made
    by the constructors of `LossComputer` instances.
    """
    # input (N, C). target (N, ) -> (N, 1).
    if inputs.ndim == 2 and targets.ndim == 1:
        targets = targets.unsqueeze(1)
    # input (N, ). target (N, 1) -> (N, ).
    elif inputs.ndim == 1 and targets.ndim == 2 and targets.size()[-1] == 1:
        targets = targets.squeeze(1)
    return targets


def _preprocess_for_loss_fn_receiving_input_target(
    inputs: Tensor,
    targets: Tensor,
    *,
    match_target_dim_to_input_dim: bool,
    target_to_dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    if inputs.ndim > 2 or targets.ndim > 2:
        raise ValueError(
            f'The dimensions of input and targets are expected to be at most 2. '
            f'But received {inputs.ndim}, {targets.ndim}.'
        )
    if match_target_dim_to_input_dim and inputs.ndim != targets.ndim:
        targets = _match_target_dim_to_input_dim_at_most_2d_case(inputs, targets)

    if target_to_dtype:
        targets = targets.to(dtype=target_to_dtype)
    return inputs, targets
