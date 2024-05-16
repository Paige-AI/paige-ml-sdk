from abc import ABC, abstractmethod
from typing import Mapping, Optional

from torch import Tensor, as_tensor


class LossesReducer(ABC):
    """Base class for losses reducers.

    LossesReducer instances are used to reduce multiple losses where each loss is identified by its key.
    """

    @abstractmethod
    def reduce(self, losses: Mapping[str, Tensor]) -> Tensor: ...

    def _check_negative_loss(self, losses: Mapping[str, Tensor]) -> None:
        """Checks if there's any negative-valued loss.

        Raises:
            ValueError if negative loss is found.
        """
        for loss in losses.values():
            if loss < 0:
                raise ValueError(
                    f'Negative loss was found: {losses}. Likely there is a bug in the losses computation.'
                )


class SumLossesReducer(LossesReducer):
    def __init__(self, weights: Optional[Mapping[str, float]] = None) -> None:
        """Reduces multiple label losses by (optionally, weighted) summation.

        Args:
            weights: Label-wise weights. Defaults to None.
        """

        self._weights = weights

    def reduce(
        self,
        losses: Mapping[str, Tensor],
    ) -> Tensor:
        """Reduces multiple label losses by summation.

        Args:
            losses: A map from label name to loss.

        Returns:
            Returns Tensor.
        """
        self._check_negative_loss(losses)
        if self._weights is None:
            return as_tensor(sum(list(losses.values())))
        else:
            return as_tensor(sum([loss * self._weights[label] for label, loss in losses.items()]))
