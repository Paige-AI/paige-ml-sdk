from typing import Mapping

import pytest
import torch
from torch import Tensor

from paige.ml_sdk.model_universe.losses.losses_reducer import SumLossesReducer


class Test_SumLossesReducer:
    @pytest.mark.parametrize(
        'losses,expected',
        (
            (
                {'label_0': torch.tensor(0.5), 'label_1': torch.tensor(1.0)},
                torch.tensor(1.5),
            ),
        ),
    )
    def test_should_sum_all_losses(self, losses: Mapping[str, Tensor], expected: Tensor) -> None:
        losses_reducer = SumLossesReducer()
        loss = losses_reducer.reduce(losses)
        torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-2)

    @pytest.mark.parametrize(
        'losses,weights,expected',
        (
            (
                {'label_0': torch.tensor(0.5), 'label_1': torch.tensor(1.0)},
                {'label_0': 1.0, 'label_1': 0.0},
                torch.tensor(0.5),
            ),
        ),
    )
    def test_should_weighted_sum_all_losses(
        self,
        losses: Mapping[str, Tensor],
        weights: Mapping[str, float],
        expected: Tensor,
    ) -> None:
        losses_reducer = SumLossesReducer(weights)
        loss = losses_reducer.reduce(losses)
        torch.testing.assert_close(loss, expected)

    def test_should_raise_value_error_if_any_loss_is_negative(self) -> None:
        losses_reducer = SumLossesReducer()
        losses = {'label_0': torch.tensor(-0.5), 'label_1': torch.tensor(1.0)}
        with pytest.raises(ValueError, match=r'.*Negative loss was found.*'):
            losses_reducer.reduce(losses)
