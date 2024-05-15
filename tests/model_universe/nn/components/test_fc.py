from typing import Tuple

import pytest
import torch
from torch import Tensor
from torch.jit import ScriptModule
from torch.nn import Linear, Module, ReLU, Sequential, Sigmoid

from paige.ml_sdk.enforce_type import enforce_type
from paige.ml_sdk.model_universe.nn.components.fc import HPSFCLayerHead, LinearLayerSpec


@pytest.fixture
def f_head() -> HPSFCLayerHead:
    in_channels = 32
    layer_specs = [
        LinearLayerSpec(dim=128, activation=ReLU()),
        LinearLayerSpec(dim=10, activation=Sigmoid()),
    ]
    return HPSFCLayerHead(in_channels=in_channels, layer_specs=layer_specs)


class TestHPSNet:
    def test_prediction_head_should_implement_graph_according_to_specs(
        self, f_head: HPSFCLayerHead
    ) -> None:
        layers = list(f_head.fc.children())

        in_channels = f_head.in_channels
        for i, spec in enumerate(f_head.layer_specs):
            assert layers[i * 2].in_features == in_channels
            assert layers[i * 2].out_features == spec.dim
            in_channels = spec.dim
            if len(layers) > (i * 2) + 1:
                assert isinstance(layers[(i * 2) + 1], type(spec.activation))

        assert isinstance(f_head.activation, type(f_head.layer_specs[-1].activation))

    def test_prediction_head_should_implement_forward_correctly(
        self, f_head: HPSFCLayerHead
    ) -> None:
        class ReferenceImplementation(Module):
            def __init__(self) -> None:
                super().__init__()

                ops = [
                    Linear(32, 128),
                    ReLU(),
                    Linear(128, 10),
                ]

                self.activation = Sigmoid()
                self.fc = Sequential(*ops)

            def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
                # This test prevents potential errors due to re-implementation of forward in HPSFCLayerHead
                logits = self.fc(x)
                activations = self.activation(logits)

                return logits, activations

        ref = ReferenceImplementation()
        f_head.load_state_dict(ref.state_dict())  # ensure weights are the same accross models.

        x = torch.randn(1, 32)
        torch.testing.assert_close(ref(x), f_head(x))

    def test_should_be_torchscriptable(self, f_head: HPSFCLayerHead) -> None:
        head_ts = enforce_type(ScriptModule, torch.jit.script(f_head))
        x = torch.randn(1, 32)
        torch.testing.assert_close(f_head(x), head_ts(x))
