from dataclasses import dataclass
from typing import List, NamedTuple, Sequence

from torch import Tensor
from torch.nn import Linear, Module, Sequential


class FCHeadOutput(NamedTuple):
    logits: Tensor
    activations: Tensor


@dataclass
class LinearLayerSpec:
    dim: int
    activation: Module


@dataclass
class FCHeadConfig:
    in_channels: int
    layer_specs: Sequence[LinearLayerSpec]


class FCHead(Module):
    def __init__(self, in_channels: int, layer_specs: Sequence[LinearLayerSpec]) -> None:
        """Creates fully connected layers to be used as a model head.

        Args:
            in_channels: Number of channels of the feature vector provided by the backbone
            layer_specs: Sequence of linear layer specifications
        """
        super().__init__()
        self.in_channels = in_channels
        self.layer_specs = layer_specs

        ops: List[Module] = []

        for layer_spec in self.layer_specs[:-1]:
            ops.append(Linear(in_channels, layer_spec.dim))
            ops.append(layer_spec.activation)

            in_channels = layer_spec.dim

        ops.append(Linear(in_channels, self.layer_specs[-1].dim))

        self.activation = self.layer_specs[-1].activation
        self.fc = Sequential(*ops)

    def __call__(self, __x: Tensor) -> FCHeadOutput:  # type: ignore
        out: FCHeadOutput = super().__call__(__x)
        return out

    def forward(self, x: Tensor) -> FCHeadOutput:  # type: ignore
        """Runs a forward pass."""
        logits = self.fc(x)
        activations = self.activation(logits)
        return FCHeadOutput(logits=logits, activations=activations)
