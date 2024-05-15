from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Sequence

from torch import Tensor
from torch.nn import Linear, Module, Sequential


class HPSNetOutput(NamedTuple):
    backbone_embedding: Tensor
    heads_logits: Dict[str, Tensor]
    heads_activations: Dict[str, Tensor]


class HPSFCLayerHeadOutput(NamedTuple):
    logits: Tensor
    activations: Tensor


@dataclass
class LinearLayerSpec:
    dim: int
    activation: Module


@dataclass
class HPSFCLayerHeadConfig:
    in_channels: int
    layer_specs: Sequence[LinearLayerSpec]


class HPSFCLayerHead(Module):
    def __init__(self, in_channels: int, layer_specs: Sequence[LinearLayerSpec]) -> None:
        """Creates Heads for HPSNet with fully connected layers.

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

    def __call__(self, __x: Tensor) -> HPSFCLayerHeadOutput:  # type: ignore
        out: HPSFCLayerHeadOutput = super().__call__(__x)
        return out

    def forward(self, x: Tensor) -> HPSFCLayerHeadOutput:  # type: ignore
        """Runs a forward pass."""
        logits = self.fc(x)
        activations = self.activation(logits)
        return HPSFCLayerHeadOutput(logits=logits, activations=activations)
