from typing import Dict, Mapping, Optional, Tuple, Union, cast

import torch.nn as nn
from torch import Tensor

from paige.ml_sdk.model_universe.nn.components.attention import (
    DotProductAttentionWithLearnedQueries,
)
from paige.ml_sdk.model_universe.nn.components.fc import (
    HPSFCLayerHead,
    HPSFCLayerHeadConfig,
)


class Agata(nn.Module):
    """Defines a two layer, feed forward, 'Aggregator with Attention' model architecture"""

    def __init__(
        self,
        in_features: int,
        layer1_out_features: int,
        layer2_out_features: int,
        activation: nn.Module,
        label_name_fclayer_head_config: Mapping[str, HPSFCLayerHeadConfig],
        scaled_attention: bool = False,
        absolute_attention: bool = False,
        n_attention_queries: int = 1,
        padding_indicator: int = 1,
    ) -> None:
        """Initializes an Agata model.

        Args:
            in_features: Dimension of an input embedding.
            layer1_out_features: Dimension of first linear layer.
            layer2_out_features: Dimension of second linear layer
            activation: Activation to be applied between both linear layers.
            label_name_fclayer_head_config: A mapping specifying the type of head for each label.
            scaled_attention: Normalizes attention values by sqrt(`layer2_out_features`).
            absolute_attention: Takes absolute value of attention.
            n_attention_queries: Number of attention queries.
            padding_indicator: Value that indicates padding positions in mask.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_features, layer1_out_features)
        self.linear2 = nn.Linear(layer1_out_features, layer2_out_features)

        self.attention = DotProductAttentionWithLearnedQueries(
            in_features=layer1_out_features,
            n_queries=n_attention_queries,
            scaled=scaled_attention,
            absolute=absolute_attention,
            reduce=True,
            padding_indicator=padding_indicator,
            return_normalized=False,  # want non-softmaxed attn scores for KL loss calc
        )

        self.activation = activation

        heads = {
            name: HPSFCLayerHead(cfg.in_channels, cfg.layer_specs)
            for name, cfg in label_name_fclayer_head_config.items()
        }
        self.heads = cast(Mapping[str, HPSFCLayerHead], nn.ModuleDict(heads))

    def forward(  # type: ignore
        self, x: Tensor, padding_masks: Optional[Tensor]
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        """Runs a forward pass.

        Args:
            x: Batch of group embeddings. Expected to be in the shape of (batch, sequence, features
            a.k.a. (b, s, f), because torch.nn.Linear operation requires features to be in the last
            dimension. Beware! torch.nn.Conv1d requires features to be in the second dimension,
            which would break compatibility; don't use Conv1d! Refs:
             - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
             - https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            padding_masks: Mask that indicates which indices in a sequence are padding and which are
                valid. Expected to be in the shape of (batch, sequence).

        Returns: A dictionary containing output tensors for the aggregated embedding, output
        head logits and activations, and attention scores.
        """
        # x.shape -> (batch, sequence, features) a.k.a (B, S, F)
        x_1, x_2 = self.forward_features(x)

        # x_3.shape -> (B, F), attn.shape -> (B, S, n_attention_queries)
        x_3, attn = self.attention(key=x_1, value=x_2, padding_mask=padding_masks)

        heads_logits, heads_activations = self._forward_output_heads(x_3)

        return {
            'backbone_embedding': x_3,
            'heads_logits': heads_logits,
            'heads_activations': heads_activations,
            'attn_scores': attn,
        }

    def forward_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Runs the layers in the network that come before attention.

        .. note::
            this API is named as such in order to achieve consistency with timm architectures,
            which all posess a `forward_features` method.
        """
        x_1 = self.linear1(x)
        x_1 = self.activation(x_1)

        x_2 = self.linear2(x_1)
        x_2 = self.activation(x_2)
        return x_1, x_2

    def _forward_output_heads(
        self, backbone_embedding: Tensor
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Forward pass for output heads.

        Args:
            backbone_embedding: Shared embedding to be distributed to output heads.

        Raises:
            TypeError: Unsupported type is output by a head.

        Returns: Dictionaries of label names mapped to head logits/activations.
        """
        heads_logits: Dict[str, Tensor] = {}
        heads_activations: Dict[str, Tensor] = {}
        for name, head in self.heads.items():
            logits, activations = head(backbone_embedding)
            heads_logits[name] = logits
            heads_activations[name] = activations

        return heads_logits, heads_activations
