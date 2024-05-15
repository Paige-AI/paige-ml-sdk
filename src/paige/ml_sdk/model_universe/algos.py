"""This file contains subclasses of paige.ml_sdk.model_universe.aggregator.Aggregator

Use these subclasses in the lightning cli like this:

```
python -m paige.ml_sdk \
--model BinClsAgata \
--model.in_features 2560 \
--model.targets [cancer, precursor] \
...
```
"""

from functools import partial
from typing import List, Literal, Optional

import torch
import torch.nn as nn

from paige.ml_sdk.model_universe.aggregator import Aggregator
from paige.ml_sdk.model_universe.losses.loss_computer import AggregatorLossComputer
from paige.ml_sdk.model_universe.metrics.metrics_computers.aggregator_metrics import (
    AggregatorBinClsGroupMetricsComputer,
    AggregatorMulticlassGroupMetricsComputer,
)
from paige.ml_sdk.model_universe.nn.agata import Agata
from paige.ml_sdk.model_universe.nn.components.fc import HPSFCLayerHeadConfig, LinearLayerSpec
from paige.ml_sdk.model_universe.nn.perceiver import PerceiverResampler, PerceiverWrapper


class BinClsAgata(Aggregator):
    def __init__(
        self,
        label_names: List[str],
        in_features: int = 2560,
        layer1_out_features: int = 512,
        layer2_out_features: int = 512,
        activation: Optional[nn.Module] = None,
        scaled_attention: bool = False,
        absolute_attention: bool = False,
        n_attention_queries: int = 1,
        padding_indicator: int = 1,
        missing_label_value: int = -999,
    ):
        loss_computers = {
            ln: AggregatorLossComputer(
                nn.CrossEntropyLoss(ignore_index=missing_label_value),
                match_target_dim_to_input_dim=False,
            )
            for ln in label_names
        }
        metrics_cls = AggregatorBinClsGroupMetricsComputer
        optimizer_partial = partial(torch.optim.AdamW, lr=0.0005)
        lr_scheduler_partial = partial(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.5)
        step_output_keys = ['heads_activations', 'instance_mask_map', 'label_map', 'loss']
        label_name_fclayer_head_config = {
            ln: HPSFCLayerHeadConfig(
                layer2_out_features, [LinearLayerSpec(dim=2, activation=nn.Sigmoid())]
            )
            for ln in label_names
        }
        activation = activation or nn.ReLU()
        model = Agata(
            in_features,
            layer1_out_features,
            layer2_out_features,
            activation,
            label_name_fclayer_head_config,
            scaled_attention,
            absolute_attention,
            n_attention_queries,
            padding_indicator,
        )
        super().__init__(
            model=model,
            loss_computers=loss_computers,
            optimizer_partial=optimizer_partial,
            lr_scheduler_partial=lr_scheduler_partial,
            train_metrics_computers=[metrics_cls(label_name=ln) for ln in label_names],
            val_metrics_computers=[metrics_cls(label_name=ln) for ln in label_names],
            test_metrics_computers=[metrics_cls(label_name=ln) for ln in label_names],
            val_step_output_keys=step_output_keys,
            train_step_output_keys=step_output_keys,
            test_step_output_keys=step_output_keys,
        )


class MultiClsAgata(Aggregator):
    def __init__(
        self,
        label_names: List[str],
        n_classes: List[int],  # order must match label_names
        in_features: int = 2560,
        layer1_out_features: int = 512,
        layer2_out_features: int = 512,
        activation: nn.Module = nn.ReLU(),
        scaled_attention: bool = False,
        absolute_attention: bool = False,
        n_attention_queries: int = 1,
        padding_indicator: int = 1,
        missing_label_value: int = -999,
    ):
        loss_computers = {
            ln: AggregatorLossComputer(
                nn.CrossEntropyLoss(ignore_index=missing_label_value),
                match_target_dim_to_input_dim=False,
            )
            for ln in label_names
        }
        metrics_cls = AggregatorMulticlassGroupMetricsComputer
        optimizer_partial = partial(torch.optim.AdamW, lr=0.0005)
        lr_scheduler_partial = partial(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.5)
        step_output_keys = ['heads_activations', 'instance_mask_map', 'label_map', 'loss']
        label_name_fclayer_head_config = {
            ln: HPSFCLayerHeadConfig(
                layer2_out_features, [LinearLayerSpec(dim=nc, activation=nn.Sigmoid())]
            )
            for ln, nc in zip(label_names, n_classes)
        }
        model = Agata(
            in_features,
            layer1_out_features,
            layer2_out_features,
            activation,
            label_name_fclayer_head_config,
            scaled_attention,
            absolute_attention,
            n_attention_queries,
            padding_indicator,
        )
        super().__init__(
            model=model,
            loss_computers=loss_computers,
            optimizer_partial=optimizer_partial,
            lr_scheduler_partial=lr_scheduler_partial,
            train_metrics_computers=[metrics_cls(label_name=ln) for ln in label_names],
            val_metrics_computers=[metrics_cls(label_name=ln) for ln in label_names],
            test_metrics_computers=[metrics_cls(label_name=ln) for ln in label_names],
            val_step_output_keys=step_output_keys,
            train_step_output_keys=step_output_keys,
            test_step_output_keys=step_output_keys,
        )


class BinClsPerceiver(Aggregator):
    def __init__(
        self,
        label_names: List[str],
        latent_seq: int = 512,
        latent_dim: int = 768,
        context_dim: int = 2560,
        mhsa_heads: int = 8,
        perceiver_depth: int = 8,
        transformer_depth: int = 6,
        share_xattn_start_layer: int = 1,
        share_tf_start_layer: int = 0,
        xattn_kv_proj: bool = True,
        xattn_kv_append_q: bool = False,
        xattn_chunked: bool = False,
        pooler: Literal['mean', 'parallel', 'mhsa'] = 'mean',
        missing_label_value: int = -999,
        init_perceiver_path: Optional[str] = None,
    ):
        loss_computers = {
            ln: AggregatorLossComputer(
                nn.CrossEntropyLoss(ignore_index=missing_label_value),
                match_target_dim_to_input_dim=False,
            )
            for ln in label_names
        }
        metrics_cls = AggregatorBinClsGroupMetricsComputer
        optimizer_partial = partial(torch.optim.AdamW, lr=0.0005)
        lr_scheduler_partial = partial(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.5)
        step_output_keys = ['heads_activations', 'instance_mask_map', 'label_map', 'loss']
        label_name_fclayer_head_config = {
            ln: HPSFCLayerHeadConfig(latent_dim, [LinearLayerSpec(dim=2, activation=nn.Sigmoid())])
            for ln in label_names
        }
        perceiver = PerceiverResampler(
            latent_seq=latent_seq,
            latent_dim=latent_dim,
            context_dim=context_dim,
            mhsa_heads=mhsa_heads,
            perceiver_depth=perceiver_depth,
            transformer_depth=transformer_depth,
            share_xattn_start_layer=share_xattn_start_layer,
            share_tf_start_layer=share_tf_start_layer,
            xattn_kv_proj=xattn_kv_proj,
            xattn_kv_append_q=xattn_kv_append_q,
            xattn_chunked=xattn_chunked,
            pooler=pooler,
        )
        model = PerceiverWrapper(
            image_resampler=perceiver,
            label_name_fclayer_head_config=label_name_fclayer_head_config,
            init_perceiver_path=init_perceiver_path,
        )
        super().__init__(
            model=model,
            loss_computers=loss_computers,
            optimizer_partial=optimizer_partial,
            lr_scheduler_partial=lr_scheduler_partial,
            train_metrics_computers=[metrics_cls(label_name=ln) for ln in label_names],
            val_metrics_computers=[metrics_cls(label_name=ln) for ln in label_names],
            test_metrics_computers=[metrics_cls(label_name=ln) for ln in label_names],
            val_step_output_keys=step_output_keys,
            train_step_output_keys=step_output_keys,
            test_step_output_keys=step_output_keys,
        )


class MultiClsPerceiver(Aggregator):
    ...
