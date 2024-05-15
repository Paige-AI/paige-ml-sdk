import pytest
import torch
import torch.nn as nn

from paige.ml_sdk.model_universe.nn.agata import Agata
from paige.ml_sdk.model_universe.nn.components.fc import HPSFCLayerHeadConfig, LinearLayerSpec


@pytest.fixture
def f_agata() -> Agata:
    label_name_fclayer_head_config = {
        'label_0': HPSFCLayerHeadConfig(
            in_channels=2,
            layer_specs=[LinearLayerSpec(dim=4, activation=nn.Softmax(dim=1))],
        ),
        'label_1': HPSFCLayerHeadConfig(
            in_channels=2,
            layer_specs=[LinearLayerSpec(dim=1, activation=nn.Sigmoid())],
        ),
    }

    return Agata(
        in_features=16,
        layer1_out_features=3,
        layer2_out_features=2,
        activation=nn.ReLU(),
        label_name_fclayer_head_config=label_name_fclayer_head_config,
        n_attention_queries=4,
    )


class Test_Agata:
    BATCH_SIZE = 8

    def test_forward_should_output_correct_shapes(
        self,
        f_agata: Agata,
    ) -> None:
        # Arrange
        group_embeddings = torch.randn(self.BATCH_SIZE, 10, f_agata.linear1.in_features)

        # Act
        output = f_agata(group_embeddings, None)

        # Assert
        assert output['attn_scores'].shape == (
            self.BATCH_SIZE,
            group_embeddings.shape[1],
            f_agata.attention.n_queries,
        )
        assert output['backbone_embedding'].shape == (
            self.BATCH_SIZE,
            f_agata.linear2.out_features,
        )
        for label_name, head in f_agata.heads.items():
            expected_shape = (self.BATCH_SIZE, head.layer_specs[-1].dim)
            assert output['heads_logits'][label_name].shape == expected_shape  # type: ignore
            assert output['heads_activations'][label_name].shape == expected_shape  # type: ignore
