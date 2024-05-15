from typing import Dict, NamedTuple, Optional
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor
from torch.nn.modules import BCEWithLogitsLoss, CrossEntropyLoss

from paige.ml_sdk.dataset_universe.collate_fns import EmbeddingAggregatorFitCollatedItems
from paige.ml_sdk.model_universe.aggregator import AggregatorOutput
from paige.ml_sdk.model_universe.losses.loss_computer import AggregatorLossComputer

MOCK_TENSOR = MagicMock(spec=Tensor, side_effect=RuntimeError('I should not have been used.'))


class TestAggregatorLossComputer:
    class _AggregatorOutput(NamedTuple):
        heads_logits: Dict[str, Tensor]
        heads_activations: Dict[str, Tensor]

    @pytest.mark.parametrize(
        'batch,model_output,target_to_dtype,expected',
        (
            pytest.param(
                EmbeddingAggregatorFitCollatedItems(
                    group_indices=MOCK_TENSOR,
                    embeddings=MOCK_TENSOR,
                    label_map={
                        'label': torch.tensor([0, 0, 1], dtype=torch.float),
                    },
                    padding_mask=MOCK_TENSOR,
                    sequence_lengths=MOCK_TENSOR,
                    instance_mask_map={'label': torch.tensor([True, True, True])},
                ),
                _AggregatorOutput(
                    heads_logits={
                        'label': torch.tensor([-1.0, -0.5, 1.0], dtype=torch.float).unsqueeze(1),
                    },
                    heads_activations={'label': MOCK_TENSOR},
                ),
                None,
                torch.tensor(0.3669).float(),
                id='compute-correctly',
            ),
            pytest.param(
                EmbeddingAggregatorFitCollatedItems(
                    group_indices=MOCK_TENSOR,
                    embeddings=MOCK_TENSOR,
                    label_map={
                        'label': torch.tensor([0, -999, 1], dtype=torch.float),
                    },
                    padding_mask=MOCK_TENSOR,
                    sequence_lengths=MOCK_TENSOR,
                    instance_mask_map={'label': torch.tensor([True, False, True])},
                ),
                _AggregatorOutput(
                    heads_logits={
                        'label': torch.tensor([-1.0, -0.5, 1.0], dtype=torch.float).unsqueeze(1),
                    },
                    heads_activations={'label': MOCK_TENSOR},
                ),
                None,
                torch.tensor(0.3133).float(),
                id='ignore-missing-label',
            ),
            pytest.param(
                EmbeddingAggregatorFitCollatedItems(
                    group_indices=MOCK_TENSOR,
                    embeddings=MOCK_TENSOR,
                    label_map={
                        'label': torch.tensor([0, 0, 1], dtype=torch.long),
                    },
                    padding_mask=MOCK_TENSOR,
                    sequence_lengths=MOCK_TENSOR,
                    instance_mask_map={'label': torch.tensor([True, True, True])},
                ),
                _AggregatorOutput(
                    heads_logits={
                        'label': torch.tensor([-1.0, -0.5, 1.0], dtype=torch.float).unsqueeze(1),
                    },
                    heads_activations={'label': MOCK_TENSOR},
                ),
                torch.float,
                torch.tensor(0.3669).float(),
                id='correct-target-dtype-for-chosen-loss',
            ),
            pytest.param(
                EmbeddingAggregatorFitCollatedItems(
                    group_indices=MOCK_TENSOR,
                    embeddings=MOCK_TENSOR,
                    label_map={
                        'label': torch.tensor([-999, -999, -999], dtype=torch.float),
                    },
                    padding_mask=MOCK_TENSOR,
                    sequence_lengths=MOCK_TENSOR,
                    instance_mask_map={'label': torch.tensor([False, False, False])},
                ),
                _AggregatorOutput(
                    heads_logits={
                        'label': torch.tensor([[0.4], [0.3], [0.6]], dtype=torch.float),
                    },
                    heads_activations={'label': MOCK_TENSOR},
                ),
                None,
                torch.tensor(0.0),
                id='all-missing-labels',
            ),
        ),
    )
    def test_should_compute_loss_with_bce_with_logits_loss(
        self,
        batch: EmbeddingAggregatorFitCollatedItems,
        model_output: AggregatorOutput,
        target_to_dtype: Optional[torch.dtype],
        expected: Tensor,
    ) -> None:
        loss_computer = AggregatorLossComputer(
            loss_fn=BCEWithLogitsLoss(),
            match_target_dim_to_input_dim=True,
            target_to_dtype=target_to_dtype,
        )
        loss = loss_computer(batch, model_output, 'label')
        torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-2)

    @pytest.mark.parametrize(
        'batch,model_output,target_to_dtype,expected',
        (
            pytest.param(
                EmbeddingAggregatorFitCollatedItems(
                    group_indices=MOCK_TENSOR,
                    embeddings=MOCK_TENSOR,
                    label_map={
                        'label': torch.tensor([0, 0, 1], dtype=torch.long),
                    },
                    padding_mask=MOCK_TENSOR,
                    sequence_lengths=MOCK_TENSOR,
                    instance_mask_map={'label': torch.tensor([True, True, True])},
                ),
                _AggregatorOutput(
                    heads_logits={
                        'label': torch.tensor(
                            [[-0.9, 1.1], [-0.55, 1.45], [0.4, 1.6]], dtype=torch.float
                        ),
                    },
                    heads_activations={'label': MOCK_TENSOR},
                ),
                None,
                torch.tensor(1.5057).float(),
                id='compute-correctly',
            ),
            pytest.param(
                EmbeddingAggregatorFitCollatedItems(
                    group_indices=MOCK_TENSOR,
                    embeddings=MOCK_TENSOR,
                    label_map={
                        'label': torch.tensor([0, -999, 1], dtype=torch.long),
                    },
                    padding_mask=MOCK_TENSOR,
                    sequence_lengths=MOCK_TENSOR,
                    instance_mask_map={'label': torch.tensor([True, False, True])},
                ),
                _AggregatorOutput(
                    heads_logits={
                        'label': torch.tensor(
                            [[-0.9, 1.1], [-0.55, 1.45], [0.4, 1.6]], dtype=torch.float
                        ),
                    },
                    heads_activations={'label': MOCK_TENSOR},
                ),
                None,
                torch.tensor(1.1951).float(),
                id='ignore-missing-label',
            ),
            pytest.param(
                EmbeddingAggregatorFitCollatedItems(
                    group_indices=MOCK_TENSOR,
                    embeddings=MOCK_TENSOR,
                    label_map={
                        'label': torch.tensor([0, 0, 1], dtype=torch.float),
                    },
                    padding_mask=MOCK_TENSOR,
                    sequence_lengths=MOCK_TENSOR,
                    instance_mask_map={'label': torch.tensor([True, True, True])},
                ),
                _AggregatorOutput(
                    heads_logits={
                        'label': torch.tensor(
                            [[-0.9, 1.1], [-0.55, 1.45], [0.4, 1.6]], dtype=torch.float
                        ),
                    },
                    heads_activations={'label': MOCK_TENSOR},
                ),
                torch.long,
                torch.tensor(1.5057).float(),
                id='correct-target-dtype-for-chosen-loss',
            ),
            pytest.param(
                EmbeddingAggregatorFitCollatedItems(
                    group_indices=MOCK_TENSOR,
                    embeddings=MOCK_TENSOR,
                    label_map={
                        'label': torch.tensor([-999, -999, -999], dtype=torch.long),
                    },
                    padding_mask=MOCK_TENSOR,
                    sequence_lengths=MOCK_TENSOR,
                    instance_mask_map={'label': torch.tensor([False, False, False])},
                ),
                _AggregatorOutput(
                    heads_logits={
                        'label': torch.tensor([[0.4], [0.3], [0.6]], dtype=torch.float),
                    },
                    heads_activations={'label': MOCK_TENSOR},
                ),
                None,
                torch.tensor(0.0),
                id='all-missing-labels',
            ),
        ),
    )
    def test_should_compute_loss_with_cross_entropy_loss(
        self,
        batch: EmbeddingAggregatorFitCollatedItems,
        model_output: AggregatorOutput,
        target_to_dtype: Optional[torch.dtype],
        expected: Tensor,
    ) -> None:
        loss_computer = AggregatorLossComputer(
            loss_fn=CrossEntropyLoss(ignore_index=-999),
            match_target_dim_to_input_dim=False,
            target_to_dtype=target_to_dtype,
        )
        loss = loss_computer(batch, model_output, 'label')
        torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-2)
