import re
from typing import Tuple

import pytest
import torch
from torch import Tensor

from paige.ml_sdk.model_universe.nn.components.attention import (
    DotProductAttentionWithLearnedQueries,
)


def _create_padding_mask(shape: Tuple[int, ...]) -> Tensor:
    mask = torch.zeros(shape)
    for i in range(shape[0]):
        mask[i, :2].fill_(0)
        mask[i, 2:].fill_(1)
    return mask


class Test_DotProductAttentionWithLearnedQueries:
    def test_should_return_correct_shapes_no_reduce(self) -> None:
        key, value = torch.randn(8, 3, 16), torch.randn(8, 3, 16)
        n_queries = 4
        dpa = DotProductAttentionWithLearnedQueries(
            in_features=key.shape[2],
            n_queries=n_queries,
            scaled=False,
            absolute=False,
            reduce=False,
        )

        out, attn = dpa(key, value)

        assert out.shape == (value.shape[0], n_queries, value.shape[2])
        assert attn.shape == (key.shape[0], key.shape[1], n_queries)

    def test_should_return_correct_shapes_reduce(self) -> None:
        key, value = torch.randn(8, 3, 16), torch.randn(8, 3, 16)
        n_queries = 4
        dpa = DotProductAttentionWithLearnedQueries(
            in_features=key.shape[2],
            n_queries=n_queries,
            scaled=False,
            absolute=False,
            reduce=True,
        )

        out, attn = dpa(key, value)

        assert out.shape == (value.shape[0], value.shape[2])
        assert attn.shape == (key.shape[0], key.shape[1], n_queries)

    def test_should_produce_attn_scaled_that_is_diff_than_unscaled(self) -> None:
        # TODO: this test is very uninformative, but accurately testing the code path used when
        # scaled=True is tricky

        key, value = torch.randn(8, 3, 16), torch.randn(8, 3, 16)
        dpa = DotProductAttentionWithLearnedQueries(
            in_features=key.shape[2], n_queries=4, scaled=False
        )
        _, attn_unscaled = dpa(key, value)

        dpa = DotProductAttentionWithLearnedQueries(
            in_features=key.shape[2], n_queries=4, scaled=True
        )
        _, attn_scaled = dpa(key, value)

        assert not torch.equal(attn_scaled, attn_unscaled)

    def test_should_return_different_output_between_absolute_and_nonabsolute_attn(self) -> None:
        # TODO: this test is very uninformative, but accurately testing the code path used when
        # scaled=True is tricky
        key, value = torch.randn(8, 3, 16), torch.randn(8, 3, 16)
        dpa = DotProductAttentionWithLearnedQueries(
            in_features=key.shape[2], n_queries=4, absolute=False
        )
        out_nonabsolute, _ = dpa(key, value)

        dpa = DotProductAttentionWithLearnedQueries(
            in_features=key.shape[2], n_queries=4, absolute=True
        )
        out_absolute, _ = dpa(key, value)

        assert not torch.equal(out_absolute, out_nonabsolute)

    @pytest.mark.parametrize(
        'key_shape,value_shape,mask_shape',
        [
            pytest.param((8, 3, 16), (8, 3, 16), (8, 3), id='2d'),
            pytest.param((8, 3, 16), (8, 3, 16), (8, 3, 1), id='3d'),
        ],
    )
    def test_should_make_masks_that_prevent_padding_from_attention(
        self,
        key_shape: Tuple[int, int, int],
        value_shape: Tuple[int, int, int],
        mask_shape: Tuple[int, ...],
    ) -> None:
        dpa = DotProductAttentionWithLearnedQueries(in_features=key_shape[2], n_queries=4)
        _, attn = dpa(
            torch.randn(key_shape),
            torch.randn(value_shape),
            padding_mask=_create_padding_mask(mask_shape),
        )

        assert attn[:, -1, :].sum() == 0.0

    @pytest.mark.parametrize(
        'key_shape,value_shape,mask_shape,error_message',
        [
            pytest.param(
                (8, 3, 16),
                (8, 3, 16),
                (8, 3, 2),
                'Expected padding_mask to have shape of .* or .*, but got shape of .*',
                id='invalid_format',
            ),
            pytest.param(
                (8, 3, 16),
                (8, 3, 16),
                (10, 5, 1),
                'Expected padding_mask to have shape of .* at first two dimensions, but got shape of .*',
                id='shape_mismatch',
            ),
        ],
    )
    def test_should_raise_error_when_masks_are_incorrectly_formatted(
        self,
        key_shape: Tuple[int, int, int],
        value_shape: Tuple[int, int, int],
        mask_shape: Tuple[int, ...],
        error_message: str,
    ) -> None:
        dpa = DotProductAttentionWithLearnedQueries(in_features=key_shape[2], n_queries=4)

        with pytest.raises(RuntimeError, match=error_message):
            _, _ = dpa(
                torch.randn(key_shape),
                torch.randn(value_shape),
                padding_mask=_create_padding_mask(mask_shape),
            )

    def test_should_produce_normalized_attn_if_return_normalized_true(self) -> None:
        key, value = torch.randn(8, 3, 16), torch.randn(8, 3, 16)
        n_queries = 4
        dpa = DotProductAttentionWithLearnedQueries(
            in_features=key.shape[2], n_queries=n_queries, return_normalized=True
        )
        mask_shape = (key.shape[0], key.shape[1])
        _, attn = dpa(key, value, padding_mask=_create_padding_mask(mask_shape))

        # Due to softmax, attn should sum to 1 on dimension 1
        torch.testing.assert_close(attn.sum(1), torch.ones((value.shape[0], n_queries)))

    def test_shoudl_produce_unnormalized_attn_if_return_normalized_false(self) -> None:
        key, value = torch.randn(8, 3, 16), torch.randn(8, 3, 16)
        n_queries = 4
        dpa = DotProductAttentionWithLearnedQueries(
            in_features=key.shape[2], n_queries=n_queries, return_normalized=False
        )
        mask_shape = (key.shape[0], key.shape[1])
        _, attn = dpa(key, value, padding_mask=_create_padding_mask(mask_shape))

        # Should not sum to one generally
        assert not torch.equal(attn.sum(1), torch.ones((value.shape[0], n_queries)))

    def test_should_raise_error_for_mismatched_key_value_lengths(self) -> None:
        key, value = torch.randn(8, 3, 16), torch.randn(8, 2, 16)
        dpa = DotProductAttentionWithLearnedQueries(in_features=key.shape[2], n_queries=4)
        with pytest.raises(
            RuntimeError,
            match=re.escape('Expected key and value to have same sequence length (dim 1).'),
        ):
            dpa(key, value)
