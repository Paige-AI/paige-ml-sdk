import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DotProductAttentionWithLearnedQueries(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_queries: int = 1,
        scaled: bool = False,
        absolute: bool = False,
        reduce: bool = False,
        padding_indicator: float = 1,
        return_normalized: bool = True,
    ) -> None:
        """Performs dot product attention given a set of keys and values with a set of learned
        queries.

        Args:
            in_features: Number of features expected in query embeddings.
            n_queries: Number of learned keys to use. Defaults to 1.
            scaled: Performs scaled dot product attention. Defaults to False.
            absolute: Applies attention to absolute value of value embeddings.
                Defaults to False.
            reduce: Sums the attended values to output a single embeddings for each
                instance in the batch. Defaults to False.
            padding_indicator: Value that indicates padding positions in mask.
                Defaults to 1.
            return_normalized: Returns the attention scores with softmax applied.
        """
        super().__init__()
        # If we think about the generalization of attention being a dot product between a set of
        # queries and a set of keys where the resulting product is applied to a set of values then
        # here we will have learned keys in the form of a weight matrix
        self.queries = nn.Linear(in_features, n_queries, bias=False)

        self.n_queries = n_queries
        self.scaled = scaled
        self.absolute = absolute
        self.reduce = reduce
        self.padding_indicator = padding_indicator
        self.return_normalized = return_normalized
        self.pad_value = float('-inf')

    def _format_padding_mask(
        self, padding_mask: Tensor, expected_shape: Tuple[int, int, int]
    ) -> Tensor:
        """Validates and formats padding masks. Masks are expected to be in the shape of
        (batch, sequence) or (batch, sequence, 1) and must have the same size in the batch and
        sequence dimensions as the query/value tensors. This function validates the format and
        shapes are as expected and repeats masks in a third dimension to enable proper use for
        masking out multiple attention keys.

        Args:
            padding_mask: (B, S) or (B, S, 1) mask to ensure a padding position does not contribute
                to attention.
            expected_shape: Shape to validate and format to.

        Raises:
            RuntimeError: If the size at the batch and sequence dimensions is incorrect.
            RuntimeError: If the format is not what is expected.

        Returns:
            Tensor: Formatted padding mask.
        """
        if padding_mask.shape[:2] != expected_shape[:2]:
            raise RuntimeError(
                f'Expected padding_mask to have shape of {expected_shape[:2]} at first two dimensions, but got shape of {padding_mask.shape[:2]}'
            )
        if len(padding_mask.shape) == 2:
            padding_mask = padding_mask.unsqueeze(-1)

        if len(padding_mask.shape) == 3:
            if padding_mask.shape[2] == 1:
                padding_mask = padding_mask.expand(-1, -1, expected_shape[2])
            elif padding_mask.shape[2] != expected_shape[2]:
                raise RuntimeError(
                    f'Expected padding_mask to have shape of {expected_shape[:2]} or {(expected_shape[0], expected_shape[1], 1)}, but got shape of {padding_mask.shape}'
                )

        return padding_mask

    def forward(  # type: ignore
        self, key: Tensor, value: Tensor, padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass to perform dot product attention with learned set of keys.

        Args:
            key: (B, S, F) where B is the batch dimension, S is the sequence length, and F is the
                embedding dimension.
            value: (B, S, F) where B is the batch dimension, S is the sequence length, and F is the
                embedding dimension.
            padding_mask: (B, S) or (B, S, 1) mask to ensure a padding position does not contribute
                to attention. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Attended values of shape (B, F) if reduce is true and (B, Q, F)
                otherwise. Attention scores of shape (B, Sk, Q).
        """
        if key.size(1) != value.size(1):
            raise RuntimeError('Expected key and value to have same sequence length (dim 1).')
        # (B, Sk, F) x (B, F, Q) -> (B, Sk, Q)
        attn = self.queries(key)

        if padding_mask is not None:
            padding_mask = self._format_padding_mask(
                padding_mask, expected_shape=(key.size(0), key.size(1), attn.size(2))
            )
            # Set attention scores at padding locations to -inf so after softmax they are
            # zero-valued
            attn[padding_mask == self.padding_indicator] = self.pad_value

        if self.scaled:
            d_k = key.size(-1)
            attn /= math.sqrt(d_k)

        normalized_attn = F.softmax(attn, dim=1)

        if self.absolute:
            value = torch.abs(value)

        # (B, Q, Sk) x (B, Sv, F) -> (B, Q, F), where Sv == Sk
        output = torch.bmm(normalized_attn.transpose(-2, -1), value)

        if self.reduce:
            # (B, Q, F) -> (B, F)
            output = output.sum(1)  # should this be mean?

        if self.return_normalized:
            attn = normalized_attn

        return output, attn
