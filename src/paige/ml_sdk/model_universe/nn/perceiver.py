from typing import Any, Dict, Literal, Mapping, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.backends.cuda import sdp_kernel

from paige.ml_sdk.model_universe.nn.components.fc import (
    HPSFCLayerHead,
    HPSFCLayerHeadConfig,
)


class CrossAttentionNoProj(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: int,
        head_dim: int,
        heads: int,
        kv_append_q: bool,
    ) -> None:
        super().__init__()

        self.query_dim = query_dim
        self.context_dim = context_dim

        self.head_dim = context_dim // heads
        self.heads = heads

        self.scale = self.head_dim**-0.5

        self.inner_dim = self.head_dim * self.heads

        self.kv_append_q = kv_append_q

        self.x_norm = nn.LayerNorm(self.query_dim)
        # self.c_norm = nn.LayerNorm(self.context_dim)

        self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=False)

        self.to_out = nn.Linear(self.inner_dim, self.query_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        c: Tensor | None = None,
        kvt: tuple[Tensor, Tensor] | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        Bx, Nx, _ = x.shape

        x = self.x_norm(x)
        # norming large contexts explodes memory; keep commented out
        # c = self.c_norm(c)

        q: Tensor = self.to_q(x)
        q = q.reshape(Bx, Nx, self.heads, self.head_dim).permute(0, 2, 1, 3)

        if c is not None and kvt is None:
            Bc, Nc, _ = c.shape
            kv = c.reshape(Bc, Nc, self.heads, self.head_dim).permute(0, 2, 1, 3)
            kvt = (kv, kv)
        elif kvt is not None and c is None:
            kv, kv = kvt
            Bc, _, Nc, _ = kv.shape
        else:
            raise ValueError(f'XOR(c, kvt) but got: {type(c)} and {type(kvt)}.')

        if self.kv_append_q is True:
            kv = torch.concat([q, kv], dim=-2)

        if attn_mask is not None:
            attn_mask = attn_mask.reshape(Bc, 1, Nx, Nc).expand(-1, self.heads, -1, -1)

            if self.kv_append_q is True:
                x_mask = torch.ones(
                    (Bc, self.heads, Nx, Nx), device=attn_mask.device, dtype=attn_mask.dtype
                )
                attn_mask = torch.concat([x_mask, attn_mask], dim=-1)

        # fwd: 0.9801454544067383 GB
        # bwd: 1.2725048065185547 GB
        q = q * self.scale
        sim = q @ kv.transpose(-2, -1)
        if attn_mask is not None:
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        a = attn @ kv

        # For some reason, more memory-intensive than above:
        # fwd: 1.3135957717895508 GB
        # bwd: 1.4147930145263672 GB
        # Limitations:
        # - FlashAttn and Mem-efficient attn don't work with attn_mask
        # - FlashAttn and Mem-efficient attn don't work without heads
        # - FlashAttn doesn't work with head_dim > 128
        # with sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        #     a: Tensor = F.scaled_dot_product_attention(q, kv, kv, attn_mask=attn_mask)

        c = a.transpose(1, 2).reshape(Bx, Nx, self.inner_dim)

        o = self.to_out(c)

        return o, kvt


class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: int,
        head_dim: int,
        heads: int,
        kv_append_q: bool,
    ) -> None:
        super().__init__()

        self.query_dim = query_dim
        self.context_dim = context_dim

        self.head_dim = head_dim
        self.heads = heads

        self.scale = self.head_dim**-0.5

        self.inner_dim = self.head_dim * self.heads

        self.kv_append_q = kv_append_q

        self.x_norm = nn.LayerNorm(self.query_dim)
        # self.c_norm = nn.LayerNorm(self.context_dim)

        self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(self.context_dim, self.inner_dim * 2, bias=False)

        self.to_out = nn.Linear(self.inner_dim, self.query_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        c: Tensor | None = None,
        kvt: tuple[Tensor, Tensor] | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        Bx, Nx, _ = x.shape

        x = self.x_norm(x)
        # norming large contexts explodes memory; keep commented out
        # c = self.c_norm(c)

        q: Tensor = self.to_q(x)
        q = q.reshape(Bx, Nx, self.heads, self.head_dim).permute(0, 2, 1, 3)

        if c is not None and kvt is None:
            Bc, Nc, _ = c.shape
            kv: Tensor = self.to_kv(c)
            kv = kv.reshape(Bc, Nc, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            kvt = (k, v)
        elif kvt is not None and c is None:
            k, v = kvt
            Bc, _, Nc, _ = k.shape
            assert (Bc, Nc) == (v.shape[0], v.shape[2])
        else:
            raise ValueError(f'XOR(c, kvt) but got: {type(c)} and {type(kvt)}.')

        if self.kv_append_q is True:
            k = torch.concat([q, k], dim=-2)
            v = torch.concat([q, v], dim=-2)

        if attn_mask is not None:
            attn_mask = attn_mask.reshape(Bc, 1, Nx, Nc).expand(-1, self.heads, -1, -1)

            if self.kv_append_q is True:
                x_mask = torch.ones(
                    (Bc, self.heads, Nx, Nx), device=attn_mask.device, dtype=attn_mask.dtype
                )
                attn_mask = torch.concat([x_mask, attn_mask], dim=-1)

        # fwd: 1.2660036087036133 GB
        # bwd: 1.7023143768310547 GB
        # q = q * self.scale
        # sim = q @ k.transpose(-2, -1)
        # if attn_mask is not None:
        #     sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        # attn = sim.softmax(dim=-1)
        # a = attn @ v

        # fwd: 1.2649354934692383 GB
        # bwd: 1.3198986053466797 GB
        # Limitations:
        # - FlashAttn and Mem-efficient attn don't work with attn_mask
        # - FlashAttn and Mem-efficient attn don't work without heads
        # - FlashAttn doesn't work with head_dim > 128
        # - FlashAttn only supports sm75 and sm8x gpu architectures. v100 is sm70.
        with sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            a: Tensor = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        c = a.transpose(1, 2).reshape(Bx, Nx, self.inner_dim)

        o = self.to_out(c)

        return o, kvt


class MHSA(nn.Module):
    def __init__(self, *, dim: int, num_heads: int):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=dim,
            vdim=dim,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        # The following only works with torch>2.1 when being finetuned as an aggregator
        with sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            x, _ = self.mha(
                x,
                x,
                x,
                key_padding_mask=None,
                need_weights=False,
                attn_mask=None,
                average_attn_weights=False,
                is_causal=False,
            )

        return x


# Feed forward


class GEGLU(nn.Module):
    def forward(self, x: Tensor):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, *, dim: int, mult: int = 1, dropout: float = 0.0):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        return self.net(self.norm(x))


# Perceiver


class PerceiverWrapper(nn.Module):
    def __init__(
        self,
        image_resampler: nn.Module,
        label_name_fclayer_head_config: Mapping[str, HPSFCLayerHeadConfig],
        init_perceiver_path: Optional[str],
    ):
        super().__init__()
        self.image_resampler = image_resampler
        if init_perceiver_path is not None:
            sd = torch.load(init_perceiver_path)
            sd = sd['state_dict']
            sd = {
                k.replace('image_resampler.', ''): v
                for k, v in sd.items()
                if 'image_resampler.' in k
            }
            print(
                f'Initializing the perceiver backbone with initialized weights from {init_perceiver_path}'
            )
            self.image_resampler.load_state_dict(sd)

        heads = {
            name: HPSFCLayerHead(cfg.in_channels, cfg.layer_specs)
            for name, cfg in label_name_fclayer_head_config.items()
        }
        self.heads = cast(Mapping[str, HPSFCLayerHead], nn.ModuleDict(heads))

    def forward(self, images: Tensor, image_pad_mask: Optional[Tensor]):
        '''
        x: shape (B, N, F)
        image_pad_mask: Mask that indicates which indices in a sequence are padding and which are
            valid. Expected to be in the shape of (batch, sequence).
        '''
        image_pad_mask = image_pad_mask.bool()
        cls_tokens, _ = self.image_resampler.forward(
            images=images, image_pad_mask=~image_pad_mask
        )  # Need to inverse the padding masks

        heads_logits, heads_activations = self._forward_output_heads(cls_tokens)

        return {
            'backbone_embedding': cls_tokens,
            'heads_logits': heads_logits,
            'heads_activations': heads_activations,
        }

    def _forward_output_heads(
        self, backbone_embedding: Tensor
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Copied from Agata
        """
        heads_logits: Dict[str, Tensor] = {}
        heads_activations: Dict[str, Tensor] = {}
        for name, head in self.heads.items():
            logits, activations = head(backbone_embedding)
            heads_logits[name] = logits
            heads_activations[name] = activations

        return heads_logits, heads_activations


class Perceiver(nn.Module):
    def __init__(
        self,
        *,
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
    ):
        super().__init__()

        assert perceiver_depth > 0
        assert share_xattn_start_layer >= 0
        assert share_tf_start_layer >= 0

        self.share_xattn_start_layer = share_xattn_start_layer
        self.share_tf_start_layer = share_tf_start_layer
        self.latent_seq = latent_seq
        self.mhsa_heads = mhsa_heads
        self.xattn_chunked = xattn_chunked

        if xattn_chunked and not xattn_kv_append_q:
            raise ValueError('if xattn_chunked=True then must set xattn_kv_append_q=True')

        # TODO: init latents correctly
        self.latents = nn.Parameter(torch.randn(latent_seq, latent_dim))

        xattn_class = CrossAttention if xattn_kv_proj is True else CrossAttentionNoProj
        get_xattn = lambda: nn.ModuleDict(
            {
                'xattn': xattn_class(
                    query_dim=latent_dim,
                    context_dim=context_dim,
                    head_dim=latent_dim,
                    heads=1,
                    kv_append_q=xattn_kv_append_q,
                ),
                'ff': FeedForward(dim=latent_dim),
            }
        )

        get_mhsa = lambda: nn.ModuleDict(
            {
                'mhsa': MHSA(dim=latent_dim, num_heads=mhsa_heads),
                'ff': FeedForward(dim=latent_dim),
            }
        )

        get_transformer = lambda: nn.ModuleList([get_mhsa() for _ in range(transformer_depth)])

        layers = []
        for i in range(perceiver_depth):
            layer = nn.ModuleDict(
                {
                    'xattn': (
                        get_xattn() if i <= self.share_xattn_start_layer else layers[-1]['xattn']
                    ),
                    'tf': (
                        get_transformer() if i <= self.share_tf_start_layer else layers[-1]['tf']
                    ),
                }
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, c: Tensor, image_pad_mask: Tensor) -> Tensor:
        assert (
            image_pad_mask.max(dim=-1)[0] != 0
        ).all(), 'image_pad_mask contains empty sequences in some batch items'
        if self.xattn_chunked:
            return self.forward_chunked_xattn(c, image_pad_mask)
        else:
            return self.forward_repeated_xattn(c, image_pad_mask)

    def forward_chunked_xattn(self, c: Tensor, image_pad_mask: Tensor) -> Tensor:
        B, N, _ = c.shape

        assert len(image_pad_mask.shape) == 2
        assert image_pad_mask.shape[:1] == c.shape[:1]

        attn_mask = image_pad_mask.reshape(B, 1, N).expand(-1, self.latent_seq, -1)

        x = self.latents.unsqueeze(0).expand(B, -1, -1)

        c_chunks = c.chunk(self.mhsa_heads, dim=-2)
        attn_mask_chunks = attn_mask.chunk(self.mhsa_heads, dim=-1)

        for i, l in enumerate(self.layers):
            c = c_chunks[i]
            attn_mask = attn_mask_chunks[i]

            xattn = l['xattn']  # type: ignore
            xattn_out, _ = xattn['xattn'](x, c=c, kvt=None, attn_mask=attn_mask)
            x = xattn_out + x
            x = xattn['ff'](x) + x

            tf = l['tf']  # type: ignore
            for mhsa in tf:
                x = mhsa['mhsa'](x) + x
                x = mhsa['ff'](x) + x

        return x

    def forward_repeated_xattn(self, c: Tensor, image_pad_mask: Tensor) -> Tensor:
        B, N, _ = c.shape

        assert len(image_pad_mask.shape) == 2
        assert image_pad_mask.shape[:1] == c.shape[:1]

        attn_mask = image_pad_mask.reshape(B, 1, N).expand(-1, self.latent_seq, -1)

        x = self.latents.unsqueeze(0).expand(B, -1, -1)

        kvt = None
        for i, l in enumerate(self.layers):
            xattn = l['xattn']  # type: ignore
            if i <= self.share_xattn_start_layer:
                xattn_out, kvt = xattn['xattn'](x, c=c, kvt=None, attn_mask=attn_mask)
            else:
                xattn_out, kvt = xattn['xattn'](x, c=None, kvt=kvt, attn_mask=attn_mask)

            x = xattn_out + x
            x = xattn['ff'](x) + x

            tf = l['tf']  # type: ignore
            for midx, mhsa in enumerate(tf):
                x = mhsa['mhsa'](x) + x
                x = mhsa['ff'](x) + x

        return x


def handle_legacy_layer_norm(
    state_dict: Mapping[str, Any], prefix: str, *args: Any, **kwargs: Any
) -> None:
    legacy_keys = {
        'cls_norm.gamma': 'cls_norm.weight',
        'cls_norm.beta': 'cls_norm.bias',
        'img_norm.gamma': 'img_norm.weight',
        'img_norm.beta': 'img_norm.bias',
    }
    for k, v in legacy_keys.items():
        if (old_key := prefix + k) in state_dict:
            assert isinstance(state_dict, MutableMapping), 'cannot mutate legacy state_dict'
            new_key = prefix + v
            state_dict[new_key] = state_dict[old_key]
            del state_dict[old_key]


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
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
    ):
        super().__init__()

        self.query_dim = latent_dim
        self.query_seq = latent_seq

        self.pooler = pooler

        if self.pooler == 'parallel':
            latent_seq = 1 + latent_seq
            self.cls_norm = nn.LayerNorm(latent_dim)
            self.img_norm = nn.LayerNorm(latent_dim)

        elif self.pooler == 'mhsa':
            self.mhsa_pooler = MHSA(dim=latent_dim, num_heads=8)
            latent_seq = 1 + latent_seq
            self.cls_norm = nn.LayerNorm(latent_dim)
            self.img_norm = nn.LayerNorm(latent_dim)

        elif self.pooler == 'mean':
            pass

        else:
            raise ValueError(f'unknown pooler: {self.pooler}')

        self._register_load_state_dict_pre_hook(handle_legacy_layer_norm)

        self.perceiver = Perceiver(
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
        )

    def forward(self, images: Tensor, *, image_pad_mask: Tensor) -> tuple[Tensor, Tensor]:
        x: Tensor = self.perceiver(c=images, image_pad_mask=image_pad_mask)

        if self.pooler == 'parallel':
            cls_emb, img_emb = x[:, 0], x[:, 1:]

            assert cls_emb.shape == (len(images), self.query_dim)
            assert img_emb.shape == (len(images), self.query_seq, self.query_dim)

            cls_emb = self.cls_norm(cls_emb)
            img_emb = self.img_norm(img_emb)

            return cls_emb, img_emb

        if self.pooler == 'mhsa':
            x = self.mhsa_pooler(x)

            cls_emb, img_emb = x[:, 0], x[:, 1:]

            assert cls_emb.shape == (len(images), self.query_dim)
            assert img_emb.shape == (len(images), self.query_seq, self.query_dim)

            cls_emb = self.cls_norm(cls_emb)
            img_emb = self.img_norm(img_emb)

            return cls_emb, img_emb

        if self.pooler == 'mean':
            assert x.shape == (len(images), self.query_seq, self.query_dim)
            return x.mean(1), x

        raise ValueError(f'unknown pooler: {self.pooler}')
