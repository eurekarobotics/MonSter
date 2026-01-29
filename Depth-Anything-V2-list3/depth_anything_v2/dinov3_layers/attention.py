
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import logging
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .utils import cat_keep_shapes, uncat_with_shapes
from .rope import rope_apply

class LinearKMaskedBias(nn.Linear):
    """Linear layer with masked K bias for DINOv3."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            bias_mask = torch.ones_like(self.bias)
            bias_mask[o // 3 : 2 * o // 3] = 0.0  # Mask K bias to 0
            self.register_buffer("bias_mask", bias_mask)

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Apply RoPE to q and k tensors."""
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        
        q = q.float()
        k = k.float()
        sin = sin.float()
        cos = cos.float()
        
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)
        
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)
        k = torch.cat((k_prefix, k), dim=-2)
        
        # Convert back to original dtype
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        # Always apply rope (DINOv3 always uses rope embeddings)
        # Removing conditional avoids ONNX If nodes that TensorRT cannot handle
        q, k = self.apply_rope(q, k, rope)
        
        # FP16-safe: pre-scale Q and K to prevent overflow in Q@K^T
        # Instead of computing (Q @ K^T) * scale where scale = 1/sqrt(d),
        # we compute (Q * sqrt(scale)) @ (K * sqrt(scale))^T which is equivalent
        # but keeps intermediate values smaller and within FP16 range
        head_dim = q.shape[-1]
        scale_factor = head_dim ** -0.25  # = (1/sqrt(d))^0.5
        logging.debug("[FP16-SAFE] Using pre-scaled attention (head_dim=%d, scale=%.4f)", head_dim, scale_factor)
        print("############"*1000)
        q = q * scale_factor
        k = k * scale_factor
        
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])
