
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
import torch.nn.functional as F
from torch import Tensor, nn

class RMSNorm(nn.Module):
    """RMSNorm"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(-1, keepdim=True).clamp(min=self.eps)
        return x_f32 * torch.rsqrt(rms)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x).type_as(x)
        return output * self.weight


class FP16SafeLayerNorm(nn.LayerNorm):
    """LayerNorm with FP32 statistics computation for FP16 stability.
    
    This is weight-compatible with nn.LayerNorm - existing checkpoints
    can be loaded directly without any modification.
    
    The mean and variance are computed in FP32 to avoid precision loss
    when working with large embedding dimensions (e.g., 1024 for vitl).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[FP16-SAFE] Using FP16SafeLayerNorm with FP32 statistics computation")
    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        # Compute LayerNorm in FP32 for numerical stability
        x = F.layer_norm(
            x.float(), 
            self.normalized_shape, 
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None, 
            self.eps
        )
        return x.to(orig_dtype)

