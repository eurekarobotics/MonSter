"""Simple unit tests for GWC volume functions."""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pytest
import torch

from core.submodule import (
    build_gwc_volume,
    build_gwc_volume_onnx,
    build_gwc_volume_onnx_v2,
    build_gwc_volume_onnx_v3,
)


@pytest.fixture
def test_inputs():
    """Create test inputs matching actual model usage."""
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    B, C, H, W = 1, 96, 120, 160  # 1/4 scale of 480x640
    maxdisp, num_groups = 48, 8   # max_disp//4 = 192//4
    
    ref = torch.randn(B, C, H, W, device=device)
    target = torch.randn(B, C, H, W, device=device)
    
    return ref, target, maxdisp, num_groups


def test_onnx_matches_original(test_inputs):
    """build_gwc_volume_onnx should match original."""
    ref, target, maxdisp, num_groups = test_inputs
    
    vol_orig = build_gwc_volume(ref, target, maxdisp, num_groups)
    vol_onnx = build_gwc_volume_onnx(ref, target, maxdisp, num_groups)
    
    assert torch.allclose(vol_orig, vol_onnx, rtol=1e-6, atol=1e-7)


def test_v2_matches_original(test_inputs):
    """build_gwc_volume_onnx_v2 should match original."""
    ref, target, maxdisp, num_groups = test_inputs
    
    vol_orig = build_gwc_volume(ref, target, maxdisp, num_groups)
    vol_v2 = build_gwc_volume_onnx_v2(ref, target, maxdisp, num_groups)
    
    assert torch.allclose(vol_orig, vol_v2, rtol=1e-6, atol=1e-7)


def test_v3_matches_original(test_inputs):
    """build_gwc_volume_onnx_v3 should match original."""
    ref, target, maxdisp, num_groups = test_inputs
    
    vol_orig = build_gwc_volume(ref, target, maxdisp, num_groups)
    vol_v3 = build_gwc_volume_onnx_v3(ref, target, maxdisp, num_groups)
    
    assert torch.allclose(vol_orig, vol_v3, rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
