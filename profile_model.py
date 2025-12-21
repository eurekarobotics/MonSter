"""
Profile MonSter model to understand time distribution.
This script measures time spent in:
1. DINO Encoder
2. Cost Volume + GRU iterations
3. Decoder + Refinement
"""
import os
import sys
sys.path.append('core')

import argparse
import time
import torch
import torch.nn.functional as F
from monster import Monster
from omegaconf import OmegaConf


def profile_model(model, image1, image2, iters=12, warmup=3, runs=10):
    """Profile model components separately."""
    
    model.eval()
    device = next(model.parameters()).device
    
    # Warmup
    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(image1, image2, iters=iters, test_mode=True)
    torch.cuda.synchronize()
    
    # Profile total time
    print(f"\nProfiling ({runs} runs)...")
    total_times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(image1, image2, iters=iters, test_mode=True)
        torch.cuda.synchronize()
        total_times.append(time.perf_counter() - start)
    
    avg_total = sum(total_times) / len(total_times)
    
    # Now profile individual components
    print("\nProfiling individual components...")
    
    # Prepare images (same as forward)
    image1_norm = (2 * (image1 / 255.0) - 1.0).contiguous()
    image2_norm = (2 * (image2 / 255.0) - 1.0).contiguous()
    
    # Profile encoder (infer_mono)
    encoder_times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                depth_mono, features_left, features_right = model.infer_mono(image1_norm, image2_norm)
        torch.cuda.synchronize()
        encoder_times.append(time.perf_counter() - start)
    
    avg_encoder = sum(encoder_times) / len(encoder_times)
    
    # Profile stems
    stem_times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            stem_2x = model.stem_2(image1_norm)
            stem_4x = model.stem_4(stem_2x)
            stem_x_list = [stem_2x, stem_4x]
        torch.cuda.synchronize()
        stem_times.append(time.perf_counter() - start)
    
    avg_stems = sum(stem_times) / len(stem_times)
    
    # Estimate GRU + Cost Volume time (total - encoder - stems - decoder overhead)
    # This is approximate since we can't easily separate
    avg_gru_loop = avg_total - avg_encoder - avg_stems  # Approximate
    
    # Print results
    print("\n" + "="*60)
    print("PROFILING RESULTS")
    print("="*60)
    print(f"\nTotal inference time: {avg_total*1000:.2f} ms")
    print(f"\nComponent breakdown:")
    print(f"  Encoder (infer_mono):     {avg_encoder*1000:.2f} ms ({avg_encoder/avg_total*100:.1f}%)")
    print(f"  Stems:                    {avg_stems*1000:.2f} ms ({avg_stems/avg_total*100:.1f}%)")
    print(f"  GRU + Cost Volume + Rest: {avg_gru_loop*1000:.2f} ms ({avg_gru_loop/avg_total*100:.1f}%)")
    print(f"\nIterations: {iters}")
    print(f"Image size: {image1.shape}")
    print("="*60)
    
    return {
        'total': avg_total,
        'encoder': avg_encoder,
        'stems': avg_stems,
        'gru_loop': avg_gru_loop
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--iters', type=int, default=12, help='Number of iterations')
    parser.add_argument('--height', type=int, default=480, help='Input height')
    parser.add_argument('--width', type=int, default=640, help='Input width')
    parser.add_argument('--runs', type=int, default=10, help='Number of profiling runs')
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.config)
    
    print(f"Loading model with encoder: {cfg.encoder}")
    
    # Create model
    model = Monster(cfg)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace('module.', '')
        cleaned_state_dict[cleaned_key] = value
    
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.cuda()
    model.eval()
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Create dummy input
    image1 = torch.randn(1, 3, args.height, args.width).cuda() * 255
    image2 = torch.randn(1, 3, args.height, args.width).cuda() * 255
    
    # Profile
    results = profile_model(model, image1, image2, iters=args.iters, runs=args.runs)
    
    # Print recommendations
    print("\nRECOMMENDATIONS:")
    if results['encoder'] / results['total'] > 0.5:
        print("  - Encoder is the bottleneck (>50% of time)")
        print("  - Consider: DINOv3 ViT-B/S, or reduce input resolution")
    else:
        print("  - GRU loop is the bottleneck")
        print("  - Consider: Reduce iterations, or optimize update blocks")


if __name__ == '__main__':
    main()
