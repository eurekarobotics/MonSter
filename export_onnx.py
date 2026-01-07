import os
os.environ['XFORMERS_AVAILABLE'] = "false"
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from core.monster import Monster, MonsterV2
from omegaconf import OmegaConf
import torch.onnx


def replace_instancenorm_with_groupnorm(model):
    """
    Replace all InstanceNorm2d/3d modules with equivalent GroupNorm modules.
    This fixes ONNX export issues with dynamic shapes while preserving trained weights.
    
    GroupNorm(num_groups=num_channels, num_channels) is mathematically identical 
    to InstanceNorm when each channel is its own group.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.InstanceNorm2d):
            # Create equivalent GroupNorm (num_groups = num_channels = instance norm)
            num_channels = module.num_features
            new_module = nn.GroupNorm(
                num_groups=num_channels,
                num_channels=num_channels,
                eps=module.eps,
                affine=module.affine
            )
            # Copy weights if affine=True
            if module.affine and module.weight is not None:
                new_module.weight.data.copy_(module.weight.data)
                new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
        elif isinstance(module, nn.InstanceNorm3d):
            num_channels = module.num_features
            new_module = nn.GroupNorm(
                num_groups=num_channels,
                num_channels=num_channels,
                eps=module.eps,
                affine=module.affine
            )
            if module.affine and module.weight is not None:
                new_module.weight.data.copy_(module.weight.data)
                new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
        else:
            # Recursively process child modules
            replace_instancenorm_with_groupnorm(module)
    return model


def precompute_rope_embeddings(model, input_height, input_width, patch_size=16):
    """
    Precompute RoPE embeddings for fixed input dimensions.
    This replaces the rope_embed forward method with a pre-cached version
    that returns constants, eliminating ONNX If nodes that TensorRT cannot handle.
    
    Args:
        model: The Monster model
        input_height: Input image height
        input_width: Input image width  
        patch_size: Patch size of the vision transformer (default 16 for DINOv3)
    """
    # Calculate patch grid dimensions
    H = input_height // patch_size
    W = input_width // patch_size
    
    def find_rope_modules(module, prefix=""):
        """Recursively find all RoPE embedding modules."""
        rope_modules = []
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            # Check if this is a RopePositionEmbedding module
            if 'rope_embed' in name.lower() or type(child).__name__ == 'RopePositionEmbedding':
                rope_modules.append((full_name, child))
            else:
                rope_modules.extend(find_rope_modules(child, full_name))
        return rope_modules
    
    rope_modules = find_rope_modules(model)
    
    for name, rope_module in rope_modules:
        print(f"Precomputing RoPE embeddings for {name} with H={H}, W={W}")
        
        with torch.no_grad():
            # Compute sin/cos once with fixed dimensions
            sin_cached, cos_cached = rope_module(H=H, W=W)
            
            # Create a replacement module that returns cached values
            class CachedRoPE(torch.nn.Module):
                def __init__(self, sin_cache, cos_cache):
                    super().__init__()
                    # Register as buffers so they're included in ONNX export
                    self.register_buffer('sin_cache', sin_cache.clone())
                    self.register_buffer('cos_cache', cos_cache.clone())
                
                def forward(self, *, H: int, W: int):
                    # Ignore H, W - return pre-cached values
                    return (self.sin_cache, self.cos_cache)
            
            cached_rope = CachedRoPE(sin_cached, cos_cached)
            cached_rope.eval()
            
            # Replace the module in the parent
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], cached_rope)
    
    print(f"Replaced {len(rope_modules)} RoPE modules with cached versions")
    return model


def export_to_onnx(checkpoint_path, config_path, output_path, input_shape=(1, 6, 480, 640), iters=32, use_v2=False):
    """
    Export trained Monster/MonsterV2 stereo model to ONNX format
    
    Args:
        checkpoint_path: Path to the trained model checkpoint (.pth file)
        config_path: Path to the model configuration file
        output_path: Output path for the ONNX model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        iters: Number of iterations (for Monster, MonsterV2 uses scale_iters from config)
        use_v2: If True, use MonsterV2 (hierarchical) instead of Monster
    """
    
    # Load configuration
    cfg = OmegaConf.load(config_path)
    
    # Initialize model
    if use_v2:
        print("Using MonsterV2 (hierarchical) model")
        model = MonsterV2(cfg, export_onnx=True)
    else:
        print("Using Monster (original) model")
        model = Monster(cfg, export_onnx=True)
    
    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path != 'none':
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key.replace('module.', '')
            cleaned_state_dict[cleaned_key] = value
        
        model.load_state_dict(cleaned_state_dict, strict=True)
        print("Model loaded successfully")
    
    # Set model to evaluation mode
    model.eval()
    
    # Replace InstanceNorm with GroupNorm for ONNX compatibility (mathematically identical)
    print("Replacing InstanceNorm with GroupNorm for ONNX export...")
    replace_instancenorm_with_groupnorm(model)
    
    # Precompute RoPE embeddings for DINOv3 models to eliminate ONNX If nodes
    # that TensorRT cannot handle
    if 'dinov3' in cfg.encoder:
        input_height = input_shape[2]
        input_width = input_shape[3]
        patch_size = 16  # DINOv3 uses patch_size=16
        print(f"Precomputing RoPE embeddings for DINOv3 (input: {input_height}x{input_width}, patch: {patch_size})...")
        precompute_rope_embeddings(model, input_height, input_width, patch_size)
    
    # Create dummy input (concatenated left and right images)
    # Shape: (1, 6, 480, 640) where 6 = 3 (left) + 3 (right) channels
    dummy_input = torch.randn(input_shape, dtype=torch.float32) * 255.0
    
    # Split input into left and right images for the model
    left = dummy_input[:, :3, :, :]  # First 3 channels
    right = dummy_input[:, 3:, :, :] # Last 3 channels
    
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model, iters=32):
            super().__init__()
            self.model = model
            self.iters = iters
        
        def forward(self, concat_input):
            # Split concatenated input into left and right
            left = concat_input[:, :3, :, :]
            right = concat_input[:, 3:, :, :]
            
            # Get disparity prediction in test mode
            # MonsterV2 ignores iters and uses scale_iters from config
            disp_pred = self.model(left, right, iters=self.iters, test_mode=True)
            
            # Return negative disparity as specified
            return -disp_pred
    
    # Wrap model for ONNX export
    wrapped_model = ONNXWrapper(model, iters=iters)

    # Export to ONNX
    print(f"Exporting model to {output_path}")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['negative_disparity'],
        verbose=True,
        training=torch.onnx.TrainingMode.EVAL,
        # dynamic_axes={
        #     'input': {0: 'batch_size'},
        #     'negative_disparity': {0: 'batch_size'}
        # }
    )
    
    print(f"Model exported successfully to {output_path}")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: (1, {input_shape[2]}, {input_shape[3]})")

def main():
    parser = argparse.ArgumentParser(description='Export Monster/MonsterV2 stereo model to ONNX')
    parser.add_argument('--checkpoint', default=None, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to model config file')
    parser.add_argument('--output', required=True, help='Output ONNX file path')
    parser.add_argument('--height', type=int, default=480, help='Input height')
    parser.add_argument('--width', type=int, default=640, help='Input width')
    parser.add_argument('--iters', type=int, default=20, help='Number of iterations for disparity prediction')
    parser.add_argument('--v2', action='store_true', help='Use MonsterV2 (hierarchical) instead of Monster')
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    
    input_shape = (1, 6, args.height, args.width)
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        input_shape=input_shape,
        iters=args.iters,
        use_v2=args.v2
    )

if __name__ == '__main__':
    main()

