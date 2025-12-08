import os
os.environ['XFORMERS_AVAILABLE'] = "false"
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from core.monster import Monster
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


def export_to_onnx(checkpoint_path, config_path, output_path, input_shape=(1, 6, 480, 640), iters=32):
    """
    Export trained Monster stereo model to ONNX format
    
    Args:
        checkpoint_path: Path to the trained model checkpoint (.pth file)
        config_path: Path to the model configuration file
        output_path: Output path for the ONNX model
        input_shape: Input tensor shape (batch_size, channels, height, width)
    """
    
    # Load configuration
    cfg = OmegaConf.load(config_path)
    
    # Initialize model
    model = Monster(cfg, export_onnx=True)
    
    # Load checkpoint
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
            disp_pred = self.model(left, right, iters=self.iters, test_mode=True)
            
            # Return negative disparity as specified
            return -disp_pred
    
    # Wrap model for ONNX export
    wrapped_model = ONNXWrapper(model, iters=iters)
    # wrapped_model = ONNXWrapper(model, iters=getattr(cfg, 'valid_iters', 12))

    # wrapped_model.cuda()
    # dummy_input = dummy_input.cuda()
    
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
    parser = argparse.ArgumentParser(description='Export Monster stereo model to ONNX')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to model config file')
    parser.add_argument('--output', required=True, help='Output ONNX file path')
    parser.add_argument('--height', type=int, default=480, help='Input height')
    parser.add_argument('--width', type=int, default=640, help='Input width')
    parser.add_argument('--iters', type=int, default=20, help='Number of iterations for disparity prediction')
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    
    input_shape = (1, 6, args.height, args.width)
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        input_shape=input_shape,
        iters=args.iters
    )

if __name__ == '__main__':
    main()
