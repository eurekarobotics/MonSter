import torch
import torch.nn as nn
from .dinov2_layers.dpt_head import DPTHead

class DINOv2BaseAdapter(nn.Module):
    """
    Base adapter for DINOv2 DPTHead.
    Handles initialization and feature reshaping.
    """
    def __init__(self, **kwargs):
        super().__init__()
        if DPTHead is None:
            raise ImportError("DPTHead not found.")
        
        # Map in_channels to embed_dims
        if "in_channels" in kwargs:
            in_channels = kwargs["in_channels"] # Don't pop!
            if isinstance(in_channels, list):
                kwargs["embed_dims"] = in_channels[0]
            else:
                kwargs["embed_dims"] = in_channels
        
        if "post_process_channels" not in kwargs:
            # Default to [128, 256, 512, 1024] which matches our DINOv2 weights
            kwargs["post_process_channels"] = [128, 256, 512, 1024]
            
        self.dpt_head = DPTHead(**kwargs)

    def _reshape_features(self, x, patch_h, patch_w):
        """
        Reshapes DINOv2 features (B, N, C) to (B, C, H, W).
        """
        processed_x = []
        for feat in x:
            if isinstance(feat, (tuple, list)):
                # Assume (patch, cls)
                patch_tokens = feat[0]
                cls_token = feat[1]
            else:
                patch_tokens = feat
                cls_token = None
                
            # Reshape (B, N, C) -> (B, C, H, W)
            B, N, C = patch_tokens.shape
            
            # Use patch_h, patch_w passed from Monster
            if N != patch_h * patch_w:
                 # Handle potential mismatch (e.g. registers)
                 pass
                 
            patch_tokens = patch_tokens.transpose(1, 2).reshape(B, C, patch_h, patch_w)
            
            # DPTHead expects (patch_tokens, cls_token)
            if cls_token is None:
                # Create dummy cls_token (B, C)
                cls_token = torch.zeros((B, C), device=patch_tokens.device, dtype=patch_tokens.dtype)
                
            processed_x.append((patch_tokens, cls_token))
        return processed_x

class DINOv2FeatureAdapter(DINOv2BaseAdapter):
    """
    Adapter to use DPTHead (configured for DINOv2) within Monster to extract FEATURES.
    Returns a list of feature maps.
    """
    def forward(self, x, patch_h, patch_w):
        processed_x = self._reshape_features(x, patch_h, patch_w)
            
        # DPTHead expects list of feature maps (or tuples)
        # Call with return_features=True to get intermediate features
        depth, features = self.dpt_head(processed_x, return_features=True)
        
        # DPTHead returns features at 1/2 resolution (relative to input)
        # Monster expects 1/4 resolution (matching stem_4x) for the first feature map
        # --> Need to downsample all features by 2x
        
        downsampled_features = []
        for feat in features:
            downsampled_features.append(nn.functional.interpolate(feat, scale_factor=0.5, mode='bilinear', align_corners=False))
            
        return downsampled_features

class DINOv2DepthAdapter(DINOv2BaseAdapter):
    """
    Adapter to use DPTHead (configured for DINOv2) within Monster to predict DEPTH.
    Returns the depth map.
    """
    def forward(self, features, patch_h, patch_w):
        reshaped_features = self._reshape_features(features, patch_h, patch_w)
        depth = self.dpt_head(reshaped_features)
        return depth


def load_dinov2_decoder_weights(adapter, backbone_name):
    import torch
    import os
    
    found_local = False
    pretrained_dir = "pretrained"
    if os.path.exists(pretrained_dir):
        prefix = f"{backbone_name}"
        for fname in os.listdir(pretrained_dir):
            if fname.startswith(prefix) and "dpt_head" in fname and fname.endswith(".pth"):
                local_path = os.path.join(pretrained_dir, fname)
                try:
                    state_dict = torch.load(local_path, map_location="cpu")
                    if "model" in state_dict:
                        state_dict = state_dict["model"]
                    elif "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    
                    # Strip 'decode_head.' prefix if present
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith("decode_head."):
                            new_state_dict[k.replace("decode_head.", "")] = v
                        else:
                            new_state_dict[k] = v
                    state_dict = new_state_dict
                    
                    missing_keys, unexpected_keys = adapter.dpt_head.load_state_dict(state_dict, strict=False)
                    
                    if len(missing_keys) > 0:
                        print(f"Warning: Missing keys when loading {local_path}: {missing_keys}")
                    
                    found_local = True
                    break
                except Exception as e:
                    print(f"Error loading {local_path}: {e}")
    
    if not found_local:
        print(f"Warning: Pre-trained depth head weights not found locally for {backbone_name}. Initializing with random weights.")
