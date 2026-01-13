import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock, BasicMultiUpdateBlock_mix2, BasicMultiUpdateBlockHierarchical
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
from core.refinement import REMP
from core.warp import disp_warp
import matplotlib.pyplot as plt

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
import os
import sys
sys.path.append('./Depth-Anything-V2-list3')
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2_decoder
from depth_anything_v2.dinov2_depth_head_adapter import (
    DINOv2FeatureAdapter,
    DINOv2DepthAdapter,
    load_dinov2_decoder_weights
)

    
def compute_scale_shift(monocular_depth, gt_depth, mask=None):
    """
    计算 monocular depth 和 ground truth depth 之间的 scale 和 shift.
    
    参数:
    monocular_depth (torch.Tensor): 单目深度图，形状为 (H, W) 或 (N, H, W)
    gt_depth (torch.Tensor): ground truth 深度图，形状为 (H, W) 或 (N, H, W)
    mask (torch.Tensor, optional): 有效区域的掩码，形状为 (H, W) 或 (N, H, W)
    
    返回:
    scale (float): 计算得到的 scale
    shift (float): 计算得到的 shift
    """
    
    flattened_depth_maps = monocular_depth.clone().view(-1).contiguous()
    sorted_depth_maps, _ = torch.sort(flattened_depth_maps)
    percentile_10_index = int(0.2 * len(sorted_depth_maps))
    threshold_10_percent = sorted_depth_maps[percentile_10_index]

    if mask is None:
        mask = (gt_depth > 0) & (monocular_depth > 1e-2) & (monocular_depth > threshold_10_percent)
    
    monocular_depth_flat = monocular_depth[mask]
    gt_depth_flat = gt_depth[mask]
    
    X = torch.stack([monocular_depth_flat, torch.ones_like(monocular_depth_flat)], dim=1)
    y = gt_depth_flat
    
    # 使用最小二乘法计算 [scale, shift]
    A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=X.device)
    b = torch.matmul(X.t(), y)
    params = torch.linalg.solve(A, b)
    
    scale, shift = params[0].item(), params[1].item()
    
    return scale, shift


def compute_scale_shift_batched(monocular_depth, gt_depth):
    """
    Batched version: compute scale and shift for all batch samples at once.
    
    Args:
        monocular_depth: (B, H, W) tensor
        gt_depth: (B, H, W) tensor
    
    Returns:
        scale: (B,) tensor
        shift: (B,) tensor
    """
    B, H, W = monocular_depth.shape
    device = monocular_depth.device
    
    # Initialize outputs
    scales = torch.ones(B, device=device, dtype=monocular_depth.dtype)
    shifts = torch.zeros(B, device=device, dtype=monocular_depth.dtype)
    
    for i in range(B):
        mono_i = monocular_depth[i]  # (H, W)
        gt_i = gt_depth[i]  # (H, W)
        
        # Compute threshold for this sample
        flattened = mono_i.view(-1)
        sorted_vals, _ = torch.sort(flattened)
        percentile_idx = int(0.2 * len(sorted_vals))
        threshold = sorted_vals[percentile_idx]
        
        # Create mask
        mask = (gt_i > 0) & (mono_i > 1e-2) & (mono_i > threshold)
        
        if mask.sum() < 10:  # Need enough points for regression
            continue
            
        mono_flat = mono_i[mask]
        gt_flat = gt_i[mask]
        
        # Least squares: [scale, shift] = (X^T X)^-1 X^T y
        X = torch.stack([mono_flat, torch.ones_like(mono_flat)], dim=1)
        y = gt_flat
        
        A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=device, dtype=X.dtype)
        b = torch.matmul(X.t(), y)
        params = torch.linalg.solve(A, b)
        
        scales[i] = params[0]
        shifts[i] = params[1]
    
    return scales, shifts


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 

        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv


class hourglass_local(nn.Module):
    """Lightweight hourglass for local cost volumes (13-17 disparities).
    
    Unlike full hourglass which does 3x downsampling, this only does 2x
    downsampling to work with small disparity dimensions. Includes feature
    attention for improved accuracy.
    """
    def __init__(self, in_channels, feat_channels=96):
        super(hourglass_local, self).__init__()
        
        # Two-level encoder-decoder
        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2),
            BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1)
        )
        
        self.conv2 = nn.Sequential(
            BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2),
            BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1)
        )
        
        # Decoder
        self.conv2_up = nn.Sequential(
            BasicConv(in_channels*4, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1),
        )
        
        self.conv1_up = nn.Sequential(
            BasicConv(in_channels*2, in_channels, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1),
        )
        
        # Skip connection aggregation
        self.agg_1 = nn.Sequential(
            BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1)
        )
        
        self.agg_0 = nn.Sequential(
            BasicConv(in_channels*2, in_channels, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1)
        )
        
        # Feature attention at each level (features will be downsampled to match)
        self.feature_att_1 = FeatureAtt(in_channels*2, feat_channels)
        self.feature_att_2 = FeatureAtt(in_channels*4, feat_channels)
        self.feature_att_up_1 = FeatureAtt(in_channels*2, feat_channels)
        self.feature_att_out = FeatureAtt(in_channels, feat_channels)
        
    def forward(self, x, features):
        """
        Args:
            x: (B, C, D, H, W) cost volume
            features: (B, feat_channels, H, W) 2D feature for attention
        """
        # Save input size for skip connections
        x_size = x.shape[2:]  # (D, H, W)
        
        # Downsample features to match each level
        feat_1x = features  # Original resolution
        feat_2x = F.avg_pool2d(features, 2)  # 1/2 resolution
        feat_4x = F.avg_pool2d(feat_2x, 2)  # 1/4 resolution
        
        # Encoder
        conv1 = self.conv1(x)  # stride=2 -> 1/2 size
        conv1_size = conv1.shape[2:]  # Save for skip connection
        conv1 = self.feature_att_1(conv1, feat_2x)
        
        conv2 = self.conv2(conv1)  # stride=2 -> 1/4 size
        conv2 = self.feature_att_2(conv2, feat_4x)
        
        # Decoder with skip connections
        # Upsample to match conv1 size
        conv2_up = self.conv2_up(conv2)
        conv2_up = F.interpolate(conv2_up, size=conv1_size, mode='trilinear', align_corners=False)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_1(conv1, feat_2x)
        
        # Upsample to match input size
        conv1_up = self.conv1_up(conv1)
        conv1_up = F.interpolate(conv1_up, size=x_size, mode='trilinear', align_corners=False)
        out = torch.cat((conv1_up, x), dim=1)
        out = self.agg_0(out)
        out = self.feature_att_out(out, feat_1x)
        
        return out


class Feat_transfer_cnet(nn.Module):
    def __init__(self, dim_list, output_dim):
        super(Feat_transfer_cnet, self).__init__()

        self.res_16x = nn.Conv2d(dim_list[0]+192, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0]+96, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0]+48, output_dim, kernel_size=3, padding=1, stride=1)

    def forward(self, features, stem_x_list):
        features_list = []
        feat_16x = self.res_16x(torch.cat((features[2], stem_x_list[0]), 1))
        feat_8x = self.res_8x(torch.cat((features[1], stem_x_list[1]), 1))
        feat_4x = self.res_4x(torch.cat((features[0], stem_x_list[2]), 1))
        features_list.append([feat_4x, feat_4x])
        features_list.append([feat_8x, feat_8x])
        features_list.append([feat_16x, feat_16x])
        return features_list



class Feat_transfer(nn.Module):
    def __init__(self, dim_list):
        super(Feat_transfer, self).__init__()
        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=int(48+dim_list[0]), out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=int(64+dim_list[0]), out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
            )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=int(192+dim_list[0]), out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(192), nn.ReLU()
            )
        self.conv32x = nn.Sequential(
            nn.Conv2d(in_channels=dim_list[0], out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(160), nn.ReLU()
            )
        self.conv_up_32x = nn.ConvTranspose2d(160,
                                192,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_16x = nn.ConvTranspose2d(192,
                                64,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_8x = nn.ConvTranspose2d(64,
                                48,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        
        self.res_16x = nn.Conv2d(dim_list[0], 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0], 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0], 48, kernel_size=1, padding=0, stride=1)




    def forward(self, features):
        features_mono_list = []
        feat_32x = self.conv32x(features[3])
        feat_32x_up = self.conv_up_32x(feat_32x)
        feat_16x = self.conv16x(torch.cat((features[2], feat_32x_up), 1)) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)
        features_mono_list.append(feat_32x)
        return features_mono_list





class Monster(nn.Module):
    def __init__(self, args, export_onnx=False):
        super().__init__()
        self.args = args
        self.export_onnx = export_onnx
        
        context_dims = args.hidden_dims

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39],
            'dinov3_vits14': [2, 5, 8, 11],
            'dinov3_vitb14': [2, 5, 8, 11],
            'dinov3_vitl14': [4, 11, 17, 23],
            # DINOv2 depth head variants (DINOv2 Encoder + DINOv2 Depth Head)
            'vits_dd': [2, 5, 8, 11],
            'vitb_dd': [2, 5, 8, 11],
            'vitl_dd': [4, 11, 17, 23],
            'vitg_dd': [9, 19, 29, 39],
        }
        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
            'dinov3_vits14': {'encoder': 'dinov3_vits14', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'dinov3_vitb14': {'encoder': 'dinov3_vitb14', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'dinov3_vitl14': {'encoder': 'dinov3_vitl14', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            # DINOv2 depth head variants
            'vits_dd': {'encoder': 'vits', 'features': 64, 'out_channels': [128, 256, 512, 1024]},
            'vitb_dd': {'encoder': 'vitb', 'features': 128, 'out_channels': [128, 256, 512, 1024]},
            'vitl_dd': {'encoder': 'vitl', 'features': 256, 'out_channels': [128, 256, 512, 1024]},
            'vitg_dd': {'encoder': 'vitg', 'features': 384, 'out_channels': [128, 256, 512, 1024]},
        }
        
        # Override out_channels if provided in args (config file)
        if hasattr(args, 'out_channels') and args.out_channels is not None:
            if args.encoder in mono_model_configs:
                mono_model_configs[args.encoder]['out_channels'] = args.out_channels
            
        dim_list_ = mono_model_configs[self.args.encoder]['features']
        dim_list = []
        dim_list.append(dim_list_)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feat_transfer = Feat_transfer(dim_list)
        self.feat_transfer_cnet = Feat_transfer_cnet(dim_list, output_dim=args.hidden_dims[0])


        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.stem_8 = nn.Sequential(
            BasicConv_IN(48, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96), nn.ReLU()
            )

        self.stem_16 = nn.Sequential(
            BasicConv_IN(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(192), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        depth_anything = DepthAnythingV2(**mono_model_configs[args.encoder])
        depth_anything_decoder = DepthAnythingV2_decoder(**mono_model_configs[args.encoder])
        
        self.mono_encoder = depth_anything.pretrained
        
        use_dinov2_decoder = 'dinov3' not in args.encoder and '_dd' in args.encoder
        
        if 'dinov3' not in args.encoder and not use_dinov2_decoder:
            # Case 1: DINOv2 + Standard Decoder (Legacy)
            state_dict_dpt = torch.load(f'./pretrained/depth_anything_v2_{args.encoder}.pth', map_location='cpu', weights_only=False)
            state_dict_dpt = torch.load(f'./pretrained/depth_anything_v2_{args.encoder}.pth', map_location='cpu', weights_only=False)
            depth_anything.load_state_dict(state_dict_dpt, strict=True)
            depth_anything_decoder.load_state_dict(state_dict_dpt, strict=False)
            
            self.mono_decoder = depth_anything.depth_head
            self.feat_decoder = depth_anything_decoder.depth_head
            
        elif use_dinov2_decoder:
            # Case 3: DINOv2 Encoder + DINOv2 Depth Head
            config = mono_model_configs[args.encoder]
            out_channels = config['out_channels']
            
            # Determine correct embed_dim for DINOv2 encoder
            if 'vits' in args.encoder: embed_dim = 384
            elif 'vitb' in args.encoder: embed_dim = 768
            elif 'vitl' in args.encoder: embed_dim = 1024
            elif 'vitg' in args.encoder: embed_dim = 1536
            else: embed_dim = config['features']
            
            # 1. Instantiate adapters
            self.mono_decoder = DINOv2DepthAdapter(
                in_channels=[embed_dim] * 4, 
                channels=256, 
                post_process_channels=out_channels,
                n_output_channels=1
            )
            self.feat_decoder = DINOv2FeatureAdapter(
                in_channels=[embed_dim] * 4,
                channels=256,
                post_process_channels=mono_model_configs[args.encoder]['out_channels'],
                use_bn=False,
                use_clstoken=False
            )
            
            # 2. Load weights
            base_encoder = args.encoder.replace('_dd', '')
            backbone_name = f"dinov2_{base_encoder}14"
            
            load_dinov2_decoder_weights(self.mono_decoder, backbone_name=backbone_name)
            load_dinov2_decoder_weights(self.feat_decoder, backbone_name=backbone_name)
            
            # 3. Load Backbone weights
            pretrained_dir = "pretrained"
            if os.path.exists(pretrained_dir):
                candidates = [
                    f"{backbone_name}_reg4_pretrain.pth",
                    f"{backbone_name}_pretrain.pth",
                    f"{backbone_name}.pth"
                ]
                for fname in candidates:
                    local_path = os.path.join(pretrained_dir, fname)
                    if os.path.exists(local_path):
                        try:
                            state_dict = torch.load(local_path, map_location="cpu", weights_only=False)
                            if "model" in state_dict:
                                state_dict = state_dict["model"]
                            elif "teacher" in state_dict:
                                state_dict = state_dict["teacher"]
                            
                            self.mono_encoder.load_state_dict(state_dict, strict=False)
                            break
                        except Exception:
                            pass
            
        else:
            # DINOv3 backbone is already loaded by DepthAnythingV2 init via dinov3_wrapper
            self.mono_decoder = depth_anything.depth_head
            self.feat_decoder = depth_anything_decoder.depth_head
            
        
        self.mono_encoder.requires_grad_(False)
        if 'dinov3' in args.encoder:
            self.mono_decoder.requires_grad_(True)
            self.feat_decoder.requires_grad_(True)
        elif use_dinov2_decoder:
            self.mono_decoder.requires_grad_(False)
            self.feat_decoder.requires_grad_(True)
        else:
            self.mono_decoder.requires_grad_(False)


        del depth_anything, depth_anything_decoder
        if 'dinov3' not in args.encoder and not use_dinov2_decoder:
            del state_dict_dpt
        self.REMP = REMP()


        self.update_block_mix_stereo = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)
        self.update_block_mix_mono = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)


        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def infer_mono(self, image1, image2):
        height_ori, width_ori = image1.shape[2:]
        
        if 'dinov3' in self.args.encoder:
             resize_image1 = image1
             resize_image2 = image2
        else:
             resize_image1 = F.interpolate(image1, scale_factor=14 / 16, mode='bilinear', align_corners=True)
             resize_image2 = F.interpolate(image2, scale_factor=14 / 16, mode='bilinear', align_corners=True)

        if 'dinov3' in self.args.encoder:
            patch_size = 16
        else:
            patch_size = 14
        patch_h, patch_w = resize_image1.shape[-2] // patch_size, resize_image1.shape[-1] // patch_size
        
        # OPTIMIZATION 3: Batch left and right images together for single encoder forward pass
        # This reduces encoder computation from 2x to 1x (~30-40% speedup)
        stacked_images = torch.cat([resize_image1, resize_image2], dim=0)  # [2B, C, H, W]
        features_stacked = self.mono_encoder.get_intermediate_layers(
            stacked_images, 
            self.intermediate_layer_idx[self.args.encoder], 
            return_class_token=True
        )
        
        # Split features back into left and right using torch.chunk (TensorRT compatible)
        # This avoids dynamic indexing feat[:B]/feat[B:] which can create ONNX If nodes
        features_left_encoder = []
        features_right_encoder = []
        for feat, cls_token in features_stacked:
            # torch.chunk splits tensor into 2 equal parts - static operation for ONNX
            feat_left, feat_right = torch.chunk(feat, 2, dim=0)
            cls_left, cls_right = torch.chunk(cls_token, 2, dim=0)
            features_left_encoder.append((feat_left, cls_left))
            features_right_encoder.append((feat_right, cls_right))
        
        depth_mono = self.mono_decoder(features_left_encoder, patch_h, patch_w)
        depth_mono = F.relu(depth_mono)
        depth_mono = F.interpolate(depth_mono, size=(height_ori, width_ori), mode='bilinear', align_corners=False)
        features_left_4x, features_left_8x, features_left_16x, features_left_32x = self.feat_decoder(features_left_encoder, patch_h, patch_w)
        features_right_4x, features_right_8x, features_right_16x, features_right_32x = self.feat_decoder(features_right_encoder, patch_h, patch_w)

        return depth_mono, [features_left_4x, features_left_8x, features_left_16x, features_left_32x], [features_right_4x, features_right_8x, features_right_16x, features_right_32x]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        # with autocast(enabled=self.args.mixed_precision):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with torch.autocast(device_type='cuda', dtype=torch.float32): 
            depth_mono, features_mono_left,  features_mono_right = self.infer_mono(image1, image2)

        scale_factor = 0.25
        size = (int(depth_mono.shape[-2] * scale_factor), int(depth_mono.shape[-1] * scale_factor))

        disp_mono_4x = F.interpolate(depth_mono, size=size, mode='bilinear', align_corners=False)

        features_left = self.feat_transfer(features_mono_left)
        features_right = self.feat_transfer(features_mono_right)
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x = self.stem_16(stem_8x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)

        stem_x_list = [stem_16x, stem_8x, stem_4x]
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))
        if self.export_onnx:
            gwc_volume = build_gwc_volume_onnx_v3(match_left, match_right, self.args.max_disp//4, 8)
        else:
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
        gwc_volume = self.corr_stem(gwc_volume)
        gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
        geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

        # Init disp from geometry encoding volume
        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        init_disp = disparity_regression(prob, self.args.max_disp//4)
        
        del prob, gwc_volume

        if not test_mode:
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)

        # cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
        cnet_list = self.feat_transfer_cnet(features_mono_left, stem_x_list)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [torch.relu(x) for x in inp_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]
        net_list_mono = [x.clone() for x in net_list]

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1).contiguous()
        disp = init_disp
        disp_preds = []
        for itr in range(iters):
            disp = disp.detach()
            if itr >= int(1):
                disp_mono_4x = disp_mono_4x.detach()
            geo_feat = geo_fn(disp, coords)
            if itr > int(iters-8):
                if itr == int(iters-7):
                    bs, _, _, _ = disp.shape
                    for i in range(bs):
                        with torch.autocast(device_type='cuda', dtype=torch.float32): 
                            scale, shift = compute_scale_shift(disp_mono_4x[i].clone().squeeze(1).to(torch.float32), disp[i].clone().squeeze(1).to(torch.float32))
                        disp_mono_4x[i] = scale * disp_mono_4x[i] + shift
                
                warped_right_mono = disp_warp(features_right[0], disp_mono_4x.clone().to(features_right[0].dtype))[0]  
                flaw_mono = warped_right_mono - features_left[0] 

                warped_right_stereo = disp_warp(features_right[0], disp.clone().to(features_right[0].dtype))[0]  
                flaw_stereo = warped_right_stereo - features_left[0] 
                geo_feat_mono = geo_fn(disp_mono_4x, coords)

            if itr <= int(iters-8):
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
            else:
                net_list, mask_feat_4, delta_disp = self.update_block_mix_stereo(net_list, inp_list, flaw_stereo, disp, geo_feat, flaw_mono, disp_mono_4x, geo_feat_mono, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
                net_list_mono, mask_feat_4_mono, delta_disp_mono = self.update_block_mix_mono(net_list_mono, inp_list, flaw_mono, disp_mono_4x, geo_feat_mono, flaw_stereo, disp, geo_feat, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
                disp_mono_4x = disp_mono_4x + delta_disp_mono
                disp_mono_4x_up = self.upsample_disp(disp_mono_4x, mask_feat_4_mono, stem_2x)
                disp_preds.append(disp_mono_4x_up)

            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)

            if itr == iters - 1:
                refine_value = self.REMP(disp_mono_4x_up, disp_up, image1, image2)
                disp_up = disp_up + refine_value
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        return init_disp, disp_preds, depth_mono


# =============================================================================
# MonsterV2: Coarse-to-Fine Hierarchical Search
# =============================================================================

class MonsterV2(nn.Module):
    """MonSter with hierarchical coarse-to-fine search.
    
    Uses local cost volumes at 1/8 and 1/4 scales centered around estimates
    from coarser scales, providing significant speedup without accuracy loss.
    
    Architecture:
        1/16: Full cost volume → iterations → 2x upsample
        1/8:  Local cost volume [est ± 15*2px] → iterations → 2x upsample  
        1/4:  Local cost volume [est ± 13*1px] → iterations → 4x upsample
    
    Args from config:
        scale_iters: List of iterations per scale [1/16, 1/8, 1/4], default [2, 2, 4]
        ndisps: Local disparity samples at [1/8, 1/4], default [15, 13]
        disp_intervals: Sample spacing at [1/8, 1/4], default [2, 1]
    """
    
    def __init__(self, args, export_onnx=False):
        super().__init__()
        self.args = args
        self.export_onnx = export_onnx
        
        # Hierarchical search config (from args or defaults)
        self.scale_iters = getattr(args, 'scale_iters', [2, 2, 4])  # 1/16, 1/8, 1/4
        self.ndisps = getattr(args, 'ndisps', [15, 13])  # For 1/8, 1/4 scales
        self.disp_intervals = getattr(args, 'disp_intervals', [2, 1])  # Pixel spacing
        
        # Encoder scale: 1.0 = full resolution, 0.5 = half resolution
        self.encoder_scale = getattr(args, 'encoder_scale', 1.0)
        
        context_dims = args.hidden_dims[::-1]
        
        # Mono depth model configs
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11], 
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39],
            'dinov3_vits14': [2, 5, 8, 11], 
            'dinov3_vitb14': [2, 5, 8, 11],
            'dinov3_vitl14': [4, 11, 17, 23],
            'vits_dd': [2, 5, 8, 11], 
            'vitb_dd': [2, 5, 8, 11],
            'vitl_dd': [4, 11, 17, 23], 
            'vitg_dd': [9, 19, 29, 39]
        }
        
        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
            'dinov3_vits14': {'encoder': 'dinov3_vits14', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'dinov3_vitb14': {'encoder': 'dinov3_vitb14', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'dinov3_vitl14': {'encoder': 'dinov3_vitl14', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vits_dd': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb_dd': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl_dd': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg_dd': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        dim_list_ = mono_model_configs[self.args.encoder]['features']
        dim_list = [dim_list_]
        
        # Single hierarchical update block
        # Uses single-layer GRU per scale with per-scale motion encoders
        self.update_block = BasicMultiUpdateBlockHierarchical(self.args, hidden_dims=args.hidden_dims)
        
        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=1) 
            for i in range(self.args.n_gru_layers)
        ])
        
        self.feat_transfer = Feat_transfer(dim_list)
        self.feat_transfer_cnet = Feat_transfer_cnet(dim_list, output_dim=args.hidden_dims[0])
        
        # Stem convolutions for image features at multiple scales
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )
        self.stem_8 = nn.Sequential(
            BasicConv_IN(48, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96), nn.ReLU()
        )
        self.stem_16 = nn.Sequential(
            BasicConv_IN(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(192), nn.ReLU()
        )
        
        # Per-scale feature matching convolutions
        # Mono features from Feat_transfer: [4x=48, 8x=64, 16x=192, 32x=160]
        # Stem features: [16x=192, 8x=96, 4x=48]
        # So concatenated: 1/16: 192+192=384, 1/8: 64+96=160, 1/4: 48+48=96
        self.conv_list = nn.ModuleList([
            BasicConv_IN(192+192, 96, kernel_size=3, padding=1, stride=1),  # 1/16: mono[2]=192 + stem_16=192
            BasicConv_IN(64+96, 96, kernel_size=3, padding=1, stride=1),    # 1/8: mono[1]=64 + stem_8=96
            BasicConv_IN(48+48, 96, kernel_size=3, padding=1, stride=1)     # 1/4: mono[0]=48 + stem_4=48
        ])
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)
        
        # Cost volume processing
        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att_list = nn.ModuleList([
            FeatureAtt(8, 192+192),  # 1/16
            FeatureAtt(8, 64+96),    # 1/8  
            FeatureAtt(8, 48+48)     # 1/4
        ])
        
        # Cost aggregation - use deeper 3D conv stacks for accuracy
        # (Not single-layer style, but not full hourglass due to local volume size constraints)
        self.cost_agg_16x = nn.Sequential(
            BasicConv(8, 16, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1),
            BasicConv(16, 16, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1),
            BasicConv(16, 8, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1)
        )
        self.cost_agg_8x = nn.Sequential(
            BasicConv(8, 16, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1),
            BasicConv(16, 16, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1),
            BasicConv(16, 8, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1)
        )
        
        # Option for full hourglass at 1/4 scale for higher accuracy
        self.use_hourglass_4x = getattr(args, 'use_hourglass_4x', False)
        if self.use_hourglass_4x:
            # Local hourglass with feature attention - designed for local cost volumes
            self.cost_agg_4x = hourglass_local(8, feat_channels=96)
            print("Using local hourglass at 1/4 scale for higher accuracy")
        else:
            # Simple 3D conv stack - faster but less accurate
            self.cost_agg_4x = nn.Sequential(
                BasicConv(8, 16, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1),
                BasicConv(16, 32, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1),
                BasicConv(32, 32, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1),
                BasicConv(32, 16, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1),
                BasicConv(16, 8, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1)
            )
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        
        # Disparity upsampling
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1))
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
        )
        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1))
        
        # 2x upsampling for scale transitions
        self.spx_gru_2x = nn.ModuleList([
            Conv2x(32, 96, True, keep_concat=False),  # 1/16 -> 1/8
            Conv2x(32, 48, True, keep_concat=False)   # 1/8 -> 1/4
        ])
        self.spx_gru_list = nn.ModuleList([
            nn.Conv2d(96, 9, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(48, 9, kernel_size=1, stride=1, padding=0)
        ])
        
        self.mask_feat_list = nn.ModuleList([
            nn.Sequential(nn.Conv2d(args.hidden_dims[0], 32, 3, padding=1), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(args.hidden_dims[1], 32, 3, padding=1), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(args.hidden_dims[2], 32, 3, padding=1), nn.ReLU(inplace=True))
        ])
        
        
        # Load mono depth model - support all encoder types
        # Override encoder in config with actual encoder name to ensure correct backbone is loaded
        depth_anything_config = {**mono_model_configs[args.encoder], 'encoder': args.encoder}
        depth_anything = DepthAnythingV2(**depth_anything_config)
        depth_anything_decoder = DepthAnythingV2_decoder(**mono_model_configs[args.encoder])
        
        self.mono_encoder = depth_anything.pretrained
        
        use_dinov2_decoder = 'dinov3' not in args.encoder and '_dd' in args.encoder
        
        if 'dinov3' not in args.encoder and not use_dinov2_decoder:
            # Case 1: DINOv2 + Standard Decoder (Legacy)
            state_dict_dpt = torch.load(f'./pretrained/depth_anything_v2_{args.encoder}.pth', map_location='cpu', weights_only=False)
            depth_anything.load_state_dict(state_dict_dpt, strict=True)
            depth_anything_decoder.load_state_dict(state_dict_dpt, strict=False)
            
            self.mono_decoder = depth_anything.depth_head
            self.feat_decoder = depth_anything_decoder.depth_head
            
        elif use_dinov2_decoder:
            # Case 2: DINOv2 Encoder + DINOv2 Depth Head
            config = mono_model_configs[args.encoder]
            out_channels = config['out_channels']
            
            # Determine correct embed_dim for DINOv2 encoder
            if 'vits' in args.encoder: embed_dim = 384
            elif 'vitb' in args.encoder: embed_dim = 768
            elif 'vitl' in args.encoder: embed_dim = 1024
            elif 'vitg' in args.encoder: embed_dim = 1536
            else: embed_dim = config['features']
            
            # 1. Instantiate adapters
            self.mono_decoder = DINOv2DepthAdapter(
                in_channels=[embed_dim] * 4, 
                channels=256, 
                post_process_channels=out_channels,
                n_output_channels=1
            )
            self.feat_decoder = DINOv2FeatureAdapter(
                in_channels=[embed_dim] * 4,
                channels=256,
                post_process_channels=mono_model_configs[args.encoder]['out_channels'],
                use_bn=False,
                use_clstoken=False
            )
            
            # 2. Load weights
            base_encoder = args.encoder.replace('_dd', '')
            backbone_name = f"dinov2_{base_encoder}14"
            
            load_dinov2_decoder_weights(self.mono_decoder, backbone_name=backbone_name)
            load_dinov2_decoder_weights(self.feat_decoder, backbone_name=backbone_name)
            
            # 3. Load Backbone weights
            pretrained_dir = "pretrained"
            if os.path.exists(pretrained_dir):
                candidates = [
                    f"{backbone_name}_reg4_pretrain.pth",
                    f"{backbone_name}_pretrain.pth",
                    f"{backbone_name}.pth"
                ]
                for fname in candidates:
                    local_path = os.path.join(pretrained_dir, fname)
                    if os.path.exists(local_path):
                        try:
                            state_dict = torch.load(local_path, map_location="cpu", weights_only=False)
                            if "model" in state_dict:
                                state_dict = state_dict["model"]
                            elif "teacher" in state_dict:
                                state_dict = state_dict["teacher"]
                            
                            self.mono_encoder.load_state_dict(state_dict, strict=False)
                            break
                        except Exception:
                            pass
            
        else:
            # Case 3: DINOv3 backbone - already loaded by DepthAnythingV2 init via dinov3_wrapper
            self.mono_decoder = depth_anything.depth_head
            self.feat_decoder = depth_anything_decoder.depth_head
        
        # Set requires_grad based on encoder type
        self.mono_encoder.requires_grad_(False)
        if 'dinov3' in args.encoder:
            self.mono_decoder.requires_grad_(True)
            self.feat_decoder.requires_grad_(True)
        elif use_dinov2_decoder:
            self.mono_decoder.requires_grad_(False)
            self.feat_decoder.requires_grad_(True)
        else:
            self.mono_decoder.requires_grad_(False)
        
        del depth_anything, depth_anything_decoder
        if 'dinov3' not in args.encoder and not use_dinov2_decoder:
            del state_dict_dpt
        
        # REMP refinement (optional, controlled by config)
        self.use_remp = getattr(args, 'use_remp', True)
        if self.use_remp:
            self.REMP = REMP()
        
        # Dual-branch update for final refinement
        self.update_block_mix_stereo = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)
        self.update_block_mix_mono = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def infer_mono(self, image1, image2):
        """Extract mono depth and multi-scale features.
        
        When encoder_scale < 1.0, runs encoder at reduced resolution for speed.
        """
        batch_size, _, height_ori, width_ori = image1.shape
        
        # Apply encoder scale if configured
        if self.encoder_scale < 1.0:
            # Scale down for encoder
            scaled_h = int(height_ori * self.encoder_scale)
            scaled_w = int(width_ori * self.encoder_scale)
            # Make divisible by patch size
            patch_size = 16 if 'dinov3' in self.args.encoder else 14
            scaled_h = (scaled_h // patch_size) * patch_size
            scaled_w = (scaled_w // patch_size) * patch_size
            resize_image1 = F.interpolate(image1, size=(scaled_h, scaled_w), mode='bilinear', align_corners=True)
            resize_image2 = F.interpolate(image2, size=(scaled_h, scaled_w), mode='bilinear', align_corners=True)
        else:
            if 'dinov3' in self.args.encoder:
                resize_image1, resize_image2 = image1, image2
                patch_size = 16
            else:
                resize_image1 = F.interpolate(image1, scale_factor=14/16, mode='bilinear', align_corners=True)
                resize_image2 = F.interpolate(image2, scale_factor=14/16, mode='bilinear', align_corners=True)
                patch_size = 14
        
        patch_size = 16 if 'dinov3' in self.args.encoder else 14
        patch_h, patch_w = resize_image1.shape[-2] // patch_size, resize_image1.shape[-1] // patch_size
        
        # Batch left and right images for efficiency
        stacked_images = torch.cat([resize_image1, resize_image2], dim=0)
        features_stacked = self.mono_encoder.get_intermediate_layers(
            stacked_images, 
            self.intermediate_layer_idx[self.args.encoder], 
            return_class_token=True
        )
        
        # Split features
        features_left_encoder, features_right_encoder = [], []
        for feat, cls_token in features_stacked:
            feat_left, feat_right = torch.chunk(feat, 2, dim=0)
            cls_left, cls_right = torch.chunk(cls_token, 2, dim=0)
            features_left_encoder.append((feat_left, cls_left))
            features_right_encoder.append((feat_right, cls_right))
        
        depth_mono = self.mono_decoder(features_left_encoder, patch_h, patch_w)
        depth_mono = F.relu(depth_mono)
        # Always upsample to original resolution
        depth_mono = F.interpolate(depth_mono, size=(height_ori, width_ori), mode='bilinear', align_corners=False)
        
        features_left = self.feat_decoder(features_left_encoder, patch_h, patch_w)
        features_right = self.feat_decoder(features_right_encoder, patch_h, patch_w)
        
        # When using encoder_scale < 1.0, upsample features to expected sizes
        if self.encoder_scale < 1.0:
            # Expected sizes: 1/4, 1/8, 1/16, 1/32 of original
            target_sizes = [
                (height_ori // 4, width_ori // 4),
                (height_ori // 8, width_ori // 8),
                (height_ori // 16, width_ori // 16),
                (height_ori // 32, width_ori // 32),
            ]
            features_left = [
                F.interpolate(feat, size=target_sizes[i], mode='bilinear', align_corners=False)
                for i, feat in enumerate(features_left)
            ]
            features_right = [
                F.interpolate(feat, size=target_sizes[i], mode='bilinear', align_corners=False)
                for i, feat in enumerate(features_right)
            ]
        
        return depth_mono, list(features_left), list(features_right)
    
    def upsample_disp_4x(self, disp, mask_feat, stem_2x):
        """Upsample disparity by 4x using learned weights."""
        xspx = self.spx_2_gru(mask_feat, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp * 4., spx_pred).unsqueeze(1)
        return up_disp
    
    def upsample_disp_2x(self, disp, mask_feat, stem, scale_idx):
        """Upsample disparity by 2x for scale transition."""
        xspx = self.spx_gru_2x[scale_idx](mask_feat, stem)
        spx_pred = self.spx_gru_list[scale_idx](xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample_2x(disp * 2., spx_pred).unsqueeze(1)
        return up_disp
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def forward(self, image1, image2, iters=None, test_mode=False):
        """Hierarchical coarse-to-fine forward pass.
        
        Args:
            image1: (B, 3, H, W) left image, range [0, 255]
            image2: (B, 3, H, W) right image
            iters: Ignored, uses self.scale_iters instead
            test_mode: If True, only return final disparity
        """
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            depth_mono, features_mono_left, features_mono_right = self.infer_mono(image1, image2)
        
        size = (image1.shape[-2], image1.shape[-1])
        
        # Multi-scale mono depth
        disp_mono_16x = F.interpolate(depth_mono, size=tuple(x // 16 for x in size), mode='bilinear', align_corners=False)
        disp_mono_8x = F.interpolate(depth_mono, size=tuple(x // 8 for x in size), mode='bilinear', align_corners=False)
        disp_mono_4x = F.interpolate(depth_mono, size=tuple(x // 4 for x in size), mode='bilinear', align_corners=False)
        disp_mono_list = [disp_mono_16x, disp_mono_8x, disp_mono_4x]
        
        # Feature transfer from mono to stereo matching
        features_left = self.feat_transfer(features_mono_left)
        features_right = self.feat_transfer(features_mono_right)
        
        # Build stem features
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x = self.stem_16(stem_8x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)
        stem_8y = self.stem_8(stem_4y)
        stem_16y = self.stem_16(stem_8y)
        
        stem_x_list = [stem_16x, stem_8x, stem_4x, stem_2x]
        stem_y_list = [stem_16y, stem_8y, stem_4y, stem_2y]
        
        # Context network from mono features
        cnet_list = self.feat_transfer_cnet(features_mono_left, stem_x_list[:3])
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) 
                    for i, conv in zip(inp_list, self.context_zqr_convs)]
        net_list_mono = [x.clone() for x in net_list]
        
        disp_preds = []
        disp_next = None
        
        # Coarse-to-fine: 1/16 -> 1/8 -> 1/4
        # features_left from feat_transfer: [4x (idx 0), 8x (idx 1), 16x (idx 2), 32x (idx 3)]
        # For scale 0 (1/16) we need idx 2, scale 1 (1/8) need idx 1, scale 2 (1/4) need idx 0
        feature_scale_idx = [2, 1, 0]  # Maps scale_idx to feature list index
        
        for scale_idx in range(3):
            # Concatenate mono features with stem features
            feat_idx = feature_scale_idx[scale_idx]
            feat_left = torch.cat((features_left[feat_idx], stem_x_list[scale_idx]), 1)
            feat_right = torch.cat((features_right[feat_idx], stem_y_list[scale_idx]), 1)
            
            # Feature matching
            match_left = self.desc(self.conv_list[scale_idx](feat_left))
            match_right = self.desc(self.conv_list[scale_idx](feat_right))
            
            if scale_idx == 0:
                # 1/16: Full cost volume
                if self.export_onnx:
                    gwc_volume = build_gwc_volume_onnx_v3(match_left, match_right, self.args.max_disp // 16, 8)
                else:
                    gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp // 16, 8)
                gwc_volume = self.corr_stem(gwc_volume)
                gwc_volume = self.corr_feature_att_list[scale_idx](gwc_volume, feat_left)
                geo_encoding_volume = self.cost_agg_16x(gwc_volume)
                
                prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
                init_disp = disparity_regression(prob, self.args.max_disp // 16)
                cur_disp = init_disp
                del prob
                
            else:
                # 1/8, 1/4: Local cost volume around previous estimate
                disp_range_samples = get_cur_disp_range_samples(
                    cur_disp=disp_next.squeeze(1),
                    ndisp=self.ndisps[scale_idx - 1],
                    disp_interval_pixel=self.disp_intervals[scale_idx - 1],
                    shape=[match_left.shape[0], match_left.shape[2], match_left.shape[3]]
                )
                
                gwc_volume = build_gwc_volume_selective(
                    match_left, match_right, disp_range_samples, 
                    self.ndisps[scale_idx - 1], 8
                )
                gwc_volume = self.corr_stem(gwc_volume)
                gwc_volume = self.corr_feature_att_list[scale_idx](gwc_volume, feat_left)
                if scale_idx == 1:
                    geo_encoding_volume = self.cost_agg_8x(gwc_volume)
                else:
                    # 1/4 scale: use hourglass with features if enabled
                    if self.use_hourglass_4x:
                        geo_encoding_volume = self.cost_agg_4x(gwc_volume, feat_left)
                    else:
                        geo_encoding_volume = self.cost_agg_4x(gwc_volume)
                cur_disp = disp_next
            
            del gwc_volume
            
            # Build geo encoding for lookup
            geo_fn = Combined_Geo_Encoding_Volume(
                match_left.float(), match_right.float(), 
                geo_encoding_volume.float(), 
                radius=self.args.corr_radius,
                num_levels=self.args.corr_levels
            )
            
            b, c, h, w = match_left.shape
            coords = torch.arange(w, device=match_left.device, dtype=match_left.dtype)
            coords = coords.reshape(1, 1, w, 1).repeat(b, h, 1, 1).contiguous()
            disp = cur_disp
            
            # Get mono disparity at this scale
            disp_mono = disp_mono_list[scale_idx]
            
            # Scale/shift mono depth to align with stereo at start of each scale
            # Skip for ONNX export - mono is already aligned at inference time
            if not self.export_onnx:
                bs = disp.shape[0]
                for bidx in range(bs):
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        scale_val, shift_val = compute_scale_shift(
                            disp_mono[bidx].squeeze(0).to(torch.float32),
                            disp[bidx].squeeze(0).to(torch.float32)
                        )
                    disp_mono_list[scale_idx][bidx] = scale_val * disp_mono[bidx] + shift_val
                disp_mono = disp_mono_list[scale_idx]
            
            # Single-layer GRU iterations at this scale
            for itr in range(self.scale_iters[scale_idx]):
                disp = disp.detach()
                
                # Lookup geo features
                if scale_idx == 0:
                    geo_feat = geo_fn(disp, coords)
                else:
                    geo_feat = geo_fn(disp, coords, self.ndisps[scale_idx - 1])
                
                # Check if this is the last iteration of the last scale for dual-branch
                is_last_scale = (scale_idx == 2)
                is_last_iter = (itr == self.scale_iters[scale_idx] - 1)
                
                if is_last_scale and is_last_iter:
                    # Dual-branch update at final iteration of 1/4 scale
                    warped_right_stereo = disp_warp(feat_right, disp.clone().to(feat_right.dtype))[0]
                    flaw_stereo = warped_right_stereo - feat_left
                    
                    warped_right_mono = disp_warp(feat_right, disp_mono.clone().to(feat_right.dtype))[0]
                    flaw_mono = warped_right_mono - feat_left
                    
                    geo_feat_mono = geo_fn(disp_mono, coords, self.ndisps[scale_idx - 1])
                    
                    # Stereo branch update (guided by mono)
                    net_list, mask_feat, delta_disp = self.update_block_mix_stereo(
                        net_list, inp_list, flaw_stereo, disp, geo_feat,
                        flaw_mono, disp_mono, geo_feat_mono,
                        iter16=False, iter08=False
                    )
                    
                    # Mono branch update (guided by stereo)
                    net_list_mono, mask_feat_mono, delta_disp_mono = self.update_block_mix_mono(
                        net_list_mono, inp_list, flaw_mono, disp_mono, geo_feat_mono,
                        flaw_stereo, disp, geo_feat,
                        iter16=False, iter08=False
                    )
                    
                    # Update mono disparity and add to predictions
                    disp_mono = disp_mono + delta_disp_mono
                    disp_mono_up = self.upsample_disp_4x(disp_mono, mask_feat_mono, stem_2x)
                    disp_preds.append(disp_mono_up)
                else:
                    # Single-layer GRU update at current scale
                    # Set iter flags based on current scale (hierarchical convention)
                    # 1/16: iter16=T, iter08=F, iter04=F
                    # 1/8:  iter16=T, iter08=T, iter04=F
                    # 1/4:  iter16=T, iter08=T, iter04=T
                    iter16 = True  # Always enabled (context flows down)
                    iter08 = (scale_idx >= 1)  # 1/8 and 1/4
                    iter04 = (scale_idx == 2)  # Only 1/4
                    
                    net_list, mask_feat, delta_disp = self.update_block(
                        net_list, inp_list, geo_feat, disp, disp_mono,
                        iter04=iter04, iter08=iter08, iter16=iter16
                    )
                
                disp = disp + delta_disp
                
                # Collect predictions
                if not test_mode or (is_last_scale and is_last_iter):
                    if scale_idx == 0:
                        # 1/16: bilinear upsample to full
                        disp_up = F.interpolate(disp * 16, size=size, mode='bilinear', align_corners=True)
                    elif scale_idx == 1:
                        # 1/8: bilinear upsample to full
                        disp_up = F.interpolate(disp * 8, size=size, mode='bilinear', align_corners=True)
                    else:
                        # 1/4: learned upsampling
                        disp_up = self.upsample_disp_4x(disp, mask_feat, stem_2x)
                    disp_preds.append(disp_up)
            
            # Upsample disparity 2x for next scale
            if scale_idx < 2:
                disp_next = F.interpolate(disp * 2, scale_factor=2, mode='bilinear', align_corners=True)
        
        # REMP refinement (configurable)
        if self.use_remp:
            disp_mono_final = F.interpolate(disp_mono_list[2], size=size, mode='bilinear', align_corners=False)
            refine_value = self.REMP(disp_mono_final, disp_up, image1, image2)
            disp_up = disp_up + refine_value
            disp_preds[-1] = disp_up
        
        if test_mode:
            return disp_up
        
        init_disp_up = F.interpolate(init_disp * 16, size=size, mode='bilinear', align_corners=True)
        return init_disp_up, disp_preds, depth_mono
