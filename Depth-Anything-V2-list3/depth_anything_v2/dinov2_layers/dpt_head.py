
import math
import copy
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# --- Utils ---

def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias='auto', norm_cfg=None, act_cfg=dict(type='ReLU'), order=('conv', 'norm', 'act')):
        super().__init__()
        self.order = order
        
        if bias == 'auto':
            bias = False if norm_cfg else True
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        
        self.norm = None
        if norm_cfg:
            if norm_cfg['type'] == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_cfg['type'] == 'SyncBN':
                self.norm = nn.BatchNorm2d(out_channels) # Fallback to BN for simplicity if SyncBN not available/needed
            else:
                raise NotImplementedError(f"Norm type {norm_cfg['type']} not implemented")
                
        self.act = None
        if act_cfg:
            if act_cfg['type'] == 'ReLU':
                self.act = nn.ReLU(inplace=True)
            elif act_cfg['type'] == 'GELU':
                self.act = nn.GELU()
            else:
                raise NotImplementedError(f"Act type {act_cfg['type']} not implemented")

    def forward(self, x):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm':
                if self.norm: x = self.norm(x)
            elif layer == 'act':
                if self.act: x = self.act(x)
        return x

def build_activation_layer(cfg):
    if cfg['type'] == 'ReLU':
        return nn.ReLU(inplace=True)
    elif cfg['type'] == 'GELU':
        return nn.GELU()
    else:
        raise NotImplementedError(f"Act type {cfg['type']} not implemented")

# --- Decode Head ---

class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg

class DepthBaseDecodeHead(BaseModule, metaclass=ABCMeta):
    def __init__(
        self,
        in_channels,
        channels=96,
        conv_cfg=None,
        act_cfg=dict(type="ReLU"),
        loss_decode=dict(type="SigLoss", valid_mask=True, loss_weight=10),
        sampler=None,
        align_corners=False,
        min_depth=1e-3,
        max_depth=None,
        norm_cfg=None,
        classify=False,
        n_bins=256,
        bins_strategy="UD",
        norm_strategy="linear",
        scale_up=False,
        **kwargs
    ):
        super(DepthBaseDecodeHead, self).__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        # self.loss_decode = ... # Skip loss for inference
        self.align_corners = align_corners
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.norm_cfg = norm_cfg
        self.classify = classify
        self.n_bins = n_bins
        self.scale_up = scale_up
        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy

        if self.classify:
            self.softmax = nn.Softmax(dim=1)
            self.conv_depth = nn.Conv2d(channels, n_bins, kernel_size=3, padding=1, stride=1)
        else:
            self.conv_depth = nn.Conv2d(channels, 1, kernel_size=3, padding=1, stride=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    @abstractmethod
    def forward(self, inputs, img_metas):
        pass

    def depth_pred(self, feat):
        """Prediction each pixel."""
        if self.classify:
            logit = self.conv_depth(feat)

            if self.bins_strategy == "UD":
                bins = torch.linspace(self.min_depth, self.max_depth, self.n_bins, device=feat.device)
            elif self.bins_strategy == "SID":
                bins = torch.logspace(self.min_depth, self.max_depth, self.n_bins, device=feat.device)

            # following Adabins, default linear
            if self.norm_strategy == "linear":
                logit = torch.relu(logit)
                eps = 0.1
                logit = logit + eps
                logit = logit / logit.sum(dim=1, keepdim=True)
            elif self.norm_strategy == "softmax":
                logit = torch.softmax(logit, dim=1)
            elif self.norm_strategy == "sigmoid":
                logit = torch.sigmoid(logit)
                logit = logit / logit.sum(dim=1, keepdim=True)

            output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)

        else:
            if self.scale_up:
                output = self.sigmoid(self.conv_depth(feat)) * self.max_depth
            else:
                output = self.relu(self.conv_depth(feat)) + self.min_depth
        return output

# --- DPT Head ---

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


class HeadDepth(nn.Module):
    def __init__(self, features):
        super(HeadDepth, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.head(x)
        return x


class ReassembleBlocks(BaseModule):
    def __init__(
        self, in_channels=768, out_channels=[96, 192, 384, 768], readout_type="ignore", patch_size=16, init_cfg=None
    ):
        super(ReassembleBlocks, self).__init__(init_cfg)

        assert readout_type in ["ignore", "add", "project"]
        self.readout_type = readout_type
        self.patch_size = patch_size

        self.projects = nn.ModuleList(
            [
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    act_cfg=None,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )
        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), build_activation_layer(dict(type="GELU")))
                )

    def forward(self, inputs):
        assert isinstance(inputs, list)
        out = []
        for i, x in enumerate(inputs):
            assert len(x) == 2
            x, cls_token = x[0], x[1]
            feature_shape = x.shape
            if self.readout_type == "project":
                x = x.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(feature_shape)
            elif self.readout_type == "add":
                x = x.flatten(2) + cls_token.unsqueeze(-1)
                x = x.reshape(feature_shape)
            else:
                pass
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class PreActResidualConvUnit(BaseModule):
    def __init__(self, in_channels, act_cfg, norm_cfg, stride=1, dilation=1, init_cfg=None):
        super(PreActResidualConvUnit, self).__init__(init_cfg)

        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=("act", "conv", "norm"),
        )

        self.conv2 = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=("act", "conv", "norm"),
        )

    def forward(self, inputs):
        inputs_ = inputs.clone()
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs_


class FeatureFusionBlock(BaseModule):
    def __init__(self, in_channels, act_cfg, norm_cfg, expand=False, align_corners=True, init_cfg=None):
        super(FeatureFusionBlock, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners

        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2

        self.project = ConvModule(self.in_channels, self.out_channels, kernel_size=1, act_cfg=None, bias=True)

        self.res_conv_unit1 = PreActResidualConvUnit(in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.res_conv_unit2 = PreActResidualConvUnit(in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg)

    def forward(self, *inputs):
        x = inputs[0]
        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = resize(inputs[1], size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = resize(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        x = self.project(x)
        return x


class DPTHead(DepthBaseDecodeHead):
    def __init__(
        self,
        embed_dims=768,
        post_process_channels=[96, 192, 384, 768],
        readout_type="ignore",
        patch_size=16,
        expand_channels=False,
        **kwargs
    ):
        super(DPTHead, self).__init__(**kwargs)

        self.in_channels = self.in_channels
        self.expand_channels = expand_channels
        self.reassemble_blocks = ReassembleBlocks(embed_dims, post_process_channels, readout_type, patch_size)

        self.post_process_channels = [
            channel * math.pow(2, i) if expand_channels else channel for i, channel in enumerate(post_process_channels)
        ]
        self.convs = nn.ModuleList()
        for channel in self.post_process_channels:
            self.convs.append(ConvModule(channel, self.channels, kernel_size=3, padding=1, act_cfg=None, bias=False))
        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(FeatureFusionBlock(self.channels, self.act_cfg, self.norm_cfg))
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = ConvModule(self.channels, self.channels, kernel_size=3, padding=1, norm_cfg=self.norm_cfg)
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels
        self.conv_depth = HeadDepth(self.channels)

    def forward(self, inputs, img_metas=None, return_features=False):
        assert len(inputs) == self.num_reassemble_blocks
        x = [inp for inp in inputs]
        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        
        # fusion_blocks are [refinenet4, refinenet3, refinenet2, refinenet1] ?
        # In DepthAnythingV2: 
        # path_4 = refinenet4(layer_4_rn)
        # path_3 = refinenet3(path_4, layer_3_rn)
        # ...
        # Here:
        # out = fusion_blocks[0](x[-1]) -> refinenet4(x[3])
        # out = fusion_blocks[i](out, x[-(i+1)])
        
        outs = []
        out = self.fusion_blocks[0](x[-1])
        outs.append(out)
        
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])
            outs.append(out)
            
        path_4 = outs[0]
        path_1 = outs[-1]
        
        features = list(reversed(outs))
        
        out = self.project(out)
        depth = self.depth_pred(out)
        
        if return_features:
            return depth, features
        return depth
