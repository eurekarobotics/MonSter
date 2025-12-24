import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels*2, out_channels*mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume
        

def build_gwc_volume_onnx(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    
    # Handle disparity 0 case
    volume[:, :, 0, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    
    # Handle disparity > 0 cases using padding and slicing
    maxdisp = min(maxdisp, W)  # Ensure maxdisp does not exceed width
    for i in range(1, maxdisp):
        shifted_ref = refimg_fea[:, :, :, i:]
        shifted_target = targetimg_fea[:, :, :, :-i]
        padded_ref = F.pad(shifted_ref, (0, i, 0, 0), mode='constant', value=0)
        padded_target = F.pad(shifted_target, (0, i, 0, 0), mode='constant', value=0)
        corr = groupwise_correlation(padded_ref, padded_target, num_groups)
        volume[:, :, i, :, i:] = corr[:, :, :, :-i]
    
    volume = volume.contiguous()
    return volume


def build_gwc_volume_onnx_v2(refimg_fea, targetimg_fea, maxdisp, num_groups):
    """ONNX-compatible GWC volume using padding and masking.
    
    More memory efficient than build_gwc_volume_onnx as it avoids creating
    separate padded tensors for ref and target per disparity.
    
    The original correlation is: ref[:,:,:,i:] * target[:,:,:,:-i]
    This means ref columns i,i+1,... correlate with target columns 0,1,...
    The result is placed at volume columns i,i+1,...
    """
    B, C, H, W = refimg_fea.shape
    channels_per_group = C // num_groups
    
    # Reshape for group-wise operation
    ref = refimg_fea.view(B, num_groups, channels_per_group, H, W)
    target = targetimg_fea.view(B, num_groups, channels_per_group, H, W)
    
    volume_list = []
    for i in range(maxdisp):
        if i == 0:
            corr = (ref * target).mean(dim=2)  # [B, num_groups, H, W]
        else:
            ref_slice = ref[:, :, :, :, i:]  # [B, num_groups, channels_per_group, H, W-i]
            target_slice = target[:, :, :, :, :-i]  # [B, num_groups, channels_per_group, H, W-i]
            corr_slice = (ref_slice * target_slice).mean(dim=2)  # [B, num_groups, H, W-i]
            corr = F.pad(corr_slice, (i, 0, 0, 0), value=0)  # [B, num_groups, H, W]
        volume_list.append(corr)
    
    volume = torch.stack(volume_list, dim=2)  # [B, num_groups, maxdisp, H, W]
    return volume.contiguous()


def build_gwc_volume_onnx_v3(refimg_fea, targetimg_fea, maxdisp, num_groups):
    """ONNX-compatible GWC volume using gather-based shifting.
    
    More memory efficient than build_gwc_volume_onnx_v2 as it avoids creating
    separate padded tensors for ref and target per disparity.
    """
    B, C, H, W = refimg_fea.shape
    channels_per_group = C // num_groups
    
    # Pad target on the left side with zeros
    # After padding: [B, C, H, W + maxdisp - 1]
    target_padded = F.pad(targetimg_fea, (maxdisp - 1, 0, 0, 0), value=0)
    W_padded = W + maxdisp - 1
    
    # Create index tensor for gathering shifted columns
    # For disparity d, we want columns [maxdisp-1-d, maxdisp-d, ..., maxdisp-1-d+W-1]
    # Shape: [maxdisp, W]
    base_col_idx = torch.arange(W, device=refimg_fea.device).view(1, W)  # [1, W]
    disp_offset = torch.arange(maxdisp - 1, -1, -1, device=refimg_fea.device).view(maxdisp, 1)  # [maxdisp, 1]
    # Column indices for each disparity: [maxdisp, W]
    col_indices = base_col_idx + disp_offset
    
    # Expand indices for gather: [B, C, H, maxdisp, W]
    col_indices = col_indices.view(1, 1, 1, maxdisp, W).expand(B, C, H, maxdisp, W)
    
    # Expand target_padded to [B, C, H, 1, W_padded] then gather along last dim
    target_expanded = target_padded.unsqueeze(3).expand(B, C, H, maxdisp, W_padded)
    
    # Gather to create shifted versions: [B, C, H, maxdisp, W]
    target_shifted = torch.gather(target_expanded, dim=4, index=col_indices)
    
    # Reshape for group-wise correlation
    ref = refimg_fea.view(B, num_groups, channels_per_group, H, W)
    target_shifted = target_shifted.view(B, num_groups, channels_per_group, H, maxdisp, W)
    
    # Compute correlation for all disparities at once
    # ref: [B, num_groups, channels_per_group, H, W]
    # target_shifted: [B, num_groups, channels_per_group, H, maxdisp, W]
    # Result: [B, num_groups, H, maxdisp, W]
    volume = (ref.unsqueeze(4) * target_shifted).mean(dim=2)
    
    # Transpose to [B, num_groups, maxdisp, H, W]
    volume = volume.permute(0, 1, 3, 2, 4)
    
    # Create mask for valid regions
    # At disparity i, columns 0 to i-1 are invalid
    disp_idx = torch.arange(maxdisp, device=refimg_fea.device).view(1, 1, maxdisp, 1, 1)
    col_idx_mask = torch.arange(W, device=refimg_fea.device).view(1, 1, 1, 1, W)
    mask = (col_idx_mask >= disp_idx).to(refimg_fea.dtype)
    
    volume = volume * mask
    
    return volume.contiguous()

def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def correlation(fea1, fea2):
    cost = torch.sum((fea1 * fea2), dim=1, keepdim=True)
    return cost

def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume



def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1))

    def forward(self, cv, feat):
        '''
        '''
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att)*cv
        return cv

def context_upsample(disp_low, up_weights):
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = disp_low.shape
        
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    disp = (disp_unfold*up_weights).sum(1)
        
    return disp

class Propagation(nn.Module):
    def __init__(self):
        super(Propagation, self).__init__()
        self.replicationpad = nn.ReplicationPad2d(1)

    def forward(self, disparity_samples):

        one_hot_filter = torch.zeros(5, 1, 3, 3, device=disparity_samples.device).float()
        one_hot_filter[0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 2] = 1.0
        disparity_samples = self.replicationpad(disparity_samples)
        aggregated_disparity_samples = F.conv2d(disparity_samples,
                                                    one_hot_filter,padding=0)
                                                    
        return aggregated_disparity_samples
        

class Propagation_prob(nn.Module):
    def __init__(self):
        super(Propagation_prob, self).__init__()
        self.replicationpad = nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))

    def forward(self, prob_volume):
        one_hot_filter = torch.zeros(5, 1, 1, 3, 3, device=prob_volume.device).float()
        one_hot_filter[0, 0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 0, 2] = 1.0

        prob_volume = self.replicationpad(prob_volume)
        prob_volume_propa = F.conv3d(prob_volume, one_hot_filter,padding=0)


        return prob_volume_propa
