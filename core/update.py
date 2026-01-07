import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):

        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)
        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, *x):
        # horizontal
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

def interp(x, dest):
    original_dtype = x.dtype
    x_fp32 = x.float()
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    with torch.cuda.amp.autocast(enabled=False):
        output_fp32 = F.interpolate(x_fp32, dest.shape[2:], **interp_args)
    if original_dtype != torch.float32:
        output = output_fp32.to(original_dtype)
    else:
        output = output_fp32
    return output

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

# def interp(x, dest):
#     interp_args = {'mode': 'bilinear', 'align_corners': True}
#     return F.interpolate(x, dest.shape[2:], **interp_args)

class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr=None, disp=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp

class BasicMultiUpdateBlock_mix(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_mix(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, flaw_stereo=None, disp=None, corr=None, flaw_mono=None, disp_mono=None, corr_mono=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp_all = self.disp_head(net[0])
        delta_disp = delta_disp_all[:, :1]
        delta_disp_mono = delta_disp_all[:, 1:2]
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp, delta_disp_mono

class BasicMotionEncoder_mix(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_mix, self).__init__()
        self.args = args
        cor_planes = 96 + args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convc1_mono = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1_mono = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.conv = nn.Conv2d(128, 64-1, 3, padding=1)
        self.conv_mono = nn.Conv2d(128, 64-1, 3, padding=1)


    def forward(self, disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono):
        cor = F.relu(self.convc1(torch.cat([corr, flaw_stereo], dim=1)))
        cor = F.relu(self.convc2(cor))
        cor_mono = F.relu(self.convc1_mono(torch.cat([corr_mono, flaw_mono], dim=1)))
        cor_mono = F.relu(self.convc2_mono(cor_mono))

        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        disp_mono_ = F.relu(self.convd1_mono(disp_mono))
        disp_mono_ = F.relu(self.convd2_mono(disp_mono_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        cor_disp_mono = torch.cat([cor_mono, disp_mono_], dim=1)

        out = F.relu(self.conv(cor_disp))
        out_mono = F.relu(self.conv_mono(cor_disp_mono))

        return torch.cat([out, disp, out_mono, disp_mono], dim=1)


class BasicMultiUpdateBlock_2(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_2(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, flaw_stereo=None, disp=None, corr=None, confidence=None, flaw_mono=None, disp_mono=None, corr_mono=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono, confidence)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp


class BasicMotionEncoder_2(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_2, self).__init__()
        self.args = args
        cor_planes = 96 + args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(int(cor_planes + 1), 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convc1_mono = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1_mono = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.conv = nn.Conv2d(129, 64-1, 3, padding=1)
        self.conv_mono = nn.Conv2d(128, 64-1, 3, padding=1)


    def forward(self, disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono, confidence):
        cor = F.relu(self.convc1(torch.cat([corr, flaw_stereo, confidence], dim=1)))
        cor = F.relu(self.convc2(cor))
        cor_mono = F.relu(self.convc1_mono(torch.cat([corr_mono, flaw_mono], dim=1)))
        cor_mono = F.relu(self.convc2_mono(cor_mono))

        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        disp_mono_ = F.relu(self.convd1_mono(disp_mono))
        disp_mono_ = F.relu(self.convd2_mono(disp_mono_))

        cor_disp = torch.cat([cor, disp_, confidence], dim=1)
        cor_disp_mono = torch.cat([cor_mono, disp_mono_], dim=1)

        out = F.relu(self.conv(cor_disp))
        out_mono = F.relu(self.conv_mono(cor_disp_mono))

        return torch.cat([out, disp, out_mono, disp_mono], dim=1)
    

class BasicMultiUpdateBlock_mono(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_mono(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr=None, disp=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp


class BasicMotionEncoder_mono(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_mono, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)


class BasicMultiUpdateBlock_mix_conf(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_mix_conf(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, flaw_stereo=None, disp=None, corr=None, flaw_mono=None, disp_mono=None, corr_mono=None, conf_stereo=None, conf_mono=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono, conf_stereo, conf_mono)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp_all = self.disp_head(net[0])
        delta_disp = delta_disp_all[:, :1]
        delta_disp_mono = delta_disp_all[:, 1:2]
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp, delta_disp_mono
    

class BasicMotionEncoder_mix_conf(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_mix_conf, self).__init__()
        self.args = args
        cor_planes = 96 + args.corr_levels * (2*args.corr_radius + 1) * (8+1)

        self.conv_conf1 = nn.Conv2d(1, 64, 7, padding=3)
        self.conv_conf2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv_conf1_mono = nn.Conv2d(1, 64, 7, padding=3)
        self.conv_conf2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.convc1 = nn.Conv2d(int(cor_planes + 64), 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convc1_mono = nn.Conv2d(int(cor_planes + 64), 64, 1, padding=0)
        self.convc2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1_mono = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.conv = nn.Conv2d(128, 64-2, 3, padding=1)
        self.conv_mono = nn.Conv2d(128, 64-2, 3, padding=1)


    def forward(self, disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono, conf_stereo, conf_mono):

        conf_stereo_ = F.relu(self.conv_conf1(conf_stereo))
        conf_stereo_ = F.relu(self.conv_conf2(conf_stereo_))

        conf_mono_ = F.relu(self.conv_conf1_mono(conf_mono))
        conf_mono_ = F.relu(self.conv_conf2_mono(conf_mono_))

        cor = F.relu(self.convc1(torch.cat([corr, flaw_stereo, conf_stereo_], dim=1)))
        cor = F.relu(self.convc2(cor))
        cor_mono = F.relu(self.convc1_mono(torch.cat([corr_mono, flaw_mono, conf_mono_], dim=1)))
        cor_mono = F.relu(self.convc2_mono(cor_mono))

        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        disp_mono_ = F.relu(self.convd1_mono(disp_mono))
        disp_mono_ = F.relu(self.convd2_mono(disp_mono_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        cor_disp_mono = torch.cat([cor_mono, disp_mono_], dim=1)

        out = F.relu(self.conv(cor_disp))
        out_mono = F.relu(self.conv_mono(cor_disp_mono))

        return torch.cat([out, disp, conf_stereo, out_mono, disp_mono, conf_mono], dim=1)


class BasicMultiUpdateBlock_mix2(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_mix2(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, flaw_stereo=None, disp=None, corr=None, flaw_mono=None, disp_mono=None, corr_mono=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net
        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp

class BasicMotionEncoder_mix2(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_mix2, self).__init__()
        self.args = args
        cor_planes = 96 + args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convc1_mono = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1_mono = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.conv = nn.Conv2d(128, 64-1, 3, padding=1)
        self.conv_mono = nn.Conv2d(128, 64-1, 3, padding=1)


    def forward(self, disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono):
        cor = F.relu(self.convc1(torch.cat([corr, flaw_stereo], dim=1)))
        cor = F.relu(self.convc2(cor))
        cor_mono = F.relu(self.convc1_mono(torch.cat([corr_mono, flaw_mono], dim=1)))
        cor_mono = F.relu(self.convc2_mono(cor_mono))

        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        disp_mono_ = F.relu(self.convd1_mono(disp_mono))
        disp_mono_ = F.relu(self.convd2_mono(disp_mono_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        cor_disp_mono = torch.cat([cor_mono, disp_mono_], dim=1)

        out = F.relu(self.conv(cor_disp))
        out_mono = F.relu(self.conv_mono(cor_disp_mono))

        return torch.cat([out, disp, out_mono, disp_mono], dim=1)


# =============================================================================
# MonsterV2 Single-Layer ConvGRU
# =============================================================================

class BasicMotionEncoderMono(nn.Module):
    """Motion encoder with mono disparity input for hierarchical updates.
    
    Takes stereo disparity, mono disparity, and correlation features as input.
    Used at each scale in the hierarchical pipeline.
    
    Args:
        args: Config with corr_levels and corr_radius
        corr_radius: Correlation lookup radius for this scale
    """
    def __init__(self, args, corr_radius):
        super(BasicMotionEncoderMono, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2 * corr_radius + 1) * (8 + 1)
        
        # Correlation feature processing
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        
        # Stereo disparity processing
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        
        # Mono disparity processing
        self.convd1_mono = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2_mono = nn.Conv2d(64, 64, 3, padding=1)
        
        # Fuse all features
        self.conv = nn.Conv2d(64 + 64 + 64, 128 - 1, 3, padding=1)

    def forward(self, disp, disp_mono, corr):
        # Process correlation
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        
        # Process stereo disparity
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))
        
        # Process mono disparity
        disp_mono_ = F.relu(self.convd1_mono(disp_mono))
        disp_mono_ = F.relu(self.convd2_mono(disp_mono_))
        
        # Fuse and output
        cor_disp = torch.cat([cor, disp_, disp_mono_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)


class BasicMultiUpdateBlockHierarchical(nn.Module):
    """Hierarchical update block with single-layer ConvGRU per scale.
    
    Hierarchical style: at each scale, only ONE GRU layer runs with its own
    motion encoder and disparity head. Context flows from coarser to finer scales.
    
    Architecture:
        - 1/16: gru16_m only → disp_head_16x
        - 1/8:  gru08_m only (with net[2] context) → disp_head_8x
        - 1/4:  gru04 only (with net[1] context) → disp_head_4x
    
    Args:
        args: Config with corr_radius list per scale, n_gru_layers, hidden_dims
        hidden_dims: List of hidden dimensions [1/16, 1/8, 1/4]
    """
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        encoder_output_dim = 128
        
        # Per-scale motion encoders (with mono disparity input)
        # args.corr_radius should be a list: [radius_16x, radius_8x, radius_4x]
        corr_radii = args.corr_radius if isinstance(args.corr_radius, list) else [args.corr_radius] * 3
        self.encoder16 = BasicMotionEncoderMono(args, corr_radii[0] if len(corr_radii) > 0 else 4)
        self.encoder8 = BasicMotionEncoderMono(args, corr_radii[1] if len(corr_radii) > 1 else 4)
        self.encoder4 = BasicMotionEncoderMono(args, corr_radii[2] if len(corr_radii) > 2 else 4)
        
        # Per-scale GRUs (single layer per scale)
        # gru16_m: takes only motion features
        self.gru16_m = ConvGRU(hidden_dims[0], encoder_output_dim)
        # gru08_m: takes motion features + context from 1/16
        self.gru08_m = ConvGRU(hidden_dims[1], encoder_output_dim + hidden_dims[0])
        # gru04: takes motion features + context from 1/8
        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1])
        
        # Per-scale disparity heads
        self.disp_head_16x = DispHead(hidden_dims[0], hidden_dim=256, output_dim=1)
        self.disp_head_8x = DispHead(hidden_dims[1], hidden_dim=256, output_dim=1)
        self.disp_head_4x = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        
        # Per-scale mask feature extractors (for upsampling)
        self.mask_feat_16 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.mask_feat_8 = nn.Sequential(
            nn.Conv2d(hidden_dims[1], 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, net, inp, corr, disp, disp_mono, iter04=True, iter08=True, iter16=True):
        """Forward pass with single-layer GRU at specified scale.
        
        Args:
            net: List of GRU hidden states [net_4x, net_8x, net_16x]
            inp: List of context inputs [inp_4x, inp_8x, inp_16x]
            corr: Correlation features at current scale
            disp: Current stereo disparity estimate
            disp_mono: Current mono disparity estimate
            iter04, iter08, iter16: Flags to select active scale
                - (False, False, True) = 1/16 scale
                - (False, True, True)  = 1/8 scale  
                - (True, True, True)   = 1/4 scale
        
        Returns:
            net: Updated hidden states
            mask_feat: Mask features for upsampling
            delta_disp: Disparity residual
        """
        if not iter04 and not iter08 and iter16:
            # 1/16 scale: ONLY gru16_m runs
            motion_features = self.encoder16(disp, disp_mono, corr)
            net[2] = self.gru16_m(net[2], *(inp[2]), motion_features)
            delta_disp = self.disp_head_16x(net[2])
            mask_feat = self.mask_feat_16(net[2])
            
        elif not iter04 and iter08 and iter16:
            # 1/8 scale: ONLY gru08_m runs, with context from 1/16
            motion_features = self.encoder8(disp, disp_mono, corr)
            net[1] = self.gru08_m(net[1], *(inp[1]), motion_features, interp(net[2], net[1]))
            delta_disp = self.disp_head_8x(net[1])
            mask_feat = self.mask_feat_8(net[1])
            
        elif iter04 and iter08 and iter16:
            # 1/4 scale: ONLY gru04 runs, with context from 1/8
            motion_features = self.encoder4(disp, disp_mono, corr)
            net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            delta_disp = self.disp_head_4x(net[0])
            mask_feat = self.mask_feat_4(net[0])
        else:
            raise ValueError(f"Invalid combination: iter04={iter04}, iter08={iter08}, iter16={iter16}")

        return net, mask_feat, delta_disp