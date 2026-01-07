import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler


class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []
        
        # Pre-compute dx tensor (optimization: avoid creating in every __call__)
        r = radius
        self.dx = torch.linspace(-r, r, 2*r+1).view(1, 1, 2*r+1, 1)
        self.device = init_fmap1.device
        self.dx = self.dx.to(self.device)

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)




    def __call__(self, disp, coords, ndisps=None):
        """Sample from geo-encoding volume at current disparity estimates.
        
        Args:
            disp: (B, 1, H, W) current disparity estimate
            coords: (B, H, W, 1) x-coordinates of reference image
            ndisps: If provided, use local mode where geo_volume has shape 
                    [B, 8, ndisps, H, W] and we sample around the center.
                    If None, use full volume mode with absolute disparity.
        
        Returns:
            out: (B, C, H, W) sampled features
        """
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        
        # Use pre-computed dx (optimization: no tensor creation in hot path)
        # Convert to disp's dtype for TensorRT compatibility (avoids NaN in FP16)
        dx = self.dx.to(dtype=disp.dtype)
        
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            
            if ndisps is not None:
                # Local mode: sample around center of local volume
                # geo_volume has disparity dim = ndisps, center is at ndisps//2
                center = torch.tensor([ndisps // 2], dtype=disp.dtype, device=disp.device)
                center = center.view(1, 1, 1, 1).expand(b * h * w, 1, 1, 1)
                x0 = dx + center
            else:
                # Full mode: use absolute disparity values
                x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    
    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr