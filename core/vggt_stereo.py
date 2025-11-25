# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
# from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.dpt_head import DPTHead

from core.vggt_loss import regression_loss


def disparity_loss(
    predictions: dict,
    disp_gt: torch.Tensor,
    valid: torch.Tensor,
    max_disp: int = 640,
):
    # (N, 1, H, W, 1)
    disp_pred = predictions["depth"]
    # (N, 1, H, W)
    conf_pred = predictions["depth_conf"]

    N, _, H, W, _ = disp_pred.shape

    # disp_gt: (N, 1, H, W)
    # valid: (N, H, W)
    disp_gt = disp_gt.view(N, 1, H, W, 1)
    valid = valid.view(N, 1, H, W)

    mag = torch.sum(disp_gt**2, dim=-1).sqrt()
    valid = ((valid > 0) & (mag < max_disp))

    # conf_loss, grad_loss, disp_loss = regression_loss(
    #     disp_pred,
    #     disp_gt,
    #     valid,
    #     conf_pred,
    #     gradient_loss_fn="grad",
    #     # gradient_loss_fn=[],
    #     gamma=0.1,
    #     # alpha=0.002,
    #     alpha=0.02,
    # )
    # total_loss = conf_loss + grad_loss + disp_loss

    epe = torch.sum((disp_pred - disp_gt) ** 2, dim=-1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    if valid.sum() == 0:
        epe = torch.tensor(0.0, device=disp_pred.device)

    metrics = {
        "train/epe": epe.mean(),
        "train/1px": (epe < 1).float().mean(),
        "train/3px": (epe < 3).float().mean(),
        "train/5px": (epe < 5).float().mean(),
    }

    # Simple L1 loss with outlier filtering
    l1_loss = torch.abs(disp_pred - disp_gt)
    valid = valid & ~torch.isnan(disp_pred.squeeze(-1))
    total_loss = l1_loss[valid].mean()
    return total_loss, metrics


class DPTFeatureExtractor(DPTHead):
    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ):
        B, S, _, H, W = images.shape

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ):
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        return out

    def scratch_forward(self, features: List[torch.Tensor]):
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_4_rn.shape[2:])
        up_path_4 = F.interpolate(
            path_4, size=layer_3_rn.shape[2:], mode="bilinear", align_corners=True
        )
        path_3 = self.scratch.refinenet3(
            up_path_4, layer_3_rn, size=layer_3_rn.shape[2:]
        )
        up_path_3 = F.interpolate(
            path_3, size=layer_2_rn.shape[2:], mode="bilinear", align_corners=True
        )
        path_2 = self.scratch.refinenet2(
            up_path_3, layer_2_rn, size=layer_2_rn.shape[2:]
        )
        up_path_2 = F.interpolate(
            path_2, size=layer_1_rn.shape[2:], mode="bilinear", align_corners=True
        )
        path_1 = self.scratch.refinenet1(
            up_path_2, layer_1_rn, size=layer_1_rn.shape[2:]
        )

        return path_1, path_2, path_3, path_4


class VGGTStereoRegression(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            patch_embed="dinov2_vitl14_reg",
        )
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=2,
            # activation="exp",
            # conf_activation="expp1",
            activation="relu",
            conf_activation="sigmoid",
        )

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        width = images.shape[-1]

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        aggregated_tokens_list = [
            tokens[:, :1, :, :] for tokens in aggregated_tokens_list
        ]
        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list,
                images=images[:, :1, :, :, :],
                patch_start_idx=patch_start_idx,
            )
            predictions["depth"] = depth * width
            predictions["depth_conf"] = depth_conf
        if not self.training:
            predictions["images"] = (
                images  # store the images for visualization during inference
            )
        return predictions

