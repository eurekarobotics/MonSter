"""Training script for MonsterV2 with DINOv3 encoder.

This script trains the hierarchical stereo matching model
with the following features:
- Hierarchical coarse-to-fine processing (1/16 → 1/8 → 1/4)
- Local cost volumes at 1/8 and 1/4 scales
- Single-layer ConvGRU per scale
- Mono disparity integration at each scale
- Optional REMP refinement
"""

import os
from torch.utils.data import dataset
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
from core.utils.utils import InputPadder
from core.monster import Monster, MonsterV2
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
import core.stereo_datasets as datasets
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from pathlib import Path


def gray_2_colormap_np(img, cmap="rainbow", max=None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img < 0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img / (max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap


def sequence_loss(
    disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192
):
    """Loss function defined over sequence of disparity predictions.
    
    For hierarchical model, disp_preds contains predictions from all scales.
    """
    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    # Initial disparity loss (from cost volume regression)
    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)
    disp_loss += 1.0 * F.smooth_l1_loss(
        disp_init_pred[init_valid], disp_gt[init_valid], reduction="mean"
    )
    
    # Sequence loss with exponential weighting
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1 + 1e-8))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [
            i_loss.shape,
            valid.shape,
            disp_gt.shape,
            disp_preds[i].shape,
        ]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        "train/epe": epe.mean(),
        "train/1px": (epe < 1).float().mean(),
        "train/3px": (epe < 3).float().mean(),
        "train/5px": (epe < 5).float().mean(),
    }
    return disp_loss, metrics


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler."""
    # Separate learning rates for decoder vs rest
    DPT_params = list(map(id, model.feat_decoder.parameters()))
    rest_params = filter(
        lambda x: id(x) not in DPT_params and x.requires_grad, model.parameters()
    )

    params_dict = [
        {"params": model.feat_decoder.parameters(), "lr": args.lr / 2.0},
        {"params": rest_params, "lr": args.lr},
    ]
    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        [args.lr / 2.0, args.lr],
        args.total_step + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    return optimizer, scheduler


@hydra.main(version_base=None, config_path="config", config_name="train_monsterv2_dinov3s")
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        log_with="tensorboard",
        kwargs_handlers=[kwargs],
        step_scheduler_with_optimizer=False,
        project_dir="./logs",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )
    accelerator.init_trackers(project_name=cfg.project_name)

    # Data loaders
    train_dataset = datasets.fetch_dataloader(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size // cfg.num_gpu,
        pin_memory=True,
        shuffle=True,
        num_workers=int(6),
        drop_last=True,
    )

    aug_params = {}
    val_dataset_1 = datasets.Middlebury(aug_params, split='MiddEval3', resolution='H')
    val_dataset_2 = datasets.EurekaV1Dataset(
        aug_params=aug_params,
        root="/home/duy/datasets/eureka-transparent/eureka",
        transparent_only=True,
    )
    val_dataset = val_dataset_1 + val_dataset_2
    booster_size = (608, 800)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(1),
        pin_memory=True,
        shuffle=False,
        num_workers=int(6),
        drop_last=False,
    )

    # Create model - use MonsterHierarchical if configured
    use_hierarchical = getattr(cfg, 'use_hierarchical', False)
    if use_hierarchical:
        print("=" * 60)
        print("Using MonsterV2 (hierarchical)")
        print(f"  scale_iters: {cfg.scale_iters}")
        print(f"  ndisps: {cfg.ndisps}")
        print(f"  disp_intervals: {cfg.disp_intervals}")
        print(f"  use_remp: {cfg.use_remp}")
        print("=" * 60)
        model = MonsterV2(cfg)
    else:
        print("Using original Monster model")
        model = Monster(cfg)

    # Print trainable parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Load checkpoint if specified
    if cfg.restore_ckpt is not None:
        assert cfg.restore_ckpt.endswith(".pth")
        print(f"Loading checkpoint from {cfg.restore_ckpt}")
        assert os.path.exists(cfg.restore_ckpt)
        checkpoint = torch.load(cfg.restore_ckpt, map_location="cpu", weights_only=False)
        ckpt = dict()
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
        for key in checkpoint:
            ckpt[key.replace("module.", "")] = checkpoint[key]
        model.load_state_dict(ckpt, strict=False)  # strict=False for architecture changes
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")
        del ckpt, checkpoint

    optimizer, lr_scheduler = fetch_optimizer(cfg, model)
    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(
        train_loader, model, optimizer, lr_scheduler, val_loader
    )
    model.to(accelerator.device)

    total_step = 0
    best_epe = float('inf')
    
    # Initial evaluation before training starts
    if cfg.restore_ckpt is not None:
        print("Evaluating loaded checkpoint before training...")
        torch.cuda.empty_cache()
        model.eval()
        elem_num, total_epe, total_out = 0, 0, 0
        for data in tqdm(
            val_loader,
            dynamic_ncols=True,
            disable=not accelerator.is_main_process,
            desc="Initial validation"
        ):
            _, left, right, disp_gt, valid = [x for x in data]
            width = left.shape[3]
            left = F.interpolate(left, size=booster_size, mode="bilinear", align_corners=False)
            right = F.interpolate(right, size=booster_size, mode="bilinear", align_corners=False)
            disp_gt = F.interpolate(disp_gt, size=booster_size, mode="nearest")
            disp_gt = disp_gt / width * booster_size[1]
            valid = F.interpolate(valid[None], size=booster_size, mode="nearest")
            valid = (disp_gt > 0) & (valid > 0.5)
            valid = valid[0]

            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)
            with torch.no_grad():
                disp_pred = model(left, right, test_mode=True)
            disp_pred = padder.unpad(disp_pred)
            
            epe = torch.abs(disp_pred - disp_gt)
            out = (epe > 2.0).float()
            epe = torch.squeeze(epe, dim=1)
            out = torch.squeeze(out, dim=1)
            epe, out = accelerator.gather_for_metrics(
                (epe[valid >= 0.5].mean(), out[valid >= 0.5].mean())
            )
            elem_num += 1
            total_epe += epe
            total_out += out
        
        init_epe = (total_epe / elem_num).mean()
        init_d1 = (100 * total_out / elem_num).mean()
        best_epe = init_epe
        print(f"Initial checkpoint EPE: {init_epe:.4f}, D1: {init_d1:.2f}%")
        accelerator.log({"val/epe": init_epe.item(), "val/d1": init_d1.item()}, total_step)
        torch.cuda.empty_cache()
    
    # Training loop
    should_keep_training = True
    while should_keep_training:
        model.train()
        accelerator.unwrap_model(model).freeze_bn()
        
        for data in tqdm(
            train_loader,
            dynamic_ncols=True,
            disable=not accelerator.is_main_process,
        ):
            _, left, right, disp_gt, valid = [x for x in data]
            
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    # Forward pass - hierarchical model ignores iters, uses scale_iters
                    disp_init_pred, disp_preds, depth_mono = model(
                        left, right, iters=cfg.train_iters
                    )
                
                loss, metrics = sequence_loss(
                    disp_preds, disp_init_pred, disp_gt, valid, max_disp=cfg.max_disp
                )
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                total_step += 1
                loss = accelerator.reduce(loss.detach(), reduction="mean")
                metrics = accelerator.reduce(metrics, reduction="mean")
                accelerator.log(
                    {
                        "train/loss": loss,
                        "train/learning_rate": optimizer.param_groups[-1]["lr"],
                    },
                    total_step,
                )
                accelerator.log(metrics, total_step)

            # Validation
            if (total_step > 0) and (total_step % cfg.val_frequency == 0):
                torch.cuda.empty_cache()
                model.eval()
                elem_num, total_epe, total_out = 0, 0, 0
                
                for data in tqdm(
                    val_loader,
                    dynamic_ncols=True,
                    disable=not accelerator.is_main_process,
                ):
                    _, left, right, disp_gt, valid = [x for x in data]
                    width = left.shape[3]
                    left = F.interpolate(left, size=booster_size, mode="bilinear", align_corners=False)
                    right = F.interpolate(right, size=booster_size, mode="bilinear", align_corners=False)
                    disp_gt = F.interpolate(disp_gt, size=booster_size, mode="nearest")
                    disp_gt = disp_gt / width * booster_size[1]
                    valid = F.interpolate(valid[None], size=booster_size, mode="nearest")
                    valid = (disp_gt > 0) & (valid > 0.5)
                    valid = valid[0]

                    padder = InputPadder(left.shape, divis_by=32)
                    left, right = padder.pad(left, right)
                    with torch.no_grad():
                        disp_pred = model(left, right, test_mode=True)
                    disp_pred = padder.unpad(disp_pred)
                    
                    epe = torch.abs(disp_pred - disp_gt)
                    out = (epe > 2.0).float()
                    epe = torch.squeeze(epe, dim=1)
                    out = torch.squeeze(out, dim=1)
                    epe, out = accelerator.gather_for_metrics(
                        (epe[valid >= 0.5].mean(), out[valid >= 0.5].mean())
                    )
                    elem_num += 1
                    total_epe += epe
                    total_out += out
                
                val_epe = (total_epe / elem_num).mean()
                val_d1 = (100 * total_out / elem_num).mean()
                accelerator.log({"val/epe": val_epe.item(), "val/d1": val_d1.item()}, total_step)

                if accelerator.is_main_process:
                    if val_epe < best_epe:
                        best_epe = val_epe
                        save_path = Path(cfg.save_path + "/best.pth")
                        model_save = accelerator.unwrap_model(model)
                        torch.save(model_save.state_dict(), save_path)
                        print(f"New best model saved with EPE: {best_epe:.4f} at step {total_step}")
                        del model_save

                model.train()
                accelerator.unwrap_model(model).freeze_bn()
                torch.cuda.empty_cache()

            if total_step % 100 == 0:
                torch.cuda.empty_cache()

            if total_step == cfg.total_step:
                should_keep_training = False
                break

    # Save final model
    if accelerator.is_main_process:
        save_path = Path(cfg.save_path + "/final.pth")
        model_save = accelerator.unwrap_model(model)
        torch.save(model_save.state_dict(), save_path)
        del model_save

    accelerator.end_training()


if __name__ == "__main__":
    main()
