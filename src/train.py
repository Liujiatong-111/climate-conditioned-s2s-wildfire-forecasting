"""Brief implementation note."""

import os
import argparse
import yaml
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from logger import TrainingLogger
from focal_loss import FocalLoss, CombinedLoss
from utils import (
    set_seed,
    compute_metrics,
    create_output_dir,
    get_model_name,
    AverageMeter,
)


# -------------------------
# helpers
# -------------------------

def _parse_lead_times_from_config(value):
    """Brief implementation note."""
    if value is None:
        return [1]
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(',') if p.strip()]
        return [int(p) for p in parts]
    raise TypeError(f"Unsupported lead_time_steps type: {type(value)}")


def _apply_ignore_mask_to_targets(y: torch.Tensor, mask: torch.Tensor, ignore_index: int = 2) -> torch.Tensor:
    """Brief implementation note."""
    mask_bool = mask.bool()

    if y.dim() == 3:
        # (B, H, W)
        return y.masked_fill(mask_bool, ignore_index)

    if y.dim() == 4:
        # (B, L, H, W)
        return y.masked_fill(mask_bool.unsqueeze(1), ignore_index)

    raise ValueError(f"Unsupported target dim: y.dim()={y.dim()} (expected 3 or 4)")


def _format_by_lead(metrics_by_lead: dict, lead_times: list, key: str) -> str:
    parts = []
    for lt in lead_times:
        v = metrics_by_lead.get(lt, {}).get(key, None)
        if v is None:
            parts.append(f"t{lt}=NA")
        else:
            parts.append(f"t{lt}={float(v):.4f}")
    return " ".join(parts)


def _mean_by_lead(metrics_by_lead: dict, lead_times: list, key: str) -> float:
    vals = []
    for lt in lead_times:
        if lt in metrics_by_lead and key in metrics_by_lead[lt]:
            vals.append(float(metrics_by_lead[lt][key]))
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


# -------------------------
# train / eval
# -------------------------

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, ignore_index: int = 2):
    """Brief implementation note."""
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for _, (x_local, x_global, x_oci, y, mask, _, _, _) in enumerate(pbar):
        
        x_local = x_local.to(device)  # (B, T, 14, 80, 80)
        x_global = x_global.to(device)  # (B, T, 14, 180, 360)
        x_oci = x_oci.to(device)  # (B, 10, 10)
        y = y.to(device)  
        mask = mask.to(device)  # (B, 80, 80)

        
        
        
        logits = model(x_local, x_global, x_oci)  

        
        if logits.dim() == 4:
            
            y_masked = _apply_ignore_mask_to_targets(y, mask, ignore_index=ignore_index)  # (B,H,W)
            loss = criterion(logits, y_masked)

        
        elif logits.dim() == 5:
            
            B, L, C, H, W = logits.shape

            
            if y.dim() == 3:
                y = y.unsqueeze(1).expand(B, L, y.size(-2), y.size(-1))

            
            if y.dim() != 4:
                raise ValueError(f"Multi-step expects y dim=4, got y.shape={tuple(y.shape)}")

            
            y_masked = _apply_ignore_mask_to_targets(y, mask, ignore_index=ignore_index)  # (B,L,H,W)

            
            
            
            #
            
            
            
            
            losses = []
            for l in range(L):
                loss_l = criterion(logits[:, l], y_masked[:, l])
                losses.append(loss_l)

            
            loss = torch.stack(losses).mean()

        else:
            raise ValueError(f"Unsupported logits dim: {logits.dim()} (shape={tuple(logits.shape)})")

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), x_local.size(0))
        pbar.set_postfix({'loss': loss_meter.avg})

    return loss_meter.avg


def evaluate_by_lead(model, data_loader, criterion, device, epoch, lead_times: list, ignore_index: int = 2, threshold: float = 0.5):
    """Brief implementation note."""
    model.eval()

    
    all_preds = {lt: [] for lt in lead_times}
    all_targets = {lt: [] for lt in lead_times}
    all_masks = {lt: [] for lt in lead_times}
    loss_meters = {lt: AverageMeter() for lt in lead_times}

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Val]" if isinstance(epoch, int) or str(epoch).isdigit() else f"{epoch}")
        for x_local, x_global, x_oci, y, mask, _, _, _ in pbar:
            x_local = x_local.to(device)
            x_global = x_global.to(device)
            x_oci = x_oci.to(device)
            y = y.to(device)
            mask = mask.to(device)

            logits = model(x_local, x_global, x_oci)

            
            if logits.dim() == 4:
                logits = logits.unsqueeze(1)

            if logits.dim() != 5:
                raise ValueError(f"Unsupported logits dim in eval: {logits.dim()} (shape={tuple(logits.shape)})")

            B, L, C, H, W = logits.shape

            
            if len(lead_times) != L:
                min_L = min(len(lead_times), L)
                if min_L != L:
                    logits = logits[:, :min_L]
                    L = min_L
                lead_times_eff = lead_times[:min_L]
            else:
                lead_times_eff = lead_times

            
            if y.dim() == 3:
                y = y.unsqueeze(1).expand(B, L, y.size(-2), y.size(-1))
            elif y.dim() == 4 and y.size(1) != L:
                min_L = min(y.size(1), L)
                y = y[:, :min_L]
                logits = logits[:, :min_L]
                lead_times_eff = lead_times_eff[:min_L]
                L = min_L

            if y.dim() != 4:
                raise ValueError(f"Unsupported y dim in eval: y.dim()={y.dim()}, y.shape={tuple(y.shape)}")

            # mask bool
            mask_bool = mask.bool()  # (B,H,W)

            
            y_masked = _apply_ignore_mask_to_targets(y, mask, ignore_index=ignore_index)  # (B,L,H,W)
            for l, lt in enumerate(lead_times_eff):
                loss_l = criterion(logits[:, l], y_masked[:, l])
                loss_meters[lt].update(loss_l.item(), B)

            
            probs = torch.softmax(logits, dim=2)[:, :, 1]  # (B,L,H,W)
            probs = probs.masked_fill(mask_bool.unsqueeze(1), 0.0)

            probs_np = probs.cpu().numpy()
            y_np = y.cpu().numpy()
            mask_np = mask.cpu().numpy()

            for l, lt in enumerate(lead_times_eff):
                all_preds[lt].append(probs_np[:, l])
                all_targets[lt].append(y_np[:, l])
                all_masks[lt].append(mask_np)

            
            
            mean_batch_loss = np.mean([loss_meters[lt].val for lt in lead_times_eff]) if len(lead_times_eff) > 0 else 0.0
            pbar.set_postfix({'loss': float(mean_batch_loss)})

    
    metrics_by_lead = {}
    for lt in lead_times:
        if len(all_preds[lt]) == 0:
            
            continue

        preds = np.concatenate(all_preds[lt], axis=0)
        targets = np.concatenate(all_targets[lt], axis=0)
        masks = np.concatenate(all_masks[lt], axis=0)

        metrics = compute_metrics(preds, targets, mask=masks, threshold=threshold)
        metrics['loss'] = loss_meters[lt].avg
        metrics_by_lead[lt] = metrics

    return metrics_by_lead


# -------------------------
# main
# -------------------------

def main(args):
    
    from dataset import SeasFirePatchDataset
    from model_multibranch_vit import create_model

    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    
    if args.lead_times is not None:
        config['data']['lead_time_steps'] = [int(x) for x in args.lead_times.split(',') if x.strip()]
        print(f"Status")

    lead_times = _parse_lead_times_from_config(config['data'].get('lead_time_steps', 1))

    
    if args.stride is not None:
        config['data']['stride'] = args.stride
        print(f"Status")

    if args.depth is not None:
        config['model']['depth'] = args.depth
        print(f"Status")

    if args.load_pretrained:
        config['model']['load_pretrained_backbone'] = True
        print(f"Status")

    if args.use_improved_head:
        config['model']['use_improved_head'] = True
        print(f"Status")

    if args.seg_head_type is not None:
        config['model']['seg_head_type'] = args.seg_head_type
        print(f"Status")

    if args.local_patch_size is not None:
        config['model']['local_patch_size'] = args.local_patch_size
        print(f"Status")

    
    config['model']['use_local'] = args.use_local
    config['model']['use_global'] = args.use_global
    config['model']['use_oci'] = args.use_oci
    print(f"Status")

    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
        print(f"Status")

    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
        print(f"Status")

    if args.lr is not None:
        config['train']['learning_rate'] = args.lr
        print(f"Status")

    if args.weight_decay is not None:
        config['train']['weight_decay'] = args.weight_decay
        print(f"Status")

    if args.optimizer is not None:
        config['train']['optimizer'] = args.optimizer
        print(f"Status")

    if args.lr_scheduler is not None:
        config['train']['lr_scheduler'] = args.lr_scheduler
        print(f"Status")

    if args.gpu_ids is not None:
        config['train']['gpu_ids'] = [int(x) for x in args.gpu_ids.split(',')]
        print(f"Status")

    print(f"Status")

    
    print("\n" + "="*80)
    print("Status")
    print("="*80)

    
    set_seed(config['train']['seed'])
    print(f"Status")

    
    
    

    
    output_dir = config['output']['save_dir']
    create_output_dir(output_dir)

    
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
        print(f"Status")
    else:
        
        depth = config['model']['depth']
        lead_times_str = '_'.join(map(str, lead_times))  
        stride = config['data'].get('stride', config['data']['patch_size'])  
        stride_str = f"_stride{stride}" if stride != config['data']['patch_size'] else ""
        checkpoint_dir = f'/data1/ljt/TeleVIT/muti-teleVit-transformer/teleVIT-0625-overlap-csjxl-codex/checkpoints/{depth}layers_lead{lead_times_str}{stride_str}'
        print(f"Status")
    create_output_dir(checkpoint_dir)

    
    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        
        lead_times_str = '_'.join(map(str, lead_times))
        stride = config['data'].get('stride', config['data']['patch_size'])
        stride_str = f"_stride{stride}" if stride != config['data']['patch_size'] else ""
        log_dir = f'/data1/ljt/TeleVIT/muti-teleVit-transformer/teleVIT-0625-overlap-csjxl-codex/logs/lead{lead_times_str}{stride_str}'
    create_output_dir(log_dir)
    print(f"Status")

    
    model_name = get_model_name(config['model']['use_local'], config['model']['use_global'], config['model']['use_oci'])
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    lead_times_str = '_'.join(map(str, lead_times))
    stride = config['data'].get('stride', config['data']['patch_size'])
    stride_str = f"_stride{stride}" if stride != config['data']['patch_size'] else ""
    experiment_name = f"{model_name}_lead{lead_times_str}{stride_str}_{timestamp}"
    print(f"Status")

    
    
    
    logger = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name, lead_times=lead_times)
    print(f"Status")

    
    gpu_ids = config['train'].get('gpu_ids', [])
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
        print(f"Status")

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')

    n_gpus = torch.cuda.device_count()
    print(f"Status")
    if n_gpus > 0:
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Status")

    
    print("\n" + "="*80)
    print("Status")
    print("="*80)

    
    
    
    

    lead_time_steps_cfg = config['data']['lead_time_steps']

    
    print(f"\n{'='*80}")
    print("Status")
    print(f"{'='*80}")
    print(f"Status")
    print(f"Status")

    temporal_steps = args.temporal_steps if args.temporal_steps is not None else config['data'].get('temporal_steps', 4)
    oci_window = args.oci_window if args.oci_window is not None else config['data']['oci_window']

    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")

    if temporal_steps != config['data'].get('temporal_steps', 4):
        print(f"Status")

    
    
    
    
    train_dataset = SeasFirePatchDataset(
        zarr_path=config['data']['zarr_path'],
        target_zarr_path=config['data']['target_zarr_path'],
        years=config['data']['train_years'],
        fire_vars=config['data']['fire_vars'],
        log_transform_vars=config['data']['log_transform_vars'],
        oci_vars=config['data']['oci_vars'],
        target_var=config['data']['target_var'],
        lead_time_steps=lead_time_steps_cfg,  
        oci_window=oci_window,
        temporal_steps=temporal_steps,  
        burn_threshold=config['data']['burn_threshold'],
        patch_size=config['data']['patch_size'],
        stride=config['data'].get('stride', None),  
        global_coarsen_factor=config['data']['global_coarsen_factor'],
        use_local=config['model']['use_local'],
        use_global=config['model']['use_global'],
        use_oci=config['model']['use_oci'],
        only_fire_patches=True,  
        use_augmentation=args.use_augmentation,  
    )

    
    
    
    
    
    

    print(f"Status")
    print(f"Status")

    
    val_size = int(len(train_dataset) * 0.15)
    print(f"Status")

    
    
    
    test_dataset = SeasFirePatchDataset(
        zarr_path=config['data']['zarr_path'],
        target_zarr_path=config['data']['target_zarr_path'],
        years=config['data']['test_years'],
        fire_vars=config['data']['fire_vars'],
        log_transform_vars=config['data']['log_transform_vars'],
        oci_vars=config['data']['oci_vars'],
        target_var=config['data']['target_var'],
        lead_time_steps=lead_time_steps_cfg,
        oci_window=oci_window,
        temporal_steps=temporal_steps,
        burn_threshold=config['data']['burn_threshold'],
        patch_size=config['data']['patch_size'],
        stride=config['data'].get('stride', None),  
        global_coarsen_factor=config['data']['global_coarsen_factor'],
        use_local=config['model']['use_local'],
        use_global=config['model']['use_global'],
        use_oci=config['model']['use_oci'],
        only_fire_patches=False,  
        use_augmentation=False,  
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
    )

    

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
    )

    
    print("\n" + "="*80)
    print("Status")
    print("="*80)

    
    #
    
    
    
    
    
    #
    
    
    
    
    #
    
    #    - Multi-Head Self-Attention (12 heads)
    #    - Feed-Forward Network (MLP)
    
    #
    
    
    #    - (B,768,T,10,10) → (B,768,10,10)
    #
    
    
    
    
    #
    
    
    
    
    

    model = create_model(config).to(device)

    
    if n_gpus > 1:
        print(f"Status")
        model = nn.DataParallel(model)
        effective_batch_size = config['train']['batch_size'] * n_gpus
        print(f"Status")
    else:
        print("Status")

    
    print("\n" + "="*50)
    print("Status")
    print("="*50)

    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=2)
        print("Status")

    elif args.loss_type == 'weighted_ce':
        class_weight = torch.tensor([1.0, args.fire_weight, 0.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=2)
        print("Status")
        print(f"Status")

    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=2)
        print(f"Status")

    elif args.loss_type == 'combined':
        criterion = CombinedLoss(
            focal_weight=0.7,
            dice_weight=0.3,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            ignore_index=2
        )
        print("Status")

    else:
        raise ValueError(f"Status")

    
    optimizer_type = config['train'].get('optimizer', 'adamw').lower()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=config['train']['weight_decay']
        )
        print(f"Status")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=config['train']['weight_decay']
        )
        print(f"Status")

    
    lr_scheduler_type = config['train'].get('lr_scheduler', 'cosine').lower()
    scheduler = None
    warmup_scheduler = None
    scheduler_monitor = None

    if lr_scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['train'].get('lr_scheduler_factor', 0.1),
            patience=config['train'].get('lr_scheduler_patience', 10),
            threshold=1e-4,
            cooldown=0,
            min_lr=0,
            verbose=True
        )
        scheduler_monitor = config['train'].get('lr_scheduler_monitor', 'train/loss')
        print(f"Status")

    elif lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train']['epochs']
        )
        print(f"Status")

    elif lr_scheduler_type == 'cosine_warmup':
        warmup_epochs = config['train'].get('warmup_epochs', 5)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['train'].get('cosine_T0', 20),
            T_mult=config['train'].get('cosine_T_mult', 2),
            eta_min=config['train'].get('cosine_eta_min', 1e-6)
        )
        print("Status")

    else:
        print("Status")

    
    print("\n" + "="*80)
    print("Status")
    print("="*80)

    
    print("Status")
    train_iter = iter(train_loader)
    x_local_sample, x_global_sample, x_oci_sample, y_sample, mask_sample, _, _, _ = next(train_iter)

    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")

    
    assert x_local_sample.shape[1] == temporal_steps, (
        f"Status"
        f"Status"
    )
    assert x_global_sample.shape[1] == temporal_steps, (
        f"Status"
        f"Status"
    )

    print(f"Status")
    print(f"Status")
    print(f"Status")

    
    if torch.isnan(x_local_sample).any():
        print(f"Status")
    if torch.isnan(x_global_sample).any():
        print(f"Status")

    
    for t in range(temporal_steps):
        if torch.all(x_local_sample[:, t] == 0):
            print(f"Status")
        if torch.all(x_global_sample[:, t] == 0):
            print(f"Status")

    
    print(f"Status")
    x_local_sample = x_local_sample.to(device)
    x_global_sample = x_global_sample.to(device)
    x_oci_sample = x_oci_sample.to(device)

    with torch.no_grad():
        logits_sample = model(x_local_sample, x_global_sample, x_oci_sample)

    print(f"Status")
    if logits_sample.dim() == 4:
        print(f"Status")
    elif logits_sample.dim() == 5:
        print(f"Status")
        assert logits_sample.shape[1] == len(lead_times), (
            f"Status"
        )

    print(f"Status")

    
    print("\n" + "="*50)
    print("Status")
    print("="*50)

    best_score = -1.0
    best_model_info = None  # (best_score, best_epoch, path)
    patience_counter = 0

    checkpoint_dir_best = os.path.join(checkpoint_dir, 'best_land')
    create_output_dir(checkpoint_dir_best)

    for epoch in range(1, config['train']['epochs'] + 1):
        
        
        val_indices = random.sample(range(len(train_dataset)), val_size)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)

        val_loader = DataLoader(
            val_subset,
            batch_size=config['train']['batch_size'],
            shuffle=False,
            num_workers=config['train']['num_workers'],
            pin_memory=True,
        )

        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, ignore_index=2)

        
        val_metrics_by_lead = evaluate_by_lead(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            lead_times=lead_times,
            ignore_index=2,
            threshold=0.5
        )

        
        val_auprc_mean = _mean_by_lead(val_metrics_by_lead, lead_times, 'auprc')
        val_auroc_mean = _mean_by_lead(val_metrics_by_lead, lead_times, 'auroc')
        val_loss_mean = _mean_by_lead(val_metrics_by_lead, lead_times, 'loss')

        print(f"\n{'='*80}")
        print(f"Status")
        print(f"{'='*80}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"\n  Validation Results (Land only):")
        print(f"  {'─'*76}")

        
        if len(lead_times) > 1:
            print(f"  {'Metric':<12} | " + " | ".join([f"t={lt:<3}" for lt in lead_times]) + f" | {'Mean':<8}")
            print(f"  {'─'*76}")

            # LOSS
            loss_str = f"  {'Loss':<12} | "
            for lt in lead_times:
                v = val_metrics_by_lead.get(lt, {}).get('loss', None)
                loss_str += f"{float(v):.4f}  | " if v is not None else "  N/A   | "
            loss_str += f"{val_loss_mean:.4f}"
            print(loss_str)

            # AUROC
            auroc_str = f"  {'AUROC':<12} | "
            for lt in lead_times:
                v = val_metrics_by_lead.get(lt, {}).get('auroc', None)
                auroc_str += f"{float(v):.4f}  | " if v is not None else "  N/A   | "
            auroc_str += f"{val_auroc_mean:.4f}"
            print(auroc_str)

            # AUPRC
            auprc_str = f"  {'AUPRC':<12} | "
            for lt in lead_times:
                v = val_metrics_by_lead.get(lt, {}).get('auprc', None)
                auprc_str += f"{float(v):.4f}  | " if v is not None else "  N/A   | "
            auprc_str += f"{val_auprc_mean:.4f}"
            print(auprc_str)

            print(f"  {'─'*76}")
        else:
            
            print(f"    Loss:  {val_loss_mean:.4f}")
            print(f"    AUROC: {val_auroc_mean:.4f}")
            print(f"    AUPRC: {val_auprc_mean:.4f}")

        
        current_lr = optimizer.param_groups[0]['lr']

        
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_metrics_by_lead=val_metrics_by_lead,
            learning_rate=current_lr
        )

        # lr scheduler
        if scheduler is not None:
            if lr_scheduler_type == 'plateau':
                if scheduler_monitor == 'train/loss':
                    scheduler.step(train_loss)
                elif scheduler_monitor == 'val/loss':
                    scheduler.step(val_loss_mean)
                elif scheduler_monitor == 'val/auprc':
                    
                    scheduler.step(-val_auprc_mean)
                else:
                    scheduler.step(train_loss)

            elif lr_scheduler_type == 'cosine_warmup':
                warmup_epochs = config['train'].get('warmup_epochs', 5)
                if epoch <= warmup_epochs and warmup_scheduler is not None:
                    warmup_scheduler.step()
                else:
                    scheduler.step()
            else:
                scheduler.step()

        
        current_score = val_auprc_mean
        if current_score > best_score:
            
            if best_model_info is not None:
                _, old_epoch, old_path = best_model_info
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"Status")

            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            checkpoint_path = os.path.join(checkpoint_dir_best, f'best_model_{model_name}_epoch{epoch}.pth')
            torch.save(model_to_save.state_dict(), checkpoint_path)

            best_score = current_score
            best_model_info = (best_score, epoch, checkpoint_path)
            patience_counter = 0
            print(f"Status")
        else:
            patience_counter += 1
            print(f"Status")

        if patience_counter >= config['train']['patience']:
            print(f"Status")
            break

        print("-" * 50)

    
    print("\n" + "="*50)
    print("Status")
    print("="*50)

    if best_model_info is None:
        print("Status")
        return

    best_score, best_epoch, best_model_path = best_model_info
    print(f"Status")
    print(f"Status")

    state_dict = torch.load(best_model_path, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    test_metrics_by_lead = evaluate_by_lead(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        epoch="Test",
        lead_times=lead_times,
        ignore_index=2,
        threshold=0.5
    )

    test_auprc_mean = _mean_by_lead(test_metrics_by_lead, lead_times, 'auprc')
    test_auroc_mean = _mean_by_lead(test_metrics_by_lead, lead_times, 'auroc')
    test_loss_mean = _mean_by_lead(test_metrics_by_lead, lead_times, 'loss')

    print(f"\n{'='*80}")
    print("FINAL TEST RESULTS (Land only)")
    print(f"{'='*80}")

    
    if len(lead_times) > 1:
        print(f"  {'Metric':<12} | " + " | ".join([f"t={lt:<3}" for lt in lead_times]) + f" | {'Mean':<8}")
        print(f"  {'─'*76}")

        # LOSS
        loss_str = f"  {'Loss':<12} | "
        for lt in lead_times:
            v = test_metrics_by_lead.get(lt, {}).get('loss', None)
            loss_str += f"{float(v):.4f}  | " if v is not None else "  N/A   | "
        loss_str += f"{test_loss_mean:.4f}"
        print(loss_str)

        # AUROC
        auroc_str = f"  {'AUROC':<12} | "
        for lt in lead_times:
            v = test_metrics_by_lead.get(lt, {}).get('auroc', None)
            auroc_str += f"{float(v):.4f}  | " if v is not None else "  N/A   | "
        auroc_str += f"{test_auroc_mean:.4f}"
        print(auroc_str)

        # AUPRC
        auprc_str = f"  {'AUPRC':<12} | "
        for lt in lead_times:
            v = test_metrics_by_lead.get(lt, {}).get('auprc', None)
            auprc_str += f"{float(v):.4f}  | " if v is not None else "  N/A   | "
        auprc_str += f"{test_auprc_mean:.4f}"
        print(auprc_str)

        # Precision
        prec_str = f"  {'Precision':<12} | "
        for lt in lead_times:
            v = test_metrics_by_lead.get(lt, {}).get('precision', None)
            prec_str += f"{float(v):.4f}  | " if v is not None else "  N/A   | "
        prec_mean = _mean_by_lead(test_metrics_by_lead, lead_times, 'precision')
        prec_str += f"{prec_mean:.4f}"
        print(prec_str)

        # Recall
        recall_str = f"  {'Recall':<12} | "
        for lt in lead_times:
            v = test_metrics_by_lead.get(lt, {}).get('recall', None)
            recall_str += f"{float(v):.4f}  | " if v is not None else "  N/A   | "
        recall_mean = _mean_by_lead(test_metrics_by_lead, lead_times, 'recall')
        recall_str += f"{recall_mean:.4f}"
        print(recall_str)

        # F1
        f1_str = f"  {'F1':<12} | "
        for lt in lead_times:
            v = test_metrics_by_lead.get(lt, {}).get('f1', None)
            f1_str += f"{float(v):.4f}  | " if v is not None else "  N/A   | "
        f1_mean = _mean_by_lead(test_metrics_by_lead, lead_times, 'f1')
        f1_str += f"{f1_mean:.4f}"
        print(f1_str)

        print(f"  {'─'*76}")
    else:
        
        lt = lead_times[0]
        metrics = test_metrics_by_lead.get(lt, {})
        print(f"  Loss:      {metrics.get('loss', 0.0):.4f}")
        print(f"  AUROC:     {metrics.get('auroc', 0.0):.4f}")
        print(f"  AUPRC:     {metrics.get('auprc', 0.0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0.0):.4f}")
        print(f"  F1:        {metrics.get('f1', 0.0):.4f}")

    print(f"{'='*80}")

    logger.log_final_test(test_metrics_by_lead=test_metrics_by_lead, best_epoch=best_epoch)

    print("Status")
    print(f"Status")
    print(f"Status")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='/data1/ljt/TeleVIT/muti-teleVit-transformer/teleVIT-0625-overlap-csjxl-codex/config.yaml',
                        help="Status")

    
    parser.add_argument('--lead-times', type=str, default=None,
                        help="Status")

    
    parser.add_argument('--depth', type=int, default=12, help="Status")
    parser.add_argument('--load-pretrained', action='store_true', default=False, help="Status")
    parser.add_argument('--use-improved-head', action='store_true', default=True, help="Status")
    parser.add_argument('--seg-head-type', type=str, default='patchwise',
                        choices=['patchwise', 'improved', 'aspp', 'progressive'],
                        help="Status")
    parser.add_argument('--local-patch-size', type=int, default=8, help='Local patch size (8 or 16)')

    
    parser.add_argument('--use-local', action='store_true', default=True, help="Status")
    parser.add_argument('--use-global', action='store_true', default=True, help="Status")
    parser.add_argument('--use-oci', action='store_true', default=True, help="Status")

    
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help="Status")
    parser.add_argument('--lr', type=float, default=1e-4, help="Status")
    parser.add_argument('--weight-decay', type=float, default=1e-6, help="Status")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help="Status")
    parser.add_argument('--lr-scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'cosine_warmup'], help="Status")

    
    parser.add_argument('--loss-type', type=str, default='combined',
                        choices=['ce', 'weighted_ce', 'focal', 'combined'],
                        help="Status")
    parser.add_argument('--fire-weight', type=float, default=4, help="Status")
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal gamma')

    
    parser.add_argument('--use-augmentation', action='store_true', default=False, help="Status")

    
    parser.add_argument('--temporal-steps', type=int, default=None,
                        help="Status")
    parser.add_argument('--oci-window', type=int, default=None,
                        help="Status")

    
    parser.add_argument('--stride', type=int, default=60,
                        help="Status")

    # GPU
    parser.add_argument('--gpu-ids', type=str, default='0,1,3,4', help="Status")

    
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help="Status")
    parser.add_argument('--log-dir', type=str, default=None,
                        help="Status")

    args = parser.parse_args()

    main(args)
