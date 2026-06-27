"""Brief implementation note."""
import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from typing import List, Tuple, Optional
from sklearn.metrics import brier_score_loss, log_loss


def create_gaussian_weight(patch_size: int, sigma: float = None) -> np.ndarray:
    """Brief implementation note."""
    if sigma is None:
        sigma = patch_size / 4

    center = patch_size / 2
    x = np.arange(patch_size)
    y = np.arange(patch_size)
    xx, yy = np.meshgrid(x, y)

    
    dist = np.sqrt((xx - center + 0.5)**2 + (yy - center + 0.5)**2)

    
    weight = np.exp(-(dist**2) / (2 * sigma**2))

    return weight.astype(np.float32)


def compute_ece(probs, targets, n_bins=10):
    """Brief implementation note."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            
            confidence_in_bin = probs[in_bin].mean()
            
            accuracy_in_bin = targets[in_bin].mean()
            
            calibration_error = abs(confidence_in_bin - accuracy_in_bin)

            
            ece += prop_in_bin * calibration_error
            
            mce = max(mce, calibration_error)

    return ece, mce


def compute_calibration_metrics(probs, targets, mask=None, n_bins=10):
    """Brief implementation note."""
    
    if probs.ndim > 1:
        probs = probs.flatten()
        targets = targets.flatten()
        if mask is not None:
            mask = mask.flatten()

    
    if mask is not None:
        valid_mask = (mask == 0)
        probs = probs[valid_mask]
        targets = targets[valid_mask]

    
    targets = targets.astype(np.int64)

    # 1. Brier Score
    brier = brier_score_loss(targets, probs)

    
    probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    logloss = log_loss(targets, probs_clipped)

    
    ece, mce = compute_ece(probs, targets, n_bins=n_bins)

    return {
        'brier_score': float(brier),
        'ece': float(ece),
        'log_loss': float(logloss),
        'mce': float(mce)
    }


def prepare_valid_prob_targets(probs, targets, mask=None):
    """Brief implementation note."""
    if probs.ndim > 1:
        probs = probs.flatten()
        targets = targets.flatten()
        if mask is not None:
            mask = mask.flatten()

    if mask is not None:
        valid_mask = (mask == 0)
        probs = probs[valid_mask]
        targets = targets[valid_mask]

    return probs.astype(np.float64), targets.astype(np.int64)


def compute_climatology_reference(targets):
    """Brief implementation note."""
    if targets.size == 0:
        return np.nan, np.nan

    climatology = float(targets.mean())
    brier_ref = float(climatology * (1.0 - climatology))
    return climatology, brier_ref


def sliding_window_inference(
    model,
    ds,
    ds_target,
    time_idx: int,
    fire_vars: List[str],
    oci_vars: List[str],
    target_var: str,
    mean_std_dict: dict,
    coord_grid_local: np.ndarray,
    coord_grid_global: Optional[np.ndarray],
    ds_global: Optional[xr.Dataset],
    temporal_steps: int,
    oci_window: int,
    lead_time_steps: List[int],
    patch_size: int = 80,
    stride: int = 40,
    batch_size: int = 16,
    device: str = 'cuda',
    use_local: bool = True,
    use_global: bool = True,
    use_oci: bool = True,
    use_gaussian_weight: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brief implementation note."""
    model.eval()

    H, W = ds['latitude'].shape[0], ds['longitude'].shape[0]
    num_leads = len(lead_time_steps)

    
    h_starts = list(range(0, H - patch_size, stride))
    if len(h_starts) == 0 or h_starts[-1] != H - patch_size:
        h_starts.append(H - patch_size)

    w_starts = list(range(0, W - patch_size, stride))
    if len(w_starts) == 0 or w_starts[-1] != W - patch_size:
        w_starts.append(W - patch_size)

    print(f"Status")
    print(f"Status")
    print(f"Status")

    
    if use_gaussian_weight:
        gaussian_weight = create_gaussian_weight(patch_size)
        print(f"Status")
    else:
        gaussian_weight = np.ones((patch_size, patch_size), dtype=np.float32)

    
    output_sum = np.zeros((num_leads, 3, H, W), dtype=np.float32)  
    output_count = np.zeros((H, W), dtype=np.float32)  

    
    patches_data = []
    patches_positions = []

    with torch.no_grad():
        for i in tqdm(h_starts, desc="Status", leave=False):
            for j in w_starts:
                
                i1 = i + patch_size
                j1 = j + patch_size

                # ========== Local Input ==========
                if use_local:
                    local_temporal = []
                    for t in range(time_idx - temporal_steps + 1, time_idx + 1):
                        local_data = []
                        for var in fire_vars:
                            data = ds[var].isel(time=t).values[i:i1, j:j1]
                            data = (data - mean_std_dict[f'{var}_mean']) / mean_std_dict[f'{var}_std']
                            local_data.append(data)
                        local_data = np.stack(local_data, axis=0)  # (10, 80, 80)
                        local_temporal.append(local_data)

                    local_temporal = np.stack(local_temporal, axis=1)  # (10, T, 80, 80)

                    
                    coord_patch = coord_grid_local[:, i:i1, j:j1]  # (4, 80, 80)
                    coord_patch = np.expand_dims(coord_patch, axis=1)  # (4, 1, 80, 80)
                    coord_patch = np.repeat(coord_patch, temporal_steps, axis=1)  # (4, T, 80, 80)

                    x_local = np.concatenate([local_temporal, coord_patch], axis=0)  # (14, T, 80, 80)
                    x_local = np.transpose(x_local, (1, 0, 2, 3))  # (T, 14, 80, 80)
                    x_local = np.nan_to_num(x_local, nan=0.0)
                else:
                    x_local = np.zeros((temporal_steps, 14, patch_size, patch_size), dtype=np.float32)

                # ========== Global Input ==========
                if use_global and ds_global is not None:
                    global_temporal = []
                    for t in range(time_idx - temporal_steps + 1, time_idx + 1):
                        global_data = []
                        for var in fire_vars:
                            data = ds_global[var].isel(time=t).values
                            global_data.append(data)
                        global_data = np.stack(global_data, axis=0)  # (10, 180, 360)
                        global_temporal.append(global_data)

                    global_temporal = np.stack(global_temporal, axis=1)  # (10, T, 180, 360)

                    coord_global = np.expand_dims(coord_grid_global, axis=1)  # (4, 1, 180, 360)
                    coord_global = np.repeat(coord_global, temporal_steps, axis=1)  # (4, T, 180, 360)

                    x_global = np.concatenate([global_temporal, coord_global], axis=0)  # (14, T, 180, 360)
                    x_global = np.transpose(x_global, (1, 0, 2, 3))  # (T, 14, 180, 360)
                    x_global = np.nan_to_num(x_global, nan=0.0)
                else:
                    x_global = np.zeros((temporal_steps, 14, 180, 360), dtype=np.float32)

                # ========== OCI Input ==========
                if use_oci:
                    oci_data = []
                    for var in oci_vars:
                        window_data = ds[var].isel(
                            time=slice(time_idx - oci_window + 1, time_idx + 1)
                        ).values

                        if window_data.ndim > 1:
                            window_data = window_data[:, 0, 0]

                        window_data = (window_data - mean_std_dict[f'{var}_mean']) / \
                                      (mean_std_dict[f'{var}_std'] + 1e-8)
                        oci_data.append(window_data)

                    x_oci = np.stack(oci_data, axis=0)  # (10, oci_window)
                    x_oci = np.nan_to_num(x_oci, nan=0.0)
                else:
                    x_oci = np.zeros((10, oci_window), dtype=np.float32)

                
                patches_data.append((x_local, x_global, x_oci))
                patches_positions.append((i, j))

                
                if len(patches_data) == batch_size or (i == h_starts[-1] and j == w_starts[-1]):
                    
                    batch_local = torch.from_numpy(np.stack([p[0] for p in patches_data])).to(device)
                    batch_global = torch.from_numpy(np.stack([p[1] for p in patches_data])).to(device)
                    batch_oci = torch.from_numpy(np.stack([p[2] for p in patches_data])).to(device)

                    
                    logits = model(batch_local, batch_global, batch_oci)  # (B, L, 3, 80, 80) or (B, 3, 80, 80)

                    
                    if logits.dim() == 4:
                        logits = logits.unsqueeze(1)  # (B, 1, 3, 80, 80)

                    
                    probs = torch.softmax(logits, dim=2)  # (B, L, 3, 80, 80)
                    probs = probs.cpu().numpy()

                    
                    for b, (pi, pj) in enumerate(patches_positions):
                        pi1 = pi + patch_size
                        pj1 = pj + patch_size

                        
                        ndvi_data = ds['ndvi'].isel(time=time_idx).values[pi:pi1, pj:pj1]
                        mask_patch = (~np.isnan(ndvi_data)).astype(np.float32)  

                        
                        weight_patch = gaussian_weight * mask_patch

                        
                        for l in range(num_leads):
                            output_sum[l, :, pi:pi1, pj:pj1] += probs[b, l] * weight_patch
                        output_count[pi:pi1, pj:pj1] += weight_patch

                    
                    patches_data = []
                    patches_positions = []

    
    output_count_safe = np.where(output_count == 0, 1, output_count)  
    output_probs = output_sum / output_count_safe[np.newaxis, np.newaxis, :, :]  # (L, 3, H, W)

    
    output_probs[:, :, output_count == 0] = 0

    
    fire_probs = output_probs[:, 1, :, :]  # (L, H, W)

    
    target_list = []
    for lead_t in lead_time_steps:
        target_time_idx = time_idx + lead_t
        target_data = ds_target[target_var].isel(time=target_time_idx).values
        target_data = np.nan_to_num(target_data, nan=0.0)
        target_binary = np.where(target_data > 0.0, 1, 0).astype(np.int64)
        target_list.append(target_binary)

    target = np.stack(target_list, axis=0)  # (L, H, W)

    
    ndvi_data = ds['ndvi'].isel(time=time_idx).values
    ndvi_mask = np.isnan(ndvi_data).astype(np.float32)  

    return fire_probs, target, ndvi_mask


def main(args):
    from dataset import SeasFirePatchDataset
    from model_multibranch_vit import create_model
    from utils import compute_metrics

    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.lead_times is not None:
        config['data']['lead_time_steps'] = [int(x) for x in args.lead_times.split(',') if x.strip()]
        print(f"Status")

    
    if isinstance(config['data']['lead_time_steps'], int):
        lead_times = [config['data']['lead_time_steps']]
    else:
        lead_times = list(config['data']['lead_time_steps'])

    print(f"\n{'='*80}")
    print("Status")
    print(f"{'='*80}")
    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Status")

    
    print(f"\n{'='*80}")
    print("Status")
    print(f"{'='*80}")
    model = create_model(config).to(device)

    
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Status")

    
    print(f"\n{'='*80}")
    print("Status")
    print(f"{'='*80}")

    
    temp_dataset = SeasFirePatchDataset(
        zarr_path=config['data']['zarr_path'],
        target_zarr_path=config['data']['target_zarr_path'],
        years=config['data']['test_years'],
        fire_vars=config['data']['fire_vars'],
        log_transform_vars=config['data']['log_transform_vars'],
        oci_vars=config['data']['oci_vars'],
        target_var=config['data']['target_var'],
        lead_time_steps=lead_times,
        oci_window=config['data']['oci_window'],
        temporal_steps=config['data'].get('temporal_steps', 4),
        burn_threshold=config['data']['burn_threshold'],
        patch_size=config['data']['patch_size'],
        stride=args.stride,
        global_coarsen_factor=config['data']['global_coarsen_factor'],
        use_local=config['model']['use_local'],
        use_global=config['model']['use_global'],
        use_oci=config['model']['use_oci'],
        only_fire_patches=False,
        use_augmentation=False,
    )

    
    mean_std_dict = temp_dataset.mean_std_dict
    coord_grid_local = temp_dataset.coord_grid_local
    coord_grid_global = temp_dataset.coord_grid_global
    ds = temp_dataset.ds
    ds_target = temp_dataset.ds_target
    ds_global = temp_dataset.ds_global if config['model']['use_global'] else None

    
    os.makedirs(args.output_dir, exist_ok=True)

    
    print(f"\n{'='*80}")
    print("Status")
    print(f"{'='*80}")

    
    time_years = ds['time'].dt.year.values
    test_years = config['data']['test_years']
    valid_mask = np.isin(time_years, test_years)
    valid_times = np.where(valid_mask)[0]

    
    temporal_steps = config['data'].get('temporal_steps', 4)
    oci_window = config['data']['oci_window']
    max_lead_time = max(lead_times)

    
    min_history_steps = max(temporal_steps - 1, oci_window)

    valid_times = valid_times[
        (valid_times >= min_history_steps) &
        (valid_times < len(ds['time']) - max_lead_time)
    ]

    print(f"Status")
    print(f"Status")
    print(f"  - oci_window: {oci_window}")
    print(f"  - max_lead_time: {max_lead_time}")
    print(f"  - min_history_steps: {min_history_steps}")
    print(f"Status")
    print(f"Status")

    
    if args.num_samples > 0:
        sample_indices = np.linspace(0, len(valid_times) - 1, args.num_samples, dtype=int)
        valid_times = valid_times[sample_indices]
        print(f"Status")

    
    all_metrics = {lt: [] for lt in lead_times}
    all_predictions = []  
    all_targets = []  
    all_masks = []  
    time_indices = []  

    for idx, time_idx in enumerate(tqdm(valid_times, desc="Status")):
        
        fire_probs, target, ndvi_mask = sliding_window_inference(
            model=model,
            ds=ds,
            ds_target=ds_target,
            time_idx=time_idx,
            fire_vars=config['data']['fire_vars'],
            oci_vars=config['data']['oci_vars'],
            target_var=config['data']['target_var'],
            mean_std_dict=mean_std_dict,
            coord_grid_local=coord_grid_local,
            coord_grid_global=coord_grid_global,
            ds_global=ds_global,
            temporal_steps=config['data'].get('temporal_steps', 4),
            oci_window=oci_window,
            lead_time_steps=lead_times,
            patch_size=config['data']['patch_size'],
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
            use_local=config['model']['use_local'],
            use_global=config['model']['use_global'],
            use_oci=config['model']['use_oci'],
            use_gaussian_weight=args.use_gaussian_weight,
        )

        
        all_predictions.append(fire_probs)  # (L, H, W)
        all_targets.append(target)  # (L, H, W)
        all_masks.append(ndvi_mask)  # (H, W)
        time_indices.append(time_idx)

        
        metrics_dict = {'time_idx': time_idx}
        for l, lt in enumerate(lead_times):
            
            metrics = compute_metrics(
                fire_probs[l:l+1],  # (1, H, W)
                target[l:l+1],  # (1, H, W)
                mask=ndvi_mask[np.newaxis, :, :],  # (1, H, W)
                threshold=0.5
            )

            
            calib_metrics = compute_calibration_metrics(
                fire_probs[l:l+1],  # (1, H, W)
                target[l:l+1],  # (1, H, W)
                mask=ndvi_mask[np.newaxis, :, :],  # (1, H, W)
                n_bins=10
            )

            
            metrics.update(calib_metrics)
            all_metrics[lt].append(metrics)

            
            metrics_dict[f'lead{lt}_auprc'] = metrics['auprc']
            metrics_dict[f'lead{lt}_auroc'] = metrics['auroc']
            metrics_dict[f'lead{lt}_f1'] = metrics['f1']
            metrics_dict[f'lead{lt}_precision'] = metrics['precision']
            metrics_dict[f'lead{lt}_recall'] = metrics['recall']
            metrics_dict[f'lead{lt}_brier_score'] = metrics['brier_score']
            metrics_dict[f'lead{lt}_ece'] = metrics['ece']
            metrics_dict[f'lead{lt}_log_loss'] = metrics['log_loss']
            metrics_dict[f'lead{lt}_mce'] = metrics['mce']

        
        print(f"Status")
        for lt in lead_times:
            m = all_metrics[lt][-1]
            print(f"  Lead={lt}: AUPRC={m['auprc']:.4f}, AUROC={m['auroc']:.4f}, F1={m['f1']:.4f}, Brier={m['brier_score']:.4f}, ECE={m['ece']:.4f}")

    
    
    print(f"\n{'='*80}")
    print("Status")
    print(f"{'='*80}")

    
    all_predictions = np.array(all_predictions)  # (N, L, H, W)
    all_targets = np.array(all_targets)  # (N, L, H, W)
    all_masks = np.array(all_masks)  # (N, H, W)
    time_indices = np.array(time_indices)  # (N,)

    
    climatology_stats = {}
    for l, lt in enumerate(lead_times):
        probs_valid, targets_valid = prepare_valid_prob_targets(
            all_predictions[:, l, :, :],
            all_targets[:, l, :, :],
            mask=all_masks
        )
        climatology, brier_ref = compute_climatology_reference(targets_valid)
        if np.isfinite(brier_ref) and brier_ref > 0:
            brier_overall = float(brier_score_loss(targets_valid, probs_valid))
            bss_overall = float(1.0 - brier_overall / brier_ref)
        else:
            brier_overall = np.nan
            bss_overall = np.nan

        climatology_stats[lt] = {
            'climatology': climatology,
            'brier_ref': brier_ref,
            'brier_overall': brier_overall,
            'bss_overall': bss_overall,
        }

    
    predictions_dir = os.path.join(args.output_dir, 'predictions')
    ground_truth_dir = os.path.join(args.output_dir, 'ground_truth')
    masks_dir = os.path.join(args.output_dir, 'masks')

    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    
    if all_predictions.ndim == 4 and all_predictions.shape[1] == 1:
        all_predictions = all_predictions[:, 0, :, :]
        all_targets = all_targets[:, 0, :, :]
        print(f"Status")
    elif all_predictions.ndim == 4:
        print(f"Status")

    
    print(f"Status")
    for idx, time_idx in enumerate(time_indices):
        
        time_value = ds['time'].isel(time=int(time_idx)).values
        date_str = pd.Timestamp(time_value).strftime('%Y%m%d')

        
        pred_path = os.path.join(predictions_dir, f'prediction_{date_str}.npy')
        np.save(pred_path, all_predictions[idx])

        
        target_path = os.path.join(ground_truth_dir, f'ground_truth_{date_str}.npy')
        np.save(target_path, all_targets[idx])

        
        mask_path = os.path.join(masks_dir, f'mask_{date_str}.npy')
        np.save(mask_path, all_masks[idx])

    
    time_indices_path = os.path.join(args.output_dir, 'time_indices.npy')
    np.save(time_indices_path, time_indices)

    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")

    
    
    print(f"\n{'='*80}")
    print("Status")
    print(f"{'='*80}")

    
    per_day_metrics = []
    for idx, time_idx in enumerate(time_indices):
        row = {'time_idx': time_idx}
        for lt in lead_times:
            m = all_metrics[lt][idx]
            brier_ref = climatology_stats[lt]['brier_ref']
            if np.isfinite(brier_ref) and brier_ref > 0:
                bss = 1.0 - m['brier_score'] / brier_ref
            else:
                bss = np.nan
            
            row[f'lead{lt}_auprc'] = m['auprc']
            row[f'lead{lt}_auroc'] = m['auroc']
            row[f'lead{lt}_f1'] = m['f1']
            row[f'lead{lt}_precision'] = m['precision']
            row[f'lead{lt}_recall'] = m['recall']
            
            row[f'lead{lt}_climatology'] = climatology_stats[lt]['climatology']
            row[f'lead{lt}_brier_ref'] = brier_ref
            row[f'lead{lt}_brier_score'] = m['brier_score']
            row[f'lead{lt}_bss'] = bss
            row[f'lead{lt}_ece'] = m['ece']
            row[f'lead{lt}_log_loss'] = m['log_loss']
            row[f'lead{lt}_mce'] = m['mce']
        per_day_metrics.append(row)

    df_per_day = pd.DataFrame(per_day_metrics)
    per_day_csv_path = os.path.join(args.output_dir, 'metrics_per_day.csv')
    df_per_day.to_csv(per_day_csv_path, index=False, float_format='%.6f')
    print(f"Status")

    
    
    avg_metrics = []
    for lt in lead_times:
        metrics_list = all_metrics[lt]
        brier_ref = climatology_stats[lt]['brier_ref']
        daily_bss = []
        if np.isfinite(brier_ref) and brier_ref > 0:
            daily_bss = [1.0 - m['brier_score'] / brier_ref for m in metrics_list]
        row = {
            'lead_time': lt,
            
            'auprc_mean': np.mean([m['auprc'] for m in metrics_list]),
            'auprc_std': np.std([m['auprc'] for m in metrics_list]),
            'auroc_mean': np.mean([m['auroc'] for m in metrics_list]),
            'auroc_std': np.std([m['auroc'] for m in metrics_list]),
            'f1_mean': np.mean([m['f1'] for m in metrics_list]),
            'f1_std': np.std([m['f1'] for m in metrics_list]),
            'precision_mean': np.mean([m['precision'] for m in metrics_list]),
            'precision_std': np.std([m['precision'] for m in metrics_list]),
            'recall_mean': np.mean([m['recall'] for m in metrics_list]),
            'recall_std': np.std([m['recall'] for m in metrics_list]),
            
            'climatology': climatology_stats[lt]['climatology'],
            'brier_ref': brier_ref,
            'brier_score_mean': np.mean([m['brier_score'] for m in metrics_list]),
            'brier_score_std': np.std([m['brier_score'] for m in metrics_list]),
            'brier_score_overall': climatology_stats[lt]['brier_overall'],
            'bss_mean': np.mean(daily_bss) if daily_bss else np.nan,
            'bss_std': np.std(daily_bss) if daily_bss else np.nan,
            'bss_overall': climatology_stats[lt]['bss_overall'],
            'ece_mean': np.mean([m['ece'] for m in metrics_list]),
            'ece_std': np.std([m['ece'] for m in metrics_list]),
            'log_loss_mean': np.mean([m['log_loss'] for m in metrics_list]),
            'log_loss_std': np.std([m['log_loss'] for m in metrics_list]),
            'mce_mean': np.mean([m['mce'] for m in metrics_list]),
            'mce_std': np.std([m['mce'] for m in metrics_list]),
        }
        avg_metrics.append(row)

    df_avg = pd.DataFrame(avg_metrics)
    avg_csv_path = os.path.join(args.output_dir, 'metrics_average.csv')
    df_avg.to_csv(avg_csv_path, index=False, float_format='%.6f')
    print(f"Status")

    
    
    print(f"\n{'='*80}")
    print("Status")
    print(f"{'='*80}")

    for lt in lead_times:
        metrics_list = all_metrics[lt]
        
        avg_auprc = np.mean([m['auprc'] for m in metrics_list])
        std_auprc = np.std([m['auprc'] for m in metrics_list])
        avg_auroc = np.mean([m['auroc'] for m in metrics_list])
        std_auroc = np.std([m['auroc'] for m in metrics_list])
        avg_f1 = np.mean([m['f1'] for m in metrics_list])
        std_f1 = np.std([m['f1'] for m in metrics_list])
        avg_precision = np.mean([m['precision'] for m in metrics_list])
        std_precision = np.std([m['precision'] for m in metrics_list])
        avg_recall = np.mean([m['recall'] for m in metrics_list])
        std_recall = np.std([m['recall'] for m in metrics_list])
        
        avg_brier = np.mean([m['brier_score'] for m in metrics_list])
        std_brier = np.std([m['brier_score'] for m in metrics_list])
        brier_ref = climatology_stats[lt]['brier_ref']
        avg_bss = 1.0 - avg_brier / brier_ref if np.isfinite(brier_ref) and brier_ref > 0 else np.nan
        overall_bss = climatology_stats[lt]['bss_overall']
        avg_ece = np.mean([m['ece'] for m in metrics_list])
        std_ece = np.std([m['ece'] for m in metrics_list])
        avg_logloss = np.mean([m['log_loss'] for m in metrics_list])
        std_logloss = np.std([m['log_loss'] for m in metrics_list])
        avg_mce = np.mean([m['mce'] for m in metrics_list])
        std_mce = np.std([m['mce'] for m in metrics_list])

        print(f"\nLead Time = {lt}:")
        print(f"  AUPRC:       {avg_auprc:.4f} ± {std_auprc:.4f}")
        print(f"  AUROC:       {avg_auroc:.4f} ± {std_auroc:.4f}")
        print(f"  F1:          {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"  Precision:   {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"  Recall:      {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"  Climatology: {climatology_stats[lt]['climatology']:.4f}")
        print(f"  Brier Ref:   {brier_ref:.4f}")
        print(f"  Brier Score: {avg_brier:.4f} ± {std_brier:.4f}")
        print(f"  BSS:         {avg_bss:.4f} (overall={overall_bss:.4f})")
        print(f"  ECE:         {avg_ece:.4f} ± {std_ece:.4f}")
        print(f"  Log Loss:    {avg_logloss:.4f} ± {std_logloss:.4f}")
        print(f"  MCE:         {avg_mce:.4f} ± {std_mce:.4f}")

    print(f"\n{'='*80}")
    print("Status")
    print(f"{'='*80}")
    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")
    print(f"Status")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Status")
    parser.add_argument('--config', type=str,
                        default='/data1/ljt/TeleVIT/muti-teleVit-transformer/teleVIT-0654-overlap-csjxl-codex/config.yaml',
                        help="Status")
    parser.add_argument('--checkpoint', type=str,
                        default='/data1/ljt/TeleVIT/muti-teleVit-transformer/teleVIT-0654-overlap-csjxl-codex/checkpoints/0.654.pth',
                        help="Status")
    parser.add_argument('--lead-times', type=str, default=None,
                        help="Status")
    parser.add_argument('--stride', type=int, default=60,
                        help="Status")
    parser.add_argument('--batch-size', type=int, default=16,
                        help="Status")
    parser.add_argument('--output-dir', type=str,
                        default='/data1/ljt/TeleVIT/muti-teleVit-transformer/teleVIT-0654-overlap-csjxl-codex/inference_results',
                        help="Status")
    parser.add_argument('--num-samples', type=int, default=-1,
                        help="Status")
    parser.add_argument('--use-gaussian-weight', action='store_true', default=True,
                        help="Status")
    parser.add_argument('--no-gaussian-weight', dest='use_gaussian_weight', action='store_false',
                        help="Status")

    args = parser.parse_args()
    main(args)
