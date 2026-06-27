"""
推理脚本：滑动窗口预测与融合
实现整图预测，消除patch拼接边界伪影
保存预测结果(npy)和指标(csv)

修改了计算指标添加一些概率指标的计算
以及结果的保存路径 一个文件夹放真实的npy数据 一个文件夹放预测的npy结果 一个文件夹放NDVI掩码

"""
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
    """
    创建2D高斯权重矩阵，用于滑动窗口融合时的加权平均

    Args:
        patch_size: patch大小
        sigma: 高斯标准差（默认为patch_size/4）

    Returns:
        weight: (patch_size, patch_size) 高斯权重矩阵，中心权重高，边缘权重低
    """
    if sigma is None:
        sigma = patch_size / 4

    center = patch_size / 2
    x = np.arange(patch_size)
    y = np.arange(patch_size)
    xx, yy = np.meshgrid(x, y)

    # 计算到中心的距离
    dist = np.sqrt((xx - center + 0.5)**2 + (yy - center + 0.5)**2)

    # 高斯权重
    weight = np.exp(-(dist**2) / (2 * sigma**2))

    return weight.astype(np.float32)


def compute_ece(probs, targets, n_bins=10):
    """
    计算 Expected Calibration Error (ECE) 和 Maximum Calibration Error (MCE)

    Args:
        probs: 预测概率 (N,)
        targets: 真值标签 (N,)
        n_bins: bins数量

    Returns:
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到在当前bin中的样本
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # 计算bin中的平均置信度
            confidence_in_bin = probs[in_bin].mean()
            # 计算bin中的实际准确率
            accuracy_in_bin = targets[in_bin].mean()
            # 计算校准误差
            calibration_error = abs(confidence_in_bin - accuracy_in_bin)

            # ECE: 加权平均
            ece += prop_in_bin * calibration_error
            # MCE: 最大值
            mce = max(mce, calibration_error)

    return ece, mce


def compute_calibration_metrics(probs, targets, mask=None, n_bins=10):
    """
    计算概率校准指标

    Args:
        probs: 预测概率 (N, H, W) 或 (N*H*W,)
        targets: 真值标签 (N, H, W) 或 (N*H*W,)
        mask: 掩码 (N, H, W) 或 (N*H*W,)，1表示忽略
        n_bins: ECE计算的bins数量

    Returns:
        dict: {
            'brier_score': float,
            'ece': float,
            'log_loss': float,
            'mce': float
        }
    """
    # 展平数据
    if probs.ndim > 1:
        probs = probs.flatten()
        targets = targets.flatten()
        if mask is not None:
            mask = mask.flatten()

    # 过滤掉masked区域
    if mask is not None:
        valid_mask = (mask == 0)
        probs = probs[valid_mask]
        targets = targets[valid_mask]

    # 确保targets是0/1
    targets = targets.astype(np.int64)

    # 1. Brier Score
    brier = brier_score_loss(targets, probs)

    # 2. Log Loss (避免log(0)，添加小的epsilon)
    probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    logloss = log_loss(targets, probs_clipped)

    # 3. ECE 和 MCE
    ece, mce = compute_ece(probs, targets, n_bins=n_bins)

    return {
        'brier_score': float(brier),
        'ece': float(ece),
        'log_loss': float(logloss),
        'mce': float(mce)
    }


def prepare_valid_prob_targets(probs, targets, mask=None):
    """
    展平并过滤掩码区域，返回有效像素上的概率和0/1标签

    Args:
        probs: 预测概率
        targets: 真值标签
        mask: 掩码，1表示忽略

    Returns:
        probs_valid: 有效区域概率，shape (N_valid,)
        targets_valid: 有效区域标签，shape (N_valid,)
    """
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
    """
    基于测试集正类比例 q 计算 climatology 和 Brier 参考值 q(1-q)
    """
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
    """
    滑动窗口推理：对整张图像进行重叠patch预测并融合

    Args:
        model: 训练好的模型
        ds: 输入数据集
        ds_target: 目标数据集
        time_idx: 时间索引
        fire_vars: 火驱动变量列表
        oci_vars: OCI变量列表
        target_var: 目标变量名
        mean_std_dict: 标准化参数字典
        coord_grid_local: 局部位置编码 (4, H, W)
        coord_grid_global: 全局位置编码 (4, H_g, W_g)
        ds_global: 全局数据集
        temporal_steps: 时间步数
        oci_window: OCI窗口大小
        lead_time_steps: 预测时间步长列表
        patch_size: patch大小
        stride: 滑动步长
        batch_size: 批处理大小
        device: 设备
        use_local/use_global/use_oci: 三分支开关

    Returns:
        output_probs: 融合后的概率图 (L, H, W)，L为lead_times数量
        target: 真值标签 (L, H, W)
        ndvi_mask: NDVI掩码 (H, W)
    """
    model.eval()

    H, W = ds['latitude'].shape[0], ds['longitude'].shape[0]
    num_leads = len(lead_time_steps)

    # 计算滑动窗口的起始位置
    h_starts = list(range(0, H - patch_size, stride))
    if len(h_starts) == 0 or h_starts[-1] != H - patch_size:
        h_starts.append(H - patch_size)

    w_starts = list(range(0, W - patch_size, stride))
    if len(w_starts) == 0 or w_starts[-1] != W - patch_size:
        w_starts.append(W - patch_size)

    print(f"  滑动窗口配置: {len(h_starts)} × {len(w_starts)} = {len(h_starts) * len(w_starts)} patches")
    print(f"  重叠率: {(patch_size - stride) / patch_size * 100:.1f}%")
    print(f"  高斯加权融合: {'启用' if use_gaussian_weight else '禁用'}")

    # 创建高斯权重矩阵（用于平滑边界）
    if use_gaussian_weight:
        gaussian_weight = create_gaussian_weight(patch_size)
        print(f"  高斯权重: sigma={patch_size/4:.1f}, 中心权重={gaussian_weight[patch_size//2, patch_size//2]:.3f}, 边缘权重={gaussian_weight[0, 0]:.3f}")
    else:
        gaussian_weight = np.ones((patch_size, patch_size), dtype=np.float32)

    # 初始化累积矩阵
    output_sum = np.zeros((num_leads, 3, H, W), dtype=np.float32)  # 累积概率
    output_count = np.zeros((H, W), dtype=np.float32)  # 累积次数

    # 准备批处理
    patches_data = []
    patches_positions = []

    with torch.no_grad():
        for i in tqdm(h_starts, desc="  滑动窗口推理", leave=False):
            for j in w_starts:
                # 提取patch数据
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

                    # 添加位置编码
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

                # 添加到批处理
                patches_data.append((x_local, x_global, x_oci))
                patches_positions.append((i, j))

                # 当批次满了或者是最后一个patch时，进行推理
                if len(patches_data) == batch_size or (i == h_starts[-1] and j == w_starts[-1]):
                    # 组装batch
                    batch_local = torch.from_numpy(np.stack([p[0] for p in patches_data])).to(device)
                    batch_global = torch.from_numpy(np.stack([p[1] for p in patches_data])).to(device)
                    batch_oci = torch.from_numpy(np.stack([p[2] for p in patches_data])).to(device)

                    # 模型推理
                    logits = model(batch_local, batch_global, batch_oci)  # (B, L, 3, 80, 80) or (B, 3, 80, 80)

                    # 统一为5维
                    if logits.dim() == 4:
                        logits = logits.unsqueeze(1)  # (B, 1, 3, 80, 80)

                    # 转换为概率
                    probs = torch.softmax(logits, dim=2)  # (B, L, 3, 80, 80)
                    probs = probs.cpu().numpy()

                    # 融合到累积矩阵
                    for b, (pi, pj) in enumerate(patches_positions):
                        pi1 = pi + patch_size
                        pj1 = pj + patch_size

                        # 获取NDVI掩码
                        ndvi_data = ds['ndvi'].isel(time=time_idx).values[pi:pi1, pj:pj1]
                        mask_patch = (~np.isnan(ndvi_data)).astype(np.float32)  # 1=有效, 0=掩码

                        # 应用高斯权重（中心权重高，边缘权重低，平滑边界）
                        weight_patch = gaussian_weight * mask_patch

                        # 累加预测（加权累加）
                        for l in range(num_leads):
                            output_sum[l, :, pi:pi1, pj:pj1] += probs[b, l] * weight_patch
                        output_count[pi:pi1, pj:pj1] += weight_patch

                    # 清空批处理
                    patches_data = []
                    patches_positions = []

    # 计算平均概率
    output_count_safe = np.where(output_count == 0, 1, output_count)  # 防止除零
    output_probs = output_sum / output_count_safe[np.newaxis, np.newaxis, :, :]  # (L, 3, H, W)

    # 对于未预测的区域，设置为0
    output_probs[:, :, output_count == 0] = 0

    # 提取火灾类别的概率（class=1）
    fire_probs = output_probs[:, 1, :, :]  # (L, H, W)

    # 获取真值标签
    target_list = []
    for lead_t in lead_time_steps:
        target_time_idx = time_idx + lead_t
        target_data = ds_target[target_var].isel(time=target_time_idx).values
        target_data = np.nan_to_num(target_data, nan=0.0)
        target_binary = np.where(target_data > 0.0, 1, 0).astype(np.int64)
        target_list.append(target_binary)

    target = np.stack(target_list, axis=0)  # (L, H, W)

    # 获取NDVI掩码
    ndvi_data = ds['ndvi'].isel(time=time_idx).values
    ndvi_mask = np.isnan(ndvi_data).astype(np.float32)  # 1=掩码, 0=有效

    return fire_probs, target, ndvi_mask


def main(args):
    from dataset import SeasFirePatchDataset
    from model_multibranch_vit import create_model
    from utils import compute_metrics

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.lead_times is not None:
        config['data']['lead_time_steps'] = [int(x) for x in args.lead_times.split(',') if x.strip()]
        print(f"[命令行参数] lead_time_steps: {config['data']['lead_time_steps']}")

    # 解析lead_times
    if isinstance(config['data']['lead_time_steps'], int):
        lead_times = [config['data']['lead_time_steps']]
    else:
        lead_times = list(config['data']['lead_time_steps'])

    print(f"\n{'='*80}")
    print("滑动窗口推理配置")
    print(f"{'='*80}")
    print(f"模型路径: {args.checkpoint}")
    print(f"Patch大小: {config['data']['patch_size']}")
    print(f"滑动步长: {args.stride}")
    print(f"重叠率: {(config['data']['patch_size'] - args.stride) / config['data']['patch_size'] * 100:.1f}%")
    print(f"批处理大小: {args.batch_size}")
    print(f"预测时间步长: {lead_times}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 创建模型
    print(f"\n{'='*80}")
    print("加载模型")
    print(f"{'='*80}")
    model = create_model(config).to(device)

    # 加载权重
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ 模型加载成功")

    # 加载数据集（用于获取标准化参数和位置编码）
    print(f"\n{'='*80}")
    print("准备数据")
    print(f"{'='*80}")

    # 创建一个临时数据集来获取标准化参数
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

    # 获取标准化参数和位置编码
    mean_std_dict = temp_dataset.mean_std_dict
    coord_grid_local = temp_dataset.coord_grid_local
    coord_grid_global = temp_dataset.coord_grid_global
    ds = temp_dataset.ds
    ds_target = temp_dataset.ds_target
    ds_global = temp_dataset.ds_global if config['model']['use_global'] else None

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 推理
    print(f"\n{'='*80}")
    print("开始推理")
    print(f"{'='*80}")

    # 获取测试集的时间索引
    time_years = ds['time'].dt.year.values
    test_years = config['data']['test_years']
    valid_mask = np.isin(time_years, test_years)
    valid_times = np.where(valid_mask)[0]

    # 排除边界（考虑 temporal_steps）
    temporal_steps = config['data'].get('temporal_steps', 4)
    oci_window = config['data']['oci_window']
    max_lead_time = max(lead_times)

    # 计算所需的最小历史步数
    min_history_steps = max(temporal_steps - 1, oci_window)

    valid_times = valid_times[
        (valid_times >= min_history_steps) &
        (valid_times < len(ds['time']) - max_lead_time)
    ]

    print(f"边界配置:")
    print(f"  - temporal_steps: {temporal_steps} (需要 {temporal_steps-1} 个历史步)")
    print(f"  - oci_window: {oci_window}")
    print(f"  - max_lead_time: {max_lead_time}")
    print(f"  - min_history_steps: {min_history_steps}")
    print(f"  - 有效时间范围: {valid_times[0]} 到 {valid_times[-1]}")
    print(f"测试集时间步数: {len(valid_times)}")

    # 选择要推理的时间步（可以全部推理或采样）
    if args.num_samples > 0:
        sample_indices = np.linspace(0, len(valid_times) - 1, args.num_samples, dtype=int)
        valid_times = valid_times[sample_indices]
        print(f"采样 {args.num_samples} 个时间步进行推理")

    # 累积指标和预测结果
    all_metrics = {lt: [] for lt in lead_times}
    all_predictions = []  # 存储所有预测结果
    all_targets = []  # 存储所有真值
    all_masks = []  # 存储所有掩码
    time_indices = []  # 存储时间索引

    for idx, time_idx in enumerate(tqdm(valid_times, desc="推理进度")):
        # 滑动窗口推理
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

        # 保存预测结果
        all_predictions.append(fire_probs)  # (L, H, W)
        all_targets.append(target)  # (L, H, W)
        all_masks.append(ndvi_mask)  # (H, W)
        time_indices.append(time_idx)

        # 计算每个lead time的指标
        metrics_dict = {'time_idx': time_idx}
        for l, lt in enumerate(lead_times):
            # 计算基础指标
            metrics = compute_metrics(
                fire_probs[l:l+1],  # (1, H, W)
                target[l:l+1],  # (1, H, W)
                mask=ndvi_mask[np.newaxis, :, :],  # (1, H, W)
                threshold=0.5
            )

            # 计算概率校准指标
            calib_metrics = compute_calibration_metrics(
                fire_probs[l:l+1],  # (1, H, W)
                target[l:l+1],  # (1, H, W)
                mask=ndvi_mask[np.newaxis, :, :],  # (1, H, W)
                n_bins=10
            )

            # 合并指标
            metrics.update(calib_metrics)
            all_metrics[lt].append(metrics)

            # 添加到当前时间步的指标字典
            metrics_dict[f'lead{lt}_auprc'] = metrics['auprc']
            metrics_dict[f'lead{lt}_auroc'] = metrics['auroc']
            metrics_dict[f'lead{lt}_f1'] = metrics['f1']
            metrics_dict[f'lead{lt}_precision'] = metrics['precision']
            metrics_dict[f'lead{lt}_recall'] = metrics['recall']
            metrics_dict[f'lead{lt}_brier_score'] = metrics['brier_score']
            metrics_dict[f'lead{lt}_ece'] = metrics['ece']
            metrics_dict[f'lead{lt}_log_loss'] = metrics['log_loss']
            metrics_dict[f'lead{lt}_mce'] = metrics['mce']

        # 打印当前时间步的指标
        print(f"\n时间步 {idx+1}/{len(valid_times)} (time_idx={time_idx}):")
        for lt in lead_times:
            m = all_metrics[lt][-1]
            print(f"  Lead={lt}: AUPRC={m['auprc']:.4f}, AUROC={m['auroc']:.4f}, F1={m['f1']:.4f}, Brier={m['brier_score']:.4f}, ECE={m['ece']:.4f}")

    # ========== 保存预测结果 (NPY) ==========
    # 【修改】每个时间步单独保存npy文件，文件名包含日期信息
    print(f"\n{'='*80}")
    print("保存预测结果")
    print(f"{'='*80}")

    # 转换为numpy数组
    all_predictions = np.array(all_predictions)  # (N, L, H, W)
    all_targets = np.array(all_targets)  # (N, L, H, W)
    all_masks = np.array(all_masks)  # (N, H, W)
    time_indices = np.array(time_indices)  # (N,)

    # 基于整个测试集计算每个lead time的 climatology Brier 参考值和 BSS
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

    # 【新增】创建三个子文件夹，分类保存不同类型的数据
    predictions_dir = os.path.join(args.output_dir, 'predictions')
    ground_truth_dir = os.path.join(args.output_dir, 'ground_truth')
    masks_dir = os.path.join(args.output_dir, 'masks')

    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # 【新增】自动适配单/多lead time的情况
    if all_predictions.ndim == 4 and all_predictions.shape[1] == 1:
        all_predictions = all_predictions[:, 0, :, :]
        all_targets = all_targets[:, 0, :, :]
        print(f"检测到单个lead time，数据维度已调整为 (N, H, W)")
    elif all_predictions.ndim == 4:
        print(f"检测到多个lead times: {all_predictions.shape[1]}")

    # 【修改】逐个时间步保存npy文件，文件名包含日期
    print(f"\n保存 {len(time_indices)} 个时间步的预测结果...")
    for idx, time_idx in enumerate(time_indices):
        # 从数据集获取日期信息
        time_value = ds['time'].isel(time=int(time_idx)).values
        date_str = pd.Timestamp(time_value).strftime('%Y%m%d')

        # 保存预测结果
        pred_path = os.path.join(predictions_dir, f'prediction_{date_str}.npy')
        np.save(pred_path, all_predictions[idx])

        # 保存真值标签
        target_path = os.path.join(ground_truth_dir, f'ground_truth_{date_str}.npy')
        np.save(target_path, all_targets[idx])

        # 保存掩码
        mask_path = os.path.join(masks_dir, f'mask_{date_str}.npy')
        np.save(mask_path, all_masks[idx])

    # 保存时间索引映射文件
    time_indices_path = os.path.join(args.output_dir, 'time_indices.npy')
    np.save(time_indices_path, time_indices)

    print(f"✓ 预测结果已保存:")
    print(f"  - predictions/ 文件夹: {len(time_indices)} 个文件")
    print(f"  - ground_truth/ 文件夹: {len(time_indices)} 个文件")
    print(f"  - masks/ 文件夹: {len(time_indices)} 个文件")
    print(f"  - time_indices.npy: 时间索引映射")

    # ========== 保存单天指标 (CSV) ==========
    # 【修改】CSV文件包含新增的4个概率校准指标
    print(f"\n{'='*80}")
    print("保存指标结果")
    print(f"{'='*80}")

    # 构建单天指标DataFrame
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
            # 基础指标
            row[f'lead{lt}_auprc'] = m['auprc']
            row[f'lead{lt}_auroc'] = m['auroc']
            row[f'lead{lt}_f1'] = m['f1']
            row[f'lead{lt}_precision'] = m['precision']
            row[f'lead{lt}_recall'] = m['recall']
            # 【新增】概率校准指标
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
    print(f"✓ 单天指标已保存: {per_day_csv_path}")

    # ========== 保存平均指标 (CSV) ==========
    # 【修改】包含新增概率校准指标的均值和标准差
    avg_metrics = []
    for lt in lead_times:
        metrics_list = all_metrics[lt]
        brier_ref = climatology_stats[lt]['brier_ref']
        daily_bss = []
        if np.isfinite(brier_ref) and brier_ref > 0:
            daily_bss = [1.0 - m['brier_score'] / brier_ref for m in metrics_list]
        row = {
            'lead_time': lt,
            # 基础指标的均值和标准差
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
            # 【新增】概率校准指标的均值和标准差
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
    print(f"✓ 平均指标已保存: {avg_csv_path}")

    # ========== 打印最终结果 ==========
    # 【修改】显示包含概率校准指标的完整结果汇总
    print(f"\n{'='*80}")
    print("最终结果汇总")
    print(f"{'='*80}")

    for lt in lead_times:
        metrics_list = all_metrics[lt]
        # 基础指标统计
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
        # 【新增】概率校准指标统计
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
    print("✓ 推理完成！")
    print(f"{'='*80}")
    print(f"输出目录: {args.output_dir}")
    print(f"包含文件:")
    print(f"  - predictions.npy: 预测概率 (N, L, H, W)")
    print(f"  - targets.npy: 真值标签 (N, L, H, W)")
    print(f"  - masks.npy: NDVI掩码 (N, H, W)")
    print(f"  - time_indices.npy: 时间索引 (N,)")
    print(f"  - metrics_per_day.csv: 每天的指标")
    print(f"  - metrics_average.csv: 平均指标")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='滑动窗口推理脚本')
    parser.add_argument('--config', type=str,
                        default='configs/config.example.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pth',
                        help='模型checkpoint路径')
    parser.add_argument('--lead-times', type=str, default=None,
                        help='多步预测的 lead time steps，逗号分隔，如 "1,2,4,8,16"。不传则使用 config.yaml 的 data.lead_time_steps')
    parser.add_argument('--stride', type=int, default=60,
                        help='滑动窗口步长（默认60，即25%重叠）')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批处理大小')
    parser.add_argument('--output-dir', type=str,
                        default='inference_results',
                        help='输出目录')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='采样推理的时间步数（-1表示全部）')
    parser.add_argument('--use-gaussian-weight', action='store_true', default=True,
                        help='是否使用高斯加权融合（默认启用，提升边界平滑度）')
    parser.add_argument('--no-gaussian-weight', dest='use_gaussian_weight', action='store_false',
                        help='禁用高斯加权融合')

    args = parser.parse_args()
    main(args)
