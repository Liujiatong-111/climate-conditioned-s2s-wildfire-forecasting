"""
训练脚本：TeleViT多时序多步预测火灾预测模型

=================================================================================
整体方案概述：
=================================================================================
目标：将TeleViT框架的遥感火灾预测性能从AUPRC≈0.60提升至0.65以上，支持多时序输入和多步预测

核心改进策略：
1. 【输入格式与数据预处理】
   - 增加历史时序深度（T=4→6/8步）
   - 多步预测标签生成（支持+1,+2,+4,+8,+16天预测）
   - 数据平衡与增强（难负样本挖掘、数据增强）

2. 【时序信息建模】
   - 时序卷积网络(TCN)：在patch embedding后沿时间轴提取特征
   - Temporal Mixer：时间-特征分解混合
   - 改进的注意力池化：动态调整各时间帧权重
   - 因子化时间注意力：分离空间和时间注意力

3. 【多步预测架构】
   - 多输出分割头：为每个预测时刻设置独立输出头
   - 共享-特定融合：主干共享+末层解耦
   - 多任务损失加权：∑ w_t·L_t
   - 分horizon评价指标

4. 【多分支协同增强】
   - 跨分支特征融合：Global/OCI特征显式融合到Local
   - 跨注意力交互：Local-Global交叉注意力
   - 类型嵌入改进：区分分支来源的位置编码

=================================================================================
本版本实现：
=================================================================================
✅ 多步预测输出：支持lead_times=[1,2,4,8,16]等多时间步长同时预测
✅ 多任务损失：对每个lead time单独计算loss并加权平均
✅ 独立输出头：每个lead time使用独立的分类器（ProgressiveSegmentationHead）
✅ 时序建模：通过TemporalAttentionPool动态加权聚合时间信息
✅ 多分支融合：Local/Global/OCI三分支通过Transformer和类型嵌入协同
✅ 分horizon评估：为每个预测时刻单独计算AUPRC/AUROC/F1等指标

模型结构流程：
输入 → [Local/Global/OCI编码] → [Token拼接+位置编码] → [Transformer编码器]
    → [时序注意力池化] → [多输出分割头] → 输出(B,L,3,80,80)

兼容性：
- 单步预测：logits (B, 3, 80, 80)
- 多步预测：logits (B, L, 3, 80, 80)，L为lead_times数量

数据格式：
- 输入：x_local(B,T,14,80,80), x_global(B,T,14,180,360), x_oci(B,10,10)
- 标签：y(B,L,80,80) 或 y(B,80,80)（自动扩展）
- 输出：logits(B,L,3,80,80)，3类为[no_fire, fire, masked]
"""

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
    """把 config 中的 lead_time_steps 规范为 List[int]。"""
    if value is None:
        return [1]
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    # yaml 里也可能写成字符串
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(',') if p.strip()]
        return [int(p) for p in parts]
    raise TypeError(f"Unsupported lead_time_steps type: {type(value)}")


def _apply_ignore_mask_to_targets(y: torch.Tensor, mask: torch.Tensor, ignore_index: int = 2) -> torch.Tensor:
    """把 mask==1 的像素位置标成 ignore_index。

    支持：
    - y: (B, H, W)
    - y: (B, L, H, W)

    mask: (B, H, W)（float/uint8/bool 均可）
    """
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
    """
    训练一个epoch（兼容单步/多步输出）

    【对应方案】多步预测架构 - 多任务损失加权

    实现细节：
    1. 模型前向传播：
       - 输入：x_local(B,T,14,80,80), x_global(B,T,14,180,360), x_oci(B,10,10)
       - 输出：logits(B,L,3,80,80) 或 (B,3,80,80)

    2. 损失计算策略：
       - 单步：直接计算CrossEntropyLoss
       - 多步：对每个lead time单独计算loss，然后平均（等权重w_t=1）
       - 公式：L = (1/L) * ∑_{l=1}^{L} CrossEntropyLoss(logits[:,l], y[:,l])

    3. 掩码处理：
       - 将mask==1的像素标记为ignore_index=2（海洋/沙漠区域）
       - 损失函数自动忽略这些像素

    Args:
        model: MultiBranchViT模型
        train_loader: 训练数据加载器
        criterion: 损失函数（支持ignore_index）
        optimizer: 优化器
        device: 设备（cuda/cpu）
        epoch: 当前epoch
        ignore_index: 忽略的类别索引（默认2）

    Returns:
        float: 平均训练损失
    """
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for _, (x_local, x_global, x_oci, y, mask, _, _, _) in enumerate(pbar):
        # ========== 数据移到GPU ==========
        x_local = x_local.to(device)  # (B, T, 14, 80, 80)
        x_global = x_global.to(device)  # (B, T, 14, 180, 360)
        x_oci = x_oci.to(device)  # (B, 10, 10)
        y = y.to(device)  # (B, L, 80, 80) 或 (B, 80, 80)
        mask = mask.to(device)  # (B, 80, 80)

        # ========== 模型前向传播 ==========
        # 【对应方案】模型结构流程：
        # 输入 → Local/Global/OCI编码 → Token拼接 → Transformer → 时序池化 → 多输出头
        logits = model(x_local, x_global, x_oci)  # (B, L, 3, 80, 80) 或 (B, 3, 80, 80)

        # ========== 单步输出：logits (B, 3, H, W) ==========
        if logits.dim() == 4:
            # 【对应方案】单步预测（向后兼容）
            y_masked = _apply_ignore_mask_to_targets(y, mask, ignore_index=ignore_index)  # (B,H,W)
            loss = criterion(logits, y_masked)

        # ========== 多步输出：logits (B, L, 3, H, W) ==========
        elif logits.dim() == 5:
            # 【对应方案】多步预测架构 - 多任务损失加权
            B, L, C, H, W = logits.shape

            # 兼容：如果数据集仍返回 (B,H,W)，则复制成 (B,L,H,W)
            if y.dim() == 3:
                y = y.unsqueeze(1).expand(B, L, y.size(-2), y.size(-1))

            # 现在期望 y: (B,L,H,W)
            if y.dim() != 4:
                raise ValueError(f"Multi-step expects y dim=4, got y.shape={tuple(y.shape)}")

            # 应用掩码：将海洋/沙漠区域标记为ignore_index
            y_masked = _apply_ignore_mask_to_targets(y, mask, ignore_index=ignore_index)  # (B,L,H,W)

            # 【核心】多任务损失计算：对每个lead time单独计算loss，然后平均
            # 实现公式：L_total = (1/L) * ∑_{l=1}^{L} L_l
            # 其中 L_l = CrossEntropyLoss(logits[:,l], y_masked[:,l])
            #
            # 优势：
            # 1. 解耦优化：每个预测步长独立优化，避免梯度冲突
            # 2. 知识共享：通过共享Transformer主干，不同步长共享特征表示
            # 3. 可扩展：可轻松调整权重w_t以强调近期或远期预测
            losses = []
            for l in range(L):
                loss_l = criterion(logits[:, l], y_masked[:, l])
                losses.append(loss_l)

            # 等权重平均（可改为加权：loss = sum(w_t * losses[t]) / sum(w_t)）
            loss = torch.stack(losses).mean()

        else:
            raise ValueError(f"Unsupported logits dim: {logits.dim()} (shape={tuple(logits.shape)})")

        # ========== 反向传播与优化 ==========
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), x_local.size(0))
        pbar.set_postfix({'loss': loss_meter.avg})

    return loss_meter.avg


def evaluate_by_lead(model, data_loader, criterion, device, epoch, lead_times: list, ignore_index: int = 2, threshold: float = 0.5):
    """
    评估函数：为每个lead time单独计算指标

    【对应方案】多步预测架构 - 分horizon评价指标

    实现细节：
    1. 模型输出处理：
       - 单步：logits(B,3,H,W) → 自动扩展为(B,1,3,H,W)
       - 多步：logits(B,L,3,H,W) → 直接使用

    2. 指标计算策略：
       - 对每个lead time单独计算：AUPRC, AUROC, Precision, Recall, F1, Accuracy
       - 只计算land区域（mask==0的像素），排除海洋/沙漠
       - 使用softmax后的class=1概率作为预测分数

    3. 返回格式：
       {
           1: {'loss': 0.32, 'auprc': 0.71, 'auroc': 0.82, ...},
           2: {'loss': 0.34, 'auprc': 0.69, 'auroc': 0.81, ...},
           4: {'loss': 0.36, 'auprc': 0.67, 'auroc': 0.79, ...},
           ...
       }

    【对应方案优势】
    - 全面评估：可观察预测性能随时间步长的衰减趋势
    - 针对性优化：识别哪些时间步长需要改进
    - 多任务对比：比较不同预测horizon的难度差异

    Args:
        model: MultiBranchViT模型
        data_loader: 数据加载器（验证集或测试集）
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch（用于显示）
        lead_times: lead time列表，如[1,2,4,8,16]
        ignore_index: 忽略的类别索引
        threshold: 二值化阈值（用于计算Precision/Recall/F1）

    Returns:
        dict: {lead_time: metrics_dict}
    """
    model.eval()

    # 容器
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

            # 统一成 (B, L, 3, H, W)
            if logits.dim() == 4:
                logits = logits.unsqueeze(1)

            if logits.dim() != 5:
                raise ValueError(f"Unsupported logits dim in eval: {logits.dim()} (shape={tuple(logits.shape)})")

            B, L, C, H, W = logits.shape

            # lead_times 与 L 不一致时：自动截断到 min(L, len(lead_times))
            if len(lead_times) != L:
                min_L = min(len(lead_times), L)
                if min_L != L:
                    logits = logits[:, :min_L]
                    L = min_L
                lead_times_eff = lead_times[:min_L]
            else:
                lead_times_eff = lead_times

            # targets 统一成 (B, L, H, W)
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

            # 1) loss：每个 lead 单独算
            y_masked = _apply_ignore_mask_to_targets(y, mask, ignore_index=ignore_index)  # (B,L,H,W)
            for l, lt in enumerate(lead_times_eff):
                loss_l = criterion(logits[:, l], y_masked[:, l])
                loss_meters[lt].update(loss_l.item(), B)

            # 2) probs：取 class=1 的概率
            probs = torch.softmax(logits, dim=2)[:, :, 1]  # (B,L,H,W)
            probs = probs.masked_fill(mask_bool.unsqueeze(1), 0.0)

            probs_np = probs.cpu().numpy()
            y_np = y.cpu().numpy()
            mask_np = mask.cpu().numpy()

            for l, lt in enumerate(lead_times_eff):
                all_preds[lt].append(probs_np[:, l])
                all_targets[lt].append(y_np[:, l])
                all_masks[lt].append(mask_np)

            # tqdm 显示一个简要 loss（mean over leads in this batch）
            # 这里不严格，主要用于观察训练是否发散
            mean_batch_loss = np.mean([loss_meters[lt].val for lt in lead_times_eff]) if len(lead_times_eff) > 0 else 0.0
            pbar.set_postfix({'loss': float(mean_batch_loss)})

    # 汇总 metrics
    metrics_by_lead = {}
    for lt in lead_times:
        if len(all_preds[lt]) == 0:
            # 兼容：如果 lead_times 比实际输出多
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
    # 延迟导入（避免在仅 import 本文件做单元测试时触发 torchvision 相关依赖问题）
    from dataset import SeasFirePatchDataset
    from model_multibranch_vit import create_model

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 允许命令行覆盖 lead_times（多任务）
    if args.lead_times is not None:
        config['data']['lead_time_steps'] = [int(x) for x in args.lead_times.split(',') if x.strip()]
        print(f"[命令行参数] lead_time_steps: {config['data']['lead_time_steps']}")

    lead_times = _parse_lead_times_from_config(config['data'].get('lead_time_steps', 1))

    # 应用命令行参数覆盖配置文件（保留你原有逻辑）
    if args.stride is not None:
        config['data']['stride'] = args.stride
        print(f"[命令行参数] 滑动窗口步长: {args.stride}")

    if args.depth is not None:
        config['model']['depth'] = args.depth
        print(f"[命令行参数] 模型深度: {args.depth} 层")

    if args.load_pretrained:
        config['model']['load_pretrained_backbone'] = True
        print(f"[命令行参数] 加载预训练权重: True")

    if args.use_improved_head:
        config['model']['use_improved_head'] = True
        print(f"[命令行参数] 使用改进的分割头: True")

    if args.seg_head_type is not None:
        config['model']['seg_head_type'] = args.seg_head_type
        print(f"[命令行参数] 分割头类型: {args.seg_head_type}")

    if args.local_patch_size is not None:
        config['model']['local_patch_size'] = args.local_patch_size
        print(f"[命令行参数] Local Patch Size: {args.local_patch_size}")

    # 消融实验参数（命令行参数覆盖配置文件）
    config['model']['use_local'] = args.use_local
    config['model']['use_global'] = args.use_global
    config['model']['use_oci'] = args.use_oci
    print(f"[命令行参数] 输入分支配置: Local={args.use_local}, Global={args.use_global}, OCI={args.use_oci}")

    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
        print(f"[命令行参数] Batch size: {args.batch_size}")

    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
        print(f"[命令行参数] Epochs: {args.epochs}")

    if args.lr is not None:
        config['train']['learning_rate'] = args.lr
        print(f"[命令行参数] 学习率: {args.lr}")

    if args.weight_decay is not None:
        config['train']['weight_decay'] = args.weight_decay
        print(f"[命令行参数] 权重衰减: {args.weight_decay}")

    if args.optimizer is not None:
        config['train']['optimizer'] = args.optimizer
        print(f"[命令行参数] 优化器: {args.optimizer}")

    if args.lr_scheduler is not None:
        config['train']['lr_scheduler'] = args.lr_scheduler
        print(f"[命令行参数] 学习率调度器: {args.lr_scheduler}")

    if args.gpu_ids is not None:
        config['train']['gpu_ids'] = [int(x) for x in args.gpu_ids.split(',')]
        print(f"[命令行参数] GPU IDs: {config['train']['gpu_ids']}")

    print(f"[命令行参数] 损失函数类型: {args.loss_type}")

    # ========== 实验配置 ==========
    print("\n" + "="*80)
    print("实验配置")
    print("="*80)

    # 设置随机种子（保证可复现性）
    set_seed(config['train']['seed'])
    print(f"随机种子: {config['train']['seed']}")

    # ========== 路径配置 ==========
    # 【对应方案】整体方案 - 项目结构优化
    # 统一使用 teleVIT_multihorizon_patch 项目路径

    # 输出目录（通用）
    output_dir = config['output']['save_dir']
    create_output_dir(output_dir)

    # Checkpoint保存目录
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
        print(f"[命令行参数] Checkpoint 目录: {checkpoint_dir}")
    else:
        # 【修改】更新为当前项目路径，包含stride信息
        depth = config['model']['depth']
        lead_times_str = '_'.join(map(str, lead_times))  # 例如: "1_2_4_8_16"
        stride = config['data'].get('stride', config['data']['patch_size'])  # 默认等于patch_size
        stride_str = f"_stride{stride}" if stride != config['data']['patch_size'] else ""
        checkpoint_dir = os.path.join('checkpoints', f'{depth}layers_lead{lead_times_str}{stride_str}')
        print(f"[自动设置] Checkpoint 目录: {checkpoint_dir}")
    create_output_dir(checkpoint_dir)

    # Log保存目录
    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        # 【修改】更新为当前项目路径，包含stride信息
        lead_times_str = '_'.join(map(str, lead_times))
        stride = config['data'].get('stride', config['data']['patch_size'])
        stride_str = f"_stride{stride}" if stride != config['data']['patch_size'] else ""
        log_dir = os.path.join('logs', f'lead{lead_times_str}{stride_str}')
    create_output_dir(log_dir)
    print(f"日志目录: {log_dir}")

    # 实验名称（包含模型配置和时间戳）
    model_name = get_model_name(config['model']['use_local'], config['model']['use_global'], config['model']['use_oci'])
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    lead_times_str = '_'.join(map(str, lead_times))
    stride = config['data'].get('stride', config['data']['patch_size'])
    stride_str = f"_stride{stride}" if stride != config['data']['patch_size'] else ""
    experiment_name = f"{model_name}_lead{lead_times_str}{stride_str}_{timestamp}"
    print(f"实验名称: {experiment_name}")

    # ========== 日志记录器 ==========
    # 【对应方案】多步预测架构 - 分horizon评价指标
    # 单个CSV文件记录所有lead times的训练/验证/测试指标
    logger = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name, lead_times=lead_times)
    print(f"CSV日志: {logger.get_log_path()}")

    # GPU 设置
    gpu_ids = config['train'].get('gpu_ids', [])
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
        print(f"指定使用 GPU: {gpu_ids}")

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')

    n_gpus = torch.cuda.device_count()
    print(f"可用 GPU 数量: {n_gpus}")
    if n_gpus > 0:
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"主设备: {device}")

    # ========== 数据集加载 ==========
    print("\n" + "="*80)
    print("数据集加载")
    print("="*80)

    # 【对应方案】输入格式与数据预处理修改
    # 1. 增加历史时序深度：temporal_steps参数控制（默认T=4）
    # 2. 多步预测标签生成：lead_time_steps支持列表，如[1,2,4,8,16]
    # 3. 数据平衡与增强：only_fire_patches和use_augmentation参数控制

    lead_time_steps_cfg = config['data']['lead_time_steps']

    # 允许命令行参数覆盖config中的值
    print(f"\n{'='*80}")
    print("时序参数配置")
    print(f"{'='*80}")
    print(f"配置文件中的 temporal_steps: {config['data'].get('temporal_steps', '未设置')}")
    print(f"命令行参数 args.temporal_steps: {args.temporal_steps}")

    temporal_steps = args.temporal_steps if args.temporal_steps is not None else config['data'].get('temporal_steps', 4)
    oci_window = args.oci_window if args.oci_window is not None else config['data']['oci_window']

    print(f"\n最终使用的时序配置:")
    print(f"  - 历史时间步数 (temporal_steps): {temporal_steps}")
    print(f"  - OCI时间窗口 (oci_window): {oci_window}")
    print(f"  - 预测时间步长 (lead_time_steps): {lead_time_steps_cfg}")
    print(f"  - 数据增强: {args.use_augmentation}")

    if temporal_steps != config['data'].get('temporal_steps', 4):
        print(f"  ⚠️  警告：使用的 temporal_steps ({temporal_steps}) 与配置文件 ({config['data'].get('temporal_steps', 4)}) 不一致！")

    # ========== 训练集 ==========
    # 【对应方案】数据平衡策略
    # only_fire_patches=True: 只使用有火的patch，缓解类别不平衡
    # use_augmentation=True: 启用随机翻转、旋转等增强，提升泛化能力
    train_dataset = SeasFirePatchDataset(
        zarr_path=config['data']['zarr_path'],
        target_zarr_path=config['data']['target_zarr_path'],
        years=config['data']['train_years'],
        fire_vars=config['data']['fire_vars'],
        log_transform_vars=config['data']['log_transform_vars'],
        oci_vars=config['data']['oci_vars'],
        target_var=config['data']['target_var'],
        lead_time_steps=lead_time_steps_cfg,  # 支持int或list，如[1,2,4,8,16]
        oci_window=oci_window,
        temporal_steps=temporal_steps,  # 历史时间步数
        burn_threshold=config['data']['burn_threshold'],
        patch_size=config['data']['patch_size'],
        stride=config['data'].get('stride', None),  # 滑动窗口步长
        global_coarsen_factor=config['data']['global_coarsen_factor'],
        use_local=config['model']['use_local'],
        use_global=config['model']['use_global'],
        use_oci=config['model']['use_oci'],
        only_fire_patches=True,  # 训练集：只使用有火patch
        use_augmentation=args.use_augmentation,  # 数据增强
    )

    # ========== 验证集配置（从训练集中随机采样）==========
    # 【优化策略】避免每个epoch重新扫描数据
    # 方案：直接从训练集的样本中随机选择验证样本
    # - 训练集已经包含所有18年的数据
    # - 每个epoch随机选择一部分样本作为验证集（例如10-20%）
    # - 无需重新加载数据，只需要索引操作

    print(f"\n验证集策略: 每个epoch从训练集中随机采样 15% 作为验证集")
    print(f"训练集总样本数: {len(train_dataset)}")

    # 计算验证集大小（15%的训练集）
    val_size = int(len(train_dataset) * 0.15)
    print(f"每个epoch验证集大小: {val_size} 个样本")

    # ========== 测试集 ==========
    # 【对应方案】最终评估
    # only_fire_patches=False: 使用所有patch，最终性能评估
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
        stride=config['data'].get('stride', None),  # 滑动窗口步长
        global_coarsen_factor=config['data']['global_coarsen_factor'],
        use_local=config['model']['use_local'],
        use_global=config['model']['use_global'],
        use_oci=config['model']['use_oci'],
        only_fire_patches=False,  # 测试集：使用所有patch
        use_augmentation=False,  # 测试集：不使用数据增强
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
    )

    # 注意：val_loader 将在每个epoch开始时通过随机采样创建（使用 Subset）

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
    )

    # ========== 模型创建 ==========
    print("\n" + "="*80)
    print("创建模型")
    print("="*80)

    # 【对应方案】完整模型架构实现
    #
    # 模型结构流程：
    # 1. 输入编码：
    #    - Local分支 (TemporalLocalEncoder): (B,T,14,80,80) → (B,400,768)
    #    - Global分支 (TemporalGlobalEncoder): (B,T,14,180,360) → (B,72,768)
    #    - OCI分支 (OCIEncoder): (B,10,10) → (B,100,768)
    #
    # 2. 多分支融合：
    #    - Token拼接: [CLS, Local, Global, OCI] → (B,573,768)
    #    - 类型嵌入: 为每个分支添加可学习的类型标识
    #    - 位置编码: 添加可学习的位置嵌入
    #
    # 3. Transformer编码器 (12层):
    #    - Multi-Head Self-Attention (12 heads)
    #    - Feed-Forward Network (MLP)
    #    - 【对应方案】时序信息建模：通过自注意力隐式学习时间依赖
    #
    # 4. 时序注意力池化 (TemporalAttentionPool):
    #    - 【对应方案】改进的注意力池化：动态加权聚合时间维度
    #    - (B,768,T,10,10) → (B,768,10,10)
    #
    # 5. 多输出分割头 (ProgressiveSegmentationHead):
    #    - 【对应方案】多步预测架构：为每个lead time独立输出
    #    - 渐进式上采样: 10×10 → 20×20 → 40×40 → 80×80
    #    - 输出: (B,L,3,80,80)，L为lead_times数量
    #
    # 【对应方案优势】
    # ✅ 多分支协同：Local细节 + Global上下文 + OCI气候指数
    # ✅ 时序建模：Transformer自注意力 + 时序注意力池化
    # ✅ 多步预测：独立输出头，解耦优化
    # ✅ 知识共享：共享Transformer主干，参数高效

    model = create_model(config).to(device)

    # 多GPU并行训练
    if n_gpus > 1:
        print(f"使用 DataParallel 进行 {n_gpus} 卡并行训练")
        model = nn.DataParallel(model)
        effective_batch_size = config['train']['batch_size'] * n_gpus
        print(f"有效 Batch Size: {effective_batch_size} ({config['train']['batch_size']}/GPU × {n_gpus} GPUs)")
    else:
        print("使用单卡训练")

    # ========== 损失函数 ==========
    print("\n" + "="*50)
    print("配置损失函数")
    print("="*50)

    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=2)
        print("使用标准 CrossEntropyLoss（无权重）")

    elif args.loss_type == 'weighted_ce':
        class_weight = torch.tensor([1.0, args.fire_weight, 0.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=2)
        print("使用加权 CrossEntropyLoss")
        print(f"  类别权重: [no_fire=1.0, fire={args.fire_weight}, masked=0.0]")

    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=2)
        print(f"使用 Focal Loss: alpha={args.focal_alpha}, gamma={args.focal_gamma}")

    elif args.loss_type == 'combined':
        criterion = CombinedLoss(
            focal_weight=0.7,
            dice_weight=0.3,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            ignore_index=2
        )
        print("使用组合损失 (Focal + Dice)")

    else:
        raise ValueError(f"未知的损失函数类型: {args.loss_type}")

    # ========== 优化器 ==========
    optimizer_type = config['train'].get('optimizer', 'adamw').lower()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=config['train']['weight_decay']
        )
        print(f"使用 Adam 优化器 (lr={config['train']['learning_rate']}, wd={config['train']['weight_decay']})")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=config['train']['weight_decay']
        )
        print(f"使用 AdamW 优化器 (lr={config['train']['learning_rate']}, wd={config['train']['weight_decay']})")

    # ========== 学习率调度 ==========
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
        print(f"使用 ReduceLROnPlateau 调度器 (monitor={scheduler_monitor})")

    elif lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train']['epochs']
        )
        print(f"使用 CosineAnnealingLR 调度器 (T_max={config['train']['epochs']})")

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
        print("使用 Warmup + CosineAnnealingWarmRestarts 调度器")

    else:
        print("不使用学习率调度器")

    # ========== 数据维度验证 ==========
    print("\n" + "="*80)
    print("数据维度验证")
    print("="*80)

    # 从训练集中获取一个批次进行验证
    print("从训练集中提取一个批次进行验证...")
    train_iter = iter(train_loader)
    x_local_sample, x_global_sample, x_oci_sample, y_sample, mask_sample, _, _, _ = next(train_iter)

    print(f"\n实际数据维度:")
    print(f"  x_local:  {x_local_sample.shape}  (预期: (B, {temporal_steps}, 14, 80, 80))")
    print(f"  x_global: {x_global_sample.shape}  (预期: (B, {temporal_steps}, 14, 180, 360))")
    print(f"  x_oci:    {x_oci_sample.shape}  (预期: (B, 10, 10))")
    print(f"  y:        {y_sample.shape}  (预期: (B, 80, 80) 或 (B, L, 80, 80))")
    print(f"  mask:     {mask_sample.shape}  (预期: (B, 80, 80))")

    # 验证时间步维度
    assert x_local_sample.shape[1] == temporal_steps, (
        f"❌ x_local 的时间步维度不匹配！"
        f"实际: {x_local_sample.shape[1]}, 预期: {temporal_steps}"
    )
    assert x_global_sample.shape[1] == temporal_steps, (
        f"❌ x_global 的时间步维度不匹配！"
        f"实际: {x_global_sample.shape[1]}, 预期: {temporal_steps}"
    )

    print(f"\n✓ 数据维度验证通过！")
    print(f"  - temporal_steps={temporal_steps} 的数据已正确加载")
    print(f"  - 每个样本包含 {temporal_steps} 个历史时间步")

    # 检查数据中是否有异常值（NaN 或全0）
    if torch.isnan(x_local_sample).any():
        print(f"  ⚠️  警告：x_local 中包含 NaN 值！")
    if torch.isnan(x_global_sample).any():
        print(f"  ⚠️  警告：x_global 中包含 NaN 值！")

    # 检查是否有全0的时间步（可能表示数据加载错误）
    for t in range(temporal_steps):
        if torch.all(x_local_sample[:, t] == 0):
            print(f"  ⚠️  警告：x_local 的时间步 {t} 全为0！")
        if torch.all(x_global_sample[:, t] == 0):
            print(f"  ⚠️  警告：x_global 的时间步 {t} 全为0！")

    # 测试模型前向传播
    print(f"\n测试模型前向传播...")
    x_local_sample = x_local_sample.to(device)
    x_global_sample = x_global_sample.to(device)
    x_oci_sample = x_oci_sample.to(device)

    with torch.no_grad():
        logits_sample = model(x_local_sample, x_global_sample, x_oci_sample)

    print(f"  模型输出维度: {logits_sample.shape}")
    if logits_sample.dim() == 4:
        print(f"  ✓ 单步预测模式: (B, 3, 80, 80)")
    elif logits_sample.dim() == 5:
        print(f"  ✓ 多步预测模式: (B, {logits_sample.shape[1]}, 3, 80, 80)")
        assert logits_sample.shape[1] == len(lead_times), (
            f"输出的 lead times 数量不匹配！实际: {logits_sample.shape[1]}, 预期: {len(lead_times)}"
        )

    print(f"\n✓ 模型前向传播测试通过！")

    # ========== 训练循环 ==========
    print("\n" + "="*50)
    print("开始训练")
    print("="*50)

    best_score = -1.0
    best_model_info = None  # (best_score, best_epoch, path)
    patience_counter = 0

    checkpoint_dir_best = os.path.join(checkpoint_dir, 'best_land')
    create_output_dir(checkpoint_dir_best)

    for epoch in range(1, config['train']['epochs'] + 1):
        # ========== 每个epoch从训练集中随机采样验证集 ==========
        # 使用 Subset 高效采样，无需重新加载数据
        val_indices = random.sample(range(len(train_dataset)), val_size)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)

        val_loader = DataLoader(
            val_subset,
            batch_size=config['train']['batch_size'],
            shuffle=False,
            num_workers=config['train']['num_workers'],
            pin_memory=True,
        )

        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, ignore_index=2)

        # 验证：多 lead 评估
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

        # 取 mean AUPRC 作为主指标
        val_auprc_mean = _mean_by_lead(val_metrics_by_lead, lead_times, 'auprc')
        val_auroc_mean = _mean_by_lead(val_metrics_by_lead, lead_times, 'auroc')
        val_loss_mean = _mean_by_lead(val_metrics_by_lead, lead_times, 'loss')

        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config['train']['epochs']} - 验证集: 随机采样 {val_size} 个样本")
        print(f"{'='*80}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"\n  Validation Results (Land only):")
        print(f"  {'─'*76}")

        # 如果是多任务，显示每个lead time的详细结果
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
            # 单任务，简洁显示
            print(f"    Loss:  {val_loss_mean:.4f}")
            print(f"    AUROC: {val_auroc_mean:.4f}")
            print(f"    AUPRC: {val_auprc_mean:.4f}")

        # 当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 写 CSV（单文件，多 lead）
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
                    # mode='min' 所以传负值
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

        # 保存 best（用 mean AUPRC）
        current_score = val_auprc_mean
        if current_score > best_score:
            # 删除旧 best
            if best_model_info is not None:
                _, old_epoch, old_path = best_model_info
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"  ✗ 删除旧最佳模型 (Epoch {old_epoch})")

            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            checkpoint_path = os.path.join(checkpoint_dir_best, f'best_model_{model_name}_epoch{epoch}.pth')
            torch.save(model_to_save.state_dict(), checkpoint_path)

            best_score = current_score
            best_model_info = (best_score, epoch, checkpoint_path)
            patience_counter = 0
            print(f"  ✓ 保存最佳模型 (mean AUPRC={best_score:.4f}, Epoch {epoch}): {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  mean AUPRC={current_score:.4f} 未超过 best={best_score:.4f} (patience={patience_counter}/{config['train']['patience']})")

        if patience_counter >= config['train']['patience']:
            print(f"\n早停触发（patience={config['train']['patience']}）")
            break

        print("-" * 50)

    # ========== 测试 ==========
    print("\n" + "="*50)
    print("在测试集上评估最佳模型")
    print("="*50)

    if best_model_info is None:
        print("\n⚠ 没有保存的最佳模型，跳过测试")
        return

    best_score, best_epoch, best_model_path = best_model_info
    print(f"加载最佳模型: Epoch {best_epoch}, mean AUPRC={best_score:.4f}")
    print(f"模型路径: {best_model_path}")

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

    # 如果是多任务，显示每个lead time的详细结果
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
        # 单任务，显示所有指标
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

    print("\n✓ 训练完成！")
    print(f"✓ 最佳模型保存在: {best_model_path}")
    print(f"✓ 日志 CSV: {logger.get_log_path()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/config.example.yaml',
                        help='配置文件路径')

    # 多 lead 覆盖
    parser.add_argument('--lead-times', type=str, default=None,
                        help='多步预测的 lead time steps，逗号分隔，如 "1,2,4,8,16"。不传则使用 config.yaml 的 data.lead_time_steps')

    # 模型参数
    parser.add_argument('--depth', type=int, default=12, help='模型深度（8 或 12 层）')
    parser.add_argument('--load-pretrained', action='store_true', default=False, help='是否加载 ImageNet 预训练权重')
    parser.add_argument('--use-improved-head', action='store_true', default=True, help='是否使用改进的分割头')
    parser.add_argument('--seg-head-type', type=str, default='patchwise',
                        choices=['patchwise', 'improved', 'aspp', 'progressive'],
                        help='分割头类型')
    parser.add_argument('--local-patch-size', type=int, default=8, help='Local patch size (8 or 16)')

    # 消融实验参数（控制三个输入分支）
    parser.add_argument('--use-local', action='store_true', default=True, help='是否使用 Local 分支')
    parser.add_argument('--use-global', action='store_true', default=True, help='是否使用 Global 分支')
    parser.add_argument('--use-oci', action='store_true', default=True, help='是否使用 OCI 分支')

    # 训练参数
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='优化器类型')
    parser.add_argument('--lr-scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'cosine_warmup'], help='学习率调度器')

    # 损失函数参数
    parser.add_argument('--loss-type', type=str, default='combined',
                        choices=['ce', 'weighted_ce', 'focal', 'combined'],
                        help='损失函数类型')
    parser.add_argument('--fire-weight', type=float, default=4, help='火灾类别权重（weighted_ce）')
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal gamma')

    # 数据增强
    parser.add_argument('--use-augmentation', action='store_true', default=False, help='是否使用数据增强')

    # 时间步参数
    parser.add_argument('--temporal-steps', type=int, default=None,
                        help='Local/Global的历史时间步数（默认None，使用config.yaml中的值）')
    parser.add_argument('--oci-window', type=int, default=None,
                        help='OCI时间窗口长度（默认None，使用config.yaml中的值）')

    # 滑动窗口参数
    parser.add_argument('--stride', type=int, default=60,
                        help='滑动窗口步长（默认60，即25%%重叠；80表示无重叠；40表示50%%重叠）')

    # GPU
    parser.add_argument('--gpu-ids', type=str, default='0,1,3,4', help='GPU IDs，逗号分隔，如 "0,1,2,3"')

    # 输出目录
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Checkpoint 保存目录（默认None，自动根据配置生成）')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='日志保存目录（默认None，自动根据配置生成）')

    args = parser.parse_args()

    main(args)
