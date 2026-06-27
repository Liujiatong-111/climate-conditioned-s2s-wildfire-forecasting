"""
工具函数：指标计算、随机种子设置等
"""
import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from typing import Dict


def set_seed(seed: int = 42):
    """设置所有随机种子，确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray = None, threshold: float = 0.5) -> Dict[str, float]:
    """
    计算二分类指标（排除掩码区域）

    Args:
        preds: 预测概率，shape (N,) 或 (N, H, W)
        targets: 真值标签，shape (N,) 或 (N, H, W)
        mask: 掩码，shape (N,) 或 (N, H, W)，1表示掩码区域（需排除），0表示有效区域
        threshold: 二值化阈值

    Returns:
        dict: 包含 accuracy, precision, recall, f1, auroc, auprc
    """
    # 展平
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    # 如果提供了掩码，只保留非掩码区域（与 televit-main 一致）
    if mask is not None:
        mask_flat = mask.reshape(-1)
        valid_mask = (mask_flat == 0)  # 0 表示有效区域
        preds_flat = preds_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]

    # 二值化
    preds_binary = (preds_flat >= threshold).astype(int)
    targets_binary = targets_flat.astype(int)

    # 计算指标（处理可能的 zero_division 警告）
    acc = accuracy_score(targets_binary, preds_binary)
    prec = precision_score(targets_binary, preds_binary, zero_division=0)
    rec = recall_score(targets_binary, preds_binary, zero_division=0)
    f1 = f1_score(targets_binary, preds_binary, zero_division=0)

    # 计算 AUROC 和 AUPRC（需要概率值，不是二值化的预测）
    # 检查是否有足够的正负样本来计算这些指标
    try:
        if len(np.unique(targets_binary)) > 1:  # 至少有两个类别
            auroc = roc_auc_score(targets_binary, preds_flat)
            auprc = average_precision_score(targets_binary, preds_flat)
        else:
            # 如果只有一个类别，无法计算 AUROC 和 AUPRC
            auroc = 0.0
            auprc = 0.0
    except Exception as e:
        # 处理任何其他异常情况
        print(f"Warning: Could not compute AUROC/AUPRC: {e}")
        auroc = 0.0
        auprc = 0.0

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc
    }


def create_output_dir(output_dir: str):
    """创建输出目录"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 输出目录: {output_dir}")


def get_model_name(use_local: bool, use_global: bool, use_oci: bool) -> str:
    """根据输入配置生成模型名称"""
    l = '1' if use_local else '0'
    g = '1' if use_global else '0'
    i = '1' if use_oci else '0'
    return f"L{l}_G{g}_I{i}"


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """打印指标"""
    print(f"{prefix}Metrics:")
    for key, value in metrics.items():
        # 如果是数字类型，使用 .4f 格式化；否则直接打印
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
