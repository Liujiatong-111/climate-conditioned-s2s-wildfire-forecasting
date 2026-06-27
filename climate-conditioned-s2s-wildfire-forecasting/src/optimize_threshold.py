"""
优化分类阈值
找到最优的概率阈值，平衡precision和recall

如果预测概率偏大，可能需要提高阈值（从0.5提高到0.6或0.7）
"""
import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

def compute_metrics_at_threshold(probs, targets, mask, threshold):
    """
    计算给定阈值下的指标
    """
    # 过滤掉masked区域
    valid_mask = (mask == 0)
    probs_valid = probs[valid_mask]
    targets_valid = targets[valid_mask]

    # 二值化预测
    preds = (probs_valid >= threshold).astype(np.int64)

    # 计算指标
    tp = ((preds == 1) & (targets_valid == 1)).sum()
    fp = ((preds == 1) & (targets_valid == 0)).sum()
    tn = ((preds == 0) & (targets_valid == 0)).sum()
    fn = ((preds == 0) & (targets_valid == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def find_optimal_threshold(probs_list, targets_list, mask_list):
    """
    搜索最优阈值
    """
    # 合并所有数据
    all_probs = np.concatenate([p.flatten() for p in probs_list])
    all_targets = np.concatenate([t.flatten() for t in targets_list])
    all_masks = np.concatenate([m.flatten() for m in mask_list])

    # 过滤掉masked区域
    valid_mask = (all_masks == 0)
    all_probs = all_probs[valid_mask]
    all_targets = all_targets[valid_mask]

    print(f"有效样本数: {len(all_targets):,}")
    print(f"正样本数: {all_targets.sum():,} ({all_targets.mean()*100:.4f}%)")

    # 计算PR曲线
    precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)

    # 计算每个阈值的F1
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    # 找到最优阈值（最大F1）
    best_idx = np.argmax(f1_scores[:-1])  # 排除最后一个点
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]

    print(f"\n{'='*60}")
    print("最优阈值搜索结果")
    print(f"{'='*60}")
    print(f"最优阈值: {best_threshold:.4f}")
    print(f"最大F1:   {best_f1:.4f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall:    {best_recall:.4f}")

    # 测试几个常用阈值
    print(f"\n{'='*60}")
    print("常用阈值对比")
    print(f"{'='*60}")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"{'-'*60}")

    test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, best_threshold]
    results = []

    for thresh in sorted(set(test_thresholds)):
        metrics = compute_metrics_at_threshold(all_probs, all_targets, all_masks, thresh)
        results.append(metrics)

        marker = " ⭐" if abs(thresh - best_threshold) < 0.01 else ""
        print(f"{thresh:<12.2f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}{marker}")

    # 绘制PR曲线
    plt.figure(figsize=(12, 5))

    # PR曲线
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.scatter([best_recall], [best_precision], c='r', s=100, zorder=5, label=f'Best (T={best_threshold:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # F1 vs Threshold
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, f1_scores[:-1], 'g-', linewidth=2)
    plt.axvline(best_threshold, color='r', linestyle='--', label=f'Best T={best_threshold:.3f}')
    plt.axvline(0.5, color='gray', linestyle=':', label='Default T=0.5')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs Threshold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    return best_threshold, results, plt


def collect_predictions(model, data_loader, device):
    """
    收集模型预测
    """
    model.eval()

    probs_list = []
    targets_list = []
    mask_list = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="收集预测"):
            x_local = batch['x_local'].to(device)
            x_global = batch['x_global'].to(device)
            x_oci = batch['x_oci'].to(device)
            y = batch['y'].numpy()  # (B, H, W)
            mask = batch['mask'].numpy()  # (B, H, W)

            # 前向传播
            logits = model(x_local, x_global, x_oci)  # (B, 3, H, W)
            probs = torch.softmax(logits, dim=1)[:, 1, :, :].cpu().numpy()  # (B, H, W) - fire class

            probs_list.append(probs)
            targets_list.append(y)
            mask_list.append(mask)

    return probs_list, targets_list, mask_list


def main(args):
    from dataset import SeasFirePatchDataset
    from model_multibranch_vit import create_model
    from torch.utils.data import DataLoader

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    print("加载模型...")
    model = create_model(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 创建验证集
    print("加载验证集...")
    val_dataset = SeasFirePatchDataset(
        zarr_path=config['data']['zarr_path'],
        target_zarr_path=config['data']['target_zarr_path'],
        years=config['data']['val_years'],
        fire_vars=config['data']['fire_vars'],
        log_transform_vars=config['data']['log_transform_vars'],
        oci_vars=config['data']['oci_vars'],
        target_var=config['data']['target_var'],
        lead_time_steps=config['data']['lead_time_steps'],
        oci_window=config['data']['oci_window'],
        temporal_steps=config['data'].get('temporal_steps', 4),
        burn_threshold=config['data']['burn_threshold'],
        patch_size=config['data']['patch_size'],
        stride=config['data']['patch_size'],
        global_coarsen_factor=config['data']['global_coarsen_factor'],
        use_local=config['model']['use_local'],
        use_global=config['model']['use_global'],
        use_oci=config['model']['use_oci'],
        only_fire_patches=False,
        use_augmentation=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 收集预测
    print("\n收集验证集预测...")
    probs_list, targets_list, mask_list = collect_predictions(model, val_loader, device)

    # 搜索最优阈值
    print("\n搜索最优阈值...")
    best_threshold, results, fig = find_optimal_threshold(probs_list, targets_list, mask_list)

    # 保存结果
    output_dir = os.path.dirname(args.checkpoint)

    # 保存图表
    fig_path = os.path.join(output_dir, 'threshold_optimization.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {fig_path}")

    # 保存文本结果
    txt_path = os.path.join(output_dir, 'optimal_threshold.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Optimal Threshold: {best_threshold:.4f}\n\n")
        f.write("Threshold Comparison:\n")
        f.write(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        f.write(f"{'-'*60}\n")
        for r in results:
            f.write(f"{r['threshold']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}\n")

        f.write(f"\n\nUsage in inference.py:\n")
        f.write(f"# 修改第416行的threshold参数:\n")
        f.write(f"metrics = compute_metrics(\n")
        f.write(f"    fire_probs[l:l+1],\n")
        f.write(f"    target[l:l+1],\n")
        f.write(f"    mask=ndvi_mask[np.newaxis, :, :],\n")
        f.write(f"    threshold={best_threshold:.4f}  # 使用最优阈值\n")
        f.write(f")\n")

    print(f"结果已保存到: {txt_path}")

    print(f"\n{'='*60}")
    print("建议:")
    print(f"{'='*60}")
    if best_threshold > 0.55:
        print(f"✓ 最优阈值 ({best_threshold:.3f}) > 0.5，说明模型预测确实偏激进")
        print(f"  建议: 在推理时使用阈值 {best_threshold:.3f} 而不是 0.5")
    elif best_threshold < 0.45:
        print(f"✓ 最优阈值 ({best_threshold:.3f}) < 0.5，说明模型预测偏保守")
        print(f"  建议: 在推理时使用阈值 {best_threshold:.3f} 而不是 0.5")
    else:
        print(f"✓ 最优阈值 ({best_threshold:.3f}) ≈ 0.5，模型校准良好")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='优化分类阈值')
    parser.add_argument('--config', type=str,
                        default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoint/best_model_L1_G1_I1_epoch7.pth',
                        help='模型checkpoint路径')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批处理大小')

    args = parser.parse_args()
    main(args)
