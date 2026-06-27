"""
可视化脚本：可视化推理结果（椭圆世界地图版本）

【修改说明】
1. 使用公开版相对路径读取推理输出
2. 新增椭圆世界地图可视化功能（Robinson/Mollweide投影）
3. 从 predictions/ 和 ground_truth/ 文件夹按日期匹配加载数据
4. 2行1列布局：上方预测，下方真值，右侧共享竖直colorbar
5. 使用scatter绘制mask==0的像素点

支持：
1. 椭圆世界地图可视化（新增）
2. 单张预测图可视化（预测概率 + 真值 + 对比）
3. 时间序列动画
4. 全球热力图
5. 指标曲线图
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Optional, Tuple
import seaborn as sns
from tqdm import tqdm

# 【新增】导入cartopy用于地图投影
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("⚠️  警告: cartopy未安装，椭圆世界地图功能不可用")
    print("   安装方法: conda install -c conda-forge cartopy")


def setup_plot_style():
    """设置绘图风格"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8


def create_fire_colormap():
    """
    创建火灾概率的colormap（专业分段配色）

    配色方案（基于概率分段）：
    - Under (< 0.15): 深紫色 #5E4FA2
    - 0.15-0.30: 蓝色 #3387BC
    - 0.30-0.45: 绿色 #AADCA4
    - 0.45-0.60: 浅黄色 #FFFEBE
    - 0.60-0.75: 橙色 #FDAD60
    - 0.75-0.90: 红色 #D43D4F
    - Over (> 0.90): 深红色 #9E0142
    """
    colors = [
        '#5E4FA2',  # 深紫色（< 0.15）
        '#3387BC',  # 蓝色（0.15-0.30）
        '#AADCA4',  # 绿色（0.30-0.45）
        '#FFFEBE',  # 浅黄色（0.45-0.60）
        '#FDAD60',  # 橙色（0.60-0.75）
        '#D43D4F',  # 红色（0.75-0.90）
        '#9E0142',  # 深红色（> 0.90）
    ]
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('fire_prob', colors, N=n_bins)
    return cmap


def get_lat_lon_grids(H: int = 720, W: int = 1440):
    """
    获取经纬度网格（0.25°分辨率）

    Args:
        H: 纬度维度（默认720，对应-90到90度）
        W: 经度维度（默认1440，对应-180到180度）

    Returns:
        lats: (H,) 纬度数组
        lons: (W,) 经度数组
        lon_grid: (H, W) 经度网格
        lat_grid: (H, W) 纬度网格
    """
    lats = np.linspace(90, -90, H)  # 从北到南
    lons = np.linspace(-180, 180, W)  # 从西到东
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lats, lons, lon_grid, lat_grid


def visualize_elliptical_world_map(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    date_str: str,
    lead_time: int,
    output_path: str,
    projection: str = 'Robinson',
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    【新增】可视化椭圆世界地图（2行1列布局，右侧竖直colorbar）
    【修改】显示所有mask==0区域的概率值，不使用阈值过滤

    Args:
        prediction: (H, W) 预测概率（0-1连续值）
        target: (H, W) 真值标签 (0/1)
        mask: (H, W) 掩码 (1=掩码, 0=有效)
        date_str: 日期字符串（如'20190406'）
        lead_time: 预测时间步长
        output_path: 输出路径
        projection: 投影类型 ('Robinson' 或 'Mollweide')
        vmin, vmax: colorbar范围
    """
    if not CARTOPY_AVAILABLE:
        print("⚠️  跳过椭圆世界地图可视化（cartopy未安装）")
        return

    # 获取经纬度网格
    H, W = prediction.shape
    lats, lons, lon_grid, lat_grid = get_lat_lon_grids(H, W)

    # 选择投影
    if projection.lower() == 'mollweide':
        proj = ccrs.Mollweide()
    else:
        proj = ccrs.Robinson()

    # 创建图形：2行1列，右侧留空间给colorbar
    # 【修改】设置白色背景
    fig = plt.figure(figsize=(16, 10), facecolor='white')

    # 创建GridSpec布局：只创建地图部分（2行1列）
    # 【修改】colorbar将手动添加到指定位置
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 1, figure=fig, hspace=0.15)

    # 上方子图：预测
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ax1.set_global()
    ax1.coastlines(linewidth=0.5, color='gray', alpha=0.7)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', alpha=0.5)
    ax1.set_facecolor('white')  # 【修改】设置地图背景为白色
    # 【新增】添加椭圆边界
    ax1.spines['geo'].set_edgecolor('black')
    ax1.spines['geo'].set_linewidth(1.5)

    # 下方子图：真值
    ax2 = fig.add_subplot(gs[1, 0], projection=proj)
    ax2.set_global()
    ax2.coastlines(linewidth=0.5, color='gray', alpha=0.7)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', alpha=0.5)
    ax2.set_facecolor('white')  # 【修改】设置地图背景为白色
    # 【新增】添加椭圆边界
    ax2.spines['geo'].set_edgecolor('black')
    ax2.spines['geo'].set_linewidth(1.5)

    # ========== 上方：预测概率（网格填充图）==========
    # 【修改】使用pcolormesh绘制网格填充图，而不是scatter点图
    # 将mask区域和概率<0.01的区域设置为NaN，显示为背景色
    prediction_masked = prediction.copy().astype(np.float32)  # 确保是浮点类型
    prediction_masked[mask == 1] = np.nan  # mask区域设为NaN
    prediction_masked[prediction_masked < 0.01] = np.nan  # 概率<0.01的区域设为NaN（显示为背景色）
    prediction_masked = np.ma.masked_invalid(prediction_masked)  # 创建masked array

    fire_cmap = create_fire_colormap()

    # 使用pcolormesh绘制网格填充图
    # 注意：pcolormesh需要网格边界，所以需要计算边界坐标
    # 对于0.25°网格，边界在每个网格中心±0.125°
    lat_edges = np.linspace(90 + 0.125, -90 - 0.125, H + 1)
    lon_edges = np.linspace(-180 - 0.125, 180 + 0.125, W + 1)

    mesh1 = ax1.pcolormesh(
        lon_edges, lat_edges,
        prediction_masked,
        cmap=fire_cmap,
        vmin=0.01,  # 【修改】colorbar最小值设为0.01，与阈值一致
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading='flat',  # 使用flat shading，每个网格单元一个颜色
        rasterized=True  # 加速渲染
    )

    ax1.set_title(f'Forecast Probability (Lead={lead_time}d, Date={date_str})',
                  fontsize=14, fontweight='bold', pad=10)

    # ========== 下方：真值（网格填充图）==========
    # 【修改】使用pcolormesh绘制网格填充图，显示0/1的真值
    # 将mask区域和target为0的区域设置为NaN，显示为背景色
    target_masked = target.copy().astype(np.float32)  # 转换为浮点类型，才能赋值NaN
    target_masked[mask == 1] = np.nan  # mask区域设为NaN
    target_masked[target_masked == 0] = np.nan  # target为0的区域也设为NaN（显示为背景色）
    target_masked = np.ma.masked_invalid(target_masked)  # 创建masked array

    mesh2 = ax2.pcolormesh(
        lon_edges, lat_edges,
        target_masked,
        cmap='Reds',  # 使用红色系colormap，只显示火点
        vmin=0,
        vmax=1,
        transform=ccrs.PlateCarree(),
        shading='flat',
        rasterized=True
    )

    ax2.set_title(f'Ground Truth (Date={date_str})',
                  fontsize=14, fontweight='bold', pad=10)

    # ========== 右侧竖直colorbar（手动定位）==========
    # 【修改】手动创建colorbar axes：缩短长度并调整位置
    # Figure坐标系：[left, bottom, width, height]，范围0-1
    # left=0.762: 向右平移1/20图宽（0.712 + 0.05 = 0.762）
    # bottom=0.28: 垂直居中，使colorbar占原长度的2/3
    # width=0.012: 更细的宽度
    # height=0.44: 原长度的2/3（0.66 × 2/3 = 0.44）
    cbar_ax = fig.add_axes([0.762, 0.28, 0.012, 0.44])
    cbar = plt.colorbar(mesh1, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Fire Probability', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # 保存
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def visualize_single_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    time_idx: int,
    lead_time: int,
    output_path: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    可视化单张预测结果（矩形地图版本）

    Args:
        prediction: (H, W) 预测概率
        target: (H, W) 真值标签 (0/1)
        mask: (H, W) 掩码 (1=掩码, 0=有效)
        time_idx: 时间索引
        lead_time: 预测时间步长
        output_path: 输出路径
        vmin, vmax: colorbar范围
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 应用掩码
    prediction_masked = np.ma.masked_where(mask == 1, prediction)
    target_masked = np.ma.masked_where(mask == 1, target)

    # 创建colormap
    fire_cmap = create_fire_colormap()

    # 1. 预测概率
    im1 = axes[0].imshow(prediction_masked, cmap=fire_cmap, vmin=vmin, vmax=vmax,
                         interpolation='nearest', aspect='auto')
    axes[0].set_title(f'Prediction (Lead={lead_time}d, Time={time_idx})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], label='Fire Probability', fraction=0.046, pad=0.04)

    # 2. 真值标签
    im2 = axes[1].imshow(target_masked, cmap='RdYlBu_r', vmin=0, vmax=1,
                         interpolation='nearest', aspect='auto')
    axes[1].set_title(f'Ground Truth (Lead={lead_time}d)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1], label='Fire (0=No, 1=Yes)', fraction=0.046, pad=0.04)

    # 3. 差异图（预测 - 真值）
    diff = prediction_masked - target_masked
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1,
                         interpolation='nearest', aspect='auto')
    axes[2].set_title(f'Difference (Pred - GT)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[2], label='Difference', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def visualize_comparison_grid(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    time_indices: np.ndarray,
    lead_times: List[int],
    output_path: str,
    num_samples: int = 4,
):
    """
    可视化多个时间步和lead times的对比网格

    Args:
        predictions: (N, L, H, W) 预测概率
        targets: (N, L, H, W) 真值标签
        masks: (N, H, W) 掩码
        time_indices: (N,) 时间索引
        lead_times: lead time列表
        output_path: 输出路径
        num_samples: 采样数量
    """
    N, L, H, W = predictions.shape

    # 采样时间步
    if N > num_samples:
        sample_indices = np.linspace(0, N - 1, num_samples, dtype=int)
    else:
        sample_indices = np.arange(N)
        num_samples = N

    # 创建网格：行=时间步，列=lead times
    fig, axes = plt.subplots(num_samples, L, figsize=(5*L, 4*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if L == 1:
        axes = axes.reshape(-1, 1)

    fire_cmap = create_fire_colormap()

    for i, sample_idx in enumerate(sample_indices):
        for j, lead_time in enumerate(lead_times):
            pred = predictions[sample_idx, j]
            target = targets[sample_idx, j]
            mask = masks[sample_idx]
            time_idx = time_indices[sample_idx]

            # 应用掩码
            pred_masked = np.ma.masked_where(mask == 1, pred)

            # 叠加真值边界
            ax = axes[i, j]
            im = ax.imshow(pred_masked, cmap=fire_cmap, vmin=0, vmax=1,
                          interpolation='nearest', aspect='auto')

            # 绘制真值火点（红色轮廓）
            target_masked = np.ma.masked_where(mask == 1, target)
            ax.contour(target_masked, levels=[0.5], colors='red', linewidths=1.5, alpha=0.8)

            ax.set_title(f'T={time_idx}, Lead={lead_time}d', fontsize=10)
            ax.axis('off')

            # 只在最右侧添加colorbar
            if j == L - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Prediction vs Ground Truth (Red Contour)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def create_animation(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    time_indices: np.ndarray,
    lead_time_idx: int,
    lead_time: int,
    output_path: str,
    fps: int = 2,
):
    """
    创建时间序列动画

    Args:
        predictions: (N, L, H, W) 预测概率
        targets: (N, L, H, W) 真值标签
        masks: (N, H, W) 掩码
        time_indices: (N,) 时间索引
        lead_time_idx: lead time索引
        lead_time: lead time值
        output_path: 输出路径（.gif）
        fps: 帧率
    """
    N = predictions.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fire_cmap = create_fire_colormap()

    def update(frame):
        for ax in axes:
            ax.clear()

        pred = predictions[frame, lead_time_idx]
        target = targets[frame, lead_time_idx]
        mask = masks[frame]
        time_idx = time_indices[frame]

        # 应用掩码
        pred_masked = np.ma.masked_where(mask == 1, pred)
        target_masked = np.ma.masked_where(mask == 1, target)

        # 预测
        im1 = axes[0].imshow(pred_masked, cmap=fire_cmap, vmin=0, vmax=1,
                            interpolation='nearest', aspect='auto')
        axes[0].set_title(f'Prediction (Time={time_idx}, Lead={lead_time}d)', fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # 真值
        im2 = axes[1].imshow(target_masked, cmap='RdYlBu_r', vmin=0, vmax=1,
                            interpolation='nearest', aspect='auto')
        axes[1].set_title(f'Ground Truth (Time={time_idx}, Lead={lead_time}d)', fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        plt.suptitle(f'Frame {frame+1}/{N}', fontsize=12)

    anim = FuncAnimation(fig, update, frames=N, interval=1000//fps, repeat=True)

    # 保存为GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"✓ 动画已保存: {output_path}")


def plot_metrics_curves(
    metrics_csv_path: str,
    lead_times: List[int],
    output_dir: str,
):
    """
    绘制指标曲线图

    Args:
        metrics_csv_path: 单天指标CSV路径
        lead_times: lead time列表
        output_dir: 输出目录
    """
    df = pd.read_csv(metrics_csv_path)

    metrics = ['auprc', 'auroc', 'f1', 'precision', 'recall']
    metric_names = ['AUPRC', 'AUROC', 'F1 Score', 'Precision', 'Recall']

    # 1. 每个指标随时间变化（不同lead times）
    for metric, metric_name in zip(metrics, metric_names):
        fig, ax = plt.subplots(figsize=(12, 6))

        for lt in lead_times:
            col_name = f'lead{lt}_{metric}'
            if col_name in df.columns:
                ax.plot(df['time_idx'], df[col_name], marker='o', markersize=3,
                       label=f'Lead={lt}d', alpha=0.8)

        ax.set_xlabel('Time Index', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} over Time (Different Lead Times)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        output_path = os.path.join(output_dir, f'{metric}_over_time.png')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"✓ 已保存: {output_path}")

    # 2. 所有指标对比（不同lead times）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        for lt in lead_times:
            col_name = f'lead{lt}_{metric}'
            if col_name in df.columns:
                ax.plot(df['time_idx'], df[col_name], marker='o', markersize=2,
                       label=f'Lead={lt}d', alpha=0.7)

        ax.set_xlabel('Time Index', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('All Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'all_metrics_comparison.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ 已保存: {output_path}")


def plot_average_metrics_bar(
    avg_metrics_csv_path: str,
    output_path: str,
):
    """
    绘制平均指标柱状图

    Args:
        avg_metrics_csv_path: 平均指标CSV路径
        output_path: 输出路径
    """
    df = pd.read_csv(avg_metrics_csv_path)

    metrics = ['auprc_mean', 'auroc_mean', 'f1_mean', 'precision_mean', 'recall_mean']
    metric_names = ['AUPRC', 'AUROC', 'F1', 'Precision', 'Recall']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.15

    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = df[metric].values
        errors = df[metric.replace('_mean', '_std')].values
        ax.bar(x + i*width, values, width, label=metric_name, yerr=errors,
               capsize=5, alpha=0.8)

    ax.set_xlabel('Lead Time (days)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Average Metrics by Lead Time', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"{int(lt)}d" for lt in df['lead_time']])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ 已保存: {output_path}")


def main(args):
    """
    【修改】主函数：支持从predictions/和ground_truth/文件夹加载数据
    """
    setup_plot_style()

    print(f"\n{'='*80}")
    print("加载推理结果")
    print(f"{'='*80}")

    # 【新增】椭圆世界地图可视化（从单独的文件夹加载）
    if args.visualize_elliptical:
        print(f"\n{'='*80}")
        print("生成椭圆世界地图可视化")
        print(f"{'='*80}")

        # 检查cartopy是否可用
        if not CARTOPY_AVAILABLE:
            print("⚠️  跳过椭圆世界地图可视化（cartopy未安装）")
            print("   安装方法: conda install -c conda-forge cartopy")
        else:
            # 获取predictions和ground_truth文件夹路径
            predictions_dir = os.path.join(args.input_dir, 'predictions')
            ground_truth_dir = os.path.join(args.input_dir, 'ground_truth')
            masks_dir = os.path.join(args.input_dir, 'masks')

            # 检查文件夹是否存在
            if not os.path.exists(predictions_dir):
                print(f"⚠️  未找到predictions文件夹: {predictions_dir}")
            elif not os.path.exists(ground_truth_dir):
                print(f"⚠️  未找到ground_truth文件夹: {ground_truth_dir}")
            elif not os.path.exists(masks_dir):
                print(f"⚠️  未找到masks文件夹: {masks_dir}")
            else:
                # 获取所有预测文件
                pred_files = sorted(glob.glob(os.path.join(predictions_dir, 'prediction_*.npy')))
                print(f"找到 {len(pred_files)} 个预测文件")

                # 创建输出目录
                elliptical_dir = os.path.join(args.output_dir, 'elliptical_maps')
                os.makedirs(elliptical_dir, exist_ok=True)

                # 【修改】处理所有测试集文件（2018和2019两年的所有数据）
                # 如果用户指定了采样数量且>0，则采样；否则处理全部
                if args.num_elliptical_samples > 0 and len(pred_files) > args.num_elliptical_samples:
                    sample_indices = np.linspace(0, len(pred_files) - 1, args.num_elliptical_samples, dtype=int)
                    pred_files = [pred_files[i] for i in sample_indices]
                    print(f"⚠️  采样模式：只处理 {len(pred_files)} 个文件（共{len(glob.glob(os.path.join(predictions_dir, 'prediction_*.npy')))}个）")
                    print(f"   如需处理全部测试集，请使用: --num-elliptical-samples -1")
                else:
                    print(f"✓ 处理全部测试集：共 {len(pred_files)} 个时间步（2018-2019年）")

                # 逐个文件生成椭圆世界地图
                for pred_file in tqdm(pred_files, desc="生成椭圆世界地图"):
                    # 从文件名提取日期
                    basename = os.path.basename(pred_file)
                    date_str = basename.replace('prediction_', '').replace('.npy', '')

                    # 构建对应的ground_truth和mask文件路径
                    gt_file = os.path.join(ground_truth_dir, f'ground_truth_{date_str}.npy')
                    mask_file = os.path.join(masks_dir, f'mask_{date_str}.npy')

                    # 检查文件是否存在
                    if not os.path.exists(gt_file):
                        print(f"⚠️  未找到对应的ground_truth文件: {gt_file}")
                        continue
                    if not os.path.exists(mask_file):
                        print(f"⚠️  未找到对应的mask文件: {mask_file}")
                        continue

                    # 加载数据
                    prediction = np.load(pred_file)  # (H, W) 或 (L, H, W)
                    target = np.load(gt_file)  # (H, W) 或 (L, H, W)
                    mask = np.load(mask_file)  # (H, W)

                    # 处理多lead time情况（取第一个lead time）
                    if prediction.ndim == 3:
                        prediction = prediction[0]  # 取第一个lead time
                        target = target[0]
                        print(f"  检测到多lead time数据，使用第一个lead time")

                    # 生成椭圆世界地图
                    output_path = os.path.join(elliptical_dir, f'elliptical_map_{date_str}.png')
                    visualize_elliptical_world_map(
                        prediction=prediction,
                        target=target,
                        mask=mask,
                        date_str=date_str,
                        lead_time=args.lead_time_value,
                        output_path=output_path,
                        projection=args.projection,
                        vmin=args.vmin,
                        vmax=args.vmax,
                    )

                print(f"✓ 已生成 {len(pred_files)} 张椭圆世界地图")
                print(f"✓ 输出目录: {elliptical_dir}")

        # 如果只生成椭圆地图，直接返回
        if args.only_elliptical:
            print(f"\n{'='*80}")
            print("✓ 可视化完成！")
            print(f"{'='*80}")
            return

    # 【原有功能】加载合并的npy文件（用于其他可视化）
    # 检查是否存在旧格式的合并文件
    predictions_file = os.path.join(args.input_dir, 'predictions.npy')
    targets_file = os.path.join(args.input_dir, 'targets.npy')
    masks_file = os.path.join(args.input_dir, 'masks.npy')
    time_indices_file = os.path.join(args.input_dir, 'time_indices.npy')

    if not os.path.exists(predictions_file):
        print(f"⚠️  未找到合并的predictions.npy文件")
        print(f"   如果只需要椭圆世界地图，请使用 --visualize-elliptical --only-elliptical 参数")
        print(f"   其他可视化功能需要合并的npy文件")
        return

    # 加载数据
    predictions = np.load(predictions_file)
    targets = np.load(targets_file)
    masks = np.load(masks_file)
    time_indices = np.load(time_indices_file)

    # 处理维度：确保是4D (N, L, H, W)
    if predictions.ndim == 3:
        predictions = predictions[:, np.newaxis, :, :]  # (N, H, W) -> (N, 1, H, W)
        targets = targets[:, np.newaxis, :, :]
        print(f"  检测到单lead time数据，已扩展维度")

    N, L, H, W = predictions.shape

    print(f"数据维度:")
    print(f"  - predictions: {predictions.shape} (N, L, H, W)")
    print(f"  - targets: {targets.shape}")
    print(f"  - masks: {masks.shape}")
    print(f"  - time_indices: {time_indices.shape}")
    print(f"\n时间步数: {N}")
    print(f"Lead times数量: {L}")
    print(f"空间尺寸: {H} × {W}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 推断lead times
    lead_times = list(range(1, L+1))  # 默认 [1, 2, ..., L]
    if args.lead_times:
        lead_times = [int(x) for x in args.lead_times.split(',')]

    print(f"Lead times: {lead_times}")

    # 1. 可视化单张预测图（采样）
    if args.visualize_samples:
        print(f"\n{'='*80}")
        print("生成单张预测图")
        print(f"{'='*80}")

        sample_dir = os.path.join(args.output_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)

        num_samples = min(args.num_samples, N)
        sample_indices = np.linspace(0, N - 1, num_samples, dtype=int)

        for sample_idx in tqdm(sample_indices, desc="生成单张预测图"):
            for l, lead_time in enumerate(lead_times):
                output_path = os.path.join(sample_dir,
                    f'prediction_t{time_indices[sample_idx]}_lead{lead_time}.png')

                visualize_single_prediction(
                    prediction=predictions[sample_idx, l],
                    target=targets[sample_idx, l],
                    mask=masks[sample_idx],
                    time_idx=time_indices[sample_idx],
                    lead_time=lead_time,
                    output_path=output_path,
                )

        print(f"✓ 已生成 {num_samples * L} 张预测图")

    # 2. 可视化对比网格
    if args.visualize_grid:
        print(f"\n{'='*80}")
        print("生成对比网格图")
        print(f"{'='*80}")

        output_path = os.path.join(args.output_dir, 'comparison_grid.png')
        visualize_comparison_grid(
            predictions=predictions,
            targets=targets,
            masks=masks,
            time_indices=time_indices,
            lead_times=lead_times,
            output_path=output_path,
            num_samples=args.grid_samples,
        )

        print(f"✓ 已保存: {output_path}")

    # 3. 创建动画
    if args.create_animation:
        print(f"\n{'='*80}")
        print("创建时间序列动画")
        print(f"{'='*80}")

        anim_dir = os.path.join(args.output_dir, 'animations')
        os.makedirs(anim_dir, exist_ok=True)

        for l, lead_time in enumerate(lead_times):
            output_path = os.path.join(anim_dir, f'animation_lead{lead_time}.gif')
            create_animation(
                predictions=predictions,
                targets=targets,
                masks=masks,
                time_indices=time_indices,
                lead_time_idx=l,
                lead_time=lead_time,
                output_path=output_path,
                fps=args.fps,
            )

    # 4. 绘制指标曲线
    if args.plot_metrics:
        print(f"\n{'='*80}")
        print("绘制指标曲线")
        print(f"{'='*80}")

        metrics_csv = os.path.join(args.input_dir, 'metrics_per_day.csv')
        if os.path.exists(metrics_csv):
            metrics_dir = os.path.join(args.output_dir, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)

            plot_metrics_curves(
                metrics_csv_path=metrics_csv,
                lead_times=lead_times,
                output_dir=metrics_dir,
            )
        else:
            print(f"⚠️  未找到指标文件: {metrics_csv}")

    # 5. 绘制平均指标柱状图
    if args.plot_average:
        print(f"\n{'='*80}")
        print("绘制平均指标柱状图")
        print(f"{'='*80}")

        avg_csv = os.path.join(args.input_dir, 'metrics_average.csv')
        if os.path.exists(avg_csv):
            output_path = os.path.join(args.output_dir, 'average_metrics_bar.png')
            plot_average_metrics_bar(
                avg_metrics_csv_path=avg_csv,
                output_path=output_path,
            )
        else:
            print(f"⚠️  未找到平均指标文件: {avg_csv}")

    print(f"\n{'='*80}")
    print("✓ 可视化完成！")
    print(f"{'='*80}")
    print(f"输出目录: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化推理结果（支持椭圆世界地图）')

    # 【修改】更新默认路径到0654项目
    parser.add_argument('--input-dir', type=str,
                        default='inference_results',
                        help='推理结果目录（包含predictions/、ground_truth/、masks/文件夹或npy文件）')
    parser.add_argument('--output-dir', type=str,
                        default='visualizations',
                        help='可视化输出目录')
    parser.add_argument('--lead-times', type=str, default=1,
                        help='Lead times（逗号分隔，如"1,2,4,8,16"）')

    # 【新增】椭圆世界地图参数
    parser.add_argument('--visualize-elliptical', action='store_true', default=True,
                        help='生成椭圆世界地图可视化（默认启用）')
    parser.add_argument('--only-elliptical', action='store_true', default=False,
                        help='只生成椭圆世界地图，跳过其他可视化')
    parser.add_argument('--projection', type=str, default='Robinson',
                        choices=['Robinson', 'Mollweide'],
                        help='地图投影类型（默认Robinson）')
    # 【修改】移除threshold参数，因为不再使用阈值过滤
    parser.add_argument('--lead-time-value', type=int, default=1,
                        help='椭圆地图标题中显示的lead time值（默认1天）')
    # 【修改】默认值改为-1，表示处理全部测试集（2018-2019年所有数据）
    parser.add_argument('--num-elliptical-samples', type=int, default=-1,
                        help='椭圆地图采样数量（-1表示全部，默认-1处理所有测试集）')
    parser.add_argument('--vmin', type=float, default=0.0,
                        help='Colorbar最小值（默认0.0）')
    parser.add_argument('--vmax', type=float, default=1.0,
                        help='Colorbar最大值（默认1.0）')

    # 【原有】其他可视化选项
    parser.add_argument('--visualize-samples', action='store_true', default=False,
                        help='生成单张预测图（矩形地图）')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='采样数量（单张预测图）')
    parser.add_argument('--visualize-grid', action='store_true', default=False,
                        help='生成对比网格图')
    parser.add_argument('--grid-samples', type=int, default=4,
                        help='网格图采样数量')
    parser.add_argument('--create-animation', action='store_true', default=False,
                        help='创建时间序列动画（GIF）')
    parser.add_argument('--fps', type=int, default=2,
                        help='动画帧率')
    parser.add_argument('--plot-metrics', action='store_true', default=False,
                        help='绘制指标曲线')
    parser.add_argument('--plot-average', action='store_true', default=False,
                        help='绘制平均指标柱状图')

    args = parser.parse_args()
    main(args)
