"""Brief implementation note."""
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


try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Status")
    print("Status")


def setup_plot_style():
    """Brief implementation note."""
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
    """Brief implementation note."""
    colors = [
        '#5E4FA2',  
        '#3387BC',  
        '#AADCA4',  
        '#FFFEBE',  
        '#FDAD60',  
        '#D43D4F',  
        '#9E0142',  
    ]
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('fire_prob', colors, N=n_bins)
    return cmap


def get_lat_lon_grids(H: int = 720, W: int = 1440):
    """Brief implementation note."""
    lats = np.linspace(90, -90, H)  
    lons = np.linspace(-180, 180, W)  
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
    """Brief implementation note."""
    if not CARTOPY_AVAILABLE:
        print("Status")
        return

    
    H, W = prediction.shape
    lats, lons, lon_grid, lat_grid = get_lat_lon_grids(H, W)

    
    if projection.lower() == 'mollweide':
        proj = ccrs.Mollweide()
    else:
        proj = ccrs.Robinson()

    
    
    fig = plt.figure(figsize=(16, 10), facecolor='white')

    
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 1, figure=fig, hspace=0.15)

    
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ax1.set_global()
    ax1.coastlines(linewidth=0.5, color='gray', alpha=0.7)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', alpha=0.5)
    ax1.set_facecolor('white')  
    
    ax1.spines['geo'].set_edgecolor('black')
    ax1.spines['geo'].set_linewidth(1.5)

    
    ax2 = fig.add_subplot(gs[1, 0], projection=proj)
    ax2.set_global()
    ax2.coastlines(linewidth=0.5, color='gray', alpha=0.7)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', alpha=0.5)
    ax2.set_facecolor('white')  
    
    ax2.spines['geo'].set_edgecolor('black')
    ax2.spines['geo'].set_linewidth(1.5)

    
    
    
    prediction_masked = prediction.copy().astype(np.float32)  
    prediction_masked[mask == 1] = np.nan  
    prediction_masked[prediction_masked < 0.01] = np.nan  
    prediction_masked = np.ma.masked_invalid(prediction_masked)  

    fire_cmap = create_fire_colormap()

    
    
    
    lat_edges = np.linspace(90 + 0.125, -90 - 0.125, H + 1)
    lon_edges = np.linspace(-180 - 0.125, 180 + 0.125, W + 1)

    mesh1 = ax1.pcolormesh(
        lon_edges, lat_edges,
        prediction_masked,
        cmap=fire_cmap,
        vmin=0.01,  
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading='flat',  
        rasterized=True  
    )

    ax1.set_title(f'Forecast Probability (Lead={lead_time}d, Date={date_str})',
                  fontsize=14, fontweight='bold', pad=10)

    
    
    
    target_masked = target.copy().astype(np.float32)  
    target_masked[mask == 1] = np.nan  
    target_masked[target_masked == 0] = np.nan  
    target_masked = np.ma.masked_invalid(target_masked)  

    mesh2 = ax2.pcolormesh(
        lon_edges, lat_edges,
        target_masked,
        cmap='Reds',  
        vmin=0,
        vmax=1,
        transform=ccrs.PlateCarree(),
        shading='flat',
        rasterized=True
    )

    ax2.set_title(f'Ground Truth (Date={date_str})',
                  fontsize=14, fontweight='bold', pad=10)

    
    
    
    
    
    
    
    cbar_ax = fig.add_axes([0.762, 0.28, 0.012, 0.44])
    cbar = plt.colorbar(mesh1, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Fire Probability', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    
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
    """Brief implementation note."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    
    prediction_masked = np.ma.masked_where(mask == 1, prediction)
    target_masked = np.ma.masked_where(mask == 1, target)

    
    fire_cmap = create_fire_colormap()

    
    im1 = axes[0].imshow(prediction_masked, cmap=fire_cmap, vmin=vmin, vmax=vmax,
                         interpolation='nearest', aspect='auto')
    axes[0].set_title(f'Prediction (Lead={lead_time}d, Time={time_idx})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], label='Fire Probability', fraction=0.046, pad=0.04)

    
    im2 = axes[1].imshow(target_masked, cmap='RdYlBu_r', vmin=0, vmax=1,
                         interpolation='nearest', aspect='auto')
    axes[1].set_title(f'Ground Truth (Lead={lead_time}d)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1], label='Fire (0=No, 1=Yes)', fraction=0.046, pad=0.04)

    
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
    """Brief implementation note."""
    N, L, H, W = predictions.shape

    
    if N > num_samples:
        sample_indices = np.linspace(0, N - 1, num_samples, dtype=int)
    else:
        sample_indices = np.arange(N)
        num_samples = N

    
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

            
            pred_masked = np.ma.masked_where(mask == 1, pred)

            
            ax = axes[i, j]
            im = ax.imshow(pred_masked, cmap=fire_cmap, vmin=0, vmax=1,
                          interpolation='nearest', aspect='auto')

            
            target_masked = np.ma.masked_where(mask == 1, target)
            ax.contour(target_masked, levels=[0.5], colors='red', linewidths=1.5, alpha=0.8)

            ax.set_title(f'T={time_idx}, Lead={lead_time}d', fontsize=10)
            ax.axis('off')

            
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
    """Brief implementation note."""
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

        
        pred_masked = np.ma.masked_where(mask == 1, pred)
        target_masked = np.ma.masked_where(mask == 1, target)

        
        im1 = axes[0].imshow(pred_masked, cmap=fire_cmap, vmin=0, vmax=1,
                            interpolation='nearest', aspect='auto')
        axes[0].set_title(f'Prediction (Time={time_idx}, Lead={lead_time}d)', fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        
        im2 = axes[1].imshow(target_masked, cmap='RdYlBu_r', vmin=0, vmax=1,
                            interpolation='nearest', aspect='auto')
        axes[1].set_title(f'Ground Truth (Time={time_idx}, Lead={lead_time}d)', fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        plt.suptitle(f'Frame {frame+1}/{N}', fontsize=12)

    anim = FuncAnimation(fig, update, frames=N, interval=1000//fps, repeat=True)

    
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"Status")


def plot_metrics_curves(
    metrics_csv_path: str,
    lead_times: List[int],
    output_dir: str,
):
    """Brief implementation note."""
    df = pd.read_csv(metrics_csv_path)

    metrics = ['auprc', 'auroc', 'f1', 'precision', 'recall']
    metric_names = ['AUPRC', 'AUROC', 'F1 Score', 'Precision', 'Recall']

    
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

        print(f"Status")

    
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

    
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('All Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'all_metrics_comparison.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Status")


def plot_average_metrics_bar(
    avg_metrics_csv_path: str,
    output_path: str,
):
    """Brief implementation note."""
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

    print(f"Status")


def main(args):
    """Brief implementation note."""
    setup_plot_style()

    print(f"\n{'='*80}")
    print("Status")
    print(f"{'='*80}")

    
    if args.visualize_elliptical:
        print(f"\n{'='*80}")
        print("Status")
        print(f"{'='*80}")

        
        if not CARTOPY_AVAILABLE:
            print("Status")
            print("Status")
        else:
            
            predictions_dir = os.path.join(args.input_dir, 'predictions')
            ground_truth_dir = os.path.join(args.input_dir, 'ground_truth')
            masks_dir = os.path.join(args.input_dir, 'masks')

            
            if not os.path.exists(predictions_dir):
                print(f"Status")
            elif not os.path.exists(ground_truth_dir):
                print(f"Status")
            elif not os.path.exists(masks_dir):
                print(f"Status")
            else:
                
                pred_files = sorted(glob.glob(os.path.join(predictions_dir, 'prediction_*.npy')))
                print(f"Status")

                
                elliptical_dir = os.path.join(args.output_dir, 'elliptical_maps')
                os.makedirs(elliptical_dir, exist_ok=True)

                
                
                if args.num_elliptical_samples > 0 and len(pred_files) > args.num_elliptical_samples:
                    sample_indices = np.linspace(0, len(pred_files) - 1, args.num_elliptical_samples, dtype=int)
                    pred_files = [pred_files[i] for i in sample_indices]
                    print(f"Status")
                    print(f"Status")
                else:
                    print(f"Status")

                
                for pred_file in tqdm(pred_files, desc="Status"):
                    
                    basename = os.path.basename(pred_file)
                    date_str = basename.replace('prediction_', '').replace('.npy', '')

                    
                    gt_file = os.path.join(ground_truth_dir, f'ground_truth_{date_str}.npy')
                    mask_file = os.path.join(masks_dir, f'mask_{date_str}.npy')

                    
                    if not os.path.exists(gt_file):
                        print(f"Status")
                        continue
                    if not os.path.exists(mask_file):
                        print(f"Status")
                        continue

                    
                    prediction = np.load(pred_file)  
                    target = np.load(gt_file)  
                    mask = np.load(mask_file)  # (H, W)

                    
                    if prediction.ndim == 3:
                        prediction = prediction[0]  
                        target = target[0]
                        print(f"Status")

                    
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

                print(f"Status")
                print(f"Status")

        
        if args.only_elliptical:
            print(f"\n{'='*80}")
            print("Status")
            print(f"{'='*80}")
            return

    
    
    predictions_file = os.path.join(args.input_dir, 'predictions.npy')
    targets_file = os.path.join(args.input_dir, 'targets.npy')
    masks_file = os.path.join(args.input_dir, 'masks.npy')
    time_indices_file = os.path.join(args.input_dir, 'time_indices.npy')

    if not os.path.exists(predictions_file):
        print(f"Status")
        print(f"Status")
        print(f"Status")
        return

    
    predictions = np.load(predictions_file)
    targets = np.load(targets_file)
    masks = np.load(masks_file)
    time_indices = np.load(time_indices_file)

    
    if predictions.ndim == 3:
        predictions = predictions[:, np.newaxis, :, :]  # (N, H, W) -> (N, 1, H, W)
        targets = targets[:, np.newaxis, :, :]
        print(f"Status")

    N, L, H, W = predictions.shape

    print(f"Status")
    print(f"  - predictions: {predictions.shape} (N, L, H, W)")
    print(f"  - targets: {targets.shape}")
    print(f"  - masks: {masks.shape}")
    print(f"  - time_indices: {time_indices.shape}")
    print(f"Status")
    print(f"Status")
    print(f"Status")

    
    os.makedirs(args.output_dir, exist_ok=True)

    
    lead_times = list(range(1, L+1))  
    if args.lead_times:
        lead_times = [int(x) for x in args.lead_times.split(',')]

    print(f"Lead times: {lead_times}")

    
    if args.visualize_samples:
        print(f"\n{'='*80}")
        print("Status")
        print(f"{'='*80}")

        sample_dir = os.path.join(args.output_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)

        num_samples = min(args.num_samples, N)
        sample_indices = np.linspace(0, N - 1, num_samples, dtype=int)

        for sample_idx in tqdm(sample_indices, desc="Status"):
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

        print(f"Status")

    
    if args.visualize_grid:
        print(f"\n{'='*80}")
        print("Status")
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

        print(f"Status")

    
    if args.create_animation:
        print(f"\n{'='*80}")
        print("Status")
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

    
    if args.plot_metrics:
        print(f"\n{'='*80}")
        print("Status")
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
            print(f"Status")

    
    if args.plot_average:
        print(f"\n{'='*80}")
        print("Status")
        print(f"{'='*80}")

        avg_csv = os.path.join(args.input_dir, 'metrics_average.csv')
        if os.path.exists(avg_csv):
            output_path = os.path.join(args.output_dir, 'average_metrics_bar.png')
            plot_average_metrics_bar(
                avg_metrics_csv_path=avg_csv,
                output_path=output_path,
            )
        else:
            print(f"Status")

    print(f"\n{'='*80}")
    print("Status")
    print(f"{'='*80}")
    print(f"Status")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Status")

    
    parser.add_argument('--input-dir', type=str,
                        default='/data1/ljt/TeleVIT/muti-teleVit-transformer/teleVIT-0654-overlap-csjxl-codex/inference_results',
                        help="Status")
    parser.add_argument('--output-dir', type=str,
                        default='/data1/ljt/TeleVIT/muti-teleVit-transformer/teleVIT-0654-overlap-csjxl-codex/visualizations',
                        help="Status")
    parser.add_argument('--lead-times', type=str, default=1,
                        help="Status")

    
    parser.add_argument('--visualize-elliptical', action='store_true', default=True,
                        help="Status")
    parser.add_argument('--only-elliptical', action='store_true', default=False,
                        help="Status")
    parser.add_argument('--projection', type=str, default='Robinson',
                        choices=['Robinson', 'Mollweide'],
                        help="Status")
    
    parser.add_argument('--lead-time-value', type=int, default=1,
                        help="Status")
    
    parser.add_argument('--num-elliptical-samples', type=int, default=-1,
                        help="Status")
    parser.add_argument('--vmin', type=float, default=0.0,
                        help="Status")
    parser.add_argument('--vmax', type=float, default=1.0,
                        help="Status")

    
    parser.add_argument('--visualize-samples', action='store_true', default=False,
                        help="Status")
    parser.add_argument('--num-samples', type=int, default=10,
                        help="Status")
    parser.add_argument('--visualize-grid', action='store_true', default=False,
                        help="Status")
    parser.add_argument('--grid-samples', type=int, default=4,
                        help="Status")
    parser.add_argument('--create-animation', action='store_true', default=False,
                        help="Status")
    parser.add_argument('--fps', type=int, default=2,
                        help="Status")
    parser.add_argument('--plot-metrics', action='store_true', default=False,
                        help="Status")
    parser.add_argument('--plot-average', action='store_true', default=False,
                        help="Status")

    args = parser.parse_args()
    main(args)

