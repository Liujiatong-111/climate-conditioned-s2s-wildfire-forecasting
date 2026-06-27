"""
可视化NDVI掩码文件
显示哪些区域被掩码（海洋、沙漠、高纬度等）
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 加载mask文件
mask_path = 'inference_results/masks/mask_20180101.npy'
mask = np.load(mask_path)

print(f"Mask shape: {mask.shape}")
print(f"Masked pixels (1): {(mask==1).sum()} ({(mask==1).sum()/mask.size*100:.1f}%)")
print(f"Valid pixels (0): {(mask==0).sum()} ({(mask==0).sum()/mask.size*100:.1f}%)")

# 创建图形
fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor='white')

# ========== 上图：掩码分布 ==========
ax1 = axes[0]

# 创建colormap：0=有效区域(绿色), 1=掩码区域(灰色)
cmap = mcolors.ListedColormap(['#2ECC71', '#95A5A6'])  # 绿色=有效, 灰色=掩码
bounds = [-0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

im1 = ax1.imshow(mask, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
ax1.set_title('NDVI Mask Distribution (Green=Valid, Gray=Masked)',
              fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('Longitude (0.25° resolution)', fontsize=12)
ax1.set_ylabel('Latitude (0.25° resolution)', fontsize=12)

# 添加纬度标注
H = mask.shape[0]
lats = np.linspace(90, -90, H)
lat_ticks = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 719]
lat_labels = [f"{lats[i]:.0f}°" for i in lat_ticks]
ax1.set_yticks(lat_ticks)
ax1.set_yticklabels(lat_labels, fontsize=9)

# 添加经度标注
W = mask.shape[1]
lons = np.linspace(-180, 180, W)
lon_ticks = [0, 240, 480, 720, 960, 1200, 1439]
lon_labels = [f"{lons[i]:.0f}°" for i in lon_ticks]
ax1.set_xticks(lon_ticks)
ax1.set_xticklabels(lon_labels, fontsize=9)

# 添加colorbar
cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1], fraction=0.046, pad=0.04)
cbar1.ax.set_yticklabels(['Valid (0)', 'Masked (1)'], fontsize=10)

# 添加关键纬度线
for lat_val in [60, 30, 0, -30, -60]:
    lat_idx = np.argmin(np.abs(lats - lat_val))
    ax1.axhline(y=lat_idx, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.text(10, lat_idx, f'{lat_val}°', color='red', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# ========== 下图：纬度带有效像素比例 ==========
ax2 = axes[1]

# 计算每个纬度的有效像素比例
valid_ratio_per_lat = []
for i in range(H):
    valid_ratio = (mask[i] == 0).sum() / W * 100
    valid_ratio_per_lat.append(valid_ratio)

# 绘制曲线
ax2.plot(lats, valid_ratio_per_lat, color='#3498DB', linewidth=2, label='Valid Pixel Ratio')
ax2.fill_between(lats, 0, valid_ratio_per_lat, color='#3498DB', alpha=0.3)

# 添加关键纬度线
for lat_val in [60, 30, 0, -30, -60]:
    ax2.axvline(x=lat_val, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.text(lat_val, 105, f'{lat_val}°', color='red', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax2.set_xlabel('Latitude (°)', fontsize=12)
ax2.set_ylabel('Valid Pixel Ratio (%)', fontsize=12)
ax2.set_title('Valid Pixel Ratio by Latitude (shows why >60° has no data)',
              fontsize=14, fontweight='bold', pad=10)
ax2.set_xlim(90, -90)
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=10)

# 添加注释
ax2.text(75, 50, 'High Latitude\n(Arctic)\nNo vegetation', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax2.text(45, 70, 'Mid Latitude\nMost fire-prone\nregions', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax2.text(-75, 10, 'High Latitude\n(Antarctic)\nNo vegetation', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()

# 保存图片
output_path = 'inference_results/mask_visualization.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
print(f"\n✓ 可视化已保存: {output_path}")
plt.close()
