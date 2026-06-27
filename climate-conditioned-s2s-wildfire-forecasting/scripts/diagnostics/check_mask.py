import numpy as np

# 加载mask文件
mask = np.load('inference_results/masks/mask_20180101.npy')

print('Mask shape:', mask.shape)
print('\nMask stats:')
print('  Total pixels:', mask.size)
print('  Masked (1):', (mask==1).sum(), f'({(mask==1).sum()/mask.size*100:.1f}%)')
print('  Valid (0):', (mask==0).sum(), f'({(mask==0).sum()/mask.size*100:.1f}%)')

print('\nChecking latitude coverage:')
H = mask.shape[0]
lats = np.linspace(90, -90, H)

for lat_threshold in [80, 70, 60, 50, 40, 30]:
    lat_idx = np.argmin(np.abs(lats - lat_threshold))
    valid_pixels = (mask[lat_idx]==0).sum()
    total_pixels = mask.shape[1]
    print(f'  Lat {lat_threshold:>3}° (row {lat_idx:>3}): valid pixels = {valid_pixels:>4}/{total_pixels} ({valid_pixels/total_pixels*100:.1f}%)')

# 检查每个纬度带的有效像素比例
print('\nLatitude band analysis:')
for lat_start in [90, 70, 60, 50, 40, 30, 0, -30, -60]:
    lat_end = lat_start - 10
    idx_start = np.argmin(np.abs(lats - lat_start))
    idx_end = np.argmin(np.abs(lats - lat_end))

    band_mask = mask[idx_start:idx_end, :]
    valid_ratio = (band_mask==0).sum() / band_mask.size * 100
    print(f'  {lat_start:>3}° to {lat_end:>3}°: {valid_ratio:.1f}% valid pixels')
