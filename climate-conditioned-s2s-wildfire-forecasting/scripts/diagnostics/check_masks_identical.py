"""
检查所有mask文件是否相同
"""
import numpy as np
import glob
import os

# 获取所有mask文件
mask_dir = 'inference_results/masks'
mask_files = sorted(glob.glob(os.path.join(mask_dir, 'mask_*.npy')))

print(f"找到 {len(mask_files)} 个mask文件")

# 加载第一个mask作为参考
ref_mask = np.load(mask_files[0])
print(f"\n参考mask: {os.path.basename(mask_files[0])}")
print(f"  Shape: {ref_mask.shape}")
print(f"  Masked pixels: {(ref_mask==1).sum()} ({(ref_mask==1).sum()/ref_mask.size*100:.1f}%)")

# 检查其他mask是否与第一个相同
all_identical = True
for i, mask_file in enumerate(mask_files[1:], 1):
    mask = np.load(mask_file)
    if not np.array_equal(mask, ref_mask):
        all_identical = False
        print(f"\n⚠️  {os.path.basename(mask_file)} 与参考mask不同！")
        break

    if i % 20 == 0:
        print(f"  已检查 {i}/{len(mask_files)-1} 个文件...")

if all_identical:
    print(f"\n✓ 所有 {len(mask_files)} 个mask文件完全相同！")
    print(f"  这是正常的，因为NDVI掩码在不同时间步是固定的")
else:
    print(f"\n✗ 发现不同的mask文件")
