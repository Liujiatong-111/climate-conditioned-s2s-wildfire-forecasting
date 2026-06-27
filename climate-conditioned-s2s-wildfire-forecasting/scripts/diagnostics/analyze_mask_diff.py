"""
详细分析mask文件的差异
"""
import numpy as np
import glob
import os

# 获取所有mask文件
mask_dir = 'inference_results/masks'
mask_files = sorted(glob.glob(os.path.join(mask_dir, 'mask_*.npy')))

print(f"找到 {len(mask_files)} 个mask文件\n")

# 加载前5个mask并比较
print("前5个mask文件的统计信息：")
print("-" * 70)

masks = []
for i in range(min(5, len(mask_files))):
    mask = np.load(mask_files[i])
    masks.append(mask)

    masked_count = (mask == 1).sum()
    valid_count = (mask == 0).sum()

    print(f"{os.path.basename(mask_files[i]):<25} | "
          f"Masked: {masked_count:>7} ({masked_count/mask.size*100:>5.2f}%) | "
          f"Valid: {valid_count:>6} ({valid_count/mask.size*100:>5.2f}%)")

# 比较差异
if len(masks) >= 2:
    print("\n差异分析：")
    print("-" * 70)
    for i in range(1, len(masks)):
        diff = (masks[0] != masks[i]).sum()
        print(f"mask_20180101 vs {os.path.basename(mask_files[i]):<20} | "
              f"不同像素: {diff:>6} ({diff/masks[0].size*100:.3f}%)")
