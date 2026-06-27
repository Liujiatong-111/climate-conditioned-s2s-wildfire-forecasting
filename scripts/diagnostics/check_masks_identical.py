"""Brief implementation note."""
import numpy as np
import glob
import os


mask_dir = 'inference_results/masks'
mask_files = sorted(glob.glob(os.path.join(mask_dir, 'mask_*.npy')))

print(f"Status")


ref_mask = np.load(mask_files[0])
print(f"Status")
print(f"  Shape: {ref_mask.shape}")
print(f"  Masked pixels: {(ref_mask==1).sum()} ({(ref_mask==1).sum()/ref_mask.size*100:.1f}%)")


all_identical = True
for i, mask_file in enumerate(mask_files[1:], 1):
    mask = np.load(mask_file)
    if not np.array_equal(mask, ref_mask):
        all_identical = False
        print(f"Status")
        break

    if i % 20 == 0:
        print(f"Status")

if all_identical:
    print(f"Status")
    print(f"Status")
else:
    print(f"Status")
