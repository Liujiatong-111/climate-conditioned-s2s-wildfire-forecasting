"""
SeasFire Dataset: 生成 80x80 patch 级别的训练/验证/测试样本
"""
from __future__ import annotations  # 支持 Python 3.10 的类型注解语法
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union
from tqdm import tqdm


def apply_augmentation(data, target, mask):
    """
    应用数据增强：随机翻转和旋转

    注意：对所有14个通道（包括位置编码）都进行相同的空间变换，
          以保持数据和位置编码的一致性

    Args:
        data: (T, 14, H, W) - T个时间步，前10个通道是 fire drivers，后4个是位置编码
        target: (H, W) - 目标标签
        mask: (H, W) - 掩码

    Returns:
        增强后的 data, target, mask
    """
    # 随机水平翻转（概率 0.5）
    if np.random.rand() > 0.5:
        # 对所有时间步和通道进行翻转，axis=3 是 W 维度
        data = np.flip(data, axis=3).copy()
        target = np.flip(target, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()

    # 随机垂直翻转（概率 0.5）
    if np.random.rand() > 0.5:
        # 对所有时间步和通道进行翻转，axis=2 是 H 维度
        data = np.flip(data, axis=2).copy()
        target = np.flip(target, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()

    # 随机旋转 90/180/270 度（概率 0.75，即有 0.25 概率不旋转）
    k = np.random.randint(0, 4)  # 0: 不旋转, 1: 90度, 2: 180度, 3: 270度
    if k > 0:
        # 对所有时间步和通道进行旋转，axes=(2, 3) 是 (H, W) 维度
        data = np.rot90(data, k=k, axes=(2, 3)).copy()
        target = np.rot90(target, k=k, axes=(0, 1)).copy()
        mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

    return data, target, mask


class SeasFirePatchDataset(Dataset):
    """
    SeasFire Patch-level Dataset

    每个样本包含：
    - x_local: (T, 14, 80, 80) - T个时间步，10 fire drivers + 4 coords
    - x_global: (T, 14, 180, 360) - T个时间步，全局粗分辨率视图
    - x_oci: (10, 10) - 10 OCIs × 10 time steps
    - y: (80, 80) - burned area binary labels (0=no fire, 1=fire)
    - mask: (80, 80) - NDVI-based mask (1=masked/sea, 0=valid)
    - patch_row_idx, patch_col_idx: patch 位置
    - time_index: 时间索引
    """

    def __init__(
        self,
        zarr_path: str,
        target_zarr_path: str,
        years: List[int],
        fire_vars: List[str],
        log_transform_vars: List[str],
        oci_vars: List[str],
        target_var: str,
        lead_time_steps: int | List[int] = 1,
        oci_window: int = 10,
        temporal_steps: int = 4,
        burn_threshold: float = 0.0,
        patch_size: int = 80,
        stride: int = None,
        global_coarsen_factor: int = 4,
        use_local: bool = True,
        use_global: bool = True,
        use_oci: bool = True,
        only_fire_patches: bool = True,
        use_augmentation: bool = False,
    ):
        """
        Args:
            zarr_path: 输入变量的 zarr 路径
            target_zarr_path: 目标变量的 zarr 路径
            years: 使用的年份列表
            fire_vars: 火驱动变量名列表
            log_transform_vars: 需要 log 变换的变量
            oci_vars: 海气指数变量名列表
            target_var: 目标变量名
            lead_time_steps: 预测时间步长（int或List[int]，例如1或[1,2,4,8,16]）
            oci_window: OCI 时间窗口
            temporal_steps: Local/Global 的时间步数（默认 4，即过去32天）
            burn_threshold: 二值化阈值
            patch_size: patch 大小（默认 80）
            stride: 滑动窗口步长（默认 None，即等于 patch_size，无重叠）
                - None 或 patch_size: 无重叠的网格划分（原始行为）
                - 40: 50%重叠的滑动窗口（推荐用于推理）
            global_coarsen_factor: 全局下采样因子
            use_local/use_global/use_oci: 三种输入开关
            only_fire_patches: 是否只使用有火的 patch（默认 True）
                - True: 只保留有火灾的 patch（缓解类别不平衡，适合训练）
                - False: 使用所有 patch（真实评估，适合验证/测试）
            use_augmentation: 是否使用数据增强（默认 False）
                - True: 对训练数据进行随机翻转和旋转（只增强前10个通道，保持位置编码不变）
                - False: 不使用数据增强
        """
        self.zarr_path = zarr_path
        self.target_zarr_path = target_zarr_path
        self.years = years
        self.fire_vars = fire_vars
        self.log_transform_vars = log_transform_vars
        self.oci_vars = oci_vars
        self.target_var = target_var

        # 标准化 lead_time_steps 为列表
        if isinstance(lead_time_steps, int):
            self.lead_time_steps = [lead_time_steps]
        elif isinstance(lead_time_steps, (list, tuple)):
            self.lead_time_steps = list(lead_time_steps)
        else:
            raise TypeError(f"lead_time_steps must be int or list, got {type(lead_time_steps)}")

        self.max_lead_time = max(self.lead_time_steps)  # 用于边界检查
        self.oci_window = oci_window
        self.temporal_steps = temporal_steps
        self.burn_threshold = burn_threshold
        self.patch_size = patch_size
        # 滑动窗口步长：默认等于patch_size（无重叠）
        self.stride = stride if stride is not None else patch_size
        self.global_coarsen_factor = global_coarsen_factor
        self.use_local = use_local
        self.use_global = use_global
        self.use_oci = use_oci
        self.only_fire_patches = only_fire_patches
        self.use_augmentation = use_augmentation

        print(f"\n加载数据集（years={years}）...")
        print(f"  多任务预测: lead_time_steps={self.lead_time_steps}")
        print(f"  滑动窗口配置: patch_size={self.patch_size}, stride={self.stride}")

        # 打开 zarr
        self.ds = xr.open_zarr(zarr_path, consolidated=True)
        self.ds_target = xr.open_zarr(target_zarr_path, consolidated=True)

        # 对需要的变量做 log 变换
        for var in log_transform_vars:
            if var in self.ds:
                self.ds[var] = np.log(self.ds[var] + 1)

        # 标准化（使用全局统计）
        print("  计算标准化参数...")
        self.mean_std_dict = {}
        for var in fire_vars:
            self.mean_std_dict[f'{var}_mean'] = float(self.ds[var].mean().values)
            self.mean_std_dict[f'{var}_std'] = float(self.ds[var].std().values)

        # 计算OCI变量的全局统计量（新增）
        print("  计算OCI标准化参数...")
        for var in oci_vars:
            self.mean_std_dict[f'{var}_mean'] = float(self.ds[var].mean().values)
            self.mean_std_dict[f'{var}_std'] = float(self.ds[var].std().values)

        # 按年份筛选 time
        time_years = self.ds['time'].dt.year.values
        valid_mask = np.isin(time_years, years)
        valid_times = np.where(valid_mask)[0]

        # 排除边界（temporal_steps + OCI 窗口 + max lead time）
        # 计算所需的前向边界：取 temporal_steps-1 和 oci_window 的最大值
        # - temporal_steps-1: 需要提取 t_idx-(temporal_steps-1) 到 t_idx 的历史数据
        # - oci_window: 需要提取 t_idx-oci_window 到 t_idx 的 OCI 数据
        min_history_steps = max(temporal_steps - 1, oci_window)

        valid_times = valid_times[
            (valid_times >= min_history_steps) &
            (valid_times < len(self.ds['time']) - self.max_lead_time)
        ]

        print(f"  边界配置:")
        print(f"    - temporal_steps={temporal_steps} (需要 {temporal_steps-1} 个历史步)")
        print(f"    - oci_window={oci_window}")
        print(f"    - max_lead_time={self.max_lead_time}")
        print(f"    - 最小历史步数: {min_history_steps}")
        print(f"    - 有效时间索引范围: {valid_times[0]} 到 {valid_times[-1]}")

        # 🚀 优化：预加载需要的时间段数据到内存
        # 注意：这里必须使用 min_history_steps 而不是 oci_window，确保有足够的历史数据
        print(f"  预加载数据到内存（时间范围: {valid_times[0]}-{valid_times[-1]}）...")
        time_slice = slice(valid_times[0] - min_history_steps, valid_times[-1] + self.max_lead_time + 1)
        self.ds = self.ds.isel(time=time_slice).load()
        self.ds_target = self.ds_target.isel(time=time_slice).load()
        # 更新 valid_times（因为切片后索引变了）
        valid_times = valid_times - valid_times[0] + min_history_steps
        print(f"  ✓ 数据已加载到内存（实际时间步范围: 0 到 {len(self.ds['time'])-1}）")

        # 获取空间尺寸
        self.n_lat = len(self.ds['latitude'])
        self.n_lon = len(self.ds['longitude'])

        # 滑动窗口 Patch 划分：计算所有可能的起始位置
        # 使用滑动窗口方式，确保完整覆盖整个图像（包括边缘）
        h_starts = list(range(0, self.n_lat - patch_size, self.stride))
        if len(h_starts) == 0 or h_starts[-1] != self.n_lat - patch_size:
            h_starts.append(self.n_lat - patch_size)  # 确保覆盖到底部边缘

        w_starts = list(range(0, self.n_lon - patch_size, self.stride))
        if len(w_starts) == 0 or w_starts[-1] != self.n_lon - patch_size:
            w_starts.append(self.n_lon - patch_size)  # 确保覆盖到右侧边缘

        self.h_starts = h_starts
        self.w_starts = w_starts
        self.n_rows = len(h_starts)
        self.n_cols = len(w_starts)

        print(f"  空间网格: {self.n_lat} × {self.n_lon}")
        print(f"  Patch 划分: {self.n_rows} 行 × {self.n_cols} 列 (滑动窗口)")
        if self.stride < patch_size:
            overlap_ratio = (patch_size - self.stride) / patch_size * 100
            print(f"  重叠率: {overlap_ratio:.1f}% (stride={self.stride}, patch_size={patch_size})")

        # 构建样本索引：(time_idx, row_idx, col_idx)
        # row_idx 和 col_idx 是在 h_starts 和 w_starts 列表中的索引
        print("  构建样本索引...")
        self.samples = []
        n_fire_patches = 0
        n_total_patches = 0

        for t_idx in tqdm(valid_times, desc="  扫描时间步"):
            for row_idx in range(self.n_rows):
                for col_idx in range(self.n_cols):
                    i0 = h_starts[row_idx]
                    j0 = w_starts[col_idx]

                    n_total_patches += 1

                    # 检查所有 lead times 的 target（只要有一个有火就算）
                    has_fire_any_lead = False
                    for lead_t in self.lead_time_steps:
                        target_time_idx = t_idx + lead_t
                        target_data = self.ds_target[target_var].isel(time=target_time_idx).values
                        patch_target = target_data[i0:i0+patch_size, j0:j0+patch_size]

                        if np.nansum(patch_target) > 0:
                            has_fire_any_lead = True
                            break  # 只要有一个lead time有火就够了

                    if has_fire_any_lead:
                        n_fire_patches += 1

                    # 根据 only_fire_patches 参数决定是否加入样本
                    if self.only_fire_patches:
                        # 只保留有火的 patch（缓解类别不平衡）
                        if has_fire_any_lead:
                            self.samples.append((t_idx, row_idx, col_idx))
                    else:
                        # 保留所有 patch（真实评估）
                        self.samples.append((t_idx, row_idx, col_idx))

        print(f"  总 patch 数: {n_total_patches}")
        print(f"  有火 patch 数: {n_fire_patches} ({100*n_fire_patches/n_total_patches:.2f}%)")
        print(f"  使用策略: {'只使用有火patch' if self.only_fire_patches else '使用所有patch'}")
        print(f"✓ 最终样本数: {len(self.samples)}")

        # 添加详细的样本统计信息（用于调试 temporal_steps 问题）
        if len(self.samples) > 0:
            sample_t_indices = [s[0] for s in self.samples]
            print(f"\n  样本时间索引统计:")
            print(f"    - 最小 t_idx: {min(sample_t_indices)}")
            print(f"    - 最大 t_idx: {max(sample_t_indices)}")
            print(f"    - 可访问的历史时间范围: {min(sample_t_indices) - temporal_steps + 1} 到 {max(sample_t_indices)}")
            print(f"    - 可访问的未来时间范围: {min(sample_t_indices)} 到 {max(sample_t_indices) + self.max_lead_time}")
            print(f"    - 数据集总时间步数: {len(self.ds['time'])}")

            # 验证边界安全性
            min_accessible_t = min(sample_t_indices) - temporal_steps + 1
            max_accessible_t = max(sample_t_indices) + self.max_lead_time
            if min_accessible_t < 0:
                print(f"    ⚠️  警告：最小可访问时间索引 {min_accessible_t} < 0，可能导致数据访问错误！")
            if max_accessible_t >= len(self.ds['time']):
                print(f"    ⚠️  警告：最大可访问时间索引 {max_accessible_t} >= {len(self.ds['time'])}，可能导致数据访问错误！")
            if min_accessible_t >= 0 and max_accessible_t < len(self.ds['time']):
                print(f"    ✓ 边界检查通过：所有样本的时间访问都在有效范围内")

        # 预计算全局经纬度编码（在 0.25° 网格上）
        print("  预计算位置编码...")
        self.coord_grid_local = self._compute_coord_grid(self.n_lat, self.n_lon)  # (4, 720, 1440)

        # 预计算全局粗分辨率数据
        if use_global:
            print("  构建全局粗分辨率视图...")
            # 消除 dask 重塑警告
            import dask
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                self.ds_global = self.ds[fire_vars].coarsen(
                    latitude=global_coarsen_factor,
                    longitude=global_coarsen_factor,
                    boundary='trim'
                ).mean()

            # 标准化
            for var in fire_vars:
                self.ds_global[var] = (
                    (self.ds_global[var] - self.mean_std_dict[f'{var}_mean']) /
                    self.mean_std_dict[f'{var}_std']
                )

            # 🚀 预加载全局数据到内存
            self.ds_global = self.ds_global.load()

            n_lat_g = len(self.ds_global['latitude'])
            n_lon_g = len(self.ds_global['longitude'])
            self.coord_grid_global = self._compute_coord_grid(n_lat_g, n_lon_g)  # (4, 180, 360)
            print(f"    全局网格: {n_lat_g} × {n_lon_g}")
        else:
            self.ds_global = None
            self.coord_grid_global = None

        print("✓ 数据集初始化完成\n")

    def _compute_coord_grid(self, n_lat: int, n_lon: int) -> np.ndarray:
        """
        计算经纬度位置编码：cos/sin(lon), cos/sin(lat)

        Returns:
            coords: (4, n_lat, n_lon)
        """
        lats = np.linspace(-90, 90, n_lat)
        lons = np.linspace(-180, 180, n_lon)

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        lon_rad = lon_grid * np.pi / 180
        lat_rad = lat_grid * np.pi / 180

        cos_lon = np.cos(lon_rad)
        sin_lon = np.sin(lon_rad)
        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)

        coords = np.stack([cos_lon, sin_lon, cos_lat, sin_lat], axis=0).astype(np.float32)
        return coords

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t_idx, row_idx, col_idx = self.samples[idx]

        # ========== 边界检查（确保数据访问安全）==========
        # 检查历史数据边界
        min_t = t_idx - self.temporal_steps + 1
        assert min_t >= 0, (
            f"时间索引越界！t_idx={t_idx}, temporal_steps={self.temporal_steps}, "
            f"需要访问 t={min_t} 但最小索引为 0。"
            f"这表明数据集初始化时的边界过滤有问题。"
        )

        # 检查未来数据边界
        max_t = t_idx + self.max_lead_time
        assert max_t < len(self.ds['time']), (
            f"时间索引越界！t_idx={t_idx}, max_lead_time={self.max_lead_time}, "
            f"需要访问 t={max_t} 但最大索引为 {len(self.ds['time'])-1}。"
            f"这表明数据集初始化时的边界过滤有问题。"
        )

        # Patch 空间范围（使用滑动窗口的起始位置）
        i0 = self.h_starts[row_idx]
        j0 = self.w_starts[col_idx]
        i1 = i0 + self.patch_size
        j1 = j0 + self.patch_size

        # ========== Local Input (多时间步) ==========
        if self.use_local:
            # 提取过去 temporal_steps 个时间步的数据
            local_temporal = []
            for t in range(t_idx - self.temporal_steps + 1, t_idx + 1):
                # 提取 10 个 fire drivers
                local_data = []
                for var in self.fire_vars:
                    data = self.ds[var].isel(time=t).values[i0:i1, j0:j1]
                    # 标准化
                    data = (data - self.mean_std_dict[f'{var}_mean']) / self.mean_std_dict[f'{var}_std']
                    local_data.append(data)
                local_data = np.stack(local_data, axis=0)  # (10, 80, 80)
                local_temporal.append(local_data)

            local_temporal = np.stack(local_temporal, axis=1)  # (10, T, 80, 80)

            # 添加位置编码（在所有时间步上复制）
            coord_patch = self.coord_grid_local[:, i0:i1, j0:j1]  # (4, 80, 80)
            coord_patch = np.expand_dims(coord_patch, axis=1)  # (4, 1, 80, 80)
            coord_patch = np.repeat(coord_patch, self.temporal_steps, axis=1)  # (4, T, 80, 80)

            x_local = np.concatenate([local_temporal, coord_patch], axis=0)  # (14, T, 80, 80)
            # 转换维度顺序: (14, T, 80, 80) -> (T, 14, 80, 80)
            x_local = np.transpose(x_local, (1, 0, 2, 3))  # (T, 14, 80, 80)
            x_local = np.nan_to_num(x_local, nan=0.0)
        else:
            x_local = np.zeros((self.temporal_steps, 14, self.patch_size, self.patch_size), dtype=np.float32)

        # ========== Global Input (多时间步) ==========
        if self.use_global and self.ds_global is not None:
            # 提取过去 temporal_steps 个时间步的数据
            global_temporal = []
            for t in range(t_idx - self.temporal_steps + 1, t_idx + 1):
                global_data = []
                for var in self.fire_vars:
                    data = self.ds_global[var].isel(time=t).values
                    global_data.append(data)
                global_data = np.stack(global_data, axis=0)  # (10, 180, 360)
                global_temporal.append(global_data)

            global_temporal = np.stack(global_temporal, axis=1)  # (10, T, 180, 360)

            # 添加位置编码（在所有时间步上复制）
            coord_global = np.expand_dims(self.coord_grid_global, axis=1)  # (4, 1, 180, 360)
            coord_global = np.repeat(coord_global, self.temporal_steps, axis=1)  # (4, T, 180, 360)

            x_global = np.concatenate([global_temporal, coord_global], axis=0)  # (14, T, 180, 360)
            # 转换维度顺序: (14, T, 180, 360) -> (T, 14, 180, 360)
            x_global = np.transpose(x_global, (1, 0, 2, 3))  # (T, 14, 180, 360)
            x_global = np.nan_to_num(x_global, nan=0.0)
        else:
            x_global = np.zeros((self.temporal_steps, 14, 180, 360), dtype=np.float32)

        # ========== OCI Input ==========
        if self.use_oci:
            oci_data = []
            for var in self.oci_vars:
                # 取 t-oci_window+1 到 t 的数据
                window_data = self.ds[var].isel(
                    time=slice(t_idx - self.oci_window + 1, t_idx + 1)
                ).values

                # 如果是空间标量（广播到整个网格），取第一个点即可
                if window_data.ndim > 1:
                    window_data = window_data[:, 0, 0]

                # 使用全局统计量标准化（改进）
                window_data = (window_data - self.mean_std_dict[f'{var}_mean']) / \
                              (self.mean_std_dict[f'{var}_std'] + 1e-8)

                oci_data.append(window_data)

            x_oci = np.stack(oci_data, axis=0)  # (10, oci_window)
            x_oci = np.nan_to_num(x_oci, nan=0.0)
        else:
            x_oci = np.zeros((10, self.oci_window), dtype=np.float32)

        # ========== Target（多个 lead times）==========
        y_list = []
        for lead_t in self.lead_time_steps:
            target_time_idx = t_idx + lead_t
            target_data = self.ds_target[self.target_var].isel(time=target_time_idx).values[i0:i1, j0:j1]

            # 二值化目标（与 televit-main 一致）
            target_data = np.nan_to_num(target_data, nan=0.0)  # NaN → 0
            y_t = np.where(target_data > self.burn_threshold, 1, 0).astype(np.int64)  # 0 或 1
            y_list.append(y_t)

        # 如果只有一个 lead time，返回 (H, W)；否则返回 (L, H, W)
        if len(self.lead_time_steps) == 1:
            y = y_list[0]  # (80, 80)
        else:
            y = np.stack(y_list, axis=0)  # (L, 80, 80)

        # ========== Mask（使用 NDVI 的 NaN，与 televit-main 一致）==========
        # televit-main: mask = np.isnan(batch.isel(time=-1)['ndvi']).values
        ndvi_data = self.ds['ndvi'].isel(time=t_idx).values[i0:i1, j0:j1]
        mask = np.isnan(ndvi_data).astype(np.float32)  # True where NDVI is NaN (sea/desert)

        # ========== 数据增强（只在训练时使用）==========
        if self.use_augmentation:
            x_local, y, mask = apply_augmentation(x_local, y, mask)

        # 转换为 Tensor
        x_local = torch.from_numpy(x_local)
        x_global = torch.from_numpy(x_global)
        x_oci = torch.from_numpy(x_oci)
        y = torch.from_numpy(y)
        mask = torch.from_numpy(mask)

        return x_local, x_global, x_oci, y, mask, row_idx, col_idx, t_idx
