"""Brief implementation note."""
from __future__ import annotations  
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union
from tqdm import tqdm


def apply_augmentation(data, target, mask):
    """Brief implementation note."""
    
    if np.random.rand() > 0.5:
        
        data = np.flip(data, axis=3).copy()
        target = np.flip(target, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()

    
    if np.random.rand() > 0.5:
        
        data = np.flip(data, axis=2).copy()
        target = np.flip(target, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()

    
    k = np.random.randint(0, 4)  
    if k > 0:
        
        data = np.rot90(data, k=k, axes=(2, 3)).copy()
        target = np.rot90(target, k=k, axes=(0, 1)).copy()
        mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

    return data, target, mask


class SeasFirePatchDataset(Dataset):
    """Brief implementation note."""

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
        """Brief implementation note."""
        self.zarr_path = zarr_path
        self.target_zarr_path = target_zarr_path
        self.years = years
        self.fire_vars = fire_vars
        self.log_transform_vars = log_transform_vars
        self.oci_vars = oci_vars
        self.target_var = target_var

        
        if isinstance(lead_time_steps, int):
            self.lead_time_steps = [lead_time_steps]
        elif isinstance(lead_time_steps, (list, tuple)):
            self.lead_time_steps = list(lead_time_steps)
        else:
            raise TypeError(f"lead_time_steps must be int or list, got {type(lead_time_steps)}")

        self.max_lead_time = max(self.lead_time_steps)  
        self.oci_window = oci_window
        self.temporal_steps = temporal_steps
        self.burn_threshold = burn_threshold
        self.patch_size = patch_size
        
        self.stride = stride if stride is not None else patch_size
        self.global_coarsen_factor = global_coarsen_factor
        self.use_local = use_local
        self.use_global = use_global
        self.use_oci = use_oci
        self.only_fire_patches = only_fire_patches
        self.use_augmentation = use_augmentation

        print(f"Status")
        print(f"Status")
        print(f"Status")

        
        self.ds = xr.open_zarr(zarr_path, consolidated=True)
        self.ds_target = xr.open_zarr(target_zarr_path, consolidated=True)

        
        for var in log_transform_vars:
            if var in self.ds:
                self.ds[var] = np.log(self.ds[var] + 1)

        
        print("Status")
        self.mean_std_dict = {}
        for var in fire_vars:
            self.mean_std_dict[f'{var}_mean'] = float(self.ds[var].mean().values)
            self.mean_std_dict[f'{var}_std'] = float(self.ds[var].std().values)

        
        print("Status")
        for var in oci_vars:
            self.mean_std_dict[f'{var}_mean'] = float(self.ds[var].mean().values)
            self.mean_std_dict[f'{var}_std'] = float(self.ds[var].std().values)

        
        time_years = self.ds['time'].dt.year.values
        valid_mask = np.isin(time_years, years)
        valid_times = np.where(valid_mask)[0]

        
        
        
        
        min_history_steps = max(temporal_steps - 1, oci_window)

        valid_times = valid_times[
            (valid_times >= min_history_steps) &
            (valid_times < len(self.ds['time']) - self.max_lead_time)
        ]

        print(f"Status")
        print(f"Status")
        print(f"    - oci_window={oci_window}")
        print(f"    - max_lead_time={self.max_lead_time}")
        print(f"Status")
        print(f"Status")

        
        
        print(f"Status")
        time_slice = slice(valid_times[0] - min_history_steps, valid_times[-1] + self.max_lead_time + 1)
        self.ds = self.ds.isel(time=time_slice).load()
        self.ds_target = self.ds_target.isel(time=time_slice).load()
        
        valid_times = valid_times - valid_times[0] + min_history_steps
        print(f"Status")

        
        self.n_lat = len(self.ds['latitude'])
        self.n_lon = len(self.ds['longitude'])

        
        
        h_starts = list(range(0, self.n_lat - patch_size, self.stride))
        if len(h_starts) == 0 or h_starts[-1] != self.n_lat - patch_size:
            h_starts.append(self.n_lat - patch_size)  

        w_starts = list(range(0, self.n_lon - patch_size, self.stride))
        if len(w_starts) == 0 or w_starts[-1] != self.n_lon - patch_size:
            w_starts.append(self.n_lon - patch_size)  

        self.h_starts = h_starts
        self.w_starts = w_starts
        self.n_rows = len(h_starts)
        self.n_cols = len(w_starts)

        print(f"Status")
        print(f"Status")
        if self.stride < patch_size:
            overlap_ratio = (patch_size - self.stride) / patch_size * 100
            print(f"Status")

        
        
        print("Status")
        self.samples = []
        n_fire_patches = 0
        n_total_patches = 0

        for t_idx in tqdm(valid_times, desc="Status"):
            for row_idx in range(self.n_rows):
                for col_idx in range(self.n_cols):
                    i0 = h_starts[row_idx]
                    j0 = w_starts[col_idx]

                    n_total_patches += 1

                    
                    has_fire_any_lead = False
                    for lead_t in self.lead_time_steps:
                        target_time_idx = t_idx + lead_t
                        target_data = self.ds_target[target_var].isel(time=target_time_idx).values
                        patch_target = target_data[i0:i0+patch_size, j0:j0+patch_size]

                        if np.nansum(patch_target) > 0:
                            has_fire_any_lead = True
                            break  

                    if has_fire_any_lead:
                        n_fire_patches += 1

                    
                    if self.only_fire_patches:
                        
                        if has_fire_any_lead:
                            self.samples.append((t_idx, row_idx, col_idx))
                    else:
                        
                        self.samples.append((t_idx, row_idx, col_idx))

        print(f"Status")
        print(f"Status")
        print(f"Status")
        print(f"Status")

        
        if len(self.samples) > 0:
            sample_t_indices = [s[0] for s in self.samples]
            print(f"Status")
            print(f"Status")
            print(f"Status")
            print(f"Status")
            print(f"Status")
            print(f"Status")

            
            min_accessible_t = min(sample_t_indices) - temporal_steps + 1
            max_accessible_t = max(sample_t_indices) + self.max_lead_time
            if min_accessible_t < 0:
                print(f"Status")
            if max_accessible_t >= len(self.ds['time']):
                print(f"Status")
            if min_accessible_t >= 0 and max_accessible_t < len(self.ds['time']):
                print(f"Status")

        
        print("Status")
        self.coord_grid_local = self._compute_coord_grid(self.n_lat, self.n_lon)  # (4, 720, 1440)

        
        if use_global:
            print("Status")
            
            import dask
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                self.ds_global = self.ds[fire_vars].coarsen(
                    latitude=global_coarsen_factor,
                    longitude=global_coarsen_factor,
                    boundary='trim'
                ).mean()

            
            for var in fire_vars:
                self.ds_global[var] = (
                    (self.ds_global[var] - self.mean_std_dict[f'{var}_mean']) /
                    self.mean_std_dict[f'{var}_std']
                )

            
            self.ds_global = self.ds_global.load()

            n_lat_g = len(self.ds_global['latitude'])
            n_lon_g = len(self.ds_global['longitude'])
            self.coord_grid_global = self._compute_coord_grid(n_lat_g, n_lon_g)  # (4, 180, 360)
            print(f"Status")
        else:
            self.ds_global = None
            self.coord_grid_global = None

        print("Status")

    def _compute_coord_grid(self, n_lat: int, n_lon: int) -> np.ndarray:
        """Brief implementation note."""
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

        
        
        min_t = t_idx - self.temporal_steps + 1
        assert min_t >= 0, (
            f"Status"
            f"Status"
            f"Status"
        )

        
        max_t = t_idx + self.max_lead_time
        assert max_t < len(self.ds['time']), (
            f"Status"
            f"Status"
            f"Status"
        )

        
        i0 = self.h_starts[row_idx]
        j0 = self.w_starts[col_idx]
        i1 = i0 + self.patch_size
        j1 = j0 + self.patch_size

        
        if self.use_local:
            
            local_temporal = []
            for t in range(t_idx - self.temporal_steps + 1, t_idx + 1):
                
                local_data = []
                for var in self.fire_vars:
                    data = self.ds[var].isel(time=t).values[i0:i1, j0:j1]
                    
                    data = (data - self.mean_std_dict[f'{var}_mean']) / self.mean_std_dict[f'{var}_std']
                    local_data.append(data)
                local_data = np.stack(local_data, axis=0)  # (10, 80, 80)
                local_temporal.append(local_data)

            local_temporal = np.stack(local_temporal, axis=1)  # (10, T, 80, 80)

            
            coord_patch = self.coord_grid_local[:, i0:i1, j0:j1]  # (4, 80, 80)
            coord_patch = np.expand_dims(coord_patch, axis=1)  # (4, 1, 80, 80)
            coord_patch = np.repeat(coord_patch, self.temporal_steps, axis=1)  # (4, T, 80, 80)

            x_local = np.concatenate([local_temporal, coord_patch], axis=0)  # (14, T, 80, 80)
            
            x_local = np.transpose(x_local, (1, 0, 2, 3))  # (T, 14, 80, 80)
            x_local = np.nan_to_num(x_local, nan=0.0)
        else:
            x_local = np.zeros((self.temporal_steps, 14, self.patch_size, self.patch_size), dtype=np.float32)

        
        if self.use_global and self.ds_global is not None:
            
            global_temporal = []
            for t in range(t_idx - self.temporal_steps + 1, t_idx + 1):
                global_data = []
                for var in self.fire_vars:
                    data = self.ds_global[var].isel(time=t).values
                    global_data.append(data)
                global_data = np.stack(global_data, axis=0)  # (10, 180, 360)
                global_temporal.append(global_data)

            global_temporal = np.stack(global_temporal, axis=1)  # (10, T, 180, 360)

            
            coord_global = np.expand_dims(self.coord_grid_global, axis=1)  # (4, 1, 180, 360)
            coord_global = np.repeat(coord_global, self.temporal_steps, axis=1)  # (4, T, 180, 360)

            x_global = np.concatenate([global_temporal, coord_global], axis=0)  # (14, T, 180, 360)
            
            x_global = np.transpose(x_global, (1, 0, 2, 3))  # (T, 14, 180, 360)
            x_global = np.nan_to_num(x_global, nan=0.0)
        else:
            x_global = np.zeros((self.temporal_steps, 14, 180, 360), dtype=np.float32)

        # ========== OCI Input ==========
        if self.use_oci:
            oci_data = []
            for var in self.oci_vars:
                
                window_data = self.ds[var].isel(
                    time=slice(t_idx - self.oci_window + 1, t_idx + 1)
                ).values

                
                if window_data.ndim > 1:
                    window_data = window_data[:, 0, 0]

                
                window_data = (window_data - self.mean_std_dict[f'{var}_mean']) / \
                              (self.mean_std_dict[f'{var}_std'] + 1e-8)

                oci_data.append(window_data)

            x_oci = np.stack(oci_data, axis=0)  # (10, oci_window)
            x_oci = np.nan_to_num(x_oci, nan=0.0)
        else:
            x_oci = np.zeros((10, self.oci_window), dtype=np.float32)

        
        y_list = []
        for lead_t in self.lead_time_steps:
            target_time_idx = t_idx + lead_t
            target_data = self.ds_target[self.target_var].isel(time=target_time_idx).values[i0:i1, j0:j1]

            
            target_data = np.nan_to_num(target_data, nan=0.0)  # NaN → 0
            y_t = np.where(target_data > self.burn_threshold, 1, 0).astype(np.int64)  
            y_list.append(y_t)

        
        if len(self.lead_time_steps) == 1:
            y = y_list[0]  # (80, 80)
        else:
            y = np.stack(y_list, axis=0)  # (L, 80, 80)

        
        # televit-main: mask = np.isnan(batch.isel(time=-1)['ndvi']).values
        ndvi_data = self.ds['ndvi'].isel(time=t_idx).values[i0:i1, j0:j1]
        mask = np.isnan(ndvi_data).astype(np.float32)  # True where NDVI is NaN (sea/desert)

        
        if self.use_augmentation:
            x_local, y, mask = apply_augmentation(x_local, y, mask)

        
        x_local = torch.from_numpy(x_local)
        x_global = torch.from_numpy(x_global)
        x_oci = torch.from_numpy(x_oci)
        y = torch.from_numpy(y)
        mask = torch.from_numpy(mask)

        return x_local, x_global, x_oci, y, mask, row_idx, col_idx, t_idx
