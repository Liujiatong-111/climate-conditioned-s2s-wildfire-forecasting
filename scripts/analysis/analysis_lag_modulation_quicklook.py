"""
Quick-look analysis for lag inconsistency and climate modulation.

This script uses the SeasFireCube zarr directly and summarizes:
1. Peak-lag distributions for local drivers vs OCI signals.
2. Spatial maps of patch-level peak lags.
3. Regional cross-correlation examples showing no universal OCI lag.
4. Phase-conditioned local response curves showing modulation.

The analysis is intentionally lightweight and aligned with the model scale:
- patch size: 80 x 80 cells (same as local branch)
- fire activity: fraction of burned pixels within each patch
- local variables: patch means
- OCI variables: global scalar indices
- deseasoning: monthly climatology removal
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DATA_PATH = Path("/home/dataset-local/QJC/ljt/TeleVIT/SeasFireCube_v3.zarr")
OUTPUT_DIR = Path(__file__).resolve().parent / "lag_modulation_quicklook_lag60"

PATCH_SIZE = 80
ACTIVE_FIRE_THRESHOLD = 1e-4
LAND_THRESHOLD = 0.2
LOCAL_LAG_MAX = 6
OCI_LAG_MAX = 60

FIRE_VARS = [
    "lst_day",
    "mslp",
    "ndvi",
    "pop_dens",
    "ssrd",
    "sst",
    "swvl1",
    "t2m_mean",
    "tp",
    "vpd",
]
OCI_VARS = [
    "oci_censo",
    "oci_ea",
    "oci_epo",
    "oci_gmsst",
    "oci_nao",
    "oci_nina34_anom",
    "oci_pdo",
    "oci_pna",
    "oci_soi",
    "oci_wp",
]
FOCUS_LOCAL_VARS = ["vpd", "rel_hum", "t2m_mean", "tp"]
FOCUS_OCI_VARS = ["oci_nina34_anom", "oci_pdo", "oci_soi", "oci_nao"]
DISPLAY_LABELS = {
    "lst_day": "LST",
    "mslp": "MSLP",
    "ndvi": "NDVI",
    "pop_dens": "POP",
    "ssrd": "SSRD",
    "sst": "SST",
    "swvl1": "SWVL1",
    "t2m_mean": "T2M",
    "tp": "TP",
    "vpd": "VPD",
    "oci_censo": "CENSO",
    "oci_ea": "EA",
    "oci_epo": "EPO",
    "oci_gmsst": "GMSST",
    "oci_nao": "NAO",
    "oci_nina34_anom": "Nino34",
    "oci_pdo": "PDO",
    "oci_pna": "PNA",
    "oci_soi": "SOI",
    "oci_wp": "WP",
}
PALETTE_NAME = "journal_muted"
PALETTES = {
    "journal_muted": {
        "Local": ("#4c628c", "#aca7ce"),
        "Global": ("#5f9b97", "#bee8dd"),
        "OCI": ("#c98f7e", "#f7e6b2"),
    },
    "journal_cool": {
        "Local": ("#3f5f8a", "#a6c0e1"),
        "Global": ("#3f8f88", "#a6dfd8"),
        "OCI": ("#7b7fb2", "#c6c5e8"),
    },
    "journal_warm": {
        "Local": ("#5f6c91", "#c4c7d9"),
        "Global": ("#6d9d94", "#c8e1db"),
        "OCI": ("#bf7f73", "#eac7b8"),
    },
}
EXAMPLE_PATCHES = [
    ((4, 9), "Eq. Africa"),
    ((4, 5), "Amazon"),
    ((3, 14), "SE Asia"),
    ((4, 14), "Maritime Continent"),
]


@dataclass
class PatchResult:
    patch_i: int
    patch_j: int
    lat_center: float
    lon_center: float
    land_fraction: float
    fire_mean: float
    metrics: Dict[str, float]


def deseason_by_month(x: np.ndarray, months: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = x.copy()
    for month in range(1, 13):
        idx = months == month
        if not np.any(idx):
            continue
        month_vals = x[idx]
        if not np.any(np.isfinite(month_vals)):
            y[idx] = np.nan
            continue
        y[idx] = month_vals - np.nanmean(month_vals)
    if not np.any(np.isfinite(y)):
        return np.full_like(y, np.nan, dtype=np.float64)
    std = np.nanstd(y)
    if not np.isfinite(std) or std < 1e-8:
        return np.zeros_like(y)
    return y / std


def lag_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    corrs = []
    for lag in range(max_lag + 1):
        if lag == 0:
            xs, ys = x, y
        else:
            xs, ys = x[:-lag], y[lag:]
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() < 30:
            corrs.append(np.nan)
            continue
        xs = xs[mask]
        ys = ys[mask]
        if np.std(xs) < 1e-8 or np.std(ys) < 1e-8:
            corrs.append(np.nan)
        else:
            corrs.append(float(np.corrcoef(xs, ys)[0, 1]))
    return np.asarray(corrs, dtype=np.float64)


def lag_shift(x: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return x.copy()
    return np.concatenate([x[:-lag], np.full(lag, np.nan)])


def fit_r2(y: np.ndarray, X: np.ndarray) -> float:
    mask = np.isfinite(y)
    for k in range(X.shape[1]):
        mask &= np.isfinite(X[:, k])
    y_valid = y[mask]
    X_valid = X[mask]
    if len(y_valid) < 40:
        return np.nan
    X_valid = np.column_stack([np.ones(len(y_valid)), X_valid])
    beta, *_ = np.linalg.lstsq(X_valid, y_valid, rcond=None)
    y_hat = X_valid @ beta
    ssr = np.sum((y_valid - y_hat) ** 2)
    sst = np.sum((y_valid - y_valid.mean()) ** 2)
    if sst < 1e-8:
        return np.nan
    return float(1.0 - ssr / sst)


def fit_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    if len(x_valid) < 20 or np.std(x_valid) < 1e-8 or np.std(y_valid) < 1e-8:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(x_valid)), x_valid])
    beta, *_ = np.linalg.lstsq(X, y_valid, rcond=None)
    y_hat = X @ beta
    ssr = np.sum((y_valid - y_hat) ** 2)
    sst = np.sum((y_valid - y_valid.mean()) ** 2)
    r2 = np.nan if sst < 1e-8 else float(1.0 - ssr / sst)
    return float(beta[1]), r2


def bin_means(x: np.ndarray, y: np.ndarray, n_bins: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    if len(x_valid) < 20:
        return np.asarray([]), np.asarray([])
    edges = np.nanpercentile(x_valid, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.asarray([]), np.asarray([])
    centers = []
    means = []
    for left, right in zip(edges[:-1], edges[1:]):
        bin_mask = mask & (x >= left) & (x <= right if right == edges[-1] else x < right)
        if bin_mask.sum() == 0:
            continue
        centers.append(np.nanmean(x[bin_mask]))
        means.append(np.nanmean(y[bin_mask]))
    return np.asarray(centers), np.asarray(means)


def patch_extent(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    lat_centers = np.array([np.mean(lats[i * PATCH_SIZE:(i + 1) * PATCH_SIZE]) for i in range(9)])
    lon_centers = np.array([np.mean(lons[j * PATCH_SIZE:(j + 1) * PATCH_SIZE]) for j in range(18)])
    return lat_centers, lon_centers


def global_background_series(da: xr.DataArray) -> np.ndarray:
    lat_weights = xr.DataArray(
        np.cos(np.deg2rad(da["latitude"].values.astype(np.float64))),
        coords={"latitude": da["latitude"]},
        dims=("latitude",),
    )
    filled = da.fillna(0.0)
    valid = xr.where(np.isfinite(da), 1.0, 0.0)
    numerator = (filled * lat_weights).sum(dim=("latitude", "longitude"))
    denominator = (valid * lat_weights).sum(dim=("latitude", "longitude"))
    ts = xr.where(denominator > 0, numerator / denominator, np.nan).compute().values
    return np.asarray(ts, dtype=np.float64)


def load_data() -> Tuple[
    Dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    all_vars = ["gwis_ba", "lsm", *FIRE_VARS, *FOCUS_LOCAL_VARS, *OCI_VARS]
    data_vars = list(dict.fromkeys(all_vars))
    ds = xr.open_zarr(str(DATA_PATH), consolidated=True)[
        data_vars
    ]

    coarse = {}
    for var in ["gwis_ba", *FIRE_VARS, *FOCUS_LOCAL_VARS]:
        if var == "gwis_ba":
            da = (ds[var] > 0).astype("float32")
        else:
            da = ds[var]
        coarse[var] = (
            da.coarsen(latitude=PATCH_SIZE, longitude=PATCH_SIZE, boundary="trim")
            .mean()
            .compute()
            .values
        )

    land = (
        ds["lsm"]
        .coarsen(latitude=PATCH_SIZE, longitude=PATCH_SIZE, boundary="trim")
        .mean()
        .compute()
        .values
    )
    months = ds["time"].dt.month.values.astype(int)
    lat_centers, lon_centers = patch_extent(ds)
    oci = {var: ds[var].values.astype(np.float64) for var in OCI_VARS}
    global_signals = {var: global_background_series(ds[var]) for var in FIRE_VARS}
    return coarse, land, months, lat_centers, lon_centers, oci, global_signals


def compute_patch_results(
    coarse: Dict[str, np.ndarray],
    land: np.ndarray,
    months: np.ndarray,
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    oci: Dict[str, np.ndarray],
    global_signals: Dict[str, np.ndarray],
) -> List[PatchResult]:
    oci_d = {var: deseason_by_month(series, months) for var, series in oci.items()}
    global_d = {var: deseason_by_month(series, months) for var, series in global_signals.items()}
    results: List[PatchResult] = []

    for i in range(coarse["gwis_ba"].shape[1]):
        for j in range(coarse["gwis_ba"].shape[2]):
            if land[i, j] < LAND_THRESHOLD:
                continue

            fire = coarse["gwis_ba"][:, i, j]
            fire_mean = float(np.nanmean(fire))
            if fire_mean < ACTIVE_FIRE_THRESHOLD:
                continue

            fire_d = deseason_by_month(fire, months)
            metrics: Dict[str, float] = {}

            for local_var in FIRE_VARS:
                local_d = deseason_by_month(coarse[local_var][:, i, j], months)
                corrs = lag_correlation(local_d, fire_d, LOCAL_LAG_MAX)
                if np.all(np.isnan(corrs)):
                    metrics[f"{local_var}_peak_lag"] = np.nan
                    metrics[f"{local_var}_peak_corr"] = np.nan
                else:
                    peak_lag = int(np.nanargmax(np.abs(corrs)))
                    metrics[f"{local_var}_peak_lag"] = peak_lag
                    metrics[f"{local_var}_peak_corr"] = float(corrs[peak_lag])

            for global_var in FIRE_VARS:
                corrs = lag_correlation(global_d[global_var], fire_d, OCI_LAG_MAX)
                if np.all(np.isnan(corrs)):
                    metrics[f"global_{global_var}_peak_lag"] = np.nan
                    metrics[f"global_{global_var}_peak_corr"] = np.nan
                else:
                    peak_lag = int(np.nanargmax(np.abs(corrs)))
                    metrics[f"global_{global_var}_peak_lag"] = peak_lag
                    metrics[f"global_{global_var}_peak_corr"] = float(corrs[peak_lag])

            for oci_var in OCI_VARS:
                corrs = lag_correlation(oci_d[oci_var], fire_d, OCI_LAG_MAX)
                if np.all(np.isnan(corrs)):
                    metrics[f"{oci_var}_peak_lag"] = np.nan
                    metrics[f"{oci_var}_peak_corr"] = np.nan
                else:
                    peak_lag = int(np.nanargmax(np.abs(corrs)))
                    metrics[f"{oci_var}_peak_lag"] = peak_lag
                    metrics[f"{oci_var}_peak_corr"] = float(corrs[peak_lag])

            # Additive vs interaction example for Nino34.
            nino_lag = int(metrics["oci_nina34_anom_peak_lag"])
            nino_shifted = lag_shift(oci_d["oci_nina34_anom"], nino_lag)
            for local_var in ["vpd", "rel_hum", "t2m_mean"]:
                local_d = deseason_by_month(coarse[local_var][:, i, j], months)
                r2_local = fit_r2(fire_d, np.column_stack([local_d]))
                r2_add = fit_r2(fire_d, np.column_stack([local_d, nino_shifted]))
                r2_int = fit_r2(fire_d, np.column_stack([local_d, nino_shifted, local_d * nino_shifted]))
                metrics[f"{local_var}_r2_local"] = r2_local
                metrics[f"{local_var}_r2_add"] = r2_add
                metrics[f"{local_var}_r2_int"] = r2_int
                metrics[f"{local_var}_delta_int"] = r2_int - r2_add

            results.append(
                PatchResult(
                    patch_i=i,
                    patch_j=j,
                    lat_center=float(lat_centers[i]),
                    lon_center=float(lon_centers[j]),
                    land_fraction=float(land[i, j]),
                    fire_mean=fire_mean,
                    metrics=metrics,
                )
            )

    return results


def write_patch_csv(results: Iterable[PatchResult], output_path: Path) -> None:
    rows = []
    for result in results:
        row = {
            "patch_i": result.patch_i,
            "patch_j": result.patch_j,
            "lat_center": result.lat_center,
            "lon_center": result.lon_center,
            "land_fraction": result.land_fraction,
            "fire_mean": result.fire_mean,
        }
        row.update(result.metrics)
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def values_for(results: List[PatchResult], key: str) -> np.ndarray:
    values = np.array([result.metrics.get(key, np.nan) for result in results], dtype=np.float64)
    return values[np.isfinite(values)]


def matrix_for(results: List[PatchResult], key: str) -> np.ndarray:
    mat = np.full((9, 18), np.nan, dtype=np.float64)
    for result in results:
        mat[result.patch_i, result.patch_j] = result.metrics.get(key, np.nan)
    return mat


def hex_to_rgb01(color: str) -> np.ndarray:
    rgba = np.array(to_rgba(color))
    return rgba[:3]


def blend_colors(color_a: str, color_b: str, n: int) -> List[str]:
    rgb_a = hex_to_rgb01(color_a)
    rgb_b = hex_to_rgb01(color_b)
    colors = []
    for t in np.linspace(0, 1, n):
        rgb = (1 - t) * rgb_a + t * rgb_b
        colors.append(tuple(rgb))
    return colors


def smooth_discrete_density(values: np.ndarray, max_lag: int, sigma: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
    support = np.arange(max_lag + 1, dtype=np.float64)
    if len(values) == 0:
        return support, np.zeros_like(support)

    values = np.asarray(values, dtype=np.int64)
    values = values[(values >= 0) & (values <= max_lag)]
    if len(values) == 0:
        return support, np.zeros_like(support)

    counts = np.bincount(values, minlength=max_lag + 1).astype(np.float64)
    density = counts / counts.sum()

    radius = int(max(2, np.ceil(3 * sigma)))
    kernel_x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(density, kernel, mode="same")
    return support, smooth


def plot_lag_distribution(results: List[PatchResult], output_path: Path) -> None:
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 18.0
    plt.rcParams["axes.labelsize"] = 21.0
    plt.rcParams["xtick.labelsize"] = 18.0
    plt.rcParams["ytick.labelsize"] = 18.0
    fig = plt.figure(figsize=(21, 11))
    ax = fig.add_subplot(111, projection="3d")

    palette = PALETTES[PALETTE_NAME]
    local_positions = np.array([0.0 + 1.12 * idx for idx in range(len(FIRE_VARS))], dtype=np.float64)
    global_positions = np.array([13.3 + 1.12 * idx for idx in range(len(FIRE_VARS))], dtype=np.float64)
    oci_positions = np.array([26.6 + 1.12 * idx for idx in range(len(OCI_VARS))], dtype=np.float64)
    group_defs = [
        ("Local", FIRE_VARS, "", local_positions),
        ("Global", FIRE_VARS, "global_", global_positions),
        ("OCI", OCI_VARS, "", oci_positions),
    ]

    max_density = 0.0

    for group_name, variables, prefix, x_positions in group_defs:
        group_colors = blend_colors(palette[group_name][0], palette[group_name][1], len(variables))
        for x_pos, var_name, color in zip(x_positions, variables, group_colors):
            key = f"{prefix}{var_name}_peak_lag"
            lag_values = values_for(results, key)
            lag_support, density = smooth_discrete_density(lag_values, OCI_LAG_MAX)
            max_density = max(max_density, float(np.nanmax(density)))

            x_curve = np.full_like(lag_support, float(x_pos), dtype=np.float64)
            ax.plot(
                x_curve,
                lag_support,
                density,
                color=color,
                linewidth=1.9,
                alpha=0.98,
            )

            verts = [(float(x_pos), float(lag_support[0]), 0.0)]
            verts.extend((float(x_pos), float(lag), float(dens)) for lag, dens in zip(lag_support, density))
            verts.append((float(x_pos), float(lag_support[-1]), 0.0))
            poly = Poly3DCollection([verts], facecolors=[(*color, 0.22)], edgecolors="none")
            ax.add_collection3d(poly)

    xticks = np.concatenate([gd[3] for gd in group_defs])
    xlabels = [DISPLAY_LABELS[var] for _, vars_, _, _ in group_defs for var in vars_]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=18.0, rotation=70, ha="right", fontweight="bold")
    ax.set_xlim(-0.85, 37.85)

    ax.set_ylim(0, OCI_LAG_MAX)
    ax.set_ylabel("Peak lag (8-day steps)", labelpad=16)
    ax.yaxis.label.set_size(22.0)
    ax.yaxis.label.set_fontweight("bold")

    ax.set_zlim(0, max_density * 1.18 if max_density > 0 else 0.5)
    ax.set_zlabel("Distribution density", labelpad=12)
    ax.zaxis.label.set_size(22.0)
    ax.zaxis.label.set_fontweight("bold")

    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=18.0, pad=10)
    ax.tick_params(axis="y", labelsize=18.0, pad=2)
    ax.tick_params(axis="z", labelsize=18.0, pad=2)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels(), *ax.get_zticklabels()]:
        tick_label.set_fontweight("bold")
    ax.view_init(elev=23, azim=-60)

    ax.xaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    ax.yaxis.pane.set_facecolor((0.99, 0.99, 0.99, 1.0))
    ax.zaxis.pane.set_facecolor((0.985, 0.985, 0.985, 1.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.84, 0.84, 0.84, 0.45)
    ax.yaxis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.35)
    ax.zaxis._axinfo["grid"]["color"] = (0.84, 0.84, 0.84, 0.45)
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0.0
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.12
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0.0
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.12
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0.0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.12

    for boundary in [10.5, 22.5]:
        ax.plot(
            [boundary, boundary],
            [0, 0],
            [0, max_density * 1.12 if max_density > 0 else 0.5],
            color="#9e9e9e",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )

    fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
    ax.set_position([-0.015, 0.01, 1.03, 0.98])
    save_kwargs = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.0}
    fig.savefig(output_path, dpi=240, **save_kwargs)
    fig.savefig(output_path.with_suffix(".svg"), **save_kwargs)
    fig.savefig(output_path.with_suffix(".pdf"), **save_kwargs)
    plt.close(fig)


def plot_lag_maps(results: List[PatchResult], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    panels = [
        ("VPD peak lag", "vpd_peak_lag", LOCAL_LAG_MAX),
        ("RH peak lag", "rel_hum_peak_lag", LOCAL_LAG_MAX),
        ("Nino34 peak lag", "oci_nina34_anom_peak_lag", OCI_LAG_MAX),
        ("PDO peak lag", "oci_pdo_peak_lag", OCI_LAG_MAX),
    ]

    for ax, (title, key, vmax) in zip(axes.flat, panels):
        mat = np.ma.masked_invalid(matrix_for(results, key))
        im = ax.imshow(
            mat,
            extent=[-180, 180, -90, 90],
            origin="upper",
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("8-day steps")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_crosscorr_examples(
    coarse: Dict[str, np.ndarray],
    months: np.ndarray,
    oci: Dict[str, np.ndarray],
    results: List[PatchResult],
    output_path: Path,
) -> None:
    nino_d = deseason_by_month(oci["oci_nina34_anom"], months)
    result_lookup = {(r.patch_i, r.patch_j): r for r in results}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, ((i, j), title) in zip(axes.flat, EXAMPLE_PATCHES):
        fire_d = deseason_by_month(coarse["gwis_ba"][:, i, j], months)
        corrs = lag_correlation(nino_d, fire_d, OCI_LAG_MAX)
        result = result_lookup[(i, j)]
        peak_lag = int(result.metrics["oci_nina34_anom_peak_lag"])
        peak_corr = result.metrics["oci_nina34_anom_peak_corr"]

        ax.plot(np.arange(len(corrs)), corrs, color="#1d91c0", linewidth=2)
        ax.scatter([peak_lag], [peak_corr], color="#d7301f", s=36, zorder=3)
        ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
        ax.set_title(
            f"{title}\n({result.lat_center:.0f}, {result.lon_center:.0f}) "
            f"peak lag={peak_lag}"
        )
        ax.set_xlabel("Lead lag (8-day steps)")
        ax.set_ylabel("Corr(Nino34, fire)")
        ax.grid(alpha=0.25)

    fig.suptitle("The same Nino34 signal peaks at different lags across regions", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_modulation_examples(
    coarse: Dict[str, np.ndarray],
    months: np.ndarray,
    oci: Dict[str, np.ndarray],
    results: List[PatchResult],
    output_path: Path,
) -> None:
    nino_d = deseason_by_month(oci["oci_nina34_anom"], months)
    result_lookup = {(r.patch_i, r.patch_j): r for r in results}
    colors = {"Cold Nino34": "#2b8cbe", "Neutral": "#7f7f7f", "Warm Nino34": "#d95f0e"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, ((i, j), title) in zip(axes.flat, EXAMPLE_PATCHES):
        fire_d = deseason_by_month(coarse["gwis_ba"][:, i, j], months)
        vpd_d = deseason_by_month(coarse["vpd"][:, i, j], months)
        result = result_lookup[(i, j)]
        lag = int(result.metrics["oci_nina34_anom_peak_lag"])
        phase = lag_shift(nino_d, lag)

        q1 = np.nanpercentile(phase, 33)
        q2 = np.nanpercentile(phase, 67)
        phase_masks = {
            "Cold Nino34": phase <= q1,
            "Neutral": (phase > q1) & (phase < q2),
            "Warm Nino34": phase >= q2,
        }

        labels = []
        for phase_name, phase_mask in phase_masks.items():
            centers, means = bin_means(vpd_d[phase_mask], fire_d[phase_mask], n_bins=5)
            slope, _ = fit_slope(vpd_d[phase_mask], fire_d[phase_mask])
            if len(centers) > 0:
                ax.plot(centers, means, marker="o", linewidth=2, color=colors[phase_name])
                labels.append(f"{phase_name}: slope={slope:+.2f}")

        ax.set_title(f"{title}\npeak Nino lag={lag}")
        ax.set_xlabel("Local VPD anomaly")
        ax.set_ylabel("Fire activity anomaly")
        ax.grid(alpha=0.25)
        ax.text(
            0.03,
            0.97,
            "\n".join(labels),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    handles = [
        plt.Line2D([0], [0], color=color, marker="o", linewidth=2, label=label)
        for label, color in colors.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("The local VPD-fire mapping changes under different climate backgrounds", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(results: List[PatchResult], output_path: Path) -> None:
    lines: List[str] = []
    lines.append(f"Valid active patches: {len(results)}")
    lines.append("")

    for group_name, key_list in [
        ("Local", [f"{var}_peak_lag" for var in FIRE_VARS]),
        ("Global", [f"global_{var}_peak_lag" for var in FIRE_VARS]),
        ("OCI", [f"{var}_peak_lag" for var in OCI_VARS]),
    ]:
        vals = np.concatenate([values_for(results, key) for key in key_list])
        lines.append(
            f"{group_name}_all_peak_lag: median={np.median(vals):.2f}, "
            f"q25={np.percentile(vals, 25):.2f}, q75={np.percentile(vals, 75):.2f}"
        )

    lines.append("")
    for local_var in ["vpd", "rel_hum", "t2m_mean"]:
        vals = values_for(results, f"{local_var}_delta_int")
        lines.append(
            f"{local_var}_delta_int: median={np.median(vals):.4f}, "
            f"q75={np.percentile(vals, 75):.4f}, max={np.max(vals):.4f}, "
            f"count_gt_0.01={(vals > 0.01).sum()}"
        )

    output_path.write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    coarse, land, months, lat_centers, lon_centers, oci, global_signals = load_data()
    results = compute_patch_results(coarse, land, months, lat_centers, lon_centers, oci, global_signals)

    write_patch_csv(results, OUTPUT_DIR / "patch_lag_summary.csv")
    write_summary(results, OUTPUT_DIR / "summary.txt")
    plot_lag_distribution(results, OUTPUT_DIR / "lag_distribution.png")
    plot_lag_maps(results, OUTPUT_DIR / "lag_maps.png")
    plot_crosscorr_examples(coarse, months, oci, results, OUTPUT_DIR / "nino_crosscorr_examples.png")
    plot_modulation_examples(coarse, months, oci, results, OUTPUT_DIR / "modulation_examples.png")

    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
