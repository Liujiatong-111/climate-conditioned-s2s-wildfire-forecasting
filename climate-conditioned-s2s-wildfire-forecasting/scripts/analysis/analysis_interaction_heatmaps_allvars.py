"""
All-variable interaction heatmaps for local x global and local x OCI pairs.

Outputs:
- two 10x10 heatmaps
- a combined two-panel figure
- csv summary for every variable pair

Metric:
- cell color: median Delta R^2 = R^2(interaction) - R^2(additive)
- cell text: percentage of active patches with Delta R^2 > 0.01
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from matplotlib.colors import TwoSlopeNorm


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config.yaml"
DATA_PATH = Path(os.environ.get("SEASFIRE_ZARR", "data/SeasFireCube_v3.zarr"))
PATCH_SUMMARY_PATH = PROJECT_DIR / "analyze" / "lag_modulation_quicklook" / "patch_lag_summary.csv"
OUTPUT_DIR = PROJECT_DIR / "analyze" / "interaction_heatmaps_allvars"

PATCH_SIZE = 80
DELTA_THRESHOLD = 0.01

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


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_transform(name: str, data: xr.DataArray, log_transform_vars: set[str]) -> xr.DataArray:
    if name in log_transform_vars:
        return np.log1p(data.clip(min=0.0))
    return data


def deseason_by_month(x: np.ndarray, months: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = x.copy()
    for month in range(1, 13):
        idx = months == month
        if not np.any(idx):
            continue
        vals = x[idx]
        if not np.any(np.isfinite(vals)):
            y[idx] = np.nan
            continue
        y[idx] = vals - np.nanmean(vals)
    if not np.any(np.isfinite(y)):
        return np.full_like(y, np.nan, dtype=np.float64)
    std = np.nanstd(y)
    if not np.isfinite(std) or std < 1e-8:
        return np.zeros_like(y)
    return y / std


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


def align_leading_signal(x: np.ndarray, lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if lag <= 0:
        return x.copy()
    return np.concatenate([np.full(lag, np.nan), x[:-lag]])


def global_background_series(da: xr.DataArray) -> np.ndarray:
    weights = xr.DataArray(
        np.cos(np.deg2rad(da["latitude"].values.astype(np.float64))),
        coords={"latitude": da["latitude"]},
        dims=("latitude",),
    )
    ts = da.weighted(weights).mean(dim=("latitude", "longitude")).compute().values
    return np.asarray(ts, dtype=np.float64)


def coarsen_mean(da: xr.DataArray) -> np.ndarray:
    return (
        da.coarsen(latitude=PATCH_SIZE, longitude=PATCH_SIZE, boundary="trim")
        .mean()
        .compute()
        .values
    )


def load_patch_rows() -> List[dict]:
    with PATCH_SUMMARY_PATH.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        out = dict(row)
        out["patch_i"] = int(row["patch_i"])
        out["patch_j"] = int(row["patch_j"])
        parsed.append(out)
    return parsed


def load_data(
    fire_vars: List[str],
    oci_vars: List[str],
    log_transform_vars: set[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    ds = xr.open_zarr(
        DATA_PATH,
        consolidated=True,
    )[["gwis_ba", *fire_vars, *oci_vars]]

    months = ds["time"].dt.month.values.astype(int)
    coarse = {}
    coarse["fire_fraction"] = coarsen_mean((ds["gwis_ba"] > 0).astype("float32"))
    for var in fire_vars:
        coarse[var] = coarsen_mean(apply_transform(var, ds[var], log_transform_vars))

    global_series = {
        var: deseason_by_month(global_background_series(apply_transform(var, ds[var], log_transform_vars)), months)
        for var in fire_vars
    }
    oci_series = {
        var: deseason_by_month(ds[var].values.astype(np.float64), months)
        for var in oci_vars
    }
    return coarse, global_series, oci_series, months


def pair_delta_r2(
    fire_d: np.ndarray,
    local_d: np.ndarray,
    climate_d: np.ndarray,
) -> float:
    r2_add = fit_r2(fire_d, np.column_stack([local_d, climate_d]))
    r2_int = fit_r2(fire_d, np.column_stack([local_d, climate_d, local_d * climate_d]))
    if not np.isfinite(r2_add) or not np.isfinite(r2_int):
        return np.nan
    return float(r2_int - r2_add)


def compute_interactions(
    patch_rows: List[dict],
    coarse: Dict[str, np.ndarray],
    global_series: Dict[str, np.ndarray],
    oci_series: Dict[str, np.ndarray],
    months: np.ndarray,
    fire_vars: List[str],
    oci_vars: List[str],
) -> List[dict]:
    rows = []
    local_cache: Dict[Tuple[str, int, int], np.ndarray] = {}
    fire_cache: Dict[Tuple[int, int], np.ndarray] = {}

    for patch in patch_rows:
        i = patch["patch_i"]
        j = patch["patch_j"]
        fire_cache[(i, j)] = deseason_by_month(coarse["fire_fraction"][:, i, j], months)
        for local_var in fire_vars:
            local_cache[(local_var, i, j)] = deseason_by_month(coarse[local_var][:, i, j], months)

    for local_var in fire_vars:
        for climate_var in fire_vars:
            deltas = []
            for patch in patch_rows:
                i = patch["patch_i"]
                j = patch["patch_j"]
                lag_raw = patch.get(f"global_{climate_var}_peak_lag", "")
                if lag_raw in ("", "nan"):
                    continue
                lag = int(float(lag_raw))
                climate_aligned = align_leading_signal(global_series[climate_var], lag)
                delta = pair_delta_r2(
                    fire_cache[(i, j)],
                    local_cache[(local_var, i, j)],
                    climate_aligned,
                )
                if np.isfinite(delta):
                    deltas.append(delta)
            rows.append(
                summarize_pair(
                    group="Global",
                    local_var=local_var,
                    climate_var=climate_var,
                    deltas=np.asarray(deltas, dtype=np.float64),
                )
            )

        for climate_var in oci_vars:
            deltas = []
            for patch in patch_rows:
                i = patch["patch_i"]
                j = patch["patch_j"]
                lag_raw = patch.get(f"{climate_var}_peak_lag", "")
                if lag_raw in ("", "nan"):
                    continue
                lag = int(float(lag_raw))
                climate_aligned = align_leading_signal(oci_series[climate_var], lag)
                delta = pair_delta_r2(
                    fire_cache[(i, j)],
                    local_cache[(local_var, i, j)],
                    climate_aligned,
                )
                if np.isfinite(delta):
                    deltas.append(delta)
            rows.append(
                summarize_pair(
                    group="OCI",
                    local_var=local_var,
                    climate_var=climate_var,
                    deltas=np.asarray(deltas, dtype=np.float64),
                )
            )
    return rows


def summarize_pair(group: str, local_var: str, climate_var: str, deltas: np.ndarray) -> dict:
    if len(deltas) == 0:
        return {
            "group": group,
            "local_var": local_var,
            "climate_var": climate_var,
            "median_delta_r2": np.nan,
            "mean_delta_r2": np.nan,
            "q75_delta_r2": np.nan,
            "positive_fraction": np.nan,
            "count_valid": 0,
        }
    return {
        "group": group,
        "local_var": local_var,
        "climate_var": climate_var,
        "median_delta_r2": float(np.nanmedian(deltas)),
        "mean_delta_r2": float(np.nanmean(deltas)),
        "q75_delta_r2": float(np.nanpercentile(deltas, 75)),
        "positive_fraction": float(np.mean(deltas > DELTA_THRESHOLD)),
        "count_valid": int(len(deltas)),
    }


def build_matrix(rows: List[dict], fire_vars: List[str], climate_vars: List[str], key: str) -> np.ndarray:
    mat = np.full((len(fire_vars), len(climate_vars)), np.nan, dtype=np.float64)
    row_map = {var: idx for idx, var in enumerate(fire_vars)}
    col_map = {var: idx for idx, var in enumerate(climate_vars)}
    for row in rows:
        i = row_map[row["local_var"]]
        j = col_map[row["climate_var"]]
        mat[i, j] = row[key]
    return mat


def choose_norm(mats: List[np.ndarray]) -> TwoSlopeNorm:
    vals = np.concatenate([mat[np.isfinite(mat)] for mat in mats if np.any(np.isfinite(mat))])
    if len(vals) == 0:
        return TwoSlopeNorm(vmin=-0.01, vcenter=0.0, vmax=0.01)
    vmax = max(np.percentile(np.abs(vals), 95), 0.01)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def annotate_cells(ax, color_mat: np.ndarray, text_mat: np.ndarray) -> None:
    for i in range(color_mat.shape[0]):
        for j in range(color_mat.shape[1]):
            val = color_mat[i, j]
            if not np.isfinite(val):
                continue
            pct = text_mat[i, j]
            text_color = "white" if abs(val) > np.nanmax(np.abs(color_mat)) * 0.45 else "#20303a"
            ax.text(
                j,
                i,
                f"{pct*100:.0f}%",
                ha="center",
                va="center",
                fontsize=8.4,
                color=text_color,
                fontweight="bold" if pct >= 0.3 else "normal",
            )


def plot_heatmap(
    ax,
    color_mat: np.ndarray,
    text_mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    norm: TwoSlopeNorm,
) -> None:
    im = ax.imshow(color_mat, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=13)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    annotate_cells(ax, color_mat, text_mat)
    for spine in ax.spines.values():
        spine.set_color("#97a6b2")
        spine.set_linewidth(0.8)
    return im


def plot_all(
    global_rows: List[dict],
    oci_rows: List[dict],
    fire_vars: List[str],
    oci_vars: List[str],
) -> None:
    row_labels = [DISPLAY_LABELS[v] for v in fire_vars]
    global_labels = [f"G-{DISPLAY_LABELS[v]}" for v in fire_vars]
    oci_labels = [DISPLAY_LABELS[v] for v in oci_vars]

    global_color = build_matrix(global_rows, fire_vars, fire_vars, "median_delta_r2")
    global_text = build_matrix(global_rows, fire_vars, fire_vars, "positive_fraction")
    oci_color = build_matrix(oci_rows, fire_vars, oci_vars, "median_delta_r2")
    oci_text = build_matrix(oci_rows, fire_vars, oci_vars, "positive_fraction")

    norm = choose_norm([global_color, oci_color])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(19, 8.8), constrained_layout=True)
    im = plot_heatmap(
        axes[0],
        global_color,
        global_text,
        row_labels,
        global_labels,
        "Local fire drivers x Global background variables",
        norm,
    )
    plot_heatmap(
        axes[1],
        oci_color,
        oci_text,
        row_labels,
        oci_labels,
        "Local fire drivers x OCI signals",
        norm,
    )
    axes[0].set_ylabel("Local fire driver")
    axes[0].set_xlabel("Global background variable")
    axes[1].set_xlabel("OCI signal")

    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Median ΔR² (interaction - additive)")

    fig.suptitle("All-variable interaction heatmaps for climate modulation", fontsize=15, y=1.02)
    fig.text(
        0.5,
        -0.01,
        f"Cell color shows median ΔR² across active patches. Cell text shows the percentage of patches with ΔR² > {DELTA_THRESHOLD:.2f}.",
        ha="center",
        fontsize=10,
        color="#44515c",
    )
    fig.savefig(OUTPUT_DIR / "interaction_heatmaps_allvars.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    for name, color_mat, text_mat, col_labels, title, xlabel in [
        ("interaction_heatmap_local_global.png", global_color, global_text, global_labels, "Local x Global interaction heatmap", "Global background variable"),
        ("interaction_heatmap_local_oci.png", oci_color, oci_text, oci_labels, "Local x OCI interaction heatmap", "OCI signal"),
    ]:
        fig, ax = plt.subplots(figsize=(10.6, 8.8), constrained_layout=True)
        im = plot_heatmap(ax, color_mat, text_mat, row_labels, col_labels, title, norm)
        ax.set_ylabel("Local fire driver")
        ax.set_xlabel(xlabel)
        cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
        cbar.set_label("Median ΔR² (interaction - additive)")
        fig.text(
            0.5,
            0.01,
            f"Cell text: % patches with ΔR² > {DELTA_THRESHOLD:.2f}",
            ha="center",
            fontsize=10,
            color="#44515c",
        )
        fig.savefig(OUTPUT_DIR / name, dpi=260, bbox_inches="tight")
        plt.close(fig)


def write_summary(rows: List[dict], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    fire_vars = list(config["data"]["fire_vars"])
    oci_vars = list(config["data"]["oci_vars"])
    log_transform_vars = set(config["data"].get("log_transform_vars", []))

    patch_rows = load_patch_rows()
    coarse, global_series, oci_series, months = load_data(fire_vars, oci_vars, log_transform_vars)
    rows = compute_interactions(patch_rows, coarse, global_series, oci_series, months, fire_vars, oci_vars)

    write_summary(rows, OUTPUT_DIR / "interaction_heatmaps_allvars_summary.csv")
    global_rows = [row for row in rows if row["group"] == "Global"]
    oci_rows = [row for row in rows if row["group"] == "OCI"]
    plot_all(global_rows, oci_rows, fire_vars, oci_vars)
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
