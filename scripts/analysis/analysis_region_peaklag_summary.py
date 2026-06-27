"""
Three-panel summary plot for region-level peak-lag patterns.

Panels:
1. Local drivers (regional means)
2. Global drivers (global background series from the same variables)
3. OCI signals

For each variable x GFED region pair:
- cell color encodes peak lag
- overlaid number is the integer peak lag
- red numbers indicate a negative peak correlation
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config.yaml"
DATA_PATH = Path("/home/dataset-local/QJC/ljt/TeleVIT/SeasFireCube_v3.zarr")
OUTPUT_DIR = Path(__file__).resolve().parent / "region_peaklag_summary"
MAX_LAG = 18

GFED_REGION_NAMES = {
    1: "BONA",
    2: "TENA",
    3: "CEAM",
    4: "NHSA",
    5: "SHSA",
    6: "EURO",
    7: "MIDE",
    8: "NHAF",
    9: "SHAF",
    10: "BOAS",
    11: "CEAS",
    12: "SEAS",
    13: "EQAS",
    14: "AUST",
}

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


def build_fire_series(ds: xr.Dataset, months: np.ndarray) -> Dict[int, np.ndarray]:
    fire_occ = (ds["gwis_ba"] > 0).astype("float32")
    region_ids = list(GFED_REGION_NAMES.keys())
    stacked = xr.concat(
        [
            fire_occ.where(ds["gfed_region"] == rid).mean(
                dim=("latitude", "longitude"),
                skipna=True,
            )
            for rid in region_ids
        ],
        dim=xr.DataArray(region_ids, dims=("region_id",), name="region_id"),
    ).compute()
    out: Dict[int, np.ndarray] = {}
    for idx, rid in enumerate(region_ids):
        out[rid] = deseason_by_month(stacked.isel(region_id=idx).values.astype(np.float64), months)
    return out


def build_local_series(
    ds: xr.Dataset,
    fire_vars: List[str],
    log_transform_vars: set[str],
    months: np.ndarray,
) -> Dict[str, Dict[int, np.ndarray]]:
    out: Dict[str, Dict[int, np.ndarray]] = {}
    region_ids = list(GFED_REGION_NAMES.keys())
    for var in fire_vars:
        da = apply_transform(var, ds[var], log_transform_vars)
        stacked = xr.concat(
            [
                da.where(ds["gfed_region"] == rid).mean(
                    dim=("latitude", "longitude"),
                    skipna=True,
                )
                for rid in region_ids
            ],
            dim=xr.DataArray(region_ids, dims=("region_id",), name="region_id"),
        ).compute()
        var_dict: Dict[int, np.ndarray] = {}
        for idx, rid in enumerate(region_ids):
            var_dict[rid] = deseason_by_month(stacked.isel(region_id=idx).values.astype(np.float64), months)
        out[var] = var_dict
    return out


def build_global_series(
    ds: xr.Dataset,
    fire_vars: List[str],
    log_transform_vars: set[str],
    months: np.ndarray,
) -> Dict[str, np.ndarray]:
    weights = xr.DataArray(
        np.cos(np.deg2rad(ds["latitude"].values.astype(np.float64))),
        coords={"latitude": ds["latitude"]},
        dims=("latitude",),
    )
    stacked = xr.concat(
        [
            xr.where(
                (xr.where(np.isfinite(apply_transform(var, ds[var], log_transform_vars)), 1.0, 0.0) * weights).sum(
                    dim=("latitude", "longitude")
                ) > 0,
                (
                    apply_transform(var, ds[var], log_transform_vars).fillna(0.0) * weights
                ).sum(dim=("latitude", "longitude"))
                / (
                    xr.where(np.isfinite(apply_transform(var, ds[var], log_transform_vars)), 1.0, 0.0) * weights
                ).sum(dim=("latitude", "longitude")),
                np.nan,
            )
            for var in fire_vars
        ],
        dim=xr.DataArray(fire_vars, dims=("variable",), name="variable"),
    ).compute()
    out: Dict[str, np.ndarray] = {}
    for idx, var in enumerate(fire_vars):
        out[var] = deseason_by_month(stacked.isel(variable=idx).values.astype(np.float64), months)
    return out


def build_oci_series(ds: xr.Dataset, oci_vars: List[str], months: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        var: deseason_by_month(ds[var].values.astype(np.float64), months)
        for var in oci_vars
    }


def summarize_group(
    group_name: str,
    variables: List[str],
    region_fire: Dict[int, np.ndarray],
    series_by_var,
) -> List[dict]:
    rows: List[dict] = []
    for var in variables:
        for rid, fire_series in region_fire.items():
            predictor = series_by_var[var][rid] if group_name == "Local" else series_by_var[var]
            corrs = lag_correlation(predictor, fire_series, MAX_LAG)
            if np.all(np.isnan(corrs)):
                peak_lag = np.nan
                peak_corr = np.nan
            else:
                peak_lag = int(np.nanargmax(np.abs(corrs)))
                peak_corr = float(corrs[peak_lag])
            rows.append(
                {
                    "group": group_name,
                    "variable": var,
                    "label": DISPLAY_LABELS.get(var, var),
                    "region_id": rid,
                    "region_name": GFED_REGION_NAMES[rid],
                    "peak_lag": peak_lag,
                    "peak_corr": peak_corr,
                }
            )
    return rows


def build_matrix(rows: List[dict], var_order: List[str]) -> tuple[np.ndarray, np.ndarray]:
    region_order = list(GFED_REGION_NAMES.values())
    x_map = {name: idx for idx, name in enumerate(region_order)}
    y_map = {var: idx for idx, var in enumerate(var_order)}
    lag_mat = np.full((len(var_order), len(region_order)), np.nan, dtype=np.float64)
    corr_mat = np.full((len(var_order), len(region_order)), np.nan, dtype=np.float64)

    for row in rows:
        y = y_map[row["variable"]]
        x = x_map[row["region_name"]]
        lag_mat[y, x] = row["peak_lag"]
        corr_mat[y, x] = row["peak_corr"]
    return lag_mat, corr_mat


def plot_panel(ax, rows: List[dict], group_title: str, var_order: List[str], cmap, norm):
    region_order = list(GFED_REGION_NAMES.values())
    y_labels = [DISPLAY_LABELS.get(v, v) for v in var_order]
    lag_mat, corr_mat = build_matrix(rows, var_order)

    im = ax.imshow(
        lag_mat,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_title(group_title, fontsize=18.5, pad=12)
    ax.set_xticks(np.arange(len(region_order)))
    ax.set_xticklabels(region_order, rotation=45, ha="right", fontsize=16.5)
    ax.set_yticks(np.arange(len(var_order)))
    ax.set_yticklabels(y_labels, fontsize=16.5)

    ax.set_xticks(np.arange(-0.5, len(region_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(var_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(lag_mat.shape[0]):
        for j in range(lag_mat.shape[1]):
            lag = lag_mat[i, j]
            corr = corr_mat[i, j]
            if not np.isfinite(lag):
                continue
            text_color = "#8b1e2d" if np.isfinite(corr) and corr < 0 else "#1f2833"
            ax.text(
                j,
                i,
                f"{int(lag)}",
                ha="center",
                va="center",
                fontsize=15.6,
                color=text_color,
                fontweight="bold",
            )

    for spine in ax.spines.values():
        spine.set_color("#9ba7b2")
        spine.set_linewidth(0.8)
    return im


def plot_summary(rows: List[dict], fire_vars: List[str], oci_vars: List[str], output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15.5,
            "axes.titlesize": 18.5,
            "axes.labelsize": 18,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    cmap = plt.get_cmap("YlGnBu")
    norm = plt.Normalize(0, MAX_LAG)
    fig, axes = plt.subplots(1, 3, figsize=(20, 8.8), constrained_layout=False)

    local_rows = [r for r in rows if r["group"] == "Local"]
    global_rows = [r for r in rows if r["group"] == "Global"]
    oci_rows = [r for r in rows if r["group"] == "OCI"]

    im = plot_panel(axes[0], local_rows, "Local Drivers", fire_vars, cmap, norm)
    plot_panel(axes[1], global_rows, "Global Drivers", fire_vars, cmap, norm)
    plot_panel(axes[2], oci_rows, "OCI Signals", oci_vars, cmap, norm)

    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    for ax in axes:
        ax.set_xlabel("")

    fig.subplots_adjust(left=0.06, right=0.985, top=0.945, bottom=0.085, wspace=0.18)

    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02, fraction=0.032, aspect=42)
    cbar.set_label("Peak lag", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    fig.savefig(output_path, dpi=260, facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
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

    ds = xr.open_zarr(
        str(DATA_PATH),
        consolidated=True,
    )[["gwis_ba", "gfed_region", *fire_vars, *oci_vars]]
    months = ds["time"].dt.month.values.astype(int)

    region_fire = build_fire_series(ds, months)
    local_series = build_local_series(ds, fire_vars, log_transform_vars, months)
    global_series = build_global_series(ds, fire_vars, log_transform_vars, months)
    oci_series = build_oci_series(ds, oci_vars, months)

    rows: List[dict] = []
    rows.extend(summarize_group("Local", fire_vars, region_fire, local_series))
    rows.extend(summarize_group("Global", fire_vars, region_fire, global_series))
    rows.extend(summarize_group("OCI", oci_vars, region_fire, oci_series))

    write_summary(rows, OUTPUT_DIR / "region_peaklag_summary.csv")
    plot_summary(rows, fire_vars, oci_vars, OUTPUT_DIR / "region_peaklag_summary.png")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
