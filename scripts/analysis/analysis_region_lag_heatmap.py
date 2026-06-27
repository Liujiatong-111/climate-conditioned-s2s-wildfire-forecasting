"""
Region x lag heatmap for global climate signals vs future fire activity.

Uses GFED regions already stored in the SeasFire cube and computes
lead-lag correlations between global climate indices and regional fire activity.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DATA_PATH = Path("/home/dataset-local/QJC/ljt/TeleVIT/SeasFireCube_v3.zarr")
OUTPUT_DIR = Path(__file__).resolve().parent / "region_lag_heatmap"
MAX_LAG = 18

SIGNALS = [
    "oci_nina34_anom",
    "oci_pdo",
    "oci_soi",
    "oci_nao",
]
DISPLAY_LABELS = {
    "oci_nina34_anom": "Nino34",
    "oci_pdo": "PDO",
    "oci_soi": "SOI",
    "oci_nao": "NAO",
}
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


def regional_fire_series(ds: xr.Dataset, region_id: int) -> np.ndarray:
    region_mask = ds["gfed_region"] == region_id
    fire_occ = (ds["gwis_ba"] > 0).astype("float32")
    fire_reg = fire_occ.where(region_mask).mean(dim=("latitude", "longitude"), skipna=True)
    return fire_reg.compute().values.astype(np.float64)


def compute_heatmap(ds: xr.Dataset, months: np.ndarray) -> tuple[Dict[str, np.ndarray], List[dict]]:
    heatmaps: Dict[str, np.ndarray] = {}
    summary_rows: List[dict] = []

    signal_anom = {
        signal: deseason_by_month(ds[signal].values.astype(np.float64), months)
        for signal in SIGNALS
    }

    region_ids = list(GFED_REGION_NAMES.keys())
    region_series = {
        rid: deseason_by_month(regional_fire_series(ds, rid), months)
        for rid in region_ids
    }

    for signal in SIGNALS:
        mat = np.full((len(region_ids), MAX_LAG + 1), np.nan, dtype=np.float64)
        for row_idx, rid in enumerate(region_ids):
            corrs = lag_correlation(signal_anom[signal], region_series[rid], MAX_LAG)
            mat[row_idx, :] = corrs
            if np.all(np.isnan(corrs)):
                peak_lag = np.nan
                peak_corr = np.nan
            else:
                peak_lag = int(np.nanargmax(np.abs(corrs)))
                peak_corr = float(corrs[peak_lag])
            summary_rows.append(
                {
                    "signal": signal,
                    "region_id": rid,
                    "region_name": GFED_REGION_NAMES[rid],
                    "peak_lag": peak_lag,
                    "peak_corr": peak_corr,
                }
            )
        heatmaps[signal] = mat

    return heatmaps, summary_rows


def plot_single_signal_heatmap(signal: str, mat: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.45,
        vmax=0.45,
        interpolation="nearest",
    )

    peak_lags = np.nanargmax(np.abs(mat), axis=1)
    ax.scatter(peak_lags, np.arange(mat.shape[0]), s=24, c="black", edgecolors="white", linewidths=0.6)

    ax.set_title(f"{DISPLAY_LABELS[signal]}: region x lag correlation with future fire activity")
    ax.set_xlabel("Lag (8-day steps)")
    ax.set_ylabel("GFED region")
    ax.set_xticks(np.arange(MAX_LAG + 1))
    ax.set_yticks(np.arange(len(GFED_REGION_NAMES)))
    ax.set_yticklabels(list(GFED_REGION_NAMES.values()))

    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Correlation")

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_multi_signal_heatmap(heatmaps: Dict[str, np.ndarray], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    region_names = list(GFED_REGION_NAMES.values())

    for ax, signal in zip(axes.flat, SIGNALS):
        mat = heatmaps[signal]
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-0.45,
            vmax=0.45,
            interpolation="nearest",
        )
        peak_lags = np.nanargmax(np.abs(mat), axis=1)
        ax.scatter(peak_lags, np.arange(mat.shape[0]), s=20, c="black", edgecolors="white", linewidths=0.5)
        ax.set_title(DISPLAY_LABELS[signal])
        ax.set_xticks(np.arange(0, MAX_LAG + 1, 2))
        ax.set_yticks(np.arange(len(region_names)))
        ax.set_yticklabels(region_names)

    fig.supxlabel("Lag (8-day steps)")
    fig.supylabel("GFED region")
    fig.suptitle("Region x lag correlation heatmaps for global climate signals", y=0.98)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def write_summary(rows: List[dict], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(DATA_PATH, consolidated=True)[["gwis_ba", "gfed_region", *SIGNALS]]
    months = ds["time"].dt.month.values.astype(int)

    heatmaps, rows = compute_heatmap(ds, months)
    plot_single_signal_heatmap(
        signal="oci_nina34_anom",
        mat=heatmaps["oci_nina34_anom"],
        output_path=OUTPUT_DIR / "region_lag_heatmap_nino34.png",
    )
    plot_multi_signal_heatmap(
        heatmaps=heatmaps,
        output_path=OUTPUT_DIR / "region_lag_heatmap_4signals.png",
    )
    write_summary(rows, OUTPUT_DIR / "region_lag_heatmap_summary.csv")

    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
