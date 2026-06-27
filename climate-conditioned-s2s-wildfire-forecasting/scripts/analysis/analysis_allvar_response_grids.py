"""
All-variable response grids in the same style as the formal modulation figure.

Outputs two 10x10 figures:
- local fire drivers x global background variables
- local fire drivers x OCI signals

Each mini-panel shows low / neutral / high climate regime response curves with
uncertainty bands, using pooled samples from the most climate-sensitive patches.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config.yaml"
DATA_PATH = Path(os.environ.get("SEASFIRE_ZARR", "data/SeasFireCube_v3.zarr"))
OUTPUT_DIR = PROJECT_DIR / "analyze" / "interaction_heatmaps_allvars"
LOCALWISE_PNG_DIR = OUTPUT_DIR / "localwise_response_grids_png"
LOCALWISE_SVG_DIR = OUTPUT_DIR / "localwise_response_grids_svg"
LOCALWISE_5LINE_PNG_DIR = OUTPUT_DIR / "localwise_response_grids_5line_png"
LOCALWISE_5LINE_SVG_DIR = OUTPUT_DIR / "localwise_response_grids_5line_svg"

PATCH_SIZE = 80
TOP_PATCHES_PER_SIGNAL = 10
BIN_EDGES = np.linspace(-2.4, 2.4, 7)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

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

PHASE_COLORS = {
    "Low regime": "#2b8cbe",
    "Neutral": "#7f7f7f",
    "High regime": "#d95f0e",
}
PHASE_COLORS_5 = {
    "Very low": "#225ea8",
    "Low": "#41b6c4",
    "Neutral": "#7f7f7f",
    "High": "#fdae6b",
    "Very high": "#d95f0e",
}

PNG_DPI = 320


def resolve_patch_summary_path() -> Path:
    candidates = [
        PROJECT_DIR / "analyze" / "lag_modulation_quicklook" / "patch_lag_summary.csv",
        PROJECT_DIR / "analyze" / "lag_modulation_quicklook_lag48" / "patch_lag_summary.csv",
        PROJECT_DIR / "analyze" / "lag_modulation_quicklook_lag60" / "patch_lag_summary.csv",
        PROJECT_DIR / "analyze" / "lag_modulation_quicklook_lag36" / "patch_lag_summary.csv",
        PROJECT_DIR / "analyze" / "lag_modulation_quicklook_lag18" / "patch_lag_summary.csv",
        PROJECT_DIR / "analyze" / "lag_modulation_quicklook_lag100" / "patch_lag_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find patch_lag_summary.csv in any lag_modulation_quicklook directory.")


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


def align_leading_signal(x: np.ndarray, lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if lag <= 0:
        return x.copy()
    return np.concatenate([np.full(lag, np.nan), x[:-lag]])


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return np.nan
    x_valid = x[mask]
    y_valid = y[mask]
    if np.std(x_valid) < 1e-8:
        return np.nan
    X = np.column_stack([np.ones(len(x_valid)), x_valid])
    beta, *_ = np.linalg.lstsq(X, y_valid, rcond=None)
    return float(beta[1])


def coarsen_mean(da: xr.DataArray) -> np.ndarray:
    return (
        da.coarsen(latitude=PATCH_SIZE, longitude=PATCH_SIZE, boundary="trim")
        .mean()
        .compute()
        .values
    )


def global_background_series(da: xr.DataArray) -> np.ndarray:
    weights = xr.DataArray(
        np.cos(np.deg2rad(da["latitude"].values.astype(np.float64))),
        coords={"latitude": da["latitude"]},
        dims=("latitude",),
    )
    filled = da.fillna(0.0)
    valid = xr.where(np.isfinite(da), 1.0, 0.0)
    numerator = (filled * weights).sum(dim=("latitude", "longitude"))
    denominator = (valid * weights).sum(dim=("latitude", "longitude"))
    ts = xr.where(denominator > 0, numerator / denominator, np.nan).compute().values
    return np.asarray(ts, dtype=np.float64)


def load_patch_rows() -> List[dict]:
    patch_summary_path = resolve_patch_summary_path()
    with patch_summary_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        out = dict(row)
        out["patch_i"] = int(row["patch_i"])
        out["patch_j"] = int(row["patch_j"])
        out["fire_mean"] = float(row["fire_mean"])
        parsed.append(out)
    return parsed


def load_data(
    fire_vars: List[str],
    oci_vars: List[str],
    log_transform_vars: set[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    ds = xr.open_zarr(
        str(DATA_PATH),
        consolidated=True,
    )[["gwis_ba", *fire_vars, *oci_vars]]

    months = ds["time"].dt.month.values.astype(int)

    coarse = {
        "fire_occurrence": (coarsen_mean((ds["gwis_ba"] > 0).astype("float32")) > 0).astype(np.float64)
    }
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


def select_top_patches(rows: List[dict], climate_var: str, prefix: str = "", top_k: int = TOP_PATCHES_PER_SIGNAL) -> List[dict]:
    corr_key = f"{prefix}{climate_var}_peak_corr"
    lag_key = f"{prefix}{climate_var}_peak_lag"
    valid = []
    for row in rows:
        corr_raw = row.get(corr_key, "")
        lag_raw = row.get(lag_key, "")
        if corr_raw in ("", "nan") or lag_raw in ("", "nan"):
            continue
        out = dict(row)
        out[corr_key] = float(corr_raw)
        out[lag_key] = int(float(lag_raw))
        valid.append(out)
    valid.sort(key=lambda r: abs(r[corr_key]), reverse=True)
    return valid[:top_k]


def binned_event_rate(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = []
    probs = []
    lowers = []
    uppers = []
    for left, right in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = np.isfinite(x) & np.isfinite(y)
        if right == BIN_EDGES[-1]:
            mask &= (x >= left) & (x <= right)
        else:
            mask &= (x >= left) & (x < right)
        n = int(mask.sum())
        if n < 8:
            continue
        p = float(np.mean(y[mask]))
        se = np.sqrt(max(p * (1.0 - p), 0.0) / n)
        centers.append(0.5 * (left + right))
        probs.append(p)
        lowers.append(max(0.0, p - 1.96 * se))
        uppers.append(min(1.0, p + 1.96 * se))
    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(probs, dtype=np.float64),
        np.asarray(lowers, dtype=np.float64),
        np.asarray(uppers, dtype=np.float64),
    )


def panel_samples(
    patch_subset: List[dict],
    coarse: Dict[str, np.ndarray],
    months: np.ndarray,
    local_var: str,
    climate_series: np.ndarray,
    lag_key: str,
) -> Dict[str, np.ndarray]:
    local_all = []
    fire_all = []
    phase_all = []
    lags = []

    for row in patch_subset:
        i = row["patch_i"]
        j = row["patch_j"]
        lag = int(row[lag_key])
        local_d = deseason_by_month(coarse[local_var][:, i, j], months)
        fire_y = coarse["fire_occurrence"][:, i, j].astype(np.float64)
        phase = align_leading_signal(climate_series, lag)
        mask = np.isfinite(local_d) & np.isfinite(fire_y) & np.isfinite(phase)
        if mask.sum() < 30:
            continue
        local_all.append(local_d[mask])
        fire_all.append(fire_y[mask])
        phase_all.append(phase[mask])
        lags.append(lag)

    if not local_all:
        return {
            "local": np.asarray([]),
            "fire": np.asarray([]),
            "phase": np.asarray([]),
            "lags": np.asarray([]),
        }

    return {
        "local": np.concatenate(local_all),
        "fire": np.concatenate(fire_all),
        "phase": np.concatenate(phase_all),
        "lags": np.asarray(lags, dtype=np.float64),
    }


def build_curves_from_masks(
    samples: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
) -> Tuple[Dict[str, dict], dict]:
    local = samples["local"]
    fire = samples["fire"]
    if len(local) == 0:
        return {}, {"median_lag": np.nan, "n_samples": 0}

    curves = {}
    for phase_name, mask in masks.items():
        x = local[mask]
        y = fire[mask]
        centers, probs, lower, upper = binned_event_rate(x, y)
        curves[phase_name] = {
            "centers": centers,
            "probs": probs,
            "lower": lower,
            "upper": upper,
            "slope": fit_slope(x, y),
            "n": int(mask.sum()),
        }
    summary = {
        "median_lag": float(np.nanmedian(samples["lags"])) if len(samples["lags"]) else np.nan,
        "n_samples": int(len(local)),
    }
    return curves, summary


def build_curves(samples: Dict[str, np.ndarray]) -> Tuple[Dict[str, dict], dict]:
    phase = samples["phase"]
    local = samples["local"]
    if len(local) == 0:
        return {}, {"median_lag": np.nan, "n_samples": 0}
    q1 = np.nanpercentile(phase, 33.0)
    q2 = np.nanpercentile(phase, 67.0)
    masks = {
        "Low regime": phase <= q1,
        "Neutral": (phase > q1) & (phase < q2),
        "High regime": phase >= q2,
    }
    return build_curves_from_masks(samples, masks)


def infer_panel_ylim(curves: Dict[str, dict], phase_names: Sequence[str]) -> Tuple[float, float]:
    values = []
    for phase_name in phase_names:
        curve = curves.get(phase_name, {})
        for key in ("lower", "upper", "probs"):
            arr = np.asarray(curve.get(key, np.asarray([])), dtype=np.float64)
            if arr.size:
                values.append(arr[np.isfinite(arr)])

    if not values:
        return 0.0, 1.0

    finite = np.concatenate([arr for arr in values if arr.size])
    if finite.size == 0:
        return 0.0, 1.0

    y_min = float(np.nanmin(finite))
    y_max = float(np.nanmax(finite))
    span = y_max - y_min

    if not np.isfinite(span) or span < 1e-8:
        center = y_min if np.isfinite(y_min) else 0.5
        half_span = 0.04
        return max(0.0, center - half_span), min(1.0, center + half_span)

    pad = max(0.015, 0.10 * span)
    y_low = max(0.0, y_min - pad)
    y_high = min(1.0, y_max + pad)

    min_span = 0.08
    if y_high - y_low < min_span:
        center = 0.5 * (y_high + y_low)
        half_span = 0.5 * min_span
        y_low = max(0.0, center - half_span)
        y_high = min(1.0, center + half_span)
        if y_high - y_low < min_span:
            if y_low <= 0.0:
                y_high = min(1.0, y_low + min_span)
            else:
                y_low = max(0.0, y_high - min_span)

    return y_low, y_high


def choose_panel_yticks(y_low: float, y_high: float) -> Tuple[float, float, np.ndarray]:
    low = float(np.floor(y_low * 10.0) / 10.0)
    high = float(np.ceil(y_high * 10.0) / 10.0)
    low = min(max(low, 0.0), 1.0)
    high = min(max(high, 0.0), 1.0)

    if high <= low:
        if high >= 1.0:
            low = max(0.0, high - 0.1)
        else:
            high = min(1.0, low + 0.1)

    span = high - low
    if span <= 0.2:
        ticks = np.asarray([low, high], dtype=np.float64)
    else:
        mid = round((low + high) / 2.0, 1)
        if mid <= low:
            mid = round(min(1.0, low + 0.1), 1)
        if mid >= high:
            mid = round(max(0.0, high - 0.1), 1)
        ticks = np.asarray([low, mid, high], dtype=np.float64)

    ticks = np.unique(np.round(ticks, 1))
    if ticks.size < 2:
        if ticks[0] >= 1.0:
            ticks = np.asarray([0.9, 1.0], dtype=np.float64)
        else:
            ticks = np.asarray([ticks[0], round(min(1.0, ticks[0] + 0.1), 1)], dtype=np.float64)

    if ticks.size > 3:
        idx = np.linspace(0, ticks.size - 1, 3).round().astype(int)
        ticks = ticks[idx]

    low = min(low, float(ticks[0]))
    high = max(high, float(ticks[-1]))
    return low, high, ticks


def apply_panel_axis_style(ax: plt.Axes, curves: Dict[str, dict], phase_names: Sequence[str]) -> None:
    y_low, y_high = infer_panel_ylim(curves, phase_names)
    y_low, y_high, ticks = choose_panel_yticks(y_low, y_high)
    ax.set_xlim(float(BIN_CENTERS[0]), float(BIN_CENTERS[-1]))
    ax.set_ylim(y_low, y_high)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in ticks], fontsize=14.8, fontweight="normal")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="x", labelsize=14.5)
    ax.tick_params(axis="y", labelleft=True, labelright=False, direction="out", pad=0.8, length=2.2, labelsize=14.8)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")
    ax.grid(alpha=0.18, linewidth=0.55)


def build_curves_five_regimes(samples: Dict[str, np.ndarray]) -> Tuple[Dict[str, dict], dict]:
    phase = samples["phase"]
    local = samples["local"]
    if len(local) == 0:
        return {}, {"median_lag": np.nan, "n_samples": 0}
    q20, q40, q60, q80 = np.nanpercentile(phase, [20.0, 40.0, 60.0, 80.0])
    masks = {
        "Very low": phase <= q20,
        "Low": (phase > q20) & (phase <= q40),
        "Neutral": (phase > q40) & (phase <= q60),
        "High": (phase > q60) & (phase <= q80),
        "Very high": phase > q80,
    }
    return build_curves_from_masks(samples, masks)


def compute_grid_results(
    patch_rows: List[dict],
    coarse: Dict[str, np.ndarray],
    months: np.ndarray,
    local_vars: List[str],
    climate_vars: List[str],
    climate_series_dict: Dict[str, np.ndarray],
    lag_prefix: str,
    build_curves_fn=build_curves,
) -> Dict[Tuple[str, str], dict]:
    top_patch_lookup = {
        climate_var: select_top_patches(patch_rows, climate_var, prefix=lag_prefix)
        for climate_var in climate_vars
    }

    panel_results: Dict[Tuple[str, str], dict] = {}
    for local_var in local_vars:
        for climate_var in climate_vars:
            lag_key = f"{lag_prefix}{climate_var}_peak_lag"
            patch_subset = top_patch_lookup[climate_var]
            samples = panel_samples(
                patch_subset=patch_subset,
                coarse=coarse,
                months=months,
                local_var=local_var,
                climate_series=climate_series_dict[climate_var],
                lag_key=lag_key,
            )
            curves, summary = build_curves_fn(samples)
            panel_results[(local_var, climate_var)] = {
                "curves": curves,
                "summary": summary,
                "n_patches": len(patch_subset),
            }
    return panel_results


def plot_response_grid(
    panel_results: Dict[Tuple[str, str], dict],
    local_vars: List[str],
    climate_vars: List[str],
    output_path: Path,
    col_prefix: str,
    title: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14.2,
            "axes.titlesize": 16.8,
            "axes.labelsize": 16.5,
            "xtick.labelsize": 14.5,
            "ytick.labelsize": 14.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    n_rows = len(local_vars)
    n_cols = len(climate_vars)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25.6, 14.8), sharex=True, sharey=False)

    for i, local_var in enumerate(local_vars):
        for j, climate_var in enumerate(climate_vars):
            ax = axes[i, j]
            panel = panel_results[(local_var, climate_var)]
            curves = panel["curves"]
            summary = panel["summary"]

            for phase_name, color in PHASE_COLORS.items():
                curve = curves.get(phase_name, {})
                centers = curve.get("centers", np.asarray([]))
                probs = curve.get("probs", np.asarray([]))
                lower = curve.get("lower", np.asarray([]))
                upper = curve.get("upper", np.asarray([]))
                if len(centers) == 0:
                    continue
                ax.plot(centers, probs, color=color, linewidth=1.7)
                ax.fill_between(centers, lower, upper, color=color, alpha=0.14)

            if i == 0:
                label = DISPLAY_LABELS[climate_var]
                if col_prefix:
                    label = f"{col_prefix}{label}"
                ax.set_title(label, pad=7.0, fontsize=16.8, fontweight="bold")
            if j == 0:
                ax.set_ylabel(DISPLAY_LABELS[local_var], fontweight="bold", fontsize=16.8, labelpad=22)

            apply_panel_axis_style(ax, curves, tuple(PHASE_COLORS.keys()))

            if not curves or all(len(curves[p]["centers"]) == 0 for p in PHASE_COLORS):
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12.5,
                    color="#7a7a7a",
                )

    handles = [
        plt.Line2D([0], [0], color=color, linewidth=2.6, label=label)
        for label, color in PHASE_COLORS.items()
    ]
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.976),
        bbox_transform=fig.transFigure,
        fontsize=16.8,
        handlelength=2.4,
        handletextpad=0.8,
        columnspacing=2.0,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.subplots_adjust(left=0.082, right=0.995, top=0.918, bottom=0.055, wspace=0.30, hspace=0.24)
    fig.savefig(
        output_path,
        dpi=PNG_DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.05,
        bbox_extra_artists=(legend,),
    )
    fig.savefig(
        output_path.with_suffix(".svg"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.05,
        bbox_extra_artists=(legend,),
    )
    plt.close(fig)


def plot_localwise_response_grid(
    local_var: str,
    global_results: Dict[Tuple[str, str], dict],
    oci_results: Dict[Tuple[str, str], dict],
    global_vars: List[str],
    oci_vars: List[str],
    png_output_path: Path,
    svg_output_path: Path,
    phase_colors: Dict[str, str] = PHASE_COLORS,
    title_suffix: str = "",
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(4, 5, figsize=(17, 13), sharex=True, sharey=True)

    global_chunks = [global_vars[:5], global_vars[5:]]
    oci_chunks = [oci_vars[:5], oci_vars[5:]]

    def draw_panel(ax, panel: dict, title: str) -> None:
        curves = panel["curves"]
        summary = panel["summary"]

        for phase_name, color in phase_colors.items():
            curve = curves.get(phase_name, {})
            centers = curve.get("centers", np.asarray([]))
            probs = curve.get("probs", np.asarray([]))
            lower = curve.get("lower", np.asarray([]))
            upper = curve.get("upper", np.asarray([]))
            if len(centers) == 0:
                continue
            ax.plot(centers, probs, color=color, linewidth=1.6)
            ax.fill_between(centers, lower, upper, color=color, alpha=0.14)

        ax.set_title(title, pad=4)
        ax.set_xlim(BIN_EDGES[0], BIN_EDGES[-1])
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(alpha=0.16, linewidth=0.5)

        lag = summary.get("median_lag", np.nan)
        if np.isfinite(lag):
            ax.text(
                0.03,
                0.93,
                f"lag={lag:.0f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7.2,
                color="#42515c",
            )
            if not curves or all(len(curves.get(p, {}).get("centers", np.asarray([]))) == 0 for p in phase_colors):
                ax.text(
                    0.5,
                0.5,
                "N/A",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#7a7a7a",
            )

    for row_offset, chunk in enumerate(global_chunks):
        for col_idx, climate_var in enumerate(chunk):
            ax = axes[row_offset, col_idx]
            draw_panel(ax, global_results[(local_var, climate_var)], f"G-{DISPLAY_LABELS[climate_var]}")

    for row_offset, chunk in enumerate(oci_chunks, start=2):
        for col_idx, climate_var in enumerate(chunk):
            ax = axes[row_offset, col_idx]
            draw_panel(ax, oci_results[(local_var, climate_var)], DISPLAY_LABELS[climate_var])

    for row_idx in range(4):
        axes[row_idx, 0].set_ylabel("Future fire\nprobability")
    for col_idx in range(5):
        axes[3, col_idx].set_xlabel("Local anomaly")

    fig.text(
        0.5,
        0.955,
        f"{DISPLAY_LABELS[local_var]} response grids under Global and OCI regimes{title_suffix}",
        ha="center",
        fontsize=16,
    )
    fig.text(0.5, 0.93, "Top 2 rows: Global background variables", ha="center", fontsize=11, color="#4c628c")
    fig.text(0.5, 0.485, "Bottom 2 rows: OCI signals", ha="center", fontsize=11, color="#c98f7e")

    handles = [
        plt.Line2D([0], [0], color=color, linewidth=1.8, label=label)
        for label, color in phase_colors.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(phase_colors), frameon=False, bbox_to_anchor=(0.5, 0.985))
    footer = "Each mini-panel shows "
    if len(phase_colors) == 3:
        footer += "low / neutral / high regime response curves with 95% CI bands."
    else:
        footer += "very-low / low / neutral / high / very-high regime response curves with 95% CI bands."
    fig.text(0.5, 0.02, footer, ha="center", fontsize=10, color="#44515c")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.07, wspace=0.16, hspace=0.22)
    fig.savefig(png_output_path, dpi=240, facecolor="white")
    fig.savefig(svg_output_path, facecolor="white")
    plt.close(fig)


def write_summary(
    panel_results: Dict[Tuple[str, str], dict],
    local_vars: List[str],
    climate_vars: List[str],
    output_path: Path,
    group_name: str,
) -> None:
    rows = []
    for local_var in local_vars:
        for climate_var in climate_vars:
            panel = panel_results[(local_var, climate_var)]
            row = {
                "group": group_name,
                "local_var": local_var,
                "climate_var": climate_var,
                "n_patches": panel["n_patches"],
                "median_lag": panel["summary"].get("median_lag", np.nan),
                "n_samples": panel["summary"].get("n_samples", np.nan),
            }
            for phase_name in PHASE_COLORS:
                key = phase_name.lower().replace(" ", "_")
                curve = panel["curves"].get(phase_name, {})
                row[f"{key}_slope"] = curve.get("slope", np.nan)
                row[f"{key}_n"] = curve.get("n", np.nan)
            rows.append(row)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCALWISE_PNG_DIR.mkdir(parents=True, exist_ok=True)
    LOCALWISE_SVG_DIR.mkdir(parents=True, exist_ok=True)
    LOCALWISE_5LINE_PNG_DIR.mkdir(parents=True, exist_ok=True)
    LOCALWISE_5LINE_SVG_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    fire_vars = list(config["data"]["fire_vars"])
    oci_vars = list(config["data"]["oci_vars"])
    log_transform_vars = set(config["data"].get("log_transform_vars", []))

    patch_rows = load_patch_rows()
    coarse, global_series, oci_series, months = load_data(fire_vars, oci_vars, log_transform_vars)

    local_global_results = compute_grid_results(
        patch_rows=patch_rows,
        coarse=coarse,
        months=months,
        local_vars=fire_vars,
        climate_vars=fire_vars,
        climate_series_dict=global_series,
        lag_prefix="global_",
    )
    local_oci_results = compute_grid_results(
        patch_rows=patch_rows,
        coarse=coarse,
        months=months,
        local_vars=fire_vars,
        climate_vars=oci_vars,
        climate_series_dict=oci_series,
        lag_prefix="",
    )
    local_global_results_5 = compute_grid_results(
        patch_rows=patch_rows,
        coarse=coarse,
        months=months,
        local_vars=fire_vars,
        climate_vars=fire_vars,
        climate_series_dict=global_series,
        lag_prefix="global_",
        build_curves_fn=build_curves_five_regimes,
    )
    local_oci_results_5 = compute_grid_results(
        patch_rows=patch_rows,
        coarse=coarse,
        months=months,
        local_vars=fire_vars,
        climate_vars=oci_vars,
        climate_series_dict=oci_series,
        lag_prefix="",
        build_curves_fn=build_curves_five_regimes,
    )

    plot_response_grid(
        panel_results=local_global_results,
        local_vars=fire_vars,
        climate_vars=fire_vars,
        output_path=OUTPUT_DIR / "response_grid_local_global_10x10.png",
        col_prefix="G-",
        title="Local fire drivers x Global background variables: 10x10 response grid",
    )
    plot_response_grid(
        panel_results=local_oci_results,
        local_vars=fire_vars,
        climate_vars=oci_vars,
        output_path=OUTPUT_DIR / "response_grid_local_oci_10x10.png",
        col_prefix="",
        title="Local fire drivers x OCI signals: 10x10 response grid",
    )
    write_summary(
        local_global_results,
        fire_vars,
        fire_vars,
        OUTPUT_DIR / "response_grid_local_global_10x10_summary.csv",
        "Global",
    )
    write_summary(
        local_oci_results,
        fire_vars,
        oci_vars,
        OUTPUT_DIR / "response_grid_local_oci_10x10_summary.csv",
        "OCI",
    )

    if os.environ.get("GENERATE_LOCALWISE", "1") != "0":
        for local_var in fire_vars:
            stem = f"{DISPLAY_LABELS[local_var].lower()}_response_grid_4x5"
            plot_localwise_response_grid(
                local_var=local_var,
                global_results=local_global_results,
                oci_results=local_oci_results,
                global_vars=fire_vars,
                oci_vars=oci_vars,
                png_output_path=LOCALWISE_PNG_DIR / f"{stem}.png",
                svg_output_path=LOCALWISE_SVG_DIR / f"{stem}.svg",
            )
            plot_localwise_response_grid(
                local_var=local_var,
                global_results=local_global_results_5,
                oci_results=local_oci_results_5,
                global_vars=fire_vars,
                oci_vars=oci_vars,
                png_output_path=LOCALWISE_5LINE_PNG_DIR / f"{stem}.png",
                svg_output_path=LOCALWISE_5LINE_SVG_DIR / f"{stem}.svg",
                phase_colors=PHASE_COLORS_5,
                title_suffix=" (5-line)",
            )
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
