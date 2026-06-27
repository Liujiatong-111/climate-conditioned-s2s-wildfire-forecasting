"""
Formal figure-3 style modulation analysis.

This script upgrades the quick-look modulation figure into a publication-style
3x3 response grid:
- rows: local variables (VPD, RH, T2M)
- cols: global climate regimes (Nino34, PDO, SOI)

For each panel, we:
1. Select the patches most sensitive to the target climate signal.
2. Align the climate index using each patch's peak lag.
3. Pool samples across those patches.
4. Estimate fire-occurrence probability across local-anomaly bins under
   low / neutral / high climate regimes.

The resulting curves are intended to show whether climate background changes
the local-to-fire response function rather than merely shifting risk upward.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.environ.get("SEASFIRE_ZARR", "data/SeasFireCube_v3.zarr"))
PATCH_SUMMARY_PATH = PROJECT_DIR / "analyze" / "lag_modulation_quicklook" / "patch_lag_summary.csv"
OUTPUT_DIR = PROJECT_DIR / "analyze" / "formal_modulation_figure3"

PATCH_SIZE = 80
TOP_PATCHES_PER_SIGNAL = 12
LOCAL_VARS = ["vpd", "rel_hum", "t2m_mean"]
SIGNAL_VARS = ["oci_nina34_anom", "oci_pdo", "oci_soi"]
DISPLAY_LABELS = {
    "vpd": "VPD anomaly",
    "rel_hum": "RH anomaly",
    "t2m_mean": "T2M anomaly",
    "oci_nina34_anom": "Nino34 regime",
    "oci_pdo": "PDO regime",
    "oci_soi": "SOI regime",
}
PHASE_COLORS = {
    "Low regime": "#2b8cbe",
    "Neutral": "#7f7f7f",
    "High regime": "#d95f0e",
}
BIN_EDGES = np.linspace(-2.4, 2.4, 7)


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


def align_leading_signal(x: np.ndarray, lag: int) -> np.ndarray:
    """Align a signal that leads fire by `lag` steps to the fire index."""
    x = np.asarray(x, dtype=np.float64)
    if lag <= 0:
        return x.copy()
    return np.concatenate([np.full(lag, np.nan), x[:-lag]])


def load_patch_summary() -> List[dict]:
    with PATCH_SUMMARY_PATH.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["patch_i"] = int(row["patch_i"])
        row["patch_j"] = int(row["patch_j"])
        row["lat_center"] = float(row["lat_center"])
        row["lon_center"] = float(row["lon_center"])
        row["fire_mean"] = float(row["fire_mean"])
    return rows


def coarsen_mean(da: xr.DataArray) -> np.ndarray:
    return (
        da.coarsen(latitude=PATCH_SIZE, longitude=PATCH_SIZE, boundary="trim")
        .mean()
        .compute()
        .values
    )


def load_data() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    ds = xr.open_zarr(
        DATA_PATH,
        consolidated=True,
    )[
        [
            "gwis_ba",
            *LOCAL_VARS,
            *SIGNAL_VARS,
        ]
    ]

    coarse = {}
    fire_binary = (ds["gwis_ba"] > 0).astype("float32")
    coarse["fire_fraction"] = coarsen_mean(fire_binary)
    coarse["fire_occurrence"] = (coarse["fire_fraction"] > 0).astype(np.float64)

    for var in LOCAL_VARS:
        coarse[var] = coarsen_mean(ds[var])

    months = ds["time"].dt.month.values.astype(int)
    signal_series = {var: deseason_by_month(ds[var].values.astype(np.float64), months) for var in SIGNAL_VARS}
    return coarse, signal_series, months


def select_top_patches(rows: List[dict], signal_var: str, top_k: int) -> List[dict]:
    corr_key = f"{signal_var}_peak_corr"
    lag_key = f"{signal_var}_peak_lag"
    valid = []
    for row in rows:
        corr_raw = row.get(corr_key, "")
        lag_raw = row.get(lag_key, "")
        if corr_raw in ("", "nan") or lag_raw in ("", "nan"):
            continue
        corr = float(corr_raw)
        lag = int(float(lag_raw))
        out = dict(row)
        out[corr_key] = corr
        out[lag_key] = lag
        valid.append(out)
    valid.sort(key=lambda r: abs(r[corr_key]), reverse=True)
    return valid[:top_k]


def binned_event_rate(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = []
    probs = []
    lower = []
    upper = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = np.isfinite(x) & np.isfinite(y)
        if right == edges[-1]:
            mask &= (x >= left) & (x <= right)
        else:
            mask &= (x >= left) & (x < right)
        n = int(mask.sum())
        if n < 10:
            continue
        p = float(np.mean(y[mask]))
        se = np.sqrt(max(p * (1.0 - p), 0.0) / n)
        centers.append(0.5 * (left + right))
        probs.append(p)
        lower.append(max(0.0, p - 1.96 * se))
        upper.append(min(1.0, p + 1.96 * se))
    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(probs, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )


def panel_samples(
    coarse: Dict[str, np.ndarray],
    signal_series: Dict[str, np.ndarray],
    months: np.ndarray,
    patch_rows: List[dict],
    local_var: str,
    signal_var: str,
) -> Dict[str, np.ndarray]:
    local_all = []
    fire_all = []
    phase_all = []
    lag_list = []

    for row in patch_rows:
        i = row["patch_i"]
        j = row["patch_j"]
        lag = int(row[f"{signal_var}_peak_lag"])

        local_d = deseason_by_month(coarse[local_var][:, i, j], months)
        fire_y = coarse["fire_occurrence"][:, i, j].astype(np.float64)
        phase = align_leading_signal(signal_series[signal_var], lag)

        mask = np.isfinite(local_d) & np.isfinite(fire_y) & np.isfinite(phase)
        if mask.sum() < 40:
            continue

        local_all.append(local_d[mask])
        fire_all.append(fire_y[mask])
        phase_all.append(phase[mask])
        lag_list.append(lag)

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
        "lags": np.asarray(lag_list, dtype=np.float64),
    }


def phase_curves(samples: Dict[str, np.ndarray]) -> Tuple[Dict[str, dict], dict]:
    local = samples["local"]
    fire = samples["fire"]
    phase = samples["phase"]
    if len(local) == 0:
        return {}, {}

    q1 = np.nanpercentile(phase, 33.0)
    q2 = np.nanpercentile(phase, 67.0)
    phase_masks = {
        "Low regime": phase <= q1,
        "Neutral": (phase > q1) & (phase < q2),
        "High regime": phase >= q2,
    }

    curves = {}
    for phase_name, mask in phase_masks.items():
        x = local[mask]
        y = fire[mask]
        centers, probs, lower, upper = binned_event_rate(x, y, BIN_EDGES)
        curves[phase_name] = {
            "centers": centers,
            "probs": probs,
            "lower": lower,
            "upper": upper,
            "slope": fit_slope(x, y),
            "n": int(mask.sum()),
        }

    summary = {
        "n_samples": int(len(local)),
        "median_lag": float(np.nanmedian(samples["lags"])) if len(samples["lags"]) else np.nan,
        "q1": float(q1),
        "q2": float(q2),
    }
    return curves, summary


def plot_response_grid(panel_results: Dict[Tuple[str, str], dict], output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
        }
    )

    fig, axes = plt.subplots(3, 3, figsize=(13.8, 10.8), sharex=True, sharey=True)
    # Fire-occurrence probability is bounded in [0, 1]. Use the full range so
    # high-probability tails are not visually clipped.
    ymax = 1.0

    for row_idx, local_var in enumerate(LOCAL_VARS):
        for col_idx, signal_var in enumerate(SIGNAL_VARS):
            ax = axes[row_idx, col_idx]
            panel = panel_results[(local_var, signal_var)]
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
                ax.plot(centers, probs, marker="o", linewidth=2.1, markersize=4.3, color=color, label=phase_name)
                ax.fill_between(centers, lower, upper, color=color, alpha=0.18)

            if row_idx == 0:
                ax.set_title(DISPLAY_LABELS[signal_var])
            if col_idx == 0:
                ax.set_ylabel("Future fire occurrence probability")
                ax.text(
                    -0.34,
                    0.5,
                    DISPLAY_LABELS[local_var],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                    color="#31414d",
                )
            if row_idx == len(LOCAL_VARS) - 1:
                ax.set_xlabel("Local anomaly (z-scored, deseasoned)")

            ax.set_xlim(BIN_EDGES[0], BIN_EDGES[-1])
            ax.set_ylim(0.0, ymax)
            ax.set_yticks(np.linspace(0.0, 1.0, 6))
            ax.grid(alpha=0.22)
            ax.text(
                0.03,
                0.97,
                f"patches={panel['n_patches']}\nmedian lag={summary.get('median_lag', np.nan):.0f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8.8,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )

            slope_lines = []
            for phase_name in ["Low regime", "Neutral", "High regime"]:
                slope = curves.get(phase_name, {}).get("slope", np.nan)
                if np.isfinite(slope):
                    slope_lines.append(f"{phase_name.split()[0]} slope={slope:+.2f}")
            if slope_lines:
                ax.text(
                    0.03,
                    0.05,
                    "\n".join(slope_lines),
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=8.2,
                    color="#33424d",
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
                )

    handles = [
        plt.Line2D([0], [0], color=color, marker="o", linewidth=2.1, label=label)
        for label, color in PHASE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "Climate regime changes the local-to-fire response function",
        fontsize=15.0,
        y=0.985,
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.12, wspace=0.06, hspace=0.08)
    fig.text(
        0.5,
        0.045,
        "Each panel pools the top climate-sensitive patches for the given signal. Curves show fire-occurrence probability under low / neutral / high lag-aligned climate regimes.",
        ha="center",
        fontsize=10,
        color="#44515c",
    )
    fig.savefig(output_path, dpi=260, facecolor="white")
    plt.close(fig)


def write_summary(panel_results: Dict[Tuple[str, str], dict], output_path: Path) -> None:
    rows = []
    for (local_var, signal_var), panel in panel_results.items():
        summary = panel["summary"]
        row = {
            "local_var": local_var,
            "signal_var": signal_var,
            "n_patches": panel["n_patches"],
            "n_samples": summary.get("n_samples", np.nan),
            "median_lag": summary.get("median_lag", np.nan),
            "phase_q33": summary.get("q1", np.nan),
            "phase_q67": summary.get("q2", np.nan),
        }
        for phase_name in ["Low regime", "Neutral", "High regime"]:
            curve = panel["curves"].get(phase_name, {})
            row[f"{phase_name.lower().replace(' ', '_')}_n"] = curve.get("n", np.nan)
            row[f"{phase_name.lower().replace(' ', '_')}_slope"] = curve.get("slope", np.nan)
        rows.append(row)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patch_rows = load_patch_summary()
    coarse, signal_series, months = load_data()

    top_patch_lookup = {
        signal_var: select_top_patches(patch_rows, signal_var, TOP_PATCHES_PER_SIGNAL)
        for signal_var in SIGNAL_VARS
    }

    panel_results: Dict[Tuple[str, str], dict] = {}
    for local_var in LOCAL_VARS:
        for signal_var in SIGNAL_VARS:
            patch_subset = top_patch_lookup[signal_var]
            samples = panel_samples(coarse, signal_series, months, patch_subset, local_var, signal_var)
            curves, summary = phase_curves(samples)
            panel_results[(local_var, signal_var)] = {
                "curves": curves,
                "summary": summary,
                "n_patches": len(patch_subset),
            }

    plot_response_grid(panel_results, OUTPUT_DIR / "formal_modulation_response_grid.png")
    write_summary(panel_results, OUTPUT_DIR / "formal_modulation_response_grid_summary.csv")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
