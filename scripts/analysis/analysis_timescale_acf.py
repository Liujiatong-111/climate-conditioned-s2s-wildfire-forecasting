"""
Group-level ACF analysis for Local / Climate / OCI inputs.

Figure goal:
- Local group: all fire_vars at local patch scale
- Climate group: the same fire_vars reduced to global background series
- OCI group: all oci_vars scalar indices

Main figure:
- median ACF curve for each group
- interquartile band for each group

Outputs are written to analyze/timescale_acf/.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config.yaml"
DEFAULT_DATA_PATH = Path("/home/dataset-local/QJC/ljt/TeleVIT/SeasFireCube_v3.zarr")
OUTPUT_DIR = Path(__file__).resolve().parent / "timescale_acf"

MAX_LAG = 60
MIN_VALID_POINTS = 120
DECORR_THRESHOLD = 1.0 / np.e
N_BOOT = 2000
BOOT_SEED = 42


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def resolve_zarr_path(config: dict) -> Path:
    candidate = Path(config["data"].get("zarr_path", ""))
    if candidate.exists():
        return candidate
    return DEFAULT_DATA_PATH


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
        month_mean = np.nanmean(month_vals)
        y[idx] = month_vals - month_mean
    std = np.nanstd(y)
    if not np.isfinite(std) or std < 1e-8:
        return np.full_like(y, np.nan, dtype=np.float64)
    return y / std


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    corrs = []
    for lag in range(max_lag + 1):
        if lag == 0:
            xs, ys = x, x
        else:
            xs, ys = x[:-lag], x[lag:]
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() < MIN_VALID_POINTS:
            corrs.append(np.nan)
            continue
        xs = xs[mask]
        ys = ys[mask]
        if np.std(xs) < 1e-8 or np.std(ys) < 1e-8:
            corrs.append(np.nan)
        else:
            corrs.append(float(np.corrcoef(xs, ys)[0, 1]))
    return np.asarray(corrs, dtype=np.float64)


def decorrelation_time(acf: np.ndarray, threshold: float = DECORR_THRESHOLD) -> float:
    valid = np.where(np.isfinite(acf) & (acf < threshold))[0]
    if len(valid) == 0:
        return np.nan
    return float(valid[0])


def global_background_series(da: xr.DataArray) -> np.ndarray:
    lat_weights = np.cos(np.deg2rad(da["latitude"]))
    ts = da.weighted(lat_weights).mean(dim=("latitude", "longitude")).compute().values
    return np.asarray(ts, dtype=np.float64)


def local_patch_curves(
    da: xr.DataArray,
    months: np.ndarray,
    patch_size: int,
) -> Tuple[np.ndarray, int]:
    patch_means = (
        da.coarsen(latitude=patch_size, longitude=patch_size, boundary="trim")
        .mean(skipna=True)
        .compute()
        .values
    )
    curves: List[np.ndarray] = []
    for i in range(patch_means.shape[1]):
        for j in range(patch_means.shape[2]):
            series = patch_means[:, i, j]
            if np.isfinite(series).sum() < MIN_VALID_POINTS:
                continue
            anomaly = deseason_by_month(series, months)
            acf = autocorr(anomaly, MAX_LAG)
            if np.isfinite(acf).sum() == 0:
                continue
            curves.append(acf)
    if not curves:
        return np.empty((0, MAX_LAG + 1), dtype=np.float64), 0
    return np.stack(curves, axis=0), len(curves)


def summarize_curves(curves: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "median": np.nanmedian(curves, axis=0),
        "q25": np.nanpercentile(curves, 25, axis=0),
        "q75": np.nanpercentile(curves, 75, axis=0),
    }


def bootstrap_median_ci(
    curves: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> Dict[str, np.ndarray]:
    n_series, n_lags = curves.shape
    median = np.nanmedian(curves, axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, n_lags), dtype=np.float64)
    for i in range(n_boot):
        indices = rng.integers(0, n_series, size=n_series)
        boot[i] = np.nanmedian(curves[indices], axis=0)
    return {
        "median": median,
        "lower": np.nanpercentile(boot, 2.5, axis=0),
        "upper": np.nanpercentile(boot, 97.5, axis=0),
    }


def apply_log_transform(da: xr.DataArray, var_name: str, log_transform_vars: List[str]) -> xr.DataArray:
    if var_name in log_transform_vars:
        return np.log(da + 1.0)
    return da


def plot_group_acf(
    group_stats: Dict[str, Dict[str, np.ndarray]],
    temporal_steps: int,
    oci_window: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(MAX_LAG + 1)

    style = {
        "Local": {"color": "#d95f0e"},
        "Climate": {"color": "#1b9e77"},
        "OCI": {"color": "#386cb0"},
    }

    for group_name in ["Local", "Climate", "OCI"]:
        stats = group_stats[group_name]
        color = style[group_name]["color"]
        ax.plot(x, stats["median"], color=color, linewidth=2.5, label=group_name)
        ax.fill_between(x, stats["q25"], stats["q75"], color=color, alpha=0.18)

    ax.axvline(oci_window, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(temporal_steps, color="#4d4d4d", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(oci_window + 0.15, 0.1, f"OCI window={oci_window}", rotation=90, va="bottom", ha="left", fontsize=9)
    ax.text(temporal_steps + 0.15, 0.1, f"temporal steps={temporal_steps}", rotation=90, va="bottom", ha="left", fontsize=9)

    ax.set_xlim(0, MAX_LAG)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Lag (8-day steps)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Time-scale mismatch across Local, Climate, and OCI groups")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    fig.savefig(output_path.with_suffix(".svg"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_group_acf_bootstrap_ci(
    group_stats: Dict[str, Dict[str, np.ndarray]],
    temporal_steps: int,
    oci_window: int,
    output_path: Path,
) -> None:
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(MAX_LAG + 1)

    style = {
        "Local": {"color": "#d95f0e"},
        "Climate": {"color": "#1b9e77"},
        "OCI": {"color": "#386cb0"},
    }

    for group_name in ["Local", "Climate", "OCI"]:
        stats = group_stats[group_name]
        color = style[group_name]["color"]
        ax.fill_between(x, stats["lower"], stats["upper"], color=color, alpha=0.08, linewidth=0)
        ax.plot(
            x,
            stats["lower"],
            color=color,
            linewidth=0.9,
            alpha=0.45,
            linestyle="--",
        )
        ax.plot(
            x,
            stats["upper"],
            color=color,
            linewidth=0.9,
            alpha=0.45,
            linestyle="--",
        )
        ax.plot(x, stats["median"], color=color, linewidth=2.6, label=group_name)

    ax.axvline(oci_window, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(temporal_steps, color="#4d4d4d", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(oci_window + 0.15, 0.1, f"OCI window={oci_window}", rotation=90, va="bottom", ha="left", fontsize=9)
    ax.text(temporal_steps + 0.15, 0.1, f"temporal steps={temporal_steps}", rotation=90, va="bottom", ha="left", fontsize=9)

    ax.set_xlim(0, MAX_LAG)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Lag (8-day steps)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Time-scale mismatch across Local, Climate, and OCI groups")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    fig.savefig(output_path.with_suffix(".svg"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def write_group_summary_csv(group_stats: Dict[str, Dict[str, np.ndarray]], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "lag", "median", "q25", "q75"])
        writer.writeheader()
        for group_name, stats in group_stats.items():
            for lag in range(MAX_LAG + 1):
                writer.writerow(
                    {
                        "group": group_name,
                        "lag": lag,
                        "median": float(stats["median"][lag]),
                        "q25": float(stats["q25"][lag]),
                        "q75": float(stats["q75"][lag]),
                    }
                )


def write_group_bootstrap_csv(group_stats: Dict[str, Dict[str, np.ndarray]], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "lag", "median", "lower95", "upper95"])
        writer.writeheader()
        for group_name, stats in group_stats.items():
            for lag in range(MAX_LAG + 1):
                writer.writerow(
                    {
                        "group": group_name,
                        "lag": lag,
                        "median": float(stats["median"][lag]),
                        "lower95": float(stats["lower"][lag]),
                        "upper95": float(stats["upper"][lag]),
                    }
                )


def write_variable_summary_csv(variable_rows: List[dict], output_path: Path) -> None:
    if not variable_rows:
        return
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(variable_rows[0].keys()))
        writer.writeheader()
        writer.writerows(variable_rows)


def write_notes(
    output_path: Path,
    zarr_path: Path,
    fire_vars: List[str],
    oci_vars: List[str],
    patch_size: int,
    temporal_steps: int,
    oci_window: int,
    variable_rows: List[dict],
) -> None:
    lines = [
        f"Data path: {zarr_path}",
        f"Patch size: {patch_size}",
        f"Max lag: {MAX_LAG} steps ({MAX_LAG * 8} days)",
        f"Current temporal_steps: {temporal_steps} ({temporal_steps * 8} days)",
        f"Current oci_window: {oci_window} ({oci_window * 8} days)",
        f"Fire vars (Local / Climate groups): {', '.join(fire_vars)}",
        f"OCI vars: {', '.join(oci_vars)}",
        "",
        "Group definitions:",
        "- Local: each fire_var at local patch scale, aggregated over valid 80x80 patch mean series.",
        "- Climate: the same fire_vars reduced to global background time series using area-weighted spatial means.",
        "- OCI: scalar oci_vars time series.",
        "",
        "Bootstrap CI figure:",
        f"- Output: {OUTPUT_DIR / 'timescale_acf_groups_bootstrap_ci.png'}",
        f"- Shading is bootstrap 95% CI of the group median ACF, resampled across variable-level curves within each group.",
        f"- n_boot={N_BOOT}, seed={BOOT_SEED}",
        "",
        "Variable-level decorrelation times (first lag where ACF < 1/e):",
    ]
    for row in variable_rows:
        lines.append(
            f"- {row['group']}/{row['variable']}: tau={row['decorrelation_time_steps']} steps, "
            f"series_count={row['series_count']}"
        )
    output_path.write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(CONFIG_PATH)
    zarr_path = resolve_zarr_path(config)
    fire_vars = list(config["data"]["fire_vars"])
    oci_vars = list(config["data"]["oci_vars"])
    log_transform_vars = list(config["data"].get("log_transform_vars", []))
    patch_size = int(config["data"].get("patch_size", 80))
    temporal_steps = int(config["data"].get("temporal_steps", 16))
    oci_window = int(config["data"].get("oci_window", 10))

    ds = xr.open_zarr(zarr_path, consolidated=True)[fire_vars + oci_vars]
    months = ds["time"].dt.month.values.astype(int)

    local_variable_curves: List[np.ndarray] = []
    climate_variable_curves: List[np.ndarray] = []
    oci_variable_curves: List[np.ndarray] = []
    variable_rows: List[dict] = []

    for var_name in fire_vars:
        da = apply_log_transform(ds[var_name], var_name, log_transform_vars)

        patch_curves, n_series = local_patch_curves(da, months, patch_size)
        if n_series > 0:
            local_curve = np.nanmedian(patch_curves, axis=0)
            local_variable_curves.append(local_curve)
            variable_rows.append(
                {
                    "group": "Local",
                    "variable": var_name,
                    "series_count": n_series,
                    "decorrelation_time_steps": decorrelation_time(local_curve),
                }
            )

        climate_series = global_background_series(da)
        climate_anom = deseason_by_month(climate_series, months)
        climate_curve = autocorr(climate_anom, MAX_LAG)
        if np.isfinite(climate_curve).sum() > 0:
            climate_variable_curves.append(climate_curve)
            variable_rows.append(
                {
                    "group": "Climate",
                    "variable": var_name,
                    "series_count": 1,
                    "decorrelation_time_steps": decorrelation_time(climate_curve),
                }
            )

    for var_name in oci_vars:
        oci_series = ds[var_name].values.astype(np.float64)
        oci_anom = deseason_by_month(oci_series, months)
        oci_curve = autocorr(oci_anom, MAX_LAG)
        if np.isfinite(oci_curve).sum() == 0:
            continue
        oci_variable_curves.append(oci_curve)
        variable_rows.append(
            {
                "group": "OCI",
                "variable": var_name,
                "series_count": 1,
                "decorrelation_time_steps": decorrelation_time(oci_curve),
            }
        )

    group_curve_arrays = {
        "Local": np.stack(local_variable_curves, axis=0),
        "Climate": np.stack(climate_variable_curves, axis=0),
        "OCI": np.stack(oci_variable_curves, axis=0),
    }
    group_stats = {group_name: summarize_curves(curves) for group_name, curves in group_curve_arrays.items()}
    group_bootstrap_stats = {
        group_name: bootstrap_median_ci(curves) for group_name, curves in group_curve_arrays.items()
    }

    plot_group_acf(
        group_stats=group_stats,
        temporal_steps=temporal_steps,
        oci_window=oci_window,
        output_path=OUTPUT_DIR / "timescale_acf_groups.png",
    )
    plot_group_acf_bootstrap_ci(
        group_stats=group_bootstrap_stats,
        temporal_steps=temporal_steps,
        oci_window=oci_window,
        output_path=OUTPUT_DIR / "timescale_acf_groups_bootstrap_ci.png",
    )
    write_group_summary_csv(group_stats, OUTPUT_DIR / "timescale_acf_group_summary.csv")
    write_group_bootstrap_csv(group_bootstrap_stats, OUTPUT_DIR / "timescale_acf_group_bootstrap_ci.csv")
    write_variable_summary_csv(variable_rows, OUTPUT_DIR / "timescale_acf_variable_summary.csv")
    write_notes(
        output_path=OUTPUT_DIR / "timescale_acf_notes.txt",
        zarr_path=zarr_path,
        fire_vars=fire_vars,
        oci_vars=oci_vars,
        patch_size=patch_size,
        temporal_steps=temporal_steps,
        oci_window=oci_window,
        variable_rows=variable_rows,
    )

    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
