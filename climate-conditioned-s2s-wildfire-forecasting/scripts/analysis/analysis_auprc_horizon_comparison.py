from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "auprc_horizon_comparison"

HORIZONS = [1, 2, 4, 8, 16]
X = np.arange(len(HORIZONS), dtype=np.float64)

LOCAL_ONLY = {
    "GRU": [0.557, 0.546, 0.538, 0.517, 0.524],
    "Conv-GRU": [0.589, 0.573, 0.551, 0.546, 0.541],
    "Conv-LSTM": [0.632, 0.606, 0.614, 0.601, 0.595],
    "U-Net++": [0.609, 0.606, 0.598, 0.591, 0.582],
    "U-TAE (21, ICCV)": [0.623, 0.620, 0.611, 0.602, 0.606],
    "FireCastNet (25)": [0.640, 0.634, 0.629, 0.627, 0.628],
    "Ours (Local Only)": [0.649, 0.635, 0.630, 0.627, 0.632],
}

FULL_INPUT = {
    "TeleViT (23, ICCV)": [0.619, 0.617, 0.609, 0.610, 0.602],
    "PMFM-kdTransformer (26)": [0.632, 0.628, 0.627, 0.621, 0.611],
    "Ours": [0.661, 0.651, 0.654, 0.656, 0.649],
}

MODEL_STYLES = {
    "GRU": {"color": "#E08D2D", "marker": "^", "linestyle": "--", "linewidth": 1.8, "markersize": 7},
    "Conv-GRU": {"color": "#5A9D52", "marker": "^", "linestyle": "--", "linewidth": 1.8, "markersize": 7},
    "Conv-LSTM": {"color": "#73AEB0", "marker": "^", "linestyle": "--", "linewidth": 1.8, "markersize": 7},
    "U-Net++": {"color": "#597AA3", "marker": "^", "linestyle": "--", "linewidth": 1.8, "markersize": 7},
    "U-TAE (21, ICCV)": {"color": "#E3BC42", "marker": "^", "linestyle": "--", "linewidth": 1.8, "markersize": 7},
    "FireCastNet (25)": {"color": "#E6A0AA", "marker": "^", "linestyle": "--", "linewidth": 1.8, "markersize": 7},
    "Ours (Local Only)": {"color": "#C12B2C", "marker": "^", "linestyle": "-", "linewidth": 2.4, "markersize": 8},
    "TeleViT (23, ICCV)": {"color": "#B08C9D", "marker": "o", "linestyle": "-", "linewidth": 1.9, "markersize": 7},
    "PMFM-kdTransformer (26)": {"color": "#4E79A7", "marker": "s", "linestyle": "-", "linewidth": 1.9, "markersize": 6.5},
    "Ours": {"color": "#8F63B6", "marker": "o", "linestyle": "-", "linewidth": 2.5, "markersize": 7.5},
}


def plot_auprc_horizon_comparison() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "axes.labelweight": "normal",
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 11.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(9.4, 6.8))

    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color="#CFCFCF", alpha=0.75)
    ax.grid(axis="x", linestyle="--", linewidth=0.8, color="#D8D8D8", alpha=0.75)

    plot_order = [
        *LOCAL_ONLY.keys(),
        *FULL_INPUT.keys(),
    ]

    for model_name in plot_order:
        if model_name in LOCAL_ONLY:
            values = LOCAL_ONLY[model_name]
        else:
            values = FULL_INPUT[model_name]
        style = MODEL_STYLES[model_name]
        ax.plot(
            X,
            values,
            label=model_name,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markerfacecolor=style["color"],
            markeredgewidth=0.4,
            markeredgecolor=style["color"],
            zorder=4 if "Ours" in model_name else 3,
        )

    ax.set_xlim(-0.12, len(HORIZONS) - 0.88)
    ax.set_ylim(0.45, 0.67)
    ax.set_xticks(X)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65])

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("AUPRC Score")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#222222")

    legend = ax.legend(
        ncol=2,
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#C9C9C9",
        columnspacing=1.5,
        handlelength=2.0,
        handletextpad=0.6,
        borderpad=0.5,
    )
    legend.get_frame().set_linewidth(0.9)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.98)

    save_kwargs = {
        "facecolor": "white",
        "bbox_inches": "tight",
        "pad_inches": 0.05,
    }
    fig.savefig(OUTPUT_DIR / "auprc_horizon_comparison.svg", **save_kwargs)
    fig.savefig(OUTPUT_DIR / "auprc_horizon_comparison.pdf", **save_kwargs)
    fig.savefig(OUTPUT_DIR / "auprc_horizon_comparison.png", dpi=320, **save_kwargs)
    plt.close(fig)


if __name__ == "__main__":
    plot_auprc_horizon_comparison()
