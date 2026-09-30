#!/usr/bin/env python3
"""Rank-ordered exposure curves for no-airflow, slow, and fast north-to-south airflow."""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PATH2OUTPUT = os.path.join("..", "output")
EXPERIMENT_NS = "choir_uq_inlet-north_outlet-south_512"
EXPERIMENT_NA = "choir_no_airflow"

INFECTION_THRESHOLD = 179
SLOW_BAND = (0.1, 0.3)
FAST_BAND = (1.2, 1.5)

C_NO_AIRFLOW = "#555555"
C_SLOW = "#E8963E"
C_FAST = "#2E86AB"
ALPHA_BAND = 0.18

HLINE_STYLE = dict(linestyle="--", color="#CC3333", linewidth=1.5, zorder=4)


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "#F9F9F9",
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def sorted_curves(df, param_ids, run_ids):
    """Return array of shape (n_curves, n_agents) with exposures sorted descending."""
    curves = []
    col = "degreeOfExposure-PID1"
    for pid in param_ids:
        for rid in run_ids:
            try:
                vals = df.xs((pid, rid), level=("id", "run_id"))[col].values
                curves.append(np.sort(vals)[::-1])
            except KeyError:
                pass
    return np.array(curves)


def rank_stats(curves):
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)
    return mean, std


def threshold_crossing_rank(mean, threshold):
    crossings = np.where(mean < threshold)[0]
    return crossings[0] if len(crossings) > 0 else len(mean)


def main():
    os.makedirs("figures", exist_ok=True)
    apply_style()

    exp_na = load_pkl(
        os.path.join(PATH2OUTPUT, EXPERIMENT_NA, "summary", "uq_experiment_post_run.pkl")
    )
    exp_ns = load_pkl(
        os.path.join(PATH2OUTPUT, EXPERIMENT_NS, "summary", "uq_experiment_post_run.pkl")
    )

    df_na = exp_na.sim_output["maxDegreeOfExposure"]
    df_ns = exp_ns.sim_output["maxDegreeOfExposure"]

    run_ids_na = df_na.index.get_level_values("run_id").unique().tolist()
    run_ids_ns = df_ns.index.get_level_values("run_id").unique().tolist()

    na_curves = sorted_curves(df_na, [0], run_ids_na)

    slow_ids = [
        i for i, v in enumerate(exp_ns.parameter_variations)
        if SLOW_BAND[0] <= v["inletVelocity"] <= SLOW_BAND[1]
    ]
    fast_ids = [
        i for i, v in enumerate(exp_ns.parameter_variations)
        if FAST_BAND[0] <= v["inletVelocity"] <= FAST_BAND[1]
    ]

    slow_curves = sorted_curves(df_ns, slow_ids, run_ids_ns)
    fast_curves = sorted_curves(df_ns, fast_ids, run_ids_ns)

    na_mean, na_std = rank_stats(na_curves)
    slow_mean, slow_std = rank_stats(slow_curves)
    fast_mean, fast_std = rank_stats(fast_curves)

    n_agents = len(na_mean)
    ranks = np.arange(1, n_agents + 1)

    na_cross = threshold_crossing_rank(na_mean, INFECTION_THRESHOLD)
    slow_cross = threshold_crossing_rank(slow_mean, INFECTION_THRESHOLD)
    fast_cross = threshold_crossing_rank(fast_mean, INFECTION_THRESHOLD)

    print(f"No airflow:      {na_cross} agents above threshold  (n={len(na_curves)} curves)")
    print(f"Slow ({SLOW_BAND[0]}–{SLOW_BAND[1]} m/s): {slow_cross} agents above threshold  (n={len(slow_curves)} curves)")
    print(f"Fast ({FAST_BAND[0]}–{FAST_BAND[1]} m/s): {fast_cross} agents above threshold  (n={len(fast_curves)} curves)")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for mean, std, color, label in (
        (na_mean, na_std, C_NO_AIRFLOW, "No airflow"),
        (slow_mean, slow_std, C_SLOW, f"Slow airflow ({SLOW_BAND[0]}–{SLOW_BAND[1]} m/s)"),
        (fast_mean, fast_std, C_FAST, f"Fast airflow ({FAST_BAND[0]}–{FAST_BAND[1]} m/s)"),
    ):
        ax.plot(ranks, mean, color=color, linewidth=2.2, zorder=3, label=label)
        ax.fill_between(
            ranks,
            np.maximum(mean - std, 1e-1),
            mean + std,
            color=color, alpha=ALPHA_BAND, zorder=2,
        )

    ax.axhline(INFECTION_THRESHOLD, **HLINE_STYLE)

    # annotate threshold crossings with vertical tick marks
    for cross, color in (
        (na_cross, C_NO_AIRFLOW),
        (slow_cross, C_SLOW),
        (fast_cross, C_FAST),
    ):
        ax.axvline(cross, color=color, linewidth=0.8, linestyle=":", alpha=0.6, zorder=2)

    #ax.set_yscale("log")
    ax.set_xlim(1, n_agents)
    ax.set_ylim(bottom=10)
    ax.set_xlabel("Agent rank (sorted by exposure, highest first)")
    ax.set_ylabel("Inhaled particles")

    legend_handles = [
        Line2D([0], [0], color=C_NO_AIRFLOW, linewidth=2.2, label="No airflow"),
        Line2D([0], [0], color=C_SLOW, linewidth=2.2,
               label=f"Slow north-to-south airflow ({SLOW_BAND[0]}–{SLOW_BAND[1]} m/s)"),
        Line2D([0], [0], color=C_FAST, linewidth=2.2,
               label=f"Fast north-to-south airflow ({FAST_BAND[0]}–{FAST_BAND[1]} m/s)"),
        Patch(facecolor="#888888", alpha=ALPHA_BAND, edgecolor="#888888", label="±1 std"),
        Line2D([0], [0], **{k: v for k, v in HLINE_STYLE.items() if k != "zorder"},
               label=f"Threshold ({INFECTION_THRESHOLD})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig("figures/rank_exposure_curves.pdf")
    plt.close(fig)
    print("Saved figures/rank_exposure_curves.pdf and .eps")


if __name__ == "__main__":
    main()