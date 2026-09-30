import os

import vimuq
from vimuq.uq.uq_experiment import UQExperiment, Stage

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PATH2OUTPUT = os.path.join("..", "output")
NR_REPS = 3


AVRG_CONFIG_COMPARISON = "B"
AVRG_PER_PARAMETER = "C"

EXPERIMENT_IDS = {
    "inlet-north_outlet-south": "choir_uq_inlet-north_outlet-south_512",
    "inlet-north_outlet-west": "choir_uq_inlet-north_outlet-west_512",
    "inlet-north_outlet-north": "choir_uq_inlet-north_outlet-north_512",
}

QOI_MEAN = "maxDegreeOfExposure_mean"
QOI_COUNT = "maxDegreeOfExposure_count_>=_179"

COLORS = {
    "inlet-north_outlet-south": "#2E86AB",
    "inlet-north_outlet-west": "#E8963E",
    "inlet-north_outlet-north": "#56A06B",
}
XLABEL_MAP = {
    "inlet-north_outlet-south": "north-to-south",
    "inlet-north_outlet-west": "north-to-west",
    "inlet-north_outlet-north": "north-to-north",
}

COLOR_MEAN = "#2E86AB"
COLOR_COUNT = "#56A06B"

HLINE_STYLE = dict(
    linestyle=(0, (4, 1.5, 1, 1.5)),
    color="#555555",
    linewidth=3.0,
    zorder=4,
)
EXPOSURE_NO_AIRFLOW = 962
COUNT_NO_AIRFLOW = 31.7

TITLE_MEAN = "Mean exposure"
TITLE_COUNT = "Highly exposed agents"

PARAM_LABELS = {
    "inletVelocity": "Inlet velocity [m/s]",
    "inlets.[side==north].width": "Inlet width [m]",
    "outlets.[side==south].width": "Outlet width [m]",
}


def param_label(key):
    if key in PARAM_LABELS:
        return PARAM_LABELS[key]
    if key.startswith("inlets") and key.endswith(".width"):
        return "Inlet width [m]"
    if key.startswith("outlets") and key.endswith(".width"):
        return "Outlet width [m]"
    return key


def param_token(key):
    if key == "inletVelocity":
        return "inletVelocity"
    if key.startswith("inlets") and key.endswith(".width"):
        return "inletWidth"
    if key.startswith("outlets") and key.endswith(".width"):
        return "outletWidth"
    return (
        key.replace(".", "_").replace("[", "").replace("]", "").replace("==", "-")
    )


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
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
    })


def load_experiment(experiment_id, avrg):
    path2dumped = vimuq.get_experiment_dir_name(
        dir_name=os.path.join(PATH2OUTPUT, experiment_id), suffix=""
    )
    return UQExperiment.read_experiment(
        path2dumped, stage=Stage.POST_ANALYSIS, avrg_technique=avrg
    )


def get_binned_data(exp, qoi, key, nr_reps):
    key_df = exp.sim_output[qoi].copy()
    values = [entry[key] for entry in exp.parameter_variations]
    repeated_values = [val for val in values for _ in range(nr_reps)]
    key_df[key] = repeated_values

    key_df["key_bin"] = pd.cut(key_df[key], bins=20)
    binned = key_df.groupby("key_bin")["degreeOfExposure-PID1"].agg(["mean", "std"])
    binned["center"] = binned.index.map(lambda x: x.mid)

    return binned["center"], binned["mean"], binned["std"]


def draw_mc_curve(ax, x, y, yerr, color, hline_val, ylim, linewidth=2.2):
    ax.plot(x, y, color=color, linewidth=linewidth, zorder=3)
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.18, zorder=2)
    ax.axhline(hline_val, **HLINE_STYLE)
    ax.set_ylim(*ylim)
    ax.margins(x=0)


def mc_legend_handles(color):
    return [
        Patch(facecolor=color, alpha=0.18, edgecolor=color, label="±1 std"),
        Line2D(
            [0], [0],
            **{k: v for k, v in HLINE_STYLE.items() if k != "zorder"},
            label="Without airflow",
        ),
    ]

def plot_config_comparison():
    exp_df = pd.DataFrame()
    for config, experiment_id in EXPERIMENT_IDS.items():
        exp = load_experiment(experiment_id, AVRG_CONFIG_COMPARISON)
        mean_series = exp.sim_output[QOI_MEAN]["degreeOfExposure-PID1"]
        count_series = exp.sim_output[QOI_COUNT]["degreeOfExposure-PID1"]
        exp_df.at[config, "mean"] = mean_series.mean()
        exp_df.at[config, "mean_std"] = mean_series.std()
        exp_df.at[config, "count"] = count_series.mean()
        exp_df.at[config, "count_std"] = count_series.std()

    configs = list(EXPERIMENT_IDS.keys())
    x = np.arange(len(configs))
    bar_colors = [COLORS[c] for c in configs]
    xticklabels = [XLABEL_MAP[c] for c in configs]

    fig, (ax_mean, ax_count) = plt.subplots(1, 2, figsize=(12, 3.8))

    for ax, value_col, std_col, title, hline, ylabel in (
            (ax_mean, "mean", "mean_std", TITLE_MEAN, EXPOSURE_NO_AIRFLOW, TITLE_MEAN),
            (ax_count, "count", "count_std", "Number of highly exposed agents", COUNT_NO_AIRFLOW, TITLE_COUNT),
    ):
        ax.bar(
            x, exp_df[value_col].values,
            yerr=exp_df[std_col].values,
            capsize=4,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.axhline(hline, label="Without airflow", **HLINE_STYLE)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Airflow configuration")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout(w_pad=4)
    fig.savefig("figures/mc_configs.png", bbox_inches="tight")
    fig.savefig("figures/mc_configs.eps", bbox_inches="tight")
    plt.show()
    print(exp_df)


# per-parameter pair (mean exposure + highly exposed count side by side)
def plot_param_pair(exp, experiment_id, key):
    xlabel = param_label(key)
    fig, (ax_mean, ax_count) = plt.subplots(1, 2, figsize=(11, 3.6))

    x, y, yerr = get_binned_data(exp, QOI_MEAN, key, NR_REPS)
    draw_mc_curve(ax_mean, x, y, yerr, COLOR_MEAN, EXPOSURE_NO_AIRFLOW, (0, 1000))
    ax_mean.set_title(TITLE_MEAN, fontweight="bold")
    ax_mean.set_xlabel(xlabel)
    ax_mean.set_ylabel(TITLE_MEAN)
    ax_mean.legend(handles=mc_legend_handles(COLOR_MEAN),
                   loc="upper right", framealpha=0.9)

    x, y, yerr = get_binned_data(exp, QOI_COUNT, key, NR_REPS)
    draw_mc_curve(ax_count, x, y, yerr, COLOR_COUNT, COUNT_NO_AIRFLOW, (0, 60))
    ax_count.set_title(TITLE_COUNT, fontweight="bold")
    ax_count.set_xlabel(xlabel)
    ax_count.set_ylabel(TITLE_COUNT)
    ax_count.legend(handles=mc_legend_handles(COLOR_COUNT),
                    loc="upper right", framealpha=0.9)

    fig.tight_layout(w_pad=4)
    fig.savefig(
        f"figures/mc_{experiment_id}_{param_token(key)}.pdf", bbox_inches="tight"
    )
    plt.close(fig)

def plot_all_configs_grid(exps):
    configs = list(EXPERIMENT_IDS.keys())

    fig = plt.figure(figsize=(10, 13))
    outer = gridspec.GridSpec(
        len(configs), 1, figure=fig,
        top=0.92, bottom=0.05, left=0.10, right=0.97, hspace=0.30,
    )

    for i, config in enumerate(configs):
        exp = exps[config]
        keys = list(exp.parameter_variations[0].keys())
        inner = gridspec.GridSpecFromSubplotSpec(
            2, len(keys), subplot_spec=outer[i], hspace=0.12, wspace=0.10,
        )

        for col, key in enumerate(keys):
            ax0 = fig.add_subplot(inner[0, col])
            x, y, yerr = get_binned_data(exp, QOI_MEAN, key, NR_REPS)
            draw_mc_curve(ax0, x, y, yerr, COLOR_MEAN,
                          EXPOSURE_NO_AIRFLOW, (0, 1000), linewidth=1.8)
            ax0.set_xticklabels([])
            if col == 0:
                ax0.set_ylabel(TITLE_MEAN)
            else:
                ax0.set_yticklabels([])

            ax1 = fig.add_subplot(inner[1, col])
            x, y, yerr = get_binned_data(exp, QOI_COUNT, key, NR_REPS)
            draw_mc_curve(ax1, x, y, yerr, COLOR_COUNT,
                          COUNT_NO_AIRFLOW, (0, 60), linewidth=1.8)
            ax1.set_xlabel(param_label(key))
            if col == 0:
                ax1.set_ylabel(TITLE_COUNT)
            else:
                ax1.set_yticklabels([])

        block_title = f"{XLABEL_MAP[config]} airflow"
        block_title = block_title[0].upper() + block_title[1:]
        pos = outer[i].get_position(fig)
        fig.text(
            (pos.x0 + pos.x1) / 2, pos.y1 + 0.008,
            block_title,
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#222222",
        )

    legend_elements = [
        Patch(facecolor=COLOR_MEAN, alpha=0.18, edgecolor=COLOR_MEAN, label="±1 std"),
        Patch(facecolor=COLOR_COUNT, alpha=0.18, edgecolor=COLOR_COUNT, label="±1 std"),
        Line2D(
            [0], [0],
            **{k: v for k, v in HLINE_STYLE.items() if k != "zorder"},
            label="Without airflow",
        ),
    ]
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, 0.975),
    )

    fig.savefig("figures/mc_grid_all_configs.pdf", bbox_inches="tight")
    plt.show()


def main():
    os.makedirs("figures", exist_ok=True)
    apply_style()

    plot_config_comparison()

    # load each experiment once for the per-parameter plots and the grid
    exps = {
        config: load_experiment(experiment_id, AVRG_PER_PARAMETER)
        for config, experiment_id in EXPERIMENT_IDS.items()
    }

    # per-parameter pairs, for every config
    for config, experiment_id in EXPERIMENT_IDS.items():
        exp = exps[config]
        for key in exp.parameter_variations[0].keys():
            plot_param_pair(exp, experiment_id, key)

    plot_all_configs_grid(exps)


if __name__ == "__main__":
    main()


