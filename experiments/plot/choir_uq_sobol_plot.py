import os

import vimuq
from vimuq.uq.uq_experiment import UQExperiment, Stage

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np


PATH2OUTPUT = os.path.join("..", "output")
AVRG = "B"

EXPERIMENT_IDS = {
    "inlet-north_outlet-south": "choir_uq_inlet-north_outlet-south_512",
    "inlet-north_outlet-west": "choir_uq_inlet-north_outlet-west_512",
    "inlet-north_outlet-north": "choir_uq_inlet-north_outlet-north_512",
}

QOIS = {
    "maxDegreeOfExposure_mean": "Mean degree of exposure",
    "maxDegreeOfExposure_count_>=_179": "Number of highly exposed agents",
}

COLORS = {
    "inlet-north_outlet-south": "#2E86AB",
    "inlet-north_outlet-west": "#E8963E",
    "inlet-north_outlet-north": "#56A06B",
}
CONFIG_LABELS = {
    "inlet-north_outlet-south": "North-to-south airflow",
    "inlet-north_outlet-west": "North-to-west airflow",
    "inlet-north_outlet-north": "North-to-north airflow",
}


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
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


def collect_sobol_data():
    """Return {qoi: {S1, S1_conf, ST, ST_conf, S2, S2_conf}} for all configs."""
    data = {}
    for qoi in QOIS:
        S1_df = pd.DataFrame()
        S1_conf_df = pd.DataFrame()
        ST_df = pd.DataFrame()
        ST_conf_df = pd.DataFrame()
        S2_df = pd.DataFrame()
        S2_conf_df = pd.DataFrame()

        for config, experiment_id in EXPERIMENT_IDS.items():
            exp = load_experiment(experiment_id, AVRG)

            S1_df[config] = exp.uq_output[qoi]["S1"]
            S1_conf_df[config] = exp.uq_output[qoi]["S1_conf"]
            ST_df[config] = exp.uq_output[qoi]["ST"]
            ST_conf_df[config] = exp.uq_output[qoi]["ST_conf"]

            config_df = exp.uq_output[qoi].reset_index().set_index("param")

            S2_df.at["inlet velocity &\n inlet width", config] = config_df.at[
                "inletVelocity", ("S2", "inlets.[side==north].width")
            ]
            S2_df.at["inlet velocity &\n outlet width", config] = config_df.at[
                "inletVelocity", config_df.columns[7]
            ]
            S2_df.at["inlet width &\n outlet width", config] = config_df.at[
                "inlets.[side==north].width", config_df.columns[7]
            ]
            S2_conf_df.at["inlet velocity &\n inlet width", config] = config_df.at[
                "inletVelocity", ("S2_conf", "inlets.[side==north].width")
            ]
            S2_conf_df.at["inlet velocity &\n outlet width", config] = config_df.at[
                "inletVelocity", config_df.columns[10]
            ]
            S2_conf_df.at["inlet width &\n outlet width", config] = config_df.at[
                "inlets.[side==north].width", config_df.columns[10]
            ]

        S1_df.index = ["inlet velocity", "inlet width", "outlet width"]
        S1_conf_df.index = S1_df.index
        ST_df.index = S1_df.index
        ST_conf_df.index = ST_df.index

        data[qoi] = {
            "S1": S1_df, "S1_conf": S1_conf_df,
            "ST": ST_df, "ST_conf": ST_conf_df,
            "S2": S2_df, "S2_conf": S2_conf_df,
        }
    return data


def plot_bars(ax, df, conf_df, ylabel):
    x = np.arange(len(df.index))
    n_configs = len(df.columns)
    width = 0.22
    offsets = np.linspace(
        -(n_configs - 1) / 2 * width, (n_configs - 1) / 2 * width, n_configs
    )
    for i, config in enumerate(df.columns):
        ax.bar(
            x + offsets[i],
            df[config].values.astype(float),
            width,
            yerr=conf_df[config].values.astype(float),
            capsize=3,
            color=COLORS[config],
            edgecolor="white",
            linewidth=0.8,
            error_kw=dict(lw=1.5, capthick=1.5, ecolor="#555555"),
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", length=0)


def save_sobol_csv(data, path="figures/sobol_indices.csv"):
    rows = []
    for qoi, indices in data.items():
        for idx_type in ["S1", "ST", "S2"]:
            df = indices[idx_type]
            conf_df = indices[f"{idx_type}_conf"]
            for param in df.index:
                for config in df.columns:
                    rows.append({
                        "qoi": qoi,
                        "index_type": idx_type,
                        "param": param.replace("\n", " "),
                        "config": config,
                        "value": df.at[param, config],
                        "conf": conf_df.at[param, config],
                    })
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
    print(f"Saved Sobol indices to {path}")


def print_sobol_data(data):
    for qoi, indices in data.items():
        print(f"\n{'='*70}")
        print(f"QOI: {qoi}")
        print(f"{'='*70}")
        for idx_type in ["S1", "ST", "S2"]:
            df = indices[idx_type]
            conf_df = indices[f"{idx_type}_conf"]
            cols_val = [f"{c}" for c in df.columns]
            cols_conf = [f"{c}_conf" for c in df.columns]
            combined = pd.concat(
                [df.rename(columns=dict(zip(df.columns, cols_val))),
                 conf_df.rename(columns=dict(zip(conf_df.columns, cols_conf)))],
                axis=1,
            )
            interleaved = []
            for c_val, c_conf in zip(cols_val, cols_conf):
                interleaved += [c_val, c_conf]
            print(f"\n  {idx_type}:")
            print(combined[interleaved].to_string(float_format=lambda x: f"{x:.4f}"))


def plot_sobol_6panel(data):
    fig = plt.figure(figsize=(11, 9.5), constrained_layout=False)
    gs_main = gridspec.GridSpec(
        3, 2, top=0.88, bottom=0.06, left=0.08, right=0.94,
        hspace=0.3, wspace=0.30, figure=fig,
    )

    qoi_keys = list(QOIS.keys())
    row_types = ["S1", "ST", "S2"]
    ylabels = [
        "1st order Sobol' index ($S_1$)",
        "Total Sobol' index ($S_T$)",
        "2nd order Sobol' index ($S_2$)",
    ]

    for row in range(3):
        for col in range(2):
            ax = fig.add_subplot(gs_main[row, col])
            plot_bars(
                ax,
                data[qoi_keys[col]][row_types[row]],
                data[qoi_keys[col]][f"{row_types[row]}_conf"],
                ylabels[row],
            )
    print_sobol_data(data)

    for col, qoi_key in enumerate(qoi_keys):
        pos = gs_main[0, col].get_position(fig)
        fig.text(
            (pos.x0 + pos.x1) / 2, 0.915,
            QOIS[qoi_key],
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#222222",
        )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[c], ec="white", lw=0.5)
        for c in EXPERIMENT_IDS
    ]
    legend_labels = [CONFIG_LABELS[c] for c in EXPERIMENT_IDS]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.005),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        facecolor="white",
        columnspacing=2.0,
        handlelength=1.6,
        handletextpad=0.6,
        fontsize=11,
    )

    fig.savefig("figures/sobol_indices_6panel.png", bbox_inches="tight")
    fig.savefig("figures/sobol_indices_6panel.eps", bbox_inches="tight")
    plt.show()


def main():
    os.makedirs("figures", exist_ok=True)
    apply_style()
    data = collect_sobol_data()
    save_sobol_csv(data)
    plot_sobol_6panel(data)


if __name__ == "__main__":
    main()