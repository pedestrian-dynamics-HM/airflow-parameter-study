import os

import vimuq
from vimuq.uq.uq_experiment import UQExperiment, Stage


PATH2OUTPUT = os.path.join("..", "output")

EXPERIMENT_IDS = {
    "inlet-north_outlet-south": "choir_uq_inlet-north_outlet-south_512",
    "inlet-north_outlet-west": "choir_uq_inlet-north_outlet-west_512",
    "inlet-north_outlet-north": "choir_uq_inlet-north_outlet-north_512",
}

SEEDS = {0: "002", 1: "006", 2: "011"}

QOI_MEAN = "maxDegreeOfExposure_mean"
QOI_COUNT = "maxDegreeOfExposure_count_>=_179"


def load_experiment(experiment_id):
    path2dumped = vimuq.get_experiment_dir_name(
        dir_name=os.path.join(PATH2OUTPUT, experiment_id), suffix=""
    )
    return UQExperiment.read_experiment(
        path2dumped, stage=Stage.POST_ANALYSIS, avrg_technique="C"
    )


def main():
    for config, experiment_id in EXPERIMENT_IDS.items():
        print(f"\nConfig: {config}")
        print(f"  {'Seed':<8} {'Mean exposure':>16} {'Highly exposed agents':>22}")
        print(f"  {'-'*8} {'-'*16} {'-'*22}")

        exp = load_experiment(experiment_id)
        mean_by_run = (
            exp.sim_output[QOI_MEAN]["degreeOfExposure-PID1"]
            .groupby(level="run_id")
            .mean()
        )
        count_by_run = (
            exp.sim_output[QOI_COUNT]["degreeOfExposure-PID1"]
            .groupby(level="run_id")
            .mean()
        )

        for run_id, seed_label in SEEDS.items():
            mean_val = mean_by_run.loc[run_id]
            count_val = count_by_run.loc[run_id]
            print(f"  {seed_label:<8} {mean_val:>16.2f} {count_val:>22.2f}")


if __name__ == "__main__":
    main()