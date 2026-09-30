import os

from vimuq.experiments.analysis import run_analysis


def experiment_01(cwd2vimuq_tests: str):
    """Checks if a previously executed experiment can be analyzed."""

    # setup experiment
    experiment_id = "experiment_01"

    path2output = os.path.join(cwd2vimuq_tests, "output")

    argv = [
        f"--experimentid {experiment_id}",
        # "--experimentsuffix YYYY-MM-DD_hh-mm-ss",
        f"--outputpath {path2output}",
    ]

    # run experiment
    run_analysis.main(argv)
    path2experiment = os.path.join(cwd2vimuq_tests, "output", experiment_id)

    # hash map
    # ToDo compare hash

    return path2experiment
