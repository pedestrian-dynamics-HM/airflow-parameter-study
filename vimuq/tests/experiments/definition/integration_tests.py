import os

from vimuq.experiments.definition import run_experiment


def experiment_01(cwd2vadere_console: str, cwd2vimuq_tests: str):
    """Starts an example UQ experiment with Vadere to check if the whole
    experiment can be executed."""

    # setup experiment
    experiment_id = "experiment_01"
    path2scenario = os.path.join(
        cwd2vimuq_tests, "model", "vadere", "scenarios", "test_queue_uq_fast.scenario"
    )
    path2output = os.path.join(cwd2vimuq_tests, "output")

    argv = [
        f"--experimentid {experiment_id}",
        "--model vadere",
        f"--modelfile {cwd2vadere_console}",
        f"--scenario {path2scenario}",
        "--parameter halfLife UniformDistribution 10 1000",
        "--parameter initialRadius UniformDistribution 0.5 2.5",
        "--output testQoI_timeStepPedIdKey_degExposure",
        "--output testQoI_pedIdKey_degExposure",
        "--output testQoI_timeStepKey_noAerosolClouds",
        "--output testQoI_noDataKey_evacTime",
        "--runlocal True",
        "--repetitions 1",
        "--uqmethod SALibSaltelliSobolSequence",
        "--samplesize 2",
        f"--outputpath {path2output}",
    ]

    # run experiment
    run_experiment.main(argv)
    path2experiment = os.path.join(cwd2vimuq_tests, "output", experiment_id)

    # hash map
    # ToDo compare hash
