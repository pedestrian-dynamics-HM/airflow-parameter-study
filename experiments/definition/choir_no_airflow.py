#!/usr/bin/env python3

# Forward propagation experiment with quasi random monte carlo sampling

import os.path
from vimuq.experiments.definition import run_experiment
from imports import relPath2output, relPath2scenarios, relPath2simulator
import time

scenario_folder_path = os.path.join(relPath2scenarios, "choir_scenarios/choir_no_airflow_3")
study_experiment_id = os.path.basename(__file__).split(".")[0] 
vadere_console_path = os.path.join(relPath2simulator, "vadere-console.jar")


argv = [
    f"--experimentid {study_experiment_id}",
    "--model vadere",
    f"--modelfile {vadere_console_path}",
    f"--outputpath {relPath2output}", 
    f"--scenario {scenario_folder_path}",
    "--parameter initialPathogenLoad UniformDistribution 1000 1000",
    "--output maxDegreeOfExposure",
    "--uqmethod TestUniformMonteCarlo",
    "--samplesize 1",
    "--runlocal True",
]

if __name__ == "__main__":
    start = time.time()
    run_experiment.main(argv)
    end = time.time()
    print(f"Total time: {end-start}")