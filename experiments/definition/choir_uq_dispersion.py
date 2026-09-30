#!/usr/bin/env python3

# Forward propagation experiment with quasi random monte carlo sampling

import os.path
from vimuq.experiments.definition import run_experiment
from imports import relPath2output, relPath2scenarios, relPath2simulator
import time

scenario_folder_path = os.path.join(relPath2scenarios, "choir_scenarios/choir_inlet-north_outlet-south_10")
study_experiment_id = os.path.basename(__file__).split(".")[0] 
vadere_console_path = os.path.join(relPath2simulator, "vadere-console.jar")

argv = [
    f"--experimentid {study_experiment_id}",
    "--model vadere",
    f"--modelfile {vadere_console_path}",
    f"--outputpath {relPath2output}", 
    f"--scenario {scenario_folder_path}",
    "--parameter airDispersionFactor UniformDistribution 0 0.006",
    "--parameter inletVelocity UniformDistribution 0.1 1.0",
    "--parameter inlets.[side==north].width UniformDistribution 0.1 1.0",
    "--parameter outlets.[side==south].width UniformDistribution 0.1 1.0",
    "--output maxDegreeOfExposure",
    "--uqmethod SALibSaltelliSobolSequence",
    "--samplesize 64",
    "--secondorder True",
    "--runlocal True",
]

if __name__ == "__main__":
    start = time.time()
    run_experiment.main(argv)
    end = time.time()
    print(f"Total time: {end-start}")