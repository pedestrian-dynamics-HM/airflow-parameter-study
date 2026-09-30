#!/usr/bin/env python3
import os

from vimuq.experiments.definition import run_experiment

relPath2experiment = ".."
relPath2output = os.path.join(relPath2experiment, "output")
relPath2simulator = os.path.join(
    "..", "..", "..", "tests", "model", "pymodel", "ishigami.py"
)

filename = os.path.basename(__file__)
experiment_id = filename.split(".")[0]

argv = [
    f"--experimentid {experiment_id}",
    f"--modelfile {relPath2simulator}",
    "--model pymodel",
    "--parameter x1 UniformDistribution -3.14159265359 3.14159265359",
    "--parameter x2 UniformDistribution -3.14159265359 3.14159265359",
    "--parameter x3 UniformDistribution -3.14159265359 3.14159265359",
    "--output Y",
    f"--outputpath {relPath2output}",
    "--uqmethod SALibSaltelliSobolSequence",
    "--samplesize 4096",  # power of 2 for better performance of sobol sequence
]

if __name__ == "__main__":
    run_experiment.main(argv)
