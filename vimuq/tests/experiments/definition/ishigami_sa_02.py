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
    f"--modelfile {relPath2simulator}",
    "--model pymodel",
    "--parameter x1 UniformDistribution -3.14159265359 3.14159265359",
    "--parameter x2 UniformDistribution -3.14159265359 3.14159265359",
    "--parameter x3 UniformDistribution -3.14159265359 3.14159265359",
    "--output Y",
    f"--outputpath {relPath2output}",
    "--uqmethod SALibSaltelliSobolSequence",
]

if __name__ == "__main__":
    n_samples = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for i, n in enumerate(n_samples):
        argv_ = argv.copy()
        argv_.append(f"--experimentid {experiment_id}_{str(i + 1).zfill(2)}")
        argv_.append(f"--samplesize {str(n)}")
        run_experiment.main(argv_)
