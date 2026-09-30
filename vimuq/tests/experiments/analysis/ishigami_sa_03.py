#!/usr/bin/env python3
import os

from vimuq.experiments.analysis import run_analysis

relPath2experiment = ".."
relPath2output = os.path.join(relPath2experiment, "output")

argv = [
    "--averagerepetitions D",  # or A (does not make a difference)
    "--outputpath {}".format(relPath2output),
]

if __name__ == "__main__":
    n_samples = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for i, n in enumerate(n_samples):
        argv_ = argv.copy()
        argv_.append(f"--samples {str(n)}")
        argv_.append(f"--experimentid ishigami_sa_03_{str(i + 1).zfill(2)}")
        run_analysis.main(argv_)
