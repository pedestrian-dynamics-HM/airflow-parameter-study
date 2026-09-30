#!/usr/bin/env python3
import os

from vimuq.experiments.analysis import run_analysis

relPath2experiment = ".."
relPath2output = os.path.join(relPath2experiment, "output")

argv = [
    "--experimentid ishigami_sa_01",
    "--averagerepetitions D",  # or A (does not make a difference)
    "--outputpath {}".format(relPath2output),
]

if __name__ == "__main__":
    n_samples = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for i in range(len(n_samples)):
        if i > 0:
            argv.remove(f"--samples {str(n_samples[i-1])}")
        argv.append(f"--samples {str(n_samples[i])}")
        run_analysis.main(argv)
