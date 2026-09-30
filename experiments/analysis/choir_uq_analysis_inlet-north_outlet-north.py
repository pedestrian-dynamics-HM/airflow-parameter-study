#!/usr/bin/env python3

from imports import relPath2output

from vimuq.experiments.analysis import run_analysis

argv = [
    "--experimentid choir_uq_inlet-north_outlet-north_512",
    "--outputpath {}".format(relPath2output),
    "--quantitiesofinterest "
    "maxDegreeOfExposure_mean "
    "maxDegreeOfExposure_count_>=_179 ",
    "--averagerepetitions B",
    "--applyfunction maxDegreeOfExposure mean id",
    "--applyfunction maxDegreeOfExposure count_>=_179 id",
]

if __name__ == "__main__":
    run_analysis.main(argv)
