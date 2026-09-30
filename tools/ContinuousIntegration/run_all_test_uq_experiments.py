import argparse
import shutil
import sys
import os

import vimuq.tests.experiments.definition.integration_tests as definition
import vimuq.tests.experiments.analysis.integration_tests as analysis

path2project = os.path.join("..", "..")
path2tests = os.path.join(path2project, "vimuq", "tests")
path2vadere_console = os.path.join(
    path2project, "experiments", "model", "vadere", "simulator", "vadere-console.jar"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vadere-jar",
    type=str,
    default=path2vadere_console,
    help="""Path to vadere-console.jar""",
)

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    definition.experiment_01(
        cwd2vadere_console=args.vadere_jar, cwd2vimuq_tests=path2tests
    )
    path2experiment = analysis.experiment_01(cwd2vimuq_tests=path2tests)

    # remove output
    if os.path.exists(path2experiment):
        shutil.rmtree(path2experiment)
        print(f"Removed {path2experiment}")
