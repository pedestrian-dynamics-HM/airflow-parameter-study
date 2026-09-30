#!/bin/bash
set -euo pipefail

CONDA_BIN="/home/sophia/miniconda3/bin/conda"
ENV_NAME="airflow-env"
VADERE_JAR="../../model/vadere/simulator/vadere-console.jar"

NAVIER_STOKES_PY="/home/sophia/Documents/vadere/vadere/VadereSimulator/src/org/vadere/simulator/models/airflow/python/navier_stokes.py"
COMPARE_ERRORS_PY="compare_errors.py"
COMPARE_EXPOSURE_PY="compare_exposure.py"

SCENARIO_PATH="../../model/vadere/scenarios/validation/restaurant_validation.scenario"
SCENARIO_NAME="$(basename "${SCENARIO_PATH}" .scenario)"
CACHE_PATH="$(dirname "${SCENARIO_PATH}")/cache"

HASHES=("03" "02" "01")
LEVELS=("1.5" "1.0" "0.5")
EDGELENS=("0.3" "0.2" "0.1")

# bounds of the airflow field: XMIN XMAX YMIN YMAX
BOUNDS=(0.2 8.7 2.3 19.8)

# Sampling lines: "x=<value>" = vertical line, "y=<value>" = horizontal line.
SAMPLING_LINES=("x=5.0" "y=8.04")

SAMPLING_ARGS=()
for line in "${SAMPLING_LINES[@]}"; do
    SAMPLING_ARGS+=(--sampling_line "$line")
done


for i in "${!HASHES[@]}"; do
    hash="${HASHES[$i]}"
    level="${LEVELS[$i]}"
    #echo "Running simulation for hash: ${hash} (level ${level})"
    #"$CONDA_BIN" run -n "$ENV_NAME" python "$NAVIER_STOKES_PY" \
    #    "$SCENARIO_PATH" "$hash" --level "$level" --fast
done

echo "Running error comparison"
"$CONDA_BIN" run -n "$ENV_NAME" python "$COMPARE_ERRORS_PY" \
    --cache_path "$CACHE_PATH" --scenario_name "$SCENARIO_NAME" \
    --hashes "${HASHES[@]}" --edgelens "${EDGELENS[@]}" \
    --bounds "${BOUNDS[@]}" \
    --scenario_path "$SCENARIO_PATH" \
    --plot_dir "output/restaurant_validation" \
    "${SAMPLING_ARGS[@]}"


echo "Running exposure comparison"
"$CONDA_BIN" run -n "$ENV_NAME" python "$COMPARE_EXPOSURE_PY" \
    --scenario_path "$SCENARIO_PATH" \
    --edgelens "${EDGELENS[@]}" \
    --vadere_jar "$VADERE_JAR" \
    --output_dir "output" \
    --skip_runs