#!/bin/bash
set -euo pipefail

CONDA_BIN="/home/sophia/miniconda3/bin/conda"
ENV_NAME="airflow-env"
VADERE_JAR="../../model/vadere/simulator/vadere-console.jar"

NAVIER_STOKES_PY="/home/sophia/Documents/vadere/vadere/VadereSimulator/src/org/vadere/simulator/models/airflow/python/navier_stokes.py"
COMPARE_ERRORS_PY="compare_errors.py"
COMPARE_EXPOSURE_SEEDS_PY="compare_exposure_seeds.py"

SCENARIO_FOLDER="../../model/vadere/scenarios/choir_scenarios/choir_inlet-north_outlet-south_3"

# airflow is identical across seeds — use the first scenario as representative
SCENARIO_PATH="$(ls "${SCENARIO_FOLDER}"/*.scenario | head -n 1)"
SCENARIO_NAME="$(basename "${SCENARIO_PATH}" .scenario)"
CACHE_PATH="$(dirname "${SCENARIO_PATH}")/cache"

HASHES=("03" "02" "01")
LEVELS=("1.0" "0.67" "0.33")
EDGELENS=("0.3" "0.2" "0.1")

# airflow domain bounds: XMIN XMAX YMIN YMAX (from scenario attributesAirFlowModel.bounds)
BOUNDS=(7.5 25.3 1.0 11.9)

# Sampling lines for velocity profile comparison
#SAMPLING_LINES=("x=16.4" "x=12.0" "y=6.45" "y=9.0" "y=3.0" "x=22.5" "x=20" "x=14")
#SAMPLING_LINES=("x=16.4" "y=6.45")
SAMPLING_LINES=("x=12" "y=6.45")
#SAMPLING_LINES=("x=12" "y=9.0")
SAMPLING_ARGS=()
for line in "${SAMPLING_LINES[@]}"; do
    SAMPLING_ARGS+=(--sampling_line "$line")
done

for i in "${!HASHES[@]}"; do
    hash="${HASHES[$i]}"
    level="${LEVELS[$i]}"
    echo "Running airflow precomputation for hash: ${hash} (level ${level})"
    "$CONDA_BIN" run -n "$ENV_NAME" python "$NAVIER_STOKES_PY" \
        "$SCENARIO_PATH" "$hash" --level "$level" --fast
done

echo "Running error comparison"
"$CONDA_BIN" run -n "$ENV_NAME" python "$COMPARE_ERRORS_PY" \
    --cache_path "$CACHE_PATH" --scenario_name "$SCENARIO_NAME" \
    --hashes "${HASHES[@]}" --edgelens "${EDGELENS[@]}" \
    --bounds "${BOUNDS[@]}" \
    --scenario_path "$SCENARIO_PATH" \
    --plot_dir "output/$(basename "$SCENARIO_FOLDER")" \
    "${SAMPLING_ARGS[@]}"

echo "Running seed-averaged exposure comparison"
"$CONDA_BIN" run -n "$ENV_NAME" python "$COMPARE_EXPOSURE_SEEDS_PY" \
    --scenario_folder "$SCENARIO_FOLDER" \
    --edgelens "${EDGELENS[@]}" \
    --vadere_jar "$VADERE_JAR" \
    --skip_runs
