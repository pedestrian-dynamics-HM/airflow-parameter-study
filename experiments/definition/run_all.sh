#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/suq-controller:$PROJECT_ROOT:$PYTHONPATH"

cd "$(dirname "$0")"

declare -a SUMMARY_SCRIPTS
declare -a SUMMARY_TIMES

run_timed() {
    local script="$1"
    local t0=$SECONDS
    pipenv run python3 "$script"
    local elapsed=$(( SECONDS - t0 ))
    printf "%-45s %ds\n" "$script" "$elapsed"
    SUMMARY_SCRIPTS+=("$script")
    SUMMARY_TIMES+=("$elapsed")
}

run_timed choir_uq_inlet-north_outlet-north.py
run_timed choir_uq_inlet-north_outlet-south.py
run_timed choir_uq_inlet-north_outlet-west.py

run_timed choir_uq_no_airflow.py

echo ""
echo "=== Summary ==="
total=0
for i in "${!SUMMARY_SCRIPTS[@]}"; do
    printf "%-45s %ds\n" "${SUMMARY_SCRIPTS[$i]}" "${SUMMARY_TIMES[$i]}"
    (( total += SUMMARY_TIMES[$i] ))
done
printf "%-45s %ds\n" "TOTAL" "$total"