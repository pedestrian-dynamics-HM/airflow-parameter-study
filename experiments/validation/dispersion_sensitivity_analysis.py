from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

SCENARIOS_DIR = SCRIPT_DIR / "../model/vadere/scenarios/validation/dispersion_sensitivity"
JAR          = SCRIPT_DIR / "../model/vadere/simulator/vadere-console.jar"
OUTPUT_ROOT  = SCRIPT_DIR / "dispersion_sensitivity_output"

INFECTION_THRESHOLD = 179

SOURCE_TO_TABLE = {
    3000: "A",   3001: "B",   3002: "C",
    3004: "T05", 3005: "T06", 3006: "T07", 3007: "T08", 3008: "T09",
    3009: "T10", 3010: "T11", 3011: "T12", 3012: "T13", 3013: "T14",
    3014: "T15", 3015: "T16", 3016: "T17", 3017: "T18",
}

TABLE_ORDER = [
    "A", "B", "C",
    "T05", "T06", "T07", "T08", "T09", "T10",
    "T11", "T12", "T13", "T14", "T15", "T16", "T17", "T18",
]


def run_vadere(scenario_file: Path, output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java", "-jar", str(JAR),
        "--loglevel", "WARN",
        "scenario-run",
        "--scenario-file", str(scenario_file),
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Vadere exited with code {result.returncode}")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        return False
    return True


def find_output_file(output_dir: Path, filename: str) -> Path | None:
    direct = output_dir / filename
    if direct.exists():
        return direct
    subdirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for d in subdirs:
        f = d / filename
        if f.exists():
            return f
    return None


def load_per_table(run_dir: Path) -> pd.DataFrame | None:
    exp_file = find_output_file(run_dir, "maxDegreeOfExposure.txt")
    src_file = find_output_file(run_dir, "pedestrianSourceIdProcessor.txt")
    if exp_file is None:
        print(f"  ERROR: maxDegreeOfExposure.txt not found in {run_dir}")
        return None
    if src_file is None:
        print(f"  ERROR: pedestrianSourceIdProcessor.txt not found in {run_dir}")
        return None

    df = pd.merge(
        pd.read_csv(exp_file, sep=" "),
        pd.read_csv(src_file, sep=" "),
        on="pedestrianId",
        how="inner",
    )
    df["table"] = df["sourceId-PID14"].map(SOURCE_TO_TABLE)
    df["highly_exposed"] = (df["degreeOfExposure-PID13"] > INFECTION_THRESHOLD).astype(int)
    return df


def compute_per_table_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby("table").agg(
        n_agents=("degreeOfExposure-PID13", "count"),
        n_highly_exposed=("highly_exposed", "sum"),
        mean_exposure=("degreeOfExposure-PID13", "mean"),
    ).reset_index()
    stats["table"] = pd.Categorical(stats["table"], categories=TABLE_ORDER, ordered=True)
    return stats.sort_values("table").reset_index(drop=True)


def print_per_table(stats: pd.DataFrame, name: str, disp: float):
    print(f"\n{'='*65}")
    print(f"Scenario: {name}  (airDispersionFactor = {disp})")
    print(f"{'='*65}")
    print(f"  {'Table':<8} {'n_agents':>10} {'n_highly_exposed':>18} {'mean_exposure':>15}")
    print(f"  {'-'*55}")
    for _, row in stats.iterrows():
        print(f"  {row['table']:<8} {int(row['n_agents']):>10} "
              f"{int(row['n_highly_exposed']):>18} {row['mean_exposure']:>15.2f}")
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<8} {int(stats['n_agents'].sum()):>10} "
          f"{int(stats['n_highly_exposed'].sum()):>18}")


def get_disp_factor(path: Path) -> float:
    with open(path) as f:
        d = json.load(f)
    atm = d["scenario"]["attributesModel"][
        "org.vadere.state.attributes.models.infection.AttributesAirTransmissionModel"
    ]
    return atm["aerosolCloudParameters"]["airDispersionFactor"]


def main():
    if not JAR.exists():
        print(f"ERROR: vadere-console.jar not found: {JAR}")
        sys.exit(1)

    scenario_files = sorted(SCENARIOS_DIR.glob("*.scenario"))
    if not scenario_files:
        print(f"ERROR: no .scenario files found in {SCENARIOS_DIR}")
        sys.exit(1)

    all_rows: list[pd.DataFrame] = []

    for scenario_path in scenario_files:
        name = scenario_path.stem
        disp = get_disp_factor(scenario_path)
        run_dir = OUTPUT_ROOT / name

        print(f"\nLoading existing output for {name} (airDispersionFactor={disp})")
        print(f"\nRunning {name} (airDispersionFactor={disp}) -> {run_dir}")
        if not run_vadere(scenario_path, run_dir):
            print(f"  Skipping analysis for {name} due to run failure.")
            continue

        df = load_per_table(run_dir)
        if df is None:
            continue

        stats = compute_per_table_stats(df)
        print_per_table(stats, name, disp)

        stats.insert(0, "scenario", name)
        stats.insert(1, "airDispersionFactor", disp)
        all_rows.append(stats)

    if not all_rows:
        print("\nNo results to save.")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_rows, ignore_index=True)
    csv_path = OUTPUT_ROOT / "dispersion_sensitivity_results.csv"
    combined.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved: {csv_path}")

    rv_dir = SCRIPT_DIR / "restaurant_validation"
    disp = get_disp_factor(rv_dir / "restaurant_validation.scenario")
    df = load_per_table(rv_dir)
    if df is None:
        print("  ERROR: could not load restaurant_validation output")
        sys.exit(1)
    stats = compute_per_table_stats(df)
    print_per_table(stats, "restaurant_validation_disp003", disp)
    stats.insert(0, "scenario", "restaurant_validation_disp003")
    stats.insert(1, "airDispersionFactor", disp)

    combined = pd.concat([pd.read_csv(csv_path), stats], ignore_index=True)
    combined.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()