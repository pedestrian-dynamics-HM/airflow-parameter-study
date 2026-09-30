from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import re

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_airflow_label(folder_name: str) -> str:
    name = re.sub(r"_\d+$", "", folder_name.lower())
    if "no_airflow" in name:
        return "no airflow"
    m = re.search(r"inlet-(\w+)_outlet-(\w+)", name)
    if m:
        return f"{m.group(1)}-to-{m.group(2)} airflow"
    return folder_name


def patch_scenario(scenario_path: Path, edge_len: float, tmp_dir: str) -> Path:
    with open(scenario_path) as f:
        data = json.load(f)

    attrs = data["scenario"]["attributesModel"]
    airflow_key = "org.vadere.state.attributes.models.airflow.AttributesAirFlowModel"
    attrs[airflow_key]["maxTriangleEdgeLen"] = edge_len

    # skip rendering pipeline — Vadere still initialises it in headless mode otherwise
    sim_attrs = data["scenario"]["attributesSimulation"]
    sim_attrs["visualizationEnabled"] = False

    # drop postvis.traj output — only maxDegreeOfExposure.txt is needed;
    # postvis.traj writes all pedestrian positions at every event timestep and
    # dominates I/O for long runs
    pw = data["processWriters"]
    pw["files"] = [f for f in pw["files"] if f.get("filename") != "postvis.traj"]
    needed_ids = {pid for f in pw["files"] for pid in f.get("processors", [])}
    pw["processors"] = [p for p in pw["processors"] if p["id"] in needed_ids]

    tmp_path = Path(tmp_dir) / f"{scenario_path.stem}_h{edge_len}.scenario"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    return tmp_path


def run_vadere(jar: Path, scenario_file: Path, output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java", "-jar", str(jar),
        "--loglevel", "WARN",
        "scenario-run",
        "--scenario-file", str(scenario_file),
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Vadere exited with code {result.returncode}")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
    #    return False
    return True


def find_exposure_file(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    direct = output_dir / "maxDegreeOfExposure.txt"
    if direct.exists():
        return direct
    subdirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for d in subdirs:
        f = d / "maxDegreeOfExposure.txt"
        if f.exists():
            return f
    return None


def load_exposure(file_path: Path) -> np.ndarray:
    return np.loadtxt(file_path, skiprows=1, usecols=1)


def compute_stats(exposure: np.ndarray, threshold: float) -> dict:
    high = exposure >= threshold
    return {
        "mean": float(np.mean(exposure)),
        "std": float(np.std(exposure, ddof=1)),
        "median": float(np.median(exposure)),
        "n_total": len(exposure),
        "n_high": int(high.sum()),
        "frac_high": float(high.mean()),
    }


def print_table(
    results: list[dict],
    edgelens: list[float],
    threshold: float,
    seed_stats: list[list[dict] | None] | None = None,
    seed_names: list[list[str] | None] | None = None,
):
    if seed_stats is not None and any(s for s in seed_stats if s):
        print()
        print(f"Exposure statistics per seed  (threshold = {threshold})")
        print("=" * 70)
        print(f"{'edge len':>10} | {'seed':>30} | {'mean':>10} | {'n_high':>8}")
        print("-" * 70)
        for h, per_seed, names in zip(edgelens, seed_stats or [], seed_names or []):
            if not per_seed:
                print(f"{h:>10.4f} | {'FAILED':>30}")
                continue
            for name, s in zip(names or [""] * len(per_seed), per_seed):
                print(f"{h:>10.4f} | {name:>30} | {s['mean']:>10.6f} | {s['n_high']:>8d}")
        print()

    print()
    print(f"Exposure statistics by mesh size, averaged over seeds  (threshold = {threshold})")
    print("=" * 80)
    print(
        f"{'edge len':>10} | {'mean':>10} | {'std':>10} | {'median':>10} | "
        f"{'n_high':>8} | {'frac_high':>10}"
    )
    print("-" * 80)
    for h, r in zip(edgelens, results):
        if r is None:
            print(f"{h:>10.4f} | {'FAILED':>10}")
            continue
        print(
            f"{h:>10.4f} | {r['mean']:>10.6f} | {r['std']:>10.6f} | "
            f"{r['median']:>10.6f} | {r['n_high']:>8.1f} | {r['frac_high']:>10.4f}"
        )
    print()

    valid = [(h, r) for h, r in zip(edgelens, results) if r is not None]
    if len(valid) >= 2:
        print("Mean exposure change vs finest mesh")
        print("-" * 40)
        ref_mean = valid[-1][1]["mean"]
        for h, r in valid[:-1]:
            diff = abs(r["mean"] - ref_mean)
            print(f"  h={h:g}: |delta mean| = {diff:.6f}")



def plot_seed_distributions(
    seed_exposure_by_name: dict,
    edgelens: list[float],
    threshold: float,
    output_dir: Path,
    airflow_label: str = "",
):
    seed_names = sorted(seed_exposure_by_name.keys())
    n_seeds = len(seed_names)
    if n_seeds == 0:
        return

    # global y/x limits shared across all subplots
    global_ymax = max(
        float(np.max(exp))
        for exposures_by_h in seed_exposure_by_name.values()
        for exp in exposures_by_h.values()
    )
    global_xmax = max(
        len(exp)
        for exposures_by_h in seed_exposure_by_name.values()
        for exp in exposures_by_h.values()
    )

    ncols = min(n_seeds, 3)
    nrows = (n_seeds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(edgelens)))

    for idx, name in enumerate(seed_names):
        ax = axes[idx // ncols][idx % ncols]
        exposures_by_h = seed_exposure_by_name[name]

        sorted_exposures = {}
        for color, h in zip(colors, edgelens):
            if h not in exposures_by_h:
                continue
            exp = np.sort(exposures_by_h[h])[::-1]
            sorted_exposures[h] = exp
            ranks = np.arange(1, len(exp) + 1)
            ax.plot(ranks, exp, label=f"h={h:g}", color=color, linewidth=1.5)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0, label="high-risk threshold")
        ax.set_title(f"Seating configuration {idx + 1}", fontsize=10)
        ax.set_xlabel("Agent rank (1 = most exposed)")
        ax.set_ylabel("Exposure (inhaled pathogens)")
        ax.set_xlim(0, global_xmax)
        ax.set_ylim(0, global_ymax * 1.15)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.25)

        # inset zoomed around the threshold crossing
        crossing_xs = []
        for h, exp in sorted_exposures.items():
            above = np.where(exp >= threshold)[0]
            if len(above) > 0:
                crossing_xs.append(int(above[-1]) + 1)
        if crossing_xs:
            x_margin = max(5, int(0.08 * global_xmax))
            zoom_x0 = max(1, min(crossing_xs) - x_margin)
            zoom_x1 = min(global_xmax, max(crossing_xs) + x_margin)

            # y range from actual data in the zoom window, with padding so lines look gentle
            zoom_y0 = 0
            zoom_y1 = 500

            axins = ax.inset_axes([0.6, 0.46, 0.36, 0.36])
            for color, h in zip(colors, edgelens):
                if h not in sorted_exposures:
                    continue
                exp = sorted_exposures[h]
                ranks = np.arange(1, len(exp) + 1)
                axins.plot(ranks, exp, color=color, linewidth=1.2)
            axins.axhline(threshold, color="red", linestyle="--", linewidth=1.0)
            axins.set_xlim(zoom_x0, zoom_x1)
            axins.set_ylim(zoom_y0, zoom_y1)
            axins.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))
            axins.tick_params(labelsize=8)
            axins.grid(True, alpha=0.25)
            ax.indicate_inset_zoom(axins, edgecolor="gray")

    for idx in range(n_seeds, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    title = f"Exposure distribution for {airflow_label} (descending rank by mesh size)" if airflow_label else "Exposure distribution by seed and mesh size (descending rank)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    for suffix, kw in [(".pdf", {}), (".png", {"dpi": 200})]:
        out_path = output_dir / f"exposure_distributions_per_seed{suffix}"
        fig.savefig(out_path, bbox_inches="tight", **kw)
        print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run all seed scenarios in a folder at different mesh sizes and compare seed-averaged exposure."
    )
    parser.add_argument(
        "--scenario_folder",
        default=str(SCRIPT_DIR / "../../model/vadere/scenarios/choir_scenarios/choir_inlet-north_outlet-west_3"),
        help="Path to folder of .scenario files (one per seed)",
    )
    parser.add_argument(
        "--edgelens",
        nargs="+",
        type=float,
        default=[0.3, 0.2, 0.1],
        help="Mesh edge lengths, coarse to fine",
    )
    parser.add_argument(
        "--vadere_jar",
        default=str(SCRIPT_DIR / "../../model/vadere/simulator/vadere-console.jar"),
        help="Path to vadere-console.jar",
    )
    parser.add_argument(
        "--output_dir",
        default=str(SCRIPT_DIR / "output"),
        help="Root directory for Vadere run output",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=179.0,
        help="Exposure threshold for 'highly exposed' classification (agents strictly above this value)",
    )
    parser.add_argument(
        "--skip_runs",
        action="store_true",
        help="Skip Vadere runs and read existing output",
    )
    args = parser.parse_args()

    scenario_folder = Path(args.scenario_folder).resolve()
    jar = Path(args.vadere_jar).resolve()
    output_root = Path(args.output_dir).resolve()

    if not scenario_folder.exists():
        print(f"Error: scenario folder not found: {scenario_folder}")
        sys.exit(1)
    if not jar.exists():
        print(f"Error: vadere-console.jar not found: {jar}")
        sys.exit(1)

    scenarios = sorted(scenario_folder.glob("*.scenario"))
    if not scenarios:
        print(f"Error: no .scenario files in {scenario_folder}")
        sys.exit(1)

    output_root = output_root / scenario_folder.name
    results = []
    all_seed_stats = []
    all_seed_names = []
    seed_exposure_by_name: dict[str, dict[float, np.ndarray]] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for h in args.edgelens:
            print(f"\nMesh edge length h={h:g}")
            seed_exposures = []
            seed_stats = []
            seed_names = []

            for scenario_path in scenarios:
                run_dir = output_root / f"h{h:g}" / scenario_path.stem
                print(f"  Seed {scenario_path.stem}")

                if not args.skip_runs:
                    tmp_scenario = patch_scenario(scenario_path, h, tmp_dir)
                    ok = run_vadere(jar, tmp_scenario, run_dir)
                    if not ok:
                        continue
                else:
                    print(f"    Skipping run (--skip_runs), reading from {run_dir}")

                exposure_file = find_exposure_file(run_dir)
                if exposure_file is None:
                    print(f"    ERROR: maxDegreeOfExposure.txt not found in {run_dir}")
                    continue

                exposure = load_exposure(exposure_file)
                stats = compute_stats(exposure, args.threshold)
                seed_exposures.append(exposure)
                seed_stats.append(stats)
                seed_names.append(scenario_path.stem)
                seed_exposure_by_name.setdefault(scenario_path.stem, {})[h] = exposure
                print(
                    f"    n={stats['n_total']}  mean={stats['mean']:.4f}  "
                    f"highly_exposed={stats['n_high']} ({stats['frac_high']:.1%})"
                )

            if not seed_exposures:
                results.append(None)
                all_seed_stats.append(None)
                all_seed_names.append(None)
                continue

            combined = np.concatenate(seed_exposures)
            agg = {
                "mean": float(np.mean(combined)),
                "std": float(np.std(combined, ddof=1)),
                "median": float(np.median(combined)),
                "n_total": len(combined),
                "n_high": float(np.mean([s["n_high"] for s in seed_stats])),
                "frac_high": float(np.mean([s["frac_high"] for s in seed_stats])),
            }
            results.append(agg)
            all_seed_stats.append(seed_stats)
            all_seed_names.append(seed_names)

    print_table(results, args.edgelens, args.threshold, all_seed_stats, all_seed_names)
    plot_seed_distributions(
        seed_exposure_by_name,
        args.edgelens,
        args.threshold,
        output_root,
        airflow_label=parse_airflow_label(scenario_folder.name),
    )


if __name__ == "__main__":
    main()
