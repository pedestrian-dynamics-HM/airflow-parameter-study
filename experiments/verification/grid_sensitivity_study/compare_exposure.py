from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


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

    tmp_path = Path(tmp_dir) / f"scenario_h{edge_len}.scenario"
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
        return False
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


def plot_exposure_distribution(
    all_exposure: list,
    edgelens: list[float],
    threshold: float,
    output_dir: Path,
    title: str = "",
):
    valid = [(h, exp) for h, exp in zip(edgelens, all_exposure) if exp is not None]
    if not valid:
        return

    global_ymax = max(float(np.max(exp)) for _, exp in valid)
    global_xmax = max(len(exp) for _, exp in valid)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(edgelens)))

    sorted_exposures = {}
    for color, (h, exp) in zip(colors, valid):
        s = np.sort(exp)[::-1]
        sorted_exposures[h] = s
        ax.plot(np.arange(1, len(s) + 1), s, label=f"h={h:g}", color=color, linewidth=1.5)

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0, label="high-risk threshold")
    ax.set_xlabel("Agent rank (1 = most exposed)")
    ax.set_ylabel("Max degree of exposure")
    ax.set_xlim(0, global_xmax)
    ax.set_ylim(0, global_ymax * 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)

    # inset zoomed around threshold crossing
    crossing_xs = []
    for h, s in sorted_exposures.items():
        above = np.where(s >= threshold)[0]
        if len(above) > 0:
            crossing_xs.append(int(above[-1]) + 1)
    if crossing_xs:
        x_margin = max(5, int(0.08 * global_xmax))
        zoom_x0 = max(1, min(crossing_xs) - x_margin)
        zoom_x1 = min(global_xmax, max(crossing_xs) + x_margin)
        y_vals = []
        for s in sorted_exposures.values():
            ranks = np.arange(1, len(s) + 1)
            y_vals.extend(s[(ranks >= zoom_x0) & (ranks <= zoom_x1)].tolist())
        if y_vals:
            axins = ax.inset_axes([0.55, 0.46, 0.36, 0.36])
            for color, (h, _) in zip(colors, valid):
                s = sorted_exposures[h]
                axins.plot(np.arange(1, len(s) + 1), s, color=color, linewidth=1.2)
            axins.axhline(threshold, color="red", linestyle="--", linewidth=1.0)
            axins.set_xlim(zoom_x0, zoom_x1)
            axins.set_ylim(0, 500)
            axins.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))
            axins.tick_params(labelsize=6)
            axins.grid(True, alpha=0.25)
            ax.indicate_inset_zoom(axins, edgecolor="gray")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix, kw in [(".pdf", {}), (".png", {"dpi": 200})]:
        out_path = output_dir / f"exposure_distribution{suffix}"
        fig.savefig(out_path, bbox_inches="tight", **kw)
        print(f"Saved: {out_path}")
    plt.close(fig)


def print_table(results: list[dict], edgelens: list[float], threshold: float):
    print()
    print(f"Exposure statistics by mesh size  (threshold = {threshold})")
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
            f"{r['median']:>10.6f} | {r['n_high']:>8d} | {r['frac_high']:>10.4f}"
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


def main():
    parser = argparse.ArgumentParser(
        description="Run Vadere scenarios at different mesh sizes and compare exposure."
    )
    parser.add_argument(
        "--scenario_path",
        default=str(SCRIPT_DIR / "scenarios" / "miller-2020-life_seed_002_nts.scenario"),
        help="Path to base scenario file",
    )
    parser.add_argument(
        "--edgelens",
        nargs="+",
        type=float,
        default=[0.6, 0.3, 0.15, 0.075],
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

    scenario_path = Path(args.scenario_path).resolve()
    jar = Path(args.vadere_jar).resolve()
    output_root = Path(args.output_dir).resolve()

    if not scenario_path.exists():
        print(f"Error: scenario not found: {scenario_path}")
        sys.exit(1)
    if not jar.exists():
        print(f"Error: vadere-console.jar not found: {jar}")
        sys.exit(1)

    scenario_name = scenario_path.stem
    output_root = output_root / scenario_name
    all_exposure = []
    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for h in args.edgelens:
            run_dir = output_root / f"h{h:g}"
            print(f"\nMesh edge length h={h:g}")

            if not args.skip_runs:
                print(f"  Patching scenario -> maxTriangleEdgeLen={h}")
                tmp_scenario = patch_scenario(scenario_path, h, tmp_dir)
                print(f"  Running Vadere -> {run_dir}")
                ok = run_vadere(jar, tmp_scenario, run_dir)
                if not ok:
                    all_exposure.append(None)
                    results.append(None)
                    continue
            else:
                print(f"  Skipping run (--skip_runs), reading from {run_dir}")

            exposure_file = find_exposure_file(run_dir)
            if exposure_file is None:
                print(f"  ERROR: maxDegreeOfExposure.txt not found in {run_dir}")
                all_exposure.append(None)
                results.append(None)
                continue

            print(f"  Reading {exposure_file}")
            exposure = load_exposure(exposure_file)
            stats = compute_stats(exposure, args.threshold)
            all_exposure.append(exposure)
            results.append(stats)
            print(
                f"  n={stats['n_total']}  mean={stats['mean']:.4f}  "
                f"std={stats['std']:.4f}  highly_exposed={stats['n_high']} "
                f"({stats['frac_high']:.1%})"
            )

    print_table(results, args.edgelens, args.threshold)
    plot_exposure_distribution(
        all_exposure, args.edgelens, args.threshold, output_root,
        title=f"Exposure distribution — {scenario_name}",
    )

if __name__ == "__main__":
    main()