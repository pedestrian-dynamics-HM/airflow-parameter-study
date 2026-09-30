import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # never open windows; figures are only saved to disk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import ticker
from matplotlib.colors import PowerNorm


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def cache_base(cache_path, scenario_name, scenario_hash):
    return f"{cache_path}/{scenario_name}_{scenario_hash}"


def load_velocity_field(cache_path, scenario_name, scenario_hash):
    """Load interpolated velocity components and their valid-point mask."""
    base = cache_base(cache_path, scenario_name, scenario_hash)
    vx_file = f"{base}_Vx.txt"
    vy_file = f"{base}_Vy.txt"

    if not os.path.exists(vx_file) or not os.path.exists(vy_file):
        print(f"Error: Could not find files for hash {scenario_hash}")
        print(f"Looking for: {vx_file}")
        sys.exit(1)

    Vx = np.loadtxt(vx_file)
    Vy = np.loadtxt(vy_file)
    valid = np.isfinite(Vx) & np.isfinite(Vy)
    return Vx, Vy, valid


def grid_from_bounds(shape, bounds):
    """Reconstruct X, Y of the regular grid from --bounds (row 0 at ymin)."""
    rows, cols = shape
    xmin, xmax, ymin, ymax = bounds
    return np.meshgrid(np.linspace(xmin, xmax, cols), np.linspace(ymin, ymax, rows))


def load_obstacles(scenario_path):
    """
    Read obstacle outlines from a Vadere .scenario (JSON) file.
    Returns a list of (N, 2) vertex arrays. Supports POLYGON, RECTANGLE, CIRCLE.
    """
    if scenario_path is None:
        return []
    try:
        with open(scenario_path, "r") as f:
            data = json.load(f)
        topo = data.get("scenario", data).get("topography", {})
        raw = topo.get("obstacles", [])
    except (OSError, ValueError, AttributeError) as err:
        print(f"WARNING: could not read obstacles from {scenario_path}: {err}")
        return []

    obstacles = []
    for obs in raw:
        shape = obs.get("shape") or obs.get("attributes", {}).get("shape")
        if not shape:
            continue
        kind = shape.get("type", "").upper()
        if kind == "POLYGON":
            obstacles.append(np.array([[p["x"], p["y"]] for p in shape["points"]]))
        elif kind == "RECTANGLE":
            x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
            obstacles.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
        elif kind == "CIRCLE":
            c, r = shape["center"], shape["radius"]
            t = np.linspace(0, 2 * np.pi, 64)
            obstacles.append(np.column_stack([c["x"] + r * np.cos(t), c["y"] + r * np.sin(t)]))
    return obstacles


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rel_l2(Vx_a, Vy_a, Vx_b, Vy_b, mask=None):
    """Relative vector L2 difference ||U_a-U_b|| / ||U_b|| on mask."""
    if mask is None:
        mask = (
            np.isfinite(Vx_a) & np.isfinite(Vy_a)
            & np.isfinite(Vx_b) & np.isfinite(Vy_b)
        )

    if not np.any(mask):
        return np.nan

    dx = Vx_a[mask] - Vx_b[mask]
    dy = Vy_a[mask] - Vy_b[mask]
    num = np.sqrt(np.sum(dx**2 + dy**2))
    den = np.sqrt(np.sum(Vx_b[mask]**2 + Vy_b[mask]**2))
    return 0.0 if den == 0 else num / den


def parse_sampling_line(spec):
    """
    argparse type for --sampling_line.

    'x=7.5' -> ('x', 7.5): vertical line at x = 7.5 m (profile along y)
    'y=4.0' -> ('y', 4.0): horizontal line at y = 4.0 m (profile along x)
    """
    try:
        axis, value = spec.split("=", 1)
        axis = axis.strip().lower()
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid sampling line '{spec}'. Expected 'x=<value>' or 'y=<value>'."
        )
    if axis not in ("x", "y"):
        raise argparse.ArgumentTypeError(
            f"Invalid axis '{axis}' in sampling line '{spec}'. Use 'x' or 'y'."
        )
    return axis, value


def sampling_lines(X, Y, specs):
    """
    Build line definitions from the actual grid coordinates.
    Column j has x = X[0, j], row i has y = Y[i, 0]. Each requested position
    is snapped to the nearest grid line.
    """
    x = X[0, :]
    y = Y[:, 0]

    lines = []
    for n, (axis, value) in enumerate(specs, start=1):
        line_id = f"L{n}"
        coords = x if axis == "x" else y
        lo, hi = float(np.min(coords)), float(np.max(coords))
        if not lo <= value <= hi:
            raise ValueError(
                f"Sampling line {axis}={value} lies outside the grid "
                f"{axis}-range [{lo:.3f}, {hi:.3f}]."
            )
        idx = int(np.argmin(np.abs(coords - value)))

        if axis == "x":
            lines.append({
                "id": line_id,
                "tag": f"x{x[idx]:.2f}",
                "name": f"Vertical line x={x[idx]:.2f} m",
                "selector": (slice(None), idx),
                "coordinate": y,
                "xlabel": "y [m]",
            })
        else:
            lines.append({
                "id": line_id,
                "tag": f"y{y[idx]:.2f}",
                "name": f"Horizontal line y={y[idx]:.2f} m",
                "selector": (idx, slice(None)),
                "coordinate": x,
                "xlabel": "x [m]",
            })

    return lines


def print_sampling_line_errors(fields, common, edgelens, line_defs):
    """
    Compare velocity fields along the given sampling lines.

    Reports both vector-component L2 error and speed-magnitude L2 error,
    always relative to the finest mesh.
    """
    Vx_ref, Vy_ref = fields[-1]

    print()
    print("Sampling-line velocity errors vs finest mesh")
    print("=" * 82)

    for line in line_defs:
        name = line["name"]
        selector = line["selector"]
        line_common = common[selector]

        print()
        print(name)
        print("-" * 82)
        if not np.any(line_common):
            print("WARNING: no commonly valid points on this line "
                  "(it may lie entirely inside a wall/obstacle).")
        print(
            f"{'edge length':<14} | "
            f"{'vector L2':<14} | "
            f"{'speed L2':<14} | "
            f"{'valid points':<12}"
        )
        print("-" * 82)

        for i, (Vx, Vy) in enumerate(fields[:-1]):
            vx = Vx[selector]
            vy = Vy[selector]
            vx_ref = Vx_ref[selector]
            vy_ref = Vy_ref[selector]
            mask = line_common

            if not np.any(mask):
                error = np.nan
                speed_error = np.nan
            else:
                dx = vx[mask] - vx_ref[mask]
                dy = vy[mask] - vy_ref[mask]
                numerator = np.sqrt(np.sum(dx**2 + dy**2))
                denominator = np.sqrt(np.sum(vx_ref[mask]**2 + vy_ref[mask]**2))
                error = 0.0 if denominator == 0 else numerator / denominator

                speed = np.sqrt(vx**2 + vy**2)
                speed_ref = np.sqrt(vx_ref**2 + vy_ref**2)
                ds = speed[mask] - speed_ref[mask]
                speed_num = np.sqrt(np.sum(ds**2))
                speed_den = np.sqrt(np.sum(speed_ref[mask]**2))
                speed_error = 0.0 if speed_den == 0 else speed_num / speed_den

            print(
                f"{edgelens[i]:<14.4f} | "
                f"{error:<14.6e} | "
                f"{speed_error:<14.6e} | "
                f"{int(mask.sum()):<12}"
            )

        print(
            f"{edgelens[-1]:<14.4f} | "
            f"{'reference':<14} | "
            f"{'reference':<14} | "
            f"{int(line_common.sum()):<12}"
        )


def plot_sampling_line_profiles(fields, common, edgelens, line_defs, output_dir, scenario_name):
    """
    Save one |U| profile figure for each sampling line.

    Invalid points (walls/obstacles/outside fluid) are written as NaN, so
    matplotlib leaves gaps rather than drawing through solid geometry.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for line in line_defs:
        selector = line["selector"]
        coord = line["coordinate"]
        mask = common[selector]

        fig, ax = plt.subplots(figsize=(8.0, 4.8))

        for (Vx, Vy), h in zip(fields, edgelens):
            speed = np.sqrt(Vx[selector]**2 + Vy[selector]**2)
            profile = np.where(mask, speed, np.nan)
            ax.plot(coord, profile, linewidth=1.4, label=f"h = {h:g} m")

        ax.set_xlabel(line["xlabel"])
        ax.set_ylabel(r"$|U|$ [m/s]")
        ax.set_title(line["name"])
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        filename = output_dir / f"{scenario_name}_sampling_{line['id']}_{line['tag']}_speed.png"
        fig.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(filename)

    print()
    print("Sampling-line figures")
    print("=" * 82)
    for filename in saved:
        print(filename)

    return saved


def plot_combined_panel(fields, valids, common, grid, obstacles, edgelens,
                        line_defs, output_dir, scenario_name):
    """
    Combined paper figure: row (a) velocity fields per mesh, row (b) speed
    profiles per sampling line across all meshes.
    """
    import matplotlib.gridspec as gridspec

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, Y = grid
    n_meshes = len(fields)
    n_lines = len(line_defs)
    x_min, x_max = float(np.min(X)), float(np.max(X))
    y_min, y_max = float(np.min(Y)), float(np.max(Y))
    domain_aspect = (y_max - y_min) / (x_max - x_min)

    base = plt.cm.Blues
    trunc_blues = mcolors.LinearSegmentedColormap.from_list(
        "trunc_blues", base(np.linspace(0.3, 1.0, 256))
    )
    mesh_colors = plt.cm.Blues(np.linspace(0.4, 1.0, n_meshes))

    vmax = 1e-12
    for (Vx, Vy), valid in zip(fields, valids):
        speed = np.sqrt(np.where(valid, Vx, np.nan) ** 2
                        + np.where(valid, Vy, np.nan) ** 2)
        vmax = max(vmax, float(np.nanmax(speed)))
    levels = np.linspace(0, vmax, 100)
    norm = PowerNorm(gamma=0.6, vmin=0, vmax=vmax)

    rc = {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
    }

    row_b_height = 2.5
    with plt.rc_context(rc):
        row_a_height = max(4.0 * domain_aspect, 3.5)
        fig_width = n_meshes * 3.8
        fig_height = row_a_height + row_b_height
        fig = plt.figure(figsize=(fig_width, fig_height))
        hspace = 0.20 + max(0.0, (1.0 - domain_aspect) * 0.8)
        gs_outer = gridspec.GridSpec(
            2, 1, figure=fig,
            height_ratios=[row_a_height, row_b_height],
            hspace=hspace,
            left=0.07, right=0.97, top=0.95, bottom=0.06,
        )
        gs_a = gridspec.GridSpecFromSubplotSpec(
            1, n_meshes, subplot_spec=gs_outer[0], wspace=0.06,
        )
        gs_b = gridspec.GridSpecFromSubplotSpec(
            1, n_lines, subplot_spec=gs_outer[1], wspace=0.40,
        )

        axes_row_a = []
        contour_mappables = []
        for j in range(n_meshes):
            ax = fig.add_subplot(gs_a[0, j])

            Vx, Vy = fields[j]
            valid = valids[j]
            vx = np.where(valid, Vx, np.nan)
            vy = np.where(valid, Vy, np.nan)
            vel_mag = np.sqrt(vx ** 2 + vy ** 2)

            Xp, Yp, Vxp, Vyp, magp = X, Y, vx, vy, vel_mag
            if Yp[-1, 0] < Yp[0, 0]:
                Xp, Yp, Vxp, Vyp, magp = (np.flipud(a) for a in (Xp, Yp, Vxp, Vyp, magp))
            if Xp[0, -1] < Xp[0, 0]:
                Xp, Yp, Vxp, Vyp, magp = (np.fliplr(a) for a in (Xp, Yp, Vxp, Vyp, magp))

            cf = ax.contourf(Xp, Yp, magp, levels=levels, cmap=trunc_blues, norm=norm)
            ax.streamplot(Xp, Yp, np.nan_to_num(Vxp), np.nan_to_num(Vyp),
                          color="white", linewidth=1.0, density=0.8,
                          arrowsize=1.0, arrowstyle="->")
            for obs in obstacles:
                obs_arr = np.asarray(obs)
                ax.fill(obs_arr[:, 0], obs_arr[:, 1], color="grey", alpha=1.0, zorder=10)

            for line in line_defs:
                axis = line["tag"][0]
                value = float(line["tag"][1:])
                if axis == "x":
                    ax.axvline(x=value, linewidth=1.0, linestyle="--",
                               color="#CC0000", zorder=5)
                else:
                    ax.axhline(y=value, linewidth=1.0, linestyle="--",
                               color="#CC0000", zorder=5)

            ax.set_title(f"$h = {edgelens[j]:g}$ m")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("x [m]")
            if j == 0:
                ax.set_ylabel("y [m]")
            else:
                ax.tick_params(labelleft=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor("white")

            axes_row_a.append(ax)
            contour_mappables.append(cf)

        cb_pad = max(0.06, 0.4 / (row_a_height * domain_aspect))
        cb_fraction = max(0.025, 0.12 / row_a_height)
        cb_shrink = min(0.9, max(0.55, 0.55 / domain_aspect))
        cb = fig.colorbar(contour_mappables[-1], ax=axes_row_a, location="bottom",
                          fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
        cb.set_label("Velocity [m/s]", fontsize=8)
        cb.locator = ticker.MaxNLocator(nbins=5)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=7)

        for j in range(n_lines):
            ax = fig.add_subplot(gs_b[0, j])

            line = line_defs[j]
            selector = line["selector"]
            coord = line["coordinate"]
            mask = common[selector]

            for k, ((Vx, Vy), h) in enumerate(zip(fields, edgelens)):
                speed = np.sqrt(Vx[selector] ** 2 + Vy[selector] ** 2)
                profile = np.where(mask, speed, np.nan)
                ax.plot(coord, profile, linewidth=1.4, color=mesh_colors[k],
                        label=f"h = {h:g} m")

            ax.set_xlabel(line["xlabel"])
            ax.set_ylabel(r"$|U|$ [m/s]")
            #ax.set_ylim(0,0.1)
            ax.set_ylim(0,0.6)
            ax.set_title(line["name"])
            ax.grid(True, color="#E0E0E0", linewidth=0.5, alpha=0.7)
            if j == n_lines - 1:
                ax.legend()

        fig.text(0.01, 0.99, "(a)", fontsize=11, fontweight="bold", va="top")
        fig.text(0.01, row_b_height / fig_height + 0.02, "(b)", fontsize=11,
                 fontweight="bold", va="top")

        png_path = output_dir / f"{scenario_name}_grid_sensitivity_panel.png"
        pdf_path = output_dir / f"{scenario_name}_grid_sensitivity_panel.pdf"
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

    print()
    print("Combined panel figure")
    print("=" * 82)
    print(png_path)
    print(pdf_path)

    return [png_path, pdf_path]


# ---------------------------------------------------------------------------
# Per-mesh field plots
# ---------------------------------------------------------------------------

def plot_results(X, Y, Vx, Vy, vel_mag, obstacles, path, mesh_2d=None):
    """
    Save mesh and streamline plots as separate figures (never shown).

    Parameters
    ----------
    X, Y, Vx, Vy, vel_mag : rectangular grid arrays (invalid points = NaN)
    obstacles : list of obstacle polygon vertices
    path : output file base path (string)
    mesh_2d : optional dict with 'coords' (N,2) and 'triangles' (M,3);
              the mesh figure is skipped if None
    """
    # streamplot needs strictly increasing rows/columns
    if Y[-1, 0] < Y[0, 0]:
        X, Y, Vx, Vy, vel_mag = (np.flipud(a) for a in (X, Y, Vx, Vy, vel_mag))
    if X[0, -1] < X[0, 0]:
        X, Y, Vx, Vy, vel_mag = (np.fliplr(a) for a in (X, Y, Vx, Vy, vel_mag))

    base = plt.cm.Blues
    trunc_blues = mcolors.LinearSegmentedColormap.from_list(
        'trunc_blues',
        base(np.linspace(0.3, 1.0, 256))
    )

    rc = {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }

    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    aspect = (x_max - x_min) / (y_max - y_min)

    plot_height = 7
    plot_width = plot_height * aspect

    def draw_obstacles(ax):
        for obs in obstacles:
            obs_arr = np.asarray(obs)
            ax.fill(obs_arr[:, 0], obs_arr[:, 1], color='grey', alpha=1.0, zorder=10)

    def format_axes(ax):
        ax.set_xlabel("x (m)")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    saved = []
    with plt.rc_context(rc):
        # Figure 1: Mesh plot
        if mesh_2d is not None:
            coords = mesh_2d['coords']
            conn = mesh_2d['triangles']

            fig_mesh, ax_mesh = plt.subplots(figsize=(plot_width, plot_height),
                                             constrained_layout=True)
            ax_mesh.triplot(coords[:, 0], coords[:, 1], conn,
                            color='k', linewidth=0.8, alpha=0.9)
            draw_obstacles(ax_mesh)
            ax_mesh.set_ylabel("y (m)")
            format_axes(ax_mesh)
            fig_mesh.savefig(path + "_mesh.svg")
            plt.close(fig_mesh)
            saved.append(path + "_mesh.svg")

        # Figure 2: Streamlines
        vmax = max(float(np.nanmax(vel_mag)), 1e-12)
        fig_stream, ax_stream = plt.subplots(figsize=(plot_width, plot_height),
                                             constrained_layout=True)
        levels = np.linspace(0, vmax, 100)
        cf1 = ax_stream.contourf(X, Y, vel_mag, levels=levels, cmap=trunc_blues,
                                 norm=PowerNorm(gamma=0.6))
        ax_stream.streamplot(X, Y, np.nan_to_num(Vx), np.nan_to_num(Vy),
                             color='white', linewidth=2.0,
                             density=1.0, arrowsize=2.0, arrowstyle='->')
        draw_obstacles(ax_stream)
        cb1 = fig_stream.colorbar(cf1, ax=ax_stream, location='bottom',
                                  fraction=0.05, pad=0.02)
        cb1.set_label('Velocity (m/s)')
        cb1.locator = ticker.MaxNLocator(nbins=max(3, min(6, int(plot_width))))  # avoid crowded ticks on narrow domains
        cb1.update_ticks()
        ax_stream.set_ylabel("y (m)")
        format_axes(ax_stream)
        fig_stream.savefig(path + "_streamlines.svg")
        fig_stream.savefig(path + "_streamlines.png")
        plt.close(fig_stream)
        saved += [path + "_streamlines.svg", path + "_streamlines.png"]

    return saved


def plot_all_meshes(fields, valids, grid, obstacles, args):
    """Save streamline figures for every run into the cache folder."""
    X, Y = grid
    print()
    print("Per-mesh field figures")
    print("=" * 82)
    for (Vx, Vy), valid, scenario_hash, h in zip(fields, valids, args.hashes, args.edgelens):
        vx = np.where(valid, Vx, np.nan)
        vy = np.where(valid, Vy, np.nan)
        vel_mag = np.sqrt(vx**2 + vy**2)

        path = cache_base(args.cache_path, args.scenario_name, scenario_hash)
        for f in plot_results(X, Y, vx, vy, vel_mag, obstacles, path):
            print(f"h = {h:g} m: {f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare mesh sensitivity of multiple airflow runs.")
    parser.add_argument("--cache_path", required=True, help="Path to cache folder")
    parser.add_argument("--scenario_name", required=True, help="Scenario name (without extension)")
    parser.add_argument(
        "--hashes", required=True, nargs="+",
        help="List of hashes ordered from COARSE to FINE. The last hash is the reference field."
    )
    parser.add_argument(
        "--edgelens", required=True, nargs="+", type=float,
        help="Bulk cell size h of each run (level * maxTriangleEdgeLen)."
    )
    parser.add_argument(
        "--scenario_path", default=None,
        help="Vadere .scenario file; used to draw obstacles in the field plots."
    )
    parser.add_argument(
        "--bounds", nargs=4, type=float, required=True,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Extent of the saved regular grid in metres."
    )
    parser.add_argument(
        "--sampling_line", action="append", type=parse_sampling_line, default=[],
        metavar="AXIS=VALUE",
        help="Sampling line, e.g. 'x=7.5' (vertical line) or 'y=4.0' "
             "(horizontal line). Can be given multiple times."
    )
    parser.add_argument(
        "--no_field_plots", action="store_true",
        help="Skip the per-mesh streamline figures."
    )
    parser.add_argument(
        "--plot_dir", default=None,
        help="Directory for sampling-line figures. Default: <cache_path>/../../output/grid_sensitivity_plots"
    )
    args = parser.parse_args()

    if len(args.hashes) != len(args.edgelens):
        parser.error("Number of hashes must match number of edge lengths.")

    xmin, xmax, ymin, ymax = args.bounds
    if xmin >= xmax or ymin >= ymax:
        parser.error("--bounds must satisfy XMIN < XMAX and YMIN < YMAX.")

    # Load all runs and form a mask containing only points that are valid for
    # every mesh. This makes all field and line comparisons use identical points.
    fields = []
    valids = []
    common = None
    ref_shape = None

    for scenario_hash in args.hashes:
        Vx, Vy, valid = load_velocity_field(
            args.cache_path, args.scenario_name, scenario_hash
        )

        if ref_shape is None:
            ref_shape = Vx.shape
        elif Vx.shape != ref_shape:
            raise ValueError(
                f"All saved regular-grid fields must have the same shape. "
                f"Expected {ref_shape}, got {Vx.shape} for hash {scenario_hash}."
            )

        fields.append((Vx, Vy))
        valids.append(valid)
        common = valid.copy() if common is None else (common & valid)

    rows, cols = ref_shape
    print(f"Grid dimensions: {rows}x{cols}")
    print(f"Common valid points: {int(common.sum())} / {common.size}")

    # Physical grid coordinates from --bounds.
    grid = grid_from_bounds(ref_shape, args.bounds)

    # Difference to the finest mesh. Keep the original full-field metric
    # unchanged; the common validity mask is used specifically for the
    # sampling-line comparisons below.
    Vx_ref, Vy_ref = fields[-1]
    print()
    print(f"{'edge length':<12} | {'rel L2 diff vs finest':<21} | {'apparent slope':<14}")
    print("-" * 54)
    prev_h = prev_d = None

    for i in range(len(args.hashes) - 1):
        d = rel_l2(*fields[i], Vx_ref, Vy_ref)
        slope = "-"
        if prev_h is not None and np.isfinite(d) and d > 1e-12:
            slope = f"{np.log(prev_d / d) / np.log(prev_h / args.edgelens[i]):.4f}"
        print(f"{args.edgelens[i]:<12.4f} | {d:<21.6e} | {slope:<14}")
        prev_h, prev_d = args.edgelens[i], d

    # Successive-grid differences, preserving the original full-field metric.
    diffs = [
        rel_l2(*fields[i], *fields[i + 1])
        for i in range(len(args.hashes) - 1)
    ]

    print()
    print(f"{'pair':<16} | {'rel L2 diff':<15} | {'r':<5} | {'observed order':<14}")
    print("-" * 60)
    for i, d in enumerate(diffs):
        h_c, h_f = args.edgelens[i], args.edgelens[i + 1]
        r = h_c / h_f
        p = "-"
        if (
            i + 1 < len(diffs)
            and np.isfinite(d) and np.isfinite(diffs[i + 1])
            and d > 1e-12 and diffs[i + 1] > 1e-12
        ):
            p = f"{np.log(d / diffs[i + 1]) / np.log(r):.4f}"
        print(f"{f'{h_c:g} -> {h_f:g}':<16} | {d:<15.6e} | {r:<5.2f} | {p:<14}")

    obstacles = load_obstacles(args.scenario_path)

    # Sampling lines
    if args.sampling_line:
        try:
            line_defs = sampling_lines(grid[0], grid[1], args.sampling_line)
        except ValueError as err:
            parser.error(str(err))

        print_sampling_line_errors(fields, common, args.edgelens, line_defs)

        plot_dir = args.plot_dir or str(Path(args.cache_path).parent.parent / "output" / "grid_sensitivity_plots")
        plot_combined_panel(
            fields, valids, common, grid, obstacles,
            args.edgelens, line_defs, plot_dir, args.scenario_name,
        )
    else:
        print()
        print("No --sampling_line given; skipping sampling-line comparison.")

    # Per-mesh streamline figures
    if not args.no_field_plots:
        plot_all_meshes(fields, valids, grid, obstacles, args)


if __name__ == "__main__":
    main()