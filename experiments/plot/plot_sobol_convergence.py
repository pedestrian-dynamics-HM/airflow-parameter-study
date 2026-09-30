"""
Sobol' index convergence analysis for a single airflow configuration.
Computes indices on nested subsets of the existing Saltelli design.
"""

import os, pickle, sys, types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.analyze import sobol as sobol_analyze


import pandas.core.indexes
if not hasattr(pandas.core.indexes, "numeric"):
    _nm = types.ModuleType("pandas.core.indexes.numeric")
    _nm.Int64Index = pd.Index; _nm.Float64Index = pd.Index; _nm.UInt64Index = pd.Index
    sys.modules["pandas.core.indexes.numeric"] = _nm; pandas.core.indexes.numeric = _nm
if "numpy.core.numeric" not in sys.modules:
    import numpy._core.numeric; sys.modules["numpy.core.numeric"] = numpy._core.numeric

_STUB = ("vimuq", "suqc")
class _Unpkl(pickle.Unpickler):
    def find_class(self, module, name):
        if not any(module.startswith(p) for p in _STUB):
            return super().find_class(module, name)
        if module not in sys.modules:
            parts = module.split(".")
            for i in range(len(parts)):
                mp = ".".join(parts[:i+1])
                if mp not in sys.modules:
                    m = types.ModuleType(mp); sys.modules[mp] = m
                    if i > 0: setattr(sys.modules[".".join(parts[:i])], parts[i], m)
        mod = sys.modules.get(module)
        if mod and hasattr(mod, name): return getattr(mod, name)
        class PH:
            def __init__(s, *a, **kw): pass
            def __setstate__(s, st):
                if isinstance(st, dict): s.__dict__.update(st)
        PH.__name__ = name; PH.__qualname__ = name
        setattr(mod, name, PH); return PH


PKL_PATH = "../output/choir_uq_inlet-north_outlet-north_512/summary/uq_experiment_post_run.pkl"
CONFIG_LABEL = "North-to-north airflow"
OUTFILE = "figures/sobol_convergence_inlet-north_outlet-north.pdf"

EXPOSURE_THRESHOLD = 179
N_SUBS = [32, 48, 64, 96, 128, 192, 256, 384, 512]

PARAM_LABELS = ["Inlet velocity", "Inlet width", "Outlet width"]
PARAM_COLORS = ["#7B2D8E", "#D4442B", "#1A8A6E"]
S2_PAIR_LABELS = ["Inlet vel. &\ninlet width",
                  "Inlet vel. &\noutlet width",
                  "Inlet width &\noutlet width"]
S2_PAIR_COLORS = ["#8B5CF6", "#EC4899", "#14B8A6"]

QOIS = {"mean": "Mean degree of exposure",
        "count": "Number of highly exposed agents"}

with open(PKL_PATH, "rb") as f:
    obj = _Unpkl(f).load()

df = obj.sim_output["maxDegreeOfExposure"]
col = df.columns[0]
g = df.groupby(level=["id", "run_id"])[col]
Y_mean = g.mean().groupby(level="id").mean().values
Y_count = g.apply(lambda x: (x >= EXPOSURE_THRESHOLD).sum()).groupby(level="id").mean().values

pvars = obj.parameter_variations
names = list(pvars[0].keys())
mat = np.array([[p[k] for k in names] for p in pvars])
problem = {"num_vars": len(names), "names": names,
           "bounds": [[mat[:,i].min(), mat[:,i].max()] for i in range(len(names))]}

D = problem["num_vars"]
step = 2 * D + 2
results = {"mean": {}, "count": {}}
for n in N_SUBS:
    ne = n * step
    if ne > len(Y_mean):
        continue
    for qk, Y in [("mean", Y_mean), ("count", Y_count)]:
        results[qk][n] = sobol_analyze.analyze(
            problem, Y[:ne], calc_second_order=True,
            num_resamples=500, conf_level=0.95)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "#F9F9F9",
    "axes.grid": True, "grid.color": "#E0E0E0",
    "grid.linewidth": 0.5, "grid.alpha": 0.7,
})

index_types = ["S1", "ST", "S2"]
ylabels = ["First-order Sobol' index ($S_1$)",
           "Total Sobol' index ($S_T$)",
           "Second-order Sobol' index ($S_2$)"]
qoi_keys = ["mean", "count"]

fig, axes = plt.subplots(3, 2, figsize=(10, 9.5), constrained_layout=True)

for col, qk in enumerate(qoi_keys):
    ns = sorted(results[qk].keys())
    for row, idx in enumerate(index_types):
        ax = axes[row, col]

        if idx in ("S1", "ST"):
            labels, colors = PARAM_LABELS, PARAM_COLORS
            vals = np.array([results[qk][n][idx] for n in ns])
            conf = np.array([results[qk][n][f"{idx}_conf"] for n in ns])
        else:
            labels, colors = S2_PAIR_LABELS, S2_PAIR_COLORS
            pairs = [(0,1),(0,2),(1,2)]
            vals = np.array([[results[qk][n]["S2"][i,j] for i,j in pairs] for n in ns])
            conf = np.array([[results[qk][n]["S2_conf"][i,j] for i,j in pairs] for n in ns])

        for i in range(vals.shape[1]):
            v, c = vals[:,i], conf[:,i]
            ax.plot(ns, v, "o-", color=colors[i], markersize=4,
                    lw=1.5, label=labels[i], zorder=3)
            ax.fill_between(ns, v-c, v+c, color=colors[i], alpha=0.15, zorder=2)

        ax.set_ylabel(ylabels[row])
        ax.set_ylim(-0.2, 1.05)
        ax.axhline(0, color="#999999", lw=0.6, ls="--", zorder=1)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        if row == 2:
            ax.set_xlabel("Base sample size $N$")
        if row == 0:
            ax.set_title(QOIS[qk], fontweight="bold", pad=8)

fig.suptitle(f"Convergence of Sobol' indices — {CONFIG_LABEL}",
             fontsize=14, fontweight="bold", y=1.03)

os.makedirs("figures", exist_ok=True)
fig.savefig(OUTFILE, bbox_inches="tight", dpi=200)
fig.savefig(OUTFILE.replace(".pdf"), bbox_inches="tight")
print(f"Saved {OUTFILE}")
plt.close()