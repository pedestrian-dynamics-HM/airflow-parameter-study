import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

INFECTION_THRESHOLD = 179

output_dir = '../../experiments/validation/restaurant_validation'
output_dir_tracer_gas = '../../experiments/validation/restaurant_validation_tracer_gas'
fig_dir = '../../experiments/validation/figures'
os.makedirs(fig_dir, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C_SIM = '#3274A1'
C_TRACER = '#E8913A'
C_CFD = '#E8915A'
C_THRESHOLD = '#CC3333'
C_INFECTED = '#CC3333'

# LOAD DATA — full agent set (all agents at all tables)
df_exp = pd.read_csv(os.path.join(output_dir, 'maxDegreeOfExposure.txt'), sep=' ')
df_src = pd.read_csv(os.path.join(output_dir, 'pedestrianSourceIdProcessor.txt'), sep=' ')

df_exp['infected'] = (df_exp['degreeOfExposure-PID13'] > INFECTION_THRESHOLD).astype(int)
result_df = pd.merge(df_exp, df_src, on='pedestrianId', how='inner')

source_to_table = {
    3000: 'A', 3001: 'B', 3002: 'C', 3004: 'T05', 3005: 'T06', 3006: 'T07',
    3007: 'T08', 3008: 'T09', 3009: 'T10', 3010: 'T11', 3011: 'T12', 3012: 'T13',
    3013: 'T14', 3014: 'T15', 3015: 'T16', 3016: 'T17', 3017: 'T18'
}
result_df['table'] = result_df['sourceId-PID14'].map(source_to_table)

avg_exposure_full = result_df.groupby('table')['degreeOfExposure-PID13'].mean()

print("highly exposed agents per table")
print(result_df.groupby('table')['infected'].sum().sort_index())
print()

df_exp_tg = pd.read_csv(
    os.path.join(output_dir_tracer_gas, 'maxDegreeOfExposure.txt'), sep=' '
)
df_src_tg = pd.read_csv(
    os.path.join(output_dir_tracer_gas, 'pedestrianSourceIdProcessor.txt'), sep=' '
)

result_tg = pd.merge(df_exp_tg, df_src_tg, on='pedestrianId', how='inner')
result_tg['table'] = result_tg['sourceId-PID14'].map(source_to_table)

avg_exposure_tg = result_tg.groupby('table')['degreeOfExposure-PID13'].mean()

tracer_gas = {
    'A2': 1.00, 'B1': 0.87, 'C1': 0.98,
    'T10': 0.55, 'T15': 0.58, 'T16': 0.70, 'T17': 0.86, 'T18': 0.73
}

# map tracer gas locations to simulation values
sim_for_tracer = {}
for loc in tracer_gas:
    if loc in ['A2', 'B1', 'C1']:
        table_name = loc[0]
        sim_for_tracer[loc] = avg_exposure_tg.get(table_name, 0.0)
    else:
        sim_for_tracer[loc] = avg_exposure_tg.get(loc, 0.0)

infected_status = {
    'A2': True, 'B1': True, 'C1': True,
    'T10': False, 'T15': False, 'T16': False, 'T17': False, 'T18': False
}

# statistical analysis
locations = list(tracer_gas.keys())
tg_vals = [tracer_gas[loc] for loc in locations]
sim_vals_for_corr = [sim_for_tracer[loc] for loc in locations]

rho, p_value = stats.spearmanr(tg_vals, sim_vals_for_corr)

print("tracer gas vs simulation comparison")
print(f"\n{'Location':<8} {'Tracer gas':>12} {'Simulation':>12} {'Sim (norm)':>12} {'Infected':>10}")
print("-" * 60)
sim_max = max(sim_vals_for_corr)
for loc in locations:
    tg = tracer_gas[loc]
    sim = sim_for_tracer[loc]
    sim_n = sim / sim_max if sim_max > 0 else 0
    inf = "yes" if infected_status[loc] else "no"
    print(f"{loc:<8} {tg:>12.2f} {sim:>12.1f} {sim_n:>12.4f} {inf:>10}")

tg_norm_mae = np.array(tg_vals) / max(tg_vals)
sim_norm_mae = np.array(sim_vals_for_corr) / max(sim_vals_for_corr)
mae_tg = np.mean(np.abs(tg_norm_mae - sim_norm_mae))

print(f"\nSpearman rho = {rho:.3f}")
print(f"p-value      = {p_value:.4f}")
print(f"n            = {len(locations)}")
print(f"MAE (norm.)  = {mae_tg:.4f}")

tg_ranking = sorted(locations, key=lambda x: tracer_gas[x], reverse=True)
sim_ranking = sorted(locations, key=lambda x: sim_for_tracer[x], reverse=True)
print(f"\nTracer gas ranking: {' > '.join(tg_ranking)}")
print(f"Simulation ranking: {' > '.join(sim_ranking)}")

# plot 1: individual agent exposure per table
table_order = ['A', 'B', 'C', 'T05', 'T06', 'T07', 'T08', 'T09',
               'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18']

all_x, all_heights, all_colors = [], [], []
tick_positions, tick_labels = [], []
current_x = 0
gap_width = 1.5

for table in table_order:
    group = result_df[result_df['table'] == table]
    if len(group) == 0:
        continue
    exposures = group['degreeOfExposure-PID13'].sort_values(ascending=False).values
    n = len(exposures)
    xs = list(range(int(current_x), int(current_x + n)))
    all_x.extend(xs)
    all_heights.extend(exposures)
    #if table in ['A', 'B', 'C']:
    all_colors.extend([C_SIM] * n)
    #else:
    #    all_colors.extend(['#7FAECE'] * n)
    tick_positions.append(current_x + (n - 1) / 2)
    tick_labels.append(table)
    current_x += n + gap_width

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(all_x, all_heights, width=0.8, color=all_colors,
       edgecolor='white', linewidth=0.3, zorder=3)
ax.axhline(INFECTION_THRESHOLD, linestyle='--', color=C_THRESHOLD, linewidth=1.5,
           label='High-risk threshold', zorder=4)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_xlabel('Agents grouped by table')
ax.set_ylabel('Exposure (inhaled particles)')
ax.yaxis.grid(True, linestyle=':', alpha=0.3)
ax.set_axisbelow(True)
ax.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_individual_exposure_per_table.eps'))
plt.savefig(os.path.join(fig_dir, 'fig_individual_exposure_per_table.pdf'))
plt.close()


# plot 2: tracer gas vs simulation
loc_sorted = sorted(locations, key=lambda x: tracer_gas[x], reverse=True)
tg_sorted = [tracer_gas[loc] for loc in loc_sorted]
sim_sorted = [sim_for_tracer[loc] for loc in loc_sorted]
inf_sorted = [infected_status[loc] for loc in loc_sorted]

tg_norm = [v / max(tg_sorted) for v in tg_sorted]
sim_norm = [v / sim_sorted[0] for v in sim_sorted]

x = np.arange(len(loc_sorted))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 3.5))

bars_tg = ax.bar(x - width / 2, tg_norm, width,
                 color=C_TRACER, alpha=0.85, edgecolor='white', linewidth=0.5,
                 label='Tracer gas (Li et al. 2021)', zorder=3)
bars_sim = ax.bar(x + width / 2, sim_norm, width,
                  color=C_SIM, alpha=0.85, edgecolor='white', linewidth=0.5,
                  label='Our model: 2D airflow in Vadere', zorder=3)

for i, inf in enumerate(inf_sorted):
    if inf:
        ax.plot(x[i], max(tg_norm[i], sim_norm[i]) + 0.04, marker='v',
                color=C_INFECTED, markersize=8, zorder=5, clip_on=False)

ax.set_xticks(x)
ax.set_xticklabels(loc_sorted)
ax.set_xlabel('Measurement location')
ax.set_ylabel('Normalized concentration')
ax.set_ylim(0, 1.08)
ax.yaxis.grid(True, linestyle=':', alpha=0.3)
ax.set_axisbelow(True)

legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc=C_TRACER, alpha=0.85, label='Tracer gas (Li et al. 2021)'),
    plt.Rectangle((0, 0), 1, 1, fc=C_SIM, alpha=0.85, label='Our model: 2D airflow in Vadere'),
    plt.Line2D([0], [0], marker='v', color=C_INFECTED, linestyle='None',
               markersize=8, label='Infected location'),
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_tracer_gas_comparison_linear.eps'))
plt.savefig(os.path.join(fig_dir, 'fig_tracer_gas_comparison_linear.pdf'))
plt.close()

# plot 3: 2D model vs 3D CFD
cfd_tables = ['A', 'B', 'C', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10',
              'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18']
cfd_values = [1.00, 0.76, 0.89, 0.07, 0.13, 0.04, 0.06,
              0.04, 0.08, 0.12, 0.09, 0.05, 0.11, 0.23,
              0.06, 0.47, 0.40]
cfd_data = dict(zip(cfd_tables, cfd_values))

max_avg = avg_exposure_full.max()
norm_sim_full = avg_exposure_full / max_avg
norm_threshold = INFECTION_THRESHOLD / max_avg

all_tables = sorted(set(list(norm_sim_full.index) + cfd_tables),
                    key=lambda t: (0, t) if len(t) == 1 else (1, t))
all_tables = [t for t in all_tables if t != 'T04']

sim_cfd_vals = [norm_sim_full.get(t, np.nan) for t in all_tables]
cfd_vals = [cfd_data.get(t, np.nan) for t in all_tables]

valid_cfd = [(s, c) for s, c in zip(sim_cfd_vals, cfd_vals) if not (np.isnan(s) or np.isnan(c))]
sim_cfd_valid, cfd_valid = zip(*valid_cfd)
mae_cfd = np.mean(np.abs(np.array(sim_cfd_valid) - np.array(cfd_valid)))
rho_cfd, p_cfd = stats.spearmanr(sim_cfd_valid, cfd_valid)
print(f"\n3D CFD vs simulation comparison")
print(f"Spearman rho = {rho_cfd:.3f}")
print(f"p-value      = {p_cfd:.4f}")
print(f"n            = {len(valid_cfd)}")
print(f"MAE (norm.)  = {mae_cfd:.4f}")

fig, ax = plt.subplots(figsize=(12, 4))
x = np.arange(len(all_tables))
bar_w = 0.35

ax.bar(x - bar_w / 2, sim_cfd_vals, bar_w, color=C_SIM,
       edgecolor='white', linewidth=0.5,
       label='This work (2D)', zorder=3)
ax.bar(x + bar_w / 2, cfd_vals, bar_w, color=C_CFD,
       edgecolor='#C06030', linewidth=0.5, hatch='///', alpha=0.70,
       label='3D CFD (Li et al., approx.)', zorder=3)
#ax.axhline(norm_threshold, linestyle='--', color=C_THRESHOLD, linewidth=1.5,
#           label='High-risk threshold (norm.)', zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(all_tables)
ax.set_xlabel('Table')
ax.set_ylabel('Average exposure (normalized)')
ax.set_ylim(0, 1.08)
ax.yaxis.grid(True, linestyle=':', alpha=0.3)
ax.set_axisbelow(True)
ax.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_exposure_vs_3dcfd.eps'))
plt.savefig(os.path.join(fig_dir, 'fig_exposure_vs_3dcfd.pdf'))
plt.close()


# plot 4: combined — (a) individual exposure, (b) tracer gas, (c) 3D CFD
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9),
                                     gridspec_kw={'height_ratios': [4, 3.5, 4]})

# (a) individual agent exposure per table
ax1.bar(all_x, all_heights, width=0.8, color=all_colors,
        edgecolor='white', linewidth=0.3, zorder=3)
ax1.axhline(INFECTION_THRESHOLD, linestyle='--', color=C_THRESHOLD, linewidth=1.5,
            label='High-risk threshold', zorder=4)
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels)
ax1.set_xlabel('Agents grouped by table')
ax1.set_ylabel('Exposure (inhaled particles)')
ax1.yaxis.grid(True, linestyle=':', alpha=0.3)
ax1.set_axisbelow(True)
ax1.legend(loc='upper right', framealpha=0.9)
ax1.text(-0.02, -0.1, '(a)', transform=ax1.transAxes, fontsize=13, fontweight='bold', va='top', ha='right')

# (b) tracer gas vs simulation
loc_sorted2 = sorted(locations, key=lambda x: tracer_gas[x], reverse=True)
tg_sorted2 = [tracer_gas[loc] for loc in loc_sorted2]
sim_sorted2 = [sim_for_tracer[loc] for loc in loc_sorted2]
inf_sorted2 = [infected_status[loc] for loc in loc_sorted2]
tg_norm2 = [v / max(tg_sorted2) for v in tg_sorted2]
sim_norm2 = [v / sim_sorted2[0] for v in sim_sorted2]

x1b = np.arange(len(loc_sorted2))
w1 = 0.35

ax2.bar(x1b - w1 / 2, tg_norm2, w1, color=C_TRACER, alpha=0.85, edgecolor='white', linewidth=0.5,
        label='Tracer gas (Li et al. 2021)', zorder=3)
ax2.bar(x1b + w1 / 2, sim_norm2, w1, color=C_SIM, alpha=0.85, edgecolor='white', linewidth=0.5,
        label='Our model: 2D airflow in Vadere', zorder=3)
for i, inf in enumerate(inf_sorted2):
    if inf:
        ax2.plot(x1b[i], max(tg_norm2[i], sim_norm2[i]) + 0.04, marker='v',
                 color=C_INFECTED, markersize=8, zorder=5, clip_on=False)
ax2.set_xticks(x1b)
ax2.set_xticklabels(loc_sorted2)
ax2.set_xlabel('Measurement location')
ax2.set_ylabel('Normalized concentration')
ax2.set_ylim(0, 1.08)
ax2.yaxis.grid(True, linestyle=':', alpha=0.3)
ax2.set_axisbelow(True)
legend_elements_b = [
    plt.Rectangle((0, 0), 1, 1, fc=C_TRACER, alpha=0.85, label='Tracer gas (Li et al. 2021)'),
    plt.Rectangle((0, 0), 1, 1, fc=C_SIM, alpha=0.85, label='Our model: 2D airflow in Vadere'),
    plt.Line2D([0], [0], marker='v', color=C_INFECTED, linestyle='None', markersize=8, label='Infected location'),
]
ax2.legend(handles=legend_elements_b, loc='upper right', framealpha=0.9)
ax2.text(-0.02, -0.1, '(b)', transform=ax2.transAxes, fontsize=13, fontweight='bold', va='top', ha='right')

# (c) 3D CFD comparison
x3 = np.arange(len(all_tables))
w2 = 0.35

ax3.bar(x3 - w2 / 2, cfd_vals, w2, color=C_CFD, edgecolor='#C06030', linewidth=0.5,
        hatch='///', alpha=0.70, label='3D CFD (Li et al. 2021)', zorder=3)
ax3.bar(x3 + w2 / 2, sim_cfd_vals, w2, color=C_SIM, alpha=0.85, edgecolor='white', linewidth=0.5,
        label='Our model: 2D airflow in Vadere', zorder=3)
for idx, value in enumerate(["A", "B", "C"]):
    ax3.plot(x3[idx], max(cfd_data[value], norm_sim_full[value]) + 0.04, marker='v',
             color=C_INFECTED, markersize=8, zorder=5, clip_on=False)
#ax3.axhline(norm_threshold, linestyle='--', color=C_THRESHOLD, linewidth=1.5,
#            label='High-risk threshold (normalized)', zorder=4)
ax3.set_xticks(x3)
ax3.set_xticklabels(all_tables)
ax3.set_xlabel('Table')
ax3.set_ylabel('Normalized exposure')
ax3.set_ylim(0, 1.08)
ax3.yaxis.grid(True, linestyle=':', alpha=0.3)
ax3.set_axisbelow(True)
legend_elements_c = [
    plt.Rectangle((0, 0), 1, 1, fc=C_CFD, edgecolor='#C06030', hatch='///', alpha=0.70, label='3D CFD (Li et al. 2021)'),
    plt.Rectangle((0, 0), 1, 1, fc=C_SIM, alpha=0.85, label='Our model: 2D airflow in Vadere'),
    plt.Line2D([0], [0], marker='v', color=C_INFECTED, linestyle='None', markersize=8, label='Infected location'),
    #plt.Line2D([0], [0], color=C_THRESHOLD, linestyle='--', label='High-risk threshold (normalized)'),
]
ax3.legend(handles=legend_elements_c, loc='upper right', framealpha=0.9)
ax3.text(-0.02, -0.1, '(c)', transform=ax3.transAxes, fontsize=13, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_exposure_comparison_combined.eps'))
plt.savefig(os.path.join(fig_dir, 'fig_exposure_comparison_combined.pdf'))
plt.close()


# save data
output_df = result_df[['pedestrianId', 'table', 'infected', 'degreeOfExposure-PID13']]
output_df.to_csv(os.path.join(output_dir, 'infections.txt'), sep='\t', index=False)
