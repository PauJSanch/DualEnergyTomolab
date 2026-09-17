import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

USE_FULL_DATASET = False

if USE_FULL_DATASET:
    rods_csv = '/home/pausanch/Documents/DualEnergyTomolab_private/rodes_gvs_samples_full.csv'
    vois_csv = 'VOIs_samples_all_together_without_threshold_new_experiment.csv'
else:
    rods_csv = 'rodes_gvs_samples.csv'
    vois_csv = 'VOIs_samples_all_together_without_threshold.csv'

# ---------- Style ----------
plt.style.use("bmh")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

rods_df = pd.read_csv(rods_csv)
vois_df = pd.read_csv(vois_csv)

rod_density = np.array([1.13, 1.16, 1.25, 1.65, 1.9])

sample_17_rods = rods_df[rods_df['Sample'] == '17_LE']
mgv_rods = sample_17_rods['Mean'].values
std_rods = sample_17_rods['Std'].values

sample_17_vois = vois_df[vois_df['Sample'] == '17_LE']
mgv_vois = sample_17_vois['Mean'].values
calculated_rho_vois = sample_17_vois['calculated_rho'].values

slope, intercept, r_value, p_value, std_err = stats.linregress(rod_density, mgv_rods)

n = len(rod_density)
mean_x = np.mean(rod_density)
ss_x = np.sum((rod_density - mean_x) ** 2)
residuals = mgv_rods - (slope * rod_density + intercept)
mse = np.sum(residuals ** 2) / (n - 2)
rmse = np.sqrt(mse)

density_range = np.linspace(1.0, 2.0, 200)
t_critical = stats.t.ppf(0.975, n - 2)

predicted_mgv = slope * density_range + intercept
prediction_interval = t_critical * rmse * np.sqrt(1 + 1/n + (density_range - mean_x)**2 / ss_x)
upper_pi = predicted_mgv + prediction_interval
lower_pi = predicted_mgv - prediction_interval

voi_idx = 0
mgv_bone = mgv_vois[voi_idx]
rho_bone = calculated_rho_vois[voi_idx]

density_at_bone_mgv = (mgv_bone - intercept) / slope

pi_half_width_at_bone = t_critical * rmse * np.sqrt(1 + 1/n + (density_at_bone_mgv - mean_x)**2 / ss_x)

density_uncertainty = pi_half_width_at_bone / slope

from matplotlib.ticker import FormatStrFormatter

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
c1 = colors[0]
c2 = colors[1]
c3 = colors[2]

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121)
ax1.errorbar(rod_density, mgv_rods, yerr=std_rods, fmt='o', color=c1,
             capsize=5, markersize=8, label='Calibration rods', zorder=5)
ax1.plot(density_range, predicted_mgv, '-', color=c2, linewidth=2, label='Linear regression')
ax1.fill_between(density_range, lower_pi, upper_pi, alpha=0.2, color=c2,
                 label='95% prediction interval')
ax1.set_xlabel('Calibration rods tabulated density (g/cm³)')
ax1.set_ylabel('Mean gray value')
ax1.set_title('SE Method - Calibration Curve (Sample 17)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax2 = fig.add_subplot(122)

ax2.plot(density_range, predicted_mgv, '-', color=c2, linewidth=2, label='Linear regression')
ax2.fill_between(density_range, lower_pi, upper_pi, alpha=0.2, color=c2,
                 label='95% prediction interval')

ax2.axhline(y=mgv_bone, color=c3, linestyle='--', linewidth=1.5,
            label=f'Bone VOI mgv = {mgv_bone:.4f}')

ax2.axvline(x=density_at_bone_mgv, color=c3, linestyle='--', linewidth=1.5,
            label=f'Calculated ρ = {density_at_bone_mgv:.3f} g/cm³')

ax2.axvspan(density_at_bone_mgv - density_uncertainty,
            density_at_bone_mgv + density_uncertainty,
            alpha=0.2, color=c3, label=f'Uncertainty: ±{density_uncertainty:.3f} g/cm³')

ax2.plot(density_at_bone_mgv, mgv_bone, 'o', color=c3, markersize=10, zorder=5)

ax2.annotate('', xy=(density_at_bone_mgv, mgv_bone),
             xytext=(density_at_bone_mgv + 0.15, mgv_bone),
             arrowprops=dict(arrowstyle='->', color=c3, lw=2))
ax2.annotate('', xy=(density_at_bone_mgv, mgv_bone),
             xytext=(density_at_bone_mgv, mgv_bone + 0.015),
             arrowprops=dict(arrowstyle='->', color=c3, lw=2))

ax2.set_xlabel('Calibration rods tabulated density (g/cm³)')
ax2.set_ylabel('Mean gray value')
ax2.set_title('Inverse Prediction - Bone VOI Density')
ax2.legend(loc='lower left')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.savefig('sample_17_se_error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Sample 17 Calibration:")
print(f"  Slope: {slope:.6f}")
print(f"  Intercept: {intercept:.6f}")
print(f"  R²: {r_value**2:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"\nBone VOI (voi_1):")
print(f"  Mean gray value: {mgv_bone:.6f}")
print(f"  Calculated density: {density_at_bone_mgv:.6f} g/cm³")
print(f"  Density uncertainty (95% PI): ±{density_uncertainty:.6f} g/cm³")
