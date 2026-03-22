"""
Exact Differences Between Methods Report

Computes and reports absolute mean differences between dual-energy (DE) and
linear fitting methods (with and without threshold) at VOI-level granularity.
"""

import pandas as pd
import numpy as np

# Load data
file_path = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/DualEnergyTomolab/combined_results_after_comments.csv"
df = pd.read_csv(file_path)

# --- Helper: merge different methods on common identifiers ---
merge_cols = ['sample', 'sample_group', 'subvolume_name']

df_DE = df[df['calculation_method_rho'] == 'DE']
df_LF_wit = df[df['calculation_method_rho'] == 'LF_wit']
df_LF_wot = df[df['calculation_method_rho'] == 'LF_wot']

# Merge DE with LF_wit
merged_wit = pd.merge(
    df_DE, df_LF_wit,
    on=merge_cols,
    suffixes=('_DE', '_LF_wit')
)

# Merge DE with LF_wot
merged_wot = pd.merge(
    df_DE, df_LF_wot,
    on=merge_cols,
    suffixes=('_DE', '_LF_wot')
)

# ============================================================
# 1) GLOBAL MEAN ABS DIFFERENCES
# ============================================================

mean_diff_wit = np.mean(np.abs(merged_wit['calculated_rho_DE'] - merged_wit['calculated_rho_LF_wit']))
mean_diff_wot = np.mean(np.abs(merged_wot['calculated_rho_DE'] - merged_wot['calculated_rho_LF_wot']))

print("\n=== 1) GLOBAL MEAN ABS DIFFERENCES ===")
print(f"DE vs LF_wit: {mean_diff_wit:.6f}")
print(f"DE vs LF_wot: {mean_diff_wot:.6f}")

# ============================================================
# 2) GROUPED BY sample_group
# ============================================================

group_wit = merged_wit.groupby('sample_group').apply(
    lambda x: np.mean(np.abs(x['calculated_rho_DE'] - x['calculated_rho_LF_wit']))
)

group_wot = merged_wot.groupby('sample_group').apply(
    lambda x: np.mean(np.abs(x['calculated_rho_DE'] - x['calculated_rho_LF_wot']))
)

print("\n=== 2) MEAN ABS DIFFERENCES BY sample_group ===")
print("\nDE vs LF_wit:")
print(group_wit)

print(f"Average across groups (DE vs LF_wit): {group_wit.mean():.6f}")

print("\nDE vs LF_wot:")
print(group_wot)

print(f"Average across groups (DE vs LF_wot): {group_wot.mean():.6f}")

# ============================================================
# 3) SAMPLE-SPECIFIC (sample = 4 and 5)
# ============================================================

for sample_id in [4, 5]:
    subset_wit = merged_wit[merged_wit['sample'] == sample_id]
    subset_wot = merged_wot[merged_wot['sample'] == sample_id]

    diff_wit = np.abs(subset_wit['calculated_rho_DE'] - subset_wit['calculated_rho_LF_wit'])
    diff_wot = np.abs(subset_wot['calculated_rho_DE'] - subset_wot['calculated_rho_LF_wot'])

    print(f"\n=== 3) SAMPLE {sample_id} ===")
    print("DE vs LF_wit differences:")
    print(diff_wit.values)
    print(f"Mean: {diff_wit.mean():.6f}")

    print("DE vs LF_wot differences:")
    print(diff_wot.values)
    print(f"Mean: {diff_wot.mean():.6f}")

# ============================================================
# 4) UNCERTAINTY DIFFERENCES (sample = 19)
# ============================================================

subset_wit_19 = merged_wit[merged_wit['sample'] == 19]
subset_wot_19 = merged_wot[merged_wot['sample'] == 19]

unc_diff_wit_19 = np.abs(subset_wit_19['uncertainty_rho_DE'] - subset_wit_19['uncertainty_rho_LF_wit'])
unc_diff_wot_19 = np.abs(subset_wot_19['uncertainty_rho_DE'] - subset_wot_19['uncertainty_rho_LF_wot'])

print("\n=== 4) UNCERTAINTY DIFFERENCES (sample = 19) ===")
print("DE vs LF_wit:")
print(unc_diff_wit_19.values)
print(f"Mean: {unc_diff_wit_19.mean():.6f}")

print("DE vs LF_wot:")
print(unc_diff_wot_19.values)
print(f"Mean: {unc_diff_wot_19.mean():.6f}")

# ============================================================
# 5) UNCERTAINTY VALUES FOR LF METHODS (sample = 19)
# ============================================================

unc_LF_wit_19 = df[
    (df['calculation_method_rho'] == 'LF_wit') &
    (df['sample'] == 19)
]['uncertainty_rho']

unc_LF_wot_19 = df[
    (df['calculation_method_rho'] == 'LF_wot') &
    (df['sample'] == 19)
]['uncertainty_rho']

print("\n=== 5) UNCERTAINTY VALUES (sample = 19) ===")

print("\nLF_wit uncertainty_rho values:")
print(unc_LF_wit_19.values)
print(f"Mean: {unc_LF_wit_19.mean():.6f}")
print(f"Std:  {unc_LF_wit_19.std():.6f}")

print("\nLF_wot uncertainty_rho values:")
print(unc_LF_wot_19.values)
print(f"Mean: {unc_LF_wot_19.mean():.6f}")
print(f"Std:  {unc_LF_wot_19.std():.6f}")
