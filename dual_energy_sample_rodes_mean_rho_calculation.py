"""
Dual-Energy Calibration Rod Analysis

Computes mean densities and method errors for calibration rods using the
dual-energy approach. Validates against known phantom densities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Getting columns' mean gray values and important data
rho_column = np.array([1.13, 1.16, 1.25, 1.65, 1.9])  # density of each column

# Prepare code to iterate
base_path = "/mnt/e/BONE_HA"

samples = [1, 4, 5, 8, 17, 18, 19, 21, 23, 24, 25, 26, 37, 38, 40, 42, 45, 46]

# Create empty dataframes
df_mean_densities_dual_energy = pd.DataFrame(columns=["Sample", "Mean_Density", "StD_Density", "Method_Error", "Group"])
df_vois_all = pd.DataFrame(columns=["Sample", "subvolume_name", "Mean", "Std", "calculated_rho", "Method_Error", "Group"])

for sample in samples:
    # Defining important paths
    sample_VOIs_rho = f'{base_path}/{sample}_rho/sample_{sample}_rho_different_vois_dual_energy.csv'
    sample_rodes_rho = f'{base_path}/{sample}_rho/sample_{sample}_rho_rode_dual_energy.csv'

    # Calculating Method error
    rodes_df = pd.read_csv(sample_rodes_rho)
    rode_densities = rodes_df["Mean"].to_numpy(dtype=float)
    method_error = np.mean(np.abs(rode_densities - rho_column))

    # Read VOI file
    vois_df = pd.read_csv(sample_VOIs_rho)

    # Calculate mean and std for sample
    voi_densities = vois_df["Mean"].to_numpy(dtype=float)
    mean_density = np.mean(voi_densities)
    std_density = np.std(voi_densities)

    # Assign group based on sample number
    if 1 <= sample <= 14:
        group = "OBC"
    elif 15 <= sample <= 22:
        group = "MG"
    elif 23 <= sample <= 26:
        group = "FRESH"
    elif 36 <= sample <= 44:
        group = "LUB"
    else:
        group = "PV"

    # Store mean summary
    df_mean_densities_dual_energy = pd.concat([
        df_mean_densities_dual_energy,
        pd.DataFrame({
            "Sample": [sample],
            "Mean_Density": [mean_density],
            "StD_Density": [std_density],
            "Method_Error": [method_error],
            "Group": [group]
        })
    ], ignore_index=True)

    # --- NEW SECTION: Collect VOIs into one big CSV ---
    # Add subvolume name if not present
    if "subvolume_name" not in vois_df.columns:
        vois_df = vois_df.reset_index().rename(columns={"index": "subvolume_name"})
        vois_df["subvolume_name"] = "voi_" + (vois_df["subvolume_name"] + 1).astype(str)

    # Add required columns
    vois_df["Sample"] = sample
    vois_df["calculated_rho"] = mean_density
    vois_df["Method_Error"] = method_error
    vois_df["Group"] = group

    # Keep only the required columns and append
    df_vois_all = pd.concat([
        df_vois_all,
        vois_df[["Sample", "subvolume_name", "Mean", "Std", "calculated_rho", "Method_Error", "Group"]]
    ], ignore_index=True)

    # Clean memory
    del rodes_df, rode_densities, method_error
    del vois_df, voi_densities, mean_density, std_density, group

# --- SAVE FILES ---
mean_densities_csv_path = os.path.join(base_path, "mean_densities_summary_group_dual_energy.csv")
df_mean_densities_dual_energy.to_csv(mean_densities_csv_path, index=False)

vois_all_csv_path = os.path.join(base_path, "VOIs_samples_all_together_dual_energy.csv")
df_vois_all.to_csv(vois_all_csv_path, index=False)

print(f"Saved mean summary to: {mean_densities_csv_path}")
print(f"Saved all VOIs to: {vois_all_csv_path}")

