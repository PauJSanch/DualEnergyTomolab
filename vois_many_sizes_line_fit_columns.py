"""
Linear Fitting for Calibration Columns (Single-Energy)

Performs linear regression between gray values and known phantom densities
for the single-energy (LE) approach. Used to convert GV to density.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Getting columns' mean gray values and important data
rho_column = np.array([1.13, 1.16, 1.25, 1.65, 1.9])  # density of each column

# Prepare code to iterate
base_path = "/mnt/e/BONE_HA/"
tomo_cera_vol = "tomo_cera/Vol"

samples = ["1_LE", "5_LE", "17_LE", "21_LE", "24_LE", "25_LE", "37_LE", "40_LE", "45_LE"]

# Create an empty dataframe to store mean densities
df_mean_densities = pd.DataFrame(columns=["Sample", "Mean_Density", "StD_Density"])

for sample in samples:
    # Getting rodes mean gvs
    rodes_gvs_csv_file_name = f"{sample}_gvs.csv"
    rodes_gvs_path = os.path.join(base_path, sample, rodes_gvs_csv_file_name)
    df_le = pd.read_csv(rodes_gvs_path)
    df_le['rho'] = rho_column.tolist()

    # Getting VOIs mean gray values
    csv_file_name = f"sample_{sample}_different_vois.csv"
    file_vois = os.path.join(base_path, sample, csv_file_name)
    df_vois = pd.read_csv(file_vois)
    
    # Line fit columns mean values
    slope, intercept, r_value, p_value, std_err = linregress(df_le["Mean"], df_le["rho"])
    
    # Calculate rho for df_vois using the fitted equation
    df_vois["calculated_rho"] = df_vois["Mean"].apply(lambda x: intercept + slope * x)

    csv_rho = f"{sample}_vois_calculated_density.csv"
    csv_rho_path = os.path.join(base_path, sample, csv_rho)
    df_vois.to_csv(csv_rho_path, index=False)

    # Calculate and append mean and std of calculated rho for the sample
    mean_rho = df_vois["calculated_rho"].mean()
    std_rho = df_vois["calculated_rho"].std()
    df_mean_densities.loc[len(df_mean_densities)] = {
    "Sample": sample,
    "Mean_Density": mean_rho,
    "StD_Density": std_rho
    }

    # Clean memory
    del rodes_gvs_csv_file_name, rodes_gvs_path, df_le, csv_file_name, file_vois, df_vois, slope, intercept, r_value, p_value, std_err, csv_rho, csv_rho_path, mean_rho, std_rho

# Save the final mean densities dataframe
mean_densities_csv_path = os.path.join(base_path, "mean_densities_summary.csv")
df_mean_densities.to_csv(mean_densities_csv_path, index=False)

