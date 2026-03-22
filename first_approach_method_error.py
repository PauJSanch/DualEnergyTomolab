"""
First Approach Method Error Analysis

Computes method errors for the single-energy linear fitting approach by
comparing measured vs. known calibration rod densities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Getting columns' mean gray values and important data
rho_column = np.array([1.13, 1.16, 1.25, 1.65, 1.9])  # density of each column

# Prepare code to iterate
base_path = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/Results/Standar_Without_Threshold"
gvs_folder = "Mean_Gray_Values"

# Open already existing file as df
mean_density_file = os.path.join(base_path, "mean_densities_summary_group_complete_without_threshold.csv")
df_mean_density = pd.read_csv(mean_density_file)

# Define the samples to be analized
samples = df_mean_density["Sample"].tolist()

# Add new column for Method_Error
df_mean_density["Method_Error"] = np.nan

for sample in samples:
    # Getting rodes mean gvs
    rodes_gvs_csv_file_name = f"{sample}_gvs.csv"
    rodes_gvs_path = os.path.join(base_path, gvs_folder, rodes_gvs_csv_file_name)
    df_le = pd.read_csv(rodes_gvs_path)
    df_le['rho'] = rho_column.tolist()

    # Line fit using rodes 1, 3, and 5 (indices 0, 2, 4)
    fit_df = df_le.iloc[[0, 2, 4]]
    slope, intercept, r_value, p_value, std_err = linregress(fit_df["Mean"], fit_df["rho"])

    # Calculate rho for rodes 2 and 4 (indices 1 and 3)
    rho_calc_2 = slope * df_le.loc[1, "Mean"] + intercept
    rho_calc_4 = slope * df_le.loc[3, "Mean"] + intercept

    # True rho values from rho_column
    rho_true_2 = df_le.loc[1, "rho"]
    rho_true_4 = df_le.loc[3, "rho"]

    # Method error value
    method_error_value = np.mean([
        np.abs(rho_calc_2 - rho_true_2),
        np.abs(rho_calc_4 - rho_true_4)
    ])

    # Add Method_Error to df_mean_density for this sample
    df_mean_density.loc[df_mean_density["Sample"] == sample, "Method_Error"] = method_error_value

    # Clean variables for next iteration
    del df_le, fit_df, slope, intercept, r_value, p_value, std_err
    del rho_calc_2, rho_calc_4, rho_true_2, rho_true_4, method_error_value

# Save the final mean densities dataframe
mean_densities_csv_path = os.path.join(base_path, "mean_densities_summary_group_without_threshold.csv")
df_mean_density.to_csv(mean_densities_csv_path, index=False)

