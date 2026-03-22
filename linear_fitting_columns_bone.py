"""
Bone-Specific Linear Calibration

Linear regression analysis for bone samples against calibration phantom.
Establishes density calibration curves with bone reference values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Definition of important numbers
qs = np.array([0.0, 0.0428, 0.1586, 0.488, 0.6349]) # proportion of HA in each column

rho_ep = 1.128  # epoxy resin density
Z_ep = 9.1  # atomic number equivalent epoxy resin

rho_ha = 3.148  # HA density
Z_ha = 16.26  # atomic number equivalent HA

rho_column = np.array([1.13, 1.16, 1.25, 1.65, 1.9])  # density of each column

# Getting gray values from FIJI measurements on three slices
file_le = "/home/pausanch/Elettra/DualEnergyTomolab/data/40_LE/40_LE_gvs_300_630_850.csv"
df_le = pd.read_csv(file_le)

# Adding the rho_column values to df_le
def map_rho(col):
    return rho_column[col - 1]

df_le["rho_column"] = df_le["Column"].apply(map_rho)

# Bone mean values
bone_means = {300: 0.097, 630: 0.095, 850: 0.093}  # Updated values for bone_mean

# Plotting
slices = df_le["Slice"].unique()

for slice_val in slices:
    df_slice = df_le[df_le["Slice"] == slice_val]

    # Extract relevant columns
    df_main = df_slice[df_slice["Column"].isin([1, 3, 5])]
    df_secondary = df_slice[df_slice["Column"].isin([2, 4])]

    # Linear fit for Columns 1, 3, and 5
    slope, intercept, _, _, _ = linregress(df_main["rho_column"], df_main["Mean"])
    fit_line = slope * df_main["rho_column"] + intercept

    plt.figure(figsize=(8, 6))

    # Plot data points and line fit
    plt.scatter(df_main["rho_column"], df_main["Mean"], color="black", label="Columns 1, 3, 5")
    plt.plot(df_main["rho_column"], fit_line, color="#1a9988", label=f"Linear Fit (Slope: {slope:.4f}, Intercept: {intercept:.4f})")
    plt.scatter(df_secondary["rho_column"], df_secondary["Mean"], color="#eb5600", s=60, label="Columns 2, 4")

    # Compute density for bone_mean and plot
    if slice_val in bone_means:
        bone_mean = bone_means[slice_val]
        bone_density = (bone_mean - intercept) / slope
        plt.scatter(bone_density, bone_mean, color='red', s=100, label=f'Bone Mean (Density: {bone_density:.2f})')
        plt.text(bone_density, bone_mean, f'{bone_density:.2f}', fontsize=12, verticalalignment='bottom', horizontalalignment='right', color='red')

    plt.xlabel("Density")
    plt.ylabel("Mean Value")
    plt.title(f"Slice {slice_val} - Mean vs Density")
    plt.legend()
    plt.grid()

plt.show()

