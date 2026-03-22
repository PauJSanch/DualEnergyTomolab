"""
Gray Value to Attenuation Coefficient Calibration

Establishes linear relationships between gray values and linear attenuation
coefficients for both LE and HE energy spectra using phantom measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Definition of importan numbers
qs = np.array([0.0, 0.0428, 0.1586, 0.488, 0.6349]) #proportion of HA on each column

rho_ep = 1.128 #epoxy resin density
Z_ep = 9.1 #atomic number equivalent epoxy resin

rho_ha = 3.148 #Ha density
Z_ha = 16.26 #atomic number equivalent HA

rho_column = np.array([1.13, 1.16, 1.25, 1.65, 1.9]) #desity of each column

mu_ep_LE = 0.2691 #absorption of epoxy resin at the mean low energy 45 keV
mu_ep_HE = 0.1904 #absorption of epoxy resin at the mean high energy 72 keV

mu_ha_LE = 0.7456 #absorption of HA at the mean low energy 45 keV
mu_ha_HE = 0.3008 #absorption of HA at the mean high energy 72 keV

# Getting absorption coeficient for EP and HA in our images
mu_ep = rho_ep * np.array([mu_ep_LE, mu_ep_HE])
mu_ha = rho_ha * np.array([mu_ha_LE, mu_ha_HE])

# Getting absorption coeficient for each column, at each energy
mu_s = np.array([[mu_ep[i] * (1 - q) + mu_ha[i] * q for q in qs] for i in range(len(mu_ep))])

# Getting gray values from FIJI measurements on three slices, at the two energies
file_he = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/data/40_HE/40_HE_gvs_300_630_850.csv"
file_le = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/data/40_LE/40_LE_gvs_300_630_850.csv"

df_he = pd.read_csv(file_he)
df_le = pd.read_csv(file_le)

df_merged = pd.merge(df_le[['Slice', 'Column', 'Mean']], df_he[['Slice', 'Column', 'Mean']], on=['Slice', 'Column'], suffixes=('_LE', '_HE'))

# Integrating column's absorption indexes to the dataframe
column_to_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

mu_s_le_values = []
mu_s_he_values = []

for index, row in df_merged.iterrows():
    column = row['Column']

    mu_s_le = mu_s[0, column_to_index[column]]  # LE absorption coefficient
    mu_s_he = mu_s[1, column_to_index[column]]  # HE absorption coefficient

    mu_s_le_values.append(mu_s_le)
    mu_s_he_values.append(mu_s_he)

df_merged['Mu_s_LE'] = mu_s_le_values
df_merged['Mu_s_HE'] = mu_s_he_values

# Line fit and plots
# Define colors for LE and HE
colors = {'LE': '#1a9988', 'HE': '#eb5600'}

# Function to plot data and linear fit for each slice
def plot_slice(df, slice_number):
    df_slice = df[df['Slice'] == slice_number]

    # Extract values for LE and HE
    x_le = df_slice['Mean_LE']
    y_le = df_slice['Mu_s_LE']
    x_he = df_slice['Mean_HE']
    y_he = df_slice['Mu_s_HE']

    # Perform linear regression for LE
    slope_le, intercept_le, _, _, _ = linregress(x_le, y_le)
    line_le = slope_le * x_le + intercept_le

    # Perform linear regression for HE
    slope_he, intercept_he, _, _, _ = linregress(x_he, y_he)
    line_he = slope_he * x_he + intercept_he

    # Plot data points
    plt.figure(figsize=(7, 5))
    plt.scatter(x_le, y_le, color=colors['LE'], label='LE Data')
    plt.scatter(x_he, y_he, color=colors['HE'], label='HE Data')

    # Plot regression lines
    plt.plot(x_le, line_le, color=colors['LE'], linestyle='--', label=f'LE Fit: y = {slope_le:.4f}x + {intercept_le:.4f}')
    plt.plot(x_he, line_he, color=colors['HE'], linestyle='--', label=f'HE Fit: y = {slope_he:.4f}x + {intercept_he:.4f}')

    # Labels and title
    plt.xlabel("Mean Gray Value")
    plt.ylabel("Mu_s")
    plt.title(f"Slice {slice_number}: Absorption vs. Gray Value")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

    # Print the slope and intercept
    print(f"Slice {slice_number}:")
    print(f"  LE Fit: y = {slope_le:.4f}x + {intercept_le:.4f}")
    print(f"  HE Fit: y = {slope_he:.4f}x + {intercept_he:.4f}")
    print("-" * 40)

# Generate plots for each slice
for slice_num in [300, 630, 850]:
    plot_slice(df_merged, slice_num)
