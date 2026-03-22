"""
X-Rho to Density Calibration

Calibrates the rotated coordinate x_rho to physical density using phantom
columns with known densities. Establishes linear transformation coefficients.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Definition of important numbers
qs = np.array([0.0, 0.0428, 0.1586, 0.488, 0.6349])  # proportion of HA in each column

rho_ep = 1.128  # epoxy resin density
Z_ep = 9.1  # atomic number equivalent epoxy resin

rho_ha = 3.148  # HA density
Z_ha = 16.26  # atomic number equivalent HA

rho_column = np.array([1.13, 1.16, 1.25, 1.65, 1.9])  # density of each column

theta = np.arctan(rho_ha / rho_ep)
M = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) # rotation matrix

# Load gray values from CSV files
file_xrho = '/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/data/40_xrho_630.csv'
file_xz = '/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/data/40_xz_630.csv'

df_xrho = pd.read_csv(file_xrho)
df_xz = pd.read_csv(file_xz)

gvs_xrho = np.array(df_xrho['Mean'])
gvs_xz = np.array(df_xz['Mean'])

# Perform linear regression
slope_density, intercept_density, r_value, p_value, std_err = linregress(rho_column, gvs_xrho)

# Generate fitted line
x_fit = np.linspace(min(rho_column), max(rho_column), 100)
y_fit = slope_density * x_fit + intercept_density

# Plot data points and fitted line
plt.figure(figsize=(8, 6))
plt.scatter(rho_column, gvs_xrho, color='black', label='Data points')
plt.plot(x_fit, y_fit, color='#1a9988', label=f'Fit: y = {slope_density:.4f}x + {intercept_density:.4f}')
plt.xlabel('Density of Column')
plt.ylabel('Gray Values Xrho')
plt.legend()
plt.grid(True)
plt.show()

# Print results
print(f'Slope (density fit): {slope_density:.4f}')
print(f'Intercept (density fit): {intercept_density:.4f}')

