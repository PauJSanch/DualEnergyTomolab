"""
Attenuation to Density Transformation (Single Slice)

Demonstrates the full dual-energy transformation pipeline on a single slice:
attenuation coefficients → basis decomposition → rotation → density.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile

# Definition of importan parameters
mu_ha_LE = 2.24
mu_ha_HE = 0.9
mu_ep_LE = 0.3
mu_ep_HE = 0.21

D = (mu_ep_HE * mu_ha_LE) - (mu_ep_LE * mu_ha_HE)

slope_HE = 15.7
interc_HE = -0.07

slope_LE = 17.4
interc_LE = -0.05

slope_density = 1.84583
interc_density = 0.455649

rho_ep = 1.128
rho_ha = 3.148

theta = np.arctan(rho_ha / rho_ep)

# I will work with slice 630
HE40_path = '/home/pausanch/Elettra/DualEnergyTomolab/data/40_HE/40_HE_630.tif'
LE40_path = '/home/pausanch/Elettra/DualEnergyTomolab/data/40_LE/40_LE_630.tif'

HE40 = cv2.imread(HE40_path, cv2.IMREAD_UNCHANGED)
LE40 = cv2.imread(LE40_path, cv2.IMREAD_UNCHANGED)

ORIGINAL_DTYPE = HE40.dtype

# Convert gvs to mus
mu_HE = (slope_HE * HE40) + interc_HE
mu_LE = (slope_LE * LE40) + interc_LE

assert ORIGINAL_DTYPE == mu_LE.dtype

# Convert mus to x_ep and x_ha
x_ep = ((mu_HE * mu_ha_LE) - (mu_LE * mu_ha_HE)) / D
x_ha = ((mu_LE * mu_ep_HE) - (mu_HE * mu_ep_LE)) / D

assert ORIGINAL_DTYPE == x_ha.dtype

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(x_ep, cmap='gray')
plt.title("x_ep")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(x_ha, cmap='gray')
plt.title("x_ha")
plt.axis("off")

# Save .tif files
x_ep_path = "/home/pausanch/Elettra/DualEnergyTomolab/data/40_ep_630.tif"
x_ha_path = "/home/pausanch/Elettra/DualEnergyTomolab/data/40_ha_630.tif"

tifffile.imwrite(x_ep_path, x_ep)
tifffile.imwrite(x_ha_path, x_ha)

#Rotating from (ep, ha) space to (x_rho, x_z) space
x_rho = (x_ep * np.cos(theta)) + (x_ha * np.sin(theta))
x_rho = x_rho.astype(np.float32)

x_z = - (x_ep * np.sin(theta)) + (x_ha * np.cos(theta))
x_z = x_z.astype(np.float32)

assert ORIGINAL_DTYPE == x_rho.dtype

x_rho_path = "/home/pausanch/Elettra/DualEnergyTomolab/data/x_rho_630.tif"
x_z_path = "/home/pausanch/Elettra/DualEnergyTomolab/data/x_z_630.tif"

tifffile.imwrite(x_rho_path, x_rho)
tifffile.imwrite(x_z_path, x_z)

# Convert x_rho to actual density, using the lineal transformation
density = (slope_density * x_rho) + interc_density

plt.figure(figsize=(10, 5))

plt.imshow(density, cmap='gray')
plt.title("Density Map")
plt.axis("off")

# Save .tif file
density_path = "/home/pausanch/Elettra/DualEnergyTomolab/data/density_630.tif"
tifffile.imwrite(density_path, density)

# Show all figures
plt.tight_layout()
plt.show()
