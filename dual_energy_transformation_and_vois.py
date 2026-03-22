"""
Dual-Energy Density Map Generation and VOI Analysis

Batch processing pipeline that converts LE and HE volumes into density maps
for multiple bone samples, then extracts and measures VOIs.
"""

import vois_many_sizes_functions as voisf
import dual_energy_transformation_functions as detf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

# Loop through all samples to get density maps
samples = [1, 4, 5, 8, 17, 18, 19, 21, 23, 24, 25, 26, 37, 38, 40, 42, 45, 46]

base_directory = '/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/data'

for sample in samples:
    # Get LE and HE volumes
    LE_path = f'{base_directory}/{sample}_LE/tomo_cera/Vol'
    HE_path = f'{base_directory}/{sample}_HE/tomo_cera/Vol'
    
    LE_vol = voisf.load_volume(LE_path)
    HE_vol = voisf.load_volume(HE_path)
    
    # Check if the folder to save the created file exists, if not, create it
    density_map_folder = f"{base_directory}/{sample}_rho"
    os.makedirs(density_map_folder, exist_ok=True)

    # Get density map
    density_map_file = f"sample_{sample}_density_map"
    density_map_path = f"{density_map_folder}/{density_map_file}"
    density_map = detf.raw_to_density_map(LE_vol, HE_vol, save_path=density_map_path)

    # Print progress message
    print(density_map_path)
    print()

    # Cleaning memory
    del LE_path, HE_path, LE_vol, HE_vol, density_map_folder, density_map_file, density_map_path, density_map
