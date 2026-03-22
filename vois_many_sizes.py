"""
Single-Energy VOI Batch Extraction

Extracts and measures multiple VOIs from single-energy (LE) volumes.
Batch processes multiple samples with VOI coordinates from CSV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import vois_many_sizes_functions as voisf
import os

# Prepare code to iterate among samples and (x, y) centers
base_path = "/mnt/e/BONE_HA/"
tomo_cera_vol = "tomo_cera/Vol"

samples = ["1_LE", "5_LE", "17_LE", "21_LE", "24_LE", "25_LE", "37_LE", "40_LE", "45_LE"]

x_y_centers_path = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/DualEnergyTomolab/samples_VOIs_centers.csv"
x_y_centers_df = pd.read_csv(x_y_centers_path) 

# Set subvolume sizw
subvolume_size = 40

# VOIs names
vois_names = [1, 2, 3, 4, 5, 6]

for sample in samples:
    # Prepare csv file for this sample
    csv_file_name = f"sample_{sample}_different_vois.csv"
    sample_csv_path = os.path.join(base_path, sample, csv_file_name)

    # Get sample meaningfull complete volume
    vol_path = os.path.join(base_path, sample, tomo_cera_vol)
    volume_stack = voisf.load_volume(vol_path)

    # Get (x, y) center for the VOIs in this sample
    sample_df = x_y_centers_df[x_y_centers_df['Sample'] == sample]
    x_centers = sample_df['Center_x'].tolist()
    y_centers = sample_df['Center_y'].tolist()

    #Iterate on all VOIs
    for x_center, y_center, voi in zip(x_centers, y_centers, vois_names):
        # Get this subvolume
        subvol = voisf.get_subvolume(original_vol=volume_stack, subvol_size=subvolume_size, starting_slice=200, center_y=y_center, center_x=x_center, selected=False)
    
        # Calculate volume, mean, and std. Save results on csv
        vol, mean, std, df = voisf.measure_subvolume(subvolume=subvol, pixel_size=20, save_csv=True, csv_path=sample_csv_path, subvolume_name=f"voi_{voi}")

        del vol, mean, std, df 

    # Delete old variables, to keep ram clean
    del vol_path, volume_stack, csv_file_name, sample_csv_path
