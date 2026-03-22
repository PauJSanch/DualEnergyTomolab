"""
VOI Visualization Tool

Interactive visualization of VOI locations overlaid on CT slices.
Highlights selected regions on tomographic data for validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import vois_many_sizes_functions as voisf
import os

# ---------- Style ----------
plt.style.use("bmh")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# Parameters
base_path = "/mnt/f/BONE_HA/"
tomo_cera_vol = "tomo_cera/Vol"
samples = [4, 5]
x_y_centers_path = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/DualEnergyTomolab/samples_VOIs_centers.csv"
output_dir = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/figures"
os.makedirs(output_dir, exist_ok=True)

x_y_centers_df = pd.read_csv(x_y_centers_path)

subvolume_size = 40
half_size = subvolume_size // 2

# Visualize and save figure per sample for LE and Dual Energy
for sample in samples:

    # --- LE data processing ---
    sample_LE = f"{sample}_LE"
    vol_path_LE = os.path.join(base_path, sample_LE, tomo_cera_vol)
    volume_stack_LE = voisf.load_volume(vol_path_LE, start_slice=500, end_slice=501)
    image_LE = volume_stack_LE[0]

    sample_df = x_y_centers_df[x_y_centers_df['Sample'] == sample]
    x_centers = sample_df['Center_x'].tolist()
    y_centers = sample_df['Center_y'].tolist()

    fig, ax = plt.subplots()
    ax.imshow(image_LE, cmap='gray')

    for x, y in zip(x_centers, y_centers):
        rect = patches.Rectangle(
            (x - half_size, y - half_size),
            subvolume_size,
            subvolume_size,
            linewidth=1.5,
            edgecolor='#D55E00',
            facecolor='none'
        )
        ax.add_patch(rect)

    ax.set_title("LE VOIs (40x40px)")

    # Remove ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    fig_filename = f"VOIs_sample_{sample}_LE_wot.png"
    fig_path = os.path.join(output_dir, fig_filename)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- Dual Energy data processing ---
    sample_DE = f"{sample}_rho"
    vol_path_DE = os.path.join(base_path, sample_DE, f"sample_{sample}_density_map")
    volume_stack_DE = voisf.load_volume(
        vol_path_DE,
        dims=(450, 2048, 2048),
        start_slice=150,
        end_slice=151
    )
    image_DE = volume_stack_DE[0]

    fig, ax = plt.subplots()
    ax.imshow(image_DE, cmap='gray')

    for x, y in zip(x_centers, y_centers):
        rect = patches.Rectangle(
            (x - half_size, y - half_size),
            subvolume_size,
            subvolume_size,
            linewidth=1.5,
            edgecolor='#D55E00',
            facecolor='none'
        )
        ax.add_patch(rect)

    ax.set_title("Dual Energy VOIs (40x40px)")

    # Remove ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    fig_filename = f"VOIs_sample_{sample}_DE.png"
    fig_path = os.path.join(output_dir, fig_filename)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
