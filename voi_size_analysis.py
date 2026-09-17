# -*- coding: utf-8 -*-
"""
VOI size analysis for LE volumes.

Extracts cubic VOIs of different sizes (10, 20, 40, 60, 80 pixels) centered at
the same VOI centers, computes mean gray value and std inside each cube, and
produces three figures:
  1. Mean gray value (± std) per VOI size for selected samples.
  2. Overlay of the chosen 40×40 size and the largest 80×80 size on slice 500.
  3. Pixel intensity distributions inside each VOI for every VOI size and sample.

Inputs:
  - VOI centers: samples_VOIs_centers.csv
  - LE volumes: /mnt/tomolab_data/BONE_HA_0/{sample}_LE/tomo_cera/Vol

Outputs saved to: Figures_after_review_SR
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import vois_many_sizes_functions as voisf

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
BASE_PATH = "/mnt/tomolab_data/BONE_HA_0"
CENTERS_CSV = "/home/pausanch/Documents/DualEnergyTomolab_private/samples_VOIs_centers.csv"
OUTPUT_DIR = "/home/pausanch/Documents/DualEnergyTomolab_private/Figures_after_review_SR"

SAMPLES = [4, 21, 42]
VOI_SIZES = [10, 20, 40, 60, 80]
CENTER_Z = 220          # z-center of the cube (matches 40×40 extraction starting at slice 200)
SLICE_FOR_OVERLAY = 500 # slice used for the visual overlay
OVERLAY_SIZES = [40, 80]  # chosen size and a larger size for comparison

# bmh stylesheet colors
BMH_COLORS = [
    "#348ABD",
    "#A60628",
    "#7A68A6",
    "#467821",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Style (matches comparative_analysis_three_methods.py) ----------
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

# ------------------------------------------------------------
# LOAD VOI CENTERS
# ------------------------------------------------------------
centers_df = pd.read_csv(CENTERS_CSV)

# ------------------------------------------------------------
# EXTRACT MEASUREMENTS FOR ALL SAMPLES, VOIs AND SIZES
# ------------------------------------------------------------
records = []

for sample in SAMPLES:
    print(f"Processing sample {sample} ...")

    vol_path = os.path.join(BASE_PATH, f"{sample}_LE", "tomo_cera", "Vol")

    # Memory-map the full LE volume (1200, 2048, 2048) float32
    volume = np.memmap(vol_path, dtype=np.float32, mode="r", shape=(1200, 2048, 2048))

    sample_centers = centers_df[centers_df["Sample"] == sample].copy()

    for _, row in sample_centers.iterrows():
        voi = int(row["VOI"])
        cx = int(row["Center_x"])
        cy = int(row["Center_y"])

        for size in VOI_SIZES:
            start_slice = CENTER_Z - size // 2

            subvol = voisf.get_subvolume(
                original_vol=volume,
                subvol_size=size,
                starting_slice=start_slice,
                center_y=cy,
                center_x=cx,
                selected=False,
            )

            _, mean_val, std_val, _ = voisf.measure_subvolume(
                subvolume=subvol,
                pixel_size=20,
                save_csv=False,
            )

            records.append({
                "Sample": sample,
                "VOI": voi,
                "Size": size,
                "Mean": mean_val,
                "Std": std_val,
            })

    del volume

results_df = pd.DataFrame(records)

# Save measurements to CSV
results_csv = os.path.join(OUTPUT_DIR, "voi_size_analysis_measurements.csv")
results_df.to_csv(results_csv, index=False)
print(f"Measurements saved to: {results_csv}")

# ------------------------------------------------------------
# FIGURE 1: Mean gray value (± std) vs VOI size, per sample
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

colors = BMH_COLORS[:6]
sample_aliases = {4: "MBC_0", 21: "MGA_3", 42: "MGB_1"}

for ax, sample in zip(axes, SAMPLES):
    sample_df = results_df[results_df["Sample"] == sample].sort_values(["VOI", "Size"])

    for i, voi in enumerate(sorted(sample_df["VOI"].unique())):
        voi_df = sample_df[sample_df["VOI"] == voi]
        ax.errorbar(
            voi_df["Size"],
            voi_df["Mean"],
            yerr=voi_df["Std"],
            fmt="o-",
            capsize=4,
            color=colors[i],
            label=f"VOI {voi}",
        )

    ax.set_xlabel("VOI side length (pixels)")
    ax.set_ylabel("Mean gray value")
    ax.set_title(f"{sample_aliases.get(sample, '')}")
    ax.legend(title="VOI", loc="best", fontsize=14)
    ax.grid(True)
    ax.set_xticks(VOI_SIZES)

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "voi_size_mean_gray_value_per_sample.png")
plt.savefig(fig1_path, dpi=200)
plt.close(fig)
print(f"Figure 1 saved to: {fig1_path}")

# ------------------------------------------------------------
# FIGURE 2: Visual overlay of 40×40 and 80×80 VOIs on slice 500
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(24, 10))

overlay_colors = {40: BMH_COLORS[0], 80: BMH_COLORS[4]}  # blue = chosen, orange = larger
overlay_labels = {40: "40×40 (chosen)", 80: "80×80"}

for ax, sample in zip(axes, SAMPLES):
    vol_path = os.path.join(BASE_PATH, f"{sample}_LE", "tomo_cera", "Vol")
    volume = np.memmap(vol_path, dtype=np.float32, mode="r", shape=(1200, 2048, 2048))
    image = volume[SLICE_FOR_OVERLAY, :, :]

    ax.imshow(image, cmap="gray")

    sample_centers = centers_df[centers_df["Sample"] == sample]

    for _, row in sample_centers.iterrows():
        cx = int(row["Center_x"])
        cy = int(row["Center_y"])

        for size in OVERLAY_SIZES:
            half = size // 2
            rect = patches.Rectangle(
                (cx - half, cy - half),
                size,
                size,
                linewidth=1.5,
                edgecolor=overlay_colors[size],
                facecolor="none",
                label=overlay_labels[size] if _ == 0 else "",
            )
            ax.add_patch(rect)

    ax.set_title(f"{sample_aliases.get(sample, '')}\nslice {SLICE_FOR_OVERLAY}")
    ax.set_xticks([])
    ax.set_yticks([])

    # Build legend manually to avoid duplicate labels
    handles = [
        patches.Patch(edgecolor=overlay_colors[s], facecolor="none", linewidth=2, label=overlay_labels[s])
        for s in OVERLAY_SIZES
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=14)

    del volume

plt.tight_layout(pad=3.0)
fig2_path = os.path.join(OUTPUT_DIR, "voi_size_overlay_40_vs_80.png")
plt.savefig(fig2_path, dpi=200)
plt.close(fig)
print(f"Figure 2 saved to: {fig2_path}")

# ------------------------------------------------------------
# FIGURE 3: Pixel intensity distributions per VOI and VOI size
# ------------------------------------------------------------
print("\nGenerating Figure 3 (pixel intensity distributions) ...")

HIST_BINS = 50

fig, axes = plt.subplots(3, 5, figsize=(30, 16))

for i, sample in enumerate(SAMPLES):
    print(f"  Figure 3 - sample {sample} ...")
    vol_path = os.path.join(BASE_PATH, f"{sample}_LE", "tomo_cera", "Vol")
    volume = np.memmap(vol_path, dtype=np.float32, mode="r", shape=(1200, 2048, 2048))

    sample_centers = centers_df[centers_df["Sample"] == sample].sort_values("VOI")

    for j, size in enumerate(VOI_SIZES):
        ax = axes[i, j]
        start_slice = CENTER_Z - size // 2

        for _, row in sample_centers.iterrows():
            voi = int(row["VOI"])
            cx = int(row["Center_x"])
            cy = int(row["Center_y"])

            subvol = voisf.get_subvolume(
                original_vol=volume,
                subvol_size=size,
                starting_slice=start_slice,
                center_y=cy,
                center_x=cx,
                selected=False,
            )
            pixels = subvol.flatten()

            ax.hist(
                pixels,
                bins=HIST_BINS,
                color=BMH_COLORS[voi - 1],
                alpha=0.4,
                histtype="stepfilled",
                label=f"VOI {voi}",
                density=True,
            )

        ax.set_title(f"{sample_aliases.get(sample, '')}\n{size}×{size}×{size}")
        ax.set_xlabel("Pixel intensity")
        if j == 0:
            ax.set_ylabel("Density")
        ax.grid(True)

    # Legend only in the rightmost panel of each row
    axes[i, -1].legend(title="VOI", loc="best", fontsize=14)

    del volume

plt.tight_layout(pad=2.0)
fig3_path = os.path.join(OUTPUT_DIR, "voi_size_intensity_distributions.png")
plt.savefig(fig3_path, dpi=200)
plt.close(fig)
print(f"Figure 3 saved to: {fig3_path}")

print("\nDone.")
