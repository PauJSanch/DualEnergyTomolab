"""
DNA, IRSF, and BPI Multi-Metric Correlation Analysis

Extended correlation analysis including DNA amount, IRSF (Infrared Splitting
Factor), and BPI (Bone Preservation Index) against density measurements.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import numpy as np

# === Global style settings ===
plt.style.use("bmh")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16
})

# === Load data ===
dna_IRSF_results = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/DualEnergyTomolab/DNA_IRSF_BPI_results.csv"
combined_results = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/DualEnergyTomolab/combined_results_after_comments.csv"
out_dir = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/DualEnergyTomolab/paper/figures"

df_dna = pd.read_csv(dna_IRSF_results)
df_combined = pd.read_csv(combined_results)

# === Aggregate density results per sample & method ===
df_density = (
    df_combined
    .groupby(["sample", "sample_alias", "sample_group", "calculation_method_rho"], as_index=False)
    .agg(mean_rho=("calculated_rho", "mean"),
         uncertainty=("uncertainty_rho", "mean"))
)

# === Merge with DNA results (BPI, IRSF) ===
df = df_density.merge(df_dna, left_on="sample", right_on="Sample", how="inner")

# Helper: merge some sample groups
def merge_groups(group):
    if group in ["LUB", "PV"]:
        return "LUB+PV"
    return group

df["sample_group_merged"] = df["sample_group"].apply(merge_groups)
methods = df["calculation_method_rho"].unique()

# === Function to create correlation plots ===
def correlation_plot(x, y, xlabel, ylabel, method, filename):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.8, edgecolor="k")

    # Linear regression line
    slope, intercept, r_value, p_value, _ = linregress(x, y)
    line_x = np.linspace(min(x), max(x), 100)
    line_y = slope * line_x + intercept

    # Use a color from bmh color cycle
    bmh_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    line_color = bmh_colors[0] if len(bmh_colors) > 0 else "C0"

    plt.plot(line_x, line_y, color=line_color, label=f"r = {r_value:.2f}, p = {p_value:.2f}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}\nMethod: {method}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{filename}", dpi=300)
    plt.close()


# === PLOTS (same as before, with Apparent Density wording) ===
fig, axes = plt.subplots(1, len(methods), figsize=(15, 5), sharey=True)
for ax, method in zip(axes, methods):
    d = df[df["calculation_method_rho"] == method]
    ax.errorbar(d["BPI"], d["mean_rho"], yerr=d["uncertainty"],
                fmt="o", capsize=3, label=method)
    ax.set_title(f"Method: {method}")
    ax.set_xlabel("BPI")
    ax.set_ylabel("Mean Apparent Density (g/cm³)")
    ax.set_xscale("log")
    ax.grid(True)
plt.suptitle("Correlation between Apparent Density and BPI (all samples)")
plt.tight_layout()
plt.savefig(f"{out_dir}/rho_vs_BPI_all_samples.png", dpi=300)

fig, axes = plt.subplots(1, len(methods), figsize=(15, 5), sharey=True)
for ax, method in zip(axes, methods):
    d = df[df["calculation_method_rho"] == method]
    for group, dg in d.groupby("sample_group_merged"):
        ax.errorbar(dg["BPI"], dg["mean_rho"], yerr=dg["uncertainty"],
                    fmt="o", capsize=3, label=group)
    ax.set_title(f"Method: {method}")
    ax.set_xlabel("BPI")
    ax.set_ylabel("Mean Apparent Density (g/cm³)")
    ax.set_xscale("log")
    ax.grid(True)
    ax.legend()
plt.suptitle("Correlation between Apparent Density and BPI by group (all samples)")
plt.tight_layout()
plt.savefig(f"{out_dir}/rho_vs_BPI_by_group.png", dpi=300)

fig, axes = plt.subplots(1, len(methods), figsize=(15, 5), sharey=True)
for ax, method in zip(axes, methods):
    d = df[df["calculation_method_rho"] == method]
    ax.errorbar(d["IRSF"], d["mean_rho"], yerr=d["uncertainty"],
                fmt="o", capsize=3, label=method)
    ax.set_title(f"Method: {method}")
    ax.set_xlabel("IRSF")
    ax.set_ylabel("Mean Apparent Density (g/cm³)")
    ax.set_xscale("log")
    ax.grid(True)
plt.suptitle("Correlation between Apparent Density and IRSF (all samples)")
plt.tight_layout()
plt.savefig(f"{out_dir}/rho_vs_IRSF_all_samples.png", dpi=300)

fig, axes = plt.subplots(1, len(methods), figsize=(15, 5), sharey=True)
for ax, method in zip(axes, methods):
    d = df[df["calculation_method_rho"] == method]
    for group, dg in d.groupby("sample_group_merged"):
        ax.errorbar(dg["IRSF"], dg["mean_rho"], yerr=dg["uncertainty"],
                    fmt="o", capsize=3, label=group)
    ax.set_title(f"Method: {method}")
    ax.set_xlabel("IRSF")
    ax.set_ylabel("Mean Apparent Density (g/cm³)")
    ax.set_xscale("log")
    ax.grid(True)
    ax.legend()
plt.suptitle("Correlation between Apparent Density and IRSF by group (all samples)")
plt.tight_layout()
plt.savefig(f"{out_dir}/rho_vs_IRSF_by_group.png", dpi=300)

# === CORRELATION ANALYSES ===
print("\nCorrelation results per calculation_method_rho:")

# rho vs IRSF
print("\nApparent Density vs IRSF:")
for method in methods:
    d = df[df["calculation_method_rho"] == method]
    r, p = pearsonr(d["mean_rho"], d["IRSF"])
    print(f"  Method: {method:15s} | r = {r:.3f}, p = {p:.3e}")
    correlation_plot(d["IRSF"], d["mean_rho"], "IRSF", "Apparent Density (g/cm³)",
                     method, f"correlation_rho_vs_IRSF_{method}.png")

# rho vs BPI
print("\nApparent Density vs BPI:")
for method in methods:
    d = df[df["calculation_method_rho"] == method]
    r, p = pearsonr(d["mean_rho"], d["BPI"])
    print(f"  Method: {method:15s} | r = {r:.3f}, p = {p:.3e}")
    correlation_plot(d["BPI"], d["mean_rho"], "BPI", "Apparent Density (g/cm³)",
                     method, f"correlation_rho_vs_BPI_{method}.png")

# BPI vs IRSF
print("\nBPI vs IRSF:")
for method in methods:
    d = df[df["calculation_method_rho"] == method]
    r, p = pearsonr(d["BPI"], d["IRSF"])
    print(f"  Method: {method:15s} | r = {r:.3f}, p = {p:.3e}")
    correlation_plot(d["IRSF"], d["BPI"], "IRSF", "BPI",
                     method, f"correlation_BPI_vs_IRSF_{method}.png")

