"""
Amplitude Measurements Across Three Methods

Analyzes amplitude (AMP) parameter correlations with bone density from three
calculation methods. Includes statistical analysis and visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import numpy as np
import os

# === Global style settings (same as reference script) ===
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

# === File paths ===
amp_results = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/DualEnergyTomolab/amp_results.csv"
combined_results = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/DualEnergyTomolab/combined_results_after_comments.csv"
out_dir = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/DualEnergyTomolab/paper/figures"

os.makedirs(out_dir, exist_ok=True)

# === Load data ===
df_amp = pd.read_csv(amp_results)
df_combined = pd.read_csv(combined_results)

# Clean column names
df_amp.columns = [c.strip() for c in df_amp.columns]
df_combined.columns = [c.strip() for c in df_combined.columns]

# Rename for consistency
df_amp = df_amp.rename(columns={
    "Sample": "sample",
    "Porosity": "porosity"
})

# Convert to numeric
df_amp["sample"] = pd.to_numeric(df_amp["sample"], errors="coerce")
df_amp["porosity"] = pd.to_numeric(df_amp["porosity"], errors="coerce")

df_amp = df_amp.dropna(subset=["sample", "porosity"])

# === Aggregate density results per sample & method ===
df_density = (
    df_combined
    .groupby(
        ["sample", "sample_alias", "sample_group", "calculation_method_rho"],
        as_index=False
    )
    .agg(
        mean_rho=("calculated_rho", "mean"),
        uncertainty=("uncertainty_rho", "mean")
    )
)

# === Merge porosity (sample-level) with density ===
df = df_density.merge(df_amp, on="sample", how="inner")

# === Helper: merge some sample groups (same logic as reference) ===
def merge_groups(group):
    if group in ["LUB", "PV"]:
        return "LUB+PV"
    return group

df["sample_group_merged"] = df["sample_group"].apply(merge_groups)

methods = df["calculation_method_rho"].unique()

# === Correlation plot helper ===
def correlation_plot(x, y, xlabel, ylabel, method, filename):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.8, edgecolor="k")

    # Linear regression
    slope, intercept, r_value, p_value, _ = linregress(x, y)
    xx = np.linspace(min(x), max(x), 100)
    yy = slope * xx + intercept

    # Use first color from bmh cycle
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    plt.plot(xx, yy, color=color,
             label=f"r = {r_value:.2f}, p = {p_value:.2e}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}\nMethod: {method}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()

# ======================================================
# === PLOTS: Porosity (AM/P) vs Apparent Density ===
# ======================================================

# --- All samples ---
fig, axes = plt.subplots(1, len(methods), figsize=(15, 5), sharey=True)

for ax, method in zip(axes, methods):
    d = df[df["calculation_method_rho"] == method]
    ax.errorbar(
        d["porosity"],
        d["mean_rho"],
        yerr=d["uncertainty"],
        fmt="o",
        capsize=3
    )
    ax.set_title(f"Method: {method}")
    ax.set_xlabel("AM/P")
    ax.set_ylabel("Mean Apparent Density (g/cm³)")
    ax.grid(True)

plt.suptitle("Correlation between Apparent Density and AM/P by group (all samples)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "rho_vs_porosity_all_samples.png"), dpi=300)

# --- By sample group ---
fig, axes = plt.subplots(1, len(methods), figsize=(15, 5), sharey=True)

for ax, method in zip(axes, methods):
    d = df[df["calculation_method_rho"] == method]
    for group, dg in d.groupby("sample_group_merged"):
        ax.errorbar(
            dg["porosity"],
            dg["mean_rho"],
            yerr=dg["uncertainty"],
            fmt="o",
            capsize=3,
            label=group
        )
    ax.set_title(f"Method: {method}")
    ax.set_xlabel("AM/P")
    ax.set_ylabel("Mean Apparent Density (g/cm³)")
    ax.grid(True)
    ax.legend()

plt.suptitle("Correlation between Apparent Density and AM/P by group (all samples)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "rho_vs_porosity_by_group.png"), dpi=300)

# ======================================================
# === CORRELATION ANALYSIS & INDIVIDUAL PLOTS ===
# ======================================================

print("\nCorrelation results per calculation_method_rho:")
print("\nApparent Density vs Porosity (AM/P):")

for method in methods:
    d = df[df["calculation_method_rho"] == method]

    r, p = pearsonr(d["mean_rho"], d["porosity"])
    print(f"  Method: {method:15s} | r = {r:.3f}, p = {p:.3e}")

    correlation_plot(
        d["porosity"],
        d["mean_rho"],
        xlabel="AM/P",
        ylabel="Apparent Density (g/cm³)",
        method=method,
        filename=f"correlation_rho_vs_porosity_{method}.png"
    )

