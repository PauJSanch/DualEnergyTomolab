"""
DNA and IRSF Correlation Analysis

Analyzes correlations between bone density (from three methods) and DNA
preservation metrics (DNA amount and IRSF - Infrared Splitting Factor).
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# === Load data ===
dna_IRSF_results = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/Results/DNA_results/DNA_IRSF_results.csv"
combined_results = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/Results/combined_results.csv"
out_dir = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/Results/figures_and_tables"

df_dna = pd.read_csv(dna_IRSF_results)
df_combined = pd.read_csv(combined_results)

# === Aggregate density results per sample & method ===
df_density = (
    df_combined
    .groupby(["sample", "sample_group", "calculation_method_rho"], as_index=False)
    .agg(mean_rho=("calculated_rho", "mean"),
         uncertainty=("uncertainty_rho", "mean"))  # assuming uncertainties are averaged
)

# === Merge with DNA results (DNA_amount, IRSF) ===
df = df_density.merge(df_dna, left_on="sample", right_on="Sample", how="inner")

# Helper: map sample groups (merge LUB and PV into one)
def merge_groups(group):
    if group in ["LUB", "PV"]:
        return "LUB+PV"
    return group

df["sample_group_merged"] = df["sample_group"].apply(merge_groups)

methods = df["calculation_method_rho"].unique()

# === PLOT 1: rho vs DNA_amount, per method (including FRESH) ===
fig, axes = plt.subplots(1, len(methods), figsize=(15, 5), sharey=True)

for ax, method in zip(axes, methods):
    d = df[df["calculation_method_rho"] == method]
    ax.errorbar(d["DNA_amount"], d["mean_rho"], yerr=d["uncertainty"],
                fmt="o", capsize=3, label=method)
    ax.set_title(f"Method: {method}")
    ax.set_xlabel("DNA amount (ng DNA / g sample)")
    ax.set_ylabel("Mean density (g/cm³)")
    ax.set_xscale("log")
    ax.grid(True)

plt.suptitle("Correlation between density and DNA amount (all samples)")
plt.tight_layout()
plt.savefig(f"{out_dir}/rho_vs_DNA_all_samples.png", dpi=300)
plt.show()


# === PLOT 2: rho vs DNA_amount, per method, grouped by sample_group (LUB+PV merged) ===
fig, axes = plt.subplots(1, len(methods), figsize=(15, 5), sharey=True)

for ax, method in zip(axes, methods):
    d = df[df["calculation_method_rho"] == method]
    for group, dg in d.groupby("sample_group_merged"):
        ax.errorbar(dg["DNA_amount"], dg["mean_rho"], yerr=dg["uncertainty"],
                    fmt="o", capsize=3, label=group)
    ax.set_title(f"Method: {method}")
    ax.set_xlabel("DNA amount (ng DNA / g sample)")
    ax.set_ylabel("Mean density (g/cm³)")
    ax.set_xscale("log")
    ax.grid(True)
    ax.legend()

plt.suptitle("Correlation between density and DNA amount by group (all samples)")
plt.tight_layout()
plt.savefig(f"{out_dir}/rho_vs_DNA_by_group.png", dpi=300)
plt.show()


# === PLOT 3: rho vs IRSF, per method (including FRESH) ===
fig, axes = plt.subplots(1, len(methods), figsize=(15, 5), sharey=True)

for ax, method in zip(axes, methods):
    d = df[df["calculation_method_rho"] == method]
    ax.errorbar(d["IRSF"], d["mean_rho"], yerr=d["uncertainty"],
                fmt="o", capsize=3, label=method)
    ax.set_title(f"Method: {method}")
    ax.set_xlabel("IRSF")
    ax.set_ylabel("Mean density (g/cm³)")
    ax.set_xscale("log")
    ax.grid(True)

plt.suptitle("Correlation between density and IRSF (all samples)")
plt.tight_layout()
plt.savefig(f"{out_dir}/rho_vs_IRSF_all_samples.png", dpi=300)
plt.show()


# === PLOT 4: rho vs IRSF, per method, grouped by sample_group (LUB+PV merged) ===
fig, axes = plt.subplots(1, len(methods), figsize=(15, 5), sharey=True)

for ax, method in zip(axes, methods):
    d = df[df["calculation_method_rho"] == method]
    for group, dg in d.groupby("sample_group_merged"):
        ax.errorbar(dg["IRSF"], dg["mean_rho"], yerr=dg["uncertainty"],
                    fmt="o", capsize=3, label=group)
    ax.set_title(f"Method: {method}")
    ax.set_xlabel("IRSF")
    ax.set_ylabel("Mean density (g/cm³)")
    ax.set_xscale("log")
    ax.grid(True)
    ax.legend()

plt.suptitle("Correlation between density and IRSF by group (all samples)")
plt.tight_layout()
plt.savefig(f"{out_dir}/rho_vs_IRSF_by_group.png", dpi=300)
plt.show()

# === Correlation analysis: rho vs IRSF, grouped by calculation_method_rho ===
print("\nCorrelation results between rho and IRSF (per calculation_method_rho):")
correlation_results = []

for method in df["calculation_method_rho"].unique():
    d = df[df["calculation_method_rho"] == method]
    r, p = pearsonr(d["mean_rho"], d["IRSF"])
    correlation_results.append((method, r, p))
    print(f"  Method: {method:15s} | r = {r:.3f}, p = {p:.3e}")

