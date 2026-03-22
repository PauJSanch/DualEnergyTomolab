"""
First Approach Reporting Figures

Generates publication-ready plots and statistical comparisons for the
first approach (single-energy linear fitting) with and without thresholding.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# First figures: Show mgv and std on bone on all samples
# Load both CSV files
df_no_thresh = pd.read_csv("VOIs_samples_all_together_without_threshold.csv")
df_thresh = pd.read_csv("VOIs_samples_all_together_with_threshold.csv")

# Create unique labels for the x-axis: VOI (Sample)
df_no_thresh["VOI_Sample"] = df_no_thresh["subvolume_name"] + " (" + df_no_thresh["Sample"] + ")"
df_thresh["VOI_Sample"] = df_thresh["subvolume_name"] + " (" + df_thresh["Sample"] + ")"

# Sort by Sample and subvolume_name to ensure same order
df_no_thresh = df_no_thresh.sort_values(by=["Sample", "subvolume_name"])
df_thresh = df_thresh.sort_values(by=["Sample", "subvolume_name"])

# Plot for data without threshold
plt.figure(figsize=(20, 6))
x_labels_no_thresh = df_no_thresh["VOI_Sample"]

plt.errorbar(x_labels_no_thresh, df_no_thresh["Mean"], yerr=df_no_thresh["Std"], fmt='o', color='#1a9988', label="Without Threshold", capsize=3)

plt.xticks(rotation=90)
plt.title("VOIs Mean ± Std (Without Threshold)", fontsize=14)
plt.xlabel("VOI (Sample)")
plt.ylabel("Mean Value")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# Plot for data with threshold
plt.figure(figsize=(20, 6))
x_labels_thresh = df_thresh["VOI_Sample"]

plt.errorbar(x_labels_thresh, df_thresh["Mean"], yerr=df_thresh["Std"], fmt='o', color='#eb5600', label="With Threshold", capsize=3)

plt.xticks(rotation=90)
plt.title("VOIs Std (With Threshold)", fontsize=14)
plt.xlabel("VOI (Sample)")
plt.ylabel("Mean Value")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# Plot for data without threshold
plt.figure(figsize=(20, 6))
x_labels_no_thresh = df_no_thresh["VOI_Sample"]

plt.plot(x_labels_no_thresh, df_no_thresh["Std"], color='#1a9988', label="Without Thresholdi Std")

plt.xticks(rotation=90)
plt.title("VOIs Std (Without Threshold)", fontsize=14)
plt.xlabel("VOI (Sample)")
plt.ylabel("Mean Value")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# Plot for data with threshold
plt.figure(figsize=(20, 6))
x_labels_thresh = df_thresh["VOI_Sample"]

plt.plot(x_labels_thresh, df_thresh["Std"], color='#eb5600', label="With Threshold Std")

plt.xticks(rotation=90)
plt.title("VOIs Mean ± Std (With Threshold)", fontsize=14)
plt.xlabel("VOI (Sample)")
plt.ylabel("Mean Value")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

del df_no_thresh, df_thresh

# Second figures: Compare DNA results with calculated density
# Load the data
df_no_thresh = pd.read_csv("mean_densities_summary_group_complete_without_threshold.csv")
df_thresh = pd.read_csv("mean_densities_summary_group_complete_with_threshold.csv")
df_dna = pd.read_csv("dna_results.csv")

# Standardize sample names
df_no_thresh["Sample"] = df_no_thresh["Sample"].str.strip()
df_thresh["Sample"] = df_thresh["Sample"].str.strip()
df_dna["Sample"] = df_dna["Sample"].str.strip()

# Merge dataframes on 'Sample'
df_no_thresh_merged = pd.merge(df_no_thresh, df_dna, on="Sample", how="inner")
df_thresh_merged = pd.merge(df_thresh, df_dna, on="Sample", how="inner")

# Plotting settings
sns.set(style="whitegrid")

# Plot WITHOUT threshold
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_no_thresh_merged,
    x="DNA_result",
    y="Mean_Density",
    hue="Group_y",
    style="Group_y",
    s=100
)
plt.xscale("log")
plt.title("Correlation Between DNA Result and Mean Density (Without Threshold)")
plt.xlabel("DNA Result (log scale)")
plt.ylabel("Mean Density")
plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot WITH threshold
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_thresh_merged,
    x="DNA_result",
    y="Mean_Density",
    hue="Group_y",
    style="Group_y",
    s=100
)
plt.xscale("log")
plt.title("Correlation Between DNA Result and Mean Density (With Threshold)")
plt.xlabel("DNA Result (log scale)")
plt.ylabel("Mean Density")
plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Group by Group_y and calculate mean values
grouped_no_thresh = df_no_thresh_merged.groupby("Group_y").agg({
    "DNA_result": "mean",
    "Mean_Density": "mean"
}).reset_index()

grouped_thresh = df_thresh_merged.groupby("Group_y").agg({
    "DNA_result": "mean",
    "Mean_Density": "mean"
}).reset_index()

# Plot group-wise mean WITHOUT threshold
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=grouped_no_thresh,
    x="DNA_result",
    y="Mean_Density",
    hue="Group_y",
    s=150
)
plt.xscale("log")
plt.title("Mean DNA Result vs Mean Density per Group (Without Threshold)")
plt.xlabel("Mean DNA Result (log scale)")
plt.ylabel("Mean Density")
plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot group-wise mean WITH threshold
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=grouped_thresh,
    x="DNA_result",
    y="Mean_Density",
    hue="Group_y",
    s=150
)
plt.xscale("log")
plt.title("Mean DNA Result vs Mean Density per Group (With Threshold)")
plt.xlabel("Mean DNA Result (log scale)")
plt.ylabel("Mean Density")
plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

del df_no_thresh, df_thresh

# Third figure: getting to understand errors
# Load rodes sample data
df = pd.read_csv("rodes_gvs_samples.csv")
rho_rodes = np.array([1.13, 1.16, 1.25, 1.65, 1.9])

df["calculated_density"] = np.nan
df["density_error"] = np.nan

# Linear fit per sample
for sample in df["Sample"].unique():
    sample_df = df[df["Sample"] == sample]
    mean_vals = sample_df["Mean"].values
    rode_ids = sample_df["Rode"].values.astype(int)

    valid_indices = rode_ids <= len(rho_rodes)
    mean_vals = mean_vals[valid_indices]
    actual_rhos = rho_rodes[rode_ids[valid_indices] - 1]

    slope, intercept, _, _, _ = linregress(mean_vals, actual_rhos)
    calc_rhos = slope * sample_df["Mean"] + intercept
    errors = np.abs(calc_rhos.values - rho_rodes[sample_df["Rode"].values - 1])

    df.loc[df["Sample"] == sample, "calculated_density"] = calc_rhos
    df.loc[df["Sample"] == sample, "density_error"] = errors

# Compute mean error per sample
error_summary = df.groupby("Sample")["density_error"].mean().reset_index()
error_summary.rename(columns={"density_error": "mean_density_error"}, inplace=True)

# Load the threshold data
df_without = pd.read_csv("mean_densities_summary_group_complete_without_threshold.csv")
df_with = pd.read_csv("mean_densities_summary_group_complete_with_threshold.csv")

# Merge with error summary
df_wo = pd.merge(df_without, error_summary, on="Sample", how="inner")
df_wt = pd.merge(df_with, error_summary, on="Sample", how="inner")

# Define a function to plot grouped bar chart
def plot_error_comparison(df_merged, threshold_label, output_file):
    samples = df_merged["Sample"]
    x = np.arange(len(samples))  # numeric x locations
    width = 0.35  # bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, df_merged["mean_density_error"], width, label="Fit Error", color='#1a9988')
    bars2 = ax.bar(x + width/2, df_merged["StD_Density"], width, label="StD_Density", color='#eb5600')

    ax.set_ylabel("Error Value")
    ax.set_xlabel("Sample")
    ax.set_title(f"Density Error vs StD_Density ({threshold_label})")
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Generate the two plots
plot_error_comparison(df_wo, "Without Threshold", "bar_comparison_without_threshold.png")
plot_error_comparison(df_wt, "With Threshold", "bar_comparison_with_threshold.png")
