"""
Density Distribution Visualization

Creates publication-ready plots showing mean densities and uncertainties
across samples, grouped by sample origin and preservation conditions.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
file_path = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/mean_densities_summary_complete.csv"
df = pd.read_csv(file_path)

# Extract numeric part of Sample for sorting and grouping
df['Sample_ID'] = df['Sample'].str.extract(r'(\d+)').astype(int)

# Define group mappings
def assign_group(sample_id):
    if 1 <= sample_id <= 14:
        return 'OBC (Karst region, 5–10 yrs)'
    elif 15 <= sample_id <= 22:
        return 'MG (WWII mass grave, Croatia)'
    elif 23 <= sample_id <= 26:
        return 'FRESH (recent)'
    elif 36 <= sample_id <= 44:
        return 'LUB (WWII mass grave, Slovenia)'
    else:
        return 'PV (aged femurs, Lombardia)'

df['Group'] = df['Sample_ID'].apply(assign_group)

# Sort by Sample_ID
df = df.sort_values('Sample_ID')
df.to_csv("mean_densities_by_sample.csv", index=False)

# Color map for groups
group_colors = {
    'OBC (Karst region, 5–10 yrs)': 'blue',
    'MG (WWII mass grave, Croatia)': 'red',
    'FRESH (recent)': 'green',
    'LUB (WWII mass grave, Slovenia)': 'orange',
    'PV (aged femurs, Lombardia)': 'purple'
}

# Plot
plt.figure(figsize=(12, 6))
for group, group_df in df.groupby('Group'):
    plt.errorbar(
        group_df['Sample'], group_df['Mean_Density'],
        yerr=group_df['StD_Density'],
        fmt='o', label=group, color=group_colors[group], capsize=5
    )

plt.xticks(rotation=45)
plt.xlabel("Sample")
plt.ylabel("Mean Density")
plt.title("Mean Bone Density per Sample with Standard Deviation")
plt.legend(title="Sample Group")
plt.tight_layout()
plt.grid(True)
plt.show()

# Calculate the mean and std of Mean_Density, grouped by Group
grouped = df.groupby('Group')['Mean_Density'].agg(['mean', 'std'])
grouped.to_csv("mean_densities_by_group.csv", index=False)

# Plot
plt.figure(figsize=(12, 6))
for group in grouped.index:
    mean_value = grouped.loc[group, 'mean']
    std_value = grouped.loc[group, 'std']

    # Plot each group's mean value as a point with error bars
    plt.errorbar(
        group, mean_value, yerr=std_value, fmt='o', label=group, color=group_colors[group], capsize=5
    )

plt.xticks(rotation=45)
plt.xlabel("Group")
plt.ylabel("Mean Density")
plt.title("Mean Bone Density per Group with Standard Deviation")
plt.legend(title="Sample Group")
plt.tight_layout()
plt.grid(True)
plt.show()
