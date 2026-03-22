"""
Threshold Impact Comparison for First Approach

Compares results from the single-energy method with and without intensity
thresholding. Analyzes differences in mean density and uncertainty.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_with = pd.read_csv("mean_densities_summary_group_complete_with_threshold.csv")
df_without = pd.read_csv("mean_densities_summary_group_complete_without_threshold.csv")

# Add a source label
df_with['Source'] = 'With threshold'
df_without['Source'] = 'Without threshold'

# Combine data
df_combined = pd.concat([df_with, df_without], ignore_index=True)

# Define color palette
palette = sns.color_palette("tab10", df_combined['Group'].nunique())

# --- 1) Plot mean density per sample (both analyses combined) ---
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=df_combined,
    x='Sample_ID',
    y='Mean_Density',
    hue='Group',
    style='Source',
    palette=palette
)
plt.errorbar(
    x=df_combined['Sample_ID'],
    y=df_combined['Mean_Density'],
    yerr=df_combined['StD_Density'],
    fmt='none',
    ecolor='gray',
    alpha=0.5
)
plt.title('Mean Density per Sample (Combined)')
plt.xlabel('Sample ID')
plt.ylabel('Mean Density')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- EXTRA: Separated plots for with and without threshold ---
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, (df, title) in zip(axs, [(df_without, 'Without Threshold'), (df_with, 'With Threshold')]):
    sns.scatterplot(
        data=df,
        x='Sample_ID',
        y='Mean_Density',
        hue='Group',
        palette=palette,
        ax=ax
    )
    ax.errorbar(
        x=df['Sample_ID'],
        y=df['Mean_Density'],
        yerr=df['StD_Density'],
        fmt='none',
        ecolor='gray',
        alpha=0.5
    )
    ax.set_title(f'Mean Density per Sample ({title})')
    ax.set_xlabel('Sample ID')
    ax.set_ylabel('Mean Density')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# --- 2) Mean density by group (with average StD as error bars) ---
group_stats = df_combined.groupby(['Group', 'Source']).agg(
    Mean_Density=('Mean_Density', 'mean'),
    StD_Density=('StD_Density', 'mean')
).reset_index()

plt.figure(figsize=(10, 6))

groups = group_stats['Group'].unique()
sources = group_stats['Source'].unique()
colors = {'With threshold': 'C0', 'Without threshold': 'C1'}
marker_style = {'With threshold': 'o', 'Without threshold': 's'}

width = 0.3  # separation between the two methods

for i, group in enumerate(groups):
    for j, source in enumerate(sources):
        subset = group_stats[(group_stats['Group'] == group) & (group_stats['Source'] == source)]
        if not subset.empty:
            x = i + (j - 0.5) * width
            y = subset['Mean_Density'].values[0]
            err = subset['StD_Density'].values[0]

            # Draw error bar first
            plt.errorbar(x, y, yerr=err, fmt='none', ecolor='black', capsize=4, zorder=1)

            # Plot the point over the error bar
            plt.scatter(x, y, color=colors[source], label=source if i == 0 else "",
                        marker=marker_style[source], s=70, zorder=2)

plt.xticks(range(len(groups)), groups, rotation=45, ha='right')
plt.xlabel('Group')
plt.ylabel('Mean Density')
plt.title('Mean Density by Group')
plt.grid(True)
plt.legend(title='Source')
plt.tight_layout()
plt.show()


# --- 3) Absolute Percentage Difference in Mean Density (With vs Without Threshold) ---
merged = pd.merge(
    df_without[['Sample_ID', 'Mean_Density', 'Group']],
    df_with[['Sample_ID', 'Mean_Density']],
    on='Sample_ID',
    suffixes=('_without', '_with')
)

merged['Percent_Difference'] = (
    abs(merged['Mean_Density_with'] - merged['Mean_Density_without']) / merged['Mean_Density_without']) * 100

plt.figure(figsize=(12, 6))
sns.barplot(
    data=merged,
    x='Sample_ID',
    y='Percent_Difference',
    hue='Group',
    palette=palette
)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Absolute Percentage Difference in Mean Density (With vs Without Threshold)')
plt.xlabel('Sample ID')
plt.ylabel('Percent Difference (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

