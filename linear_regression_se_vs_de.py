# -*- coding: utf-8 -*-
"""
Linear regression plots comparing SE (LF_wit, LF_wot) vs DE calculated_rho.
Style matches comparative_analysis_three_methods.py.
Outputs saved to Figures_after_review_SR.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINED_CSV = os.path.join(BASE_DIR, "combined_results_after_comments.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Figures_after_review_SR")

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

# ---------- Load data ----------
df = pd.read_csv(COMBINED_CSV)
df.columns = [c.strip() for c in df.columns]

for c in ["calculated_rho", "uncertainty_rho"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["calculated_rho", "uncertainty_rho", "calculation_method_rho", "sample"])

# Split into SE and DE data
se_df = df[df["calculation_method_rho"].isin(["LF_wit", "LF_wot"])][
    ["sample", "subvolume_name", "calculation_method_rho", "calculated_rho", "uncertainty_rho"]
].copy()
de_df = df[df["calculation_method_rho"] == "DE"][
    ["sample", "subvolume_name", "calculated_rho", "uncertainty_rho"]
].copy()

# ---------- Per-method colours for the data points ----------
METHOD_COLORS = {
    "LF_wit": "#A60628",
    "LF_wot": "#7A68A6",
}

# ---------- Helper ----------
def plot_linear_regression(se_method):
    color = METHOD_COLORS[se_method]
    se_subset = se_df[se_df["calculation_method_rho"] == se_method].copy()

    # Pair SE and DE measurements by sample and subvolume
    merged = se_subset.merge(
        de_df, on=["sample", "subvolume_name"], suffixes=("_se", "_de")
    )

    x = merged["calculated_rho_de"].values
    y = merged["calculated_rho_se"].values
    x_err = merged["uncertainty_rho_de"].values
    y_err = merged["uncertainty_rho_se"].values

    # Linear regression (ordinary least squares)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    # R^2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.errorbar(
        x, y,
        xerr=x_err,
        yerr=y_err,
        fmt="o",
        capsize=3,
        alpha=0.7,
        color=color,
        label="Data",
    )

    # Regression line
    x_min, x_max = np.min(x), np.max(x)
    x_line = np.linspace(x_min, x_max, 100)
    y_line = slope * x_line + intercept
    ax.plot(
        x_line, y_line,
        color="black",
        linewidth=2,
        label=f"Fit: y = {slope:.3f}x + {intercept:.3f}",
    )

    # Padded axis limits so that error bars fit comfortably
    x_range = x_max - x_min
    x_pad_low = x_min - 0.05 * x_range
    x_pad_high = x_max + 0.05 * x_range
    y_range = np.max(y) - np.min(y)
    y_pad_low = np.min(y) - 0.05 * y_range
    y_pad_high = np.max(y) + 0.05 * y_range

    ax.set_xlim([x_pad_low, x_pad_high])
    ax.set_ylim([y_pad_low, y_pad_high])
    ax.set_xlabel("DE apparent density (g/cm³)")
    ax.set_ylabel(f"{se_method} apparent density (g/cm³)")
    ax.set_title(f"Linear regression: {se_method} vs DE\n$R^2$ = {r2:.3f}")
    ax.legend(loc="upper left")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"linear_regression_{se_method}_vs_DE.png"),
        dpi=200,
    )
    plt.close(fig)

    return slope, intercept, r2


def plot_group_regression(se_method):
    """Regression using one mean point per sample_group (mean ± SEM)."""
    color = METHOD_COLORS[se_method]

    # Group-level statistics: mean, std, count per sample_group and method
    group_stats = (
        df.groupby(["sample_group", "calculation_method_rho"])["calculated_rho"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    group_stats["sem"] = group_stats["std"] / np.sqrt(group_stats["count"])

    se_group = group_stats[group_stats["calculation_method_rho"] == se_method][
        ["sample_group", "mean", "sem"]
    ].rename(columns={"mean": "rho_se", "sem": "sem_se"})
    de_group = group_stats[group_stats["calculation_method_rho"] == "DE"][
        ["sample_group", "mean", "sem"]
    ].rename(columns={"mean": "rho_de", "sem": "sem_de"})

    merged = se_group.merge(de_group, on="sample_group")

    x = merged["rho_de"].values
    y = merged["rho_se"].values
    x_err = merged["sem_de"].values
    y_err = merged["sem_se"].values

    # Linear regression (ordinary least squares)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    # R^2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.errorbar(
        x, y,
        xerr=x_err,
        yerr=y_err,
        fmt="o",
        capsize=3,
        alpha=0.7,
        color=color,
        label="Data",
    )

    # Regression line
    x_min, x_max = np.min(x), np.max(x)
    x_line = np.linspace(x_min, x_max, 100)
    y_line = slope * x_line + intercept
    ax.plot(
        x_line, y_line,
        color="black",
        linewidth=2,
        label=f"Fit: y = {slope:.3f}x + {intercept:.3f}",
    )

    # Padded axis limits
    x_range = x_max - x_min
    x_pad_low = x_min - 0.05 * x_range
    x_pad_high = x_max + 0.05 * x_range
    y_range = np.max(y) - np.min(y)
    y_pad_low = np.min(y) - 0.05 * y_range
    y_pad_high = np.max(y) + 0.05 * y_range

    ax.set_xlim([x_pad_low, x_pad_high])
    ax.set_ylim([y_pad_low, y_pad_high])
    ax.set_xlabel("DE mean apparent density (g/cm³)")
    ax.set_ylabel(f"{se_method} mean apparent density (g/cm³)")
    ax.set_title(f"Group-level regression: {se_method} vs DE\n$R^2$ = {r2:.3f}")
    ax.legend(loc="upper left")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"group_regression_{se_method}_vs_DE.png"),
        dpi=200,
    )
    plt.close(fig)

    return slope, intercept, r2


# ---------- Individual subvolume regression plots ----------
print("--- Individual subvolume regressions ---")
for method in ["LF_wit", "LF_wot"]:
    slope, intercept, r2 = plot_linear_regression(method)
    print(
        f"{method} vs DE: "
        f"slope = {slope:.4f}, intercept = {intercept:.4f}, R² = {r2:.4f}"
    )

# ---------- Group-level regression plots ----------
print("\n--- Group-level regressions (one point per sample_group) ---")
for method in ["LF_wit", "LF_wot"]:
    slope, intercept, r2 = plot_group_regression(method)
    print(
        f"{method} vs DE: "
        f"slope = {slope:.4f}, intercept = {intercept:.4f}, R² = {r2:.4f}"
    )

# ---------- Signed SE - DE difference statistics ----------
print("\n--- SE - DE signed differences ---")
stds = []
for method in ["LF_wit", "LF_wot"]:
    se_subset = se_df[se_df["calculation_method_rho"] == method].copy()
    merged = se_subset.merge(
        de_df, on=["sample", "subvolume_name"], suffixes=("_se", "_de")
    )
    diff = merged["calculated_rho_se"].values - merged["calculated_rho_de"].values
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    stds.append(std_diff)
    print(
        f"{method}: mean diff = {mean_diff:.4f} g/cm³, "
        f"std = {std_diff:.4f} g/cm³"
    )

print(f"Average std across SE methods: {np.mean(stds):.4f} g/cm³")
print(f"\nDone. Figures saved to: {OUTPUT_DIR}")
