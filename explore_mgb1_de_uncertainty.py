# -*- coding: utf-8 -*-
"""
Exploratory plots to investigate why MGB_1 has a high DE regression-based uncertainty.
Uses: combined_results_after_comments_NEW_DE_uncertainty.csv
Column of interest: compatible_uncertainty (plotted as "regression-based uncertainty").
Figures saved to: Figures_after_review_SR
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "combined_results_after_comments_NEW_DE_uncertainty.csv")
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
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

for c in ["mean_gvs", "std_gvs", "calculated_rho", "uncertainty_rho", "compatible_uncertainty"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Keep only DE rows (the only ones with regression-based uncertainty)
de_df = df[df["calculation_method_rho"] == "DE"].copy()
de_df["relative_uncertainty"] = 100 * de_df["compatible_uncertainty"] / de_df["calculated_rho"]

# Split MGB_1 from the rest for highlighting
mgb1_df = de_df[de_df["sample_alias"] == "MGB_1"].copy()
other_df = de_df[de_df["sample_alias"] != "MGB_1"].copy()

# ---------- Helper ----------
def save_and_close(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=200)
    plt.close()

# ---------- Plot 1: regression-based uncertainty vs rho ----------
plt.figure(figsize=(10, 7))
plt.scatter(
    other_df["calculated_rho"], other_df["compatible_uncertainty"],
    color="#348ABD", alpha=0.7, edgecolors="k", s=80, label="Other DE samples"
)
plt.scatter(
    mgb1_df["calculated_rho"], mgb1_df["compatible_uncertainty"],
    color="#A60628", edgecolors="k", s=120, label="MGB_1", zorder=5
)
plt.xlabel("Calculated apparent density (g/cm³)")
plt.ylabel("Regression-based uncertainty")
plt.title("DE regression-based uncertainty vs calculated density")
plt.legend()
plt.grid(True)
save_and_close("de_regression_uncertainty_vs_rho.png")

# ---------- Plot 2: regression-based uncertainty vs mean_gvs ----------
plt.figure(figsize=(10, 7))
plt.scatter(
    other_df["mean_gvs"], other_df["compatible_uncertainty"],
    color="#348ABD", alpha=0.7, edgecolors="k", s=80, label="Other DE samples"
)
plt.scatter(
    mgb1_df["mean_gvs"], mgb1_df["compatible_uncertainty"],
    color="#A60628", edgecolors="k", s=120, label="MGB_1", zorder=5
)
plt.xlabel("Mean gray-value signal (GVS)")
plt.ylabel("Regression-based uncertainty")
plt.title("DE regression-based uncertainty vs mean GVS")
plt.legend()
plt.grid(True)
save_and_close("de_regression_uncertainty_vs_mean_gvs.png")

# ---------- Plot 3: regression-based uncertainty vs std_gvs ----------
plt.figure(figsize=(10, 7))
plt.scatter(
    other_df["std_gvs"], other_df["compatible_uncertainty"],
    color="#348ABD", alpha=0.7, edgecolors="k", s=80, label="Other DE samples"
)
plt.scatter(
    mgb1_df["std_gvs"], mgb1_df["compatible_uncertainty"],
    color="#A60628", edgecolors="k", s=120, label="MGB_1", zorder=5
)
plt.xlabel("Standard deviation of GVS")
plt.ylabel("Regression-based uncertainty")
plt.title("DE regression-based uncertainty vs GVS standard deviation")
plt.legend()
plt.grid(True)
save_and_close("de_regression_uncertainty_vs_std_gvs.png")

# ---------- Plot 4: regression-based uncertainty vs original uncertainty_rho ----------
plt.figure(figsize=(10, 7))
plt.scatter(
    other_df["uncertainty_rho"], other_df["compatible_uncertainty"],
    color="#348ABD", alpha=0.7, edgecolors="k", s=80, label="Other DE samples"
)
plt.scatter(
    mgb1_df["uncertainty_rho"], mgb1_df["compatible_uncertainty"],
    color="#A60628", edgecolors="k", s=120, label="MGB_1", zorder=5
)
plt.xlabel("Original uncertainty_rho")
plt.ylabel("Regression-based uncertainty")
plt.title("DE regression-based uncertainty vs original uncertainty")
plt.legend()
plt.grid(True)
save_and_close("de_regression_uncertainty_vs_original_unc.png")

# ---------- Plot 5: mean regression-based uncertainty per sample ----------
sample_stats = (
    de_df.groupby(["sample", "sample_alias", "sample_group"])["compatible_uncertainty"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .sort_values("mean")
)

plt.figure(figsize=(14, 7))
colors = ["#A60628" if alias == "MGB_1" else "#348ABD" for alias in sample_stats["sample_alias"]]
plt.bar(range(len(sample_stats)), sample_stats["mean"], color=colors, alpha=0.8, edgecolor="k")
plt.xticks(range(len(sample_stats)), sample_stats["sample_alias"], rotation=75, ha="right")
plt.xlabel("Sample")
plt.ylabel("Mean regression-based uncertainty")
plt.title("Mean DE regression-based uncertainty per sample")
plt.grid(True, axis="y")
save_and_close("de_mean_regression_uncertainty_per_sample.png")

# ---------- Plot 6: relative regression-based uncertainty per sample ----------
sample_rel = (
    de_df.groupby(["sample", "sample_alias", "sample_group"])["relative_uncertainty"]
    .mean()
    .reset_index()
    .sort_values("relative_uncertainty")
)

plt.figure(figsize=(14, 7))
colors = ["#A60628" if alias == "MGB_1" else "#348ABD" for alias in sample_rel["sample_alias"]]
plt.bar(range(len(sample_rel)), sample_rel["relative_uncertainty"], color=colors, alpha=0.8, edgecolor="k")
plt.xticks(range(len(sample_rel)), sample_rel["sample_alias"], rotation=75, ha="right")
plt.xlabel("Sample")
plt.ylabel("Mean relative regression-based uncertainty (%)")
plt.title("Mean DE relative regression-based uncertainty per sample")
plt.grid(True, axis="y")
save_and_close("de_mean_relative_regression_uncertainty_per_sample.png")

# ---------- Plot 7: std_gvs vs mean_gvs colored by regression-based uncertainty ----------
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    other_df["mean_gvs"], other_df["std_gvs"],
    c=other_df["compatible_uncertainty"], cmap="viridis",
    alpha=0.8, edgecolors="k", s=80, vmin=de_df["compatible_uncertainty"].min(), vmax=de_df["compatible_uncertainty"].max()
)
plt.scatter(
    mgb1_df["mean_gvs"], mgb1_df["std_gvs"],
    color="#A60628", edgecolors="k", s=120, label="MGB_1", zorder=5
)
plt.colorbar(scatter, label="Regression-based uncertainty")
plt.xlabel("Mean GVS")
plt.ylabel("Std GVS")
plt.title("Std GVS vs mean GVS (color = regression-based uncertainty)")
plt.legend()
plt.grid(True)
save_and_close("de_std_gvs_vs_mean_gvs_colored_by_uncertainty.png")

# ---------- Plot 8 & 9: underlying DE calibration regression for selected samples ----------

# Certified rod densities (must match the calibration set in error_calculation_DE_after_bone_comments.py)
ROD_DENSITY = np.array([1.13, 1.16, 1.25, 1.65, 1.9])
BASE_ROD_PATH = "/mnt/tomolab_data/BONE_HA_0"


def load_rod_values(sample_name):
    """Load measured DE rod values for a given sample."""
    rod_file = os.path.join(
        BASE_ROD_PATH,
        f"{sample_name}_rho",
        f"sample_{sample_name}_rho_rode_dual_energy.csv",
    )
    if not os.path.exists(rod_file):
        print(f"Rod file not found: {rod_file}")
        return None
    rod_df = pd.read_csv(rod_file)
    if "Mean" not in rod_df.columns:
        print(f"'Mean' column not found in {rod_file}")
        return None
    return rod_df["Mean"].to_numpy(dtype=float)


def plot_de_calibration_regression(sample_name, sample_alias):
    """Plot certified density vs measured DE rod density and show the VOI prediction."""
    rod_values = load_rod_values(sample_name)
    if rod_values is None or len(rod_values) != len(ROD_DENSITY):
        print(f"Skipping {sample_alias}: rod values unavailable or mismatched.")
        return

    # Fit: certified density = a + b * measured DE density
    X = sm.add_constant(rod_values)
    model = sm.OLS(ROD_DENSITY, X).fit()

    # VOI density for this sample
    voi_values = de_df[de_df["sample"] == sample_name]["calculated_rho"].unique()
    if len(voi_values) == 0:
        print(f"Skipping {sample_alias}: no VOI density found.")
        return
    voi_density = float(voi_values[0])

    # Prediction for the VOI and its 95% prediction interval
    roi_X = sm.add_constant([[voi_density]], has_constant="add")
    prediction = model.get_prediction(roi_X)
    pred_summary = prediction.summary_frame(alpha=0.05)
    pred_mean = float(pred_summary["mean"].values[0])
    lower = float(pred_summary["obs_ci_lower"].values[0])
    upper = float(pred_summary["obs_ci_upper"].values[0])
    half_width = (upper - lower) / 2.0

    # Regression line spanning the rod range and the VOI point
    x_min = min(rod_values.min(), voi_density)
    x_max = max(rod_values.max(), voi_density)
    x_line = np.linspace(x_min, x_max, 100)
    y_line = model.params[0] + model.params[1] * x_line

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        rod_values, ROD_DENSITY,
        color="#348ABD", s=100, edgecolors="k", zorder=5,
        label="Rod calibration points",
    )
    ax.plot(
        x_line, y_line,
        color="black", linewidth=2,
        label=f"Fit: certified = {model.params[1]:.3f}·measured + {model.params[0]:.3f}",
    )
    ax.errorbar(
        voi_density, pred_mean,
        yerr=half_width,
        fmt="o", color="#A60628", ecolor="#A60628",
        capsize=5, capthick=2, markersize=10,
        label=f"VOI prediction ({sample_alias})\n± regression-based uncertainty = {half_width:.4f}",
        zorder=6,
    )

    ax.set_xlabel("Measured DE density")
    ax.set_ylabel("Certified density (g/cm³)")
    ax.set_title(f"DE calibration regression for {sample_alias}")
    ax.legend(loc="best")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"de_calibration_regression_{sample_alias}.png"),
        dpi=200,
    )
    plt.close(fig)

    # Print regression details
    print(f"\nSample {sample_name} ({sample_alias}):")
    print(f"  Measured rod densities: {rod_values}")
    print(f"  Certified rod densities: {ROD_DENSITY}")
    print(f"  Regression: certified = {model.params[0]:.4f} + {model.params[1]:.4f}·measured")
    print(f"  R² = {model.rsquared:.4f}")
    print(f"  VOI measured density: {voi_density:.4f}")
    print(f"  Predicted certified density: {pred_mean:.4f}")
    print(f"  Regression-based uncertainty (half-width of 95% PI): {half_width:.4f}")


print("--- DE calibration regression for MGB_1 and a comparison sample ---")
plot_de_calibration_regression(42, "MGB_1")
plot_de_calibration_regression(40, "MGB_3")


# ---------- Printed summary ----------
print("MGB_1 DE measurements:")
print(
    mgb1_df[[
        "sample", "subvolume_name", "sample_alias", "sample_group",
        "mean_gvs", "std_gvs", "calculated_rho", "uncertainty_rho",
        "compatible_uncertainty", "relative_uncertainty"
    ]].to_string(index=False)
)

print("\nMean regression-based uncertainty per DE sample:")
print(sample_stats.to_string(index=False))

print(f"\nDone. Figures saved to: {OUTPUT_DIR}")
