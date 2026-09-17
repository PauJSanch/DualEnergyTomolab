# -*- coding: utf-8 -*-
"""
Calculate the absolute difference between SE (LF_wit / LF_wot) and DE
calculated_rho values, matched by sample and subvolume_name, and report
mean +/- standard deviation for all numerical comparisons in Note.txt.

Uses: combined_results_after_comments.csv
"""

import os
import pandas as pd
import numpy as np

# Path to the CSV (relative to the script location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "combined_results_after_comments.csv")

# Load data
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
for c in ["calculated_rho", "uncertainty_rho"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with missing density/uncertainty
df = df.dropna(subset=["calculated_rho", "uncertainty_rho", "calculation_method_rho", "sample"])

# ------------------------------------------------------------------------
# 1) Per-pair absolute SE - DE density difference (Figure 4a, paragraph 1)
# ------------------------------------------------------------------------
se_methods = ["LF_wit", "LF_wot"]
se_df = df[df["calculation_method_rho"].isin(se_methods)][
    ["sample", "subvolume_name", "calculated_rho"]
].copy()
de_df = df[df["calculation_method_rho"] == "DE"][
    ["sample", "subvolume_name", "calculated_rho"]
].copy()

merged = se_df.merge(de_df, on=["sample", "subvolume_name"], suffixes=("_se", "_de"))
merged["abs_diff"] = (merged["calculated_rho_se"] - merged["calculated_rho_de"]).abs()

mean_abs_diff = merged["abs_diff"].mean()
std_abs_diff = merged["abs_diff"].std(ddof=1)
n_pairs = len(merged)

print("=" * 60)
print("1) Per-pair SE - DE density difference (Figure 4a)")
print("=" * 60)
print(f"Number of SE-DE pairs: {n_pairs}")
print(f"Mean absolute difference: {mean_abs_diff:.4f} g/cm³")
print(f"Standard deviation of absolute differences: {std_abs_diff:.4f} g/cm³")
print(f"Mean +/- SD: {mean_abs_diff:.2f} +/- {std_abs_diff:.2f} g/cm³")
print(f"\nAll absolute differences (g/cm³):")
print(merged["abs_diff"].to_string(index=False))

# ------------------------------------------------------------------------
# 2) Group-averaged absolute SE - DE density difference (Figure 4b)
# ------------------------------------------------------------------------
def weighted_mean(values, sigmas):
    """Inverse-variance weighted mean."""
    values = np.asarray(values, float)
    sigmas = np.asarray(sigmas, float)
    ok = np.isfinite(values) & np.isfinite(sigmas) & (sigmas > 0)
    if ok.sum() == 0:
        return np.nan
    if ok.sum() == 1:
        return values[ok][0]
    w = 1.0 / sigmas[ok] ** 2
    return np.sum(w * values[ok]) / np.sum(w)

by_group_method = (
    df.groupby(["sample_group", "calculation_method_rho"], dropna=False)
    .apply(lambda x: weighted_mean(x["calculated_rho"].values, x["uncertainty_rho"].values))
    .reset_index(name="weighted_mean_rho")
)

de_groups = by_group_method[by_group_method["calculation_method_rho"] == "DE"].set_index("sample_group")["weighted_mean_rho"]
group_diffs = []
for se_method in se_methods:
    se_groups = by_group_method[by_group_method["calculation_method_rho"] == se_method].set_index("sample_group")["weighted_mean_rho"]
    common_groups = de_groups.index.intersection(se_groups.index)
    for g in common_groups:
        group_diffs.append(abs(de_groups.loc[g] - se_groups.loc[g]))

group_diffs = np.asarray(group_diffs, float)
mean_group_diff = np.mean(group_diffs)
std_group_diff = np.std(group_diffs, ddof=1)

print("\n" + "=" * 60)
print("2) Group-averaged SE - DE density difference (Figure 4b)")
print("=" * 60)
print(f"Number of group-method comparisons: {len(group_diffs)}")
print(f"Mean absolute group-level deviation: {mean_group_diff:.4f} g/cm³")
print(f"Standard deviation of group-level deviations: {std_group_diff:.4f} g/cm³")
print(f"Mean +/- SD: {mean_group_diff:.2f} +/- {std_group_diff:.2f} g/cm³")
print(f"\nGroup-level absolute deviations (g/cm³):")
print("\n".join(f"{d:.4f}" for d in group_diffs))

# ------------------------------------------------------------------------
# 3) MGA_1 uncertainty comparison (Figure 6b, paragraph 3)
# ------------------------------------------------------------------------
mga1_df = df[df["sample_alias"] == "MGA_1"].copy()
if mga1_df.empty:
    mga1_df = df[df["sample"] == 19].copy()

mga1_de = mga1_df[mga1_df["calculation_method_rho"] == "DE"][["subvolume_name", "uncertainty_rho"]].copy()
mga1_de = mga1_de.rename(columns={"uncertainty_rho": "unc_de"})

mga1_se = mga1_df[mga1_df["calculation_method_rho"].isin(se_methods)][
    ["subvolume_name", "calculation_method_rho", "uncertainty_rho"]
].copy()

mga1_wit = mga1_se[mga1_se["calculation_method_rho"] == "LF_wit"][["subvolume_name", "uncertainty_rho"]].copy()
mga1_wit = mga1_wit.rename(columns={"uncertainty_rho": "unc_wit"})

mga1_wot = mga1_se[mga1_se["calculation_method_rho"] == "LF_wot"][["subvolume_name", "uncertainty_rho"]].copy()
mga1_wot = mga1_wot.rename(columns={"uncertainty_rho": "unc_wot"})

mga1_merged = mga1_de.merge(mga1_wit, on="subvolume_name").merge(mga1_wot, on="subvolume_name")
mga1_merged["unc_se_avg"] = (mga1_merged["unc_wit"] + mga1_merged["unc_wot"]) / 2.0
mga1_merged["diff_de_minus_se"] = mga1_merged["unc_de"] - mga1_merged["unc_se_avg"]

mean_diff = mga1_merged["diff_de_minus_se"].mean()
std_diff = mga1_merged["diff_de_minus_se"].std(ddof=1)
mean_se_unc = mga1_merged[["unc_wit", "unc_wot"]].values.mean()
std_se_unc = np.std(mga1_merged[["unc_wit", "unc_wot"]].values, ddof=1)
mean_de_unc = mga1_merged["unc_de"].mean()
std_de_unc = mga1_merged["unc_de"].std(ddof=1)

print("\n" + "=" * 60)
print("3) MGA_1 uncertainty comparison (Figure 6b)")
print("=" * 60)
print(f"DE uncertainty - mean SE uncertainty: {mean_diff:.4f} g/cm³")
print(f"Standard deviation of difference: {std_diff:.4f} g/cm³")
print(f"Mean +/- SD: {mean_diff:.3f} +/- {std_diff:.3f} g/cm³")
print(f"\nMean SE uncertainty (across the two SE variants): {mean_se_unc:.4f} g/cm³")
print(f"Standard deviation of SE uncertainties: {std_se_unc:.4f} g/cm³")
print(f"Mean +/- SD: {mean_se_unc:.3f} +/- {std_se_unc:.3f} g/cm³")
print(f"\nDE uncertainty for MGA_1: {mean_de_unc:.4f} +/- {std_de_unc:.4f} g/cm³")
print(f"\nPer-subvolume values (g/cm³):")
print(mga1_merged[["subvolume_name", "unc_wit", "unc_wot", "unc_de", "diff_de_minus_se"]].to_string(index=False))
