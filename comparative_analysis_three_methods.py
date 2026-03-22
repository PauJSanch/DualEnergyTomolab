"""
Comparative Analysis of Three Density Calculation Methods

Compares three approaches for bone density quantification:
1. Linear fitting with threshold (LF_wit)
2. Linear fitting without threshold (LF_wot)
3. Dual-energy decomposition (DE)

Generates weighted statistics, uncertainty analysis, and comparative plots.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Style ----------
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

# ---------- Config ----------
COMBINED_CSV = "/home/pausanch/Elettra/DualEnergyTomolab/combined_results_after_comments.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(COMBINED_CSV), "figures_and_tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load data ----------
df = pd.read_csv(COMBINED_CSV)
df.columns = [c.strip() for c in df.columns]

required_cols = [
    "sample", "sample_group", "subvolume_name", "sample_alias",
    "mean_gvs", "std_gvs",
    "calculated_rho", "uncertainty_rho", "calculation_method_rho",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in combined_results_after_comments.csv: {missing}")

for c in ["mean_gvs", "std_gvs", "calculated_rho", "uncertainty_rho"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = (
    df.dropna(subset=["calculated_rho", "uncertainty_rho",
                      "calculation_method_rho", "sample"])
      .query("uncertainty_rho >= 0")
      .copy()
)

# ---------- Sample order (by sample_alias) ----------
sample_order_df = (
    df[["sample", "sample_alias"]]
    .drop_duplicates()
    .sort_values("sample_alias")
)
samples_sorted = sample_order_df["sample"].tolist()
sample_aliases = sample_order_df["sample_alias"].tolist()

# ---------- Helpers ----------
def weighted_mean_and_uncertainty(values, sigmas):
    values = np.asarray(values, float)
    sigmas = np.asarray(sigmas, float)
    ok = np.isfinite(values) & np.isfinite(sigmas) & (sigmas > 0)

    if ok.sum() >= 2:
        w = 1.0 / sigmas[ok]**2
        return np.sum(w * values[ok]) / np.sum(w), np.sqrt(1.0 / np.sum(w))
    if ok.sum() == 1:
        i = np.where(ok)[0][0]
        return values[i], sigmas[i]

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan
    if finite.size == 1:
        return finite[0], np.nan
    return finite.mean(), finite.std(ddof=1) / math.sqrt(finite.size)

def grouped_weighted_stats(df_in, group_cols):
    return (
        df_in.groupby(group_cols, dropna=False)
        .apply(lambda x: pd.Series(
            weighted_mean_and_uncertainty(
                x["calculated_rho"].values,
                x["uncertainty_rho"].values
            ),
            index=["mean_calculated_rho", "uncertainty_mean_rho"]
        ))
        .reset_index()
    )

# ---------- (1) By sample & method ----------
by_sample_method = grouped_weighted_stats(df, ["sample", "calculation_method_rho"])
by_sample_method.to_csv(
    os.path.join(OUTPUT_DIR, "weighted_density_by_sample_method.csv"),
    index=False
)

methods = by_sample_method["calculation_method_rho"].unique()
x = np.arange(len(samples_sorted))
offset_step = 0.15 if len(methods) > 1 else 0.0

plt.figure(figsize=(14, 6))
for i, m in enumerate(methods):
    d = (
        by_sample_method[by_sample_method["calculation_method_rho"] == m]
        .set_index("sample")
        .reindex(samples_sorted)
    )
    xi = x + (i - (len(methods) - 1) / 2) * offset_step
    plt.errorbar(
        xi,
        d["mean_calculated_rho"],
        yerr=d["uncertainty_mean_rho"],
        fmt="o",
        capsize=3,
        label=str(m),
    )

plt.xticks(x, sample_aliases, rotation=75, ha="right")
plt.xlabel("Sample")
plt.ylabel("Apparent Density (g/cm³)")
plt.title("Calculated apparent density by sample (per method)")
plt.legend(title="Method")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_1_by_sample.png"), dpi=200)

# ---------- (2) By sample_group & method ----------
by_group_method = grouped_weighted_stats(df, ["sample_group", "calculation_method_rho"])
by_group_method.to_csv(
    os.path.join(OUTPUT_DIR, "weighted_density_by_sample_group_method.csv"),
    index=False
)

groups_sorted = sorted(by_group_method["sample_group"].astype(str).unique())
x = np.arange(len(groups_sorted))

plt.figure(figsize=(10, 6))
for i, m in enumerate(methods):
    d = (
        by_group_method[by_group_method["calculation_method_rho"] == m]
        .assign(sample_group=lambda x: x["sample_group"].astype(str))
        .set_index("sample_group")
        .reindex(groups_sorted)
    )
    xi = x + (i - (len(methods) - 1) / 2) * offset_step
    plt.errorbar(
        xi,
        d["mean_calculated_rho"],
        yerr=d["uncertainty_mean_rho"],
        fmt="o",
        capsize=4,
        label=str(m),
    )

plt.xticks(x, groups_sorted)
plt.xlabel("Sample group")
plt.ylabel("Apparent Density (g/cm³)")
plt.title("Calculated apparent density by sample group (per method)")
plt.legend(title="Method")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_2_by_group.png"), dpi=200)

# ---------- (3) % uncertainty ----------
df["percent_unc"] = 100 * df["uncertainty_rho"] / df["calculated_rho"]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["percent_unc"])

# (3a) Boxplot
plt.figure(figsize=(8, 6))
method_order = sorted(df["calculation_method_rho"].astype(str).unique())
plt.boxplot(
    [df.loc[df["calculation_method_rho"].astype(str) == m, "percent_unc"]
     for m in method_order],
    labels=method_order,
    showfliers=False,
)
plt.xlabel("Method")
plt.ylabel("% uncertainty of apparent density")
plt.title("Row-level % uncertainty per method")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_3a_row_level_percent_unc.png"), dpi=200)

# (3b) Per sample
percent_by_sample = (
    df.groupby(["sample", "calculation_method_rho"])["percent_unc"]
    .mean()
    .reset_index()
)
percent_by_sample.to_csv(
    os.path.join(OUTPUT_DIR, "percent_unc_by_sample_method.csv"),
    index=False
)

plt.figure(figsize=(14, 6))
x = np.arange(len(samples_sorted))
for i, m in enumerate(methods):
    d = (
        percent_by_sample[percent_by_sample["calculation_method_rho"] == m]
        .set_index("sample")
        .reindex(samples_sorted)
    )
    xi = x + (i - (len(methods) - 1) / 2) * offset_step
    width = offset_step * 0.9 if offset_step > 0 else 0.6
    plt.bar(xi, d["percent_unc"], width=width, label=str(m))

plt.xticks(x, sample_aliases, rotation=75, ha="right")
plt.xlabel("Sample")
plt.ylabel("Average % uncertainty")
plt.title("Average % uncertainty per sample (per method)")
plt.legend(title="Method")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_3b_percent_by_sample.png"), dpi=200)

# ---------- Done ----------
plt.show()
print(f"Done. Outputs saved to: {OUTPUT_DIR}")

