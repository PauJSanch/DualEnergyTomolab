# -*- coding: utf-8 -*-

"""
Create updated uncertainty figures using:
- uncertainty_rho for LF_wit and LF_wot
- compatible_uncertainty for DE

Outputs:
1. figure_3a_percent_uncertainty_compatible_DE.png
2. figure_3b_percent_uncertainty_by_sample_compatible_DE.png
3. figure_3c_percent_uncertainty_by_group_compatible_DE.png
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STYLE
# ============================================================

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


# ============================================================
# FILES
# ============================================================

COMBINED_CSV = (
    "/home/pausanch/Documents/DualEnergyTomolab_private/"
    "combined_results_after_comments_NEW_DE_uncertainty.csv"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(COMBINED_CSV),
    "updated_uncertainty_figures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(COMBINED_CSV)

df.columns = [c.strip() for c in df.columns]


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_cols = [
    "sample",
    "sample_alias",
    "sample_group",
    "calculated_rho",
    "uncertainty_rho",
    "compatible_uncertainty",
    "calculation_method_rho",
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_cols = [
    "calculated_rho",
    "uncertainty_rho",
    "compatible_uncertainty",
]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")


# ============================================================
# CREATE UNIFIED UNCERTAINTY COLUMN
# ============================================================

df["plot_uncertainty"] = df["uncertainty_rho"]

# Replace DE uncertainties
df.loc[
    df["calculation_method_rho"] == "DE",
    "plot_uncertainty"
] = df.loc[
    df["calculation_method_rho"] == "DE",
    "compatible_uncertainty"
]


# ============================================================
# CLEAN DATA
# ============================================================

df = (
    df
    .dropna(subset=["plot_uncertainty", "calculated_rho"])
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["plot_uncertainty", "calculated_rho"])
    .query("plot_uncertainty >= 0")
    .copy()
)


# ============================================================
# PERCENT UNCERTAINTY
# ============================================================

df["percent_unc"] = (
    100 * df["plot_uncertainty"] / df["calculated_rho"]
)

df = (
    df
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["percent_unc"])
)


# ============================================================
# SAMPLE ORDER
# ============================================================

sample_order_df = (
    df[["sample", "sample_alias"]]
    .drop_duplicates()
    .sort_values("sample_alias")
)

samples_sorted = sample_order_df["sample"].tolist()

sample_aliases = sample_order_df["sample_alias"].tolist()


# ============================================================
# GROUP ORDER
# ============================================================

groups_sorted = sorted(
    df["sample_group"].astype(str).unique()
)


# ============================================================
# METHODS
# ============================================================

method_order = sorted(
    df["calculation_method_rho"].astype(str).unique()
)

offset_step = 0.15 if len(method_order) > 1 else 0.0


# ============================================================
# FIGURE 3A
# ============================================================

plt.figure(figsize=(8, 6))

plt.boxplot(
    [
        df.loc[
            df["calculation_method_rho"].astype(str) == m,
            "percent_unc"
        ]
        for m in method_order
    ],
    labels=method_order,
    showfliers=False,
)

plt.xlabel("Method")

plt.ylabel("% uncertainty of apparent density")

plt.title(
    "Row-level % uncertainty per method\n"
    "(regression-based uncertainty estimation)"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure_3a_percent_uncertainty_compatible_DE.png"
    ),
    dpi=300
)


# ============================================================
# FIGURE 3B - PER SAMPLE
# ============================================================

percent_by_sample = (
    df.groupby(["sample", "calculation_method_rho"])["percent_unc"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(14, 6))

x = np.arange(len(samples_sorted))

for i, m in enumerate(method_order):

    d = (
        percent_by_sample[
            percent_by_sample["calculation_method_rho"] == m
        ]
        .set_index("sample")
        .reindex(samples_sorted)
    )

    xi = x + (i - (len(method_order) - 1) / 2) * offset_step

    width = offset_step * 0.9 if offset_step > 0 else 0.6

    plt.bar(
        xi,
        d["percent_unc"],
        width=width,
        label=str(m)
    )

plt.xticks(
    x,
    sample_aliases,
    rotation=75,
    ha="right"
)

plt.xlabel("Sample")

plt.ylabel("Average % uncertainty")

plt.title(
    "Average % uncertainty per sample\n"
    "(regression-based uncertainty estimation)"
)

plt.legend(title="Method")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure_3b_percent_uncertainty_by_sample_compatible_DE.png"
    ),
    dpi=300
)


# ============================================================
# FIGURE 3C - PER SAMPLE GROUP
# ============================================================

percent_by_group = (
    df.groupby(["sample_group", "calculation_method_rho"])["percent_unc"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))

x = np.arange(len(groups_sorted))

for i, m in enumerate(method_order):

    d = (
        percent_by_group[
            percent_by_group["calculation_method_rho"] == m
        ]
        .assign(sample_group=lambda x: x["sample_group"].astype(str))
        .set_index("sample_group")
        .reindex(groups_sorted)
    )

    xi = x + (i - (len(method_order) - 1) / 2) * offset_step

    width = offset_step * 0.9 if offset_step > 0 else 0.6

    plt.bar(
        xi,
        d["percent_unc"],
        width=width,
        label=str(m)
    )

plt.xticks(x, groups_sorted)

plt.xlabel("Sample group")

plt.ylabel("Average % uncertainty")

plt.title(
    "Average % uncertainty per sample group\n"
    "(regression-based uncertainty estimation)"
)

plt.legend(title="Method")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure_3c_percent_uncertainty_by_group_compatible_DE.png"
    ),
    dpi=300
)


# ============================================================
# DONE
# ============================================================

plt.show()

print("\nDONE")

print(f"\nFigures saved to:\n{OUTPUT_DIR}")
