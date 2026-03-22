"""
Comprehensive Error Propagation Pipeline

Implements full uncertainty quantification across the density measurement
pipeline. Propagates statistical and systematic errors from calibration
through final density calculations.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import statsmodels.api as sm
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------- USER ADJUSTABLE SETTINGS ----------------
# Base results folder (change if needed)
base_dir = r"/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/Results"

# Rod certified densities (as in your plan)
rod_density = np.array([1.13, 1.16, 1.25, 1.65, 1.9])  # length must match rod files

# Output combined file
df_combined_file = os.path.join(base_dir, "combined_results.csv")

# ---------------- helper functions ----------------
def read_first_existing(paths):
    """Return pd.read_csv(path) for the first path that exists, else None."""
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

def find_column_case_insensitive(df, candidate_names):
    """
    Return the first matching column name (case-insensitive) from candidate_names found in df.
    If none found, raise KeyError.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidate_names:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    raise KeyError(f"None of columns {candidate_names} found in dataframe. Available: {list(df.columns)}")

def safe_float(x):
    """Try to convert to float, return np.nan if fails."""
    try:
        return float(x)
    except Exception:
        return np.nan

def parse_sample_index(sample_value):
    """
    Try to extract an integer sample index from the sample column.
    Accepts numeric types or strings like '1', 'S1', 'sample_01', etc.
    """
    if pd.isna(sample_value):
        return None
    # direct int
    try:
        return int(sample_value)
    except Exception:
        s = str(sample_value)
        # find digits
        import re
        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))
    return None

def sample_to_group(idx):
    """
    Map numeric sample index to group name.
    NOTE: I used a reasonable mapping based on your notes (there were a couple of typos).
    Adjust ranges below if your real mapping is different.
    """
    if idx is None:
        return "UNKNOWN"
    if 1 <= idx <= 14:
        return "OBC"
    if 15 <= idx <= 22:
        return "MG"
    if 23 <= idx <= 35:   # adjusted assumption (was a typo in your file)
        return "FRESH"
    if 36 <= idx <= 44:
        return "LUB"
    if 45 <= idx <= 46:
        return "PV"
    return "UNKNOWN"

# ---------------- prepare combined dataframe ----------------
results = []

cols = [
    "sample", "sample_group", "subvolume_name",
    "mean_gvs", "std_gvs", "calculated_rho",
    "uncertainty_rho", "calculation_method_rho"
]
df_combined = pd.DataFrame(columns=cols)

# ---------------- paths to input files (with fallbacks) ----------------
# LF with threshold VOIs
lf_with_candidates = [
    os.path.join(base_dir, "Standar_With_Threshold", "VOIs_samples_all_together_with_threshold.csv"),
    os.path.join(base_dir, "Standard_With_Threshold", "VOIs_samples_all_together_with_threshold.csv"),
    os.path.join(base_dir, "Standar_With_Threshold", "VOIs_samples_all_together_with_threshold .csv"),
]

# LF without threshold VOIs
lf_without_candidates = [
    os.path.join(base_dir, "Standar_Without_Threshold", "VOIs_samples_all_together_without_threshold.csv"),
    os.path.join(base_dir, "Standard_Without_Threshold", "VOIs_samples_all_together_without_threshold.csv"),
    os.path.join(base_dir, "Standar_With_Threshold", "VOIs_samples_all_together_without_threshold.csv"),
]

# Dual energy VOIs
de_candidates = [
    os.path.join(base_dir, "Dual_Energy", "VOIs_samples_all_together_dual_energy.csv"),
    os.path.join(base_dir, "DualEnergy", "VOIs_samples_all_together_dual_energy.csv"),
]

# ------------------ PROCESS LF WITH THRESHOLD ------------------
df_lf_with = read_first_existing(lf_with_candidates)
if df_lf_with is None:
    logging.warning("LF with-threshold VOI file not found in any fallback path.")
else:
    # determine column names (flexible)
    sample_col = find_column_case_insensitive(df_lf_with, ["Sample", "sample"])
    subvol_col = find_column_case_insensitive(df_lf_with, ["subvolume_name", "Subvolume", "subvolume"])
    mean_col = find_column_case_insensitive(df_lf_with, ["Mean", "mean", "mean_gvs"])
    std_col = find_column_case_insensitive(df_lf_with, ["Std", "std", "std_gvs"])

    logging.info(f"Processing LF (with threshold) file: {sample_col}, {subvol_col}, {mean_col}, {std_col}")

    for _, row in df_lf_with.iterrows():
        sample_val = row[sample_col]
        sample_idx = parse_sample_index(sample_val)
        subvol = row[subvol_col]
        mean_gv = safe_float(row[mean_col])
        std_gv = safe_float(row[std_col])

        # candidate rod gray file paths (note the naming from your plan)
        rod_gray_candidates = [
            os.path.join(base_dir, "Standar_With_Threshold", "Mean_Gray_Values", f"{sample_val}_LE_gvs.csv"),
            os.path.join(base_dir, "Standard_With_Threshold", "Mean_Gray_Values", f"{sample_val}_LE_gvs.csv"),
            os.path.join(base_dir, "Standar_With_Threshold", "Mean_Gray_Values", f"sample_{sample_val}_LE_gvs.csv"),
        ]
        df_rod_gray = read_first_existing(rod_gray_candidates)
        if df_rod_gray is None:
            logging.warning(f"Rod gray file not found for sample {sample_val} (LF_with). Paths tried: {rod_gray_candidates}")
            predicted_density = np.nan
            error = np.nan
        else:
            try:
                rod_mean_col = find_column_case_insensitive(df_rod_gray, ["Mean", "mean", "mean_gvs"])
                rod_gray = df_rod_gray[rod_mean_col].to_numpy(dtype=float).ravel()
            except KeyError:
                logging.error(f"Rod file for sample {sample_val} missing a 'Mean' column. Available columns: {df_rod_gray.columns}")
                rod_gray = np.asarray([])

            # check sizes
            if rod_gray.size != rod_density.size:
                logging.error(f"Rod gray size ({rod_gray.size}) != rod_density size ({rod_density.size}) for sample {sample_val}. Skipping prediction.")
                predicted_density = np.nan
                error = np.nan
            else:
                # Fit regression: y = rod_density, x = rod_gray
                X = sm.add_constant(rod_gray)  # shape (n,2)
                model = sm.OLS(rod_density, X).fit()
                roi_X = sm.add_constant([[mean_gv]], has_constant="add")
                prediction = model.get_prediction(roi_X)
                pred_summary = prediction.summary_frame(alpha=0.05)
                predicted_density = float(pred_summary["mean"].values[0])
                # obs_ci_lower / upper are prediction interval bounds
                lower = float(pred_summary["obs_ci_lower"].values[0])
                upper = float(pred_summary["obs_ci_upper"].values[0])
                error = float((upper - lower) / 2.0)

        results.append({
            "sample": sample_val,
            "sample_group": sample_to_group(sample_idx),
            "subvolume_name": subvol,
            "mean_gvs": mean_gv,
            "std_gvs": std_gv,
            "calculated_rho": predicted_density,
            "uncertainty_rho": error,
            "calculation_method_rho": "LF_wit"
        })


# ------------------ PROCESS LF WITHOUT THRESHOLD ------------------
df_lf_without = read_first_existing(lf_without_candidates)
if df_lf_without is None:
    logging.warning("LF without-threshold VOI file not found in any fallback path.")
else:
    sample_col = find_column_case_insensitive(df_lf_without, ["Sample", "sample"])
    subvol_col = find_column_case_insensitive(df_lf_without, ["subvolume_name", "Subvolume", "subvolume"])
    mean_col = find_column_case_insensitive(df_lf_without, ["Mean", "mean", "mean_gvs"])
    std_col = find_column_case_insensitive(df_lf_without, ["Std", "std", "std_gvs"])

    logging.info(f"Processing LF (without threshold) file: {sample_col}, {subvol_col}, {mean_col}, {std_col}")

    for _, row in df_lf_without.iterrows():
        sample_val = row[sample_col]
        sample_idx = parse_sample_index(sample_val)
        subvol = row[subvol_col]
        mean_gv = safe_float(row[mean_col])
        std_gv = safe_float(row[std_col])

        rod_gray_candidates = [
            os.path.join(base_dir, "Standar_Without_Threshold", "Mean_Gray_Values", f"{sample_val}_LE_gvs.csv"),
            os.path.join(base_dir, "Standard_Without_Threshold", "Mean_Gray_Values", f"{sample_val}_LE_gvs.csv"),
            os.path.join(base_dir, "Standar_With_Threshold", "Mean_Gray_Values", f"{sample_val}_LE_gvs.csv"),
        ]
        df_rod_gray = read_first_existing(rod_gray_candidates)
        if df_rod_gray is None:
            logging.warning(f"Rod gray file not found for sample {sample_val} (LF_wot). Paths tried: {rod_gray_candidates}")
            predicted_density = np.nan
            error = np.nan
        else:
            try:
                rod_mean_col = find_column_case_insensitive(df_rod_gray, ["Mean", "mean", "mean_gvs"])
                rod_gray = df_rod_gray[rod_mean_col].to_numpy(dtype=float).ravel()
            except KeyError:
                logging.error(f"Rod file for sample {sample_val} missing a 'Mean' column. Available columns: {df_rod_gray.columns}")
                rod_gray = np.asarray([])

            if rod_gray.size != rod_density.size:
                logging.error(f"Rod gray size ({rod_gray.size}) != rod_density size ({rod_density.size}) for sample {sample_val}. Skipping prediction.")
                predicted_density = np.nan
                error = np.nan
            else:
                X = sm.add_constant(rod_gray)
                model = sm.OLS(rod_density, X).fit()
                roi_X = sm.add_constant([[mean_gv]], has_constant="add")
                prediction = model.get_prediction(roi_X)
                pred_summary = prediction.summary_frame(alpha=0.05)
                predicted_density = float(pred_summary["mean"].values[0])
                lower = float(pred_summary["obs_ci_lower"].values[0])
                upper = float(pred_summary["obs_ci_upper"].values[0])
                error = float((upper - lower) / 2.0)

        results.append({
            "sample": sample_val,
            "sample_group": sample_to_group(sample_idx),
            "subvolume_name": subvol,
            "mean_gvs": mean_gv,
            "std_gvs": std_gv,
            "calculated_rho": predicted_density,
            "uncertainty_rho": error,
            "calculation_method_rho": "LF_wot"
        })

# ------------------ PROCESS DUAL ENERGY (DE) ------------------
df_de = read_first_existing(de_candidates)
if df_de is None:
    logging.warning("Dual energy VOI file not found.")
else:
    sample_col = find_column_case_insensitive(df_de, ["Sample", "sample"])
    subvol_col = find_column_case_insensitive(df_de, ["subvolume_name", "Subvolume", "subvolume"])
    mean_col = find_column_case_insensitive(df_de, ["Mean", "mean", "mean_gvs"])
    std_col = find_column_case_insensitive(df_de, ["Std", "std", "std_gvs"])

    logging.info(f"Processing Dual Energy file: {sample_col}, {subvol_col}, {mean_col}, {std_col}")

    for _, row in df_de.iterrows():
        sample_val = row[sample_col]
        sample_idx = parse_sample_index(sample_val)
        subvol = row[subvol_col]
        mean_gv = safe_float(row[mean_col])
        std_gv = safe_float(row[std_col])

        # DE rod gray (measured densities) file candidate (note 'sample_{sample}' naming from your plan)
        rod_gray_candidates = [
            os.path.join(base_dir, "Dual_Energy", "Mean_Gray_Values", f"sample_{sample_val}_rho_rode_dual_energy.csv"),
            os.path.join(base_dir, "Dual_Energy", "Mean_Gray_Values", f"{sample_val}_rho_rode_dual_energy.csv"),
            os.path.join(base_dir, "DualEnergy", "Mean_Gray_Values", f"sample_{sample_val}_rho_rode_dual_energy.csv"),
        ]
        df_rod_gray = read_first_existing(rod_gray_candidates)
        if df_rod_gray is None:
            logging.warning(f"DE rod gray file not found for sample {sample_val}. Paths tried: {rod_gray_candidates}")
            uncertainty = np.nan
        else:
            try:
                rod_mean_col = find_column_case_insensitive(df_rod_gray, ["Mean", "mean", "mean_gvs"])
                rod_gray_vals = df_rod_gray[rod_mean_col].to_numpy(dtype=float).ravel()
            except KeyError:
                logging.error(f"DE rod file for sample {sample_val} missing a 'Mean' column. Available columns: {df_rod_gray.columns}")
                rod_gray_vals = np.asarray([])

            if rod_gray_vals.size != rod_density.size:
                logging.error(f"DE rod_gray size ({rod_gray_vals.size}) != rod_density size ({rod_density.size}) for sample {sample_val}. Skipping interpolation.")
                uncertainty = np.nan
            else:
                residuals = rod_gray_vals - rod_density
                abs_residuals = np.abs(residuals)
                # interpolation of absolute residuals vs rod_density
                interp_error = interp1d(rod_density, abs_residuals, kind="linear", fill_value="extrapolate")
                # bone_density is mean_gv (ROI mean)
                uncertainty = np.abs(float(interp_error(mean_gv)))

        results.append({
            "sample": sample_val,
            "sample_group": sample_to_group(sample_idx),
            "subvolume_name": subvol,
            "mean_gvs": mean_gv,
            "std_gvs": std_gv,
            "calculated_rho": mean_gv,         # DE result is directly the ROI mean
            "uncertainty_rho": uncertainty,    # interpolated local error
            "calculation_method_rho": "DE"
        })


# ------------------ SAVE combined csv ------------------
# Write to disk
os.makedirs(base_dir, exist_ok=True)

# Build df_combined from results
df_combined = pd.DataFrame(results, columns=cols)

# Ensure numeric columns are numeric
for col in ["mean_gvs", "std_gvs", "calculated_rho", "uncertainty_rho"]:
    if col in df_combined.columns:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

# Force DE rows: calculated_rho = mean_gvs
df_combined.loc[df_combined["calculation_method_rho"] == "DE", "calculated_rho"] = \
    df_combined.loc[df_combined["calculation_method_rho"] == "DE", "mean_gvs"]

# Save to CSV
df_combined.to_csv(df_combined_file, index=False)
logging.info(f"Combined results saved to: {df_combined_file}")

