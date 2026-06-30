# ============================================================
# DE compatible uncertainty calculation
# Stores new uncertainty in a NEW column:
# compatible_uncertainty
# ============================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------

# Certified rod densities
rod_density = np.array([1.13, 1.16, 1.25, 1.65, 1.9])

# Original combined file
combined_file = "/home/pausanch/Documents/DualEnergyTomolab_private/combined_results_after_comments.csv"

# Base folder containing rod measurements
base_rod_path = "/mnt/tomolab_data/BONE_HA_0"

# Output file
output_file = "/home/pausanch/Documents/DualEnergyTomolab_private/combined_results_after_comments_NEW_DE_uncertainty.csv"


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(combined_file)

# Create new empty column
df["compatible_uncertainty"] = np.nan


# ------------------------------------------------------------
# FUNCTION TO LOAD ROD VALUES
# ------------------------------------------------------------

def load_rod_values(sample_name):

    rod_file = os.path.join(
        base_rod_path,
        f"{sample_name}_rho",
        f"sample_{sample_name}_rho_rode_dual_energy.csv"
    )

    if not os.path.exists(rod_file):
        print(f"Rod file not found:\n{rod_file}")
        return None

    rod_df = pd.read_csv(rod_file)

    if "Mean" not in rod_df.columns:
        print(f"'Mean' column not found in {rod_file}")
        return None

    rod_values = rod_df["Mean"].to_numpy(dtype=float)

    return rod_values


# ------------------------------------------------------------
# CALCULATE COMPATIBLE UNCERTAINTY FOR DE ROWS
# ------------------------------------------------------------

for idx, row in df.iterrows():

    # Only calculate for DE rows
    if row["calculation_method_rho"] != "DE":
        continue

    sample_name = row["sample"]

    # DE density already calculated
    voi_density = float(row["calculated_rho"])

    # Load measured rod values
    rod_values = load_rod_values(sample_name)

    if rod_values is None:
        continue

    if len(rod_values) != len(rod_density):
        print(f"Rod size mismatch for sample {sample_name}")
        continue

    # --------------------------------------------------------
    # Regression:
    # certified density = a + b * measured DE density
    # --------------------------------------------------------

    X = sm.add_constant(rod_values)

    model = sm.OLS(rod_density, X).fit()

    # Prediction for VOI density
    roi_X = sm.add_constant([[voi_density]], has_constant="add")

    prediction = model.get_prediction(roi_X)

    pred_summary = prediction.summary_frame(alpha=0.05)

    lower = float(pred_summary["obs_ci_lower"].values[0])
    upper = float(pred_summary["obs_ci_upper"].values[0])

    # Half-width of 95% prediction interval
    compatible_uncertainty = (upper - lower) / 2.0

    # Store new uncertainty
    df.at[idx, "compatible_uncertainty"] = compatible_uncertainty


# ------------------------------------------------------------
# SAVE UPDATED FILE
# ------------------------------------------------------------

df.to_csv(output_file, index=False)

print("\nDONE")
print(f"\nUpdated file saved to:\n{output_file}")
