import pandas as pd
import numpy as np

# ============================================================
# FILE
# ============================================================

csv_file = "/home/pausanch/Documents/DualEnergyTomolab_private/combined_results_after_comments_NEW_DE_uncertainty.csv"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(csv_file)

df.columns = [c.strip() for c in df.columns]

# ============================================================
# TARGET SAMPLE
# ============================================================

target_sample = "MGB_1"

# ============================================================
# SELECT SAMPLE
# ============================================================

sample_df = df[df["sample_alias"] == target_sample].copy()

# ============================================================
# GET DE UNCERTAINTY
# ============================================================

de_vals = sample_df.loc[
    sample_df["calculation_method_rho"] == "DE",
    "compatible_uncertainty"
].astype(float)

de_mean = de_vals.mean()

# ============================================================
# GET SE UNCERTAINTIES
# ============================================================

se_vals = sample_df.loc[
    sample_df["calculation_method_rho"] != "DE",
    "uncertainty_rho"
].astype(float)

se_mean = se_vals.mean()

# ============================================================
# DIFFERENCE
# ============================================================

difference = de_mean - se_mean

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n===================================")
print(f"Sample: {target_sample}")
print("===================================\n")

print(f"Mean DE compatible uncertainty : {de_mean:.3f} g/cm³")
print(f"Mean SE uncertainty            : {se_mean:.3f} g/cm³")
print(f"Difference (DE - SE)           : {difference:.3f} g/cm³")

print("\n===================================\n")

# ============================================================
# OPTIONAL SENTENCE FOR PAPER
# ============================================================

sentence = (
    f"For sample {target_sample}, the error associated with the "
    f"DE method is {difference:.3f} g/cm³ higher than that obtained "
    f"using the SE approaches, which average {se_mean:.3f} g/cm³ "
    f"across the two SE variants."
)

print(sentence)
