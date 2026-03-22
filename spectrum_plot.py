"""
X-ray Spectrum Visualization

Plots and analyzes LE and HE X-ray energy spectra from SPEK data files.
Displays photon counts vs. energy for both acquisition energies.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# File paths
le_file = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/spectrum_values/LE_spek_data.txt"
he_file = "/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/spectrum_values/HE_spek_data.txt"

def load_spectrum(file_path):
    """Extract energy and photon counts from a SPEK data file."""
    energy = []
    counts = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        spectrum_start = False
        for line in lines:
            if "**** CALCULATED SPECTRUM ****" in line:
                spectrum_start = True
                continue
            if spectrum_start:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                try:
                    energy.append(float(parts[0]))
                    counts.append(float(parts[1]))
                except ValueError:
                    pass
    return np.array(energy), np.array(counts)

# Load both spectra
E_LE, N_LE = load_spectrum(le_file)
E_HE, N_HE = load_spectrum(he_file)

# Normalize photon counts
N_LE_norm = N_LE / np.max(N_LE)
N_HE_norm = N_HE / np.max(N_HE)

# Set plot style and font
plt.style.use("bmh")
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14

# Plot smooth lines with filled areas
plt.figure(figsize=(10, 6))

plt.plot(E_LE, N_LE_norm, label="Low Energy Spectrum (LE)", linewidth=2, color="C0")
plt.fill_between(E_LE, N_LE_norm, alpha=0.6, color="C0")

plt.plot(E_HE, N_HE_norm, label="High Energy Spectrum (HE)", linewidth=2, color="C1")
plt.fill_between(E_HE, N_HE_norm, alpha=0.6, color="C1")

# Labels, title, legend
plt.xlabel("Energy [keV]")
plt.ylabel("Normalized Photon Count")
plt.title("X-ray Spectra (Normalized)")
plt.legend(fontsize=14)
plt.grid(True, linewidth=0.5)

# Save figure
output_dir = os.path.dirname(le_file)
save_path = os.path.join(output_dir, "normalized_spectra_filled.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Figure saved as: {save_path}")

