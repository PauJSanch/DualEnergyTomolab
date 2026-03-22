# DualEnergyTomolab

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Quantitative dual-energy CT analysis for bone density measurements using synchrotron tomography data.

## Overview

This project implements a dual-energy computed tomography (DECT) pipeline for quantitative density measurements of bone samples. The method decomposes materials into basis components (epoxy resin and hydroxyapatite) and converts tomographic grayscale data into calibrated physical density maps.

### Key Features

- **Dual-energy decomposition**: Converts low-energy (LE) and high-energy (HE) CT data into quantitative density maps
- **Calibration pipeline**: Gray values → attenuation coefficients → basis materials → density
- **Volume of Interest (VOI) analysis**: Automated extraction and measurement of cubic subvolumes
- **Multi-method comparison**: Three different approaches for density calculation with uncertainty quantification
- **Error propagation**: Statistical framework for uncertainty estimation across the full pipeline

## Methodology

The dual-energy CT approach uses two X-ray energy spectra (LE ~45 keV, HE ~72 keV) to decompose materials:

1. **Calibration** (`calibration_gvs_to_mus.py`, `calibration_xrho_to_density.py`):
   - Convert gray values to linear attenuation coefficients using phantom measurements
   - Establish relationship between rotated coordinates (x_rho) and physical density

2. **Basis Material Decomposition** (`dual_energy_transformation_functions.py`):
   - Solve 2×2 system: μ = x_ep·μ_ep + x_ha·μ_ha
   - Decompose into epoxy resin and hydroxyapatite components

3. **Coordinate Transformation**:
   - Rotate from (epoxy, HA) space to (density, effective-Z) space
   - Apply calibrated linear transformation to obtain density in g/cm³

4. **VOI Measurements** (`vois_many_sizes_functions.py`):
   - Extract cubic subvolumes at specified anatomical locations
   - Compute mean density and standard deviation for each VOI

## Project Structure

### Core Modules

#### `dual_energy_transformation_functions.py`
Main transformation algorithm implementing the dual-energy decomposition method.

- **`raw_to_density_map(LE_vol, HE_vol, save_path=None)`**: Converts dual-energy volumes to calibrated density maps

#### `vois_many_sizes_functions.py`
Utilities for volume loading, subvolume extraction, and statistical measurements.

- **`load_volume(vol_path, dims, start_slice, end_slice, selected=False)`**: Loads raw volume data with memory mapping
- **`get_subvolume(original_vol, subvol_size, starting_slice, center_y, center_x, selected=False)`**: Extracts cubic VOIs
- **`measure_subvolume(subvolume, pixel_size, save_csv, csv_path, subvolume_name)`**: Computes volume and intensity statistics

### Analysis Scripts

#### Calibration
- **`calibration_gvs_to_mus.py`**: Establishes linear relationship between gray values and attenuation coefficients for LE/HE energies
- **`calibration_xrho_to_density.py`**: Calibrates rotated coordinate (x_rho) to physical density using phantom rods
- **`mus_to_xhaxep_to_density.py`**: Full pipeline from attenuation coefficients to density via basis decomposition

#### Density Calculations
- **`dual_energy_vois_many_sizes.py`**: Batch processing script for extracting and measuring VOIs from dual-energy density maps
- **`dual_energy_transformation_and_vois.py`**: Combined transformation and VOI extraction workflow
- **`dual_energy_sample_rodes_mean_rho_calculation.py`**: Computes mean densities for calibration rod samples

#### Comparative Analysis
- **`comparative_analysis_three_methods.py`**: Compares three density calculation approaches:
  1. First approach (with/without threshold)
  2. Intermediate method
  3. Dual-energy decomposition method

  Generates weighted density statistics by sample and sample group, with uncertainty quantification.

- **`comparative_analysis_three_methods_comments.py`**: Enhanced version with additional documentation

#### Error Analysis
- **`error_propagation_all.py`**: Comprehensive error propagation framework for uncertainty estimation across the full pipeline
- **`first_approach_method_error.py`**: Uncertainty analysis specific to the first approach method

#### Visualization & Reporting
- **`plots_to_report_first_approach.py`**: Generates publication-ready figures for the first approach
- **`spectrum_plot.py`**: Visualizes X-ray energy spectra
- **`show_vois.py`**: Interactive VOI visualization tool
- **`vois_many_sizes_density_plots.py`**: Density distribution plots for multiple VOI sizes
- **`exact_differences_report.py`**: Detailed comparison report between methods
- **`comparison_first_approach_with_without_threshold.py`**: Analyzes impact of thresholding on results

#### Specialized Analysis
- **`vois_many_sizes.py`**: VOI analysis for single-energy datasets
- **`vois_many_sizes_line_fit_columns.py`**: Linear fitting for calibration columns
- **`linear_fitting_columns_bone.py`**: Bone-specific linear calibration fitting
- **`dna_irsf_three_methods.py`**: DNA/IRSF metric calculations across three methods
- **`dna_irsf_bpi_three_methods.py`**: Extended metrics including BPI calculations
- **`amp_three_methods.py`**: Amplitude measurements comparison

### Data Files

#### Input Data
- **`samples_VOIs_centers.csv`**: (x, y) coordinates for VOI centers across samples
- **`rodes_gvs_samples.csv`**: Gray value measurements for calibration rods

#### Results
- **`combined_results_after_comments.csv`**: Combined density measurements with metadata from all methods
- **`VOIs_samples_all_together_with_threshold.csv`**: VOI measurements using threshold filtering
- **`VOIs_samples_all_together_without_threshold.csv`**: VOI measurements without threshold filtering
- **`mean_densities_summary_group_complete_with_threshold.csv`**: Group-level density statistics (with threshold)
- **`mean_densities_summary_group_complete_without_threshold.csv`**: Group-level density statistics (without threshold)
- **`amp_results.csv`**: Amplitude analysis results
- **`porosity_results.csv`**: Porosity calculations

## Workflow

### Basic Pipeline

1. **Calibration** (one-time setup):
   ```bash
   python calibration_gvs_to_mus.py          # GV → μ calibration
   python calibration_xrho_to_density.py     # x_rho → ρ calibration
   ```

2. **Density Map Generation**:
   ```python
   import dual_energy_transformation_functions as detf

   # Load LE and HE volumes
   LE_vol = np.memmap('path/to/LE.raw', dtype=np.float32, shape=(450, 1600, 1600))
   HE_vol = np.memmap('path/to/HE.raw', dtype=np.float32, shape=(450, 1600, 1600))

   # Generate density map
   density_map = detf.raw_to_density_map(LE_vol, HE_vol, save_path='density_map.raw')
   ```

3. **VOI Analysis**:
   ```python
   import vois_many_sizes_functions as voisf

   # Load density map
   vol = voisf.load_volume('density_map.raw', dims=(450, 1600, 1600),
                           start_slice=0, end_slice=449)

   # Extract subvolume
   subvol = voisf.get_subvolume(vol, subvol_size=40, starting_slice=200,
                                 center_y=800, center_x=800)

   # Measure properties
   volume, mean_density, std, df = voisf.measure_subvolume(
       subvol, pixel_size=20, save_csv=True,
       csv_path='results.csv', subvolume_name='sample_01_voi_1'
   )
   ```

4. **Multi-method Comparison**:
   ```bash
   python comparative_analysis_three_methods.py
   ```

### Batch Processing

For processing multiple samples with multiple VOIs per sample:

```bash
python dual_energy_vois_many_sizes.py
```

This script:
- Iterates over sample directories
- Loads density maps for each sample
- Extracts VOIs based on coordinates from `samples_VOIs_centers.csv`
- Computes statistics and saves to per-sample CSV files

## Dependencies

- NumPy
- Pandas
- Matplotlib
- SciPy
- statsmodels

## Physical Constants

### Basis Materials
- **Epoxy resin**: ρ = 1.128 g/cm³, Z_eff = 9.1
- **Hydroxyapatite (HA)**: ρ = 3.148 g/cm³, Z_eff = 16.26

### Energy Spectra
- **Low energy (LE)**: Mean ~45 keV
- **High energy (HE)**: Mean ~72 keV

### Calibration Phantom
Five columns with varying HA concentrations:
- Column 1: 0% HA (pure epoxy), ρ = 1.13 g/cm³
- Column 2: 4.28% HA, ρ = 1.16 g/cm³
- Column 3: 15.86% HA, ρ = 1.25 g/cm³
- Column 4: 48.8% HA, ρ = 1.65 g/cm³
- Column 5: 63.49% HA, ρ = 1.9 g/cm³

## Data Format

### Raw Volume Files
- **Format**: Binary raw files (`.raw`)
- **Data type**: float32
- **Dimensions**: Typically (depth, height, width) = (450-1200, 1600-2048, 1600-2048)
- **Voxel size**: 20 μm isotropic

### CSV Output Format
Measurement files contain:
- `subvolume_name`: VOI identifier
- `Volume`: Physical volume in μm³
- `Mean`: Mean density (g/cm³) or gray value
- `Std`: Standard deviation

## Notes

- Memory-mapped file access enables processing of large volumes without loading into RAM
- All density values are apparent densities (not corrected for porosity)
- Interactive visualization available via `selected=True` parameter in VOI functions
- Statistical uncertainties propagated through full pipeline using error_propagation_all.py

## Authors

Paula Sanchez (Elettra Sincrotrone Trieste)

## Citation

If you use this software in your research, please cite:

```bibtex
@software{sanchez2026dualenergy,
  author = {Sanchez, Paula},
  title = {DualEnergyTomolab: Quantitative Dual-Energy CT Analysis for Bone Density Measurements},
  year = {2026},
  institution = {Elettra Sincrotrone Trieste},
  url = {https://github.com/PauJSanch/DualEnergyTomolab}
}
```

Or in text format:

TODO:
> Sanchez, P. (2026). DualEnergyTomolab: Quantitative Dual-Energy CT Analysis for Bone Density Measurements. Elettra Sincrotrone Trieste. https://github.com/PauJSanch/DualEnergyTomolab

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

**You are free to:**
- Share — copy and redistribute the material
- Adapt — remix, transform, and build upon the material

**Under the following terms:**
- **Attribution** — You must give appropriate credit and cite this work
- **NonCommercial** — You may not use the material for commercial purposes

For commercial licensing inquiries, please contact [institution/email].

See the [LICENSE](LICENSE) file for full details.

## References

Dual-energy CT decomposition methodology based on:
- Basis material decomposition for quantitative bone imaging
- Synchrotron micro-CT for high-resolution density measurements
