"""
Dual-Energy CT Transformation Functions

This module implements the dual-energy CT transformation algorithm that converts
low-energy (LE) and high-energy (HE) volumetric data into quantitative density maps.
The method decomposes materials into epoxy resin and hydroxyapatite components.
"""

import numpy as np


def raw_to_density_map(LE_vol, HE_vol, save_path=None):
    """
    Converts low-energy (LE) and high-energy (HE) volumes into a density map.

    This function implements a dual-energy CT decomposition algorithm that:
    1. Converts gray values to linear attenuation coefficients
    2. Decomposes into basis materials (epoxy resin and hydroxyapatite)
    3. Rotates to density-atomic number space
    4. Calibrates to physical density values

    The transformation uses pre-calibrated coefficients derived from phantom
    measurements with known material compositions.

    Parameters
    ----------
    LE_vol : np.ndarray
        3D array representing the low-energy volume (grayscale values).
        Expected energy: ~45 keV.
    HE_vol : np.ndarray
        3D array representing the high-energy volume (grayscale values).
        Expected energy: ~72 keV. Must have same shape as LE_vol.
    save_path : str, optional
        If provided, saves the output density map as a raw binary file.

    Returns
    -------
    density_map : np.ndarray
        3D array of float32 representing the calibrated density map in g/cm³.

    Notes
    -----
    Physical constants:
    - mu_ha_LE/HE: Hydroxyapatite attenuation coefficients (cm²/g)
    - mu_ep_LE/HE: Epoxy resin attenuation coefficients (cm²/g)
    - rho_ep: Epoxy resin density (1.128 g/cm³)
    - rho_ha: Hydroxyapatite density (3.148 g/cm³)

    The rotation angle theta transforms from (epoxy, HA) space to
    (density, effective-Z) space for improved density quantification.
    """
    # Basis material attenuation coefficients (cm²/g)
    mu_ha_LE = 2.24  # Hydroxyapatite at low energy
    mu_ha_HE = 0.9   # Hydroxyapatite at high energy
    mu_ep_LE = 0.3   # Epoxy resin at low energy
    mu_ep_HE = 0.21  # Epoxy resin at high energy

    # Determinant for solving the 2x2 system of equations
    D = (mu_ep_HE * mu_ha_LE) - (mu_ep_LE * mu_ha_HE)

    # Calibration coefficients: gray value (GV) to attenuation coefficient (mu)
    slope_HE = 15.7
    interc_HE = -0.07
    slope_LE = 17.4
    interc_LE = -0.05

    # Calibration coefficients: rotated coordinate (x_rho) to density (g/cm³)
    slope_density = 1.84583
    interc_density = 0.455649

    # Physical densities of basis materials
    rho_ep = 1.128  # Epoxy resin density (g/cm³)
    rho_ha = 3.148  # Hydroxyapatite density (g/cm³)

    # Rotation angle for coordinate transformation
    theta = np.arctan(rho_ha / rho_ep)

    # Store original dtype for validation
    ORIGINAL_DTYPE = LE_vol.dtype

    # Step 1: Convert gray values to linear attenuation coefficients
    mu_HE = (slope_HE * HE_vol) + interc_HE
    mu_LE = (slope_LE * LE_vol) + interc_LE

    assert mu_LE.dtype == ORIGINAL_DTYPE, f"Dtype changed in mu_LE: {mu_LE.dtype} != {ORIGINAL_DTYPE}"

    # Step 2: Solve for basis material decomposition (x_ep, x_ha)
    # This inverts the system: mu = x_ep * mu_ep + x_ha * mu_ha
    x_ep = ((mu_HE * mu_ha_LE) - (mu_LE * mu_ha_HE)) / D
    x_ha = ((mu_LE * mu_ep_HE) - (mu_HE * mu_ep_LE)) / D

    assert x_ha.dtype == ORIGINAL_DTYPE, f"Dtype changed in x_ha: {x_ha.dtype} != {ORIGINAL_DTYPE}"

    # Step 3: Rotate from (epoxy, HA) space to (density, effective-Z) space
    # x_rho correlates with physical density, x_z with atomic number
    x_rho = (x_ep * np.cos(theta)) + (x_ha * np.sin(theta))
    x_z = -(x_ep * np.sin(theta)) + (x_ha * np.cos(theta))

    assert x_rho.dtype == ORIGINAL_DTYPE, f"Dtype changed in x_rho: {x_rho.dtype} != {ORIGINAL_DTYPE}"

    # Step 4: Convert x_rho to calibrated density using linear relationship
    density_map = (slope_density * x_rho) + interc_density
    density_map = density_map.astype(np.float32)

    assert density_map.dtype == np.float32, f"Dtype changed in density_map: {density_map.dtype} != np.float32"

    # Optionally save to binary file
    if save_path:
        density_map.tofile(save_path)

    return density_map

