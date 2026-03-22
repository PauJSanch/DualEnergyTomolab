"""
Volume of Interest (VOI) Analysis Functions

This module provides utilities for loading, extracting, and measuring volumes
from raw tomographic data. It supports interactive visualization and statistical
analysis of cubic subvolumes for quantitative CT measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def load_volume(vol_path, dims=(1200, 2048, 2048), start_slice=350, end_slice=800, selected=False):
    """
    Loads a raw volume file and returns a 3D stack of slices.

    Uses memory mapping for efficient handling of large tomographic datasets.
    Optionally provides interactive visualization with a slice-by-slice viewer.

    Parameters
    ----------
    vol_path : str
        Path to the raw binary volume file (expected dtype: float32).
    dims : tuple of int, default=(1200, 2048, 2048)
        Volume dimensions as (depth, height, width).
    start_slice : int, default=350
        Starting slice index for extraction.
    end_slice : int, default=800
        Ending slice index for extraction (exclusive).
    selected : bool, default=False
        If True, displays an interactive slider viewer for the volume.

    Returns
    -------
    volume_sliced : np.ndarray
        3D array of shape (end_slice-start_slice, height, width) containing
        the extracted volume stack.
    """
    # Memory-map the raw file to avoid loading entire volume into RAM
    volume = np.memmap(vol_path, dtype=np.float32, mode='r', shape=dims)

    # Extract the region of interest along the depth axis
    volume_sliced = volume[start_slice:end_slice, :, :]

    if selected:
        # Visualize the selected slices using a slider
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)  # Leave space for the slider

        # Initialize the slice to display
        z_index = start_slice + (end_slice - start_slice) // 2

        # Display the selected slice
        img = ax.imshow(volume_sliced[z_index - start_slice, :, :], cmap="gray")
        ax.set_title(f"Slice {z_index}")

        # Add a colorbar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label("Intensity")

        # Slider for scrolling through slices
        ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])  # Position of the slider
        slider = Slider(ax_slider, "Slice", 0, end_slice - start_slice - 1, valinit=z_index - start_slice, valstep=1)

        # Update function for the slider
        def update(val):
            slice_idx = int(slider.val)
            img.set_data(volume_sliced[slice_idx, :, :])
            ax.set_title(f"Slice {slice_idx + start_slice}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

    # Return the 3D numpy array for further use
    return volume_sliced

def get_subvolume(original_vol, subvol_size, starting_slice, center_y, center_x, selected=False):
    """
    Extracts a cubic subvolume from a larger 3D volume.

    The subvolume is centered at specified (x, y) coordinates and extends
    for a defined number of slices. Useful for analyzing specific regions
    of interest (ROIs) within tomographic data.

    Parameters
    ----------
    original_vol : np.ndarray
        The original 3D volume as a numpy array.
    subvol_size : int
        Edge length of the cubic subvolume (in voxels). The subvolume will
        have dimensions (subvol_size, subvol_size, subvol_size).
    starting_slice : int
        Z-axis slice index where the subvolume begins.
    center_y : int
        Y-coordinate of the subvolume center in the original volume.
    center_x : int
        X-coordinate of the subvolume center in the original volume.
    selected : bool, default=False
        If True, displays the original volume with the subvolume location
        highlighted, using an interactive slice viewer.

    Returns
    -------
    subvolume : np.ndarray
        3D array of shape (subvol_size, subvol_size, subvol_size) containing
        the extracted cubic region.
    """
    # Extract cubic region centered at (center_x, center_y) with specified depth
    subvolume = original_vol[starting_slice:starting_slice + subvol_size,
                             center_y - subvol_size // 2 : center_y + subvol_size // 2,
                             center_x - subvol_size // 2 : center_x + subvol_size // 2]

    if selected:
        # Visualize the entire original volume and highlight the location of the subvolume using a slider
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)  # Leave space for the slider

        # Initialize the slice to display (start with starting_slice)
        z_index = starting_slice + subvol_size // 2

        # Display the selected slice of the original volume
        img = ax.imshow(original_vol[z_index, :, :], cmap="gray")
        ax.set_title(f"Slice {z_index}")

        # Add a colorbar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label("Intensity")

        # Highlight the location of the subvolume in the original volume
        ax.add_patch(plt.Rectangle(
            (center_x - subvol_size // 2, center_y - subvol_size // 2),
            subvol_size, subvol_size,
            linewidth=2, edgecolor='r', facecolor='none', linestyle='--'
        ))

        # Slider for scrolling through the slices of the original volume
        ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])  # Position of the slider
        slider = Slider(ax_slider, "Slice", starting_slice, starting_slice + subvol_size - 1, valinit=z_index, valstep=1)

        # Update function for the slider
        def update(val):
            slice_idx = int(slider.val)
            img.set_data(original_vol[slice_idx, :, :])
            ax.set_title(f"Slice {slice_idx}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

    # Return the cropped subvolume for further use
    return subvolume

def measure_subvolume(subvolume, pixel_size=20, save_csv=False, csv_path="measurements.csv", subvolume_name="subvolume"):
    """
    Computes physical volume and statistical properties of a subvolume.

    Calculates the physical volume in cubic micrometers and computes
    intensity statistics (mean and standard deviation). Results can be
    saved to a CSV file for batch processing and analysis.

    Parameters
    ----------
    subvolume : np.ndarray
        3D array representing the subvolume to measure.
    pixel_size : float, default=20
        Voxel edge length in micrometers (assumes isotropic voxels).
    save_csv : bool, default=False
        If True, appends measurements to a CSV file (creates file if needed).
    csv_path : str, default="measurements.csv"
        Path where CSV results will be saved.
    subvolume_name : str, default="subvolume"
        Identifier for this subvolume in the output data.

    Returns
    -------
    volume_um3 : float
        Physical volume in cubic micrometers (µm³).
    mean_intensity : float
        Mean voxel intensity across the subvolume.
    std_intensity : float
        Standard deviation of voxel intensities.
    df : pd.DataFrame
        DataFrame containing the measurements. If save_csv=True, this data
        is also appended to the specified CSV file.

    Notes
    -----
    For density maps, mean_intensity corresponds to the average apparent
    density in g/cm³. The standard deviation provides a measure of
    material heterogeneity within the VOI.
    """
    # Calculate physical volume from voxel count
    num_voxels = np.prod(subvolume.shape)
    volume_um3 = num_voxels * (pixel_size ** 3)

    # Compute intensity statistics
    mean_intensity = np.mean(subvolume)
    std_intensity = np.std(subvolume)

    # Format results as DataFrame
    new_data = pd.DataFrame([[subvolume_name, volume_um3, mean_intensity, std_intensity]],
                             columns=["subvolume_name", "Volume", "Mean", "Std"])

    # Append to existing CSV or create new file
    if save_csv:
        try:
            df = pd.read_csv(csv_path)
            df = pd.concat([df, new_data], ignore_index=True)
        except FileNotFoundError:
            df = new_data
        df.to_csv(csv_path, index=False)
    else:
        df = new_data

    return volume_um3, mean_intensity, std_intensity, df
