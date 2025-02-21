"""
Visualization of Global Maps and Tables
---------------------------------------
This script provides functionality to process, analyze, and visualize climate model data, specifically:
- Generating global maps for various variables and time periods.
- Computing and visualizing ensemble statistics.
- Creating scatter plots to compare ensemble mean performance against OBS and ERA5 land and compare RX5day/annual precipitation to other variables.
- Producing summary tables of global means, percentage changes, and regime shifts.

Functions:
# ========== Utility Functions ==========
- `add_box`: Adds a styled annotation box to a plot.
- `ScatterBoxHandler`: Custom legend handler for scatter-filled legend boxes.
- `scatter_legend`: Generates scatter-filled legend entries for plots.
- `cbar_global_map`: Configures and adds colorbars to global map visualizations.

# ========== Analytical Functions ==========
- `subdivide_bgws`: Segments BGWS data into subdivisions for better analysis.

# ========== Map Plotting Functions ==========
- `plot_var_data_on_map`: Visualizes a variable on a global map for a specific model and period.
- `plot_agreement_mask`: Adds ensemble agreement masks to maps, showing high consensus areas.
- `plot_bgws_sub_change`: Visualizes BGWS subcomponent changes across regions.
- `plot_bgws_flip`: Highlights BGWS regime shifts (positive to negative or vice versa).

# ========== Scatter Plot Functions ==========
- `plot_performance_scatter`: Plots BGWS scatter of OBS/ERA5 land compared to Ensemble mean.
- `plot_scatter_comparison_rx5day_ratio`: Compares scatters of RX5day/annual precipitation (RX5day ratio) against mean precipitation, RX5day and BGWS.

# ========== Table Functions ==========
- `global_mean_table`: Computes and saves global means for selected variables across scenarios.
- `percentage_changes_table`: Calculates and exports area-weighted percentage changes.
- `flip_changes_table`: Generates and exports data showing BGWS regime flips.

Author: Simon P. Heselschwerdt
Date: 2025-02-06
Dependencies: numpy, pandas, matplotlib, seaborn, cartopy, xarray, scipy
"""

# ========== Imports ==========

import os
import numpy as np
import pandas as pd

import seaborn as sns
import cartopy.crs as ccrs
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as patches

from scipy.stats import gaussian_kde
from scipy.stats import pearsonr

# ========== Import Custom Functions ==========

import colormaps_and_utilities as col_uti
import compute_statistics as comp_stats

# ========== Global Settings and Constants ==========

col_map_limits = {
    'historical': {
        'bgws_ensmean': {'vmin': -60, 'vmax': 60, 'steps': 20},
        'bgws_ensstd': {'vmin': 0, 'vmax': 40, 'steps': 5},
        'pr': {'vmin': 0, 'vmax': 10, 'steps': 1},
        'mrro': {'vmin': 0, 'vmax': 3, 'steps': 0.25},
        'tran': {'vmin': 0, 'vmax': 2, 'steps': 0.2},
        'RX5day': {'vmin': 0, 'vmax': 250, 'steps': 25},
        'evapo': {'vmin': 0, 'vmax': 3, 'steps': 0.5},
        'vpd': {'vmin': 0, 'vmax': 30, 'steps': 2.5},
        'wue': {'vmin': 0, 'vmax': 6, 'steps': 1},
        'gpp': {'vmin': 0, 'vmax': 10, 'steps': 1},
        'lai': {'vmin': 0, 'vmax': 6, 'steps': 1}
    },
    'ssp370': {
        'bgws_ensmean': {'vmin': -60, 'vmax': 60, 'steps': 20},
        'bgws_ensstd': {'vmin': 0, 'vmax': 40, 'steps': 5},
        'pr': {'vmin': 0, 'vmax': 10, 'steps': 1},
        'mrro': {'vmin': 0, 'vmax': 3, 'steps': 0.25},
        'tran': {'vmin': 0, 'vmax': 2, 'steps': 0.2},
        'RX5day': {'vmin': 0, 'vmax': 250, 'steps': 25},
        'evapo': {'vmin': 0, 'vmax': 3, 'steps': 0.5},
        'vpd': {'vmin': 0, 'vmax': 30, 'steps': 2.5},
        'wue': {'vmin': 0, 'vmax': 6, 'steps': 1},
        'gpp': {'vmin': 0, 'vmax': 10, 'steps': 1},
        'lai': {'vmin': 0, 'vmax': 6, 'steps': 1}
    },
    'ssp370-historical': {
        'bgws_ensmean': {'vmin': -10, 'vmax': 10, 'steps': 5},
        'bgws_ensstd': {'vmin': 0, 'vmax': 40, 'steps': 5},
        'pr': {'vmin': -0.8, 'vmax': 0.8, 'steps': 0.2},
        #'pr': {'vmin': -50, 'vmax': 50, 'steps': 10}, #activate for relative change
        'mrro': {'vmin': -0.6, 'vmax': 0.6, 'steps': 0.2},
        #'mrro': {'vmin': -100, 'vmax': 100, 'steps': 25}, #activate for relative change
        'tran': {'vmin': -0.3, 'vmax': 0.3, 'steps': 0.1},
        #'tran': {'vmin': -100, 'vmax': 100, 'steps': 25}, #activate for relative change
        'RX5day': {'vmin': -40, 'vmax': 40, 'steps': 10},
        'evapo': {'vmin': -0.3, 'vmax': 0.3, 'steps': 0.1},
        'mrso': {'vmin': -15, 'vmax': 15, 'steps': 5},
        'vpd': {'vmin': 0, 'vmax': 10, 'steps': 1},
        'lai': {'vmin': -1, 'vmax': 1, 'steps': 0.5},
        'wue': {'vmin': -2, 'vmax': 2, 'steps': 0.5}
    }
}

# ========== Utility Functions ==========

def add_box(fig, x, y, box_width, box_height, boxstyle, linewidth, edgecolor, facecolor, zorder):
    rect = patches.FancyBboxPatch(
            (x, y),
            box_width,       
            box_height,       
            boxstyle=boxstyle,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
            transform=fig.transFigure,
            zorder=zorder 
        )
    return rect

class ScatterBoxHandler(HandlerBase):
    """
    Custom legend handler for a scatter-filled box.
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Create the outer box
        rect = Rectangle(
            xy=(-xdescent - 5, -ydescent - 10),
            width=width*1.2,
            height=height* 1.8,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            lw=orig_handle.get_linewidth(),
            transform=trans,
        )

        # Add staggered scatter points inside the box
        num_rows = 4
        num_cols = 4
        scatter_radius = width / 20  # Adjust scatter size relative to box size
        scatter_spacing_x = width / num_cols + 2.5
        scatter_spacing_y = height / num_rows + 2.5
        points = []

        for row in range(num_rows):
            for col in range(num_cols):
                x_offset = scatter_spacing_x / 2 if row % 2 else 0  # Shift every other row
                x = -xdescent - 1 + scatter_spacing_x * col + x_offset
                y = -ydescent - 4 + scatter_spacing_y * row
                points.append(
                    Circle(
                        (x, y),
                        radius=scatter_radius,
                        transform=trans,
                        facecolor="grey",
                        edgecolor="none",
                    )
                )

        # Return the box and scatter points
        return [rect] + points


def scatter_legend(ax, label, facecolor="white", edgecolor="black", lw=0.5):
    """
    Create a legend entry with a scatter-filled box.

    Parameters
    ----------
    ax : matplotlib Axes
        The axis to which the legend is added.
    label : str
        The legend text.
    facecolor : str, optional
        Face color of the box, by default "white".
    edgecolor : str, optional
        Edge color of the box, by default "black".
    lw : float, optional
        Line width of the box edge, by default 0.5.

    Returns
    -------
    matplotlib.patches.Rectangle
        The legend handle.
    """
    return Rectangle((0, 0), 1, 1, facecolor=facecolor, edgecolor=edgecolor, lw=lw, label=label)

def cbar_global_map(img, fig, ax_main, period, model, variable, vmin, vmax, steps):
    # Add colorbar and legend
    if (period == 'historical' and variable == 'bgws' and model != 'Ensemble std') or (period == 'ssp370' and variable == 'bgws' and model != 'Ensemble std') or (period == 'ssp370-historical' and variable != 'vpd' and model != 'Ensemble std'):
        extend = 'both'
    else:
        extend = 'max'

    # Add colorbar
    if variable == 'bgws' and model == 'Ensemble mean' or variable == 'bgws' and model == 'OBS' or variable == 'bgws' and model == 'ERA5':
        cbar = fig.colorbar(img, ax=ax_main, orientation='horizontal', fraction=0.046, pad=0.05, extend=extend, drawedges=True)
    elif variable == 'pr' or variable == 'mrro' or variable == 'tran': 
        cbar = fig.colorbar(img, ax=ax_main, orientation='horizontal', fraction=0.08, pad=0.08, extend=extend, drawedges=True)
    else:
        cbar = fig.colorbar(img, ax=ax_main, orientation='horizontal', fraction=0.06, pad=0.065, extend=extend, drawedges=True)

    # Get full variable name and unit to set label
    display_variable = col_uti.get_global_map_var_name(period, variable)
    if variable == 'bgws' and model == 'Ensemble mean' or variable == 'bgws' and model == 'OBS' or variable == 'bgws' and model == 'ERA5':
        cbar.set_label(display_variable, fontsize=26, weight='bold', labelpad=15)
    elif variable == 'pr' or variable == 'mrro' or variable == 'tran':
        cbar.set_label(display_variable, fontsize=62, weight='bold', labelpad=15)
    else:
        cbar.set_label(display_variable, fontsize=42, weight='bold', labelpad=15)
    
    # Set ticks 
    if variable == 'bgws' and model == 'Ensemble mean' or variable == 'bgws' and model == 'OBS' or variable == 'bgws' and model == 'ERA5':
        cbar.ax.tick_params(labelsize=24)
    elif variable == 'pr' or variable == 'mrro' or variable == 'tran': 
        cbar.ax.tick_params(labelsize=40)
    else:
        cbar.ax.tick_params(labelsize=30)
    
    # Define the ticks and their corresponding labels
    if (period == 'ssp370-historical' and variable != 'vpd') or variable == 'bgws':
        cbar_ticks_steps = steps 
    else:
        cbar_ticks_steps = steps * 2
    cbar_ticks = np.arange(vmin, vmax+cbar_ticks_steps, cbar_ticks_steps)

    if steps < 1 and steps != 0.25: 
        cbar_ticklabels = [f"{tick:.1f}" if abs(tick) > 1e-10 else "0.0" for tick in cbar_ticks] 
    elif steps > 1:
        cbar_ticklabels = [f"{tick:.0f}" if abs(tick) > 1e-10 else "0" for tick in cbar_ticks] 
    else:
        cbar_ticklabels = [f"{tick}" for tick in cbar_ticks]

    # Set the ticks and labels on the colorbar
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels)

# ========== Analytical Functions ==========

def subdivide_bgws(ds_dict_historical, ds_dict_change):
    ds_historical = ds_dict_historical['Ensemble mean'].bgws
    ds_change = ds_dict_change['Ensemble mean'].bgws
    
    if 'member_id' in ds_historical.coords:
        ds_historical = ds_historical.drop('member_id')
    if 'member_id' in ds_change.coords:
        ds_change = ds_change.drop('member_id')

    # Create masks for the current bgws values
    mask_bgws_positive = ds_historical > 0
    mask_bgws_negative = ds_historical < 0

    # Define subdivisions
    subdivisions = {
        'Historical Blue Water Regime': (ds_historical.where(mask_bgws_positive), ds_change.where(mask_bgws_positive)),
        'Historical Green Water Regime': (ds_historical.where(mask_bgws_negative), ds_change.where(mask_bgws_negative))
    }

    return ds_historical, ds_change, subdivisions

# ========== Map Plotting Functions ==========

def plot_var_data_on_map(ds_dict, model, variable, period, dpi=300, filetype='pdf', filepath=None):
    """
    Plot a variable on a map for a specified model and period.

    Parameters:
    ds_dict (dict): Dictionary containing datasets for different models.
    model (str): Model name to plot.
    variable (str): Variable name to plot.
    period (str): Period name (for saving purposes).
    save_fig (bool): Whether to save the figure. Default is False.

    Returns:
    str: Filepath of the saved figure or a message if not saved.
    """
    # Initialize the plot with a cartopy projection
    fig = plt.figure(figsize=(30, 15))
    ax_main = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Drop 'member_id' coordinate if it exists
    ds_dict_cleaned = {name: ds.drop('member_id', errors='ignore') for name, ds in ds_dict.items()}
    
    # Select the dataset for the specified model
    ds = ds_dict_cleaned.get(model)
    if ds is None:
        raise ValueError(f"Model '{model}' not found in ds_dict.")

    # Get colormap
    if variable == 'bgws' and model == 'Ensemble mean':
        cmap_var = 'bgws_ensmean'
    elif variable == 'bgws' and model == 'Ensemble std':
        cmap_var = 'bgws_ensstd'
    else:
        cmap_var = variable 
        
    vmin = col_map_limits[period][cmap_var]['vmin']
    vmax = col_map_limits[period][cmap_var]['vmax']
    steps = col_map_limits[period][cmap_var]['steps']
    cmap, cmap_norm = col_uti.create_colormap(cmap_var, period, vmin, vmax, steps)

    # Plot the selected variable from the dataset
    img = ds[variable].plot(ax=ax_main, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), add_colorbar=False)
  
    # Add coastlines and gridlines for context
    ax_main.coastlines()
    ax_main.tick_params(axis='both', which='major', labelsize=20)
    gridlines = ax_main.gridlines(draw_labels=True, color='black', alpha=0.1, linestyle='--')
    gridlines.top_labels = gridlines.right_labels = False
    if variable == 'bgws' and model == 'Ensemble mean' or variable == 'bgws' and model == 'OBS' or variable == 'bgws' and model == 'ERA5':
        gridlines.xlabel_style = {'size': 24}
        gridlines.ylabel_style = {'size': 24}
    elif variable == 'pr' or variable == 'mrro' or variable == 'tran': 
        gridlines.xlabel_style = {'size': 38}
        gridlines.ylabel_style = {'size': 38}
    else:
        gridlines.xlabel_style = {'size': 30}
        gridlines.ylabel_style = {'size': 30}
    

    if model == 'Ensemble mean' or model == 'Ensemble median':
        if period == 'ssp370-historical':
            plot_agreement_mask(ds_dict_cleaned, ds, model, variable, ax_main)
            
    # Add colorbar
    cbar_global_map(img, fig, ax_main, period, model, variable, vmin, vmax, steps)

    # Plot figure
    plt.show()

    # Save figure
    if filepath is not None:
        filename = f'{model}_{variable}_map_rel.{filetype}'
        col_uti.save_fig(fig, filepath, filename, dpi=dpi)
        print(f"Figure saved under {filepath}{filename}")
    else:
        print("Figure not saved! If you want to save the figure change filepath='your filepath/' in the function call.")
        
def plot_agreement_mask(ds_dict_cleaned, ds, model, variable, ax_main):
    # Compute the agreement in the sign of change across all models, excluding the Ensemble mean and median
    models_to_include = [m for m in ds_dict_cleaned.keys() if "Ensemble" not in m and m != model]
    concatenated_data = xr.concat([ds_dict_cleaned[m][variable] for m in models_to_include], dim='model')

    # Check for positive and negative changes
    positive_agreement = (concatenated_data > 0).mean(dim='model', skipna=True)
    negative_agreement = (concatenated_data < 0).mean(dim='model', skipna=True)

    # Calculate the maximum agreement (should be at least 50%)
    agreement = xr.where(positive_agreement > negative_agreement, positive_agreement, negative_agreement)

    # Since the agreement should be at least 50%, ensure no values are below 0.54545455 / Values below come from grid cells with missing data for some models so leave it marked as uncertain
    agreement = agreement.where(agreement >= 0.54545455, np.nan)
    
    # Mark the regions where the agreement is below 70% -> less than 8 of 11 models agree
    agreement_threshold = 0.7
    low_agreement_mask = agreement < agreement_threshold

    lon, lat = np.meshgrid(ds.lon, ds.lat)
    
    ax_main.scatter(lon[low_agreement_mask], lat[low_agreement_mask], color='grey', marker='D', s=0.8, transform=ccrs.PlateCarree(), label='Low Ensemble Agreement')
    
    # Add the custom scatter-filled legend entry
    scatter_handle = scatter_legend(
        ax_main, label="Low Ensemble Agreement", facecolor="white", edgecolor="black", lw=0.5
    )
    
    # Add the legend
    ax_main.legend(
        handles=[scatter_handle],
        handler_map={type(scatter_handle): ScatterBoxHandler()},
        fontsize=26,
        loc="lower right",
    )

'''def plot_bgws_sub_change(ds_dict_historical, ds_dict_change, dpi=300, filetype='pdf', filepath=None):
    """
    Function to plot subdivisions based on current and change datasets for a specific variable and region.
    Each subdivision is represented by a different color on the map.

    Parameters:
    ds_dict_current (dict): Dictionary containing current datasets for different models.
    ds_dict_change (dict): Dictionary containing change datasets for different models.
    model (str): Model name to plot.
    region (int): Region index to plot.
    variable (str): Variable name to plot.
    save_fig (bool): Whether to save the figure. Default is False.

    Returns:
    str: Filepath of the saved figure or a message if not saved.
    """ 
    
    # Initialize the plot with a cartopy projection
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add coastlines and gridlines for context
    ax.coastlines()
    ax.tick_params(axis='both', which='major', labelsize=20)
    gridlines = ax.gridlines(draw_labels=True, color='black', alpha=0.1, linestyle='--')
    gridlines.top_labels = gridlines.right_labels = False
    gridlines.xlabel_style = {'size': 24}
    gridlines.ylabel_style = {'size': 24}

    # Add box
    rect = add_box(fig, 0.1361, 0.191, 0.175, 0.228, "round,pad=0.004", 1, 'gray', 'white', 0)
    fig.patches.append(rect)

    # Subdivide data
    ds_historical, ds_change, subdivisions = subdivide_bgws(ds_dict_historical, ds_dict_change)

    # Get subdivision colormaps
    bw_cmap, _, gw_cmap, _ = col_uti.create_bgws_change_colormaps(-10, 10, 5)

    # Plot subdivision data
    for i, (name, (subdivision, change)) in enumerate(subdivisions.items()):
        norm = plt.Normalize(vmin=-10, vmax=10)

        if name == 'Historical Blue Water Regime':
            colmap = bw_cmap
        else:
            colmap = gw_cmap
       
        # Apply colormap based on the change values
        pcm = change.plot(ax=ax, add_colorbar=False, cmap=colmap, norm=norm, transform=ccrs.PlateCarree(), add_labels=False)
        
        # Add colorbar for each subdivision, stacked horizontally
        cbar_ax = fig.add_axes([0.175, 0.238 + i * 0.12, 0.1, 0.02], zorder=2)
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', extend='both', drawedges=True)
        cbar.set_ticks([-10, 0, 10])
        cbar.ax.set_xticklabels(['-10', '0', '10'])
        
        # Add the label to the left of the colorbar
        fig.text(0.145 + i * -0.0075, 0.278 + i * 0.12, name, va='center', ha='left', fontsize=26, zorder=2)
        cbar.set_label('$\Delta$ BGWS [%]', fontsize=20)
        cbar.ax.tick_params(labelsize=14)

    plot_agreement_mask(ds_dict_change, ds_historical, 'Ensemble mean', 'bgws', ax)

    # Plot figure
    plt.show()

    # Save figure
    if filepath is not None:
        filename = f'BGWS_subdivision_change_map.{filetype}'
        col_uti.save_fig(fig, filepath, filename, dpi=dpi)
        print(f"Figure saved under {filepath}{filename}")
    else:
        print("Figure not saved! If you want to save the figure change filepath='your filepath/' in the function call.")'''


def plot_bgws_sub_change(ds_dict_historical, ds_dict_change, dpi=300, filetype='pdf', filepath=None):
    """
    Function to plot subdivisions based on current and change datasets for a specific variable and region.
    Each subdivision is represented by a different color on the map.

    Parameters:
    ds_dict_current (dict): Dictionary containing current datasets for different models.
    ds_dict_change (dict): Dictionary containing change datasets for different models.
    model (str): Model name to plot.
    region (int): Region index to plot.
    variable (str): Variable name to plot.
    save_fig (bool): Whether to save the figure. Default is False.

    Returns:
    str: Filepath of the saved figure or a message if not saved.
    """ 
    
    # Initialize the plot with a cartopy projection
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add coastlines and gridlines for context
    ax.coastlines()
    ax.tick_params(axis='both', which='major', labelsize=20)
    gridlines = ax.gridlines(draw_labels=True, color='black', alpha=0.1, linestyle='--')
    gridlines.top_labels = gridlines.right_labels = False
    gridlines.xlabel_style = {'size': 24}
    gridlines.ylabel_style = {'size': 24}

    # Add box
    rect = add_box(fig, 0.1361, 0.191, 0.19, 0.27, "round,pad=0.004", 1, 'gray', 'white', 0)
    fig.patches.append(rect)

    # Subdivide data
    ds_historical, ds_change, subdivisions = subdivide_bgws(ds_dict_historical, ds_dict_change)

    # Get subdivision colormaps
    bw_cmap, _, gw_cmap, _ = col_uti.create_bgws_change_colormaps(-10, 10, 5)

    # Plot subdivision data
    for i, (name, (subdivision, change)) in enumerate(subdivisions.items()):
        norm = plt.Normalize(vmin=-10, vmax=10)

        if name == 'Historical Blue Water Regime':
            colmap = bw_cmap
        else:
            colmap = gw_cmap
       
        # Apply colormap based on the change values
        pcm = change.plot(ax=ax, add_colorbar=False, cmap=colmap, norm=norm, transform=ccrs.PlateCarree(), add_labels=False)
        
        # Add colorbar for each subdivision, stacked horizontally
        cbar_ax = fig.add_axes([0.182, 0.27 + i * 0.12, 0.1, 0.02], zorder=2)
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', extend='both', drawedges=True)
        cbar.set_ticks([-10, 0, 10])
        cbar.ax.set_xticklabels(['-10', '0', '10'])
        
        # Add the label to the left of the colorbar
        fig.text(0.157 + i * -0.0075, 0.32 + i * 0.12, name, va='center', ha='left', fontsize=26, zorder=2)
        cbar.set_label('$\Delta$ BGWS [%]', fontsize=20)
        cbar.ax.tick_params(labelsize=14)

    
    plot_agreement_mask(ds_dict_change, ds_historical, 'Ensemble mean', 'bgws', ax)

    # Plot figure
    plt.show()

    # Save figure
    if filepath is not None:
        filename = f'BGWS_subdivision_change_map.{filetype}'
        col_uti.save_fig(fig, filepath, filename, dpi=dpi)
        print(f"Figure saved under {filepath}{filename}")
    else:
        print("Figure not saved! If you want to save the figure change filepath='your filepath/' in the function call.")

def plot_bgws_flip(ds_dict, ds_dict_change, dpi=300, filetype='pdf', filepath=None):
    """
    Plot a variable on a map for a specified model and period.

    Parameters:
    ds_dict (dict): Dictionary containing datasets for different models.
    model (str): Model name to plot.
    variable (str): Variable name to plot.
    period (str): Period name (for saving purposes).
    save_fig (bool): Whether to save the figure. Default is False.

    Returns:
    str: Filepath of the saved figure or a message if not saved.
    """
    # Initialize the plot with a cartopy projection
    fig = plt.figure(figsize=(30, 15))
    ax_main = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ds_historical = ds_dict['historical']['Ensemble mean'].bgws
    ds_ssp370 = ds_dict['ssp370']['Ensemble mean'].bgws

    # Compute the sign of each dataset
    sign_historical = np.sign(ds_historical)
    sign_ssp370 = np.sign(ds_ssp370)
    
    # Create a mask where the signs differ
    sign_change_mask = sign_historical != sign_ssp370
    
    # Apply the mask to ds_ssp370 (keep only grid cells with sign changes)
    ds_ssp370_masked = ds_ssp370.where(sign_change_mask)

    # Get colormap
    vmin = col_map_limits['ssp370-historical']['bgws_ensmean']['vmin']
    vmax = col_map_limits['ssp370-historical']['bgws_ensmean']['vmax']
    steps = col_map_limits['ssp370-historical']['bgws_ensmean']['steps']
    cmap, cmap_norm = col_uti.create_colormap('bgws_ensmean', 'ssp370-historical', vmin, vmax, steps)

    # Plot the selected variable from the dataset
    img = ds_ssp370_masked.plot(ax=ax_main, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), add_colorbar=False)
  
    # Add coastlines and gridlines for context
    ax_main.coastlines()
    ax_main.tick_params(axis='both', which='major', labelsize=20)
    gridlines = ax_main.gridlines(draw_labels=True, color='black', alpha=0.1, linestyle='--')
    gridlines.top_labels = gridlines.right_labels = False
    gridlines.xlabel_style = {'size': 24}
    gridlines.ylabel_style = {'size': 24}

    # Initialize the new dictionary
    ds_dict_masked = {}
    
    # Loop through the models in ds_dict_change
    for model in ds_dict_change['ssp370-historical']:
        # Access the original dataset
        ds_change = ds_dict_change['ssp370-historical'][model].bgws
        
        # Apply the mask (using the same sign change mask logic)
        ds_masked = ds_change.where(sign_change_mask)
        
        # Ensure the structure matches the original dictionary
        if 'ssp370-historical' not in ds_dict_masked:
            ds_dict_masked['ssp370-historical'] = {}
        if model not in ds_dict_masked['ssp370-historical']:
            ds_dict_masked['ssp370-historical'][model] = {}
        ds_dict_masked['ssp370-historical'][model]['bgws'] = ds_masked

    plot_agreement_mask(ds_dict_masked['ssp370-historical'], ds_ssp370, 'Ensemble mean', 'bgws', ax_main)
            
    # Add colorbar
    cbar_global_map(img, fig, ax_main, 'ssp370', 'Ensemble mean', 'bgws', vmin, vmax, steps)

    # Plot figure
    plt.show()

    # Save figure
    if filepath is not None:
        filename = f'BGWS_flip_map.{filetype}'
        col_uti.save_fig(fig, filepath, filename, dpi=dpi)
        print(f"Figure saved under {filepath}{filename}")
    else:
        print("Figure not saved! If you want to save the figure change filepath='your filepath/' in the function call.")

# ========== Scatter Plot Functions ==========

def plot_performance_scatter(reference, model, variable, ref_label, model_label, dpi=300, output_file=None):
    """
    Creates a scatter plot comparing two xarray datasets for a specific variable with density-based transparency.

    Parameters:
    - reference (xarray.Dataset): Reference dataset (e.g., OBS or ERA5_Land).
    - model (xarray.Dataset): Model dataset (e.g., CMIP6).
    - variable (str): Variable name to compare.
    - ref_label (str): Label for the reference dataset.
    - model_label (str): Label for the model dataset.
    - output_file (str, optional): File path to save the plot. If None, displays the plot.
    """
    # Extract the variable from both datasets
    reference_data = reference[variable].values.flatten()
    model_data = model[variable].values.flatten()

    # Mask invalid values (NaN or Inf) from both datasets
    valid_mask = np.isfinite(reference_data) & np.isfinite(model_data)
    reference_data = reference_data[valid_mask]
    model_data = model_data[valid_mask]

    # Calculate global mean for each dataset
    global_mean_reference = comp_stats.compute_spatial_statistic(reference, 'mean')[variable].values
    global_mean_model = comp_stats.compute_spatial_statistic(model, 'mean')[variable].values

    # Set plot limits
    lims = [-100, 100]

    # Calculate Pearson correlation coefficient
    correlation, _ = pearsonr(reference_data, model_data)
    mean_bias = np.mean(model_data - reference_data)

    # Calculate density using KDE
    xy = np.vstack([reference_data, model_data])
    kde = gaussian_kde(xy)(xy)

    # Normalize density to [0, 1] for transparency adjustment
    kde_normalized = (kde - kde.min()) / (kde.max() - kde.min())

    # Create scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(
        reference_data,
        model_data,
        alpha=np.clip(kde_normalized, 0.1, 1.0),  # Ensure a minimum transparency of 0.2
        color='darkblue',
        s=30,
        edgecolor='darkblue'
    )

    # Add global mean markers
    plt.scatter(global_mean_reference, global_mean_model, color='red', s=100, edgecolor='red', label='Global Mean')

    # Add 1:1 line
    plt.plot(lims, lims, 'k-', lw=0.75)

    # Set axis limits
    plt.xlim(lims)
    plt.ylim(lims)

    if ref_label == 'ERA5_land':
        ref_display = 'ERA5 Land'
    else:
        ref_display = 'Obs'

    # Add labels and title
    plt.xlabel(f"{ref_display} – BGWS [%] ", fontsize=20)
    plt.ylabel(f"{model_label} – BGWS [%]", fontsize=20)

    # Add Pearson correlation coefficient as text
    plt.text(
        0.05, 0.95, rf"$\mathrm{{Pearson's\ }} r = {correlation:.2f}$", fontsize=16,
        transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    )

    # Customize grid and ticks
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=15)

    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Scatter plot saved to {output_file}")
    else:
        plt.show()

def plot_scatter_comparison_rx5day_ratio(rx5day_ratio, ds, var_name, dpi=300, output_file=None):
    """
    Creates a scatter plot comparing RX5day ratio to a selected variable with density-based transparency.

    Parameters:
    - rx5day_ratio (np.array): RX5day-to-Annual Precipitation ratio.
    - var (np.array): Comparison variable values.
    - output_file (str, optional): File path to save the plot. If None, displays the plot.
    """
    # Get var data
    var_data= ds[var_name].values
    
    # Flatten and mask invalid values (NaN or Inf)
    rx5day_ratio = rx5day_ratio.flatten() * 100
    var_data = var_data.flatten()
    valid_mask = np.isfinite(rx5day_ratio) & np.isfinite(var_data)
    rx5day_ratio = rx5day_ratio[valid_mask]
    var_data = var_data[valid_mask]

    # Set plot limits
    #if var 
    #lims = [-60, 60]

    # Calculate Pearson correlation coefficient
    correlation, _ = pearsonr(rx5day_ratio, var_data)

    # Calculate density using KDE
    xy = np.vstack([rx5day_ratio, var_data])
    kde = gaussian_kde(xy)(xy)

    # Normalize density to [0, 1] for transparency adjustment
    kde_normalized = (kde - kde.min()) / (kde.max() - kde.min())

    # Create scatter plot
    plt.figure(figsize=(10, 10))

    # Split data for color coding if bgws
    if var_name == 'bgws':
        below_zero = var_data < 0
        above_zero = var_data > 0

        # Plot points below zero in green
        plt.scatter(
            rx5day_ratio[below_zero],
            var_data[below_zero],
            alpha=np.clip(kde_normalized[below_zero], 0.1, 1.0),
            color=(30/255, 130/255, 30/255),
            s=30,
            edgecolor='none'
        )
    
        # Plot points above zero in blue
        plt.scatter(
            rx5day_ratio[above_zero],
            var_data[above_zero],
            alpha=np.clip(kde_normalized[above_zero], 0.1, 1.0),
            color=(55/255, 140/255, 225/255),
            s=30,
            edgecolor='none'
        )
         # Set plot limits
        lims_bgws = [-50, 90]
        lims_ratio = [0,60]
    
        # Add horizontal line at y=0
        plt.axhline(0, color='black', linestyle='-', linewidth=1)


    else:
        plt.scatter(
            rx5day_ratio,
            var_data,
            alpha=np.clip(kde_normalized, 0.1, 1.0),  # Minimum transparency of 0.1
            color='blue',
            s=30,
            edgecolor='none'
        )

    # Add labels and title
    plt.xlabel("RX5day / Annual Precipitation Ratio [%]", fontsize=16)
    if var_name == 'pr':
        plt.ylabel("Precipitation [mm/day]", fontsize=16)
    elif var_name == 'RX5day':
        plt.ylabel("RX5day [mm]", fontsize=16)
    elif var_name == 'bgws':
        plt.ylabel("BGWS [%]", fontsize=16)
    else:
        plt.ylabel("", fontsize=16)
        print(f'No y-axis label for {var_name}')

    # Add Pearson correlation coefficient as text
    plt.text(
        0.05, 0.95, rf"$\mathrm{{Pearson's\ }} r = {correlation:.2f}$", fontsize=16,
        transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    )

    # Customize grid and ticks
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Scatter plot saved to {output_file}")
    else:
        plt.show()

# ========== Table Functions ==========

def global_mean_table(ds_dict, ds_dict_change, variables, filepath):
    """
    Generate a table with global means for selected variables, considering both historical and future scenarios, and save it as a CSV.

    Parameters:
    ds_dict (dict): Dictionary containing the data for different models and experiments.
    ds_dict_change (dict): Dictionary containing the change data between future and historical periods.
    variables (list): List of variable names to include in the table.
    filepath (str): Directory where the CSV file will be saved.
    """
    # Models to skip
    skip_models = {'OBS', 'ERA5_land', 'Ensemble std'}

    # Initialize the index for rows (model names), excluding the models to be skipped
    row_index = [model for model in ds_dict[list(ds_dict.keys())[0]].keys() if model not in skip_models]

    # Initialize the MultiIndex for columns
    col_tuples = []
    
    for var in variables:
        display_name = col_uti.get_global_map_var_name('historical', var)
        col_tuples.append((display_name, 'Historical'))
        col_tuples.append((display_name, 'SSP3-7.0'))
        col_tuples.append((display_name, 'Change'))

    # Create a MultiIndex for the columns
    columns = pd.MultiIndex.from_tuples(col_tuples)

    # Initialize the DataFrame to hold the results
    results = pd.DataFrame(index=row_index, columns=columns)

    # Populate the DataFrame
    for model in row_index:  # Only iterate over the filtered models
        for var in variables:
            try:
                # Calculate the weighted mean for historical data
                historical_mean = comp_stats.compute_spatial_statistic(ds_dict['historical'][model][var], statistic='mean')
                historical_mean_rounded = round(historical_mean.item(), 2)

                # Calculate the weighted mean for SSP3-7.0 data
                future_mean = comp_stats.compute_spatial_statistic(ds_dict['ssp370'][model][var], statistic='mean')
                future_mean_rounded = round(future_mean.item(), 2)

                # Calculate the change between future and historical
                change_mean = comp_stats.compute_spatial_statistic(ds_dict_change['ssp370-historical'][model][var], statistic='mean')
                change_mean_rounded = round(change_mean.item(), 2)

                # Assign the values to the DataFrame
                results.loc[model, (col_uti.get_global_map_var_name('historical', var), 'Historical')] = historical_mean_rounded
                results.loc[model, (col_uti.get_global_map_var_name('historical', var), 'SSP3-7.0')] = future_mean_rounded
                results.loc[model, (col_uti.get_global_map_var_name('historical', var), 'Change')] = change_mean_rounded
            
            except KeyError:
                print(f"Skipping variable {var} for model {model} due to missing data.")

    if filepath is not None:
        # Ensure the save directory exists
        os.makedirs(filepath, exist_ok=True)
        filename = 'global_mean_table.csv'
        savepath = os.path.join(filepath, filename)
    
        # Save the DataFrame to a CSV file
        results.to_csv(savepath)
    
        print(f"Table saved to {savepath}")

    return results


def percentage_changes_table(ds_dict_current, ds_dict_change, variable='bgws', filepath=None):
    """
    Save the area percentage changes data to a CSV file for each model, organized by historical state and change.

    Parameters:
    ds_dict_current (dict): Dictionary containing current datasets for different models.
    ds_dict_change (dict): Dictionary containing change datasets for different models.
    variable (str): Variable name to compute the changes.
    save_dir (str): Directory where the CSV file will be saved.
    """
    # Models to skip
    skip_models = {'OBS', 'ERA5_land', 'Ensemble std'}
    
    # Filter models
    models = [model for model in ds_dict_current.keys() if model not in skip_models]

    # Prepare the structure of the results with simplified labels
    columns = pd.MultiIndex.from_product(
        [('Positive', 'Negative'), 
         ('Positive', 'Negative')],
        names=['Historical State', 'Change']
    )
    results = pd.DataFrame(index=models, columns=columns)

    # Populate the DataFrame
    for model in models:
        ds_current = ds_dict_current[model].drop('member_id', errors='ignore')
        ds_change = ds_dict_change[model].drop('member_id', errors='ignore')

        result = comp_stats.compute_area_percentage_changes(ds_current, ds_change, variable)
        
        results.loc[model, ('Positive', 'Positive')] = result['Positive']['Positive']
        results.loc[model, ('Positive', 'Negative')] = result['Positive']['Negative']
        results.loc[model, ('Negative', 'Positive')] = result['Negative']['Positive']
        results.loc[model, ('Negative', 'Negative')] = result['Negative']['Negative']

    if filepath is not None:
        # Ensure the save directory exists
        os.makedirs(filepath, exist_ok=True)
        filename = f'percentage_changes_{variable}_area_table.csv'
        savepath = os.path.join(filepath, filename)
    
        # Save the DataFrame to a CSV file
        results.to_csv(savepath)
    
        print(f"Table saved to {savepath}")
    
    return results

def flip_changes_table(ds_dict_current, ds_dict_future, variable='bgws', filepath=None):
    """
    Save the flip changes data to a CSV file for each model.

    Parameters:
    ds_dict_current (dict): Dictionary containing historical datasets for different models.
    ds_dict_future (dict): Dictionary containing future datasets for different models.
    variable (str): Variable name to compute the changes.
    filepath (str): Directory where the CSV file will be saved.
    """
    # Models to skip
    skip_models = {'OBS', 'ERA5_land', 'Ensemble std'}
    
    # Filter models
    models = [model for model in ds_dict_current.keys() if model not in skip_models]

    results = {'Model': []}

    for model in models:
        try:
            ds_current = ds_dict_current[model][variable].drop('member_id', errors='ignore')
            ds_future = ds_dict_future[model][variable].drop('member_id', errors='ignore')

            result = comp_stats.compute_flip_changes(ds_current, ds_future, variable)
            
            results['Model'].append(model)
            for key, value in result.items():
                if key not in results:
                    results[key] = []
                results[key].append(value)
        except KeyError:
            print(f"Skipping model '{model}' due to missing data.")

    df = pd.DataFrame(results)

    if filepath is not None:
        # Ensure the save directory exists
        os.makedirs(filepath, exist_ok=True)
        filename = f'flip_changes_{variable}_table.csv'
        savepath = os.path.join(filepath, filename)
    
        # Save the DataFrame to a CSV file
        df.to_csv(savepath)
    
        print(f"Table saved to {savepath}")
        
    return df