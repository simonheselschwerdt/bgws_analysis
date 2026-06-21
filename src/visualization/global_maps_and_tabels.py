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
Date: 21.06.2026
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
        'bgws_tran_mean': {'vmin': -60, 'vmax': 60, 'steps': 20},
        'r_over_p_mean': {'vmin': 0, 'vmax': 60, 'steps': 5},
        'et_over_p_mean': {'vmin': 0, 'vmax': 60, 'steps': 5},
        'bgws_ensmean': {'vmin': -60, 'vmax': 60, 'steps': 20},
        'bgws_ensstd': {'vmin': 0, 'vmax': 40, 'steps': 5}, 
        'pr_seasonality': {'vmin': 0, 'vmax': 6, 'steps': 1},
        'precipitation_intensity_index': {'vmin': -3, 'vmax': 3, 'steps': 1},
        'pr_mean': {'vmin': 0, 'vmax': 10, 'steps': 1},
        'mrro_mean': {'vmin': 0, 'vmax': 3, 'steps': 0.25},
        'tran_mean': {'vmin': 0, 'vmax': 2, 'steps': 0.2},
        'RX5day': {'vmin': 0, 'vmax': 250, 'steps': 25},
        'evapo': {'vmin': 0, 'vmax': 3, 'steps': 0.5},
        'vpd': {'vmin': 0, 'vmax': 30, 'steps': 2.5},
        'wue': {'vmin': 0, 'vmax': 6, 'steps': 1},
        'gpp': {'vmin': 0, 'vmax': 10, 'steps': 1},
        'lai': {'vmin': 0, 'vmax': 6, 'steps': 1},
        'mrro_pr': {'vmin': 0, 'vmax': 60, 'steps': 10},
        'tran_pr': {'vmin': 0, 'vmax': 60, 'steps': 10},
        'rx5day_ratio': {'vmin': 0, 'vmax': 30, 'steps': 2.5},
        'mrsos': {'vmin': 0, 'vmax': 30, 'steps': 2.5},
        'tas': {'vmin': 0, 'vmax': 50, 'steps': 5},
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
        'lai': {'vmin': 0, 'vmax': 6, 'steps': 1},
        'mrsos': {'vmin': 0, 'vmax': 30, 'steps': 2.5},
        'tas': {'vmin': 0, 'vmax': 40, 'steps': 5},
    },
    'ssp370_ff-historical': {
        'bgws_tran_mean': {'vmin': -10, 'vmax': 10, 'steps': 5},
        'bgws_ensmean': {'vmin': -10, 'vmax': 10, 'steps': 5},
        'bgws_ensstd': {'vmin': 0, 'vmax': 40, 'steps': 5},
        'pr_mean': {'vmin': -0.6, 'vmax': 0.6, 'steps': 0.2},
        #'pr': {'vmin': -50, 'vmax': 50, 'steps': 10}, #activate for relative change
        'mrro_mean': {'vmin': -0.6, 'vmax': 0.6, 'steps': 0.2},
        #'mrro': {'vmin': -100, 'vmax': 100, 'steps': 25}, #activate for relative change
        'tran_mean': {'vmin': -0.6, 'vmax': 0.6, 'steps': 0.2},
        #'tran': {'vmin': -100, 'vmax': 100, 'steps': 25}, #activate for relative change
        'RX5day': {'vmin': -40, 'vmax': 40, 'steps': 10},
        'evapo': {'vmin': -0.3, 'vmax': 0.3, 'steps': 0.1},
        'mrso': {'vmin': -15, 'vmax': 15, 'steps': 5},
        'vpd': {'vmin': 0, 'vmax': 10, 'steps': 1},
        'lai': {'vmin': -1, 'vmax': 1, 'steps': 0.5},
        'wue': {'vmin': -2, 'vmax': 2, 'steps': 0.5},
        'mrro_pr': {'vmin': -10, 'vmax': 10, 'steps': 2.5},
        'tran_pr': {'vmin': -10, 'vmax': 10, 'steps': 2.5},
        'mrsos': {'vmin': -5, 'vmax': 5, 'steps': 2.5},
        'tas': {'vmin': 0, 'vmax': 10, 'steps': 2.5},
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
    def __init__(self, xdescent_offset_box=0, ydescent_offset_box=0, box_width_offset=0, box_height_offset=0, scatter_spacing_x_offset=0, scatter_spacing_y_offset=0, scatter_vertical_offset=0, scatter_horizontal_offset=0):
        super().__init__()
        self.xdescent_offset = xdescent_offset_box
        self.ydescent_offset = ydescent_offset_box
        self.box_width_offset = box_width_offset
        self.box_height_offset = box_height_offset
        self.scatter_spacing_x_offset = scatter_spacing_x_offset
        self.scatter_spacing_y_offset = scatter_spacing_y_offset
        self.scatter_vertical_offset = scatter_vertical_offset
        self.scatter_horizontal_offset = scatter_horizontal_offset

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):

        rect = Rectangle(
            xy=(xdescent + self.xdescent_offset,
                ydescent + self.ydescent_offset),
            width=width * self.box_width_offset, 
            height=height * self.box_height_offset, 
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            lw=orig_handle.get_linewidth(),
            transform=trans,
        )

        num_rows = 4
        num_cols = 4
        scatter_radius = width / 20
        scatter_spacing_x = width / num_cols + self.scatter_spacing_x_offset
        scatter_spacing_y = height / num_rows + self.scatter_spacing_y_offset

        points = []
        for row in range(num_rows):
            for col in range(num_cols):
                x_offset = (scatter_spacing_x / 2) if (row % 2) else 0
                x = xdescent + scatter_spacing_x * col + x_offset - self.scatter_horizontal_offset
                y = ydescent + scatter_spacing_y * row + self.scatter_vertical_offset #-1
                points.append(
                    Circle(
                        (x, y),
                        radius=scatter_radius,
                        transform=trans,
                        facecolor="grey",
                        edgecolor="none",
                    )
                )

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
    if (period == 'historical' and variable == 'bgws' and model != '12 model ensemble std') or (period == 'ssp370' and variable == 'bgws' and model != '12 model ensemble std') or (period == 'ssp370-historical' and variable != 'vpd' and model != '12 model ensemble std'):
        extend = 'both'
    else:
        extend = 'max'

    # Add colorbar
    if variable == 'bgws' and model == '12 model ensemble mean' or variable == 'bgws' and model == 'OBS' or variable == 'bgws' and model == 'ERA5':
        cbar = fig.colorbar(img, ax=ax_main, orientation='horizontal', fraction=0.046, pad=0.05, extend=extend, drawedges=True)
    elif variable == 'pr' or variable == 'mrro' or variable == 'tran': 
        cbar = fig.colorbar(img, ax=ax_main, orientation='horizontal', fraction=0.08, pad=0.08, extend=extend, drawedges=True)
    else:
        cbar = fig.colorbar(img, ax=ax_main, orientation='horizontal', fraction=0.06, pad=0.065, extend=extend, drawedges=True)

    # Get full variable name and unit to set label
    display_variable = col_uti.get_global_map_var_name(period, variable)
    if variable == 'bgws' and model == '12 model ensemble mean' or variable == 'bgws' and model == 'OBS' or variable == 'bgws' and model == 'ERA5':
        cbar.set_label(display_variable, fontsize=26, weight='bold', labelpad=15)
    elif variable == 'pr' or variable == 'mrro' or variable == 'tran':
        cbar.set_label(display_variable, fontsize=62, weight='bold', labelpad=15)
    else:
        cbar.set_label(display_variable, fontsize=42, weight='bold', labelpad=15)
    
    # Set ticks 
    if variable == 'bgws' and model == '12 model ensemble mean' or variable == 'bgws' and model == 'OBS' or variable == 'bgws' and model == 'ERA5':
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
    elif steps > 1 and steps != 2.5:
        cbar_ticklabels = [f"{tick:.0f}" if abs(tick) > 1e-10 else "0" for tick in cbar_ticks] 
    else:
        cbar_ticklabels = [f"{tick}" for tick in cbar_ticks]

    # Set the ticks and labels on the colorbar
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels)

# ========== Analytical Functions ==========

def subdivide_bgws(ds_dict_historical, ds_dict_change):
    ds_historical = ds_dict_historical['12 model ensemble mean'].bgws_tran_mean
    ds_change = ds_dict_change["ssp370_ff-historical"]['12 model ensemble mean'].bgws_tran_mean
    
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
        #filename = f'{model}_{variable}_map_rel.{filetype}'
        filename = f'{model}_{variable}_map.{filetype}'
        col_uti.save_fig(fig, filepath, filename, dpi=dpi)
        print(f"Figure saved under {filepath}{filename}")
    else:
        print("Figure not saved! If you want to save the figure change filepath='your filepath/' in the function call.")


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
    rect = add_box(fig, 0.1375, 0.191, 0.175, 0.228, "round,pad=0.004", 1, 'gray', 'white', 0)
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

        # Side badges (slightly smaller font than the cbar label fontsize=20)
        add_side_badges(fig, cbar_ax, font_size=16, arrow_lw=3.0)

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
        print("Figure not saved! If you want to save the figure change filepath='your filepath/' in the function call.")


def add_side_badges(fig, cbar_ax, *,
                    box_w=0.034, box_h=0.03, pad=0.004,
                    face_green=(0.88, 0.96, 0.88), edge_green=(0.05, 0.55, 0.10),
                    face_blue =(0.88, 0.92, 0.98), edge_blue =(0.10, 0.35, 0.80),
                    font_size=16, arrow_lw=4.0, head_size=24):
    """
    Add left (green) and right (blue) label boxes next to a colorbar axis.
    Each box contains a vertical arrow on the left (colored), and 'Green\nWater' / 'Blue\nWater' text.
    Arrow spans from the top of 'Green' to the bottom of 'Water'.
    """
    cb_pos   = cbar_ax.get_position(fig)
    y_center = cb_pos.y0 + cb_pos.height / 2.0
    y0 = y_center - box_h / 2.0 - 0.005  # apply shift here

    left_x0  = cb_pos.x0 - pad - box_w
    right_x0 = cb_pos.x1 + pad

    # Create the two small axes
    left_ax  = fig.add_axes([left_x0,  y0, box_w, box_h],  zorder=5)
    right_ax = fig.add_axes([right_x0, y0, box_w, box_h],  zorder=5)

    # Style helpers
    def style_ax(ax, face, edge):
        ax.set_facecolor(face)
        for sp in ax.spines.values():
            sp.set_edgecolor(edge); sp.set_linewidth(2.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        return edge  # return edge color for arrow/text

    green_col = style_ax(left_ax,  face_green, edge_green)
    blue_col  = style_ax(right_ax, face_blue,  edge_blue)

    # Common layout inside the small box (axes coords)
    # Place text centered; two lines. Leave ~0.18 margin left for the arrow.
    text_x   = 0.62
    text_y   = 0.50
    line_sp  = 0.92

    # Arrow geometry (vertical, left of text). Arrow points upward.
    # Tail near bottom of "Water", head near top of "Green".
    arrow_x  = 0.21
    y_head   = 0.78   # near top of 'Green'
    y_tail   = 0.28   # near bottom of 'Water'

    # LEFT badge (Green)
    left_ax.text(text_x, text_y, "Green\nWater",
                 ha="center", va="center", fontsize=font_size,
                 linespacing=line_sp, color=green_col)
    left_ax.annotate("",
        xy=(arrow_x, y_head), xytext=(arrow_x, y_tail),
        xycoords="axes fraction",
        arrowprops=dict(
        arrowstyle='-|>',
        mutation_scale=head_size,   # ↑ head size (try 20–36)
        lw=arrow_lw,              # ↑ shaft stroke
        color=green_col,
        shrinkA=0, shrinkB=0
    ))

    # RIGHT badge (Blue)
    right_ax.text(text_x, text_y, "Blue\nWater",
                  ha="center", va="center", fontsize=font_size,
                  linespacing=line_sp, color=blue_col)
    right_ax.annotate("",
        xy=(arrow_x, y_head), xytext=(arrow_x, y_tail),
        xycoords="axes fraction",
        arrowprops=dict(
        arrowstyle='-|>',
        mutation_scale=head_size,   # ↑ head size (try 20–36)
        lw=arrow_lw,              # ↑ shaft stroke
        color=blue_col,
        shrinkA=0, shrinkB=0
    ))

    return left_ax, right_ax


def plot_bgws_change_with_contours(
    ds_dict_change,
    ds_dict_historical,
    model,
    period,
    dpi=300,
    filetype='pdf',
    filepath=None,
    contour_step=20
):
    """
    Plot BGWS change with colormap scaling from col_map_limits,
    and overlay historical BGWS contours.

    Parameters
    ----------
    ds_dict_change : dict
        Dictionary containing BGWS change datasets.
    ds_dict_historical : dict
        Dictionary containing historical BGWS datasets.
    model : str
        Model name to plot (e.g. "Ensemble mean").
    period : str
        Experiment/period for colormap scaling.
    contour_step : int
        Step size for historical contours (e.g. 10 or 20).
    """

    # ---- setup ----
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Drop 'member_id' if exists
    ds_change = {k: v.drop('member_id', errors='ignore') for k, v in ds_dict_change.items()}
    ds_hist   = {k: v.drop('member_id', errors='ignore') for k, v in ds_dict_historical.items()}

    # Get datasets
    dsc = ds_change.get(model)
    dsh = ds_hist.get(model)
    if dsc is None or dsh is None:
        raise ValueError(f"Model '{model}' not found in both change and historical dictionaries.")

    # ---- colormap via your utility ----
    cmap_var = "bgws_ensmean"   # define a name for change maps
    vmin = col_map_limits[period][cmap_var]['vmin']
    vmax = col_map_limits[period][cmap_var]['vmax']
    steps = col_map_limits[period][cmap_var]['steps']/2
    cmap, cmap_norm = col_uti.create_colormap(cmap_var, period, vmin, vmax, steps)

    # ---- plot change ----
    img = dsc['bgws'].plot(
        ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, norm=cmap_norm,
        transform=ccrs.PlateCarree(), add_colorbar=False
    )

    # ---- add historical BGWS zero contour ----
    cs = ax.contour(
        dsh['bgws'].lon, dsh['bgws'].lat, dsh['bgws'],
        levels=[0],                   # only the zero line
        colors='black', linewidths=1.2,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(cs, inline=True, fontsize=12, fmt="0")


    # ---- map styling ----
    ax.coastlines()
    ax.tick_params(axis='both', which='major', labelsize=20)
    gl = ax.gridlines(draw_labels=True, color='black', alpha=0.1, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 30}
    gl.ylabel_style = {'size': 30}

    # Ensemble agreement stippling (optional)
    if model in ('Ensemble mean', 'Ensemble median') and period == 'ssp370-historical':
        plot_agreement_mask(ds_dict_change, dsh, model, 'bgws', ax)

    # ---- colorbar using the same function as your other maps ----
    cbar_global_map(img, fig, ax, period, model, 'bgws', vmin, vmax, steps)

    plt.show()

    # ---- save ----
    if filepath is not None:
        fname = f"{model}_BGWS_change_with_histcontours.{filetype}"
        try:
            col_uti.save_fig(fig, filepath, fname, dpi=dpi)
            print(f"Figure saved under {filepath}{fname}")
        except NameError:
            full = filepath + fname
            fig.savefig(full, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved under {full}")
    else:
        print("Figure not saved! If you want to save the figure change filepath='your filepath/' in the function call.")


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patches as mpatches

def plot_bgws_with_change_overlay(
    ds_dict,
    model,
    period,
    change_target='mrro',              # 'mrro' or 'tran' (or any var present as a change field)
    threshold=0.0,                     # ignore small-magnitude changes |Δ| <= threshold
    overlay_style='hatch',             # 'hatch' or 'dots'
    dpi=300,
    filetype='pdf',
    filepath=None
):
    """
    Plot BGWS and overlay grid-cell increase/decrease for a selected change variable.

    Parameters
    ----------
    ds_dict : dict
        Dictionary containing datasets for different models (flat or nested).
    model : str
        Model name to plot (e.g., 'Ensemble mean' or a specific model key).
    period : str
        Period key used for colormap limits (e.g., 'ssp370-historical').
    change_target : str
        Base variable whose change to overlay (e.g., 'mrro' or 'tran').
    change_var : str or None
        Name of the change variable in the dataset. If None, tries 'd_<target>',
        then '<target>_change', then '<target>_delta'.
    threshold : float
        Minimum absolute change to flag as increase/decrease.
    overlay_style : str
        'hatch' (contour hatching) or 'dots' (scatter points).
    dpi : int
        Figure DPI for saving.
    filetype : str
        Output file type ('pdf', 'png', ...).
    filepath : str or None
        Folder path to save into. If None, the figure is not saved.

    Returns
    -------
    None (shows the figure; optionally saves it)
    """

    # ---- helpers ----
    def _find_model_dataset(dct, key):
        # drop 'member_id' if present; support nested dicts
        cleaned = {}
        for k, v in dct.items():
            if isinstance(v, dict):
                cleaned[k] = {kk: vv.drop('member_id', errors='ignore') for kk, vv in v.items()}
            else:
                cleaned[k] = v.drop('member_id', errors='ignore')
        if key in cleaned:
            return cleaned[key]
        # try nested: experiments -> models
        for _, vv in cleaned.items():
            if isinstance(vv, dict) and key in vv:
                return vv[key]
        raise ValueError(f"Model '{key}' not found in ds_dict.")

    # ---- fetch dataset & variables ----
    ds = _find_model_dataset(ds_dict, model)
    if 'bgws' not in ds:
        raise ValueError("Dataset must contain 'bgws' to plot the base map.")

    # get colormap settings for BGWS (using your utilities)
    if model == 'Ensemble mean':
        cmap_key = 'bgws_ensmean'
    elif model == 'Ensemble std':
        cmap_key = 'bgws_ensstd'
    else:
        cmap_key = 'bgws'

    change_var = change_target

    vmin = col_map_limits[period][cmap_key]['vmin']
    vmax = col_map_limits[period][cmap_key]['vmax']
    steps = col_map_limits[period][cmap_key]['steps']
    cmap, cmap_norm = col_uti.create_colormap(cmap_key, period, vmin, vmax, steps)

    # resolve change variable (positive -> increase; negative -> decrease)
    dvar_name = change_target
    dvar = ds[dvar_name]

    # apply threshold for robustness
    inc_mask = xr.where(dvar > threshold, 1, np.nan)
    dec_mask = xr.where(dvar < -threshold, 1, np.nan)

    # grab coords (assumes 2D lat/lon or 1D lat/lon)
    lat = ds[ds['bgws'].dims[-2]] if 'lat' not in ds.coords else ds['lat']
    lon = ds[ds['bgws'].dims[-1]] if 'lon' not in ds.coords else ds['lon']
    # Broadcast if needed
    if lat.ndim == 1 and lon.ndim == 1:
        LON, LAT = np.meshgrid(lon.values, lat.values)
    else:
        LON, LAT = lon.values, lat.values

    # ---- figure & map ----
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # base BGWS field
    img = ds['bgws'].plot(
        ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
        transform=ccrs.PlateCarree(), add_colorbar=False
    )

    # coastlines & gridlines
    ax.coastlines()
    ax.tick_params(axis='both', which='major', labelsize=20)
    gl = ax.gridlines(draw_labels=True, color='black', alpha=0.1, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 30}
    gl.ylabel_style = {'size': 30}

    # optional ensemble agreement mask (reuse your function)
    #if model in ('Ensemble mean', 'Ensemble median') and period == 'ssp370-historical':
    #    plot_agreement_mask(ds_dict, ds, model, 'bgws', ax)

    # ---- overlay increase/decrease ----
    if overlay_style == 'hatch':
        # use contourf hatching; need numeric arrays with NaNs
        # increases: forward slashes; decreases: backslashes
        # NOTE: set facecolor='none' so only hatches are visible.
        # increases
        if np.isfinite(inc_mask).any():
            cs_inc = ax.contourf(
                LON, LAT, inc_mask.values,
                levels=[0.5, 1.5],
                colors='none',
                hatches=['///'],
                transform=ccrs.PlateCarree()
            )
        # decreases
        if np.isfinite(dec_mask).any():
            cs_dec = ax.contourf(
                LON, LAT, dec_mask.values,
                levels=[0.5, 1.5],
                colors='none',
                hatches=['\\\\\\'],
                transform=ccrs.PlateCarree()
            )

        # legend patches for hatches
        legend_handles = []
        legend_handles.append(mpatches.Patch(facecolor='none', hatch='///', label=f'{change_target} increase'))
        legend_handles.append(mpatches.Patch(facecolor='none', hatch='\\\\\\', label=f'{change_target} decrease'))
        ax.legend(handles=legend_handles, loc='lower left', frameon=True, fontsize=18)

    elif overlay_style == 'dots':
        # scatter dots at cell centres (black = increase, white edge = decrease)
        # tune 's' for density
        if np.isfinite(inc_mask).any():
            ax.scatter(
                LON[np.isfinite(inc_mask.values)], LAT[np.isfinite(inc_mask.values)],
                s=4, c='k', alpha=0.6, transform=ccrs.PlateCarree(), label=f'{change_target} increase'
            )
        if np.isfinite(dec_mask).any():
            ax.scatter(
                LON[np.isfinite(dec_mask.values)], LAT[np.isfinite(dec_mask.values)],
                s=4, facecolors='none', edgecolors='k', linewidths=0.5, alpha=0.6,
                transform=ccrs.PlateCarree(), label=f'{change_target} decrease'
            )
        ax.legend(loc='lower left', frameon=True, fontsize=18)
    else:
        raise ValueError("overlay_style must be 'hatch' or 'dots'.")

    # colorbar (reuse your utility)
    cbar_global_map(img, fig, ax, period, model, 'bgws', vmin, vmax, steps)

    plt.show()

    # save
    if filepath is not None:
        filename = f"{model}_bgws_{change_target}_overlay_{overlay_style}.{filetype}"
        col_uti.save_fig(fig, filepath, filename, dpi=dpi)
        print(f"Figure saved under {filepath}{filename}")
    else:
        print("Figure not saved! If you want to save the figure change filepath='your filepath/' in the function call.")

def plot_bgws_with_dual_agreement_overlay(
    ds_dict,
    model,
    period,
    change_target='mrro',      # 'mrro' or 'tran'
    threshold_bgws=0.0,        # mask |ΔBGWS| <= this
    threshold_var=0.0,         # mask |Δvar| <= this
    dpi=300,
    filetype='pdf',
    filepath=None
):
    """
    Plot BGWS and overlay colored scatters where BOTH BGWS and the selected variable
    have robust agreement in the sign of change.

    Parameters
    ----------
    ds_dict : dict
        Dictionary of datasets for models.
    model : str
        Model key (e.g. "Ensemble mean").
    period : str
        Experiment/period name.
    change_target : str
        Either 'mrro' (runoff) or 'tran' (transpiration).
    threshold_bgws : float
        Mask |ΔBGWS| <= threshold.
    threshold_var : float
        Mask |Δvar| <= threshold.
    """

    # ---- helpers ----
    def _find_model_dataset(dct, key):
        cleaned = {}
        for k, v in dct.items():
            if isinstance(v, dict):
                cleaned[k] = {kk: vv.drop('member_id', errors='ignore') for kk, vv in v.items()}
            else:
                cleaned[k] = v.drop('member_id', errors='ignore')
        if key in cleaned:
            return cleaned[key]
        for _, vv in cleaned.items():
            if isinstance(vv, dict) and key in vv:
                return vv[key]
        raise ValueError(f"Model '{key}' not found.")

    # ---- dataset & variables ----
    ds = _find_model_dataset(ds_dict, model)
    if 'bgws' not in ds:
        raise ValueError("Dataset must contain 'bgws'.")

    dB = ds['bgws']
    dV = ds[change_target]

    # sign masks
    sB = xr.where(dB >  threshold_bgws,  1, xr.where(dB < -threshold_bgws, -1, np.nan))
    sV = xr.where(dV >  threshold_var,   1, xr.where(dV < -threshold_var, -1, np.nan))
    valid = np.isfinite(sB) & np.isfinite(sV)

    # ---- categories ----
    # Only keep cells where both valid
    inc_mask = (sV == 1) & valid
    dec_mask = (sV == -1) & valid

    # coords
    lat = ds['lat'] if 'lat' in ds.coords else ds[dB.dims[-2]]
    lon = ds['lon'] if 'lon' in ds.coords else ds[dB.dims[-1]]
    if lat.ndim == 1 and lon.ndim == 1:
        LON, LAT = np.meshgrid(lon.values, lat.values)
    else:
        LON, LAT = lon.values, lat.values

    # ---- figure ----
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # base BGWS
    cmap_key = 'bgws_ensmean' if model == 'Ensemble mean' else 'bgws'
    vmin = col_map_limits[period][cmap_key]['vmin']
    vmax = col_map_limits[period][cmap_key]['vmax']
    steps = col_map_limits[period][cmap_key]['steps']
    cmap, cmap_norm = col_uti.create_colormap(cmap_key, period, vmin, vmax, steps)

    img = dB.plot(
        ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
        transform=ccrs.PlateCarree(), add_colorbar=False
    )

    # coastlines/gridlines
    ax.coastlines()
    ax.tick_params(axis='both', which='major', labelsize=20)
    gl = ax.gridlines(draw_labels=True, color='black', alpha=0.1, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 30}
    gl.ylabel_style = {'size': 30}

    # ---- overlay scatters ----
    col_inc = "red"

    ax.scatter(
        LON[inc_mask.values], LAT[inc_mask.values],
        s=3, c=col_inc, alpha=0.2,
        transform=ccrs.PlateCarree(),
        label=f"{change_target} increase"
    )

    # legend
    ax.legend(loc='lower left', frameon=True, fontsize=18)

    # colorbar
    cbar_global_map(img, fig, ax, period, model, 'bgws', vmin, vmax, steps)

    plt.show()

    # save
    if filepath is not None:
        filename = f"{model}_bgws_{change_target}_dualoverlay.{filetype}"
        col_uti.save_fig(fig, filepath, filename, dpi=dpi)
        print(f"Figure saved under {filepath}{filename}")



import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple

def plot_bgws_quadrant_map(
    ds_dict,
    model,
    period,
    change_var='mrro',               # 'mrro' or 'tran' (or other var)
    threshold_bgws=0.0,              # mask |ΔBGWS| <= this
    threshold_change=0.0,            # mask |Δvar|  <= this
    magnitude_source='var',          # 'bgws' | 'var' | 'joint'
    alpha_low=0.60,                  # low magnitude alpha
    alpha_med=0.8,                  # medium magnitude alpha
    alpha_high=1.00,                 # high magnitude alpha
    dpi=300,
    filetype='pdf',
    filepath=None
):
    """
    Four-quadrant categorical map with magnitude encoded by alpha (lightness).
    - NaNs/threshold-masked cells -> fully transparent; background set to white.
    - Joint-sign hues use your exact paper colors (no hue shifts).
    - Magnitude legend is neutral grey (not blue) and merged into the main legend.
    - Ensemble-agreement stippling is suppressed where thresholds mask changes.
    """

    # ---------- helpers ----------
    def _drop_member_id(dct):
        out = {}
        for k, v in dct.items():
            if isinstance(v, dict):
                out[k] = {kk: vv.drop('member_id', errors='ignore') for kk, vv in v.items()}
            else:
                out[k] = v.drop('member_id', errors='ignore')
        return out

    def _get_ds(dct, key):
        dct = _drop_member_id(dct)
        if key in dct and isinstance(dct[key], xr.Dataset):
            return dct[key]
        for _, vv in dct.items():
            if isinstance(vv, dict) and key in vv:
                return vv[key]
        raise ValueError(f"Model '{key}' not found.")

    def _rgb01(t):
        r, g, b = t
        return (r/255.0, g/255.0, b/255.0)

    def _rgba(rgb, a):
        return (*_rgb01(rgb), float(a))

    # ---------- dataset & variables ----------
    ds = _get_ds(ds_dict, model)
   
    dB = ds['bgws']
    dV = ds[change_var]

    # ---------- sign thresholds & valid mask ----------
    sB = xr.where(dB >  threshold_bgws,  1, xr.where(dB < -threshold_bgws, -1, np.nan))
    sV = xr.where(dV >  threshold_change, 1, xr.where(dV < -threshold_change, -1, np.nan))
    valid = np.isfinite(sB) & np.isfinite(sV)

    # Quadrants: 0=++, 1=+-, 2=-+, 3=--
    quad = xr.full_like(dB, np.nan, dtype=float)
    quad = xr.where(valid & (sB== 1) & (sV== 1), 0, quad)  # ++
    quad = xr.where(valid & (sB== 1) & (sV==-1), 1, quad)  # +-
    quad = xr.where(valid & (sB==-1) & (sV== 1), 2, quad)  # -+
    quad = xr.where(valid & (sB==-1) & (sV==-1), 3, quad)  # --

    # ---------- magnitude ----------
    absB = np.abs(dB)
    absV = np.abs(dV)

    if magnitude_source == 'bgws':
        mag = absB.where(valid)
    elif magnitude_source == 'var':
        mag = absV.where(valid)
    elif magnitude_source == 'joint':
        valsB = absB.values[np.isfinite(absB.values)]
        valsV = absV.values[np.isfinite(absV.values)]
        scaleB = np.nanpercentile(valsB, 90) if valsB.size else 1.0
        scaleV = np.nanpercentile(valsV, 90) if valsV.size else 1.0
        scaleB = scaleB if scaleB > 0 else 1.0
        scaleV = scaleV if scaleV > 0 else 1.0
        mag = xr.ufuncs.maximum(absB/scaleB, absV/scaleV).where(valid)
    else:
        raise ValueError("magnitude_source must be 'bgws', 'var', or 'joint'.")

  
    ## mag_strategy == 'quantile':
    vals = mag.values[np.isfinite(mag.values)]
    if vals.size == 0:
        edges = np.array([0, 1, 2, 3], dtype=float)
    else:
        q1, q2 = np.nanpercentile(vals, [33.33, 66.67])
        edges = np.array([vals.min(), q1, q2, vals.max()], dtype=float)
        eps = 1e-12
        edges[1] = max(edges[1], edges[0] + eps)
        edges[2] = max(edges[2], edges[1] + eps)
        edges[3] = max(edges[3], edges[2] + eps)
   
    mag_bin = xr.full_like(mag, np.nan, dtype=float)
    mag_bin = xr.where((mag >= edges[0]) & (mag < edges[1]), 0, mag_bin)  # low
    mag_bin = xr.where((mag >= edges[1]) & (mag < edges[2]), 1, mag_bin)  # medium
    mag_bin = xr.where((mag >= edges[2]) & (mag <= edges[3]), 2, mag_bin) # high

    # Combined class index: 0..11 (4 hues × 3 alpha levels)
    combo = xr.full_like(quad, np.nan, dtype=float)
    for q in range(4):
        for m in range(3):
            combo = xr.where((quad == q) & (mag_bin == m), q*3 + m, combo)

    # ---------- EXACT paper colours & alpha mapping ----------
    # High magnitude uses base “over/under” colour at full opacity.
    # Medium and low use the same RGB with alpha_med / alpha_low.

    # BLUE (ΔBGWS↑, Δvar↑): use OVER blue
    BLUE_OVER  = (40, 125, 210)

    # VIOLET (ΔBGWS↑, Δvar↓): use OVER violet
    VIO_OVER   = (130, 79, 158)

    # GREEN (ΔBGWS↓, Δvar↑): use UNDER green
    GREEN_UNDER = (15, 115, 15)

    # BROWN (ΔBGWS↓, Δvar↓): use UNDER brown
    BROWN_UNDER = (150, 140, 100)

    # Build RGBA list: order ++, +-, -+, --
    cat_rgbs = [BLUE_OVER, VIO_OVER, GREEN_UNDER, BROWN_UNDER]
    alphas = [alpha_low, alpha_med, alpha_high]  # low, medium, high

    colors = []
    for rgb in cat_rgbs:
        for a in alphas:
            colors.append(_rgba(rgb, a))

    cmap = ListedColormap(colors, name="quad_mag_alpha")
    # Make NaNs fully transparent so background (white) shows through
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    # ---------- plot ----------
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_facecolor('white')  # masked/NaN areas appear white

    combo.plot(ax=ax, cmap=cmap, add_colorbar=False, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.tick_params(axis='both', which='major', labelsize=20)
    gl = ax.gridlines(draw_labels=True, color='black', alpha=0.1, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 30}
    gl.ylabel_style = {'size': 30}
    
    # ---------- ensemble agreement overlay (respect thresholds) ----------
    if model in ('Ensemble mean', 'Ensemble median') and period == 'ssp370-historical':
        try:
            # 1) mask the dataset you pass as 'ds'
            ds_masked = ds.copy(deep=False)
            ds_masked['bgws'] = ds_masked['bgws'].where(valid)
    
            # 2) build a ds_dict masked on the SAME valid grid cells
            def _mask_ds_dict(dct, var, mask):
                out = {}
                for k, v in dct.items():
                    if isinstance(v, dict):  # nested (e.g., experiments -> models)
                        out[k] = {}
                        for kk, vv in v.items():
                            vv2 = vv.copy(deep=False)
                            if var in vv2:
                                vv2[var] = vv2[var].where(mask)
                            out[k][kk] = vv2
                    else:  # flat (model -> dataset)
                        vv2 = v.copy(deep=False)
                        if var in vv2:
                            vv2[var] = vv2[var].where(mask)
                        out[k] = vv2
                return out
    
            ds_dict_masked = _mask_ds_dict(ds_dict, 'bgws', valid)
    
            # 3)  Compute masks (mirrors your agreement logic; respects thresholds)
            masks = dual_agreement_masks(
                ds_dict_masked, model,
                var_bgws_change='bgws',
                var_flux_change=change_var,
                valid_mask=valid,            # ensures no stippling below thresholds
                min_majority=0.70,
                min_participation=0.54545455
            )

            # 2) Plot ONE overlay of your choice
            #    'conservative' -> union (either uncertain)
            #    'stringent'    -> intersection (both uncertain)
            #    'bivariate'    -> 3 symbols (BGWS / flux / both)
            leg_unc = plot_dual_agreement_overlay(
                ax=ax,
                ds=ds,
                masks=masks,
                mode="conservative",             # or 'stringent' or 'bivariate'
                change_target_label="runoff",    # or "transpiration"
                legend=True,
                legend_loc="lower right",
                fontsize=22,
            )
    
        except Exception as e:
            print(f"[agreement overlay skipped] {e}")

   # --- category legend (stacked vertically) ---
    cat_patches = [
        mpatches.Patch(color=_rgba(BLUE_OVER,  alpha_med),  label="Δ BGWS ↑ and Δ Runoff ↑"  if change_var.lower() in ('mrro')
                       else "Δ BGWS ↑ and Δ Transpiration ↑"),
        mpatches.Patch(color=_rgba(VIO_OVER,   alpha_med),  label="Δ BGWS ↑ and Δ Runoff ↓"  if change_var.lower() in ('mrro')
                       else "Δ BGWS ↑ and Δ Transpiration ↓"),
        mpatches.Patch(color=_rgba(GREEN_UNDER,alpha_med),  label="Δ BGWS ↓ and Δ Runoff ↑"  if change_var.lower() in ('mrro')
                       else "Δ BGWS ↓ and Δ Transpiration ↑"),
        mpatches.Patch(color=_rgba(BROWN_UNDER,alpha_med),  label="Δ BGWS ↓ and Δ Runoff ↓"  if change_var.lower() in ('mrro')
                       else "Δ BGWS ↓ and Δ Transpiration ↓"),
    ]
    
    # --- magnitude chips (one row in legend via HandlerTuple) ---
    GREY = (128/255, 128/255, 128/255)
    mag_patches = (
        mpatches.Patch(color=(GREY[0], GREY[1], GREY[2], alpha_low)),
        mpatches.Patch(color=(GREY[0], GREY[1], GREY[2], alpha_med)),
        mpatches.Patch(color=(GREY[0], GREY[1], GREY[2], alpha_high)),
    )
    
    handles = [*cat_patches, mag_patches]
    labels  = [
        cat_patches[0].get_label(),
        cat_patches[1].get_label(),
        cat_patches[2].get_label(),
        cat_patches[3].get_label(),
        "Joint magnitude (tertiles)",
    ]
    
    leg = ax.legend(
        handles=handles,
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.8)},  # <-- draws tuple side-by-side
        ncol=1,                     # categories stacked; magnitude is one row
        loc='lower left',
        frameon=True,
        facecolor='white',
        framealpha=0.9,
        fontsize=16,                   # ← smaller text
        prop={'size': 16},             # ← forces smaller text reliably
        handlelength=2.4,
        handletextpad=0.8,
        borderpad=0.8,
        labelspacing=0.6,  
    )

    if leg_unc is not None:
        ax.add_artist(leg_unc) 


    plt.show()

    # ---------- save ----------
    if filepath is not None:
        fname = f"{model}_quadrant_mag_bgws_{change_var}_{magnitude_source}.{filetype}"
        try:
            col_uti.save_fig(fig, filepath, fname, dpi=dpi)
            print(f"Figure saved under {filepath}{fname}")
        except NameError:
            full = filepath + fname
            fig.savefig(full, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved under {full}")
    else:
        print("Figure not saved! If you want to save the figure change filepath='your filepath/' in the function call.")

import numpy as np
import xarray as xr
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

def _collect_member_fields(ds_dict_in, var_name, exclude_key):
    """Gather per-model DataArrays for var_name, skipping 'Ensemble*' and the current model key."""
    fields = []
    for k, v in ds_dict_in.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if ("Ensemble" in str(kk)) or (kk == exclude_key):
                    continue
                if isinstance(vv, xr.Dataset) and (var_name in vv):
                    fields.append(vv[var_name].drop('member_id', errors='ignore'))
        else:
            if ("Ensemble" in str(k)) or (k == exclude_key):
                continue
            if isinstance(v, xr.Dataset) and (var_name in v):
                fields.append(v[var_name].drop('member_id', errors='ignore'))
    return fields

def compute_low_agreement_mask(
    ds_dict_in,
    var_name,
    current_model,
    min_majority=0.70,
    min_participation=0.54545455,
):
    """
    Boolean mask (True = low sign-agreement) for 'var_name', mirroring your existing logic:
    - Compare fraction of models with positive vs negative Δ
    - Treat cells with insufficient participation (<~0.545) as NaN (no stipple)
    - Low agreement if majority < min_majority (e.g., 0.70)
    """
    fields = _collect_member_fields(ds_dict_in, var_name, current_model)
    if not fields:
        raise ValueError(f"No member fields found for '{var_name}' to compute agreement.")
    stack = xr.concat(fields, dim='model')

    pos = (stack > 0).mean(dim='model', skipna=True)
    neg = (stack < 0).mean(dim='model', skipna=True)
    agree = xr.where(pos > neg, pos, neg)
    agree = agree.where(agree >= min_participation)  # keep NaN where too few members

    low = agree < min_majority
    return xr.where(np.isfinite(agree), low, False)

def dual_agreement_masks(
    ds_dict_masked,            # dict where BOTH change vars were already masked by your thresholds
    model,                     # current ensemble/model key
    var_bgws_change,           # e.g., 'd_bgws'
    var_flux_change,           # e.g., 'd_mrro' or 'd_tran'
    valid_mask=None,           # optional additional mask (e.g., |Δ| thresholds for BOTH vars)
    min_majority=0.70,
    min_participation=0.54545455,
):
    """Return dict of low-agreement masks for BGWS, flux, union, intersection."""
    low_bgws = compute_low_agreement_mask(
        ds_dict_masked, var_bgws_change, model,
        min_majority=min_majority, min_participation=min_participation
    )
    low_flux = compute_low_agreement_mask(
        ds_dict_masked, var_flux_change, model,
        min_majority=min_majority, min_participation=min_participation
    )
    if valid_mask is not None:
        low_bgws = (low_bgws & valid_mask).fillna(False)
        low_flux = (low_flux & valid_mask).fillna(False)

    return {
        "bgws":        low_bgws,
        "flux":        low_flux,
        "union":       (low_bgws | low_flux).fillna(False),
        "intersection":(low_bgws & low_flux).fillna(False),
    }

def plot_dual_agreement_overlay(
    ax,
    ds,                          # dataset for lon/lat
    masks,                       # output of dual_agreement_masks(...)
    mode="conservative",         # 'conservative', 'stringent', or 'bivariate'
    change_target_label="runoff",# used only in legend text for bivariate mode
    lon_name="lon",
    lat_name="lat",
    legend=True,
    legend_loc="lower right",
    fontsize=30,
):
    """
    Draw one overlay:
      - conservative: union of low-agreement (either var uncertain) -> grey diamonds
      - stringent:    intersection (both uncertain) -> black dots
      - bivariate:    three symbols (BGWS-only, flux-only, both)
    Returns the legend handle so you can re-add it after drawing your main legend.
    """
    # Prepare lon/lat grids
    if (lon_name in ds.coords) and (lat_name in ds.coords) and ds[lon_name].ndim == 1 and ds[lat_name].ndim == 1:
        LON, LAT = np.meshgrid(ds[lon_name].values, ds[lat_name].values)
    else:
        sample = next(iter(masks.values()))
        lat_dim, lon_dim = sample.dims[-2], sample.dims[-1]
        LAT = ds[lat_dim].values if ds[lat_dim].ndim == 2 else np.meshgrid(ds[lon_dim].values, ds[lat_dim].values)[1]
        LON = ds[lon_dim].values if ds[lon_dim].ndim == 2 else np.meshgrid(ds[lon_dim].values, ds[lat_dim].values)[0]

    leg_handle = None

    if mode == "conservative":
        m = masks["union"]
        ii = np.where(m.values)
        if ii[0].size:
            ax.scatter(LON[ii], LAT[ii], color='grey', marker='D', s=1.5, alpha=1.0,
                       transform=ccrs.PlateCarree(), zorder=11)
            if legend:
                h = scatter_legend(
                    ax, label="Low Ensemble Agreement",
                    facecolor="white", edgecolor="black", lw=0.7
                )
                
                # 2. Choose your offsets – same ones you used before
                handler = ScatterBoxHandler(
                    xdescent_offset_box = -2.9,     # example values
                    ydescent_offset_box = -8,
                    box_width_offset     = 1.2,  # ← smaller box
                    box_height_offset    = 1.8,  # ← smaller box
                    scatter_spacing_x_offset = 2,
                    scatter_spacing_y_offset = 2,
                    scatter_vertical_offset  = -4,
                    scatter_horizontal_offset= 0
                )
                
                # 3. Attach legend with handler map
                leg_handle = ax.legend(
                    handles=[h],
                    handler_map={type(h): handler},
                    fontsize=20,                        # smaller font
                    loc=legend_loc,
                    frameon=True,
                    facecolor="white",
                    edgecolor="gray"
                )


    elif mode == "stringent":
        m = masks["intersection"]
        ii = np.where(m.values)
        if ii[0].size:
            ax.scatter(LON[ii], LAT[ii], color='k', marker='.', s=7, alpha=0.8,
                       transform=ccrs.PlateCarree(), zorder=11)
            if legend:
                h = scatter_legend(ax, label="Low Ensemble Agreement",
                                   facecolor="black", edgecolor="black", lw=0.5)
                leg_handle = ax.legend(handles=[h],
                                       handler_map={type(h): ScatterBoxHandler()},
                                       fontsize=fontsize, loc=legend_loc, frameon=True)

    elif mode == "bivariate":
        # BGWS-only
        only_bgws = (masks["bgws"] & ~masks["flux"])
        ii = np.where(only_bgws.values)
        if ii[0].size:
            ax.scatter(LON[ii], LAT[ii], c='0.25', marker='D', s=6, alpha=0.8,
                       transform=ccrs.PlateCarree(), zorder=11, linewidths=0)

        # Flux-only
        only_flux = (~masks["bgws"] & masks["flux"])
        ii = np.where(only_flux.values)
        if ii[0].size:
            ax.scatter(LON[ii], LAT[ii], facecolors='none', edgecolors='0.25', marker='s', s=7, alpha=0.9,
                       transform=ccrs.PlateCarree(), zorder=11, linewidths=0.7)

        # Both
        both = (masks["bgws"] & masks["flux"])
        ii = np.where(both.values)
        if ii[0].size:
            ax.scatter(LON[ii], LAT[ii], c='k', marker='.', s=8, alpha=0.85,
                       transform=ccrs.PlateCarree(), zorder=12, linewidths=0)

        if legend:
            # Three entries using your scatter handle for a consistent legend style
            h_bgws = scatter_legend(ax, label="Low Ensemble Agreement: BGWS",
                                    facecolor="0.3", edgecolor="0.3", lw=0.5)
            h_flux = scatter_legend(ax, label=f"Low Ensemble Agreement: {change_target_label}",
                                    facecolor="white", edgecolor="0.3", lw=0.7)
            h_both = scatter_legend(ax, label="Low Ensemble Agreement: Both",
                                    facecolor="black", edgecolor="black", lw=0.5)
            leg_handle = ax.legend(handles=[h_bgws, h_flux, h_both],
                                   handler_map={type(h_bgws): ScatterBoxHandler()},
                                   fontsize=fontsize, loc=legend_loc, frameon=True)
    else:
        raise ValueError("mode must be 'conservative', 'stringent', or 'bivariate'.")

    return leg_handle



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

    ds_historical = ds_dict['historical']['12 model ensemble mean'].bgws_tran_mean
    ds_ssp370 = ds_dict['ssp370_ff']['12 model ensemble mean'].bgws_tran_mean

    # Compute the sign of each dataset
    sign_historical = np.sign(ds_historical)
    sign_ssp370 = np.sign(ds_ssp370)
    
    # Create a mask where the signs differ
    sign_change_mask = sign_historical != sign_ssp370
    
    # Apply the mask to ds_ssp370 (keep only grid cells with sign changes)
    ds_ssp370_masked = ds_ssp370.where(sign_change_mask)

    # Get colormap
    vmin = col_map_limits['ssp370_ff-historical']['bgws_tran_mean']['vmin']
    vmax = col_map_limits['ssp370_ff-historical']['bgws_tran_mean']['vmax']
    steps = col_map_limits['ssp370_ff-historical']['bgws_tran_mean']['steps']
    cmap, cmap_norm = col_uti.create_colormap('bgws_ensmean', 'ssp370_ff-historical', vmin, vmax, steps)

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
    for model in ds_dict_change['ssp370_ff-historical']:
        # Access the original dataset
        ds_change = ds_dict_change['ssp370_ff-historical'][model].bgws_tran_mean
        
        # Apply the mask (using the same sign change mask logic)
        ds_masked = ds_change.where(sign_change_mask)
        
        # Ensure the structure matches the original dictionary
        if 'ssp370_ff-historical' not in ds_dict_masked:
            ds_dict_masked['ssp370_ff-historical'] = {}
        if model not in ds_dict_masked['ssp370_ff-historical']:
            ds_dict_masked['ssp370_ff-historical'][model] = {}
        ds_dict_masked['ssp370_ff-historical'][model]['bgws'] = ds_masked
            
    # Add colorbar
    cbar_global_map(img, fig, ax_main, 'ssp370_ff', '12 model ensemble mean', 'bgws_tran_mean', vmin, vmax, steps)

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
    if variable == 'bgws':
        lims = [-100, 100]
    elif variable == 'tran_pr' or  variable == 'mrro_pr':
        lims = [0, 100]
    elif variable == 'pr':
        lims = [0, 15]
    elif variable == 'mrro':
        lims = [0, 7.5]
    elif variable == 'tran':
        lims = [0, 3]
    else:
        lims = [0, 100]

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
        ref_display = 'ERA5 land'
    else:
        ref_display = 'Observation-based'

    if model_label == 'Ensemble Mean':
        model_display= 'CMIP6 ensemble mean'
    else:
        model_display = model_label

    if variable == 'bgws':
        display_var = 'BGWS'
    elif variable == 'tran_pr':
        display_var = 'Transpiration / Precipitation'
    elif variable == 'mrro_pr':
        display_var = 'Runoff / Precipitation'
    elif variable == 'pr':
        display_var = 'Precipitation'
    elif variable == 'mrro':
        display_var = 'Runoff'
    elif variable == 'tran':
        display_var = 'Transpiration'
    else:
        display_var = variable

    if variable == 'pr':
        unit = 'mm'
    else: 
        unit = '%'
        
    # Add labels and title
    plt.xlabel(f"{ref_display} {display_var} [{unit}]", fontsize=20)
    plt.ylabel(f"{model_display} {display_var} [{unit}]", fontsize=20)

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
    rx5day_ratio = rx5day_ratio.flatten() #* 100
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
    if var_name == 'bgws_tran_mean':
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
    skip_models = {'OBS', 'ERA5_land', '12 model ensemble std', '12 model ensemble median'}

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
                historical_mean_rounded = round(historical_mean.values.item(), 2)

                # Calculate the weighted mean for SSP3-7.0 data
                future_mean = comp_stats.compute_spatial_statistic(ds_dict['ssp370_ff'][model][var], statistic='mean')
                future_mean_rounded = round(future_mean.values.item(), 2)

                # Calculate the change between future and historical
                change_mean = comp_stats.compute_spatial_statistic(ds_dict_change['ssp370_ff-historical'][model][var], statistic='mean')
                change_mean_rounded = round(change_mean.values.item(), 2)

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


def percentage_changes_table(ds_dict_current, ds_dict_change, variable='bgws_tran_mean', filepath=None):
    """
    Save the area percentage changes data to a CSV file for each model, organized by historical state and change.

    Parameters:
    ds_dict_current (dict): Dictionary containing current datasets for different models.
    ds_dict_change (dict): Dictionary containing change datasets for different models.
    variable (str): Variable name to compute the changes.
    save_dir (str): Directory where the CSV file will be saved.
    """
    # Models to skip
    skip_models = {'OBS', 'ERA5_land', '12 model ensemble std', '12 model ensemble median'}
    
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

def flip_changes_table(ds_dict_current, ds_dict_future, variable='bgws_tran_mean', filepath=None):
    """
    Save the flip changes data to a CSV file for each model.

    Parameters:
    ds_dict_current (dict): Dictionary containing historical datasets for different models.
    ds_dict_future (dict): Dictionary containing future datasets for different models.
    variable (str): Variable name to compute the changes.
    filepath (str): Directory where the CSV file will be saved.
    """
    # Models to skip
    skip_models = {'OBS', 'ERA5_land', '12 model ensemble std', '12 model ensemble median'}
    
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


import numpy as np
import xarray as xr
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


def _sign_nozero(da):
    """Return sign, but treat zeros as NaN so they do not count as agreement."""
    return xr.where(da > 0, 1, xr.where(da < 0, -1, np.nan))


def get_disagreement_with_both_refs_mask(da_ensmean, da_obs, da_era5):
    """
    True where ensemble-mean sign disagrees with BOTH OBS and ERA5-Land.
    Only evaluated where all three fields are finite and non-zero.
    """
    s_ens = _sign_nozero(da_ensmean)
    s_obs = _sign_nozero(da_obs)
    s_era = _sign_nozero(da_era5)

    valid = np.isfinite(s_ens) & np.isfinite(s_obs) & np.isfinite(s_era)

    disagree = valid & (s_ens != s_obs) & (s_ens != s_era)
    return disagree

def plot_boolean_scatter_mask(
    ax,
    mask,
    stipple_step=2,
    scatter_size=10,
    marker="o",
    facecolor="white",
    edgecolor="black",
    alpha=0.9,
    linewidth=0.5,
    zorder=30,
):
    """
    Plot a boolean mask as scatter points, subsampled every `stipple_step`.
    """
    if "lat" in mask.dims and "lon" in mask.dims:
        mask_sub = mask.isel(
            lat=slice(0, None, stipple_step),
            lon=slice(0, None, stipple_step),
        )
    else:
        dims = list(mask.dims)
        d1, d2 = dims[-2], dims[-1]
        mask_sub = mask.isel(
            {d1: slice(0, None, stipple_step), d2: slice(0, None, stipple_step)}
        )

    pts = mask_sub.where(mask_sub).stack(points=mask_sub.dims).dropna("points")

    if pts.sizes.get("points", 0) == 0:
        return

    x = pts["lon"].values
    y = pts["lat"].values

    ax.scatter(
        x, y,
        s=scatter_size,
        marker=marker,
        facecolors=facecolor,
        edgecolors=edgecolor,
        alpha=alpha,
        linewidths=linewidth,
        transform=ccrs.PlateCarree(),
        zorder=zorder,
        clip_on=True,
    )


def add_bgws_disagreement_legend(fig, cbar_ax, fontsize=14):
    """
    Add a figure-level legend for BGWS disagreement scatter.
    Place it to the right of the upper BGWS colorbar.
    """

    disagree_handle = scatter_legend(
        cbar_ax,
        label="Ensemble mean sign disagreement\nwith both reference datasets",
        facecolor="white",
        edgecolor="black",
        lw=0.8,
    )

    leg = fig.legend(
        handles=[disagree_handle],
        labels=[disagree_handle.get_label()],
        handler_map={
            type(disagree_handle): ScatterBoxHandler(
                ydescent_offset_box=-4,
                box_width_offset=1.2,
                box_height_offset=2,
                scatter_spacing_y_offset=1.5,
                scatter_spacing_x_offset=0.5,
                scatter_vertical_offset=-1,
                scatter_horizontal_offset=-4,
            )
        },
        loc="center left",
        bbox_to_anchor=(
            cbar_ax.get_position().x1-0.16,
            cbar_ax.get_position().y0 + cbar_ax.get_position().height / 2 + 0.08
        ),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=fontsize,
    )

    for t in leg.get_texts():
        t.set_fontweight("bold")

    return leg

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def figure1(ds_dict, ds_ref_mean, dpi=300, filetype="pdf", savepath=None):

    fig = plt.figure(figsize=(16, 13))

    outer_gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[1.25, 0.9],
        hspace=0.3
    )

    # --------------------
    # Datasets
    # --------------------
    ds_main = ds_dict["historical"]["12 model ensemble mean"].drop("member_id", errors="ignore")
    ds_obs = ds_ref_mean["historical"]["OBS"].drop("member_id", errors="ignore")
    ds_era = ds_ref_mean["historical"]["ERA5_land"].drop("member_id", errors="ignore")

    # --------------------
    # Regime inset dataset
    # --------------------
    regime = build_regime_map(ds_main)

    # --------------------
    # PANEL a
    # --------------------
    ax_a = fig.add_subplot(outer_gs[0, 0], projection=ccrs.PlateCarree())
    cbar_ax = fig.add_axes([0.23, 0.42, 0.54, 0.025])

    f1_upper_panel(
        ds_dict,
        "12 model ensemble mean",
        "bgws_tran_mean",
        "historical",
        ax=ax_a,
        fig=fig,
        cbar_ax=cbar_ax
    )

    ax_a.text(
        0.01, 0.99, "(a)",
        transform=ax_a.transAxes,
        fontsize=19,
        fontweight="bold",
        ha="left",
        va="top"
    )

    # BGWS disagreement with BOTH OBS and ERA5-Land
    bgws_disagree_mask = get_disagreement_with_both_refs_mask(
        ds_main["bgws_tran_mean"],
        ds_obs["bgws_tran_mean"],
        ds_era["bgws_tran_mean"],
    )

    plot_boolean_scatter_mask(
        ax_a,
        bgws_disagree_mask,
        stipple_step=2,     # use 3 if still too dense
        scatter_size=8,
        marker="o",
        facecolor="black",
        edgecolor="white",
        alpha=0.9,
        linewidth=0.4,
        zorder=35,
    )

    add_bgws_disagreement_legend(fig, cbar_ax, fontsize=14)

    # inset regime map
    add_regime_inset(
        fig,
        ax_a,
        regime,
        loc="lower left",
        width_frac=0.30,
        height_frac=0.42,
        title="Hydroclimatic regimes"
    )
    # --------------------
    # PANELS b, c
    # --------------------
    bottom_gs = outer_gs[1].subgridspec(1, 2, wspace=0.08)
    bottom_axes = [
        fig.add_subplot(bottom_gs[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(bottom_gs[0, 1], projection=ccrs.PlateCarree()),
    ]

    vars_bottom = ["r_over_p_mean", "et_over_p_mean"]
    labels = ["(b)", "(c)"]

    for i, (ax, var, lab) in enumerate(zip(bottom_axes, vars_bottom, labels)):
        f1_lower_panel(
            ds_dict,
            "12 model ensemble mean",
            var,
            "historical",
            ax=ax,
            fig=fig,
            show_ylabels=(i == 0)
        )

        ax.text(
            0.01, 0.87, lab,
            transform=ax.transAxes,
            fontsize=19,
            fontweight="bold",
            ha="left",
            va="bottom"
        )

    if savepath:
        fname = f"fig_1_{dpi}dpi_updated.{filetype}"
        col_uti.save_fig(fig, savepath, fname, dpi=dpi)
        print(f"Figure saved under {savepath}{fname}")

    plt.show()


def f1_upper_panel(ds_dict, model, variable, period, ax, fig, cbar_ax):
    ds_dict_cleaned = {
        name: ds.drop("member_id", errors="ignore")
        for name, ds in ds_dict[period].items()
    }
    ds = ds_dict_cleaned.get(model)

    if ds is None:
        raise ValueError(f"Model '{model}' not found.")

    if variable == "bgws_tran_mean" and model == "12 model ensemble mean":
        cmap_var = "bgws_ensmean"
    elif variable == "bgws_tran_mean" and model == "12 model ensemble std":
        cmap_var = "bgws_ensstd"
    else:
        cmap_var = variable

    vmin = col_map_limits[period][cmap_var]["vmin"]
    vmax = col_map_limits[period][cmap_var]["vmax"]
    steps = col_map_limits[period][cmap_var]["steps"]

    cmap, cmap_norm = col_uti.create_colormap(
        cmap_var, period, vmin, vmax, steps
    )

    img = ds[variable].plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        add_colorbar=False
    )

    ax.coastlines()
    for spine in ax.spines.values():
        spine.set_edgecolor("0.6")
        spine.set_linewidth(0.5)

    ax.tick_params(axis="both", which="major", labelsize=18)

    gl = ax.gridlines(
        draw_labels=True,
        linestyle="--",
        color="black",
        alpha=0.1
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 15}
    gl.ylabel_style = {"size": 15}

    cbar = fig.colorbar(
        img,
        cax=cbar_ax,
        orientation="horizontal",
        drawedges=True,
        extend="both"
    )

    var_label = col_uti.get_global_map_var_name(period, variable)
    cbar.set_label(var_label, fontsize=19, weight="bold", labelpad=10)

    ticks = np.arange(vmin, vmax + steps, steps)
    ticklabels = [f"{t:.0f}" for t in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=16)

    return img


def f1_lower_panel(ds_dict, model, variable, period, ax, fig, show_ylabels=True):
    ds_dict_cleaned = {
        name: ds.drop("member_id", errors="ignore")
        for name, ds in ds_dict[period].items()
    }
    ds = ds_dict_cleaned.get(model)

    if ds is None:
        raise ValueError(f"Model '{model}' not found.")

    cmap_var = variable
    vmin = col_map_limits[period][cmap_var]["vmin"]
    vmax = col_map_limits[period][cmap_var]["vmax"]
    steps = col_map_limits[period][cmap_var]["steps"]

    cmap, cmap_norm = col_uti.create_colormap(
        cmap_var, period, vmin, vmax, steps
    )

    # convert ratios to %
    ds_p = ds[variable] * 100
    
    img = ds_p.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        add_colorbar=False
    )

    ax.coastlines(color="black", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("0.6")
        spine.set_linewidth(0.5)

    ax.gridlines(draw_labels=False, linestyle="--", color="black", alpha=0.1)

    # remove +/-180 to avoid cramped labels
    xticks = [-120, -60, 0, 60, 120]
    yticks = [-40, 0, 40, 80]

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=15)

    if not show_ylabels:
        ax.set_yticks([], crs=ccrs.PlateCarree())
        ax.tick_params(left=False, labelleft=False)
        
    cbar = fig.colorbar(
        img,
        ax=ax,
        orientation="horizontal",
        pad=0.12,
        fraction=0.06,
        shrink=0.72,
        drawedges=True,
        extend="max"
    )

    label = col_uti.get_global_map_var_name(period, variable)
    cbar.set_label(label, fontsize=19, weight="bold")

    ticks = np.arange(vmin, vmax + (steps * 2), steps * 2)
    ticklabels = [f"{t:.1f}" if steps < 1 else f"{t:.0f}" for t in ticks]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=16)

    return img


def _zscore(da):
    return (da - da.mean(skipna=True)) / da.std(skipna=True)


def _tas_to_celsius(da):
    units = str(da.attrs.get("units", "")).lower()
    if "k" in units and "deg" not in units and "c" not in units:
        return da - 273.15

    # fallback heuristic
    try:
        m = float(da.mean().compute()) if hasattr(da.data, "compute") else float(da.mean())
    except Exception:
        m = np.nan

    if np.isfinite(m) and m > 100:
        return da - 273.15

    return da



def _zscore(da):
    return (da - da.mean(skipna=True)) / da.std(skipna=True)

def _zscore(da):
    mean = da.mean(skipna=True)
    std = da.std(skipna=True)
    if float(std) == 0 or not np.isfinite(std):
        return xr.zeros_like(da)
    return (da - mean) / std

def build_regime_map(
    ds,
    water_var="mrsos_mean",
    energy_var="tas_mean",          # can change to "rsds_mean" if needed
    seasonality_var="pr_seasonality",
):
    """
    Very simple regime classification based on three standardized indicators:

      1 = water-limited                -> low soil moisture
      2 = energy-limited               -> low temperature
      3 = pr-seasonality regulated     -> high precipitation seasonality

    Each grid cell is assigned to whichever score is strongest.
    """

    required = [water_var, energy_var, seasonality_var]
    missing = [v for v in required if v not in ds]
    if missing:
        raise ValueError(f"Missing variables in dataset: {missing}")

    water = ds[water_var]
    energy = ds[energy_var]
    seasonality = ds[seasonality_var]

    valid = np.isfinite(water) & np.isfinite(energy) & np.isfinite(seasonality)

    # simple standardized scores
    water_score = -_zscore(water)           # low mrsos -> stronger water limitation
    energy_score = -_zscore(energy)         # low tas -> stronger energy limitation
    seasonality_score = _zscore(seasonality)  # high seasonality -> stronger regime 3

    scores = xr.concat(
        [water_score, energy_score, seasonality_score],
        dim="regime_score"
    )

    # avoid argmax crashing on cells where all values are NaN
    scores = scores.where(valid, -np.inf)

    # 0,1,2 -> 1,2,3
    regime = scores.argmax(dim="regime_score") + 1
    regime = regime.where(valid).astype("float32")

    regime.name = "regime"
    regime.attrs["flag_values"] = [1, 2, 3]
    regime.attrs["flag_meanings"] = (
        "water_limited energy_limited precipitation_seasonality_regulated"
    )
    regime.attrs["description"] = (
        f"Simple regime map based on strongest standardized score among "
        f"low {water_var}, low {energy_var}, and high {seasonality_var}"
    )

    return regime

import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

def add_regime_inset(
    fig,
    parent_ax,
    regime,
    loc="lower left",

    # text block anchor
    width_frac=0.30,
    height_frac=0.18,
    shift_x=-0.025,
    shift_y=0.005,

    # text content
    title="Hydroclimatic regimes",
    title_y=1.2,
    label_y_top=1,
    label_spacing=0.118,
    box_x=0.08,
    text_x=0.18,
    box_w=0.08,
    box_h=0.07,
    fontsize_title=17,
    fontsize_labels=15,
    labels=("Energy-limited", "Water-limited", "Seasonality-regulated"),

    # fully independent map position (relative to parent_ax size)
    map_left_frac=-0.085,
    map_bottom_frac=0.012,
    map_width_frac=0.46,
    map_height_frac=0.28,
):
    """
    Add a regime inset with a text block and a fully independent map.

    Text block position is controlled by:
      loc, width_frac, height_frac, shift_x, shift_y

    Map position is controlled independently by:
      map_left_frac, map_bottom_frac, map_width_frac, map_height_frac

    All *_frac values for the map are relative to the parent_ax size.
    """

    pos = parent_ax.get_position()

    # -------------------------
    # Text block position
    # -------------------------
    text_w = pos.width * width_frac
    text_h = pos.height * height_frac

    margin_x = pos.width * 0.008
    margin_y = pos.height * -0.01

    if loc == "lower right":
        text_left = pos.x1 - text_w - margin_x
        text_bottom = pos.y0 + margin_y
    elif loc == "lower left":
        text_left = pos.x0 + margin_x
        text_bottom = pos.y0 + margin_y
    elif loc == "upper right":
        text_left = pos.x1 - text_w - margin_x
        text_bottom = pos.y1 - text_h - margin_y
    elif loc == "upper left":
        text_left = pos.x0 + margin_x
        text_bottom = pos.y1 - text_h - margin_y
    else:
        raise ValueError("loc must be one of: lower right, lower left, upper right, upper left")

    text_left += pos.width * shift_x
    text_bottom += pos.height * shift_y

    ax_text = fig.add_axes([text_left, text_bottom, text_w, text_h])
    ax_text.set_facecolor("none")
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    for spine in ax_text.spines.values():
        spine.set_visible(False)

    ax_text.text(
        0.5, title_y, title,
        ha="center", va="top",
        fontsize=fontsize_title, fontweight="bold"
    )

    alpha_regime = 0.8

    regime_colors = {
        1: (180/255, 160/255, 120/255, alpha_regime),  # water-limited
        2: (55/255, 140/255, 225/255, alpha_regime),   # energy-limited
        3: (0.54, 0.42, 0.65, alpha_regime),           # seasonality-regulated
    }
    
    legend_items = [
        ("Energy-limited", regime_colors[2]),
        ("Water-limited", regime_colors[1]),
        ("Seasonality-regulated", regime_colors[3]),
    ]
    
    label_y_positions = [label_y_top - i * label_spacing for i in range(len(legend_items))]
    
    for y, (lab, col) in zip(label_y_positions, legend_items):
        rect = mpatches.Rectangle(
            (box_x, y - box_h / 2),
            box_w, box_h,
            facecolor=col,
            edgecolor="0.2",
            linewidth=0.8,
            transform=ax_text.transAxes,
            clip_on=False,
        )
        ax_text.add_patch(rect)
    
        ax_text.text(
            text_x, y, lab,
            ha="left", va="center",
            fontsize=fontsize_labels
        )

    # -------------------------
    # Fully independent map position
    # -------------------------
    map_left = pos.x0 + pos.width * map_left_frac
    map_bottom = pos.y0 + pos.height * map_bottom_frac
    map_w = pos.width * map_width_frac
    map_h = pos.height * map_height_frac

    ax_map = fig.add_axes(
        [map_left, map_bottom, map_w, map_h],
        projection=ccrs.PlateCarree()
    )
    ax_map.set_facecolor("none")

    cmap = mcolors.ListedColormap([
        regime_colors[1],  # regime value 1 = water-limited
        regime_colors[2],  # regime value 2 = energy-limited
        regime_colors[3],  # regime value 3 = seasonality-regulated
    ])
    norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)

    regime.plot(
        ax=ax_map,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False
    )

    ax_map.coastlines(linewidth=0.35, color="black")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_xlabel("")
    ax_map.set_ylabel("")
    ax_map.gridlines(draw_labels=False, alpha=0.15)

    for spine in ax_map.spines.values():
        spine.set_edgecolor("0.25")
        spine.set_linewidth(0.6)

    return ax_text, ax_map

## FIG 2 ##

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# =========================================================
# Helpers
# =========================================================

def _extract_da(obj, varname):
    """
    Return DataArray from Dataset/DataArray input.
    """
    if isinstance(obj, xr.DataArray):
        return obj
    if isinstance(obj, xr.Dataset):
        if varname in obj:
            return obj[varname]
        raise KeyError(f"Variable '{varname}' not found in dataset.")
    raise TypeError("Expected xarray Dataset or DataArray.")


def _get_member_model_names(ds_period):
    """
    Return only individual model names from ds_dict_change[period],
    excluding ensemble mean / median / std products.
    """
    bad_terms = [
        "ensemble mean",
        "ensemble median",
        "ensemble std",
    ]

    model_names = []
    for k in ds_period.keys():
        k_low = k.lower()
        if any(term in k_low for term in bad_terms):
            continue
        model_names.append(k)

    return model_names


def plot_boolean_scatter_mask(
    ax,
    mask,
    stipple_step=2,
    scatter_size=8,
    marker="o",
    facecolor="black",
    edgecolor="white",
    alpha=0.9,
    linewidth=0.4,
    zorder=35,
):
    """
    Plot a boolean mask as subsampled scatter points.
    """
    if "lat" in mask.dims and "lon" in mask.dims:
        mask_sub = mask.isel(
            lat=slice(0, None, stipple_step),
            lon=slice(0, None, stipple_step),
        )
    else:
        dims = list(mask.dims)
        d1, d2 = dims[-2], dims[-1]
        mask_sub = mask.isel(
            {d1: slice(0, None, stipple_step), d2: slice(0, None, stipple_step)}
        )

    pts = mask_sub.where(mask_sub).stack(points=mask_sub.dims).dropna("points")

    if pts.sizes.get("points", 0) == 0:
        return

    x = pts["lon"].values
    y = pts["lat"].values

    ax.scatter(
        x, y,
        s=scatter_size,
        marker=marker,
        facecolors=facecolor,
        edgecolors=edgecolor,
        alpha=alpha,
        linewidths=linewidth,
        transform=ccrs.PlateCarree(),
        zorder=zorder,
        clip_on=True,
    )


def get_low_sign_agreement_mask(
    ds_dict_change,
    period,
    variable,
    ensemble_key="12 model ensemble mean",
    min_agree_models=8,
):
    """
    Return mask where fewer than `min_agree_models` agree on sign of change.

    Agreement is based on the larger of:
      number of models with positive change
      number of models with negative change

    Excludes ensemble mean / median / std from the agreement calculation.
    """
    ds_period = ds_dict_change[period]

    model_names = _get_member_model_names(ds_period)

    da_models = xr.concat(
        [ds_period[k].drop("member_id", errors="ignore")[variable] for k in model_names],
        dim="model"
    )

    da_mean = ds_period[ensemble_key].drop("member_id", errors="ignore")[variable]

    pos_count = (da_models > 0).sum("model")
    neg_count = (da_models < 0).sum("model")
    max_same_sign = xr.where(pos_count >= neg_count, pos_count, neg_count)

    valid = np.isfinite(da_mean)
    low_agree = (max_same_sign < min_agree_models).where(valid)

    return low_agree, da_models.sizes["model"]


def add_low_agreement_legend(fig, anchor_ax, min_agree_models, n_models, fontsize=12,
                             dx=-0.08, dy=0.08):
    low_agree_handle = scatter_legend(
        anchor_ax,
        label=f"Low ensemble\nsign agreement (<{min_agree_models}/{n_models})",
        facecolor="white",
        edgecolor="black",
        lw=0.8,
    )

    leg = fig.legend(
        handles=[low_agree_handle],
        labels=[low_agree_handle.get_label()],
        handler_map={
            type(low_agree_handle): ScatterBoxHandler(
                ydescent_offset_box=-4,
                box_width_offset=1.2,
                box_height_offset=2,
                scatter_spacing_y_offset=1.5,
                scatter_spacing_x_offset=0.5,
                scatter_vertical_offset=-1,
                scatter_horizontal_offset=-4,
            )
        },
        loc="center left",
        bbox_to_anchor=(
            anchor_ax.get_position().x1 + dx,
            anchor_ax.get_position().y0 + anchor_ax.get_position().height + dy
        ),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=fontsize,
    )

    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("0.5")
    frame.set_linewidth(0.6)
    frame.set_alpha(1.0)

    for t in leg.get_texts():
        t.set_fontweight("bold")

    return leg


from matplotlib.patches import Rectangle

def add_bgws_summary_inset(
    fig,
    parent_ax,
    da_hist_bgws,
    da_change_bgws,
    low_agree_mask=None,
    left_frac=0.70,
    bottom_frac=0.28,
    width_frac=0.16,
    height_frac=0.22,
    fontsize=10,
):
    """
    Small summary bar chart for four categories:
      Greener, Less green, Less blue, Bluer

    Bar height = total land-area fraction in category
    Horizontal line = robust land-area fraction in category
    Hatched upper part = uncertain fraction in category

    Area-weighted using cosine(latitude).
    """
    pos = parent_ax.get_position()

    ax_inset = fig.add_axes([
        pos.x0 + pos.width * left_frac,
        pos.y0 + pos.height * bottom_frac,
        pos.width * width_frac,
        pos.height * height_frac,
    ])

    valid = np.isfinite(da_hist_bgws) & np.isfinite(da_change_bgws)

    if low_agree_mask is not None:
        low_agree_mask = low_agree_mask.fillna(False).astype(bool)
        robust_mask = (~low_agree_mask) & valid
    else:
        robust_mask = valid

    lat_weights = np.cos(np.deg2rad(da_change_bgws["lat"]))
    weights = xr.ones_like(da_change_bgws) * lat_weights

    hist_blue = da_hist_bgws > 0
    hist_green = da_hist_bgws < 0
    dpos = da_change_bgws > 0
    dneg = da_change_bgws < 0

    cat_masks = {
        "Greener": hist_green & dneg,
        "Less green": hist_green & dpos,
        "Less blue": hist_blue & dneg,
        "Bluer": hist_blue & dpos,
    }

    total_w = weights.where(valid).sum(skipna=True)

    total_vals = []
    robust_vals = []

    for cat_mask in cat_masks.values():
        total_frac = weights.where(valid & cat_mask).sum(skipna=True) / total_w * 100.0
        robust_frac = weights.where(robust_mask & cat_mask).sum(skipna=True) / total_w * 100.0

        total_vals.append(float(total_frac))
        robust_vals.append(float(robust_frac))

    labels = ["Greener", "Less\ngreen", "Less\nblue", "Bluer"]
    colors = [
        (30/255, 130/255, 30/255),     # Greener
        (0.54, 0.42, 0.65),            # Less green
        (180/255, 160/255, 120/255),   # Less blue
        (55/255, 140/255, 225/255),    # Bluer
    ]

    x = np.arange(len(labels))
    bar_width = 0.72

    bars = ax_inset.bar(
        x,
        total_vals,
        width=bar_width,
        color=colors,
        edgecolor="0.2",
        linewidth=0.6,
        alpha=0.8,
        zorder=2
    )

    # robust line + uncertain hatched top for each bar
    for xi, total_v, robust_v, bar in zip(x, total_vals, robust_vals, bars):
        uncertain_height = total_v - robust_v

        # stronger robust line: white underlay + black line + end caps
        if robust_v > 0:
            x0 = xi - bar_width / 2
            x1 = xi + bar_width / 2

            # white halo underlay
            #ax_inset.hlines(
            #    robust_v,
            #    x0, x1,
            #    colors="white",
            #    linestyles="-",
            #    linewidth=3.0,
            #    zorder=4
            #)

            # black main line
            ax_inset.hlines(
                robust_v,
                x0, x1,
                colors="black",
                linestyles="-",
                linewidth=1,
                zorder=5
            )

            # small end caps
            #cap_h = 0.7
            #ax_inset.vlines(
            #    [x0, x1],
            #    robust_v - cap_h / 2,
            #    robust_v + cap_h / 2,
            #    colors="black",
            #    linewidth=1,
            #    zorder=5
            #)

        # hatched uncertain part only above robust line and inside bar
        if uncertain_height > 0:
            rect = Rectangle(
                (xi - bar_width / 2, robust_v),
                bar_width,
                uncertain_height,
                facecolor="none",
                edgecolor="0.0",
                hatch="..",
                linewidth=0.0,
                zorder=3
            )
            ax_inset.add_patch(rect)

    ax_inset.set_xticks(x)
    ax_inset.set_xticklabels(
        labels,
        fontsize=fontsize - 1,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        multialignment="center",
    )

    ax_inset.tick_params(axis="x", pad=2)
    ax_inset.set_ylabel("Land area [%]", fontsize=fontsize)
    ax_inset.set_yticks([0, 10, 20, 30, 40])
    ax_inset.set_yticklabels(["0", "10", "20", "30", "40"], fontsize=fontsize - 1)

    ax_inset.grid(
        axis="y",
        linestyle="--",
        color="0.7",
        alpha=0.6,
        linewidth=0.6
    )

    ax_inset.tick_params(axis="y", labelsize=fontsize - 1)

    ymax = max(total_vals) * 1.20 if len(total_vals) > 0 else 100
    ax_inset.set_ylim(0, ymax)

    ## -------------------------
    # tiny inset-specific key
    # -------------------------
    key_x0 = 0.23
    key_y_patch = 0.90
    key_text_x = 0.4
    
    # low-agreement hatched patch key
    key_patch = Rectangle(
        (key_x0, key_y_patch - 0.015),
        0.12,
        0.11,
        transform=ax_inset.transAxes,
        facecolor="white",     # or "none"
        edgecolor="black",
        hatch="..",
        linewidth=0.5,
        zorder=6,
        clip_on=False,
    )
    ax_inset.add_patch(key_patch)
    plt.rcParams["hatch.linewidth"] = 1
    
    ax_inset.text(
        key_text_x, key_y_patch + 0.03,
        "Low agreement",
        transform=ax_inset.transAxes,
        fontsize=fontsize - 1,
        va="center",
        ha="left",
    )

    for spine in ax_inset.spines.values():
        spine.set_edgecolor("0.5")
        spine.set_linewidth(0.8)

    ax_inset.set_facecolor("white")
    ax_inset.patch.set_alpha(0.95)

    return ax_inset

def add_f2_top_colorbars(fig, ax_top, pcm_blue, pcm_green):
    """
    Add the two BGWS colorbars below panel (a), side by side.
    """
    pos = ax_top.get_position()

    cbar_y = pos.y0 - 0.078
    cbar_h = 0.020
    cbar_w = 0.2

    cbar1_left = pos.x0 + pos.width * 0.3 - cbar_w / 2
    cbar2_left = pos.x0 + pos.width * 0.7 - cbar_w / 2

    cax1 = fig.add_axes([cbar1_left, cbar_y, cbar_w, cbar_h])
    cax2 = fig.add_axes([cbar2_left, cbar_y, cbar_w, cbar_h])

    cbar1 = fig.colorbar(
        pcm_blue,
        cax=cax1,
        orientation="horizontal",
        extend="both",
        drawedges=True,
    )
    cbar2 = fig.colorbar(
        pcm_green,
        cax=cax2,
        orientation="horizontal",
        extend="both",
        drawedges=True,
    )

    for cbar, cax, title in [
        (cbar1, cax1, "Historical blue water regime"),
        (cbar2, cax2, "Historical green water regime"),
    ]:
        cbar.set_ticks([-10, -5, 0, 5, 10])
        cbar.set_ticklabels(["-10", "-5", "0", "5", "10"])
        cbar.set_label(r"$\Delta$BGWS [ppts]", fontsize=19, weight="bold")
        cbar.ax.tick_params(labelsize=16)
        cax.set_title(title, fontsize=19, pad=10, weight="bold")

    return cax1, cax2


# =========================================================
# Panel functions
# =========================================================

def f2_upper_panel(
    ds_dict_historical,
    ds_dict_change,
    ax,
    fig,
    period="ssp370_ff-historical",
    hist_key="12 model ensemble mean",
    change_ens_key="12 model ensemble mean",
    min_agree_models=8,
    stipple_step=2,
    scatter_size=8,
):
    """
    Upper panel of Figure 2:
    ΔBGWS split into historical blue-water and green-water regimes.
    """

    ax.coastlines(color="black", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("0.6")
        spine.set_linewidth(0.5)

    gl = ax.gridlines(draw_labels=True, linestyle="--", color="black", alpha=0.1)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 16}
    gl.ylabel_style = {"size": 16}
    ax.tick_params(axis="both", which="major", labelsize=16)

    ds_hist, ds_change_new, subdivisions = subdivide_bgws(
        ds_dict_historical, ds_dict_change
    )

    da_hist_bgws = _extract_da(ds_hist, "bgws_tran_mean")
    da_change_bgws = ds_dict_change[period][change_ens_key].drop("member_id", errors="ignore")["bgws_tran_mean"]

    bw_cmap, _, gw_cmap, _ = col_uti.create_bgws_change_colormaps(-10, 10, 5)
    norm = plt.Normalize(vmin=-10, vmax=10)

    pcm_blue = None
    pcm_green = None

    for name, (_, change_da) in subdivisions.items():
        cmap_here = bw_cmap if "Blue" in name else gw_cmap

        pcm = change_da.plot(
            ax=ax,
            add_colorbar=False,
            cmap=cmap_here,
            norm=norm,
            transform=ccrs.PlateCarree()
        )

        if "Blue" in name:
            pcm_blue = pcm
        else:
            pcm_green = pcm

    low_agree_mask, n_models = get_low_sign_agreement_mask(
        ds_dict_change,
        period,
        "bgws_tran_mean",
        ensemble_key=change_ens_key,
        min_agree_models=min_agree_models,
    )

    plot_boolean_scatter_mask(
        ax,
        low_agree_mask,
        stipple_step=stipple_step,
        scatter_size=scatter_size,
        marker="o",
        facecolor="black",
        edgecolor="white",
        alpha=0.9,
        linewidth=0.4,
        zorder=35,
    )

    return pcm_blue, pcm_green, da_hist_bgws, da_change_bgws, low_agree_mask, n_models


def f2_lower_panel(
    ds_dict_change,
    model,
    variable,
    period,
    ax,
    fig,
    min_agree_models=8,
    stipple_step=3,
    scatter_size=8,
    show_ylabels=True,
):
    """
    Lower change panel (pr_mean, mrro_mean, tran_mean)
    with Figure 1-style low-agreement scatter.
    """
    ds_dict_cleaned = {
        name: ds.drop("member_id", errors="ignore")
        for name, ds in ds_dict_change[period].items()
    }
    ds = ds_dict_cleaned.get(model)

    if ds is None:
        raise ValueError(f"Model '{model}' not found in ds_dict_change.")

    cmap_var = variable
    vmin = col_map_limits[period][cmap_var]["vmin"]
    vmax = col_map_limits[period][cmap_var]["vmax"]
    steps = col_map_limits[period][cmap_var]["steps"]

    cmap, cmap_norm = col_uti.create_colormap(
        cmap_var, period, vmin, vmax, steps
    )

    img = ds[variable].plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
    )

    ax.coastlines(color="black", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("0.6")
        spine.set_linewidth(0.5)

    ax.gridlines(draw_labels=False, linestyle="--", color="black", alpha=0.1)

    xticks = [-120, 0, 120]#[-120, -60, 0, 60, 120]
    yticks = [-40, 0, 40, 80]

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="major", labelsize=16)

    if not show_ylabels:
        ax.set_yticks([], crs=ccrs.PlateCarree())
        ax.tick_params(left=False, labelleft=False)

    low_agree_mask, _ = get_low_sign_agreement_mask(
        ds_dict_change,
        period,
        variable,
        ensemble_key=model,
        min_agree_models=min_agree_models,
    )

    plot_boolean_scatter_mask(
        ax,
        low_agree_mask,
        stipple_step=stipple_step,
        scatter_size=scatter_size,
        marker="o",
        facecolor="black",
        edgecolor="white",
        alpha=0.9,
        linewidth=0.5,
        zorder=35,
    )

    pos = ax.get_position()

    left = pos.x0 + 0.12 * pos.width
    bottom = pos.y0 - 0.05
    width = 0.76 * pos.width
    height = 0.018
    
    cax = fig.add_axes([left, bottom, width, height])
    
    cbar = fig.colorbar(
        img,
        cax=cax,
        orientation="horizontal",
        drawedges=True,
        extend="both",
    )

    label = col_uti.get_global_map_var_name(period, variable)
    cbar.set_label(label, fontsize=19, weight="bold")

    ticks = np.arange(vmin, vmax + steps, steps * 2 if steps > 1 else steps)
    ticklabels = [f"{t:.1f}" if steps < 1 else f"{t:.0f}" for t in ticks]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=16)

    return img


# =========================================================
# Main Figure 2
# =========================================================

def figure2(
    ds_dict_historical,
    ds_dict_change,
    dpi=300,
    filetype="pdf",
    savepath=None,
    hist_key="12 model ensemble mean",
    change_ens_key="12 model ensemble mean",
    period="ssp370_ff-historical",
    min_agree_models=8,
    stipple_step=2,
    scatter_size_top=8,
    scatter_size_bottom=6,
):
    """
    Figure 2:
      (a) ΔBGWS subdivided by historical blue-water vs green-water regime
      (b) ΔP
      (c) ΔR
      (d) ΔEt

    Low-agreement scatter marks regions where fewer than `min_agree_models`
    agree on the sign of change.
    """
    fig = plt.figure(figsize=(16, 13))

    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[1.10, 0.95],
        hspace=0.06,
        wspace=0.10,
    )

    # --------------------
    # PANEL a
    # --------------------
    ax_a = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())

    pcm_blue, pcm_green, da_hist_bgws, da_change_bgws, low_agree_mask, n_models = f2_upper_panel(
        ds_dict_historical=ds_dict_historical,
        ds_dict_change=ds_dict_change,
        ax=ax_a,
        fig=fig,
        period=period,
        hist_key=hist_key,
        change_ens_key=change_ens_key,
        min_agree_models=min_agree_models,
        stipple_step=stipple_step,
        scatter_size=scatter_size_top,
    )
    
    ax_a.text(
        0.01, 0.99, "(a)",
        transform=ax_a.transAxes,
        fontsize=19,
        fontweight="bold",
        ha="left",
        va="top"
    )

    add_bgws_summary_inset(
        fig,
        ax_a,
        da_hist_bgws=da_hist_bgws,
        da_change_bgws=da_change_bgws,
        low_agree_mask=low_agree_mask,
        left_frac=0.06,
        bottom_frac=0.16,
        width_frac=0.2,
        height_frac=0.3,
        fontsize=16,
    )

    cax1, cax2 = add_f2_top_colorbars(fig, ax_a, pcm_blue, pcm_green)

    add_low_agreement_legend(
        fig,
        anchor_ax=cax2,
        min_agree_models=min_agree_models,
        n_models=n_models,
        fontsize=14,
    )

    # --------------------
    # PANELS b, c, d
    # --------------------
    vars_bottom = ["pr_mean", "mrro_mean", "tran_mean"]
    labels = ["(b)", "(c)", "(d)"]

    for i, (var, lab) in enumerate(zip(vars_bottom, labels)):
        ax = fig.add_subplot(gs[1, i], projection=ccrs.PlateCarree())

        f2_lower_panel(
            ds_dict_change=ds_dict_change,
            model=change_ens_key,
            variable=var,
            period=period,
            ax=ax,
            fig=fig,
            min_agree_models=min_agree_models,
            stipple_step=stipple_step+2,
            scatter_size=scatter_size_bottom,
            show_ylabels=(i == 0),
        )

        ax.text(
            0.01, 0.99, lab,
            transform=ax.transAxes,
            fontsize=19,
            fontweight="bold",
            ha="left",
            va="top"
        )

    if savepath:
        fname = f"fig_2_{dpi}dpi_updated.{filetype}"
        col_uti.save_fig(fig, savepath, fname, dpi=dpi)
        print(f"Figure saved under {savepath}{fname}")

    plt.show()

def get_axis_scale(ax):
    """
    Returns relative width & height of the axis in figure coordinates.
    Used to scale legend font sizes.
    """
    bbox = ax.get_position()
    width = bbox.width
    height = bbox.height
    return min(width, height)   # use the smaller dimension for scaling

def plot_agreement_mask(ds_dict_cleaned, ds, model, variable, ax_main, fontsize, scatter_size, xdescent_offset, ydescent_offset, box_width_offset, box_height_offset, scatter_spacing_x_offset, scatter_spacing_y_offset, scatter_vertical_offset, scatter_horizontal_offset):
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
    
    # scatter
    ax_main.scatter(
        lon[low_agreement_mask], lat[low_agreement_mask],
        color='grey', marker='D', s=scatter_size,  
        transform=ccrs.PlateCarree()
    )

    # ---------- ADAPTIVE LEGEND SIZE ----------
    scale = get_axis_scale(ax_main)   # compute panel size

    # Scatter legend handle
    scatter_handle = scatter_legend(
        ax_main, label="Low Ensemble Agreement",
        facecolor="white", edgecolor="black", lw=0.5
    )

    ax_main.legend(
        handles=[scatter_handle],
        handler_map={type(scatter_handle): ScatterBoxHandler(xdescent_offset, ydescent_offset, box_width_offset, box_height_offset, scatter_spacing_x_offset, scatter_spacing_y_offset, scatter_vertical_offset, scatter_horizontal_offset)},
        fontsize=fontsize,
        loc="lower right",
    )

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs


# =========================================================
# Reuse from Fig. 2
# =========================================================

def plot_boolean_scatter_mask(
    ax,
    mask,
    stipple_step=2,
    scatter_size=8,
    marker="o",
    facecolor="black",
    edgecolor="white",
    alpha=0.9,
    linewidth=0.4,
    zorder=35,
):
    """
    Plot boolean mask as subsampled scatter points.
    """
    if "lat" in mask.dims and "lon" in mask.dims:
        mask_sub = mask.isel(
            lat=slice(0, None, stipple_step),
            lon=slice(0, None, stipple_step),
        )
    else:
        dims = list(mask.dims)
        d1, d2 = dims[-2], dims[-1]
        mask_sub = mask.isel(
            {d1: slice(0, None, stipple_step), d2: slice(0, None, stipple_step)}
        )

    pts = mask_sub.where(mask_sub).stack(points=mask_sub.dims).dropna("points")

    if pts.sizes.get("points", 0) == 0:
        return

    ax.scatter(
        pts["lon"].values,
        pts["lat"].values,
        s=scatter_size,
        marker=marker,
        facecolors=facecolor,
        edgecolors=edgecolor,
        alpha=alpha,
        linewidths=linewidth,
        transform=ccrs.PlateCarree(),
        zorder=zorder,
        clip_on=True,
    )


# =========================================================
# Helpers for Fig. 4/5 combined
# =========================================================

def _drop_member_id(dct):
    out = {}
    for k, v in dct.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.drop("member_id", errors="ignore") for kk, vv in v.items()}
        else:
            out[k] = v.drop("member_id", errors="ignore")
    return out


def _get_period_dict(ds_dict, period=None):
    """
    Return the period-level dict of model datasets.
    Works for:
      ds_dict[period][model]
    or
      ds_dict[model]
    """
    ds_dict = _drop_member_id(ds_dict)

    if period is not None and period in ds_dict and isinstance(ds_dict[period], dict):
        return ds_dict[period]

    if all(isinstance(v, xr.Dataset) for v in ds_dict.values()):
        return ds_dict

    for _, v in ds_dict.items():
        if isinstance(v, dict):
            return v

    raise ValueError("Could not infer period-specific dataset dictionary.")


def _get_ds(ds_dict, key, period=None):
    ds_period = _get_period_dict(ds_dict, period)
    if key in ds_period:
        return ds_period[key]
    raise ValueError(f"Model '{key}' not found.")


def _get_member_model_names(ds_dict, period=None):
    ds_period = _get_period_dict(ds_dict, period)

    bad_terms = (
        "ensemble mean",
        "ensemble median",
        "ensemble std",
    )

    model_names = []
    for k in ds_period.keys():
        if any(term in k.lower() for term in bad_terms):
            continue
        model_names.append(k)

    return model_names


def _make_quadrant(da_bgws, da_change, threshold_bgws=0.0, threshold_change=0.0):
    """
    4-category quadrant classification:
      0 = BGWS+, change+
      1 = BGWS+, change-
      2 = BGWS-, change+
      3 = BGWS-, change-
    """
    sB = xr.where(
        da_bgws > threshold_bgws, 1,
        xr.where(da_bgws < -threshold_bgws, -1, np.nan)
    )
    sV = xr.where(
        da_change > threshold_change, 1,
        xr.where(da_change < -threshold_change, -1, np.nan)
    )

    valid = np.isfinite(sB) & np.isfinite(sV)

    quad = xr.full_like(da_bgws, np.nan, dtype=float)
    quad = xr.where(valid & (sB == 1)  & (sV == 1),  0, quad)
    quad = xr.where(valid & (sB == 1)  & (sV == -1), 1, quad)
    quad = xr.where(valid & (sB == -1) & (sV == 1),  2, quad)
    quad = xr.where(valid & (sB == -1) & (sV == -1), 3, quad)

    return quad, valid


def get_low_quadrant_agreement_mask(
    ds_dict,
    period,
    change_var,
    threshold_bgws=0.0,
    threshold_change=0.0,
    min_agree_models=8,
    min_participation_frac=0.55,
):
    """
    Low-agreement mask based on quadrant agreement across member models.

    A grid cell is 'low agreement' when:
      - fewer than `min_agree_models` members agree on the same quadrant, or
      - participation is too low
    """
    ds_period = _get_period_dict(ds_dict, period)
    model_names = _get_member_model_names(ds_dict, period)

    if len(model_names) == 0:
        raise ValueError("No member models found for agreement calculation.")

    quads = []
    for name in model_names:
        ds_m = ds_period[name]

        quad_m, _ = _make_quadrant(
            ds_m["bgws_tran_mean"],
            ds_m[change_var],
            threshold_bgws=threshold_bgws,
            threshold_change=threshold_change,
        )
        quads.append(quad_m.expand_dims(model=[name]))

    quad_models = xr.concat(quads, dim="model")

    valid_count = quad_models.notnull().sum("model")

    quad_counts = xr.concat(
        [(quad_models == q).sum("model") for q in range(4)],
        dim="quadrant"
    )
    best_count = quad_counts.max("quadrant")

    min_participation = int(np.ceil(min_participation_frac * len(model_names)))

    low_agree = (
        (best_count < min_agree_models) |
        (valid_count < min_participation)
    ).where(valid_count > 0)

    return low_agree, len(model_names)


def add_uncertainty_legend(ax, min_agree_models, n_models, loc="lower right", fontsize=11):
    handle = Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markerfacecolor="black",
        markeredgecolor="white",
        markeredgewidth=0.7,
        markersize=7,
        label=f"Low ensemble agreement (<{min_agree_models}/{n_models})",
    )

    leg = ax.legend(
        handles=[handle],
        loc=loc,
        fontsize=fontsize,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
    )
    return leg

def compute_quadrant_area_fractions(
    ds_dict,
    model,
    period,
    change_var,
    threshold_bgws=0.0,
    threshold_change=0.0,
    min_agree_models=8,
    min_participation_frac=0.55,
):
    """
    Returns area fractions [% of valid land area] for the 4 quadrants:
      total fraction
      robust fraction
      uncertain fraction = total - robust
    """

    ds = _get_ds(ds_dict, model, period=period)

    dB = ds["bgws_tran_mean"]
    dV = ds[change_var]

    quad, valid = _make_quadrant(
        dB, dV,
        threshold_bgws=threshold_bgws,
        threshold_change=threshold_change,
    )

    # low-agreement mask from ensemble members
    low_agree_mask, n_models = get_low_quadrant_agreement_mask(
        ds_dict,
        period=period,
        change_var=change_var,
        threshold_bgws=threshold_bgws,
        threshold_change=threshold_change,
        min_agree_models=min_agree_models,
        min_participation_frac=min_participation_frac,
    )

    low_agree_mask = low_agree_mask.fillna(False).astype(bool)
    robust_mask = valid & (~low_agree_mask)

    # latitude weighting
    lat_weights = np.cos(np.deg2rad(dB["lat"]))
    weights = xr.ones_like(dB) * lat_weights

    total_w = weights.where(valid).sum(skipna=True)

    total_fracs = []
    robust_fracs = []
    uncertain_fracs = []

    for q in range(4):
        qmask = quad == q

        total_q = weights.where(valid & qmask).sum(skipna=True)
        robust_q = weights.where(robust_mask & qmask).sum(skipna=True)

        total_pct = float(total_q / total_w * 100.0) if float(total_w) > 0 else 0.0
        robust_pct = float(robust_q / total_w * 100.0) if float(total_w) > 0 else 0.0
        uncertain_pct = total_pct - robust_pct

        total_fracs.append(total_pct)
        robust_fracs.append(robust_pct)
        uncertain_fracs.append(max(0.0, uncertain_pct))

    return {
        "total": total_fracs,
        "robust": robust_fracs,
        "uncertain": uncertain_fracs,
        "n_models": n_models,
    }

def add_quadrant_fraction_inset(
    fig,
    parent_ax,
    fractions,
    change_var="mrro_mean",
    left_frac=0.63,
    bottom_frac=0.08,
    width_frac=0.23,
    height_frac=0.25,
    fontsize=10,
):
    """
    Small inset bar plot:
      colored lower part = robust fraction
      white upper part   = non-robust fraction
      total bar height   = total fraction in quadrant
    """

    pos = parent_ax.get_position()

    ax_inset = fig.add_axes([
        pos.x0 + pos.width * left_frac,
        pos.y0 + pos.height * bottom_frac,
        pos.width * width_frac,
        pos.height * height_frac,
    ])

    # same quadrant colors as maps
    BLUE   = (40/255, 125/255, 210/255)   # BGWS↑ + change↑
    PURPLE = (130/255, 79/255, 158/255)   # BGWS↑ + change↓
    GREEN  = (15/255, 115/255, 15/255)    # BGWS↓ + change↑
    BROWN  = (150/255, 140/255, 100/255)  # BGWS↓ + change↓

    colors = [BLUE, PURPLE, GREEN, BROWN]

    total_vals = fractions["total"]
    robust_vals = fractions["robust"]
    uncertain_vals = fractions["uncertain"]

    x = np.arange(4)
    bar_width = 0.72

    # robust colored part
    ax_inset.bar(
        x,
        robust_vals,
        width=bar_width,
        color=colors,
        edgecolor="black",
        alpha=0.8,
        linewidth=0.7,
        zorder=3,
    )

    # uncertain white top
    ax_inset.bar(
        x,
        uncertain_vals,
        width=bar_width,
        bottom=robust_vals,
        color="white",
        edgecolor="black",
        linewidth=0.7,
        zorder=4,
    )

    # top line for total
    for xi, total_v in zip(x, total_vals):
        ax_inset.hlines(
            total_v,
            xi - bar_width/2,
            xi + bar_width/2,
            colors="black",
            linewidth=1.0,
            zorder=5,
        )

    if change_var == "mrro_mean":
        labels = ["B↑R↑", "B↑R↓", "B↓R↑", "B↓R↓"]
    else:
        labels = ["B↑T↑", "B↑T↓", "B↓T↑", "B↓T↓"]

    ax_inset.set_xticks([])
    ax_inset.tick_params(axis="x", length=0)
    ax_inset.set_ylabel("Land area [%]", fontsize=fontsize, labelpad=-0.1)

    #ymax = max(total_vals) if len(total_vals) else 10
    #ax_inset.set_ylim(0, max(8, ymax * 1.18))

    # y-axis formatting
    ax_inset.set_ylim(0, 50)
    ax_inset.set_yticks([25, 50])
    ax_inset.set_yticklabels(["25", "50"], fontsize=fontsize - 1)
    
    #ax_inset.tick_params(axis="y", labelsize=fontsize - 1)
    ax_inset.grid(axis="y", linestyle="--", color="0.7", alpha=0.6, linewidth=0.6)
    ax_inset.set_axisbelow(True)

    for spine in ax_inset.spines.values():
        spine.set_edgecolor("0.5")
        spine.set_linewidth(0.6)

    ax_inset.set_facecolor("white")
    ax_inset.patch.set_alpha(0.96)

    return ax_inset

    from matplotlib.patches import Rectangle

from matplotlib.patches import Rectangle

def add_robust_masked_legend(
    fig,
    parent_ax,
    left_frac=0.52,
    bottom_frac=0.035,
    width_frac=0.42,
    height_frac=0.05,
    fontsize=16,
    robust_facecolor="0.65",
):
    """
    One-line legend in the lower-right area of a map panel.
    No surrounding box.
    """
    pos = parent_ax.get_position()

    ax_leg = fig.add_axes([
        pos.x0 + pos.width * left_frac,
        pos.y0 + pos.height * bottom_frac,
        pos.width * width_frac,
        pos.height * height_frac,
    ])

    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")

    box_w = 0.07
    box_h = 0.8
    y = 0.6

    # Robust
    ax_leg.add_patch(
        Rectangle(
            (0.18, y - box_h / 2),
            box_w,
            box_h,
            facecolor=robust_facecolor,
            edgecolor="black",
            linewidth=0.7,
        )
    )
    ax_leg.text(
        0.28, y,
        "Robust",
        ha="left",
        va="center",
        fontsize=fontsize,
    )

    # Masked
    ax_leg.add_patch(
        Rectangle(
            (0.56, y - box_h / 2),
            box_w,
            box_h,
            facecolor="white",
            edgecolor="black",
            linewidth=0.7,
        )
    )
    ax_leg.text(
        0.65, y,
        "Masked",
        ha="left",
        va="center",
        fontsize=fontsize,
    )

    return ax_leg
# =========================================================
# Updated map panel
# =========================================================

def plot_bgws_quadrant_map_on_axis(
    ax,
    ds_dict,
    model,
    period,
    change_var="mrro_mean",
    threshold_bgws=0.0,
    threshold_change=0.0,
    magnitude_source="var",
    alpha_low=0.60,
    alpha_med=0.80,
    alpha_high=1.00,
    min_agree_models=8,
    min_participation_frac=0.55,
    agreement_mode="mask",   # "stipple" or "mask"
    stipple_step=2,
    scatter_size=8,
    show_category_legend=False,
    show_uncertainty_legend=False,
    show_ylabels=True,
    legend_fontsize=11,
    show_summary_inset=False,
    summary_inset_left_frac=0.08,
    summary_inset_bottom_frac=0.09,
    summary_inset_width_frac=0.175,
    summary_inset_height_frac=0.3,
    summary_inset_fontsize=14,
    fig=None,
):
    """
    Updated version:
    - supports masking or stippling for uncertainty
    - no legends by default
    - fixed lat/lon label locations
    """

    def _rgb01(t):
        r, g, b = t
        return (r / 255.0, g / 255.0, b / 255.0)

    def _rgba(rgb, a):
        return (*_rgb01(rgb), float(a))

    # --------------------
    # data
    # --------------------
    ds = _get_ds(ds_dict, model, period=period)
    dB = ds["bgws_tran_mean"]
    dV = ds[change_var]

    quad, valid = _make_quadrant(
        dB, dV,
        threshold_bgws=threshold_bgws,
        threshold_change=threshold_change,
    )

    # --------------------
    # 4 category map only
    # --------------------
    plot_data = quad.copy()

    # --------------------
    # uncertainty
    # --------------------
    low_agree_mask = None
    n_models = None

    is_ensemble_product = any(
        term in model.lower() for term in ("ensemble mean", "ensemble median")
    )

    if is_ensemble_product:
        low_agree_mask, n_models = get_low_quadrant_agreement_mask(
            ds_dict,
            period=period,
            change_var=change_var,
            threshold_bgws=threshold_bgws,
            threshold_change=threshold_change,
            min_agree_models=min_agree_models,
            min_participation_frac=min_participation_frac,
        )

        low_agree_bool = low_agree_mask.fillna(False).astype(bool)

        if agreement_mode == "mask":
            plot_data = plot_data.where(~low_agree_bool)

    # --------------------
    # colours
    # --------------------
    BLUE_OVER   = (40, 125, 210)
    VIO_OVER    = (130, 79, 158)
    GREEN_UNDER = (15, 115, 15)
    BROWN_UNDER = (150, 140, 100)

    cat_rgbs = [BLUE_OVER, VIO_OVER, GREEN_UNDER, BROWN_UNDER]
    alphas = [alpha_low, alpha_med, alpha_high]

    colors = [
        _rgba(BLUE_OVER, 0.8),
        _rgba(VIO_OVER, .8),
        _rgba(GREEN_UNDER, .8),
        _rgba(BROWN_UNDER, .8),
    ]

    cmap = ListedColormap(colors)
    cmap.set_bad((1, 1, 1, 0))

    # --------------------
    # plot
    # --------------------
    plot_data.plot(
        ax=ax,
        cmap=cmap,
        add_colorbar=False,
        transform=ccrs.PlateCarree(),
    )

    ax.set_facecolor("white")
    ax.coastlines(linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, color="black", alpha=0.10, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_ylabels
    gl.bottom_labels = True

    gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([-40, 0, 40, 80])

    gl.xlabel_style = {"size": 15}
    gl.ylabel_style = {"size": 15}
    ax.tick_params(axis="both", labelsize=15)

    if agreement_mode == "stipple" and low_agree_mask is not None:
        plot_boolean_scatter_mask(
            ax,
            low_agree_bool,
            stipple_step=stipple_step,
            scatter_size=scatter_size,
            marker="o",
            facecolor="black",
            edgecolor="white",
            alpha=0.90,
            linewidth=0.35,
            zorder=40,
        )

    if show_summary_inset and fig is not None and is_ensemble_product:
        fractions = compute_quadrant_area_fractions(
            ds_dict=ds_dict,
            model=model,
            period=period,
            change_var=change_var,
            threshold_bgws=threshold_bgws,
            threshold_change=threshold_change,
            min_agree_models=min_agree_models,
            min_participation_frac=min_participation_frac,
        )

        add_quadrant_fraction_inset(
            fig=fig,
            parent_ax=ax,
            fractions=fractions,
            change_var=change_var,
            left_frac=summary_inset_left_frac,
            bottom_frac=summary_inset_bottom_frac,
            width_frac=summary_inset_width_frac,
            height_frac=summary_inset_height_frac,
            fontsize=summary_inset_fontsize,
        )

    return ax

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import textwrap


# ---------------------------------------------------------
# Optional global font settings
# Set these to match your manuscript style
# ---------------------------------------------------------
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'Nimbus Sans'
# plt.rcParams["font.size"] = 10


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------
def _rgb01(rgb):
    return tuple(v / 255.0 for v in rgb)


def _axes_line_height(ax, fontsize_pt, leading=1.22):
    """
    Convert font size to an approximate line height in axes coordinates.
    """
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    return (fontsize_pt / 72.0 * fig.dpi) / bbox.height * leading


def _chars_for_width(ax, width_axes, fontsize_pt, pad_frac=0.10):
    """
    Approximate number of characters that fit in a given cell width.
    """
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    width_px = bbox.width * width_axes * (1 - 2 * pad_frac)
    chars = int(width_px / (0.56 * fontsize_pt))
    return max(chars, 16)


def _wrap_bullets(items, width):
    wrapped = []
    for item in items:
        wrapped.append(
            textwrap.fill(
                item,
                width=width,
                initial_indent="• ",
                subsequent_indent="  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(wrapped)


def _wrap_regions(regions, width):
    txt = ", ".join(regions)
    return textwrap.fill(
        txt,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )

import matplotlib.patheffects as pe
# ---------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------
def _draw_header_cell(
    ax,
    x0,
    y0,
    w,
    h,
    title,
    bg,
    arrow=None,
    lw=1,
    title_fs=20,
    arrow_fs=34,
    arrow_zone_frac=0.22,   # reserved width for arrow
    arrow_x_frac=0.10,      # arrow position inside whole cell
):
    ax.add_patch(
        Rectangle((x0, y0), w, h, facecolor=bg, edgecolor="black", linewidth=0)
    )

    if arrow is not None:
        # thicker-looking arrow without increasing size
        ax.text(
            x0 + arrow_x_frac * w,
            y0 + 0.5 * h,
            arrow,
            ha="center",
            va="center",
            fontsize=arrow_fs,
            fontweight="bold",
            path_effects=[
                pe.Stroke(linewidth=lw, foreground="black"),
                pe.Normal(),
            ],
        )

        # center title in the remaining title area, not by a hard-coded x
        title_x0 = x0 + arrow_zone_frac * w
        title_w = w - arrow_zone_frac * w

        ax.text(
            title_x0 + 0.40 * title_w,
            y0 + 0.5 * h,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight="bold",
        )
    else:
        ax.text(
            x0 + 0.50 * w,
            y0 + 0.58 * h,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight="bold",
        )


def _draw_bgws_label_cell(
    ax,
    x0,
    y0,
    w,
    h,
    arrow,
    bg,
    lw=1,
    bgws_fs=20,
    arrow_fs=34,
):
    ax.add_patch(
        Rectangle((x0, y0), w, h, facecolor=bg, edgecolor="black", linewidth=0)
    )

    ax.text(
        x0 + 0.50 * w,
        y0 + 0.64 * h,
        arrow,
        ha="center",
        va="center",
        fontsize=arrow_fs,
        fontweight="bold",
        path_effects=[
            pe.Stroke(linewidth=lw, foreground="black"),
            pe.Normal(),
        ],
    )
    ax.text(
        x0 + 0.50 * w,
        y0 + 0.34 * h,
        "BGWS",
        ha="center",
        va="center",
        fontsize=bgws_fs,
        fontweight="bold",
    )


def _with_alpha(color, alpha):
    if len(color) == 4:
        return (color[0], color[1], color[2], alpha)
    return (*color, alpha)


def _draw_content_cell(
    ax,
    x0,
    y0,
    w,
    h,
    facecolor,
    chip_color,
    implications,
    body_fs=11.6,
    heading_fs=12.2,
    chip_w_frac=0.16,
    chip_h_frac=0.10,
    lw=1,
    face_alpha=0.12,
    chip_alpha=0.95,
):
    ax.add_patch(
        Rectangle(
            (x0, y0), w, h,
            facecolor=_with_alpha(facecolor, face_alpha),
            edgecolor="black",
            linewidth=0
        )
    )

    pad_x = 0.045 * w
    pad_top = 0.055 * h

    chip_w = chip_w_frac * w
    chip_h = chip_h_frac * h
    chip_x = x0 + 0.124 * w - 0.5 * chip_w
    chip_y = y0 + h - pad_top - chip_h

    ax.add_patch(
        Rectangle(
            (chip_x, chip_y),
            chip_w,
            chip_h,
            facecolor=_with_alpha(chip_color, chip_alpha),
            edgecolor="black",
            linewidth=lw,
        )
    )

    wrap_imp = _chars_for_width(ax, w, body_fs, pad_frac=0.11)
    imp_txt = _wrap_bullets(implications, wrap_imp)

    y_text = chip_y - 0.04 * h

    ax.text(
        x0 + pad_x,
        y_text,
        imp_txt,
        ha="left",
        va="top",
        fontsize=body_fs,
        linespacing=1.28,
    )


def draw_fig5_concept_table(
    ax,
    body_fontsize=13,
    header_fontsize=18,
    bgws_fontsize=18,
):
    """
    Compact, lighter table for the combined figure.
    No example regions.
    """

    BLUE = _rgb01((40, 125, 210))
    PURPLE = _rgb01((130, 79, 158))
    GREEN = _rgb01((15, 115, 15))
    BROWN = _rgb01((150, 140, 100))

    RUNOFF_BG = _rgb01((221, 233, 246))
    TRAN_BG = _rgb01((225, 238, 220))
    BGWS_UP_BG = _rgb01((216, 230, 245))
    BGWS_DOWN_BG = _rgb01((221, 236, 216))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # slightly inset so the panel feels lighter/smaller
    x_left = 0
    x_right = 0.999
    y_bottom = 0.005
    y_top = 1

    total_w = x_right - x_left
    total_h = y_top - y_bottom

    left_w = 0.065 * total_w
    header_h = 0.13 * total_h
    row_h = (total_h - header_h) / 2.0
    col_w = (total_w - left_w) / 4.0

    # -------------------------
    # Header row
    # -------------------------
    _draw_header_cell(
        ax, x_left, y_top - header_h, left_w, header_h,
        title="", bg=_with_alpha(RUNOFF_BG, 0), arrow=None, title_fs=header_fontsize
    )

    _draw_header_cell(
        ax, x_left + left_w + 0 * col_w, y_top - header_h, col_w, header_h,
        title="Runoff", bg=_with_alpha(RUNOFF_BG, 0.4), arrow="↑",
        title_fs=header_fontsize, arrow_fs=32
    )
    _draw_header_cell(
        ax, x_left + left_w + 1 * col_w, y_top - header_h, col_w, header_h,
        title="Runoff", bg=_with_alpha(RUNOFF_BG, 0.4), arrow="↓",
        title_fs=header_fontsize, arrow_fs=32
    )
    _draw_header_cell(
        ax, x_left + left_w + 2 * col_w, y_top - header_h, col_w, header_h,
        title="Transpiration", bg=_with_alpha(TRAN_BG, 0.4), arrow="↑",
        title_fs=header_fontsize, arrow_fs=32
    )
    _draw_header_cell(
        ax, x_left + left_w + 3 * col_w, y_top - header_h, col_w, header_h,
        title="Transpiration", bg=_with_alpha(TRAN_BG, 0.4), arrow="↓",
        title_fs=header_fontsize, arrow_fs=30
    )

    # -------------------------
    # Left BGWS labels
    # -------------------------
    _draw_bgws_label_cell(
        ax, x_left, y_bottom + row_h, left_w, row_h,
        arrow="↑", bg=_with_alpha(BGWS_UP_BG, 0.4), bgws_fs=bgws_fontsize, arrow_fs=30
    )
    _draw_bgws_label_cell(
        ax, x_left, y_bottom, left_w, row_h,
        arrow="↓", bg=_with_alpha(BGWS_DOWN_BG, 0.4), bgws_fs=bgws_fontsize, arrow_fs=30
    )

    # -------------------------
    # Cell content
    # -------------------------
    cells = [
        dict(
            col=0, row=1, bg=RUNOFF_BG, chip=BLUE,
            implications=[
                "Larger blue water share and more runoff",
                "Stronger runoff response per unit precipitation",
                "May increase flood pressure in high flow periods",
            ],
        ),
        dict(
            col=1, row=1, bg=RUNOFF_BG, chip=PURPLE,
            implications=[
                "Larger blue water share and less runoff",
                "Transpiration declines more than runoff",
                "Suggests stronger runoff sensitivity to intense rainfall",
            ],
        ),
        dict(
            col=2, row=1, bg=TRAN_BG, chip=BLUE,
            implications=[
                "Larger blue water share and more transpiration",
                "May reflect higher atmospheric or phenological demand",
                "May raise blue water scarcity in dry periods",
            ],
        ),
        dict(
            col=3, row=1, bg=TRAN_BG, chip=PURPLE,
            implications=[
                "Larger blue water share and less transpiration",
                "Lower vegetation water use shifts partitioning toward runoff",
                "May reduce evaporative cooling and increase heat stress",
            ],
        ),
        dict(
            col=0, row=0, bg=RUNOFF_BG, chip=GREEN,
            implications=[
                "Larger green water share and more runoff",
                "Transpiration increases more strongly than runoff",
                "Runoff may remain elevated where water supply is sufficient",
            ],
        ),
        dict(
            col=1, row=0, bg=RUNOFF_BG, chip=BROWN,
            implications=[
                "Larger green water share and less runoff",
                "Points to lower blue water availability",
                "May increase pressure on water supply",
            ],
        ),
        dict(
            col=2, row=0, bg=TRAN_BG, chip=GREEN,
            implications=[
                "Larger green water share and more transpiration",
                "Reflects a stronger transpiration response",
                "May strengthen land-atmosphere coupling",
            ],
        ),
        dict(
            col=3, row=0, bg=TRAN_BG, chip=BROWN,
            implications=[
                "Larger green water share and less transpiration",
                "Could suggest limits on plant water use",
                "May reduce evaporative cooling and vegetation buffering",
            ],
        ),
        ]
    for cell in cells:
        x0 = x_left + left_w + cell["col"] * col_w
        y0 = y_bottom + row_h if cell["row"] == 1 else y_bottom

        _draw_content_cell(
            ax=ax,
            x0=x0,
            y0=y0,
            w=col_w,
            h=row_h,
            facecolor=cell["bg"],
            chip_color=cell["chip"],
            implications=cell["implications"],
            body_fs=body_fontsize,
            heading_fs=body_fontsize,# + 0.6,
            face_alpha=0.4,
            chip_alpha=0.8,
        )

    # -------------------------
    # draw table grid once
    # -------------------------
    x_lines = [
        x_left,
        x_left + left_w,
        x_left + left_w + col_w,
        x_left + left_w + 2 * col_w,
        x_left + left_w + 3 * col_w,
        x_right,
    ]
    
    y_lines = [
        y_bottom,
        y_bottom + row_h,
        y_top - header_h,
        y_top,
    ]
    
    # vertical lines
    ax.vlines(
        x_lines,
        y_bottom,
        y_top,
        colors="black",
        linewidth=1.0,
        zorder=10,
    )
    
    # horizontal lines
    ax.hlines(
        y_lines,
        x_left,
        x_right,
        colors="black",
        linewidth=1.0,
        zorder=10,
    )


# ---------------------------------------------------------
# Standalone figure
# ---------------------------------------------------------
def figure5_vector(
    dpi=300,
    filetype="pdf",
    savepath=None,
):
    fig = plt.figure(figsize=(15.5, 8.8))
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.92])

    draw_fig5_concept_table(
        ax,
        body_fontsize=13,
        header_fontsize=19,
        bgws_fontsize=19,
    )

    if savepath:
        fname = f"fig_5_vector_updated.{filetype}"
        fig.savefig(f"{savepath}/{fname}", dpi=dpi, bbox_inches="tight")
        print(f"Figure saved under {savepath}/{fname}")

    return fig


# ---------------------------------------------------------
# Example
# ---------------------------------------------------------
if __name__ == "__main__":
    fig = figure5_vector(dpi=300, filetype="png", savepath=".")
    plt.show()


# =========================================================
# Combined Fig. 4 + Fig. 5
# =========================================================

def figure4_5_combined(
    ds_dict,
    model,
    period="ssp370-historical",
    threshold_bgws=0,
    threshold_change=0,
    magnitude_source="joint",
    min_agree_models=8,
    min_participation_frac=0.55,
    agreement_mode="mask",
    stipple_step=2,
    scatter_size=7,
    dpi=300,
    filetype="pdf",
    savepath=None,
):
    """
    Top row:
      (a) BGWS x Runoff
      (b) BGWS x Transpiration

    Bottom row:
      (c) compact conceptual synthesis table
    """

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, .9],
        hspace=-0.1,
        wspace=0.03,
    )

    # --------------------
    # panel a
    # --------------------
    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    plot_bgws_quadrant_map_on_axis(
        ax=ax_a,
        ds_dict=ds_dict,
        model=model,
        period=period,
        change_var="mrro_mean",
        threshold_bgws=threshold_bgws,
        threshold_change=threshold_change,
        magnitude_source=magnitude_source,
        min_agree_models=min_agree_models,
        min_participation_frac=min_participation_frac,
        agreement_mode=agreement_mode,
        stipple_step=stipple_step,
        scatter_size=scatter_size,
        show_category_legend=False,
        show_uncertainty_legend=False,
        show_ylabels=True,
        legend_fontsize=10.5,
        show_summary_inset=True,
        fig=fig,
    )
    add_robust_masked_legend(
        fig,
        ax_a,
    )
    ax_a.text(
        0.01, 1.1, "(a)",
        transform=ax_a.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax_a.set_title(r"$\Delta$BGWS × $\Delta$Runoff", fontsize=19, pad=7, fontweight="bold",)

    # --------------------
    # panel b
    # --------------------
    ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    plot_bgws_quadrant_map_on_axis(
        ax=ax_b,
        ds_dict=ds_dict,
        model=model,
        period=period,
        change_var="tran_mean",
        threshold_bgws=threshold_bgws,
        threshold_change=threshold_change,
        magnitude_source=magnitude_source,
        min_agree_models=min_agree_models,
        min_participation_frac=min_participation_frac,
        agreement_mode=agreement_mode,
        stipple_step=stipple_step,
        scatter_size=scatter_size,
        show_category_legend=False,
        show_uncertainty_legend=False,
        show_ylabels=False,
        legend_fontsize=10.5,
        fig=fig,
        show_summary_inset=True,
    )
    add_robust_masked_legend(
        fig,
        ax_b,
    )
    ax_b.text(
        0.01, 1.1, "(b)",
        transform=ax_b.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax_b.set_title(r"$\Delta$BGWS × $\Delta$Transpiration", fontsize=19, pad=7, fontweight="bold")

    # --------------------
    # panel c
    # --------------------
    ax_c = fig.add_subplot(gs[1, :])
    draw_fig5_concept_table(
        ax_c,
        body_fontsize=14,
        header_fontsize=19,
        bgws_fontsize=19,
    )
    ax_c.text(
        0.006, 0.93, "(c)",
        transform=ax_c.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    if savepath:
        fname = f"fig_4_5_combined_{dpi}dpi_updated.{filetype}"
        col_uti.save_fig(fig, savepath, fname, dpi=dpi)
        print(f"Figure saved under {savepath}{fname}")

    plt.show()
    return fig

import matplotlib.ticker as mticker

def figure4(
    ds_dict,
    model,
    period='ssp370-historical',
    threshold_bgws=0,
    threshold_change=0,
    magnitude_source='joint',
    dpi=150,
    filetype='pdf',
    savepath=None
):
    """
    Creates a combined figure:
    Panel a = BGWS × MRRO quadrant map
    Panel b = BGWS × TRAN quadrant map
    """

    fig = plt.figure(figsize=(26, 20))
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1, 1],
        hspace=0.22
    )

    # ----------------------------------------------------
    # PANEL a — BGWS × MRRO
    # ----------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    plot_bgws_quadrant_map_on_axis(
        ax_a, ds_dict, model, period,
        change_var='mrro',
        threshold_bgws=threshold_bgws,
        threshold_change=threshold_change,
        magnitude_source=magnitude_source
    )

    ax_a.text(0.005, 0.92, "(a)",
            transform=ax_a.transAxes,
            fontsize=34, #fontweight='bold',
            ha='left', va='bottom', zorder=20)

    # ----------------------------------------------------
    # PANEL b — BGWS × TRAN
    # ----------------------------------------------------
    ax_b = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    plot_bgws_quadrant_map_on_axis(
        ax_b, ds_dict, model, period,
        change_var='tran',
        threshold_bgws=threshold_bgws,
        threshold_change=threshold_change,
        magnitude_source=magnitude_source
    )

    ax_b.text(0.005, 0.92, "(b)",
            transform=ax_b.transAxes,
            fontsize=34,# fontweight='bold',
            ha='left', va='bottom', zorder=20)

    # ----------------------------------------------------
    # SAVE
    # ----------------------------------------------------
    if savepath:
        fname = f"fig_4_{dpi}dpi_updated.{filetype}"
        col_uti.save_fig(fig, savepath, fname, dpi=dpi)
        print(f"Figure saved under {savepath}{fname}")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

def figure_S1(
    ds_dict_ens,
    ds_dict_obs,
    dpi=300,
    filetype="pdf",
    filepath=None
):

    fig = plt.figure(figsize=(30, 18))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[3, 1],
        height_ratios=[1, 1],
        wspace=0.08,
        hspace=0.4
    )

    # ---------------------------------------------------
    # (a) ERA5_land BGWS map
    # ---------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    plot_bgws_map_S1(
        fig=fig,
        ax=ax_a,
        ds_dict_obs=ds_dict_obs,
        model="ERA5_land",
        period="historical"
    )
    ax_a.text(0.005, 1.01, "(a)",
            transform=ax_a.transAxes,
            fontsize=28, #fontweight='bold',
            ha='left', va='bottom', zorder=20)

    # --- fine-tune panel (a) ---
    adjust_panel_position(
        ax_a,
        dx=0.00,     # move left/right
        dy=0.00,     # move up/down
        dw=0.05,     # change width
        dh=0.08      # change height
    )

    # ---------------------------------------------------
    # (b) ERA5_land scatter
    # ---------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    plot_scatter_S1(
        ax=ax_b,
        reference=ds_dict_obs["ERA5_land"],
        model=ds_dict_ens["12 model ensemble mean"],
        variable="bgws",
        ref_label="ERA5_land",
        model_label="Ensemble Mean"
    )
    ax_b.text(0.005, 1.01, "(b)",
            transform=ax_b.transAxes,
            fontsize=28, #fontweight='bold',
            ha='left', va='bottom', zorder=20)

    # --- fine-tune panel (b) ---
    adjust_panel_position(
        ax_b,
        dx=0.00,
        dy=0.046,
        dw=-0.01,
        dh=-0.011
    )

    # ---------------------------------------------------
    # (c) OBS BGWS map
    # ---------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    plot_bgws_map_S1(
        fig=fig,
        ax=ax_c,
        ds_dict_obs=ds_dict_obs,
        model="OBS",
        period="historical"
    )
    ax_c.text(0.005, 1.01, "(c)",
            transform=ax_c.transAxes,
            fontsize=28, #fontweight='bold',
            ha='left', va='bottom', zorder=20)

    # --- fine-tune panel (c) ---
    adjust_panel_position(
        ax_c,
        dx=0.00,     # move left/right
        dy=0.00,     # move up/down
        dw=0.05,     # change width
        dh=0.08      # change height
    )

    # ---------------------------------------------------
    # (d) OBS scatter
    # ---------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    plot_scatter_S1(
        ax=ax_d,
        reference=ds_dict_obs["OBS"],
        model=ds_dict_ens["12 model ensemble mean"],
        variable="bgws",
        ref_label="OBS",
        model_label="Ensemble Mean"
    )
    ax_d.text(0.005, 1.01, "(d)",
            transform=ax_d.transAxes,
            fontsize=28, #fontweight='bold',
            ha='left', va='bottom', zorder=20)

    # --- fine-tune panel (d) ---
    adjust_panel_position(
        ax_d,
        dx=0.00,
        dy=0.046,
        dw=-0.01,
        dh=-0.011
    )

    # ---------------------------------------------------
    # Final layout + save
    # ---------------------------------------------------
    #plt.tight_layout()

    if filepath is not None:
        filename = f"figure_S1_{dpi}dpi.{filetype}"
        plt.savefig(
            f"{filepath}{filename}",
            dpi=dpi,
            bbox_inches="tight"
        )
        print(f"Figure saved under {filepath}{filename}")

    plt.show()




def plot_bgws_map_S1(fig, ax, ds_dict_obs, model, period):
    """
    model must be 'ERA5_land' or 'OBS'
    """

    # Clean member_id if present
    ds_dict_cleaned = {
        name: ds.drop('member_id', errors='ignore')
        for name, ds in ds_dict_obs.items()
    }

    ds = ds_dict_cleaned.get(model)
    if ds is None:
        raise ValueError(f"Model '{model}' not found in ds_dict_obs.")

    variable = "bgws"

    # --- FORCE ensemble colormap limits ---
    cmap_var = "bgws_ensmean"

    vmin = col_map_limits[period][cmap_var]["vmin"]
    vmax = col_map_limits[period][cmap_var]["vmax"]
    steps = col_map_limits[period][cmap_var]["steps"]

    cmap, cmap_norm = col_uti.create_colormap(
        cmap_var, period, vmin, vmax, steps
    )

    img = ds[variable].plot(
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        add_colorbar=False
    )

    ax.coastlines()
    ax.tick_params(axis="both", which="major", labelsize=22)

    # ✅ LOCK MAP EXTENT HERE (THIS IS THE RIGHT PLACE)
    #ax.set_global()
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())


    gridlines = ax.gridlines(
        draw_labels=True, color="black", alpha=0.1, linestyle="--"
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 20}
    gridlines.ylabel_style = {"size": 20}

    # Colorbar
    extend = 'both'
    cbar = fig.colorbar(img, ax=ax, orientation='horizontal', fraction=0.046, pad=0.1, extend=extend, drawedges=True)
    display_variable = col_uti.get_global_map_var_name(period, variable)
    cbar.set_label(display_variable, fontsize=26, weight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=24)
    cbar_ticks_steps = steps 
    cbar_ticks = np.arange(vmin, vmax+cbar_ticks_steps, cbar_ticks_steps)
    cbar_ticklabels = [f"{tick:.0f}" if abs(tick) > 1e-10 else "0" for tick in cbar_ticks] 
    # Set the ticks and labels on the colorbar
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels)
    
from matplotlib.ticker import FixedLocator

def plot_scatter_S1(ax, reference, model, variable, ref_label, model_label):
    reference_data = reference[variable].values.flatten()
    model_data = model[variable].values.flatten()

    valid_mask = np.isfinite(reference_data) & np.isfinite(model_data)
    reference_data = reference_data[valid_mask]
    model_data = model_data[valid_mask]

    global_mean_reference = comp_stats.compute_spatial_statistic(
        reference, "mean"
    )[variable].values

    global_mean_model = comp_stats.compute_spatial_statistic(
        model, "mean"
    )[variable].values

    if variable == "bgws":
        lims = [-100, 100]
        unit = "%"
    else:
        lims = [0, 15]
        unit = "mm"

    correlation, _ = pearsonr(reference_data, model_data)

    xy = np.vstack([reference_data, model_data])
    kde = gaussian_kde(xy)(xy)
    kde_normalized = (kde - kde.min()) / (kde.max() - kde.min())

    ax.scatter(
        reference_data,
        model_data,
        alpha=np.clip(kde_normalized, 0.1, 1.0),
        color="darkblue",
        s=25,
        edgecolor="darkblue"
    )

    ax.scatter(
        global_mean_reference,
        global_mean_model,
        color="red", s=90, edgecolor="red", label="Global mean"
    )

    ax.plot(lims, lims, "k-", lw=0.75)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    if ref_label == "ERA5_land":
        ref_display = "ERA5-Land"
    else:
        ref_display = "Observation-based"

    if model_label == "12 model ensemble mean":
        model_display = "CMIP6 ensemble mean"
    else:
        model_display = model_label

    ax.set_xlabel(
        f"{ref_display} BGWS [{unit}]", fontsize=26,
        labelpad=12,
        fontweight="bold"  
    )

    ax.set_ylabel(
        f"{model_display} BGWS [{unit}]", fontsize=26, 
        labelpad=12,
        fontweight="bold"   
    )


    ax.text(
        0.05, 0.95,
        rf"$\mathrm{{Pearson's\ }} r = {correlation:.2f}$",
        fontsize=20,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="white",
                  edgecolor="black",
                  boxstyle="round,pad=0.3")
    )

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", which="major", labelsize=18)
    
    # Define ticks explicitly (aligned with 1:1 line)
    ticks = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
    
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    # Hide the lowest y-tick label (-100) so it appears only once
    yticklabels = ["" if t == -100 else f"{t:g}" for t in ticks]
    ax.set_yticklabels(yticklabels)
    
    #ax.tick_params(axis="both", which="major", labelsize=18)


def adjust_panel_position(ax, dx=0.0, dy=0.0, dw=0.0, dh=0.0):
    """
    Fine-tune axis position.

    dx : shift right (+) / left (-)
    dy : shift up (+) / down (-)
    dw : change width
    dh : change height
    """
    pos = ax.get_position()
    ax.set_position([
        pos.x0 + dx,
        pos.y0 + dy,
        pos.width + dw,
        pos.height + dh
    ])


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_bgws_validation_figure(
    data_dict,
    var_name=None,
    ens_key="12 model ensemble mean",
    era_key="ERA5_land",
    obs_key="OBS",
    mode="change",  # "change" or "state"
    eps=1.0,
    map_titles=None,
    cmap_main="PRGn_r",   # ignored if use_segmented_cmap=True
    vmin=None,
    vmax=None,
    cbar_label=None,
    lon_ticks=(-120, -60, 0, 60, 120),
    lat_ticks=(-40, 0, 40, 80),
    figsize=(19, 10.0),
    scatter_s=5,
    scatter_alpha=0.22,
    scatter_color="0.35",
    show_fit_line=False,
    show_one_to_one=True,
    scatter_percentile_range=(2.5, 97.5),
    savepath=None,
    dpi=300,
    scatter_ylabel="CMIP6 ensemble mean BGWS [ppts]",
    scatter_xlabel_era="ERA5-Land BGWS [ppts]",
    scatter_xlabel_obs="Observation-based BGWS [ppts]",
    use_segmented_cmap=True,
    map_levels=None,
    map_bin_width=None,
    cbar_tick_step=None,
):
    """
    2-row validation figure:
      row 1: ensemble mean, ERA5-Land, OBS maps
      row 2: agreement map (left), two larger square scatter panels (right)

    Returns
    -------
    fig, outdict
    """

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------
    def _get_da(obj, var_name=None):
        if isinstance(obj, xr.DataArray):
            return obj
        if isinstance(obj, xr.Dataset):
            if var_name is None:
                if len(obj.data_vars) == 1:
                    return obj[list(obj.data_vars)[0]]
                raise ValueError(
                    "Dataset has multiple variables. Please set var_name explicitly."
                )
            return obj[var_name]
        raise TypeError(f"Unsupported object type: {type(obj)}")

    def _infer_lat_lon_names(da):
        lat_candidates = ["lat", "latitude", "y"]
        lon_candidates = ["lon", "longitude", "x"]

        lat_name = next((k for k in lat_candidates if k in da.coords), None)
        lon_name = next((k for k in lon_candidates if k in da.coords), None)

        if lat_name is None or lon_name is None:
            raise ValueError("Could not infer lat/lon coordinate names.")
        return lat_name, lon_name

    def _add_panel_label(ax, label, dx=0.00, dy=0.008, fontsize=20):
        fig = ax.figure
        pos = ax.get_position()
        fig.text(
            pos.x0 - dx, pos.y1 + dy,
            label,
            ha="left", va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )

    def _add_map_style(ax, left_labels=True, bottom_labels=True):
        ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="0.4")

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="0.5",
            alpha=0.6,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)
        gl.xformatter = LongitudeFormatter(zero_direction_label=False)
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {"size": 14}
        gl.ylabel_style = {"size": 14}

    def _sign_with_eps(da, eps):
        return xr.where(da > eps, 1, xr.where(da < -eps, -1, 0))

    def _compute_agreement_map(ens, era, obs, eps):
        """
        Classes:
          0 = weak / near-zero in at least one dataset
          1 = all three agree positive
          2 = all three agree negative
          3 = partial agreement
              (exactly one of ERA5/OBS disagrees with ensemble sign)
          4 = no agreement
              (both ERA5 and OBS disagree with ensemble sign)
        """
        s_ens = _sign_with_eps(ens, eps)
        s_era = _sign_with_eps(era, eps)
        s_obs = _sign_with_eps(obs, eps)

        valid = np.isfinite(ens) & np.isfinite(era) & np.isfinite(obs)
        out = xr.full_like(s_ens, np.nan, dtype=float)

        any_zero = (s_ens == 0) | (s_era == 0) | (s_obs == 0)
        all_pos = (s_ens == 1) & (s_era == 1) & (s_obs == 1)
        all_neg = (s_ens == -1) & (s_era == -1) & (s_obs == -1)

        # partial agreement: exactly one of ERA5/OBS differs from ensemble sign
        partial = (
            (s_ens != 0) & (s_era != 0) & (s_obs != 0) &
            (
                ((s_era == s_ens) & (s_obs != s_ens)) |
                ((s_obs == s_ens) & (s_era != s_ens))
            )
        )

        # no agreement: BOTH ERA5 and OBS disagree with the ensemble sign
        no_agree = (
            valid &
            (s_ens != 0) & (s_era != 0) & (s_obs != 0) &
            (s_era != s_ens) & (s_obs != s_ens)
        )

        out = xr.where(valid & any_zero, 0, out)
        out = xr.where(valid & all_pos, 1, out)
        out = xr.where(valid & all_neg, 2, out)
        out = xr.where(valid & partial, 3, out)
        out = xr.where(valid & no_agree, 4, out)

        return out

    def _flatten_valid(a, b):
        aa, bb = xr.align(a, b, join="inner")
        valid = np.isfinite(aa) & np.isfinite(bb)
        x = aa.where(valid).values.ravel()
        y = bb.where(valid).values.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    def _pearson_r(x, y):
        if len(x) < 2:
            return np.nan
        return np.corrcoef(x, y)[0, 1]

    def _get_scatter_limits(arr, prange=(2.5, 97.5)):
        lo = np.nanpercentile(arr, prange[0])
        hi = np.nanpercentile(arr, prange[1])

        lo = min(lo, 0.0)
        hi = max(hi, 0.0)

        if hi <= lo:
            hi = lo + 1.0

        pad = 0.04 * (hi - lo)
        return (lo - pad, hi + pad)

    def _agreement_area_percentages(agreement):
        lat_name, lon_name = _infer_lat_lon_names(agreement)
        weights = np.cos(np.deg2rad(agreement[lat_name]))
        weights_2d = xr.broadcast(weights, agreement)[0]
    
        valid = np.isfinite(agreement)
        total = weights_2d.where(valid).sum().values.item()
    
        if total == 0:
            return np.full(5, np.nan)
    
        out = []
        for i in range(5):
            w = weights_2d.where(valid & (agreement == i)).sum().values.item()
            out.append(100.0 * w / total)
        return np.array(out)

    def _scatter_panel(
        ax,
        x,
        y,
        xlabel,
        ylabel=None,
        title="",
        show_ylabel=True,
        lims=None,
    ):
        ax.scatter(
            x, y,
            s=scatter_s,
            alpha=scatter_alpha,
            color=scatter_color,
            edgecolors="none",
            rasterized=True,
        )

        ax.axhline(0, color="0.25", lw=1.0, ls=":")
        ax.axvline(0, color="0.25", lw=1.0, ls=":")

        if show_one_to_one:
            ax.plot([lims[0], lims[1]], [lims[0], lims[1]], color="k", lw=1.0, ls="--")

        if show_fit_line and len(x) > 1:
            p = np.polyfit(x, y, 1)
            xx = np.linspace(lims[0], lims[1], 200)
            yy = p[0] * xx + p[1]
            ax.plot(xx, yy, color="tab:red", lw=1.2)

        xm = np.nanmean(x)
        ym = np.nanmean(y)
        
        mean_handle = ax.scatter(
            [xm], [ym],
            s=70, color="red", edgecolor="black", linewidth=0.5, zorder=10,
            label="spatial mean",
        )

        r = _pearson_r(x, y)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_box_aspect(1)

        ax.set_xlabel(xlabel, fontsize=15)
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=15)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

        ax.set_title(title, fontsize=18, fontweight="bold", pad=8)
        ax.tick_params(labelsize=14)
        ax.grid(True, ls="--", lw=0.45, alpha=0.45)

        mean_handle = Line2D(
            [], [], linestyle="none", marker="o", markersize=8,
            markerfacecolor="red", markeredgecolor="black",
            label="spatial mean"
        )
        
        pearson_handle = Line2D(
            [], [], linestyle="none",
            label=rf"Pearson's $r$ = {r:.2f}"
        )
        
        ax.legend(
            handles=[pearson_handle, mean_handle],
            loc="lower right",
            bbox_to_anchor=(0.99, 0.01),
            frameon=True,
            framealpha=0.92,
            edgecolor="0.75",
            fontsize=13.5,
            borderpad=0.45,
            handlelength=1.0,
            handletextpad=0.5,
            labelspacing=0.4,
        )

    # ---------------------------------------------------------
    # load and align data
    # ---------------------------------------------------------
    da_ens = _get_da(data_dict[ens_key], var_name)
    da_era = _get_da(data_dict[era_key], var_name)
    da_obs = _get_da(data_dict[obs_key], var_name)

    da_ens, da_era, da_obs = xr.align(da_ens, da_era, da_obs, join="inner")
    _infer_lat_lon_names(da_ens)

    if map_titles is None:
        map_titles = ["12-model ensemble mean", "ERA5-Land", "Obs.-based product"]

    if cbar_label is None:
        cbar_label = r"$\Delta$BGWS [ppts]" if mode == "change" else r"BGWS [%]"

    if mode == "change":
        agree_labels = [
            "All agree: pos. change",
            "All agree: neg. change",
            "Partial agreement",
            "Ref. datasets disagree",
            f"Near-zero signal (|x| < {eps:g})",
            "",
        ]
    else:
        agree_labels = [
            "All agree: BGWS > 0",
            "All agree: BGWS < 0",
            "Partial agreement",
            "Ref. datasets disagree",
            f"Near-zero signal (|x| < {eps:g})",
            "",
        ]

    agreement = _compute_agreement_map(da_ens, da_era, da_obs, eps=eps)

    x_era, y_era = _flatten_valid(da_era, da_ens)
    x_obs, y_obs = _flatten_valid(da_obs, da_ens)

    all_scatter = np.concatenate([x_era, y_era, x_obs, y_obs])
    all_scatter = all_scatter[np.isfinite(all_scatter)]
    lims = _get_scatter_limits(all_scatter, prange=scatter_percentile_range)

    # ---------------------------------------------------------
    # segmented colormap for top-row maps
    # ---------------------------------------------------------
    if use_segmented_cmap:
        under = (15/255, 115/255, 15/255)            # deeper green
        deep_negative = (30/255, 130/255, 30/255)    # green   -> value -6
        light_negative = (240/255, 255/255, 240/255) # light green -> around -1..0

        midpoint = (1.0, 1.0, 1.0)                   # around 0

        light_positive = (240/255, 250/255, 255/255) # light blue -> around 0..+1
        deep_positive = (55/255, 140/255, 225/255)   # blue   -> value +6
        over = (40/255, 125/255, 210/255)            # deeper blue

        if map_levels is None:
            if vmin is None or vmax is None:
                raise ValueError(
                    "For segmented cmap, please provide vmin and vmax (or map_levels)."
                )

            if map_bin_width is None:
                map_bin_width = 1.0

            map_levels = np.arange(vmin, vmax + map_bin_width, map_bin_width)

            # make sure vmax is included exactly
            if map_levels[-1] < vmax:
                map_levels = np.append(map_levels, vmax)

        # anchor colors at selected values
        # normalized to [0,1] assuming symmetric range here
        if map_bin_width is None:
            map_bin_width = map_levels[1] - map_levels[0]

        anchor_vals = np.array(
            [vmin, -map_bin_width, 0.0, map_bin_width, vmax],
            dtype=float
        )
        anchor_pos = (anchor_vals - vmin) / (vmax - vmin)

        anchor_colors = [
            deep_negative,   # vmin
            light_negative,  # -1
            midpoint,        # 0
            light_positive,  # +1
            deep_positive,   # vmax
        ]

        base_cmap = LinearSegmentedColormap.from_list(
            "bgws_segmented_interp",
            list(zip(anchor_pos, anchor_colors))
        )

        # 12 interval centers for [-6,-5], ..., [5,6]
        bin_centers = 0.5 * (map_levels[:-1] + map_levels[1:])
        sample_pos = (bin_centers - vmin) / (vmax - vmin)
        interval_colors = [base_cmap(p) for p in sample_pos]

        # need 12 interior colors + 2 extension bins bookkeeping
        cmap_top = ListedColormap(
            [interval_colors[0]] + interval_colors + [interval_colors[-1]]
        )
        cmap_top.set_under(under)
        cmap_top.set_over(over)

        norm_top = BoundaryNorm(map_levels, cmap_top.N, extend="both")
    else:
        cmap_top = cmap_main
        norm_top = None

    # ---------------------------------------------------------
    # figure layout
    # ---------------------------------------------------------
    fig = plt.figure(figsize=figsize)

    # more space between rows -> lower panels sit a bit lower
    outer = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[1.0, 0.98],
        hspace=0.4,
    )

    top = outer[0].subgridspec(1, 3, wspace=0.04)

    # bottom row split into two blocks:
    # left = agreement map
    # right = two scatter panels
    bottom = outer[1].subgridspec(
        1, 2,
        width_ratios=[1.34, 1.1],
        wspace=0.16,   # larger gap between agreement map and panel (e)
    )

    bottom_right = bottom[0, 1].subgridspec(
        1, 2,
        wspace=0.06,   # very small gap between the two scatters
    )

    proj = ccrs.PlateCarree()

    ax1 = fig.add_subplot(top[0, 0], projection=proj)
    ax2 = fig.add_subplot(top[0, 1], projection=proj)
    ax3 = fig.add_subplot(top[0, 2], projection=proj)

    ax4 = fig.add_subplot(bottom[0, 0], projection=proj)
    ax5 = fig.add_subplot(bottom_right[0, 0])
    ax6 = fig.add_subplot(bottom_right[0, 1], sharey=ax5)

    # move scatter panels slightly upward for better alignment with panel (d)
    dy = 0.002
    for ax in (ax5, ax6):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    # ---------------------------------------------------------
    # top row maps
    # ---------------------------------------------------------
    map_axes = [ax1, ax2, ax3]
    map_fields = [da_ens, da_era, da_obs]

    for i, (ax, field, title, plabel) in enumerate(zip(map_axes, map_fields, map_titles, panel_labels[:3])):
        if use_segmented_cmap:
            im = field.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap_top,
                norm=norm_top,
                add_colorbar=False,
                rasterized=True,
            )
        else:
            im = field.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap_top,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
                rasterized=True,
            )

        _add_map_style(ax, left_labels=(i == 0), bottom_labels=True)
        ax.set_title(title, fontsize=18, fontweight="bold", pad=8)
        _add_panel_label(ax, plabel)

    # shorter shared segmented colorbar for top row
    cax = fig.add_axes([0.348, 0.54, 0.33, 0.025])
    if use_segmented_cmap:
        if cbar_tick_step is None:
            cbar_tick_step = 2 * map_bin_width if map_bin_width is not None else None

        cb_ticks = (
            np.arange(vmin, vmax + cbar_tick_step, cbar_tick_step)
            if cbar_tick_step is not None else None
        )
    else:
        cb_ticks = None

    cb = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        extend="both",
        boundaries=map_levels if use_segmented_cmap else None,
        ticks=cb_ticks,
        spacing="uniform",
    )
    cb.set_label(cbar_label, fontsize=18, fontweight="bold")
    cb.ax.tick_params(labelsize=14)

    # ---------------------------------------------------------
    # agreement map
    # ---------------------------------------------------------
    agreement_alpha = 0.8
    agree_colors = [
        (217/255, 217/255, 217/255, agreement_alpha),  # weak
        (69/255, 117/255, 180/255, agreement_alpha),   # all positive
        (26/255, 152/255, 80/255, agreement_alpha),    # all negative
        (253/255, 174/255, 97/255, agreement_alpha),   # partial
        (215/255, 48/255, 39/255, agreement_alpha),    # no agreement
    ]
    cmap_agree = ListedColormap(agree_colors)
    norm_agree = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap_agree.N)

    agreement.plot.pcolormesh(
        ax=ax4,
        transform=ccrs.PlateCarree(),
        cmap=cmap_agree,
        norm=norm_agree,
        add_colorbar=False,
        rasterized=True,
    )
    _add_map_style(ax4, left_labels=True, bottom_labels=True)
    ax4.set_title("Agreement map", fontsize=18, fontweight="bold", pad=8)
    _add_panel_label(ax4, panel_labels[3])
    agree_pct = _agreement_area_percentages(agreement)

    # inset bar chart with land-area percentages by agreement category
    inset = ax4.inset_axes([0.074, 0.05, 0.195, 0.38])

    bar_order = [1, 2, 3, 4, 0]
    
    agree_pct_plot = agree_pct[bar_order]
    inset_colors = [agree_colors[i] for i in bar_order]
    
    xbar = np.arange(len(bar_order))
    bars = inset.bar(
        xbar,
        agree_pct_plot,
        color=inset_colors,
        edgecolor="0.4",
        linewidth=0.4,
    )

    inset.set_ylabel("Land area [%]", fontsize=13)
    inset.set_xticks([])
    inset.tick_params(axis="y", labelsize=13)
    inset.tick_params(axis="x", length=0)
    
    inset.set_axisbelow(True)
    inset.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, color="0.6")
    inset.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

    ymax = np.nanmax(agree_pct)
    inset.set_ylim(0, max(10, ymax * 1.22))
    inset.set_facecolor("white")

    for spine in inset.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("0.5")

    inset.set_axisbelow(True)
    inset.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, color="0.6")
    
    for rect, val in zip(bars, agree_pct_plot):
        inset.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.6,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    legend_handles = [
        Patch(facecolor=agree_colors[1], edgecolor="none", label=agree_labels[0]),  # row1 col1
        Patch(facecolor=agree_colors[2], edgecolor="none", label=agree_labels[1]),  # row1 col2
        Patch(facecolor=agree_colors[3], edgecolor="none", label=agree_labels[2]),  # row1 col3
        Patch(facecolor=agree_colors[4], edgecolor="none", label=agree_labels[3]),  # row2 col1
        Patch(facecolor=agree_colors[0], edgecolor="none", label=agree_labels[4]),  # row2 col2
        Patch(facecolor="white", edgecolor="white", alpha=0.0, label=agree_labels[5]),  # row2 col3 blank
    ]

    ax4.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=14.0,
        frameon=False,
        handlelength=1.4,
        columnspacing=2.0,
        handletextpad=0.6,
    )

    # ---------------------------------------------------------
    # scatter panels
    # ---------------------------------------------------------
    _scatter_panel(
        ax5,
        x_era, y_era,
        xlabel=scatter_xlabel_era,
        ylabel=scatter_ylabel,
        title="ERA5-Land",
        show_ylabel=True,
        lims=lims,
    )
    _add_panel_label(ax5, panel_labels[4])

    _scatter_panel(
        ax6,
        x_obs, y_obs,
        xlabel=scatter_xlabel_obs,
        ylabel=scatter_ylabel,
        title="Obs.-based",
        show_ylabel=False,
        lims=lims,
    )
    _add_panel_label(ax6, panel_labels[5])

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, {
        "agreement": agreement,
        "scatter_era": (x_era, y_era),
        "scatter_obs": (x_obs, y_obs),
        "scatter_limits": lims,
    }

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_partitioning_validation_figure(
    data_dict,
    var_name,
    ens_key="12 model ensemble mean",
    era_key="ERA5_land",
    obs_key="OBS",
    map_titles=None,
    vmin=0,
    vmax=60,
    map_bin_width=5,
    cbar_tick_step=10,
    cbar_label=None,
    lon_ticks=(-120, -60, 0, 60, 120),
    lat_ticks=(-40, 0, 40, 80),
    figsize=(19, 8.8),
    scatter_s=5,
    scatter_alpha=0.22,
    scatter_color="0.35",
    show_fit_line=False,
    show_one_to_one=True,
    scatter_percentile_range=(2.5, 97.5),
    savepath=None,
    dpi=300,
    scatter_ylabel=None,
    scatter_xlabel_era=None,
    scatter_xlabel_obs=None,
):
    """
    Validation figure for positive partitioning variables such as:
      - r_over_p_mean
      - et_over_p_mean

    Layout:
      row 1: ensemble mean, ERA5-Land, OBS maps
      row 2: two scatter panels
    """

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------
    def _get_da(obj, var_name=None):
        if isinstance(obj, xr.DataArray):
            return obj
        if isinstance(obj, xr.Dataset):
            if var_name is None:
                if len(obj.data_vars) == 1:
                    return obj[list(obj.data_vars)[0]]
                raise ValueError("Dataset has multiple variables. Please set var_name explicitly.")
            return obj[var_name]
        raise TypeError(f"Unsupported object type: {type(obj)}")

    def _infer_lat_lon_names(da):
        lat_candidates = ["lat", "latitude", "y"]
        lon_candidates = ["lon", "longitude", "x"]

        lat_name = next((k for k in lat_candidates if k in da.coords), None)
        lon_name = next((k for k in lon_candidates if k in da.coords), None)

        if lat_name is None or lon_name is None:
            raise ValueError("Could not infer lat/lon coordinate names.")
        return lat_name, lon_name

    def _add_panel_label(ax, label, dx=0.00, dy=0.008, fontsize=20):
        fig = ax.figure
        pos = ax.get_position()
        fig.text(
            pos.x0 - dx, pos.y1 + dy,
            label,
            ha="left", va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )

    def _add_map_style(ax, left_labels=True, bottom_labels=True):
        ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="0.4")

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="0.5",
            alpha=0.6,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)
        gl.xformatter = LongitudeFormatter(zero_direction_label=False)
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {"size": 14}
        gl.ylabel_style = {"size": 14}

    def _flatten_valid(a, b):
        aa, bb = xr.align(a, b, join="inner")
        valid = np.isfinite(aa) & np.isfinite(bb)
        x = aa.where(valid).values.ravel()
        y = bb.where(valid).values.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    def _pearson_r(x, y):
        if len(x) < 2:
            return np.nan
        return np.corrcoef(x, y)[0, 1]

    def _get_scatter_limits(arr, prange=(2.5, 97.5)):
        lo = np.nanpercentile(arr, prange[0])
        hi = np.nanpercentile(arr, prange[1])

        lo = min(lo, 0.0)
        hi = max(hi, 0.0)

        if hi <= lo:
            hi = lo + 1.0

        pad = 0.04 * (hi - lo)
        return (lo - pad, hi + pad)

    def _build_positive_segmented_cmap(var_name, vmin, vmax, map_levels):
        # beige -> dark green for ET/P
        if var_name == "et_over_p_mean":
            light_positive = (246/255, 232/255, 195/255)
            deep_positive = (0/255, 60/255, 48/255)
            over = (0/255, 45/255, 30/255)
    
        # beige -> dark blue for R/P
        elif var_name == "r_over_p_mean":
            light_positive = (246/255, 232/255, 195/255)
            deep_positive = (0/255, 62/255, 125/255)
            over = (0/255, 52/255, 105/255)
    
        else:
            raise ValueError(
                "This function currently supports only 'et_over_p_mean' and 'r_over_p_mean'."
            )
    
        n_intervals = len(map_levels) - 1
    
        base_cmap = LinearSegmentedColormap.from_list(
            f"{var_name}_segmented",
            [light_positive, deep_positive]
        )
    
        # sample away from exact 0 so first two bins are visually distinct
        sample_pos = np.linspace(0.08, 1.0, n_intervals)
        interval_colors = [base_cmap(p) for p in sample_pos]
    
        # for extend='max', BoundaryNorm needs one extra color
        cmap = ListedColormap(interval_colors + [interval_colors[-1]])
        cmap.set_over(over)
        cmap.set_under(interval_colors[0])
    
        norm = BoundaryNorm(map_levels, cmap.N, extend="max")
        return cmap, norm

    def _scatter_panel(
        ax,
        x,
        y,
        xlabel,
        ylabel=None,
        title="",
        show_ylabel=True,
        lims=None,
    ):
        ax.scatter(
            x, y,
            s=scatter_s,
            alpha=scatter_alpha,
            color=scatter_color,
            edgecolors="none",
            rasterized=True,
        )

        ax.axhline(0, color="0.25", lw=1.0, ls=":")
        ax.axvline(0, color="0.25", lw=1.0, ls=":")

        if show_one_to_one:
            ax.plot([lims[0], lims[1]], [lims[0], lims[1]], color="k", lw=1.0, ls="--")

        if show_fit_line and len(x) > 1:
            p = np.polyfit(x, y, 1)
            xx = np.linspace(lims[0], lims[1], 200)
            yy = p[0] * xx + p[1]
            ax.plot(xx, yy, color="tab:red", lw=1.2)

        xm = np.nanmean(x)
        ym = np.nanmean(y)
        ax.scatter(
            [xm], [ym],
            s=70, color="red", edgecolor="black", linewidth=0.5, zorder=10
        )

        r = _pearson_r(x, y)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_box_aspect(1)

        ax.set_xlabel(xlabel, fontsize=15)
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=15)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

        ax.set_title(title, fontsize=18, fontweight="bold", pad=8)
        ax.tick_params(labelsize=14)
        ax.grid(True, ls="--", lw=0.45, alpha=0.45)

        mean_handle = Line2D(
            [], [], linestyle="none", marker="o", markersize=8,
            markerfacecolor="red", markeredgecolor="black",
            label="spatial mean"
        )
        pearson_handle = Line2D(
            [], [], linestyle="none",
            label=rf"Pearson's $r$ = {r:.2f}"
        )

        ax.legend(
            handles=[pearson_handle, mean_handle],
            loc="lower right",
            bbox_to_anchor=(0.99, 0.01),
            frameon=True,
            framealpha=0.92,
            edgecolor="0.75",
            fontsize=13.5,
            borderpad=0.45,
            handlelength=1.0,
            handletextpad=0.5,
            labelspacing=0.4,
        )

    # ---------------------------------------------------------
    # load and align data
    # ---------------------------------------------------------
    da_ens = _get_da(data_dict[ens_key], var_name)
    da_era = _get_da(data_dict[era_key], var_name)
    da_obs = _get_da(data_dict[obs_key], var_name)

    da_ens, da_era, da_obs = xr.align(da_ens, da_era, da_obs, join="inner")

    # convert fractions to percent
    da_ens = da_ens * 100.0
    da_era = da_era * 100.0
    da_obs = da_obs * 100.0
    
    _infer_lat_lon_names(da_ens)

    if map_titles is None:
        map_titles = ["12-model ensemble mean", "ERA5-Land", "Obs.-based product"]

    if cbar_label is None:
        if var_name == "et_over_p_mean":
            cbar_label = r"$E_t/P$ [%]"
        elif var_name == "r_over_p_mean":
            cbar_label = r"$R/P$ [%]"
        else:
            cbar_label = "[%]"

    if scatter_ylabel is None:
        if var_name == "et_over_p_mean":
            scatter_ylabel = r"CMIP6 ensemble mean $E_t/P$ [%]"
        elif var_name == "r_over_p_mean":
            scatter_ylabel = r"CMIP6 ensemble mean $R/P$ [%]"

    if scatter_xlabel_era is None:
        if var_name == "et_over_p_mean":
            scatter_xlabel_era = r"ERA5-Land $E_t/P$ [%]"
        elif var_name == "r_over_p_mean":
            scatter_xlabel_era = r"ERA5-Land $R/P$ [%]"

    if scatter_xlabel_obs is None:
        if var_name == "et_over_p_mean":
            scatter_xlabel_obs = r"Obs.-based $E_t/P$ [%]"
        elif var_name == "r_over_p_mean":
            scatter_xlabel_obs = r"Obs.-based $R/P$ [%]"

    x_era, y_era = _flatten_valid(da_era, da_ens)
    x_obs, y_obs = _flatten_valid(da_obs, da_ens)

    all_scatter = np.concatenate([x_era, y_era, x_obs, y_obs])
    all_scatter = all_scatter[np.isfinite(all_scatter)]
    lims = _get_scatter_limits(all_scatter, prange=scatter_percentile_range)

    # ---------------------------------------------------------
    # colormap
    # ---------------------------------------------------------
    map_levels = np.arange(vmin, vmax + map_bin_width, map_bin_width)
    if map_levels[-1] < vmax:
        map_levels = np.append(map_levels, vmax)

    cmap_top, norm_top = _build_positive_segmented_cmap(
        var_name=var_name,
        vmin=vmin,
        vmax=vmax,
        map_levels=map_levels,
    )

    cb_ticks = np.arange(vmin, vmax + cbar_tick_step, cbar_tick_step)

    # ---------------------------------------------------------
    # figure layout
    # ---------------------------------------------------------
    fig = plt.figure(figsize=figsize)

    outer = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[1.0, 0.98],
        hspace=0.42,
    )

    top = outer[0].subgridspec(1, 3, wspace=0.04)

    # bottom row centered two scatters
    bottom = outer[1].subgridspec(
        1, 5,
        width_ratios=[0.65, 1.0, 0.08, 1.0, 0.65],
        wspace=0.0,
    )

    proj = ccrs.PlateCarree()

    ax1 = fig.add_subplot(top[0, 0], projection=proj)
    ax2 = fig.add_subplot(top[0, 1], projection=proj)
    ax3 = fig.add_subplot(top[0, 2], projection=proj)

    ax5 = fig.add_subplot(bottom[0, 1])
    ax6 = fig.add_subplot(bottom[0, 3], sharey=ax5)

    dy = 0.002
    for ax in (ax5, ax6):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    # ---------------------------------------------------------
    # top row maps
    # ---------------------------------------------------------
    map_axes = [ax1, ax2, ax3]
    map_fields = [da_ens, da_era, da_obs]

    for i, (ax, field, title, plabel) in enumerate(zip(map_axes, map_fields, map_titles, panel_labels[:3])):
        im = field.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap_top,
            norm=norm_top,
            add_colorbar=False,
            rasterized=True,
        )

        _add_map_style(ax, left_labels=(i == 0), bottom_labels=True)
        ax.set_title(title, fontsize=18, fontweight="bold", pad=8)
        _add_panel_label(ax, plabel)

    cax = fig.add_axes([0.348, 0.54, 0.33, 0.025])
    cb = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        extend="max",
        boundaries=map_levels,
        ticks=cb_ticks,
        spacing="uniform",
    )
    cb.set_label(cbar_label, fontsize=18, fontweight="bold")
    cb.ax.tick_params(labelsize=14)

    # ---------------------------------------------------------
    # scatter panels
    # ---------------------------------------------------------
    _scatter_panel(
        ax5,
        x_era, y_era,
        xlabel=scatter_xlabel_era,
        ylabel=scatter_ylabel,
        title="ERA5-Land",
        show_ylabel=True,
        lims=lims,
    )
    _add_panel_label(ax5, panel_labels[3])

    _scatter_panel(
        ax6,
        x_obs, y_obs,
        xlabel=scatter_xlabel_obs,
        ylabel=scatter_ylabel,
        title="Obs.-based",
        show_ylabel=False,
        lims=lims,
    )
    _add_panel_label(ax6, panel_labels[4])

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, {
        "scatter_era": (x_era, y_era),
        "scatter_obs": (x_obs, y_obs),
        "scatter_limits": lims,
        "map_levels": map_levels,
    }

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_historical_hydroeco_ensemble_maps(
    data,
    variables=None,
    ens_key="12 model ensemble mean",
    lon_ticks=(-120, -60, 0, 60, 120),
    lat_ticks=(-40, 0, 40, 80),
    figsize=(18, 15),
    savepath=None,
    dpi=300,
):
    """
    Plot historical mean-state maps for multiple hydroecological variables
    in a 4x3 panel layout with one segmented colorbar below each panel.
    """

    # ---------------------------------------------------------
    # defaults
    # ---------------------------------------------------------
    if variables is None:
        variables = [
            "mrro_mean",
            "tran_mean",
            "evapo_mean",
            "pr_mean",
            "pr_seasonality",
            "RX5day",
            "mrsos_mean",
            "lai_mean",
            "wue_mean",
            "vpd_mean",
            "vpd_seasonality",
            "clt_mean",
        ]

    display_names = {
        "mrro_mean": r"$R$ [mm day$^{-1}$]",
        "tran_mean": r"$E_t$ [mm day$^{-1}$]",
        "evapo_mean": r"$E_{nt}$ [mm day$^{-1}$]",
        "pr_mean": r"$P$ [mm day$^{-1}$]",
        "pr_seasonality": r"$P_{seas}$ [mm day$^{-1}$]",
        "RX5day": r"RX5day [mm]",
        "mrsos_mean": r"SM$_{surf}$ [mm]",
        "lai_mean": r"LAI [m$^2$ m$^{-2}$]",
        "wue_mean": r"WUE [gC m$^{-2}$ mm$^{-1}$]",
        "vpd_mean": r"VPD [hPa]",
        "vpd_seasonality": r"VPD$_{seas}$ [hPa]",
        "clt_mean": r"CLT [%]",
    }

    # requested colorbar setup
    cbar_specs = {
        "mrro_mean": {"vmin": 0.0, "vmax": 2.5, "step": 0.25},
        "tran_mean": {"vmin": 0.0, "vmax": 2.5, "step": 0.25},
        "evapo_mean": {"vmin": 0.0, "vmax": 2.5, "step": 0.25},
        "pr_mean": {"vmin": 0.0, "vmax": 5.0, "step": 0.5},
        "pr_seasonality": {"vmin": 0.0, "vmax": 5.0, "step": 0.5},
        "RX5day": {"vmin": 20.0, "vmax": 160.0, "step": 20.0},
        "mrsos_mean": {"vmin": 10.0, "vmax": 35.0, "step": 5.0},
        "lai_mean": {"vmin": 0.0, "vmax": 5.0, "step": 0.5},
        "wue_mean": {"vmin": 3.0, "vmax": 6.0, "step": 0.5},
        "vpd_mean": {"vmin": 0.0, "vmax": 20.0, "step": 2.5},
        "vpd_seasonality": {"vmin": 0.0, "vmax": 10.0, "step": 1.0},
        "clt_mean": {"vmin": 20.0, "vmax": 80.0, "step": 10.0},
    }

    # ---------------------------------------------------------
    # helper functions
    # ---------------------------------------------------------
    def _get_ds(obj):
        if isinstance(obj, xr.Dataset):
            return obj
        if isinstance(obj, dict):
            out = obj[ens_key]
            if isinstance(out, xr.Dataset):
                return out
            raise TypeError(f"Expected xr.Dataset at data['{ens_key}'], got {type(out)}")
        raise TypeError(f"Unsupported data type: {type(obj)}")

    def _infer_lat_lon_names(da):
        lat_candidates = ["lat", "latitude", "y"]
        lon_candidates = ["lon", "longitude", "x"]

        lat_name = next((k for k in lat_candidates if k in da.coords), None)
        lon_name = next((k for k in lon_candidates if k in da.coords), None)

        if lat_name is None or lon_name is None:
            raise ValueError("Could not infer lat/lon coordinate names.")
        return lat_name, lon_name

    def _add_map_style(ax, left_labels=True, bottom_labels=True):
        ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="0.4")

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="0.5",
            alpha=0.6,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)
        gl.xformatter = LongitudeFormatter(zero_direction_label=False)
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {"size": 13}
        gl.ylabel_style = {"size": 13}

    def _add_panel_label_inside(ax, label, fontsize=17):
        ax.text(
            0.01, 1.13, label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize,
            fontweight="bold",
            #bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2.2),
            zorder=50,
        )

    def _get_tick_labels_every_second(boundaries):
        labels = []
        for i, val in enumerate(boundaries):
            if i % 2 == 0:
                if float(val).is_integer():
                    labels.append(f"{int(val)}")
                else:
                    labels.append(f"{val:g}")
            else:
                labels.append("")
        return labels

    # ---------------------------------------------------------
    # load dataset
    # ---------------------------------------------------------
    ds = _get_ds(data)

    for var in variables:
        if var not in ds:
            raise KeyError(f"Variable '{var}' not found in dataset.")

    sample = ds[variables[0]]
    _infer_lat_lon_names(sample)

    # ---------------------------------------------------------
    # figure layout
    # ---------------------------------------------------------
    nvars = len(variables)
    ncols = 3
    nrows = int(np.ceil(nvars / ncols))

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)

    # tighter vertical spacing
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        hspace=0,
        wspace=0.04,
    )

    panel_labels = [f"({chr(97+i)})" for i in range(nvars)]

    # ---------------------------------------------------------
    # plot panels
    # ---------------------------------------------------------
    for i, var in enumerate(variables):
        row = i // ncols
        col = i % ncols

        ax = fig.add_subplot(gs[row, col], projection=proj)
        da = ds[var]

        spec = cbar_specs[var]
        vmin = spec["vmin"]
        vmax = spec["vmax"]
        steps = spec["step"]

        cmap, cmap_norm = col_uti.create_colormap(
            var=var,
            period="historical",
            vmin=vmin,
            vmax=vmax,
            steps=steps,
        )

        im = da.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=cmap_norm,
            add_colorbar=False,
            rasterized=True,
        )

        _add_map_style(
            ax,
            left_labels=(col == 0),
            bottom_labels=(row == nrows - 1),
        )

        _add_panel_label_inside(ax, panel_labels[i])

        # colorbar below each panel
        pos = ax.get_position()
        cbar_h = 0.012

        # slightly tighter between rows, but lower for last row
        if row == nrows - 1:
            cbar_pad = 0.040
        else:
            cbar_pad = 0.018

        cax = fig.add_axes([
            pos.x0 + 0.03,
            pos.y0 - cbar_pad,
            pos.width - 0.06,
            cbar_h
        ])

        boundaries = np.arange(vmin, vmax + steps, steps)
        if boundaries[-1] < vmax:
            boundaries = np.append(boundaries, vmax)

        cb = fig.colorbar(
            im,
            cax=cax,
            orientation="horizontal",
            boundaries=boundaries,
            ticks=boundaries,
            spacing="uniform",
            extend="max",
        )
        cb.ax.set_xticklabels(_get_tick_labels_every_second(boundaries))
        cb.ax.tick_params(labelsize=12, length=3, pad=1)

        # variable label below colorbar
        cb.set_label(display_names.get(var, var), fontsize=16, labelpad=4, fontweight="bold",)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_change_hydroeco_ensemble_maps(
    data,
    rel_change_data=None,
    variables=None,
    ens_key="12 model ensemble mean",
    lon_ticks=(-120, -60, 0, 60, 120),
    lat_ticks=(-40, 0, 40, 80),
    figsize=(18, 15),
    savepath=None,
    dpi=300,
    min_agree_models=8,
    stipple_step=4,
    stipple_size=5,
    stipple_facecolor="black",
    stipple_edgecolor="white",
    stipple_alpha=0.9,
    stipple_linewidth=1,
    add_stipple_legend=True,
):
    """
    Plot change maps for multiple hydroecological variables.

    Parameters
    ----------
    data : dict
        Absolute-change dict for the plotted period, including individual models
        and ensemble mean.
    rel_change_data : dict or None
        Relative-change dict for the same period, used only for
        pr_mean, mrro_mean, tran_mean if provided.
    """

    if variables is None:
        variables = [
            "mrro_mean",
            "tran_mean",
            "evapo_mean",
            "pr_mean",
            "pr_seasonality",
            "RX5day",
            "mrsos_mean",
            "lai_mean",
            "wue_mean",
            "vpd_mean",
            "vpd_seasonality",
            "clt_mean",
        ]

    relative_vars = {"pr_mean", "mrro_mean", "tran_mean"}

    display_names_abs = {
        "mrro_mean": r"$\Delta R$ [mm day$^{-1}$]",
        "tran_mean": r"$\Delta E_t$ [mm day$^{-1}$]",
        "evapo_mean": r"$\Delta E_{nt}$ [mm day$^{-1}$]",
        "pr_mean": r"$\Delta P$ [mm day$^{-1}$]",
        "pr_seasonality": r"$\Delta P_{seas}$ [mm day$^{-1}$]",
        "RX5day": r"$\Delta$RX5day [mm]",
        "mrsos_mean": r"$\Delta$SM$_{surf}$ [mm]",
        "lai_mean": r"$\Delta$LAI [m$^2$ m$^{-2}$]",
        "wue_mean": r"$\Delta$WUE [gC m$^{-2}$ mm$^{-1}$]",
        "vpd_mean": r"$\Delta$VPD [hPa]",
        "vpd_seasonality": r"$\Delta$VPD$_{seas}$ [hPa]",
        "clt_mean": r"$\Delta$CLT [ppts]",
    }

    display_names_rel = {
        "mrro_mean": r"Relative change in $R$ [%]",
        "tran_mean": r"Relative change in $E_t$ [%]",
        "pr_mean": r"Relative change in $P$ [%]",
    }

    cbar_specs_abs = {
        "mrro_mean": {"vmin": -0.5, "vmax": 0.5, "step": 0.1, "positive_only": False},
        "tran_mean": {"vmin": -0.5, "vmax": 0.5, "step": 0.1, "positive_only": False},
        "evapo_mean": {"vmin": -0.5, "vmax": 0.5, "step": 0.1, "positive_only": False},
        "pr_mean": {"vmin": -1.0, "vmax": 1.0, "step": 0.2, "positive_only": False},
        "pr_seasonality": {"vmin": -1.0, "vmax": 1.0, "step": 0.2, "positive_only": False},
        "RX5day": {"vmin": -20.0, "vmax": 20.0, "step": 5.0, "positive_only": False},
        "mrsos_mean": {"vmin": -5.0, "vmax": 5.0, "step": 1.0, "positive_only": False},
        "lai_mean": {"vmin": -1.0, "vmax": 1.0, "step": 0.2, "positive_only": False},
        "wue_mean": {"vmin": -1.5, "vmax": 1.5, "step": 0.25, "positive_only": False},
        "vpd_mean": {"vmin": 0.0, "vmax": 5.0, "step": 0.5, "positive_only": True},
        "vpd_seasonality": {"vmin": -2.0, "vmax": 2.0, "step": 0.5, "positive_only": False},
        "clt_mean": {"vmin": -10.0, "vmax": 10.0, "step": 2.0, "positive_only": False},
    }

    cbar_specs_rel = {
        "mrro_mean": {"vmin": -100.0, "vmax": 100.0, "step": 20.0, "positive_only": False},
        "tran_mean": {"vmin": -100.0, "vmax": 100.0, "step": 20.0, "positive_only": False},
        "pr_mean": {"vmin": -50.0, "vmax": 50.0, "step": 10.0, "positive_only": False},
    }

    def _get_ds(obj):
        if isinstance(obj, xr.Dataset):
            return obj
        if isinstance(obj, dict):
            out = obj[ens_key]
            if isinstance(out, xr.Dataset):
                return out
            raise TypeError(f"Expected xr.Dataset at data['{ens_key}'], got {type(out)}")
        raise TypeError(f"Unsupported data type: {type(obj)}")

    def _get_member_model_names(data_dict):
        bad_terms = ["ensemble mean", "ensemble median", "ensemble std"]
        return [k for k in data_dict if not any(term in k.lower() for term in bad_terms)]

    def _infer_lat_lon_names(da):
        lat_candidates = ["lat", "latitude", "y"]
        lon_candidates = ["lon", "longitude", "x"]
        lat_name = next((k for k in lat_candidates if k in da.coords), None)
        lon_name = next((k for k in lon_candidates if k in da.coords), None)
        if lat_name is None or lon_name is None:
            raise ValueError("Could not infer lat/lon coordinate names.")
        return lat_name, lon_name

    def _add_map_style(ax, left_labels=True, bottom_labels=True):
        ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="0.4")

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="0.5",
            alpha=0.6,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)
        gl.xformatter = LongitudeFormatter(zero_direction_label=False)
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {"size": 13}
        gl.ylabel_style = {"size": 13}

    def _add_panel_label_inside(ax, label, fontsize=17):
        ax.text(
            0.015, 1.13, label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize,
            fontweight="bold",
            zorder=50,
        )

    def _get_tick_labels_every_second(boundaries):
        labels = []
        for i, val in enumerate(boundaries):
            if i % 2 == 0:
                labels.append(f"{val:g}" if not float(val).is_integer() else f"{int(val)}")
            else:
                labels.append("")
        return labels

    def _make_diverging_change_cmap(vmin, vmax, steps):
        under = (84/255, 48/255, 5/255)
        deep_negative = (100/255, 64/255, 21/255)
        light_negative = (246/255, 232/255, 195/255)

        over = (0/255, 60/255, 48/255)
        deep_positive = (16/255, 76/255, 64/255)
        light_positive = (199/255, 234/255, 229/255)

        boundaries, n = col_uti.get_colmap_set_up(vmin, vmax, steps)
        cmap, cmap_norm = col_uti.create_two_gradient_colormap(
            deep_negative, light_negative,
            deep_positive, light_positive,
            boundaries, n, under, over
        )
        return cmap, cmap_norm

    def _make_positive_change_cmap(vmin, vmax, steps):
        light_positive = (199/255, 234/255, 229/255)
        deep_positive = (16/255, 76/255, 64/255)
        over = (0/255, 60/255, 48/255)

        boundaries, n = col_uti.get_colmap_set_up(vmin, vmax, steps)
        cmap, cmap_norm = col_uti.create_one_gradient_colormap(
            light_positive, deep_positive, boundaries, n, over
        )
        return cmap, cmap_norm

    def _get_low_sign_agreement_mask(data_dict, variable, min_agree_models=8):
        model_names = _get_member_model_names(data_dict)

        da_models = xr.concat(
            [
                data_dict[k].drop_vars("member_id", errors="ignore")[variable]
                for k in model_names
            ],
            dim="model"
        )

        da_mean = data_dict[ens_key].drop_vars("member_id", errors="ignore")[variable]

        pos_count = (da_models > 0).sum("model")
        neg_count = (da_models < 0).sum("model")
        max_same_sign = xr.where(pos_count >= neg_count, pos_count, neg_count)

        valid = np.isfinite(da_mean)
        low_agree = (max_same_sign < min_agree_models).where(valid)

        return low_agree.fillna(False), da_models.sizes["model"]

    def _plot_boolean_scatter_mask(
        ax,
        mask,
        stipple_step=4,
        scatter_size=5,
        marker="o",
        facecolor="black",
        edgecolor="white",
        alpha=0.9,
        linewidth=0.8,
        zorder=35,
    ):
        if "lat" in mask.dims and "lon" in mask.dims:
            mask_sub = mask.isel(
                lat=slice(0, None, stipple_step),
                lon=slice(0, None, stipple_step),
            )
        else:
            dims = list(mask.dims)
            d1, d2 = dims[-2], dims[-1]
            mask_sub = mask.isel({d1: slice(0, None, stipple_step), d2: slice(0, None, stipple_step)})

        pts = mask_sub.where(mask_sub).stack(points=mask_sub.dims).dropna("points")
        if pts.sizes.get("points", 0) == 0:
            return

        x = pts["lon"].values
        y = pts["lat"].values

        ax.scatter(
            x, y,
            s=scatter_size * 2.0,
            marker=marker,
            facecolors="white",
            edgecolors="white",
            alpha=1.0,
            linewidths=0.0,
            transform=ccrs.PlateCarree(),
            zorder=zorder,
            clip_on=True,
        )

        ax.scatter(
            x, y,
            s=scatter_size,
            marker=marker,
            facecolors=facecolor,
            edgecolors=edgecolor,
            alpha=alpha,
            linewidths=linewidth,
            transform=ccrs.PlateCarree(),
            zorder=zorder + 0.1,
            clip_on=True,
        )

    # validate data
    ds = _get_ds(data)
    for var in variables:
        source = rel_change_data if (rel_change_data is not None and var in relative_vars) else data
        if var not in _get_ds(source):
            raise KeyError(f"Variable '{var}' not found in plotting source.")

    sample_source = rel_change_data if rel_change_data is not None else data
    sample = _get_ds(sample_source)[variables[0 if variables[0] not in relative_vars else list(variables).index(variables[0])]]
    _infer_lat_lon_names(sample)

    nvars = len(variables)
    ncols = 3
    nrows = int(np.ceil(nvars / ncols))

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        hspace=0.0,
        wspace=0.04,
        bottom=0.13,   # a bit more space for centered legend
    )

    panel_labels = [f"({chr(97+i)})" for i in range(nvars)]
    n_models_for_legend = None
    bottom_row_axes = []

    for i, var in enumerate(variables):
        row = i // ncols
        col = i % ncols

        use_rel = (rel_change_data is not None and var in relative_vars)
        source_dict = rel_change_data if use_rel else data
        ds_plot = _get_ds(source_dict)
        da = ds_plot[var]

        spec = cbar_specs_rel[var] if use_rel else cbar_specs_abs[var]
        vmin = spec["vmin"]
        vmax = spec["vmax"]
        steps = spec["step"]
        positive_only = spec["positive_only"]

        if positive_only:
            cmap, cmap_norm = _make_positive_change_cmap(vmin, vmax, steps)
        else:
            cmap, cmap_norm = _make_diverging_change_cmap(vmin, vmax, steps)

        ax = fig.add_subplot(gs[row, col], projection=proj)

        if row == nrows - 1:
            bottom_row_axes.append(ax)

        im = da.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=cmap_norm,
            add_colorbar=False,
            rasterized=True,
        )

        _add_map_style(
            ax,
            left_labels=(col == 0),
            bottom_labels=(row == nrows - 1),
        )
        _add_panel_label_inside(ax, panel_labels[i])

        if not positive_only:
            low_agree_mask, n_models = _get_low_sign_agreement_mask(
                data_dict=source_dict,
                variable=var,
                min_agree_models=min_agree_models,
            )
            n_models_for_legend = n_models

            _plot_boolean_scatter_mask(
                ax,
                low_agree_mask,
                stipple_step=stipple_step,
                scatter_size=stipple_size,
                marker="o",
                facecolor=stipple_facecolor,
                edgecolor=stipple_edgecolor,
                alpha=stipple_alpha,
                linewidth=stipple_linewidth,
                zorder=35,
            )

        pos = ax.get_position()
        cbar_h = 0.012
        cbar_pad = 0.040 if row == nrows - 1 else 0.018

        cax = fig.add_axes([
            pos.x0 + 0.03,
            pos.y0 - cbar_pad,
            pos.width - 0.06,
            cbar_h
        ])

        boundaries = np.arange(vmin, vmax + steps, steps)
        if boundaries[-1] < vmax:
            boundaries = np.append(boundaries, vmax)

        cb = fig.colorbar(
            im,
            cax=cax,
            orientation="horizontal",
            boundaries=boundaries,
            ticks=boundaries,
            spacing="uniform",
            extend="max" if positive_only else "both",
        )
        cb.ax.set_xticklabels(_get_tick_labels_every_second(boundaries))
        cb.ax.tick_params(labelsize=12, length=3, pad=1)

        cb.set_label(
            display_names_rel[var] if use_rel else display_names_abs[var],
            fontsize=16,
            labelpad=4,
            fontweight="bold",
        )

    # centered low-agreement legend below last row using your custom style
    if add_stipple_legend and (n_models_for_legend is not None) and bottom_row_axes:
        anchor_ax = bottom_row_axes[len(bottom_row_axes) // 2]

        low_agree_handle = scatter_legend(
            anchor_ax,
            label=f"Low ensemble\nsign agreement (<{min_agree_models}/{n_models_for_legend})",
            facecolor="white",
            edgecolor="black",
            lw=0.8,
        )

        leg = fig.legend(
            handles=[low_agree_handle],
            labels=[low_agree_handle.get_label()],
            handler_map={
                type(low_agree_handle): ScatterBoxHandler(
                    ydescent_offset_box=-4,
                    box_width_offset=1.2,
                    box_height_offset=2,
                    scatter_spacing_y_offset=1.8,
                    scatter_spacing_x_offset=0.6,
                    scatter_vertical_offset=-1,
                    scatter_horizontal_offset=-4,
                )
            },
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            bbox_transform=fig.transFigure,
            frameon=False,
            fontsize=16,
            ncol=1,
        )

        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("0.5")
        frame.set_linewidth(0.6)
        frame.set_alpha(1.0)

        for t in leg.get_texts():
            t.set_fontweight("bold")

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_rx5day_ratio_map_and_scatters(
    ds_or_dict,
    ens_key="12 model ensemble mean",
    ratio_var="rx5day_ratio",
    pr_var="pr_mean",
    bgws_var="bgws_tran_mean",
    lon_ticks=(-120, -60, 0, 60, 120),
    lat_ticks=(-40, 0, 40, 80),
    figsize=(14, 9.5),
    savepath=None,
    dpi=300,
    scatter_s=10,
    scatter_alpha_min=0.08,
    scatter_alpha_max=0.85,
):
    """
    Figure with:
      - top: historical map of RX5day / annual precipitation ratio [%]
      - bottom: 2 scatter plots with common y-axis = RX5day / annual precipitation ratio [%]
                and x = pr_mean, bgws_tran_mean
    """

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------
    def _get_ds(obj):
        if isinstance(obj, xr.Dataset):
            return obj
        if isinstance(obj, dict):
            out = obj[ens_key]
            if isinstance(out, xr.Dataset):
                return out
            raise TypeError(f"Expected xr.Dataset at data['{ens_key}'], got {type(out)}")
        raise TypeError(f"Unsupported data type: {type(obj)}")

    def _infer_lat_lon_names(da):
        lat_candidates = ["lat", "latitude", "y"]
        lon_candidates = ["lon", "longitude", "x"]

        lat_name = next((k for k in lat_candidates if k in da.coords), None)
        lon_name = next((k for k in lon_candidates if k in da.coords), None)

        if lat_name is None or lon_name is None:
            raise ValueError("Could not infer lat/lon coordinate names.")
        return lat_name, lon_name

    def _add_map_style(ax, left_labels=True, bottom_labels=True):
        ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="0.4")

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="0.5",
            alpha=0.6,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)
        gl.xformatter = LongitudeFormatter(zero_direction_label=False)
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {"size": 13}
        gl.ylabel_style = {"size": 13}

    def _add_panel_label(ax, label, fontsize=18):
        ax.text(
            0.015, 0.985, label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2.0),
            zorder=50,
        )

    def _flatten_valid(a, b):
        aa, bb = xr.align(a, b, join="inner")
        valid = np.isfinite(aa) & np.isfinite(bb)
        x = aa.where(valid).values.ravel()
        y = bb.where(valid).values.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    def _pearson_r(x, y):
        if len(x) < 2:
            return np.nan
        if np.allclose(np.nanstd(x), 0) or np.allclose(np.nanstd(y), 0):
            return np.nan
        return np.corrcoef(x, y)[0, 1]

    def _density_alpha(x, y, bins=70, amin=0.08, amax=0.85):
        if len(x) == 0:
            return np.array([])

        xedges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
        yedges = np.linspace(np.nanmin(y), np.nanmax(y), bins + 1)

        H, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])

        ix = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
        iy = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)

        dens = H[ix, iy].astype(float)

        if dens.max() > dens.min():
            dens = (dens - dens.min()) / (dens.max() - dens.min())
        else:
            dens = np.ones_like(dens)

        return amin + dens * (amax - amin)

    def _scatter_panel(ax, x, y, xlabel, show_ylabel=False, is_bgws=False):
        r = _pearson_r(x, y)

        if is_bgws:
            above = x > 0
            below = x < 0

            alpha_above = _density_alpha(
                x[above], y[above],
                amin=scatter_alpha_min, amax=scatter_alpha_max
            ) if np.any(above) else np.array([])

            alpha_below = _density_alpha(
                x[below], y[below],
                amin=scatter_alpha_min, amax=scatter_alpha_max
            ) if np.any(below) else np.array([])

            if np.any(below):
                green = np.tile(np.array([[30/255, 130/255, 30/255, 1.0]]), (below.sum(), 1))
                green[:, 3] = alpha_below
                ax.scatter(
                    x[below], y[below],
                    s=scatter_s, c=green, edgecolors="none", rasterized=True
                )

            if np.any(above):
                blue = np.tile(np.array([[55/255, 140/255, 225/255, 1.0]]), (above.sum(), 1))
                blue[:, 3] = alpha_above
                ax.scatter(
                    x[above], y[above],
                    s=scatter_s, c=blue, edgecolors="none", rasterized=True
                )

            ax.axvline(0, color="0.2", lw=1.0, ls=":")
        else:
            alpha_vals = _density_alpha(
                x, y, amin=scatter_alpha_min, amax=scatter_alpha_max
            )
            gray = np.tile(np.array([[0.35, 0.35, 0.35, 1.0]]), (len(x), 1))
            gray[:, 3] = alpha_vals

            ax.scatter(
                x, y,
                s=scatter_s,
                c=gray,
                edgecolors="none",
                rasterized=True,
            )

        ax.set_xlabel(xlabel, fontsize=15, fontweight="bold")
        if show_ylabel:
            ax.set_ylabel(r"RX5day / $P_{\mathrm{annual}}$ [%]", fontsize=15, fontweight="bold")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

        ax.set_box_aspect(1)

        ax.tick_params(labelsize=13)
        ax.grid(True, ls="--", lw=0.45, alpha=0.45)

        ax.text(
            0.97, 0.94,
            rf"Pearson's $r$ = {r:.2f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=13.5,
            bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.92),
        )

    # ---------------------------------------------------------
    # load data
    # ---------------------------------------------------------
    ds = _get_ds(ds_or_dict)

    required = [ratio_var, pr_var, bgws_var]
    for v in required:
        if v not in ds:
            raise KeyError(f"Variable '{v}' not found in dataset.")

    da_ratio = ds[ratio_var] * 100
    da_pr = ds[pr_var]
    da_bgws = ds[bgws_var]

    _infer_lat_lon_names(da_ratio)

    x_pr, y_ratio_pr = _flatten_valid(da_pr, da_ratio)
    x_bgws, y_ratio_bgws = _flatten_valid(da_bgws, da_ratio)

    # ---------------------------------------------------------
    # map colormap
    # ---------------------------------------------------------
    vals = da_ratio.values
    vals = vals[np.isfinite(vals)]
    vmax = float(np.nanpercentile(vals, 99)) if vals.size else 25.0
    vmax = max(25.0, min(30.0, vmax))
    vmin = 0.0
    step = 2.5

    boundaries = np.arange(vmin, vmax + step, step)
    if boundaries[-1] < vmax:
        boundaries = np.append(boundaries, vmax)

    light = (246/255, 232/255, 195/255) # Light Beige
    deep = (103/255, 0/255, 31/255) #Dark Purple
    over = (64/255, 0/255, 20/255)  # Darker Purple

    n_intervals = len(boundaries) - 1
    base_cmap = LinearSegmentedColormap.from_list("rx5ratio_base", [light, deep])
    sample_pos = np.linspace(0.08, 1.0, n_intervals)
    interval_colors = [base_cmap(p) for p in sample_pos]

    cmap = ListedColormap(interval_colors + [interval_colors[-1]])
    cmap.set_over(over)
    norm = BoundaryNorm(boundaries, cmap.N, extend="max")

    # ---------------------------------------------------------
    # figure layout
    # ---------------------------------------------------------
    fig = plt.figure(figsize=figsize)

    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, 0.82],
        hspace=0.44,
    )
    
    top = outer[0].subgridspec(1, 1)
    
    # spacer | scatter | scatter | spacer
    bottom = outer[1].subgridspec(
        1, 4,
        width_ratios=[0.42, 1.0, 1.0, 0.42],
        wspace=0,
    )
    
    proj = ccrs.PlateCarree()
    ax_map = fig.add_subplot(top[0, 0], projection=proj)
    ax1 = fig.add_subplot(bottom[0, 1])
    ax2 = fig.add_subplot(bottom[0, 2], sharey=ax1)

    # ---------------------------------------------------------
    # top map
    # ---------------------------------------------------------
    im = da_ratio.plot.pcolormesh(
        ax=ax_map,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
        rasterized=True,
    )

    _add_map_style(ax_map, left_labels=True, bottom_labels=True)
    _add_panel_label(ax_map, "(a)")

    pos = ax_map.get_position()
    cax = fig.add_axes([pos.x0 + 0.12, pos.y0 - 0.06, pos.width - 0.24, 0.022])

    cb = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        boundaries=boundaries,
        ticks=boundaries[::2],
        spacing="uniform",
        extend="max",
    )
    cb.set_label(
        r"RX5day / $P_{\mathrm{annual}}$ [%]",
        fontsize=16,
        fontweight="bold"
    )
    cb.ax.tick_params(labelsize=12)

    # ---------------------------------------------------------
    # scatter panels
    # ---------------------------------------------------------
    _scatter_panel(
        ax1,
        x_pr, y_ratio_pr,
        xlabel=r"$P$ [mm day$^{-1}$]",
        show_ylabel=True,
        is_bgws=False,
    )
    _add_panel_label(ax1, "(b)")

    _scatter_panel(
        ax2,
        x_bgws, y_ratio_bgws,
        xlabel=r"BGWS [%]",
        show_ylabel=False,
        is_bgws=True,
    )
    _add_panel_label(ax2, "(c)")

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_bgws_ensemble_std_two_panel(
    historical_dict,
    change_dict,
    hist_key="12 model ensemble std",
    change_key="12 model ensemble std",
    var_name="bgws_tran_mean",
    lon_ticks=(-120, -60, 0, 60, 120),
    lat_ticks=(-40, 0, 40, 80),
    figsize=(14, 5.8),
    savepath=None,
    dpi=300,
):
    """
    Two-panel figure of ensemble std for BGWS:
      (a) historical
      (b) future-minus-historical change

    Parameters
    ----------
    historical_dict : dict
        Dict for historical period, containing ensemble std dataset at hist_key.
    change_dict : dict
        Dict for change period, containing ensemble std dataset at change_key.
    """

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------
    def _get_da(obj, var_name):
        if isinstance(obj, xr.DataArray):
            return obj
        if isinstance(obj, xr.Dataset):
            if var_name not in obj:
                raise KeyError(f"Variable '{var_name}' not found in dataset.")
            return obj[var_name]
        raise TypeError(f"Unsupported object type: {type(obj)}")

    def _infer_lat_lon_names(da):
        lat_candidates = ["lat", "latitude", "y"]
        lon_candidates = ["lon", "longitude", "x"]

        lat_name = next((k for k in lat_candidates if k in da.coords), None)
        lon_name = next((k for k in lon_candidates if k in da.coords), None)

        if lat_name is None or lon_name is None:
            raise ValueError("Could not infer lat/lon coordinate names.")
        return lat_name, lon_name

    def _add_map_style(ax, left_labels=True, bottom_labels=True):
        ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="0.4")

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="0.5",
            alpha=0.6,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)
        gl.xformatter = LongitudeFormatter(zero_direction_label=False)
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {"size": 13}
        gl.ylabel_style = {"size": 13}

    def _add_panel_label(ax, label, fontsize=18):
        ax.text(
            0.015, 1.15, label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2.0),
            zorder=50,
        )

    # ---------------------------------------------------------
    # load data
    # ---------------------------------------------------------
    da_hist = _get_da(historical_dict[hist_key], var_name)
    da_change = _get_da(change_dict[change_key], var_name)

    da_hist, da_change = xr.align(da_hist, da_change, join="inner")
    _infer_lat_lon_names(da_hist)

    # ---------------------------------------------------------
    # color ranges and colormaps
    # ---------------------------------------------------------
    from matplotlib.colors import BoundaryNorm

    bounds = np.arange(0, 35 + 5, 5)
    cmap = plt.get_cmap("magma_r", len(bounds))
    norm = BoundaryNorm(bounds, cmap.N, extend="max")

    # ---------------------------------------------------------
    # figure layout
    # ---------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, wspace=0.06)

    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ax2 = fig.add_subplot(gs[0, 1], projection=proj)

    # ---------------------------------------------------------
    # panel (a): historical std
    # ---------------------------------------------------------
    im1 = da_hist.plot.pcolormesh(
        ax=ax1,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
        rasterized=True,
    )
    _add_map_style(ax1, left_labels=True, bottom_labels=True)
    _add_panel_label(ax1, "(a)")
    ax1.set_title("Historical", fontsize=18, fontweight="bold", pad=8)

    pos1 = ax1.get_position()
    cax1 = fig.add_axes([pos1.x0 + 0.10, pos1.y0 - 0.1, pos1.width - 0.20, 0.03])    

    cb1 = fig.colorbar(
        im1,
        cax=cax1,
        orientation="horizontal",
        boundaries=bounds,
        ticks=bounds[::2],
        spacing="uniform",
        extend="max",
    )
    cb1.set_label(r"BGWS ensemble standard deviation [%]", fontsize=18, fontweight="bold")
    cb1.ax.tick_params(labelsize=14)

    # ---------------------------------------------------------
    # panel (b): change std
    # ---------------------------------------------------------
    im2 = da_change.plot.pcolormesh(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
        rasterized=True,
    )
    _add_map_style(ax2, left_labels=False, bottom_labels=True)
    _add_panel_label(ax2, "(b)")
    ax2.set_title(r"$\Delta$SSP3-7.0", fontsize=20, fontweight="bold", pad=8)

    pos2 = ax2.get_position()
    cax2 = fig.add_axes([pos2.x0 + 0.10, pos2.y0 - 0.1, pos2.width - 0.20, 0.03])

    cb2 = fig.colorbar(
        im2,
        cax=cax2,
        orientation="horizontal",
        boundaries=bounds,
        ticks=bounds[::2],
        spacing="uniform",
        extend="max",
    )
    cb2.set_label(r"$\Delta$BGWS ensemble standard deviation [ppts]", fontsize=18, fontweight="bold")
    cb2.ax.tick_params(labelsize=14)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig

import copy
import numpy as np
import xarray as xr


def build_relative_change_dict(
    masked_ds_dict,
    historical_key="historical",
    future_key="ssp370_ff",
    ens_key="12 model ensemble mean",
    variables=("pr_mean", "mrro_mean", "tran_mean"),
    out_key="ssp370_ff-historical_rel",
    eps=1e-12,
):
    """
    Build a dict containing relative change (%) for selected variables only,
    for each individual model and a recomputed ensemble mean.

    Relative change is:
        (future - historical) / historical * 100

    Returns
    -------
    rel_change_dict : dict
        {
            out_key: {
                model_name: xr.Dataset,
                ...,
                ens_key: xr.Dataset
            }
        }
    """
    bad_terms = ["ensemble mean", "ensemble median", "ensemble std"]

    hist_period = masked_ds_dict[historical_key]
    fut_period = masked_ds_dict[future_key]

    model_names = [
        k for k in fut_period.keys()
        if not any(term in k.lower() for term in bad_terms)
    ]

    rel_period = {}
    rel_member_datasets = []

    for model in model_names:
        if model not in hist_period:
            continue

        ds_hist = hist_period[model].drop_vars("member_id", errors="ignore")
        ds_fut = fut_period[model].drop_vars("member_id", errors="ignore")

        missing = [v for v in variables if (v not in ds_hist or v not in ds_fut)]
        if missing:
            raise KeyError(f"{model} is missing variables: {missing}")

        ds_rel = xr.Dataset()

        for var in variables:
            hist_da, fut_da = xr.align(ds_hist[var], ds_fut[var], join="inner")

            # avoid division by ~0
            denom = xr.where(np.abs(hist_da) > eps, hist_da, np.nan)
            rel_da = ((fut_da - hist_da) / denom) * 100.0
            rel_da.name = var
            rel_da.attrs = hist_da.attrs.copy()
            rel_da.attrs["long_name"] = f"Relative change in {var}"
            rel_da.attrs["units"] = "%"

            ds_rel[var] = rel_da

        rel_period[model] = ds_rel
        rel_member_datasets.append(ds_rel)

    if len(rel_member_datasets) == 0:
        raise ValueError("No model relative-change datasets were created.")

    # recompute ensemble mean from individual relative-change model datasets
    ens_rel = xr.concat(rel_member_datasets, dim="model").mean("model", skipna=True)
    rel_period[ens_key] = ens_rel

    return {out_key: rel_period}

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_bgws_regime_shift_figure(
    ds_dict,
    ds_dict_change,
    historical_period="historical",
    future_period="ssp370_ff",
    change_period="ssp370_ff-historical",
    hist_key="12 model ensemble mean",
    future_key="12 model ensemble mean",
    change_key="12 model ensemble mean",
    var_name="bgws_tran_mean",
    min_agree_models=8,
    stipple_step=5,
    stipple_size=8,
    lon_ticks=(-120, -60, 0, 60, 120),
    lat_ticks=(-40, 0, 40, 80),
    figsize=(14, 7.5),
    savepath=None,
    dpi=300,
):
    """
    Composite figure for BGWS regime shifts:
      - main map: future BGWS only where the BGWS sign differs from historical
      - inset bar chart: land-area fraction for blue->green and green->blue shifts,
        with 95% CI across ensemble members
      - stippling: low ensemble sign agreement in ΔBGWS (< min_agree_models / n_models)
    """

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------
    def _get_da(obj, var_name):
        if isinstance(obj, xr.DataArray):
            return obj
        if isinstance(obj, xr.Dataset):
            if var_name not in obj:
                raise KeyError(f"Variable '{var_name}' not found.")
            return obj[var_name]
        raise TypeError(f"Unsupported object type: {type(obj)}")

    def _infer_lat_lon_names(da):
        lat_candidates = ["lat", "latitude", "y"]
        lon_candidates = ["lon", "longitude", "x"]
        lat_name = next((k for k in lat_candidates if k in da.coords), None)
        lon_name = next((k for k in lon_candidates if k in da.coords), None)
        if lat_name is None or lon_name is None:
            raise ValueError("Could not infer lat/lon coordinate names.")
        return lat_name, lon_name

    def _add_map_style(ax, left_labels=True, bottom_labels=True):
        ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="0.4")

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="0.5",
            alpha=0.6,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)
        gl.xformatter = LongitudeFormatter(zero_direction_label=False)
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {"size": 13}
        gl.ylabel_style = {"size": 13}

    def _add_panel_label(ax, label, fontsize=18):
        ax.text(
            0.015, 0.985, label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2.0),
            zorder=50,
        )

    def _get_member_model_names(ds_period):
        bad_terms = ["ensemble mean", "ensemble median", "ensemble std"]
        return [k for k in ds_period.keys() if not any(term in k.lower() for term in bad_terms)]

    def _get_low_sign_agreement_mask(ds_dict_change, period, variable, min_agree_models=8):
        ds_period = ds_dict_change[period]
        model_names = _get_member_model_names(ds_period)

        da_models = xr.concat(
            [
                ds_period[k].drop_vars("member_id", errors="ignore")[variable]
                for k in model_names
            ],
            dim="model"
        )

        da_mean = ds_period[change_key].drop_vars("member_id", errors="ignore")[variable]

        pos_count = (da_models > 0).sum("model")
        neg_count = (da_models < 0).sum("model")
        max_same_sign = xr.where(pos_count >= neg_count, pos_count, neg_count)

        valid = np.isfinite(da_mean)
        low_agree = (max_same_sign < min_agree_models).where(valid)

        return low_agree.fillna(False), da_models.sizes["model"]

    def _plot_boolean_scatter_mask(
        ax,
        mask,
        stipple_step=4,
        scatter_size=7,
        zorder=35,
    ):
        if "lat" in mask.dims and "lon" in mask.dims:
            mask_sub = mask.isel(
                lat=slice(0, None, stipple_step),
                lon=slice(0, None, stipple_step),
            )
        else:
            dims = list(mask.dims)
            d1, d2 = dims[-2], dims[-1]
            mask_sub = mask.isel({d1: slice(0, None, stipple_step), d2: slice(0, None, stipple_step)})

        pts = mask_sub.where(mask_sub).stack(points=mask_sub.dims).dropna("points")
        if pts.sizes.get("points", 0) == 0:
            return

        x = pts["lon"].values
        y = pts["lat"].values

        ax.scatter(
            x, y,
            s=scatter_size * 2.0,
            marker="o",
            facecolors="white",
            edgecolors="white",
            linewidths=0.0,
            alpha=1.0,
            transform=ccrs.PlateCarree(),
            zorder=zorder,
            clip_on=True,
        )
        ax.scatter(
            x, y,
            s=scatter_size,
            marker="o",
            facecolors="black",
            edgecolors="white",
            linewidths=0.4,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            zorder=zorder + 0.1,
            clip_on=True,
        )

    def _shift_fraction_for_pair(hist_da, fut_da):
        lat_name, _ = _infer_lat_lon_names(hist_da)
        weights = np.cos(np.deg2rad(hist_da[lat_name]))
        weights_2d = xr.broadcast(weights, hist_da)[0]

        valid = np.isfinite(hist_da) & np.isfinite(fut_da)
        hist_pos = hist_da > 0
        hist_neg = hist_da < 0
        fut_pos = fut_da > 0
        fut_neg = fut_da < 0

        pos_to_neg = valid & hist_pos & fut_neg
        neg_to_pos = valid & hist_neg & fut_pos

        total = weights_2d.where(valid).sum(skipna=True).values.item()
        if total == 0:
            return np.nan, np.nan

        p2n = weights_2d.where(pos_to_neg).sum(skipna=True).values.item() / total * 100.0
        n2p = weights_2d.where(neg_to_pos).sum(skipna=True).values.item() / total * 100.0
        return p2n, n2p

    def _ensemble_shift_stats(ds_dict, historical_period, future_period, var_name):
        hist_period = ds_dict[historical_period]
        fut_period = ds_dict[future_period]
        model_names = [
            k for k in _get_member_model_names(fut_period)
            if k in hist_period
        ]

        p2n_vals = []
        n2p_vals = []

        for model in model_names:
            da_hist_m = _get_da(
                hist_period[model].drop_vars("member_id", errors="ignore"),
                var_name
            )
            da_fut_m = _get_da(
                fut_period[model].drop_vars("member_id", errors="ignore"),
                var_name
            )
            da_hist_m, da_fut_m = xr.align(da_hist_m, da_fut_m, join="inner")
            p2n, n2p = _shift_fraction_for_pair(da_hist_m, da_fut_m)
            p2n_vals.append(p2n)
            n2p_vals.append(n2p)

        return np.array(p2n_vals, dtype=float), np.array(n2p_vals, dtype=float)

    def _ci95(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        if n <= 1:
            return np.nan
        se = np.std(vals, ddof=1) / np.sqrt(n)
        return 1.96 * se

    # ---------------------------------------------------------
    # load data
    # ---------------------------------------------------------
    da_hist = _get_da(ds_dict[historical_period][hist_key].drop_vars("member_id", errors="ignore"), var_name)
    da_fut = _get_da(ds_dict[future_period][future_key].drop_vars("member_id", errors="ignore"), var_name)
    da_change = _get_da(ds_dict_change[change_period][change_key].drop_vars("member_id", errors="ignore"), var_name)

    da_hist, da_fut, da_change = xr.align(da_hist, da_fut, da_change, join="inner")
    _infer_lat_lon_names(da_hist)

    sign_change_mask = np.sign(da_hist) != np.sign(da_fut)
    da_fut_masked = da_fut.where(sign_change_mask)

    low_agree_mask, n_models = _get_low_sign_agreement_mask(
        ds_dict_change, change_period, var_name, min_agree_models=min_agree_models
    )
    low_agree_mask = low_agree_mask.where(sign_change_mask, False)

    p2n_mean, n2p_mean = _shift_fraction_for_pair(da_hist, da_fut)
    p2n_members, n2p_members = _ensemble_shift_stats(
        ds_dict, historical_period, future_period, var_name
    )

    p2n_ci = _ci95(p2n_members)
    n2p_ci = _ci95(n2p_members)

    # ---------------------------------------------------------
    # colormap for future BGWS state
    # ---------------------------------------------------------
    vmin = -6
    vmax = 6
    step = 1

    cmap, norm = col_uti.create_colormap(
        var="bgws_ensmean",
        period="ssp370_ff-historical",
        vmin=vmin,
        vmax=vmax,
        steps=step,
    )

    bounds = np.arange(vmin, vmax + step, step)
    if bounds[-1] < vmax:
        bounds = np.append(bounds, vmax)

    # ---------------------------------------------------------
    # figure layout
    # ---------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())

    im = da_fut_masked.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
        rasterized=True,
    )

    _add_map_style(ax, left_labels=True, bottom_labels=True)
    #_add_panel_label(ax, "(a)")

    _plot_boolean_scatter_mask(
        ax,
        low_agree_mask,
        stipple_step=stipple_step,
        scatter_size=stipple_size,
    )

    # ---------------------------------------------------------
    # colorbar (narrower)
    # ---------------------------------------------------------
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + 0.24, pos.y0 - 0.08, pos.width - 0.48, 0.03])

    cb = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        boundaries=bounds,
        ticks=np.arange(vmin, vmax + 1, 2),
        spacing="uniform",
        extend="both",
    )
    cb.set_label(r"SSP3-7.0 (2071-2100) BGWS [%]", fontsize=18, fontweight="bold")
    cb.ax.tick_params(labelsize=14)

    # ---------------------------------------------------------
    # low agreement legend
    # ---------------------------------------------------------
    add_low_agreement_legend(
        fig=fig,
        anchor_ax=cax,
        min_agree_models=min_agree_models,
        n_models=n_models,
        fontsize=15,
        dx=0,
        dy=0.08,
    )

    # ---------------------------------------------------------
    # inset bar chart with 95% CI
    # ---------------------------------------------------------
    inset = ax.inset_axes([0.07, 0.12, 0.15, 0.25])

    labels = ["Blue\n→ Green", "Green\n→ Blue"]
    vals = [p2n_mean, n2p_mean]
    errs = [p2n_ci, n2p_ci]
    colors = [
        (30/255, 130/255, 30/255),
        (55/255, 140/255, 225/255),
    ]

    x = np.arange(2)
    inset.bar(
        x,
        vals,
        color=colors,
        edgecolor="0.25",
        linewidth=0.6,
        alpha=0.9,
        width=0.7,
        zorder=2,
    )

    inset.errorbar(
        x,
        vals,
        yerr=errs,
        fmt="none",
        ecolor="black",
        elinewidth=2,
        capsize=4,
        capthick=1.2,
        zorder=4,
    )

    inset.set_ylabel("Land area [%]", fontsize=15)
    inset.set_xticks(x)
    inset.set_xticklabels(labels, fontsize=15)
    inset.tick_params(axis="y", labelsize=13)

    inset.set_yticks([2.5, 5.0, 7.5, 10.0])
    inset.set_yticklabels(["", "5", "", "10"])

    inset.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, color="0.6")
    inset.set_axisbelow(True)
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.95)

    ymax = np.nanmax(np.array(vals) + np.nan_to_num(np.array(errs), nan=0.0))
    ymax = max(8.5, ymax * 1.25)
    inset.set_ylim(0, ymax)

    for spine in inset.spines.values():
        spine.set_edgecolor("0.5")
        spine.set_linewidth(0.8)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig