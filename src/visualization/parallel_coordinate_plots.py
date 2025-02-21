"""
Parallel Coordinate Plots for Climate Model Analysis
-----------------------------------------------------
This script provides functions for generating parallel coordinate plots to visualize 
climate model predictions and changes in different water regimes.

Functions:
- create_parallel_coordinate_plots: Main function to generate and optionally save parallel coordinate plots.
- plot_region: Handles plotting of a specific region.
- get_cmap: Creates a colormap for visualization based on historical values.
- plot_data: Plots data points for each model and variable.
- plot_individual_data_point: Plots a single data point with a specific color scheme.
- plot_connection: Draws connecting lines between data points.
- adjust_axes: Configures the appearance and limits of plot axes.
- get_region_name: Extracts and formats region names for plot titles.
- calculate_global_min_max: Computes the global max/min values across models for normalization.
- extract_variables_info: Extracts selected variables and determines how they are displayed.
- determine_change_type: Determines whether to use absolute or relative change.
- round_value_based_on_position: Rounds values dynamically based on magnitude.
- adjust_array_values: Adjusts array values to appropriate axis limits.
- adjust_color_darker: Darkens a given color for better contrast.
- add_legend_and_colorbar: Adds a legend and colorbar to the plot for better readability.

Usage: Import this module and call `create_parallel_coordinate_plots` with the necessary data.

Author: [Simon P. Heselschwerdt]
Date: [2025-06-02]
Dependencies: matplotlib, numpy, xarray, os, sys
"""
# ========== Imports ==========

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import matplotlib.patches as patches
import numpy as np
import xarray as xr
import math
import os
import sys

# ========== Import Custom Functions ==========

import colormaps_and_utilities as col_uti
import save_data_as_nc as save_dat

# ========== Functions ==========

def create_parallel_coordinate_plots(ddict_change_mean, ddict_historical_mean, selected_vars, yaxis_limits, savepath=None):
    
    models = list(ddict_change_mean.keys())
    variables, display_variables, change_type = extract_variables_info(ddict_change_mean, selected_vars)
  
    for subdiv_idx in range(ddict_change_mean['Ensemble mean'].sizes['subdivision']):
        fig, axes = plt.subplots(1, len(variables), sharey=False, figsize=(28, 12))

        plot_region(fig, axes, models, ddict_change_mean, ddict_historical_mean, 
                    variables, display_variables, change_type, subdiv_idx, yaxis_limits)

        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=32)
    
        add_legend_and_colorbar(fig, axes, models, ddict_historical_mean, subdiv_idx)

        if savepath is not None:
            if subdiv_idx == 0:
                subdiv = 'bw_regime'
            else:
                 subdiv = 'gw_regime'
            filename = f'para_coord_{subdiv}.pdf'
            col_uti.save_fig(fig, savepath, filename, dpi=300)
            print(f"Figure saved under {savepath}{filename}")
        else:
            print("Figure not saved! If you want to save the figure change filepath='your filepath/' in the function call.")

        plt.show()


def plot_region(fig, axes, models, ddict_change_mean, ddict_historical_mean, 
                variables, display_variables, change_type, subdiv_idx, yaxis_limits):
    """Plot a single region on the axes."""
    global_max_min_values, global_mm_min, global_mm_max = calculate_global_min_max(ddict_change_mean, variables, subdiv_idx)
    
    adjust_axes(axes, variables, global_max_min_values, global_mm_min,
                global_mm_max, display_variables, yaxis_limits)
    
    plot_data(fig, axes, models, variables, ddict_change_mean, ddict_historical_mean, subdiv_idx)
    
def get_cmap(ddict_historical_mean, models, subdiv_idx):
    
    historical_values = []
    
    for model_idx, model_name in enumerate(models, start=1):    
        ds_historical = ddict_historical_mean[model_name]
        
        historical_value = ds_historical['bgws'].isel(subdivision=subdiv_idx).values.item()  # Use historical values for hue
        historical_values.append(abs(historical_value))  

    historical_max = max(historical_values)  

    # Define a list of thresholds for vmax (descending order for easier logic)
    thresholds = [80, 70, 60, 50, 40, 30, 20, 10]  # Add more thresholds as needed
    
    # Find the largest threshold that is less than or equal to historical_max
    vmax = next((threshold for threshold in thresholds if threshold <= historical_max), thresholds[-1])
            
    # Define steps based on vmax
    if vmax == 10:
        steps = 2
    elif vmax in [20, 40]:
        steps = 5
    elif vmax > 40:
        steps = 5
    else:
        steps = 5

    bgws_cmap, bgws_cmap_norm = col_uti.create_colormap('bgws_ensmean', 'historical', -vmax, vmax, steps)

    return bgws_cmap, bgws_cmap_norm, vmax, steps

def plot_data(fig, axes, models, variables, ddict_change_mean, ddict_historical_mean, subdiv_idx=None):
    """Plot data points for each model and variable."""
    for model_idx, model_name in enumerate(models, start=1):
        ds_change = ddict_change_mean[model_name]
        ds_historical = ddict_historical_mean[model_name]

        prev_xy = None
        
        for var_idx, variable in enumerate(variables):
            if variable in ds_change.data_vars:
                value = ds_change[variable].isel(subdivision=subdiv_idx).values.item()
                historical_value = ds_historical['bgws'].isel(subdivision=subdiv_idx).values.item() 
                change_value = ds_change['bgws'].isel(subdivision=subdiv_idx).values.item() 

                if np.isnan(value):
                    prev_xy = None
                    continue

                bgws_cmap, bgws_cmap_norm, vmax, steps = get_cmap(ddict_historical_mean, models, subdiv_idx)

                color = bgws_cmap(bgws_cmap_norm(historical_value))
                current_xy = (var_idx, value)
                plot_individual_data_point(axes[var_idx], model_name, current_xy, model_idx, change_value, historical_value)
                
                if prev_xy:
                    if model_name != 'Ensemble mean' and model_name != 'Ensemble median':
                        plot_connection(fig, axes, var_idx, prev_xy, current_xy, change_value, historical_value)
                
                prev_xy = current_xy
            else:
                prev_xy = None

def plot_individual_data_point(ax, model_name, current_xy, model_idx, value, historical_value):
    """Plot a single data point."""
    if historical_value < 0:
        color = (30/255, 130/255, 30/255) if value < 0 else (160/255, 113/255, 120/255)
    else:
        color = (180/255, 160/255, 120/255) if value < 0 else (55/255, 140/255, 225/255)
        
    if model_name.lower() == "ensemble mean":
        ax.plot(current_xy[0], current_xy[1], 'D', mec='red', mfc='none', markersize=24, mew=4, zorder=5)
    elif model_name.lower() == "ensemble median":
        ax.plot(current_xy[0], current_xy[1], 'o', mec='black', mfc='none', markersize=30, mew=4, zorder=4)
    else:
        dot_color = '#f0f0f0' if current_xy[0] % 2 == 0 else 'white'
        ax.plot(current_xy[0], current_xy[1], 'o', color=dot_color, markersize=38, zorder=1)
        ax.annotate(str(model_idx), xy=current_xy, xytext=(0, 0), textcoords='offset points',
                    fontsize=38, weight='bold', color=color,
                    ha='center', va='center', zorder=2)


def plot_connection(fig, axes, var_idx, prev_xy, current_xy, value, historical_value):
    """Plot a connection between two data points."""
    if historical_value < 0:
        color = (30/255, 130/255, 30/255) if value < 0 else (160/255, 113/255, 120/255)
    else:
        color = (180/255, 160/255, 120/255) if value < 0 else (55/255, 140/255, 225/255)
        
    con = ConnectionPatch(
        xyA=prev_xy, xyB=current_xy, coordsA="data", coordsB="data",
        axesA=axes[var_idx-1], axesB=axes[var_idx],
        linestyle='-', shrinkA=17, shrinkB=17, color=color, linewidth=2
    )
    fig.add_artist(con)

def adjust_axes(axes, variables, global_max_min_values, global_mm_min, global_mm_max,
                display_variables, yaxis_limits):
    """Adjust axes properties and add variable names."""
    for j, var in enumerate(variables):
        if yaxis_limits and var in yaxis_limits:
            max_abs_value = yaxis_limits[var]
        else:
            max_abs_value = max(abs(global_max_min_values[var][0]), abs(global_max_min_values[var][1])) * 1.05

        if np.isnan(max_abs_value):
            continue

        ticks = adjust_array_values(np.linspace(-max_abs_value, max_abs_value, num=5))
        
        axes[j].set_ylim(ticks[0], ticks[-1])
        axes[j].set_yticks(ticks)
        axes[j].spines['left'].set_bounds(ticks[0], ticks[-1])
        axes[j].tick_params(axis='y', labelsize=24)

        axes[j].set_xlim(j - 1, j + 1)
        axes[j].set_xticks([j])
        axes[j].spines['top'].set_visible(False)
        axes[j].spines['bottom'].set_visible(False)
        axes[j].spines['right'].set_visible(False)
        axes[j].set_xticklabels([display_variables[var]])  # Adjust font size of x-axis labels in main function
        axes[j].tick_params(axis='x', length=10, color='white')
        
        axes[j].axhline(y=0, color='gray', linestyle='-', linewidth=1)

        if j % 2 == 0:
            axes[j].set_facecolor('#f0f0f0')

        if j == 0:
            axes[j].set_ylabel("End-of-century response", fontsize=36)
        
        if j > 0:
            axes[j].set_yticklabels([f'{tick:.2f}'.rstrip('0').rstrip('.') if '.' in f'{tick:.2f}' and tick != 0 else '' for tick in ticks])
            ticks = axes[j].yaxis.get_major_ticks()
            tick_to_modify = 2
            ticks[tick_to_modify].tick1line.set_markersize(50)
            ticks[tick_to_modify].tick1line.set_markeredgecolor('gray')
            ticks[tick_to_modify].tick1line.set_markeredgewidth(1)
        else:
            axes[j].set_yticklabels([f'{tick:.2f}'.rstrip('0').rstrip('.') if '.' in f'{tick:.2f}' else f'{tick:.2f}' for tick in ticks])

def get_region_name(ddict_change_mean, subdiv_idx):
    """Fetch region name for the plot title."""
    subdivision_name = str(ddict_change_mean[list(ddict_change_mean.keys())[0]].coords['subdivision'][subdiv_idx].item())
    
    return subdivision_name

def calculate_global_min_max(ddict_change_mean, variables, subdiv_idx):
    """Calculate global max and min values for each variable across all subdivisions."""
    global_max_min_values = {}
    global_mm_max = -np.inf
    global_mm_min = np.inf

    for var in variables:
        global_max = -np.inf
        global_min = np.inf
        for model_name, ds in ddict_change_mean.items():
            if var in ds.data_vars:
                selected_data = ds[var].isel(subdivision=subdiv_idx)

                if selected_data.size == 1:
                    value = selected_data.values.item()
                    if not np.isnan(value):
                        global_max = max(global_max, value)
                        global_min = min(global_min, value)
                else:
                    print(f"Skipping regime {subdiv_idx} for model {model_name} and variable {var} due to multiple values or NaNs")
                    continue
        global_max_min_values[var] = (global_min, global_max)

    return global_max_min_values, global_mm_min, global_mm_max

def extract_variables_info(ds_dict, selected_vars):
    ensemble = ds_dict['Ensemble mean']
    variables = [var for var in ensemble.data_vars.keys() if var not in ['bgws', 'region', 'abbrevs', 'names', 'member_id']] if selected_vars is None else [var for var in selected_vars if var in ensemble.data_vars.keys()]
    display_variables = col_uti.get_var_name_parallel_coordinate_plot(variables)
    change_type = determine_change_type(ds_dict)
    return variables, display_variables, change_type
    
def determine_change_type(ds_dict):
    """
    Determines the type of change (relative or absolute) based on the units of the first 
    variable found in the provided dataset dictionary.
    """
    try:
        first_scenario = list(ds_dict.keys())[0]
        first_model = list(ds_dict[first_scenario].keys())[0]
        first_dataset = ds_dict[first_scenario][first_model]
        
        if isinstance(first_dataset, xr.Dataset):
            first_var_name = list(first_dataset.data_vars)[0]
            units = first_dataset[first_var_name].attrs.get('units', '')
        elif isinstance(first_dataset, xr.DataArray):
            units = first_dataset.attrs.get('units', '')
        else:
            raise ValueError("Unexpected data structure. Expected xarray Dataset or DataArray.")
        
        return 'rel_change' if units == '%' else 'abs_change'
    except Exception as e:
        print(f"Error determining change type: {e}")
        return 'abs_change'  # Default to 'abs_change' in case of error or unexpected data structure

def round_value_based_on_position(x):
    if x == 0:
        return 0
    abs_x = abs(x)
    if abs_x >= 10:
        return math.ceil(x)
    elif abs_x >= 1:
        return max(1.5, math.ceil(x * 2) / 2)
    else:
        digit_pos = -int(math.floor(math.log10(abs_x)))
        increment = 10 ** (-digit_pos)
        return math.ceil(x / increment) * increment

def adjust_array_values(arr):
    abs_max_val = np.max(np.abs(arr))
    new_max = round_value_based_on_position(abs_max_val)
    return np.array([-new_max, -new_max / 2, 0, new_max / 2, new_max])

def adjust_color_darker(color, adjustment=30):
    """
    Darken a given RGB color by subtracting a specified value from each component.

    Parameters:
    - color: tuple, the RGB color to adjust (values between 0 and 1).
    - adjustment: int, the amount to subtract from each RGB component (default is 10).

    Returns:
    - tuple: The darkened RGB color (values between 0 and 1).
    """
    return tuple(max(0, (c * 255 - adjustment) / 255) for c in color)

def add_legend_and_colorbar(fig, ax, models, ddict_historical_mean, subdiv_idx):
    # Upper Legend (Ensemble mean, median, and line styles)
    upper_legend_position = [0.736, 0.34, 0.1, 0.125]  # Adjust position as needed
    upper_legend_ax = fig.add_axes(upper_legend_position, frame_on=False, zorder=2)

    # Check if values are only negative or only positive
    historical_values = [
    ddict_historical_mean[model]['bgws'].isel(subdivision=subdiv_idx).values.item()
    for model in models
    if not np.isnan(
        ddict_historical_mean[model]['bgws'].isel(subdivision=subdiv_idx).values.item()
    )]
    
    if all(value >= 0 for value in historical_values):
        upper_legend_elements = [
            plt.Line2D([0], [0], marker='D', markeredgecolor='red', markerfacecolor='none', label='Ensemble mean', markersize=12, linestyle='None', lw=10),
            plt.Line2D([0], [0], marker='o', mec='black', mfc='none', label='Ensemble median', markersize=15, linestyle='None', mew=2),
            plt.Line2D([0], [0], color=(55/255, 140/255, 225/255), label='$+\,\Delta$ BGWS', linestyle='-', linewidth=4),
            plt.Line2D([0], [0], color=(180/255, 160/255, 120/255), label='$-\,\Delta$ BGWS', linestyle='-', linewidth=4)  
        ]
    
    elif all(value <= 0 for value in historical_values):
        upper_legend_elements = [
            plt.Line2D([0], [0], marker='D', markeredgecolor='red', markerfacecolor='none', label='Ensemble mean', markersize=12, linestyle='None', lw=10),
            plt.Line2D([0], [0], marker='o', mec='black', mfc='none', label='Ensemble median', markersize=15, linestyle='None', mew=2),
            plt.Line2D([0], [0], color=(160/255, 113/255, 120/255), label='$+\,\Delta$ BGWS', linestyle='-', linewidth=4),
            plt.Line2D([0], [0], color=(30/255, 130/255, 30/255), label='$-\,\Delta$ BGWS', linestyle='-', linewidth=4)  
        ]
    
    upper_legend = upper_legend_ax.legend(handles=upper_legend_elements, fontsize=24, loc='center', ncol=2,
                                          columnspacing=1, handletextpad=0.4, borderaxespad=0.5) # columnspacing=1, handletextpad=0.5, borderaxespad=0.5)
    upper_legend_ax.axis('off')
    upper_legend.get_frame().set_facecolor('none')
    upper_legend.get_frame().set_edgecolor('none')
    
    # Define starting position and spacing for the model name annotations
    start_x = 0.683 # Right side of the figure; adjust as needed
    start_y = 0.35 # Starting height; adjust as needed
    spacing_y = 0.038 # Vertical spacing between model names; adjust as needed
    num_columns = 2 # Number of columns for model names
    column_width = 0.115 # Horizontal space between columns

    # Filter out the ensemble mean and median for the model names annotation
    model_names = [model for model in models if model not in ["Ensemble mean", "Ensemble median"]]

    # Calculate how many names per column
    names_per_column = len(model_names) // num_columns + (len(model_names) % num_columns > 0)

    # Loop through the model names to place them as text annotations
    for idx, model_name in enumerate(model_names):
        column = idx // names_per_column
        row = idx % names_per_column
        x_position = start_x + column * column_width  # Adjust x based on column
        y_position = start_y - row * spacing_y  # Adjust y based on row

        # Place text annotation
        fig.text(x_position, y_position, f"{idx + 1}: {model_name}", fontsize=24, transform=fig.transFigure, ha='left', va='top', zorder=2)
 

    # Create a rectangle box behind text and colorbar
    rect = patches.Rectangle(
        (0.68, 0.128),  # x, y
        0.22,  # box_width
        0.314,  # box_height
        linewidth=1,
        edgecolor='gray',
        facecolor='white',
        transform=fig.transFigure,
        zorder=1  # Ensure the box is behind the text and colorbar
    )
    fig.patches.append(rect)