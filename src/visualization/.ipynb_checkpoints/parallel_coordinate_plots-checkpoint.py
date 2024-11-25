"""
src/visualization/parallel_coordinate_plots.py

This script provides functions to create parallel coordinate plots.

Functions:
- create_parallel_coordinate_plots: Main function for creating the parallel coordinate plots.
- plot_region: Plot a single region.
- plot_data: Plot data points on the axes.
- adjust_axes: Adjust the axes based on global min/max values.
- get_region_name: Fetch the name of the region for plot titles.
- set_plot_title: Set the title for the plot.
- extract_variables_info: Extract and prepare variables for plotting.
- determine_change_type: Determine whether the change type is relative or absolute.
- prepare_display_variables: Prepare display variables with proper formatting.
- round_value_based_on_position: Round a value based on its magnitude.
- adjust_array_values: Adjust array values for plotting.

Usage:
    Import this module in your scripts.
"""

import colormaps as colmap
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import matplotlib.patches as patches
import numpy as np
import xarray as xr
import math
import os


# Define the BGWS colormap
def create_bgws_colormap(vmax, steps):
    #deep_blue = (60/255, 145/255, 230/255)
    deep_blue = (30/255, 30/255, 180/255)  # RGB: (0, 0, 150)
    light_blue = (160/255, 220/255, 255/255)  # RGB: (160, 220, 255)

    #light_blue = (240/255, 250/255, 255/255)
    #deep_green = (34/255, 139/255, 34/255)
    deep_green = (0/255, 100/255, 0/255)  # RGB: (0, 100, 0)

    light_green = (160/255, 240/255, 160/255)  # Brighter and saturated green

    #light_green = (240/255, 255/255, 240/255)
    
    boundaries = np.arange(-30, 30+steps, 5)
    cmap_name = 'BGWS colormap'

    cmap, norm = colmap.create_two_gradient_colormap(deep_green, light_green, deep_blue, light_blue, boundaries, cmap_name)
    
    # Dynamically darken the under and over colors
    under_color = adjust_color_darker(deep_green)  # Darken the lightest green for under
    over_color = adjust_color_darker(deep_blue)     # Darken the deepest blue for over

    # Set the under and over colors
    cmap.set_under(under_color)  # Darker color for values below the colormap range
    cmap.set_over(over_color)    # Darker color for values above the colormap range

    return cmap, norm
    
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

def create_parallel_coordinate_plots(
    ddict_change_regions_mean,
    ddict_historical_regions_mean,
    selected_indices='ALL',
    selected_vars=None,
    common_scale_for_mm_day=True,
    legend=True,
    save_fig=False,
    subdiv=False,
    selected_subdiv_idx=None,  # New parameter for selecting specific subdivision
    yaxis_limits=None  # New parameter for y-axis limits
):

    """
    Main function for creating the parallel coordinate plots.

    Parameters:
    - ddict_change_regions_mean: Dictionary with mean regional variables changes for all models.
    - ddict_historical_regions_mean: Dictionary with mean regional historical variables for all models.
    - selected_indices: Region index to select a specific region. Default is 'ALL' meaning all regions are plotted.
    - selected_vars: Select specific variables. Default is None.
    - common_scale_for_mm_day: Set y-axis to common scale for all variables in mm/day. Default is True.
    - legend: Defines if legend should be included or not. Default is True.
    - save_fig: Defines if figure should be saved. Default is False.
    - subdiv: Define if the dataset has subdivisions. Default is False.
    - selected_subdiv_idx: Select a specific subdivision index to plot. Default is None, meaning all subdivisions are plotted.
    - yaxis_limits: Define y-axis limits if min-max scale of variable is not wanted. Default is None.

    """
    models = list(ddict_change_regions_mean.keys())
    variables, display_variables, change_type, region_names = extract_variables_info(ddict_change_regions_mean, selected_vars)
    
    selected_indices = (
        ddict_change_regions_mean['Ensemble mean'].region.values.tolist()
        if selected_indices == "ALL" else selected_indices
    )
    
    for region_idx in selected_indices:
        if subdiv:
            # If selected_subdiv_idx is provided, plot only that specific subdivision
            if selected_subdiv_idx is not None:
                subdiv_indices = [selected_subdiv_idx]
            else:
                subdiv_indices = range(ddict_change_regions_mean['Ensemble mean'].dims['subdivision'])
                
            for subdiv_idx in subdiv_indices:
                fig, axes = plt.subplots(1, len(variables), sharey=False, figsize=(28, 12))

                plot_region(
                    fig, axes, models, region_idx, ddict_change_regions_mean,
                    ddict_historical_regions_mean, variables, display_variables,
                    change_type, common_scale_for_mm_day, subdiv_idx, yaxis_limits
                )

                for ax in axes:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=32)
            
                if legend:
                    region_name = get_region_name(ddict_change_regions_mean, region_idx, subdiv_idx=subdiv_idx)
                    add_legend_and_colorbar(fig, axes, models, region_name, ddict_historical_regions_mean, region_idx, subdiv_idx=subdiv_idx)
                
                if save_fig:
                    save_figure(fig, change_type, region_names, region_idx, subdiv_idx, legend=legend, common_scale_for_mm_day=common_scale_for_mm_day)

                plt.show()
                
        elif not subdiv:
            fig, axes = plt.subplots(1, len(variables), sharey=False, figsize=(26, 12))

            plot_region(
                fig, axes, models, region_idx, ddict_change_regions_mean,
                ddict_historical_regions_mean, variables, display_variables,
                change_type, common_scale_for_mm_day, subdiv_idx=None, yaxis_limits=yaxis_limits
            )

            for ax in axes:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=26)
                
            if legend:
                region_name = get_region_name(ddict_change_regions_mean, region_idx, subdiv_idx=None)
                add_legend_and_colorbar(fig, axes, models, region_name)

            if save_fig:
                save_figure(fig, change_type, region_names, region_idx, subdiv_idx=None, legend=legend, common_scale_for_mm_day=common_scale_for_mm_day)
                
            plt.show()


def plot_region(
    fig, axes, models, selected_region,
    ddict_change_regions_mean, ddict_historical_regions_mean,
    variables, display_variables, change_type,
    use_common_scale_for_mm_day, subdiv_idx=None, yaxis_limits=None
):
    """Plot a single region on the axes."""
    mm_day_variables = ['pr', 'mrro', 'evspsbl', 'tran', 'evapo']
    global_max_min_values, global_mm_min, global_mm_max = calculate_global_min_max(
        ddict_change_regions_mean, variables, selected_region, mm_day_variables, subdiv_idx
    )
    
    adjust_axes(
        axes, variables, global_max_min_values, global_mm_min,
        global_mm_max, display_variables, use_common_scale_for_mm_day, mm_day_variables, yaxis_limits=yaxis_limits
    )
    
    #norm = Normalize(vmin=-10, vmax=10)
    plot_data(
        fig, axes, models, variables, ddict_change_regions_mean,
        ddict_historical_regions_mean, selected_region, subdiv_idx
    )


def get_cmap(ddict_historical_regions_mean, models, selected_region, subdiv_idx=None):
    
    historical_values = []
    
    for model_idx, model_name in enumerate(models, start=1):    
        ds_historical = ddict_historical_regions_mean[model_name]
        
        if subdiv_idx is None:
            historical_value = ds_historical['bgws'].isel(region=selected_region).values.item()  # Use historical values for hue
        else:
            historical_value = ds_historical['bgws'].isel(region=selected_region, subdivision=subdiv_idx).values.item() 
            
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

    bgws_cmap, bgws_cmap_norm = create_bgws_colormap(vmax, steps)

    return bgws_cmap, bgws_cmap_norm, vmax, steps

def plot_data(
    fig, axes, models, variables,
    ddict_change_regions_mean, ddict_historical_regions_mean,
    selected_region, subdiv_idx=None
):
    """Plot data points for each model and variable."""
    for model_idx, model_name in enumerate(models, start=1):
        ds_change = ddict_change_regions_mean[model_name]
        ds_historical = ddict_historical_regions_mean[model_name]

        prev_xy = None
        
        for var_idx, variable in enumerate(variables):
            if variable in ds_change.data_vars:
                if subdiv_idx is None:
                    value = ds_change[variable].isel(region=selected_region).values.item()
                    historical_value = ds_historical['bgws'].isel(region=selected_region).values.item()  # Use historical values for hue
                    change_value = ds_change['bgws'].isel(region=selected_region).values.item()  # Use historical values for hue 
                else:
                    value = ds_change[variable].isel(region=selected_region, subdivision=subdiv_idx).values.item()
                    historical_value = ds_historical['bgws'].isel(region=selected_region, subdivision=subdiv_idx).values.item() 
                    change_value = ds_change['bgws'].isel(region=selected_region, subdivision=subdiv_idx).values.item() 

                
                if np.isnan(value):
                    prev_xy = None
                    continue

                bgws_cmap, bgws_cmap_norm, vmax, steps = get_cmap(ddict_historical_regions_mean, models, selected_region, subdiv_idx=subdiv_idx)

                color = bgws_cmap(bgws_cmap_norm(historical_value))
                current_xy = (var_idx, value)
                plot_individual_data_point(axes[var_idx], model_name, current_xy, color, model_idx)
                
                if prev_xy:
                    plot_connection(fig, axes, var_idx, prev_xy, current_xy, color, change_value)
                
                prev_xy = current_xy
            else:
                prev_xy = None

def plot_individual_data_point(
    ax, model_name, current_xy, color, model_idx
):
    """Plot a single data point."""
    if model_name.lower() == "ensemble mean":
        ax.plot(current_xy[0], current_xy[1], 'D', mec='red', mfc='none', markersize=22, mew=2, zorder=5)
    elif model_name.lower() == "ensemble median":
        ax.plot(current_xy[0], current_xy[1], 'o', mec='orange', mfc='none', markersize=28, mew=2, zorder=4)
    else:
        dot_color = '#f0f0f0' if current_xy[0] % 2 == 0 else 'white'
        ax.plot(current_xy[0], current_xy[1], 'o', color=dot_color, markersize=38, zorder=1)
        ax.annotate(str(model_idx), xy=current_xy, xytext=(0, 0), textcoords='offset points',
                    fontsize=38, weight='bold', color=color,
                    ha='center', va='center', zorder=2)


def plot_connection(
    fig, axes, var_idx, prev_xy,
    current_xy, color, value
):
    """Plot a connection between two data points."""
    linestyle = (0, (5, 5)) if value < 0 else '-'
    con = ConnectionPatch(
        xyA=prev_xy, xyB=current_xy, coordsA="data", coordsB="data",
        axesA=axes[var_idx-1], axesB=axes[var_idx],
        linestyle=linestyle, shrinkA=17, shrinkB=17, color=color, linewidth=2
    )
    fig.add_artist(con)


def adjust_axes(
    axes, variables, global_max_min_values, global_mm_min, global_mm_max,
    display_variables, use_common_scale_for_mm_day, mm_day_variables, yaxis_limits=None
):
    """Adjust axes properties and add variable names."""
    for j, var in enumerate(variables):
        if yaxis_limits and var in yaxis_limits:
            max_abs_value = yaxis_limits[var]
        elif var in mm_day_variables and use_common_scale_for_mm_day:
            max_abs_value = max(abs(global_mm_min), abs(global_mm_max)) * 1.05
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
        axes[j].set_xticklabels([display_variables[var]]) # Adjust font size of x-axis labels in main function
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



def get_region_name(ddict_change_regions_mean, selected_region, subdiv_idx=None):
    """Fetch region name for the plot title."""
    if subdiv_idx is not None:
        region_name = ddict_change_regions_mean[list(ddict_change_regions_mean.keys())[0]].coords['names'].isel(region=selected_region).item()
        subdivision_name = str(ddict_change_regions_mean[list(ddict_change_regions_mean.keys())[0]].coords['subdivision'][subdiv_idx].item())
        return region_name, subdivision_name
    else:
        return ddict_change_regions_mean[list(ddict_change_regions_mean.keys())[0]].coords['names'].isel(region=selected_region).item()


def set_plot_title(fig, region_name):
    """Set the overall title for the plot."""
    fig.suptitle(region_name, fontsize=32)


def calculate_global_min_max(ddict_change_regions_mean, variables, selected_region, mm_day_variables, subdiv_idx=None):
    """Calculate global max and min values for each variable across all subdivisions."""
    global_max_min_values = {}
    global_mm_max = -np.inf
    global_mm_min = np.inf

    for var in variables:
        global_max = -np.inf
        global_min = np.inf
        for model_name, ds in ddict_change_regions_mean.items():
            if var in ds.data_vars:
                if subdiv_idx is None:
                    selected_data = ds[var].isel(region=selected_region)
                else:
                    selected_data = ds[var].isel(region=selected_region, subdivision=subdiv_idx)

                if selected_data.size == 1:
                    value = selected_data.values.item()
                    if not np.isnan(value):
                        if var in mm_day_variables:
                            global_mm_max = max(global_mm_max, np.abs(value))
                            global_mm_min = min(global_mm_min, -np.abs(value))
                        global_max = max(global_max, value)
                        global_min = min(global_min, value)
                else:
                    print(f"Skipping region {selected_region} for model {model_name} and variable {var} due to multiple values or NaNs")
                    continue
        global_max_min_values[var] = (global_min, global_max)

    return global_max_min_values, global_mm_min, global_mm_max


def extract_variables_info(ds_dict, selected_vars):
    ensemble = ds_dict['Ensemble mean']
    variables = [var for var in ensemble.data_vars.keys() if var not in ['bgws', 'region', 'abbrevs', 'names', 'member_id']] if selected_vars is None else [var for var in selected_vars if var in ensemble.data_vars.keys()]
    display_variables = prepare_display_variables(variables)
    change_type = determine_change_type(ds_dict)
    region_names = ds_dict[list(ds_dict.keys())[0]].names
    return variables, display_variables, change_type, region_names


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


def prepare_display_variables(variables):
    var_map = {
        'tas': ('T', '°C'),
        'vpd': ('VPD', 'hPa'),
        'gpp': ('GPP', r'\frac{\frac{gC}{m^2}}{day}'),  
        'pr': ('P', r'\frac{mm}{day}'),
        'mrro': ('R', r'\frac{mm}{day}'),
        'evspsbl': ('ET', r'\frac{mm}{day}'),
        'tran': ('Tran', r'\frac{mm}{day}'),
        'evapo': ('E', r'\frac{mm}{day}'),
        'lai': ('LAI', r'\frac{m^2}{m^2}'),
        'mrso': ('SM', '\%'),
        'rgtr': ('P/T', r'\frac{GPP}{T}'),
        'et_partitioning': ('EP', r'\frac{E-Tran}{ET}'),
        'RX5day': ('RX5day', 'mm'),
        'gsl': ('GSL', 'months'),
        'wue': ('WUE', r'\frac{\frac{mm}{gC}}{m^2}'),
        'bgws': ('BGWS', '\%'),
        'tbgw': ('TBGW', '\%')
    }
    display_variables = {}
    for var in variables:
        if var in var_map:
            abbreviation, units = var_map[var]
            display_variables[var] = f"${{\Delta\, \mathrm{{\it{{{abbreviation}}}}}}}$ \n $\\left[{units}\\right]$"
        else:
            print(f"Variable '{var}' not found in var_map.")
            display_variables[var] = var
    return display_variables


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


def add_legend_and_colorbar(fig, ax, models, region_name, ddict_historical_regions_mean, selected_region, subdiv_idx=None):
    # Set figure caption
    #set_plot_title(fig, region_name)
    #fig.text(0.6, 0.93, region_name, ha='center', va='top', fontsize=22, wrap=True)
    
    # Upper Legend (Ensemble mean, median, and line styles)
    upper_legend_position = [0.755, 0.385, 0.1, 0.125]  # Adjust position as needed
    upper_legend_ax = fig.add_axes(upper_legend_position, frame_on=False, zorder=2)
    upper_legend_elements = [
        plt.Line2D([0], [0], marker='D', markeredgecolor='red', markerfacecolor='none', label='Ensemble mean', markersize=10, linestyle='None', lw=2),
        plt.Line2D([0], [0], marker='o', mec='orange', mfc='none', label='Ensemble median', markersize=14, linestyle='None', mew=2),
        plt.Line2D([0], [0], color='black', label='$+\,\Delta$ BGWS', linestyle='-', linewidth=2),
        plt.Line2D([0], [0], color='black', label='$-\,\Delta$ BGWS', linestyle='--', linewidth=2)  
    ]
    upper_legend = upper_legend_ax.legend(handles=upper_legend_elements, fontsize=18, loc='center', ncol=2,
                                          columnspacing=1, handletextpad=0.5, borderaxespad=0.5)
    upper_legend_ax.axis('off')
    upper_legend.get_frame().set_facecolor('none')
    upper_legend.get_frame().set_edgecolor('none')
    
    # Define starting position and spacing for the model name annotations
    start_x = 0.725  # Right side of the figure; adjust as needed
    start_y = 0.405  # Starting height; adjust as needed
    spacing_y = 0.0315  # Vertical spacing between model names; adjust as needed
    num_columns = 2  # Number of columns for model names
    column_width = 0.095  # Horizontal space between columns

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
        fig.text(x_position, y_position, f"{idx + 1}: {model_name}", fontsize=18, transform=fig.transFigure, ha='left', va='top', zorder=2)
 
    # Add the colorbar below the lower legend
    colorbar_position = [0.726, 0.16, 0.18, 0.03]  # right, up, length, width ADATP A BIT WHEN TAKING ONLY HALF OF THE COLORBAR 
    cbar_ax = fig.add_axes(colorbar_position, zorder=2) 

    # Create a rectangle box behind text and colorbar
    rect = patches.Rectangle(
        (0.72, 0.1),  # x, y
        0.188,  # box_width
        0.38,  # box_height
        linewidth=1,
        edgecolor='gray',
        facecolor='white',
        transform=fig.transFigure,
        zorder=1  # Ensure the box is behind the text and colorbar
    )
    fig.patches.append(rect)

    bgws_cmap, bgws_cmap_norm, vmax, steps = get_cmap(ddict_historical_regions_mean, models, selected_region, subdiv_idx=subdiv_idx)

    # Check if values are only negative or only positive
    historical_values = [
    ddict_historical_regions_mean[model]['bgws'].isel(region=selected_region, subdivision=subdiv_idx).values.item()
    if subdiv_idx is not None else 
    ddict_historical_regions_mean[model]['bgws'].isel(region=selected_region).values.item()
    for model in models
    if not np.isnan(
        ddict_historical_regions_mean[model]['bgws'].isel(region=selected_region, subdivision=subdiv_idx).values.item()
        if subdiv_idx is not None else 
        ddict_historical_regions_mean[model]['bgws'].isel(region=selected_region).values.item()
    )]

    print(historical_values)

    if all(value >= 0 for value in historical_values):
        # Only positive values: Cut the colormap from 0 to vmax
        cmap_half = LinearSegmentedColormap.from_list('pos_half', bgws_cmap(np.linspace(0.5, 1, 128)))

        deep_blue = (0/255, 0/255, 150/255)  # RGB: (0, 0, 150)
        over_color = adjust_color_darker(deep_blue)
        cmap_half.set_over(over_color)  # Darker color for values below the colormap range

        boundaries = np.arange(0, 30 + steps, steps)
        # Create a BoundaryNorm for the adjusted color map
        norm = BoundaryNorm(boundaries, cmap_half.N)
        # Create the colorbar
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_half), cax=cbar_ax, orientation='horizontal', extend='max')
    
    elif all(value <= 0 for value in historical_values):
        # Only negative values: Use green shades from -vmax to 0
        # Define green shades by slicing the colormap's green part (0 to mid-point)
        cmap_half = LinearSegmentedColormap.from_list('neg_half', bgws_cmap(np.linspace(0, 0.49, 128)))  # Green half
        
        # Dynamically darken the under and over colors
        deep_green = (0/255, 100/255, 0/255)  # RGB: (0, 100, 0)
        under_color = adjust_color_darker(deep_green)
        cmap_half.set_under(under_color)  # Darker color for values below the colormap range

        boundaries = np.arange(-30, 0 + steps, 5)
        norm = BoundaryNorm(boundaries, cmap_half.N)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_half), cax=cbar_ax, orientation='horizontal', extend='min')
        
    else:
        # Mixed values: Use the full colormap
        cmap_half = bgws_cmap
        boundaries = np.arange(-vmax, vmax + steps, steps)
        # Create a BoundaryNorm for the adjusted color map
        norm = BoundaryNorm(boundaries, cmap_half.N)
        # Create the colorbar
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_half), cax=cbar_ax, orientation='horizontal', extend='both')
    

    cbar.set_label("Historical Blue-Green Water Share [%]", fontsize=20, weight='bold')

    # Adjust the ticks based on the boundaries and steps
    if all(value >= 0 for value in historical_values) or all(value < 0 for value in historical_values):
        # Skip every second tick (for positive or negative values only)
        cbar_ticks = boundaries[::2]
    else:
        # Full range with standard steps
        cbar_ticks = boundaries

    cbar_ticklabels = [f"{tick}" for tick in cbar_ticks]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels)
    
    cbar.ax.tick_params(labelsize=20)

def save_figure(fig, change, region_names, region_idx, subdiv_idx, legend, common_scale_for_mm_day):
    # Caption and figure saving
    region_name = (region_names.isel(region=region_idx).item()).replace("/", "_")

    savepath=f'/work/ch0636/g300115/phd_project/paper_1/results/CMIP6/ssp370-historical/parallel_coordinate_plots/{region_name}/'

    os.makedirs(savepath, exist_ok=True)

    if subdiv_idx is None:
        if legend:
            if common_scale_for_mm_day:
                filename = f'{change}.pdf'
            else:
                filename = f'{change}_no_common_y_scale.pdf' 
        else:
            if common_scale_for_mm_day:
                filename = f'{change}_without_legend.pdf' 
    
            else:
                filename = f'{change}_without_legend_no_common_y_scale.pdf' 
    else:
        if legend:
            if common_scale_for_mm_day:
                filename = f'{subdiv_idx}_{change}.pdf'
            else:
                #filename = f'{region_name}_{subdiv_idx}_{change}_no_common_y_scale.pdf' 
                filename = f'{subdiv_idx}_historical.pdf' 
        else:
            if common_scale_for_mm_day:
                filename = f'{subdiv_idx}_{change}_without_legend.pdf' 
    
            else:
                filename = f'{subdiv_idx}_{change}_without_legend_no_common_y_scale.pdf' 
    
    filepath = os.path.join(savepath, filename)
    fig.savefig(filepath, dpi=600, bbox_inches='tight', format='pdf')
    print(f'Figure saved at: {filepath}')
