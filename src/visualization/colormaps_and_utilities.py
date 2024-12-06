"""
src/visualization/colormaps_and_utilities.py

This script provides utility functions for colormap creation, color mapping, and figure saving 
in support of global map visualizations.

Functions:
- `save_fig`: Saves a matplotlib figure to a specified path.
- `map_colors_to_display_names`: Maps base colors to predictor variable display names.
- `get_var_name_parallel_coordinate_plot`: Maps variable names to display names for parallel coordinate plots.
- `get_global_map_var_name`: Maps variable names to display names for global maps.
- `create_diverging_colormap`: Creates a diverging colormap (negative to positive with white center).
- `create_two_gradient_colormap_with_white_transition`: Creates a colormap with two gradients with a white transition at zero.
- `create_two_gradient_colormap`: Creates a colormap with two gradients for negative and positive values.
- `create_one_gradient_colormap`: Creates a single gradient colormap.
- `create_palette_from_colormap`: Generates colors for coefficients based on a colormap.
- `create_colormap`: Builds specific colormaps based on variable and period.
- `get_colmap_set_up`: Prepares colormap boundaries and bin counts.
- `create_bgws_change_colormaps`: Generates colormaps specifically for BGWS regime changes.

Usage:
    Import this module in your scripts and call the required functions.

Author: Simon P. Heselschwerdt
Date: 2024-12-02
Dependencies: numpy, matplotlib
"""

# ========== Imports ==========
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# ========== Utility Functions ==========

def save_fig(fig, savepath, filename, dpi):
    os.makedirs(savepath, exist_ok=True)
    filepath = os.path.join(savepath, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')

def map_colors_to_display_names(base_colors, predictor_vars):
    """
    Maps custom colors to the formatted display variable names.

    Parameters:
    - base_colors (dict): Dictionary with original variable names as keys and color names as values.
    - predictor_vars (list): List of predictor variable names.

    Returns:
    - dict: Dictionary with formatted display names as keys and colors as values.
    """
    display_names = get_var_name_parallel_coordinate_plot(predictor_vars)
    display_colors = {display_names[var]: base_colors[var] for var in predictor_vars if var in base_colors}
    return display_colors

def get_var_name_parallel_coordinate_plot(variables):
    var_map = {
        'bgws': ('BGWS', r'%'),
        'RX5day': ('RX5day', r'mm'),
        'pr': ('P', r'\frac{mm}{day}'),
        'mrro': ('R', r'\frac{mm}{day}'),
        'tran': ('Tran', r'\frac{mm}{day}'),
        'evapo': ('E', r'\frac{mm}{day}'),
        'mrso': ('SM', r'%'),
        'lai': ('LAI', r'\frac{m^2}{m^2}'),
        'wue': ('WUE', r'\frac{GPP}{Tran}'),
        'vpd': ('VPD', r'hPa')
    }
    display_variables = {}
    for var in variables:
        if var in var_map:
            abbreviation, units = var_map[var]
            display_variables[var] = f"${{\Delta\, \mathrm{{\it{{{abbreviation}}}}}}}$"
        else:
            print(f"Variable '{var}' not found in var_map.")
            display_variables[var] = var
    return display_variables

def get_global_map_var_name(period, variable):
    var_map = {
        'pr': ('Precipitation', r'mm day$^{-1}$'),
        'mrro': ('Runoff', r'mm day$^{-1}$'),
        'tran': ('Transpiration', r'mm day$^{-1}$'),
        'bgws': ('Blue-Green Water Share', r'%'),
        'RX5day': ('RX5day', r'mm'),
        'evapo': ('Evaporation', r'mm day$^{-1}$'),
        'evspsbl': ('Evapotranspiration', r'mm day$^{-1}$'),
        'vpd': ('Vapour Pressure Deficit', r'hPa'),
        'mrso': ('Soil Moisture', r'%'), # Only change 
        'lai': ('Leaf Area Index', r'm$^{2}$ m$^{-2}$'),
        'gpp': ('Gross Primary Productivity', r'gC m$^{-2}$ day$^{-1}$'),  
        'wue': ('Water Use Efficiency', r'gC m$^{-2}$ mm$^{-1}$')
    }
    if variable in var_map:
        long_name, unit = var_map[variable]
        if period == 'historical' or period == 'ssp370':
            display_variable = f"{long_name} [{unit}]"
        else: 
            display_variable = f"$\Delta$ {long_name} [{unit}]"
    else:
        print(f"Variable '{variable}' not found in var_map.")
        display_variable = variable 
    return display_variable

# ========== Colormap Creation Functions ==========

def create_diverging_colormap(deep_blue, deep_green):
    """
    Create a diverging colormap from deep_blue to white to deep_green.

    Parameters:
    - deep_blue: tuple, the color for positive values.
    - deep_green: tuple, the color for negative values.

    Returns:
    - colormap: LinearSegmentedColormap, the created colormap.
    """
    # Define the colors at three points: deep_green -> white -> deep_blue
    colors = [deep_green, (1, 1, 1), deep_blue]
    
    # Create a linear segmented colormap
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors)
    
    return cmap

def create_two_gradient_colormap_with_white_transition(dark_color_m, light_color_m, dark_color_p, light_color_p, boundaries, n=6):
    """
    Create a colormap with two gradients combined, one for positive and one for negative values,
    with white at zero.

    Parameters:
    - dark_color_m: tuple, the darker color for the negative gradient.
    - light_color_m: tuple, the lighter color for the negative gradient.
    - dark_color_p: tuple, the darker color for the positive gradient.
    - light_color_p: tuple, the lighter color for the positive gradient.
    - boundaries: array-like, the boundaries for the bins.
    - n: int, number of colors to sample from each gradient.

    Returns:
    - colormap: LinearSegmentedColormap, the created colormap.
    - norm: BoundaryNorm, the corresponding BoundaryNorm for the colormap.
    """
    total_bins = len(boundaries) - 1
    n = max(n, total_bins)  # Ensure n is at least the number of bins

    # Create color gradients
    positive_map = LinearSegmentedColormap.from_list("+ map", [light_color_p, dark_color_p], N=n)
    negative_map = LinearSegmentedColormap.from_list("- map", [dark_color_m, light_color_m], N=n)

    # Sample gradients
    p_colors = [positive_map(i) for i in np.linspace(0, 1, n)]
    n_colors = [negative_map(i) for i in np.linspace(0, 1, n)]

    # Insert white color at the midpoint of the combined gradient
    combined_grad = n_colors + [(1, 1, 1)] + p_colors

    colormap = LinearSegmentedColormap.from_list('cmap', combined_grad, N=len(combined_grad))
    norm = BoundaryNorm(boundaries, ncolors=len(combined_grad), clip=True)

    return colormap, norm

def create_two_gradient_colormap(dark_color_m, light_color_m, dark_color_p, light_color_p, boundaries, n=6, under_color=None, over_color=None):
    """
    Create a colormap with two gradients combined, one for positive and one for negative values.

    Parameters:
    - dark_color_m: tuple, the darker color for the negative gradient.
    - light_color_m: tuple, the lighter color for the negative gradient.
    - dark_color_p: tuple, the darker color for the positive gradient.
    - light_color_p: tuple, the lighter color for the positive gradient.
    - boundaries: array-like, the boundaries for the bins.
    - n: int, number of colors to sample from each gradient.

    Returns:
    - colormap: LinearSegmentedColormap, the created colormap.
    - norm: BoundaryNorm, the corresponding BoundaryNorm for the colormap.
    """
    total_bins = len(boundaries) - 1
    n = max(n, total_bins)  # Ensure n is at least the number of bins

    positive_map = LinearSegmentedColormap.from_list("+ map", [light_color_p, dark_color_p], N=n)
    negative_map = LinearSegmentedColormap.from_list("- map", [dark_color_m, light_color_m], N=n)

    # Sample gradients 
    p_colors = [positive_map(i) for i in np.linspace(0, 1, n)]
    n_colors = [negative_map(i) for i in np.linspace(0, 1, n)]

    # Combine both gradients
    combined_grad = n_colors + p_colors
    
    colormap = LinearSegmentedColormap.from_list('cmap', combined_grad, N=len(combined_grad))
    norm = BoundaryNorm(boundaries, ncolors=len(combined_grad), extend='both')

    # Set explicit under and over colors
    if under_color:
        colormap.set_under(under_color)
    if over_color:
        colormap.set_over(over_color)

    return colormap, norm


def create_one_gradient_colormap(light_color, dark_color, boundaries, n=6, over_color=None):
    """
    Create a colormap with two gradients combined, one for positive and one for negative values.

    Parameters:
    - light_color: tuple, the lighter color.
    - dark_color: tuple, the darker color.
    - boundaries: array-like, the boundaries for the bins.
    - n: int, number of colors to sample.

    Returns:
    - colormap: LinearSegmentedColormap, the created colormap.
    - norm: BoundaryNorm, the corresponding BoundaryNorm for the colormap.
    """
    total_bins = len(boundaries) - 1
    n = max(n, total_bins)  # Ensure n is at least the number of bins

    colormap = LinearSegmentedColormap.from_list("cmap", [light_color, dark_color], N=n)

    # Sample gradients 
    colors = [colormap(i) for i in np.linspace(0, 1, n)]

    if over_color:
        colormap.set_over(over_color)
    
    norm = BoundaryNorm(boundaries, ncolors=len(colors), clip=True)

    return colormap, norm

def create_palette_from_colormap(bgws_cmap, bgws_cmap_norm, coefficients):
    # Generate colors for each coefficient
    return [bgws_cmap(bgws_cmap_norm(coeff)) for coeff in coefficients]

def create_colormap(var, period, vmin, vmax, steps):
    if period == 'historical' or period == 'ssp370':
        if var == 'bgws':
            under = (15/255, 115/255, 15/255)  # deeper green
            deep_negative = (30/255, 130/255, 30/255) # green
            light_negative = (240/255, 255/255, 240/255) # light green
            
            over = (40/255, 125/255, 210/255)  # deeper blue
            deep_positive = (55/255, 140/255, 225/255) # blue
            light_positive = (240/255, 250/255, 255/255) # light blue

            boundaries, n = get_colmap_set_up(vmin, vmax, steps)

            cmap, cmap_norm = create_two_gradient_colormap(deep_negative, light_negative, 
                                                           deep_positive, light_positive, 
                                                           boundaries, n, under, over)

        elif var == 'pr' or var == 'RX5day' or var == 'mrro':
            # Color stops for the gradient: Light Beige -> Dark Blue
            light_positive = (246/255, 232/255, 195/255) # Light Beige
            deep_positive = (0/255, 62/255, 125/255) # Dark Blue
            over = (0/255, 52/255, 105/255)  # Darker Blue

            boundaries, n = get_colmap_set_up(vmin, vmax, steps)
            cmap, cmap_norm = create_one_gradient_colormap(light_positive, deep_positive, boundaries, n, over)

        elif var == 'evapo' or var == 'vpd' or var == 'wue':
            # Color stops for the gradient: Light Beige -> Dark Purple
            light_positive = (246/255, 232/255, 195/255) # Light Beige
            deep_positive = (103/255, 0/255, 31/255) #Dark Purple
            over = (80/255, 0/255, 20/255)  # Darker Purple

            boundaries, n = get_colmap_set_up(vmin, vmax, steps)

            cmap, cmap_norm = create_one_gradient_colormap(light_positive, deep_positive, boundaries, n, over)

        elif var == 'tran' or var == 'gpp' or var == 'lai':
            # Color stops for the gradient: Light Beige -> Dark Green
            light_positive = (246/255, 232/255, 195/255) # Light Beige
            deep_positive = (0/255, 60/255, 48/255) # Dark Green
            over = (0/255, 45/255, 30/255)  # Darker Green

            boundaries, n = get_colmap_set_up(vmin, vmax, steps)
            cmap, cmap_norm = create_one_gradient_colormap(light_positive, deep_positive, boundaries, n, over)
        else:
            raise ValueError(f"Colormap for {var} not defined")
    
    elif period == 'ssp370-historical':
        if var == 'vpd': # VPD has only positive change
            # Color stops for the gradient: Light Beige -> Dark Purple
            light_positive = (246/255, 232/255, 195/255) # Light Beige
            deep_positive = (103/255, 0/255, 31/255) #Dark Purple
            over = (64/255, 0/255, 20/255)  # Darker Purple

            boundaries, n = get_colmap_set_up(vmin, vmax, steps)
            cmap, cmap_norm = create_one_gradient_colormap(light_positive, deep_positive, boundaries, n, over)
        elif var == 'bgws':
            under = (15/255, 115/255, 15/255)  # deeper green
            deep_negative = (30/255, 130/255, 30/255) # green
            light_negative = (240/255, 255/255, 240/255) # light green
            
            over = (40/255, 125/255, 210/255)  # deeper blue
            deep_positive = (55/255, 140/255, 225/255) # blue
            light_positive = (240/255, 250/255, 255/255) # light blue

            boundaries, n = get_colmap_set_up(vmin, vmax, steps)

            cmap, cmap_norm = create_two_gradient_colormap(deep_negative, light_negative, 
                                                           deep_positive, light_positive, 
                                                           boundaries, n, under, over)
        else:
            under = (84/255, 48/255, 5/255) 
            deep_negative = (100/255, 64/255, 21/255)  
            light_negative = (246/255, 232/255, 195/255)  
            
            over = (0/255, 60/255, 48/255) 
            deep_positive = (16/255, 76/255, 64/255) 
            light_positive = (199/255, 234/255, 229/255)

            boundaries, n = get_colmap_set_up(vmin, vmax, steps)
            
            cmap, cmap_norm = create_two_gradient_colormap(deep_negative, light_negative, 
                                                                  deep_positive, light_positive, 
                                                                  boundaries, n, under, over)

    return cmap, cmap_norm          

def get_colmap_set_up(vmin, vmax, steps):
    # Define boundaries and steps for the bins
    if vmin == 0:
        boundaries = np.arange(0, vmax+steps, steps)
    else: 
        boundaries = np.arange(vmin, vmax+steps, steps)
    
    n = int((abs(vmin) + vmax) / steps)

    return boundaries, n

def create_bgws_change_colormaps(vmin, vmax, steps):
    
    # Blue Water Regime Changemap
    over = (40/255, 125/255, 210/255)  # deeper blue
    deep_positive = (55/255, 140/255, 225/255) # blue
    light_positive = (240/255, 250/255, 255/255) # light blue

    under = (150/255, 140/255, 100/255)  # deeper brown
    deep_negative = (180/255, 160/255, 120/255) # brown
    light_negative = (255/255, 249/255, 236/255) # light brown
    
    # Define boundaries and steps for the bins
    boundaries, n = get_colmap_set_up(vmin, vmax, steps)

    bw_cmap, bw_cmap_norm = create_two_gradient_colormap(deep_negative, light_negative, 
                                                         deep_positive, light_positive, 
                                                         boundaries, n, under, over)

    # Green Water Regime Changemap
    under = (15/255, 115/255, 15/255)  # deeper green
    deep_negative = (30/255, 130/255, 30/255) # green
    light_negative = (240/255, 255/255, 240/255) # light green
    
    over = (140/255, 95/255, 100/255)  # deeper violet
    deep_positive = (160/255, 113/255, 120/255) # violet
    light_positive = (255/255, 237/255, 240/255) # light violet
    
    # Define boundaries and steps for the bins
    boundaries, n = get_colmap_set_up(vmin, vmax, steps)

    gw_cmap, gw_cmap_norm = create_two_gradient_colormap(deep_negative, light_negative, 
                                                         deep_positive, light_positive, 
                                                         boundaries, n, under, over)
    
    return bw_cmap, bw_cmap_norm, gw_cmap, gw_cmap_norm