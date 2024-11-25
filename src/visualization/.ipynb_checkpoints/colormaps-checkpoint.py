"""
src/visualization/colormaps.py

This script provides functions create colormaps.

Functions:
- 

Usage:
    Import this module in your scripts.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

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

def create_two_gradient_colormap_with_white_transition(dark_color_m, light_color_m, dark_color_p, light_color_p, boundaries, cmap_name, n=6):
    """
    Create a colormap with two gradients combined, one for positive and one for negative values,
    with white at zero.

    Parameters:
    - dark_color_m: tuple, the darker color for the negative gradient.
    - light_color_m: tuple, the lighter color for the negative gradient.
    - dark_color_p: tuple, the darker color for the positive gradient.
    - light_color_p: tuple, the lighter color for the positive gradient.
    - boundaries: array-like, the boundaries for the bins.
    - cmap_name: str, the name for the colormap.
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

    colormap = LinearSegmentedColormap.from_list(cmap_name, combined_grad, N=len(combined_grad))
    norm = BoundaryNorm(boundaries, ncolors=len(combined_grad), clip=True)

    return colormap, norm

def create_two_gradient_colormap(dark_color_m, light_color_m, dark_color_p, light_color_p, boundaries, cmap_name, n=6, under_color=None, over_color=None):
    """
    Create a colormap with two gradients combined, one for positive and one for negative values.

    Parameters:
    - dark_color_m: tuple, the darker color for the negative gradient.
    - light_color_m: tuple, the lighter color for the negative gradient.
    - dark_color_p: tuple, the darker color for the positive gradient.
    - light_color_p: tuple, the lighter color for the positive gradient.
    - boundaries: array-like, the boundaries for the bins.
    - cmap_name: str, the name for the colormap.
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
    
    colormap = LinearSegmentedColormap.from_list(cmap_name, combined_grad, N=len(combined_grad))
    #norm = BoundaryNorm(boundaries, ncolors=len(combined_grad), clip=True)
    norm = BoundaryNorm(boundaries, ncolors=len(combined_grad), extend='both')

    # Set explicit under and over colors
    if under_color:
        colormap.set_under(under_color)
    if over_color:
        colormap.set_over(over_color)

    return colormap, norm


def create_one_gradient_colormap(light_color, dark_color, boundaries, cmap_name, n=6):
    """
    Create a colormap with two gradients combined, one for positive and one for negative values.

    Parameters:
    - light_color: tuple, the lighter color.
    - dark_color: tuple, the darker color.
    - boundaries: array-like, the boundaries for the bins.
    - cmap_name: str, the name for the colormap.
    - n: int, number of colors to sample.

    Returns:
    - colormap: LinearSegmentedColormap, the created colormap.
    - norm: BoundaryNorm, the corresponding BoundaryNorm for the colormap.
    """
    total_bins = len(boundaries) - 1
    n = max(n, total_bins)  # Ensure n is at least the number of bins

    colormap = LinearSegmentedColormap.from_list("color map", [light_color, dark_color], N=n)

    # Sample gradients 
    colors = [colormap(i) for i in np.linspace(0, 1, n)]
    
    norm = BoundaryNorm(boundaries, ncolors=len(colors), clip=True)

    return colormap, norm

colorbar_dict = {
    'range': {'vmax': 40, 'steps': 10, 'cmap': 'BrBG', 'unit': '%'}
}