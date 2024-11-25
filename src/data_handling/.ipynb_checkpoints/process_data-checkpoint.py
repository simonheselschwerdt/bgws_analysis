"""
src/data_handling/process_data.py

This script provides functions to process data. Users can select specific periods, compute different statistics, 
apply region masks or compute the bgws variable.

Functions:
- select_period
- drop_var
- standardize
- apply_region_mask
- compute_change
- compute_bgws

Usage:
    Import this module in your scripts to process data for the analysis.
"""

import xarray as xr
import os
import copy
import numpy as np
import regionmask
from itertools import permutations
import pandas as pd
import dask
from dask.diagnostics import ProgressBar

# ------- SELECT PERIOD ---------

# Constants
SEASONS_TO_MONTHS = {
    'nh_winter': [12, 1, 2],
    'nh_spring': [3, 4, 5],
    'nh_summer': [6, 7, 8],
    'nh_fall': [9, 10, 11]
}

MONTH_NAMES = {
    1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J',
    7: 'J', 8: 'A', 9: 'S', 10: 'O', 11: 'N', 12: 'D'
}

def get_time_selection_months(time_selection):
    """
    Determine the months and time selection name based on the provided time selection.

    Parameters:
    - time_selection: Selection of time spans within the dataset.

    Returns:
    - Tuple containing a list of months and a time selection name.
    """
    months = []
    time_selection_name = 'whole_year'
    
    if time_selection is None:
        months = list(range(1, 13))  # All months
    elif isinstance(time_selection, int):
        time_selection_name = MONTH_NAMES[time_selection]
        months = [time_selection]
    elif isinstance(time_selection, str):
        if 'and' in time_selection:
            seasons = time_selection.lower().split('and')
            time_selection_name = ''
            for season in seasons:
                season = season.strip()
                months.extend(SEASONS_TO_MONTHS.get(season, []))
                time_selection_name += ''.join(MONTH_NAMES[m] for m in SEASONS_TO_MONTHS.get(season, []))
        else:
            months = SEASONS_TO_MONTHS.get(time_selection.lower(), [])
            time_selection_name = ''.join(MONTH_NAMES[m] for m in months)
    elif isinstance(time_selection, list):
        time_selection_name = ''.join(MONTH_NAMES[m] for m in time_selection if m in MONTH_NAMES)
        months = time_selection
    else:
        raise ValueError("time_selection must be None, an integer, a string representing a single season, "
                         "a string with multiple seasons separated by 'and', or a list of integers.")
    return months, time_selection_name

def select_time_period(ds, start_year, end_year):
    """
    Select the data within the specified time period.

    Parameters:
    - ds: The dataset to select from.
    - start_year: The start year of the period.
    - end_year: The end year of the period.

    Returns:
    - The dataset containing data only within the specified time period.
    """
    if start_year and end_year:
        start_date = f'{start_year}-01-01'
        end_date = f'{end_year}-12-31'
        ds = ds.sel(time=slice(start_date, end_date))
    return ds

def select_months(ds, months):
    """
    Filter the data to include only the specified months.

    Parameters:
    - ds: The dataset to filter.
    - months: List of months to include.

    Returns:
    - The filtered dataset.
    """
    if months:
        month_mask = ds['time.month'].isin(months)
        ds = ds.where(month_mask, drop=True)
    return ds


def select_period(ds_dict, start_year=None, end_year=None, specific_months_or_seasons=None):
    """
    Select period and optionally compute yearly sums.

    Parameters:
    - ds_dict: Dictionary with xarray datasets.
    - start_year: The start year of the period.
    - end_year: The end year of the period.
    - specific_months_or_seasons (int, list, str, None): Single month (int), list of months (list), multiple seasons (str) to select,
                                             or None to not select any specific month or season.

    Returns:
    - A dictionary containing copies of the original datasets with data only for the selected time selection.
    """
    # Create a deep copy of the original ds_dict to avoid modifying it directly
    ds_dict_copy = copy.deepcopy(ds_dict)

    # Get the months and time_selection name based on the provided time_selection
    months, time_selection_name = get_time_selection_months(specific_months_or_seasons)

    for key, ds in ds_dict_copy.items():
        # Select the time period
        ds = select_time_period(ds, start_year, end_year)
        
        # Select the months
        ds = select_months(ds, months)

        # Store the original attributes of each variable
        original_attrs = {var: ds[var].attrs for var in ds.data_vars}

        # Reassign the original attributes back to each variable
        for var in ds.data_vars:
            ds[var].attrs = original_attrs[var]

        ds_dict_copy[key] = ds
        ds_dict_copy[key].attrs['months'] = time_selection_name

    return ds_dict_copy


def drop_var(ds_dict, var):
    for name, ds in ds_dict.items():
        ds_dict[name] = ds.drop(var)
        
    return ds_dict

def standardize(ds_dict):
    '''
    Helper function to standardize datasets of a dictionary
    '''
    ds_dict_stand = {}
    
    for name, ds in ds_dict.items():
        attrs = ds.attrs
        ds_stand = (ds - ds.mean()) / ds.std()

        # Preserve variable attributes from the original dataset
        for var in ds.variables:
            if var in ds_stand.variables:
                ds_stand[var].attrs = ds[var].attrs

        ds_stand.attrs = attrs
        ds_dict_stand[name] = ds_stand
        
    return ds_dict_stand

def apply_region_mask(ds_dict, with_global=False):
    """
    Applies the AR6 land region mask to datasets in the provided dictionary, adds a region dimension,
    and optionally includes a 'Global' aggregation.

    Args:
        ds_dict (dict): A dictionary of xarray datasets organized by experiments and models.
        with_global (bool): If True, includes a 'Global' region with aggregated data.

    Returns:
        dict: A new dictionary where keys are the same as in the input dictionary,
              and each value is an xarray Dataset with a region dimension added to each variable,
              and optionally includes a 'Global' region.
    """

    land_regions = regionmask.defined_regions.ar6.land
    
    if with_global:
        global_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    
    ds_masked_dict = {}

    for experiment in ds_dict.keys():
        ds_masked_dict[experiment] = {}
        
        for ds_name, ds in ds_dict[experiment].items():
            ds_masked = xr.Dataset()  # Initiate an empty Dataset for the masked data
            
            for var in ds:
                # Get the binary mask
                mask = land_regions.mask_3D(ds[var])
                
                # Expand the mask to include the time dimension if it exists
                if 'time' in ds[var].dims:
                    mask = mask.expand_dims({'time': ds[var].time}, axis=0)

                var_attrs = ds[var].attrs
    
                # Multiply the original data with the mask to get the masked data
                masked_var = ds[var] * mask
    
                # Replace 0s with NaNs
                masked_var = masked_var.where(masked_var != 0)
                
                if with_global:
                    # Convert the global mask to 3D to match the regional mask dimensions
                    glob_mask = global_mask.mask_3D(ds[var])
                    
                    # Expand the global mask to include the time dimension if it exists
                    if 'time' in ds[var].dims:
                        glob_mask = glob_mask.expand_dims({'time': ds[var].time}, axis=0)
                    
                    global_masked_var = ds[var] * glob_mask
                    
                    # Replace 0s with NaNs
                    global_masked_var = global_masked_var.where(global_masked_var != 0)

                    # Combine masked data
                    masked_var = xr.concat([masked_var, global_masked_var], dim='region')

                    # Rename the 'names' and 'abbrevs' coordinate
                    masked_var = masked_var.assign_coords(names=('region', ['Global' if name == 'land' else name for name in masked_var['names'].values]))
                    masked_var = masked_var.assign_coords(abbrevs=('region', ['glob' if abbrevs == 'lnd' else abbrevs for abbrevs in masked_var['abbrevs'].values]))
    
                # Remove regions that contain only NaN values
                non_nan_regions = ~masked_var.isnull().all(dim=['lat', 'lon', 'time'] if 'time' in ds[var].dims else ['lat', 'lon'])
                masked_var = masked_var.isel(region=non_nan_regions)
                
                # Add the masked variable to the output Dataset
                ds_masked[var] = masked_var
                ds_masked[var].attrs = var_attrs
    
            # Copy dataset attributes
            ds_masked.attrs.update(ds.attrs)
            
            # Add the modified dataset to the dictionary
            ds_masked_dict[experiment][ds_name] = ds_masked

    return ds_masked_dict

def is_numeric(data):
    try:
        _ = data.astype(float)
        return True
    except (ValueError, TypeError):
        return False

def compute_change_dict(ds_dict, var_rel_change=None):
    """
    Computes the change (absolute or relative) between historical and future datasets.

    Parameters:
    ds_dict (dict): A dictionary of xarray datasets with keys representing periods (e.g., 'historical', 'ssp245').
    var_rel_change (str or list, optional): Variables for which to compute relative change. 
                                            If 'all', compute relative change for all variables. 
                                            If None, compute absolute change for all variables.

    Returns:
    dict: A dictionary of xarray datasets with computed changes.
    """
    periods = list(ds_dict.keys())
    if 'historical' not in periods or len(periods) < 2:
        print("The dictionary must contain at least 'historical' and one future period key.")
        return {}

    ds_dict_change = {}
    historical_period = copy.deepcopy(ds_dict['historical'])

    for period in periods:
        if period == 'historical':
            continue

        ds_hist = copy.deepcopy(historical_period)
        ds_future = copy.deepcopy(ds_dict[period])

        # Remove any existing ensemble statistics
        keys_to_remove = [key for key in ds_hist.keys() if key.startswith('Ensemble ')]
        if keys_to_remove:
            print(f"Ensemble mean or median removed for keys: {keys_to_remove}")
        for key in keys_to_remove:
            ds_hist.pop(key, None)
            ds_future.pop(key, None)

        ds_dict_change[f'{period}-historical'] = {}

        for model in ds_hist.keys():
            common_vars = set(ds_hist[model].data_vars).intersection(ds_future[model].data_vars)
            ds_change = ds_hist[model].copy(deep=True)
            
            if var_rel_change == 'all':
                var_rel_change = common_vars

            for var in common_vars:
                if var == 'mrso' or (var_rel_change is not None and var in var_rel_change):
                    # Compute relative change where ds_hist is not zero
                    rel_change = (ds_future[model][var] - ds_hist[model][var]) / ds_hist[model][var].where(ds_hist[model][var] != 0) * 100
                    ds_change[var].data = rel_change.data
                    ds_change[var].attrs['units'] = '%'
                else:
                    # Compute absolute change
                    abs_change = ds_future[model][var] - ds_hist[model][var]
                    ds_change[var].data = abs_change.data
            
            ds_change.attrs = ds_future[model].attrs
            ds_change.attrs['computed_from'] = f'historical and {period}'
            ds_dict_change[f'{period}-historical'][model] = ds_change

    return ds_dict_change

def compute_bgws(ds_dict):
    """
    Computes the Blue Green Water Share (BGWS) for the given datasets.

    Parameters:
    ds_dict (dict): A dictionary of xarray datasets, potentially nested with experiments and models.

    Returns:
    dict: The input dictionary with the computed BGWS added, excluding any ensemble data.
    """
    def compute_bgws_for_ds(ds):
        bgws = ((ds['mrro'] - ds['tran']) / ds['pr']) * 100
        
        # Replace infinite values with NaN
        bgws = xr.where(np.isinf(bgws), float('nan'), bgws)
        
        # Set all values above 200 and below -200 to NaN
        bgws = xr.where(bgws > 200, float('nan'), bgws)
        bgws = xr.where(bgws < -200, float('nan'), bgws)
        
        ds['bgws'] = bgws
        ds['bgws'].attrs = {'long_name': 'Blue-Green Water Share', 'units': ''}
        return ds

    # Create a new dictionary to avoid modifying the input directly
    ds_dict_clean = {}

    # Check if the dictionary is nested with experiments
    for key, value in ds_dict.items():
        if isinstance(value, dict):  # Nested with experiments
            ds_dict_clean[key] = {}
            for model, ds in value.items():
                if not model.startswith('Ensemble '):
                    ds_dict_clean[key][model] = compute_bgws_for_ds(ds)
                else:
                    print(f"Ignored ensemble data for {key} - {model}")
        else:  # Directly datasets
            if not key.startswith('Ensemble '):
                ds_dict_clean[key] = compute_bgws_for_ds(value)
            else:
                print(f"Ignored ensemble data for {key}")

    return ds_dict_clean

def compute_tbgw(ds_dict):
    """
    Computes the Total Blue-Green Water (TBGW) (sum of runoff and transpiration divided by precipitation) for the given datasets.

    Parameters:
    ds_dict (dict): A dictionary of xarray datasets, potentially nested with experiments and models.

    Returns:
    dict: The input dictionary with the computed TBGW added, excluding any ensemble data.
    """
    def compute_tbgw_for_ds(ds):
        tbgw = ((ds['mrro'] + ds['tran']) / ds['pr']) * 100
        
        # Replace infinite values with NaN
        tbgw = xr.where(np.isinf(tbgw), float('nan'), tbgw)
        
        # Set all values above 200 and below 0 to NaN (adjust threshold if needed)
        tbgw = xr.where(tbgw > 200, float('nan'), tbgw)
        tbgw = xr.where(tbgw < 0, float('nan'), tbgw)
        
        ds['tbgw'] = tbgw
        ds['tbgw'].attrs = {'long_name': 'Total Blue-Green Water', 'units': ''}
        return ds

    # Create a new dictionary to avoid modifying the input directly
    ds_dict_clean = {}

    # Check if the dictionary is nested with experiments
    for key, value in ds_dict.items():
        if isinstance(value, dict):  # Nested with experiments
            ds_dict_clean[key] = {}
            for model, ds in value.items():
                if not model.startswith('Ensemble '):
                    ds_dict_clean[key][model] = compute_tbgw_for_ds(ds)
                else:
                    print(f"Ignored ensemble data for {key} - {model}")
        else:  # Directly datasets
            if not key.startswith('Ensemble '):
                ds_dict_clean[key] = compute_tbgw_for_ds(value)
            else:
                print(f"Ignored ensemble data for {key}")

    return ds_dict_clean

def subdivide_ds_dict_regions(ds_dict_base_region, ds_dict_change_region, base_id, change_id, variable):
    """
    Subdivide datasets for multiple models based on the specified variable.

    Parameters:
    ds_dict_region: Dictionary of historical datasets indexed by model.
    ds_dict_change_region: Dictionary of change datasets indexed by model.
    variable: Variable used to determine the subdivision.

    Returns:
    Dictionary of subdivided datasets.
    """
    ds_dict_change_region_sub = {}
    ds_dict_change_region_sub[change_id] = {}
    
    for model, ds_region in ds_dict_base_region.items():
        ds_dict_change_region_sub[change_id][model] = subdivide_ds_by_region(
            ds_region, ds_dict_change_region[model], base_id, change_id, 
            variable
        )

    return ds_dict_change_region_sub

def subdivide_ds_by_region(ds_base, ds_change, base_id, change_id, variable='bgws'):
    """
    Subdivide a single dataset into regions based on historical data of a variable.
    
    Parameters:
    ds_historical: Historical dataset.
    ds_change: Change dataset.
    variable: Variable used for subdivision.

    Returns:
    Expanded dataset with subdivisions.
    """
    mask_positive = ds_base[variable] > 0
    mask_negative = ds_base[variable] < 0

    subdivisions_masks = xr.DataArray(
        np.array([mask_positive, mask_negative]),
        dims=['subdivision', 'lat', 'lon', 'region'],
        coords={
            'subdivision': [f'Positive {base_id.capitalize()} {variable.upper()}', 
                            f'Negative {base_id.capitalize()} {variable.upper()}'],
            'lat': ds_base.lat,
            'lon': ds_base.lon,
            'region': ds_base.region
        }
    )

    def expand_dataset(ds, subdivisions_masks):
        expanded_vars = {}
        for name, var in ds.data_vars.items():
            # Expand the variable along the subdivision dimension
            expanded_var = var.expand_dims(subdivision=subdivisions_masks.subdivision)
            # Apply the masks to create the subdivisions
            expanded_var = expanded_var.where(subdivisions_masks)
            expanded_vars[name] = expanded_var

        # Create the expanded dataset
        expanded_ds = xr.Dataset(expanded_vars, coords={**ds.coords, 'subdivision': subdivisions_masks.subdivision})
        return expanded_ds

    return expand_dataset(ds_change, subdivisions_masks)
