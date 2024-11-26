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

from __future__ import annotations
import xclim.indices
from xclim import testing

import xarray as xr
import os
import copy
import numpy as np
import regionmask
from itertools import permutations
import pandas as pd
import dask
from dask.diagnostics import ProgressBar
import xesmf as xe
import sys


# ========== Configure Paths ==========
# Define the full path to the directories containing utility scripts and configurations
data_handling_dir = '../../src/data_handling'
config_file = '../../src'

# Add directories to sys.path for importing custom modules
sys.path.append(data_handling_dir)
sys.path.append(config_file)

# Import custom utility functions and configurations
import load_data as ld
import process_data as pro_dat
import save_data_as_nc as sd
from config import DATA_DIR, DEFAULT_EXPERIMENT, DEFAULT_TEMP_RES, DEFAULT_MODEL, DEFAULT_VARIABLE


def consis_time(ds_dict, ref_ds):
    """
    Creates consistent time coordinate based on a reference dataset

    Args:
        ds_dict (dict): A dictionary of xarray datasets, where each key is the name of the dataset 
                        and each value is the dataset itself.
        ref_ds (xarray): A xarray dataset as reference for the consistent time coordinate

    Returns:
        dict: A dictionary with a new time coordinate depending on the reference dataset.
    """
    time = ref_ds.time
    
    for i, (name, ds) in enumerate(ds_dict.items()):
        # Create consistent time coordinate using the first time coordinate for all following models
        if not ds['time'].equals(time):
            ds['time'] = time
            # Add comment about changes to data 
            if 'log' in ds.attrs:
                log_old = ds.attrs['log']
                ds.attrs['log'] = f'Time coordinate changed to format cftime.DatetimeNoLeap(1850, 1, 16, 12, 0, 0, 0, has_year_zero=True). // {log_old}'
            else:
                ds.attrs['log'] = 'Time coordinate changed to format cftime.DatetimeNoLeap(1850, 1, 16, 12, 0, 0, 0, has_year_zero=True).'
        else:
            print('Time variable is already in the requested format')
            
        # Update the dictionary with the modified dataset
        ds_dict[name] = ds
            
    return ds_dict

def regrid(ds_dict, method='conservative'):
    """
    Combines different grid labels via interpolation with xesmf

    Args:
        ds_dict (dict): A dictionary of xarray datasets, where each key is the name of the dataset 
                        and each value is the dataset itself.
        method (str): Interpolation method for xesmf, by default 'conservative'. Other options are "nearest_s2d", 
                        "patch", "bilinear" or "conservative_normed".

    Returns:
        dict: A dictionary of combined datasets (usually will combine across different variable ids).
    """
    
    lon_degree = 1
    lat_degree = 1
    
    # Define your output grid (with cell boundaries)
    out_lon = np.arange(-180, 180, lon_degree) 
    out_lat = np.arange(-90, 90, lat_degree)
    
    

    # Calculate boundaries as midpoints between centers
    out_lon_b = (out_lon[:-1] + out_lon[1:]) / 2  # longitude boundaries
    out_lat_b = (out_lat[:-1] + out_lat[1:]) / 2  # latitude boundaries

    # Add boundary points at the start and end of the grid
    out_lon_b = np.concatenate([out_lon[0:1] - (out_lon[1]-out_lon[0])/2, out_lon_b, out_lon[-1:] + (out_lon[-1]-out_lon[-2])/2])
    out_lat_b = np.concatenate([out_lat[0:1] - (out_lat[1]-out_lat[0])/2, out_lat_b, out_lat[-1:] + (out_lat[-1]-out_lat[-2])/2])

    # Create a grid Dataset
    ds_out = xr.Dataset(
        {
            'lat': (['lat'], out_lat),
            'lon': (['lon'], out_lon),
            'lat_b': (['lat_b'], out_lat_b),
            'lon_b': (['lon_b'], out_lon_b),
        }
    )

    for i, (name, ds) in enumerate(ds_dict.items()):
        # Check if lat/lon in dataset
        if ('lat' in ds.coords) and ('lon' in ds.coords):

            # Check if regridding is necessary
            if ds['lat'].equals(ds_out['lat']) and ds['lon'].equals(ds_out['lon']): # regridding not necessary
                    ds_dict[name] =  ds 
                    print(f"Grid of {name} already requested format")
            else:
                # Initialize regridder
                regridder = xe.Regridder(ds, ds_out, method, ignore_degenerate=True, periodic=True)

                # Regrid data
                reg_ds = regridder(ds, keep_attrs=True)

                # Assign attributes
                reg_ds.attrs.update(ds.attrs)
                
                 # Add comment about changes to data
                if 'log' in reg_ds.attrs:
                    log_old = reg_ds.attrs['log']
                    reg_ds.attrs['log'] = f'Regridded to {lon_degree}x{lat_degree}° lonxlat grid using {method} interpolation. // {log_old}'
                else:
                    reg_ds.attrs['log'] = 'Regridded to {lon_degree}x{lat_degree}° lonxlat grid using {method} interpolation.'

                # Update the ds_dict with the regridded dataset
                ds_dict[name] = reg_ds
        else:
            raise ValueError(f"No lat and lon in dataset '{name}'.")

    
    return ds_dict

def apply_landmask(ds_dict, filename, savepath):
    
    # Load landmask
    filename = 'land_sea_mask_1x1_grid.nc'
    savepath = '/work/ch0636/g300115/phd_project/common/data/external/landmask/'
    print(os.path.join(savepath, filename))
    landmask = ld.open_dataset(os.path.join(savepath, filename))
    landmask.landseamask.plot()

    # Apply landmask on data
    for i, (name, ds) in enumerate(ds_dict.items()):
        masked_ds = ds * landmask.landseamask
        print(f'Landmask applied on {name}.')
        
        masked_ds.attrs = ds.attrs
        # Add comment about changes to data 
        if 'log' in masked_ds.attrs:
            log_old = masked_ds.attrs['log']
            masked_ds.attrs['log'] = f'IMERG Land-Sea Mask NetCDF 25%-landmask applied. // {log_old}'
        else:
            masked_ds.attrs['log'] = 'IMERG Land-Sea Mask NetCDF 25%-landmask applied.'   
            
        for var in ds.variables:
            masked_ds[var].attrs = ds[var].attrs
        
        ds_dict[name] = masked_ds
    
    return ds_dict

def remove_antarctica_greenland_iceland(ds_dict):
    # Define the regions using regionmask
    land_regions = regionmask.defined_regions.ar6.land
    
    # Identify the index for the GIC Greenland/Iceland region
    gic_region_idx = land_regions.map_keys('Greenland/Iceland')
    
    for name, ds in ds_dict.items():
        # Get the mask for the GIC Greenland/Iceland region
        gic_region_mask = land_regions.mask(ds)
        
        # Slice the dataset to exclude Antarctica latitudes
        ds_no_antarctica = ds.sel(lat=(ds.lat > -60))
        
        # Apply the mask to exclude the GIC Greenland/Iceland region
        mask = gic_region_mask == gic_region_idx
        ds_no_gic = ds_no_antarctica.where(~mask, drop=True)

        print(f'Regions removed from {name}.')
        
        # Add comment about changes to data
        if 'log' in ds_no_gic.attrs:
            log_old = ds_no_gic.attrs['log']
            ds_no_gic.attrs['log'] = f'Dataset sliced along lat 60 to remove Antarctica and excluded Greenland and Iceland using regionmask. // {log_old}'
        else:
            ds_no_gic.attrs['log'] = 'Dataset sliced along lat 60 to remove Antarctica and excluded Greenland and Iceland using regionmask.'
        
        ds_dict[name] = ds_no_gic
    
    return ds_dict

def set_units(ds_dict, conv_units):
    """
     Convert units for specified variables
    """

    ds_dict_copy = copy.deepcopy(ds_dict)  
    
    for i, (name, ds) in enumerate(ds_dict_copy.items()):

        for var in list(conv_units.keys()):

            if var in ds.variables:
                old_unit = ds[var].units
                
                if conv_units[var] == ds[var].units:
                    print(f'Unit {var} for model {name} already in the requested format')
                    
                elif var == 'lai':
                    # Keep existing attributes and only modify the units attribute
                    attrs = ds[var].attrs
                    attrs['units'] = conv_units[var]
                    attrs['equation'] = 'leaf area / ground area'
                    ds[var].attrs = attrs
                    ds = create_log(ds, name, var, old_unit)


                elif conv_units[var] == 'gC/m²/day':
                    if ds[var].units == 'kg/m²/s' or 'kg m-2 s-1':
                    
                        # Keep existing attributes and only modify the units attribute
                        attrs = ds[var].attrs
                        attrs['units'] = conv_units[var]
                        ds[var] = ds[var] * 1000 * 60 * 60 * 24 
                        ds[var].attrs = attrs
                        ds = create_log(ds, name, var, old_unit)
                        
                elif conv_units[var] == 'mm/day':
                    if ds[var].units == 'kg/m²/s' or ds[var].units =='kg m-2 s-1':
    
                        # Keep existing attributes and only modify the units attribute
                        attrs = ds[var].attrs
                        attrs['units'] = conv_units[var]
                        ds[var] = ds[var] * 60 * 60 * 24 
                        ds[var].attrs = attrs
                        ds = create_log(ds, name, var, old_unit)
                
                elif conv_units[var] == 'hPa':
                    if ds[var].units == 'Pa':
    
                        # Keep existing attributes and only modify the units attribute
                        attrs = ds[var].attrs
                        attrs['units'] = conv_units[var]
                        ds[var] = ds[var] / 100 
                        ds[var].attrs = attrs
                        ds = create_log(ds, name, var, old_unit)
                        
                elif conv_units[var] == 'ppm':
                    if ds[var].units == 'kg':
                        # Keep existing attributes and only modify the units attribute
                        attrs = ds[var].attrs
                        attrs['units'] = conv_units[var]
                        attrs['long_name'] = 'CO2 concentration'
                        # Constants
                        molar_mass_co2 = 44.01  # Molar mass of CO2 in grams per mole (g/mol)
                        moles_of_air = 2.13e20  # Volume of the atmosphere in moles (may vary, check CMIP6 documentation)
                        # Convert co2mass from kg to moles
                        ds[var] = ((ds[var] / molar_mass_co2) / moles_of_air) * 1e6
                        ds[var].attrs = attrs
                        ds = create_log(ds, name, var, old_unit)
                            
                elif conv_units[var] == '°C':
                    if ds[var].units == 'K':
                        # Keep existing attributes and only modify the units attribute
                        attrs = ds[var].attrs
                        attrs['units'] = conv_units[var]
                        # Convert co2mass from kg to moles
                        ds[var] = ds[var] - 273.15
                        ds[var].attrs = attrs
                        ds = create_log(ds, name, var, old_unit)
                
                elif conv_units[var] == 'mm':
                    if ds[var].units == 'kg/m²':
                        # Keep existing attributes and only modify the units attribute
                        attrs = ds[var].attrs
                        attrs['units'] = conv_units[var]
                        # Convert co2mass from kg to moles
                        ds[var] = ds[var] / 1e3  # This now represents mm of water
                        ds[var].attrs = attrs
                        ds = create_log(ds, name, var, old_unit)

                else: 
                    raise ValueError(f"No unit conversion for variable '{var}' specified.")

            else:
                print(f"No variable '{var}' in ds_dict.")
        
        ds_dict_copy[name] = ds
                
    return ds_dict_copy

def create_log(ds, name, var, old_unit):

        if 'log' in ds.attrs:
            log_old = ds.attrs['log']
            ds.attrs['log'] = f'Unit of {var} converted from {old_unit} to {ds[var].units}. // {log_old}'
        else:
            ds.attrs['log'] = f'Unit of {var} converted from {old_unit} to {ds[var].units}.'

        print(f"Unit of {var} for model {name} converted from {old_unit} to {ds[var].units}.")
        return ds


#### Compute variables

def compute_evapo(ds_dict):

    ds_dict_copy = copy.deepcopy(ds_dict)
    
    for name, ds in ds_dict_copy.items():
        # Compute evapo as the difference between evspsbl and tran
        ds['evapo'] = ds['evspsbl'] - ds['tran']
        
        # Compute evapo as the difference between evspsbl and tran
        ds['evapo'] = ds['evspsbl'] - ds['tran']
        
        # Assign attributes to evapo
        ds['evapo'].attrs = {'standard_name': 'water_evaporation_flux',
                             'long_name': 'Evaporation',
                             'comment': 'Evaporation at surface: flux of water into the atmosphere due to conversion of both liquid and solid phases to vapor (from underlying surface) - excluding transpiration',
                             'cell_methods': 'area: time: mean (interval: 5 minutes)',
                             'cell_measures': 'area: areacella',
                             'units': 'mm/day',
                             'log': 'Negative values are set to 0'}
        
        # Update the dictionary with the modified dataset
        ds_dict_copy[name] = ds
        
    return ds_dict_copy

def compute_RX5day(ds_dict, freq='YS'):
    new_ds_dict = {}
    
    for name, ds in ds_dict.items():
        # Compute the RX5day index
        rx5day = xclim.indicators.icclim.RX5day(pr=ds.pr, freq=freq)

        # Create a new dataset with the computed RX5day
        new_ds = xr.Dataset({'RX5day': rx5day})
        
        # Assigning time coordinates from the computed RX5day to the new dataset
        new_ds = new_ds.assign_coords(time=rx5day.time)
        
        # Store the new dataset in the dictionary
        new_ds_dict[name] = new_ds

        new_ds_dict[name].attrs = ds.attrs
        
        new_ds_dict[name].attrs['table_id'] = freq

    return new_ds_dict

def convert_units_wue(ds, var_name, target_units):
    """
    Convert units of a variable to the target units if necessary.

    Parameters:
    ds (xarray.Dataset): The dataset containing the variable.
    var_name (str): The name of the variable to convert.
    target_units (str): The target units ('mm/day' or 'gC/m²/day').

    Returns:
    xarray.DataArray: The variable with converted units.
    """
    var = ds[var_name]
    if 'units' in var.attrs:
        current_units = var.attrs['units']
        if var_name == 'tran' and current_units != target_units:
            if current_units == 'kg/m²/s':
                var = var * 86400  # Convert from kg/m²/s to mm/day
                var.attrs['units'] = 'mm/day'
        elif var_name == 'gpp' and current_units != target_units:
            # Assuming gpp might be in other units, add conversion logic if needed
            pass
        var.attrs['units'] = target_units
    return var

def compute_wue(ds_dict, wue_threshold=10):
    """
    Compute the Water Use Efficiency (WUE) for each dataset in the dictionary.

    Parameters:
    ds_dict (dict): Dictionary of datasets, each containing 'tran' (transpiration) and 'gpp' (gross primary production).
    wue_threshold (float): The maximum realistic value for WUE, above which values will be set to NaN.

    Returns:
    dict: Updated dictionary with WUE included in each dataset.
    """
    ds_dict_copy = copy.deepcopy(ds_dict)
    models_with_wue = []
    realistic_range = (0.1, 5)  # Set a realistic range for WUE values in gC/m²/mm

    for name, ds in ds_dict_copy.items():
        # Check if all required variables are present
        required_vars = ['tran', 'gpp']
        missing_vars = [var for var in required_vars if var not in ds]

        if missing_vars:
            print(f'WUE not computed for {name} as variable(s) {", ".join(missing_vars)} is/are missing')
            continue

        # Convert units if necessary
        ds['tran'] = convert_units_wue(ds, 'tran', 'mm/day')
        ds['gpp'] = convert_units_wue(ds, 'gpp', 'gC/m²/day')

        # Resample to yearly sums (assuming 'time' is the dimension in your dataset)
        yearly_gpp = ds['gpp'].resample(time='1Y').sum(dim='time')
        yearly_tran = ds['tran'].resample(time='1Y').sum(dim='time')

        # Compute WUE using the yearly sums
        wue_yearly = yearly_gpp / yearly_tran

        # Optionally, you can set a threshold to remove extreme values
        wue = xr.where(wue_yearly > wue_threshold, np.nan, wue_yearly)

        # Assign attributes to the WUE variable
        attrs = {
            "description": "This dataset contains the Water Use Efficiency (WUE) computed from gross primary production (gpp) and transpiration (tran).",
            "units": "gC/m²/mm",
            "long_name": "Water Use Efficiency",
            "calculation": f"WUE was computed using the formula WUE = gpp / tran with WUE values above {wue_threshold} gC/m²/mm set to NaN.",
            "source": "Data sourced from the CMIP6 archive.",
        }

        # Add WUE to the original dataset
        wue_var = xr.DataArray(wue, dims=yearly_gpp.dims, coords=yearly_gpp.coords, attrs=attrs)
        ds['wue'] = wue_var

        # Save the updated dataset back
        ds_dict_copy[name] = ds
        models_with_wue.append(name)

        print(f'WUE computed and saved for {name}')

    return ds_dict_copy

def convert_units_vpd(ds, var_name, target_units):
    """
    Convert units of a variable to the target units.

    Parameters:
    ds (xarray.Dataset): The dataset containing the variable.
    var_name (str): The name of the variable to convert.
    target_units (str): The target units ('Pa' or 'C').

    Returns:
    xarray.DataArray: The variable with converted units.
    """
    var = ds[var_name]
    if 'units' in var.attrs:
        current_units = var.attrs['units']
        if var_name == 'ps' and current_units == 'hPa' and target_units == 'Pa':
            var = var * 100
        elif var_name == 'tas' and current_units == 'K' and target_units == 'C':
            var = var - 273.15
        elif var_name == 'tas' and current_units == 'C' and target_units == 'K':
            var = var + 273.15
        var.attrs['units'] = target_units
    return var

def compute_vpd(ds_dict):
    """
    Compute the Vapor Pressure Deficit (VPD) for each dataset in the dictionary.

    Parameters:
    ds_dict (dict): Dictionary of datasets, each containing 'huss' (specific humidity), 'ps' (surface air pressure),
                    and 'tas' (near-surface air temperature).

    Returns:
    dict: Updated dictionary with VPD included in each dataset.
    """
    import copy
    import xarray as xr
    import numpy as np

    ds_dict_copy = copy.deepcopy(ds_dict)
    
    for name, ds in ds_dict_copy.items():
        # Check if all required variables are present
        required_vars = ['huss', 'ps', 'tas']
        missing_vars = [var for var in required_vars if var not in ds]
        
        if missing_vars:
            print(f'VPD not computed for {name} as variable(s) {", ".join(missing_vars)} is/are missing')
            continue
        
        # Convert units if necessary
        ds['ps'] = convert_units_vpd(ds, 'ps', 'Pa')
        ds['tas'] = convert_units_vpd(ds, 'tas', 'C')
        
        # Ensure temperature is in Celsius for the Buck equation
        T = ds['tas']
        
        # Compute saturation vapor pressure (e_s) using the Buck equation
        e_s = 611.21 * np.exp((18.678 - T / 234.5) * (T / (257.14 + T)))
        
        # Compute actual vapor pressure (e_a)
        q = ds['huss']
        p_s = ds['ps']
        e_a = (q * p_s) / (0.622 + (0.378 * q))
        
        # Compute VPD
        vpd = e_s - e_a
        
        # Ensure VPD is not negative
        vpd = xr.where(vpd < 0, 0, vpd)
        
        # Convert VPD from Pa to hPa
        vpd_hPa = vpd / 100
        
        # Assign attributes to the VPD variable
        attrs = {
            "description": "This dataset contains the Vapor Pressure Deficit (VPD) computed from specific humidity (huss) and surface air pressure (ps), and converted to hPa.",
            "units": "hPa",
            "long_name": "Vapor Pressure Deficit",
            "calculation": "VPD was computed using the formula VPD = e_s - e_a, where e_s is the saturation vapor pressure and e_a is the actual vapor pressure, and converted to hPa.",
            "source": "Data sourced from the CMIP6 archive.",
            "created_by": "Simon P.Heselschwerdt"
        }
        
        # Add VPD to the original dataset
        vpd_var = xr.DataArray(vpd_hPa, dims=ds['tas'].dims, coords=ds['tas'].coords, attrs=attrs)
        ds['vpd'] = vpd_var
        
        # Save the updated dataset back
        ds_dict_copy[name] = ds
        
        print(f'VPD computed and saved for {name}')
    
    return ds_dict_copy

def remove_variables(ds_dict, variables_to_remove):
    """
    Removes specified variables from each dataset in the dictionary.

    Parameters:
    ds_dict (dict): Dictionary of xarray datasets.
    variables_to_remove (list): List of variable names to check and remove from the datasets.

    Returns:
    dict: Updated dictionary with specified variables removed from each dataset.
    """
    import copy

    # Create a deep copy of the input dictionary to avoid modifying the original
    ds_dict_copy = copy.deepcopy(ds_dict)

    for name, ds in ds_dict_copy.items():
        for var in variables_to_remove:
            if var in ds:
                del ds[var]  # Remove the variable if it exists in the dataset
                print(f"Variable '{var}' removed from dataset '{name}'")

    return ds_dict_copy




############

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
