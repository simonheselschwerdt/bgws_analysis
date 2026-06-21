"""
src/data_handling/process_data.py

Functions:
- consis_time
- create_log
- regrid
- apply_landmask
- remove_antarctica
- remove_antarctica_greenland_iceland
- set_units
- remove_variables
- drop_redundant
- consistent_time_coordinate
- preprocess_data_and_save
- compute_and_save_evapo
- load_and_get_ensemble
- select_period
- get_time_selection_months
- select_months
- compute_diff_dict

Author: Simon P. Heselschwerdt
Date: 2026-02-26
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
import xesmf as xe
import sys
import copy
import datetime
import cftime
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

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
import compute_statistics as comp_stats
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

def regrid(ds_dict, lon_lat_degree=1, method='conservative'):
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
    
    lon_res = lat_res = lon_lat_degree
    
    # Define output grid (with cell boundaries) 
    out_lon = np.arange(-180, 180, lon_res) 
    out_lat = np.arange(-90, 90, lat_res) 
    
    # Calculate boundaries as midpoints between centers 
    out_lon_b = (out_lon[:-1] + out_lon[1:]) / 2 # longitude boundaries 
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
                    reg_ds.attrs['log'] = f'Regridded to {lon_res}x{lat_res}° lonxlat grid using {method} interpolation. // {log_old}'
                else:
                    reg_ds.attrs['log'] = f'Regridded to {lon_res}x{lat_res}° lonxlat grid using {method} interpolation.'

                # Update the ds_dict with the regridded dataset
                ds_dict[name] = reg_ds
        else:
            raise ValueError(f"No lat and lon in dataset '{name}'.")
   
    return ds_dict

def apply_landmask(ds_dict, filename, savepath):
    """
    Apply a land mask to a dataset using `regionmask` / land-sea information.
    
    Implementation is unchanged; this docstring is added for clarity only.
    """
    
    # Load landmask
    #filename = 'land_sea_mask_1x1_grid.nc'
    #savepath = '/work/ch0636/g300115/phd_project/common/data/landmasks/landmask/'
    print(os.path.join(savepath, filename))
    landmask = ld.open_dataset(os.path.join(savepath, filename))
    #landmask.landseamask.plot()

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
    """
    Remove Antarctica, Greenland and Iceland from the dataset (unchanged implementation).
    """
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

def remove_antarctica(ds_dict):
    """
    Remove Antarctica from the dataset (unchanged implementation).
    """
    for name, ds in ds_dict.items():
        # Slice the dataset to exclude Antarctica latitudes
        ds_no_antarctica = ds.sel(lat=(ds.lat > -60))
        
        print(f'Antarctica removed from {name}.')
        
        # Add comment about changes to data
        if 'log' in ds_no_antarctica.attrs:
            log_old = ds_no_antarctica.attrs['log']
            ds_no_antarctica.attrs['log'] = f'Dataset sliced along lat 60 to remove Antarctica. // {log_old}'
        else:
            ds_no_antarctica.attrs['log'] = 'Dataset sliced along lat 60 to remove Antarctica.'
        
        ds_dict[name] = ds_no_antarctica
    
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
                    if ds[var].units in ['kg/m²', 'kg m-2']:
                        # Keep existing attributes and only modify the units attribute
                        attrs = ds[var].attrs
                        attrs['units'] = 'mm'
                        # No conversion needed; 1 kg/m² = 1 mm of water
                        ds[var].attrs = attrs
                        ds = create_log(ds, name, var, old_unit)

                elif conv_units[var] == 'μm':
                    old_unit = ds[var].attrs.get('units', 'unknown')
                    attrs = ds[var].attrs
                    attrs['units'] = 'μm'
                
                    if old_unit in ['kg/m²', 'kg m-2']:
                        # Convert: kg/m² → mm → μm
                        ds[var] = ds[var] / 1e3  # kg/m² → mm
                        ds[var] = ds[var] * 1e3  # mm → μm
                    elif old_unit in ['mm']:
                        # Convert: mm → μm
                        ds[var] = ds[var] * 1e3
                    else:
                        raise ValueError(f"Unsupported unit for μm conversion: {old_unit}")
                
                    ds[var].attrs = attrs
                    ds = create_log(ds, name, var, old_unit)

                else: 
                    raise ValueError(f"No unit conversion for variable '{var}' specified.")

            else:
                print(f"No variable '{var}' in ds_dict.")
        
        ds_dict_copy[name] = ds
                
    return ds_dict_copy

def create_log(ds, name, var, old_unit):
        """
        Helper to append/initialize a short provenance string in dataset attributes.
        
        This is used by some processing steps (e.g., unit conversion) to track modifications.
        """

        if 'log' in ds.attrs:
            log_old = ds.attrs['log']
            ds.attrs['log'] = f'Unit of {var} converted from {old_unit} to {ds[var].units}. // {log_old}'
        else:
            ds.attrs['log'] = f'Unit of {var} converted from {old_unit} to {ds[var].units}.'

        print(f"Unit of {var} for model {name} converted from {old_unit} to {ds[var].units}.")
        return ds


#### Compute variables

from pathlib import Path
import xarray as xr

def variable_already_saved(DATA_DIR, experiment, temp_res, model_name, var_name, verify_var=True):
    """
    Check whether a saved NetCDF exists for (experiment/temp_res/model_name) and contains var_name.

    Looks in:
      1) processed/<experiment>/<temp_res>/<var_name>/<model_name>.nc   (fast path)
      2) recursively under processed/<experiment>/<temp_res>/ for *<model_name>*.nc

    Returns
    -------
    (exists: bool, path: str | None)
    """
    base = Path(DATA_DIR) / "processed" / experiment / temp_res
    if not base.exists():
        return False, None

    # 1) Fast path: your save_files layout
    expected = base / var_name / f"{model_name}.nc"
    if expected.exists():
        if not verify_var:
            return True, str(expected)
        try:
            with xr.open_dataset(expected, decode_times=False) as ds:
                return (var_name in ds.variables), str(expected) if (var_name in ds.variables) else (False, None)
        except Exception:
            # file exists but can't be opened -> treat as not usable
            return False, None

    # 2) Fallback: recursive search for model file anywhere under temp_res
    # (handles other directory conventions)
    candidates = list(base.rglob(f"*{model_name}*.nc"))

    for fp in candidates:
        if not verify_var:
            return True, str(fp)
        try:
            with xr.open_dataset(fp, decode_times=False) as ds:
                if var_name in ds.variables:
                    return True, str(fp)
        except Exception:
            continue

    return False, None



def compute_and_save_evapo(selected_model, experiment, member_id):
    """
    Compute `evapo` using the existing pipeline and save to disk (unchanged implementation).
    """
    data_state = 'processed'              
    variables=["evspsbl", "tran"]
    model_name = f"{selected_model}_{member_id}"

    print(model_name)

    exists, path = variable_already_saved(DATA_DIR, experiment, "month", model_name, "evapo")
    print(exists, path)
    if exists:
        print(f"[SKIP] evapo already saved for {model_name} ({experiment}) in {path}")
        return None

    # Step 1.1: Load the datasets
    print("Loading datasets...")
    with ProgressBar():
        ds_dict_evapo = dask.compute(
            ld.load_multiple_models_and_experiments(
                DATA_DIR, data_state, [experiment], DEFAULT_TEMP_RES, [model_name], variables
            )
        )[0]
    
    for name, ds in ds_dict_evapo[experiment].items():
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
        ds_dict_evapo[experiment][name] = ds

    ds_dict_evapo[experiment] = remove_variables(ds_dict_evapo[experiment], ["evspsbl", "tran"])

    data_path = f"processed/{experiment}/month/"
    file_path = os.path.join(DATA_DIR, data_path)

    print(f"Saving files to: {file_path}{list(ds_dict_evapo[experiment].keys())[0]}")
     
    # Save the processed datasets and remove any existing files at the target path
    sd.save_files(ds_dict_evapo[experiment], file_path)
        
    return ds_dict_evapo

def remove_variables(ds_dict, variables_to_remove):
    """
    Removes specified variables from each dataset in the dictionary.

    Parameters:
    ds_dict (dict): Dictionary of xarray datasets.
    variables_to_remove (list): List of variable names to check and remove from the datasets.

    Returns:
    dict: Updated dictionary with specified variables removed from each dataset.
    """
    for name, ds in ds_dict.items():
        for var in variables_to_remove:
            if var in ds:
                del ds[var]  # Remove the variable if it exists in the dataset
                print(f"Variable '{var}' removed from dataset '{name}'")
            else: 
                print(f"Variable '{var}' not removed because not in dataset '{name}'")

    return ds_dict

import numpy as np
import xarray as xr

def convert_units_vpd(ds, var_name, target_units):
    """
    Convert units of ps and tas to required target units.
    - ps: Pa
    - tas: C (for Buck equation)
    """
    var = ds[var_name]
    current_units = (var.attrs.get("units", "") or "").strip()

    # normalize some common spellings
    current_units_norm = current_units.lower().replace(" ", "")

    if var_name == "ps" and target_units == "Pa":
        # common: hPa, hectopascal, mb
        if current_units_norm in ("hpa", "hectopascal", "hectopascals", "mb", "mbar"):
            var = var * 100.0
            var.attrs["units"] = "Pa"
        elif current_units_norm in ("pa",):
            var.attrs["units"] = "Pa"

    elif var_name == "tas":
        if target_units == "C":
            # Kelvin -> Celsius
            if current_units_norm in ("k", "kelvin"):
                var = var - 273.15
                var.attrs["units"] = "C"
            elif current_units_norm in ("c", "degc", "degreec", "degrees_c", "celsius"):
                var.attrs["units"] = "C"
        elif target_units == "K":
            if current_units_norm in ("c", "degc", "degreec", "degrees_c", "celsius"):
                var = var + 273.15
                var.attrs["units"] = "K"
            elif current_units_norm in ("k", "kelvin"):
                var.attrs["units"] = "K"

    return var

def compute_and_save_vpd(selected_model, experiment, member_id):
    """
    Compute VPD from tas, huss, ps for one model/member/experiment and save to processed/{experiment}/month/.
    """
    data_state = "processed"
    variables = ["tas", "huss", "ps"]
    model_name = f"{selected_model}_{member_id}"

    print(model_name)

    # ✅ skip if already saved
    exists, path = variable_already_saved(DATA_DIR, experiment, "month", model_name, "vpd")
    if exists:
        print(f"[SKIP] vpd already saved for {model_name} ({experiment}) in {path}")
        return None

    # Load datasets
    print("Loading datasets...")
    with ProgressBar():
        ds_dict_vpd = dask.compute(
            ld.load_multiple_models_and_experiments(
                DATA_DIR, data_state, [experiment], DEFAULT_TEMP_RES, [model_name], variables
            )
        )[0]

    # Compute for each dataset
    for name, ds in ds_dict_vpd[experiment].items():
        required_vars = ["huss", "ps", "tas"]
        missing = [v for v in required_vars if v not in ds]
        if missing:
            print(f"[WARN] VPD not computed for {name}, missing: {', '.join(missing)}")
            continue

        # Convert units
        ds["ps"] = convert_units_vpd(ds, "ps", "Pa")
        ds["tas"] = convert_units_vpd(ds, "tas", "C")

        T = ds["tas"]     # Celsius
        q = ds["huss"]    # kg/kg
        p = ds["ps"]      # Pa

        # Buck (1996) saturation vapor pressure over water (Pa)
        e_s = 611.21 * np.exp((18.678 - T / 234.5) * (T / (257.14 + T)))

        # Actual vapor pressure from specific humidity (Pa)
        e_a = (q * p) / (0.622 + 0.378 * q)

        vpd = e_s - e_a
        vpd = xr.where(vpd < 0, 0, vpd)

        # Convert to hPa
        vpd_hpa = vpd / 100.0

        vpd_hpa.attrs = {
            "standard_name": "vapor_pressure_deficit",
            "long_name": "Vapor Pressure Deficit",
            "units": "hPa",
            "comment": (
                "Computed as VPD = e_s(T) - e_a(q,p). "
                "e_s uses Buck (1996) equation (Pa), e_a derived from specific humidity and surface pressure (Pa). "
                "Negative values set to 0."
            ),
            "source_variables": "tas, huss, ps",
            "created_by": "Simon P. Heselschwerdt",
        }

        ds["vpd"] = vpd_hpa

        # Update dict
        ds_dict_vpd[experiment][name] = ds
        print(f"✓ VPD computed for {name}")

    # Optional: remove inputs (match evapo pattern)
    ds_dict_vpd[experiment] = remove_variables(ds_dict_vpd[experiment], ["tas", "huss", "ps"])

    # Save
    data_path = f"processed/{experiment}/month/"
    file_path = os.path.join(DATA_DIR, data_path)

    print(f"Saving files to: {file_path}{list(ds_dict_vpd[experiment].keys())[0]}")
    sd.save_files(ds_dict_vpd[experiment], file_path)

    return ds_dict_vpd

import re

def convert_units_wue(ds, var_name, target_units):
    """
    Convert tran and gpp to target units:
      - tran -> mm/day  (assuming 1 kg m-2 = 1 mm water)
      - gpp  -> gC/m2/day (common CMIP: kg m-2 s-1; treat as kgC m-2 s-1 if it's gpp)
    """
    var = ds[var_name]
    current_units = (var.attrs.get("units", "") or "").strip()
    u = current_units.lower().replace("²", "2").replace("−", "-").replace(" ", "")

    if var_name == "tran":
        # kg m-2 s-1 -> mm/day
        if u in ("kgm-2s-1", "kg/m2/s", "kgm-2s-1", "kgm-2s-1"):
            var = var * 86400.0
            var.attrs["units"] = "mm/day"
        elif u in ("mm/day", "mmd-1", "mmd-1"):
            var.attrs["units"] = "mm/day"

        # enforce
        var.attrs["units"] = target_units
        return var

    if var_name == "gpp":
        # kg m-2 s-1 -> gC m-2 day-1  (×1000 g/kg ×86400 s/day)
        if u in ("kgm-2s-1", "kg/m2/s", "kgm-2s-1"):
            var = var * 1000.0 * 86400.0
            var.attrs["units"] = "gC/m2/day"
        elif u in ("gc/m2/day", "gcm-2d-1", "gcm-2day-1"):
            var.attrs["units"] = "gC/m2/day"

        # enforce
        var.attrs["units"] = "gC/m2/day"
        return var

    return var

import os
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar

def compute_and_save_wue(selected_model, experiment, member_id, wue_threshold=10.0, epsilon=1e-6):
    """
    Compute WUE = gpp / (tran + eps) and save to processed/{experiment}/month/.
    """
    data_state = "processed"
    variables = ["tran", "gpp"]
    model_name = f"{selected_model}_{member_id}"

    print(model_name)

    # ✅ skip if already saved
    exists, path = variable_already_saved(DATA_DIR, experiment, "month", model_name, "wue")
    if exists:
        print(f"[SKIP] wue already saved for {model_name} ({experiment}) in {path}")
        return None

    # Load datasets
    print("Loading datasets...")
    with ProgressBar():
        ds_dict_wue = dask.compute(
            ld.load_multiple_models_and_experiments(
                DATA_DIR, data_state, [experiment], DEFAULT_TEMP_RES, [model_name], variables
            )
        )[0]

    # Compute
    for name, ds in ds_dict_wue[experiment].items():
        missing = [v for v in ("tran", "gpp") if v not in ds]
        if missing:
            print(f"[WARN] WUE not computed for {name}, missing: {', '.join(missing)}")
            continue

        ds["tran"] = convert_units_wue(ds, "tran", "mm/day")
        ds["gpp"]  = convert_units_wue(ds, "gpp",  "gC/m2/day")

        tran = ds["tran"]
        gpp  = ds["gpp"]

        wue = gpp / (tran + epsilon)

        # your thresholds
        wue = xr.where((wue > wue_threshold) | (wue < 0.1), np.nan, wue)

        wue.attrs = {
            "standard_name": "water_use_efficiency",
            "long_name": "Water Use Efficiency",
            "units": "gC/m2/mm",
            "comment": (
                f"WUE = gpp/(tran+{epsilon}). Input units converted to gC/m2/day and mm/day. "
                f"Values > {wue_threshold} or < 0.1 set to NaN."
            ),
            "source_variables": "gpp, tran",
            "created_by": "Simon P. Heselschwerdt",
        }

        ds["wue"] = wue
        ds_dict_wue[experiment][name] = ds
        print(f"✓ WUE computed for {name}")

    # Optional: remove inputs
    ds_dict_wue[experiment] = remove_variables(ds_dict_wue[experiment], ["tran", "gpp"])

    # Save
    out_dir = os.path.join(DATA_DIR, f"processed/{experiment}/month/")
    print(f"Saving files to: {out_dir}{list(ds_dict_wue[experiment].keys())[0]}")
    sd.save_files(ds_dict_wue[experiment], out_dir)

    return ds_dict_wue


import os
import dask
from dask.diagnostics import ProgressBar
import xarray as xr
import xclim

import gc
import dask
import xarray as xr
from dask.diagnostics import ProgressBar

import os
import gc
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


import os
import gc
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


def compute_and_save_rx5day_ratio(selected_model, experiment, member_id):
    DATA_DIR = "/work/ch0636/g300115/phd_project/common/data"
    model_name = f"{selected_model}_{member_id}"

    rx5_path = os.path.join(
        DATA_DIR, f"processed/{experiment}/year/RX5day/{model_name}.nc"
    )
    pr_path = os.path.join(
        DATA_DIR, f"processed/{experiment}/month/pr/{model_name}.nc"
    )
    out_dir = os.path.join(
        DATA_DIR, f"processed/{experiment}/year/rx5day_ratio"
    )
    out_path = os.path.join(out_dir, f"{model_name}.nc")

    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        print(f"[SKIP] rx5day_ratio already exists for {model_name} ({experiment})")
        return

    if not os.path.exists(rx5_path):
        print(f"[MISSING] RX5day file not found: {rx5_path}")
        return

    if not os.path.exists(pr_path):
        print(f"[MISSING] monthly pr file not found: {pr_path}")
        return

    print(f"Loading RX5day and monthly pr for {model_name} ({experiment})")

    with xr.open_dataset(rx5_path) as ds_rx5, xr.open_dataset(pr_path) as ds_pr:
        if "RX5day" not in ds_rx5:
            raise ValueError(f"'RX5day' not found in {rx5_path}")
        if "pr" not in ds_pr:
            raise ValueError(f"'pr' not found in {pr_path}")

        rx5 = ds_rx5["RX5day"]
        pr = ds_pr["pr"]

        # Convert monthly mean pr to daily mean precipitation
        pr_daily = pr

        # Simple annual mean daily precipitation from the 12 monthly means
        annual_mean_pr = pr_daily.resample(time="YS").mean(skipna=True)
        annual_mean_pr.name = "annual_mean_pr"

        # Annual precipitation = annual mean daily precipitation * 365
        annual_pr = annual_mean_pr * 365.0
        annual_pr.name = "annual_pr"
        annual_pr.attrs = pr_daily.attrs.copy()
        annual_pr.attrs["long_name"] = "Annual precipitation from simple annual mean daily precipitation * 365"
        annual_pr.attrs["units"] = "mm"

        # Align years with RX5day
        #rx5, annual_pr = xr.align(rx5, annual_pr, join="inner")

        # Compute ratio
        ratio = xr.where(annual_pr > 0, rx5 / annual_pr, np.nan)
        ratio.name = "rx5day_ratio"
        ratio.attrs = {
            "long_name": "RX5day divided by annual precipitation",
            "description": "Annual precipitation computed as simple annual mean daily precipitation from monthly means multiplied by 365",
            "units": "1",
            "source_rx5day_file": rx5_path,
            "source_pr_file": pr_path,
        }

        out = xr.Dataset({"rx5day_ratio": ratio})        
        out = out.assign_coords(time=ratio.time)
        out.attrs = ds_rx5.attrs.copy()

        print(f"Saving: {out_path}")
        with ProgressBar():
            out.compute().to_netcdf(out_path)

        out.close()
        del out, ratio, annual_pr, annual_mean_pr, pr_daily, pr, rx5

    gc.collect()
    print(f"Finished rx5day_ratio for {model_name} ({experiment})")

def compute_and_save_rx_indices(selected_model, experiment, member_id, freq="YS"):

    data_state = "processed"
    variables = ["pr"]
    model_name = f"{selected_model}_{member_id}"
    temp_res_toggle = "year"

    # Skip if both exist
    ex1, _ = variable_already_saved(DATA_DIR, experiment, temp_res_toggle, model_name, "RX1day")
    ex5, _ = variable_already_saved(DATA_DIR, experiment, temp_res_toggle, model_name, "RX5day")

    #ex1=False
    #ex5=False
    if ex1 and ex5:
        print(f"[SKIP] RX1day and RX5day already saved for {model_name} ({experiment})")
        return

    print("Loading daily pr...")

    with ProgressBar():
        ds_dict = dask.compute(
            ld.load_multiple_models_and_experiments(
                DATA_DIR, data_state, [experiment], "day", [model_name], variables
            )
        )[0]

    base = os.path.join(DATA_DIR, f"processed/{experiment}/{temp_res_toggle}/")

    # Process each dataset individually (important for memory)
    for name, ds in ds_dict[experiment].items():

        if "pr" not in ds:
            continue

        pr = ds["pr"]

        # ---------- RX1day ----------
        if not ex1:

            print(f"Computing RX1day for {name}")

            rx1 = xclim.indicators.icclim.RX1day(pr=pr, freq=freq)

            out1 = xr.Dataset({"RX1day": rx1}).assign_coords(time=rx1.time)
            out1.attrs = ds.attrs.copy()

            # FORCE COMPUTE + SAVE
            with ProgressBar():
                out1.compute().to_netcdf(
                    os.path.join(base, f"RX1day/{name}.nc")
                )

            # Explicit close + delete
            out1.close()
            del out1, rx1

        # ---------- RX5day ----------
        if not ex5:

            print(f"Computing RX5day for {name}")

            rx5 = xclim.indicators.icclim.RX5day(pr=pr, freq=freq)

            out5 = xr.Dataset({"RX5day": rx5}).assign_coords(time=rx5.time)
            out5.attrs = ds.attrs.copy()

            with ProgressBar():
                out5.compute().to_netcdf(
                    os.path.join(base, f"RX5day/{name}.nc")
                )

            out5.close()
            del out5, rx5

        # ----- CLOSE INPUT -----
        ds.close()
        del ds, pr

        # VERY IMPORTANT
        gc.collect()

    # Remove master dict
    del ds_dict
    gc.collect()

    print("Finished RX index computation")



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
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

def preprocess_data_and_save(ds_dict, model, member_id, selected_scenario, freq, regrid_to = 1, landmask=True, apply_only_ant_mask=False, filepath=None):
    """
    Run the preprocessing chain used in the notebook and save outputs.
    
    This function wraps the existing regridding/masking/unit logic and writes per-member files.
    Implementation is unchanged; only docstring was added.
    """
    ds_dict = drop_redundant(ds_dict, droplist=None)
    ds_dict = pro_dat.regrid(ds_dict, lon_lat_degree=regrid_to, method='conservative')
    # The reference file has to exist before you run this code meaning that this is the first time you are loading this data at least the reference file (in
    # my case always BCC pr) has to be saved before.
    if model != "BCC-CSM2-MR":
        ds_dict = consistent_time_coordinate(ds_dict, member_id, selected_scenario, freq,  reference_file_path=f'processed/{selected_scenario}/{freq}/pr/BCC-CSM2-MR_r1i1p1f1.nc')

    # Define variable and conversion unit
    conv_units = {'pr': 'mm/day',
                  'gpp': 'gC/m²/day', 
                  'mrro': 'mm/day',
                  'tran': 'mm/day',
                  'lai': '',
                  'mrsos': 'mm',
                  'mrso': 'mm',
                  'evspsbl': 'mm/day',
                  'tas': '°C'
                  
                }
    ds_dict = pro_dat.set_units(ds_dict, conv_units)
        
    if landmask:
        ds_dict = pro_dat.apply_landmask(
            ds_dict,
            filename="land_sea_mask_1x1_grid.nc",
            savepath="/work/ch0636/g300115/phd_project/common/data/landmasks/imerg/",
        )
    else:
        key = list(ds_dict.keys())[0]
        ds = ds_dict[key]
        rename_map = {var: f"{var}_no_landmask" for var in ds.data_vars}
        ds_dict[key] = ds.rename(rename_map)
    
    if apply_only_ant_mask:
        ds_dict = pro_dat.remove_antarctica(ds_dict)
    else:
        ds_dict = pro_dat.remove_antarctica_greenland_iceland(ds_dict)

    if filepath is not None:
        sd.save_files_with_member_id(ds_dict, filepath, member_id)
    
    return ds_dict

def drop_redundant(ds_dict, droplist=None):
    """
    Drop redundant coordinates/variables from a dataset (unchanged implementation).
    """
    # Step 4.1: Define redundant coordinates and variables
    if droplist is None:
        drop_list = [
            "member_id", "type", "nbnd", "bnds", "height", "depth", "lat_bnds",
            "lon_bnds", "time_bnds", "time_bounds", "depth_bnds", "sdepth_bounds",
            "depth_bounds", "hist_interval", "axis_nbounds", "dcpp_init_year", "areacella", "sftlf"
        ]
    else:
        drop_list = droplist
    
    # Drop the defined coordinates and variables
    ds_dict = drop_redundant_vars(ds_dict, drop_list)
    
    # Step 4.2: Merge datasets with different `table_id` but the same `source_id`
    ds_dict = merge_source_id_data(ds_dict)

    return ds_dict

def drop_redundant_vars(ds_dict, drop_list): 
    """
    Remove redundant coordinates and variables from datasets in a dictionary.

    Parameters:
    ds_dict (dict): Dictionary containing dataset names as keys and xarray.Dataset objects as values.
    drop_list (list): List of redundant coordinate or variable names to be removed from the datasets.

    Returns:
    dict: Dictionary with the same keys as the input ds_dict and modified xarray.Dataset objects with redundant elements removed.
    """
    for ds_name, ds_data in ds_dict.items():
        
        if 'sdepth' in ds_data.coords:
            if 'depth' in ds_data.coords:
                ds_data = ds_data.drop('depth')
            if 'depth' in ds_data.dims:
                ds_data = ds_data.drop_dims('depth')
            ds_data = ds_data.rename({'sdepth': 'depth'})
            print(f'sdepth changed to depth for model {ds_data.source_id}')
            # Add comment about changes to data 
            if 'log' in ds_data.attrs:
                log_old = ds_data.attrs['log']
                ds_data.attrs['log'] = f'Coordinate name changed from sdepth to depth. // {log_old}'
            else:
                ds_data.attrs['log'] = 'Coordinate name changed from sdepth to depth.'
            
        if 'solth' in ds_data.coords:
            if 'depth' in ds_data.coords:
                ds_data = ds_data.drop('depth')
            if 'depth' in ds_data.dims:
                ds_data = ds_data.drop_dims('depth')
            ds_data = ds_data.rename({'solth': 'depth'})
            print(f'solth changed to depth for model {ds_data.source_id}')
            # Add comment about changes to data 
            if 'log' in ds_data.attrs:
                log_old = ds_data.attrs['log']
                ds_data.attrs['log'] = f'Coordinate name changed from solth to depth. // {log_old}'
            else:
                ds_data.attrs['log'] = 'Coordinate name changed from solth to depth.'
   
        
        if 'mrsol' in ds_data and 'depth' in drop_list or 'tsl' in ds_data and 'depth' in drop_list:
            drop_list.remove('depth')
                      
        for coord in drop_list:
            if coord in ds_data.coords:
                ds_data = ds_data.drop(coord).squeeze()
                print(f'Dropped coordinate: {coord}')
                # Add comment about changes to data 
                if 'log' in ds_data.attrs:
                    log_old = ds_data.attrs['log']
                    ds_data.attrs['log'] = f'Dropped: {coord}. // {log_old}'
                else:
                    ds_data.attrs['log'] = f'Dropped: {coord}.'
            if coord in ds_data.variables:
                ds_data = ds_data.drop_vars(coord).squeeze()
                print(f'Dropped variable: {coord}')
                # Add comment about changes to data 
                if 'log' in ds_data.attrs:
                    log_old = ds_data.attrs['log']
                    ds_data.attrs['log'] = f'Dropped: {coord}. // {log_old}'
                else:
                    ds_data.attrs['log'] = f'Dropped: {coord}.'
            
        # Check if the coords were dropped successfully and use squeeze if their length is 1
        for coord in drop_list:
            if coord in ds_data.dims:
                print(f"Coordinate {coord} was not dropped.")
                if ds_data.dims[coord] == 1:
                    ds_data = ds_data.squeeze(coord, drop=True)
                    print(f"Squeezed coordinate: {coord}")
                    # Add comment about changes to data 
                    if 'log' in ds_data.attrs:
                        log_old = ds_data.attrs['log']
                        ds_data.attrs['log'] = f'Dropped: {coord}. // {log_old}'
                    else:
                        ds_data.attrs['log'] = f'Dropped: {coord}.'
            
        # Update the dictionary with the modified dataset
        ds_dict[ds_name] = ds_data
    
    return ds_dict

def merge_source_id_data(ds_dict):
    """
    Merge datasets with the same source_id (name of the CMIP6 model) as CMIP6 data is stored in different table id's. This function is mainly used to merge two 
    different xarray datasets for 'table_id' Amon and Lmon into a single xarray dataset as this makes future investigations easier. Other table_id's
    can also be merged; however, be careful when the same variable exists in both datasets.

    Args:
        ds_dict (dict): A dictionary of xarray datasets, where each key is the name of the dataset 
                        and each value is the dataset itself.

    Returns:
        dict: A merged dictionary with a single dataset for each CMIP6 model/source_id.
    """
    merged_dict = {}
    for dataset_name, dataset in ds_dict.items():
        source_id = dataset.attrs['source_id']
        table_id = dataset.attrs['table_id']
        print(f"Merging dataset '{dataset_name}' with source_id '{source_id}' and table_id '{table_id}'...")
       
        if source_id in merged_dict:
            if source_id == merged_dict[source_id].attrs['source_id'] and table_id != merged_dict[source_id].attrs['table_id']:
                merg_model_name = merged_dict[source_id].attrs['intake_esm_dataset_key']
                merg_model_table_id = merged_dict[source_id].attrs['table_id']
                 
                # Replace coordinates lat, lon, time of dataset only when different to datasets in merged_dict
                dataset = replace_coordinates(merged_dict[source_id], dataset)

                # Merge data    
                with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                    merged_dict[source_id] = xr.merge([merged_dict[source_id], dataset])

                if len(list(merged_dict.keys())) == 1:
                    print(f"Datasets '{merg_model_name}' ('{merg_model_table_id}') and '{dataset_name}' ('{table_id}') are merged to 'ds_dict' with key '{source_id}'.")
                else:
                    print(f"Datasets '{dataset_name}' ('{table_id}') is merged with 'ds_dict'.")

        else:
            merged_dict[source_id] = dataset
            print(f"Dataset '{dataset_name}' ('{table_id}') is saved in 'ds_dict'.")

    return merged_dict

import pandas as pd
import cftime
import xarray as xr

def _is_cftime_array(values) -> bool:
    return len(values) > 0 and isinstance(values[0], cftime.datetime)

def _make_start_end_for_ds(ds: xr.Dataset, start_str: str, end_str: str):
    tvals = ds["time"].values

    # parse user dates
    s = pd.Timestamp(start_str)
    e = pd.Timestamp(end_str)

    if _is_cftime_array(tvals):
        t0 = tvals[0]
        CF = type(t0)
        has_yz = getattr(t0, "has_year_zero", False)

        # keep dataset time-of-day (common in CMIP: 12:00)
        hh = getattr(t0, "hour", 0)
        mm = getattr(t0, "minute", 0)
        ss = getattr(t0, "second", 0)

        start = CF(s.year, s.month, s.day, hh, mm, ss, has_year_zero=has_yz)

        # end-of-day so we never exclude the last day
        end   = CF(e.year, e.month, e.day, 23, 59, 59, has_year_zero=has_yz)
        return start, end

    else:
        start = pd.Timestamp(start_str)
        # inclusive end-of-day
        end = pd.Timestamp(end_str) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return start, end

def slice_time_to_target(
    ds: xr.Dataset,
    expected_n: int | None = None,
    start_str: str = "2015-01-01",
    end_str: str = "2100-12-31",
    label: str = "",
):
    """
    - Prints current time info (dtype/type, start/end, ntime)
    - If expected_n is provided and ds has more timesteps than expected_n, slices by [start_str, end_str]
    - Prints after slicing info + mismatch warnings
    """
    if "time" not in ds.coords and "time" not in ds.dims:
        print(f"[{label}] No 'time' coordinate/dimension found. Skipping.")
        return ds

    time = ds["time"]
    n0 = ds.sizes.get("time", None)

    # Determine "type" for display
    tvals = time.values
    if _is_cftime_array(tvals):
        time_kind = f"cftime ({type(tvals[0]).__name__})"
        t0, t1 = tvals[0], tvals[-1]
    else:
        time_kind = f"numpy datetime64 ({tvals.dtype})"
        t0, t1 = pd.Timestamp(tvals[0]).to_pydatetime(), pd.Timestamp(tvals[-1]).to_pydatetime()

    print(f"[{label}] BEFORE: ntime={n0}, time_kind={time_kind}, start={t0}, end={t1}")

    # Decide whether we need to slice
    need_slice = False
    if expected_n is not None and n0 is not None:
        if n0 != expected_n:
            print(f"[{label}] MISMATCH: ntime={n0} vs expected={expected_n}")
        # You said “if it extends my timestamps” — typically means it's longer than expected
        if n0 > expected_n:
            need_slice = True
    else:
        # fallback: slice if end exceeds target end date
        # (works when expected_n isn't known)
        target_end = pd.Timestamp(end_str)
        if not _is_cftime_array(tvals):
            if pd.Timestamp(tvals[-1]) > target_end:
                need_slice = True
        else:
            # for cftime we just attempt slice; safe enough
            need_slice = True

    if not need_slice:
        print(f"[{label}] No slicing applied.")
        return ds

    # Build matching start/end objects and slice
    start, end = _make_start_end_for_ds(ds, start_str, end_str)
    ds2 = ds.sel(time=slice(start, end))

    n1 = ds2.sizes.get("time", None)
    tvals2 = ds2["time"].values
    if _is_cftime_array(tvals2):
        t0b, t1b = tvals2[0], tvals2[-1]
        time_kind2 = f"cftime ({type(tvals2[0]).__name__})"
    else:
        t0b, t1b = pd.Timestamp(tvals2[0]).to_pydatetime(), pd.Timestamp(tvals2[-1]).to_pydatetime()
        time_kind2 = f"numpy datetime64 ({tvals2.dtype})"

    print(f"[{label}] AFTER : ntime={n1}, time_kind={time_kind2}, start={t0b}, end={t1b}")

    if expected_n is not None and n1 != expected_n:
        print(f"[{label}] STILL MISMATCH after slicing: ntime={n1} vs expected={expected_n}")

    return ds2
    

import numpy as np
import pandas as pd
import xarray as xr
import cftime
import calendar as pycal

def _is_cftime_time(ds: xr.Dataset) -> bool:
    vals = ds["time"].values
    return len(vals) > 0 and isinstance(vals[0], cftime.datetime)

def _days_in_month(year: int, month: int, calendar_name: str) -> int:
    # Minimal support for common CMIP calendars
    if calendar_name in ("360_day", "360"):
        return 30
    if calendar_name in ("noleap", "365_day"):
        # Feb always 28
        if month == 2:
            return 28
        return pycal.monthrange(year, month)[1]
    # default: proleptic_gregorian/gregorian/standard
    return pycal.monthrange(year, month)[1]

def _calendar_name(ds: xr.Dataset) -> str:
    # try to read CF calendar attribute
    cal = ds["time"].encoding.get("calendar", None) or ds["time"].attrs.get("calendar", None)
    return cal or "standard"

def _make_like_template(ds: xr.Dataset, year: int, month: int):
    """Create one timestamp matching ds.time type AND the day/hour/min/sec pattern of ds.time[0]."""
    t0 = ds["time"].values[0]
    cal = _calendar_name(ds)

    if isinstance(t0, cftime.datetime):
        CF = type(t0)
        day = min(getattr(t0, "day", 1), _days_in_month(year, month, cal))
        return CF(
            year, month, day,
            getattr(t0, "hour", 0),
            getattr(t0, "minute", 0),
            getattr(t0, "second", 0),
            has_year_zero=getattr(t0, "has_year_zero", False),
        )
    else:
        t0 = pd.Timestamp(t0)
        day = min(t0.day, pycal.monthrange(year, month)[1])
        return pd.Timestamp(year=year, month=month, day=day,
                            hour=t0.hour, minute=t0.minute, second=t0.second, nanosecond=t0.nanosecond)

def _month_list(start_ym: tuple[int,int], end_ym: tuple[int,int]) -> list[tuple[int,int]]:
    """Inclusive list of (year, month) from start to end."""
    sy, sm = start_ym
    ey, em = end_ym
    out = []
    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        out.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out

def pad_monthly_missing_with_nans(
    ds: xr.Dataset,
    target_start: str,
    target_end: str,
    label: str = ""
) -> xr.Dataset:
    """
    Pads missing MONTHS at the beginning and/or end of ds.time to cover [target_start, target_end],
    creating NaN maps for the missing months only.
    """
    if "time" not in ds.dims:
        print(f"[{label}] No time dimension; skipping padding.")
        return ds

    # Convert target bounds to year/month
    ts = pd.Timestamp(target_start)
    te = pd.Timestamp(target_end)
    target_start_ym = (ts.year, ts.month)
    target_end_ym   = (te.year, te.month)

    # Get ds start/end year/month (works for both datetime64 and cftime)
    t0 = ds["time"].values[0]
    t1 = ds["time"].values[-1]
    if isinstance(t0, cftime.datetime):
        ds_start_ym = (t0.year, t0.month)
        ds_end_ym   = (t1.year, t1.month)
    else:
        t0p = pd.Timestamp(t0)
        t1p = pd.Timestamp(t1)
        ds_start_ym = (t0p.year, t0p.month)
        ds_end_ym   = (t1p.year, t1p.month)

    # Determine missing head months (target_start .. month before ds_start)
    missing_head = []
    if ds_start_ym > target_start_ym:
        head_end = (ds_start_ym[0], ds_start_ym[1] - 1)
        if head_end[1] == 0:
            head_end = (head_end[0] - 1, 12)
        missing_head = _month_list(target_start_ym, head_end)

    # Determine missing tail months (month after ds_end .. target_end)
    missing_tail = []
    if ds_end_ym < target_end_ym:
        tail_start = (ds_end_ym[0], ds_end_ym[1] + 1)
        if tail_start[1] == 13:
            tail_start = (tail_start[0] + 1, 1)
        missing_tail = _month_list(tail_start, target_end_ym)

    if not missing_head and not missing_tail:
        print(f"[{label}] No missing months relative to target window. No padding.")
        return ds

    # Build missing time coordinates matching ds timestamp "style"
    head_times = [_make_like_template(ds, y, m) for (y, m) in missing_head]
    tail_times = [_make_like_template(ds, y, m) for (y, m) in missing_tail]

    # Build NaN fillers using ds structure
    fillers = []
    if head_times:
        filler_head = ds.isel(time=slice(0, 0)).reindex(time=head_times)
        fillers.append(filler_head)
    fillers.append(ds)
    if tail_times:
        filler_tail = ds.isel(time=slice(0, 0)).reindex(time=tail_times)
        fillers.append(filler_tail)

    out = xr.concat(fillers, dim="time")

    # Document what happened
    out.attrs["time_alignment_note"] = (
        f"{label}: padded missing monthly timesteps with NaNs. "
        f"missing_head={len(head_times)} ({missing_head[0] if missing_head else None}..{missing_head[-1] if missing_head else None}), "
        f"missing_tail={len(tail_times)} ({missing_tail[0] if missing_tail else None}..{missing_tail[-1] if missing_tail else None})."
    )

    print(
        f"[{label}] Padded: +{len(head_times)} months at start, +{len(tail_times)} months at end. "
        f"Final ntime={out.sizes['time']}."
    )
    return out

import numpy as np
import pandas as pd
import xarray as xr
import cftime

def pad_daily_missing_with_nans(ds: xr.Dataset, target_start: str, target_end: str, label: str = ""):
    if "time" not in ds.dims:
        print(f"[{label}] No time dimension; skipping padding.")
        return ds

    t0 = ds["time"].values[0]
    t1 = ds["time"].values[-1]

    # dataset time-of-day pattern (e.g. 12:00)
    if isinstance(t0, cftime.datetime):
        CF = type(t0)
        cal = ds["time"].encoding.get("calendar") or ds["time"].attrs.get("calendar") or "standard"
        has_yz = getattr(t0, "has_year_zero", False)
        hh, mm, ss = t0.hour, t0.minute, t0.second

        # full daily axis (midnight) in the dataset calendar
        full = xr.cftime_range(target_start, target_end, freq="D", calendar=cal)

        # compare by date components
        ds_start = (t0.year, t0.month, t0.day)
        ds_end   = (t1.year, t1.month, t1.day)

        def ymd(x): return (x.year, x.month, x.day)

        head_days = [d for d in full if ymd(d) < ds_start]
        tail_days = [d for d in full if ymd(d) > ds_end]

        # convert to dataset-style timestamps (same hour/min/sec)
        head_times = [CF(d.year, d.month, d.day, hh, mm, ss, has_year_zero=has_yz) for d in head_days]
        tail_times = [CF(d.year, d.month, d.day, hh, mm, ss, has_year_zero=has_yz) for d in tail_days]

    else:
        t0p = pd.Timestamp(t0)
        t1p = pd.Timestamp(t1)
        offset = pd.Timedelta(hours=t0p.hour, minutes=t0p.minute, seconds=t0p.second, nanoseconds=t0p.nanosecond)

        full = pd.date_range(target_start, target_end, freq="D")

        head_days = full[full.date < t0p.date()]
        tail_days = full[full.date > t1p.date()]

        head_times = (head_days + offset).to_numpy()
        tail_times = (tail_days + offset).to_numpy()

    if not head_times and not tail_times:
        print(f"[{label}] No missing days relative to target window. No padding.")
        return ds

    fillers = []
    if head_times:
        fillers.append(ds.isel(time=slice(0, 0)).reindex(time=head_times))
    fillers.append(ds)
    if tail_times:
        fillers.append(ds.isel(time=slice(0, 0)).reindex(time=tail_times))

    out = xr.concat(fillers, dim="time")
    out.attrs["time_alignment_note"] = (
        f"{label}: padded missing DAILY timesteps with NaNs. "
        f"missing_head={len(head_times)}, missing_tail={len(tail_times)}."
    )

    print(f"[{label}] Padded daily: +{len(head_times)} start, +{len(tail_times)} end. Final ntime={out.sizes['time']}.")
    return out

def drop_feb29_if_needed(ds, expected_n=None, label=""):
    """
    Drop Feb 29 if it explains the mismatch (Gregorian/standard vs noleap reference).
    Only drops if removing Feb29 makes ds.time length match expected_n.
    """
    if "time" not in ds.coords:
        return ds

    feb29 = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
    n_feb29 = int(feb29.sum().item()) if hasattr(feb29.sum(), "item") else int(feb29.sum())

    if n_feb29 == 0:
        return ds

    if expected_n is not None:
        n0 = ds.sizes["time"]
        if (n0 - n_feb29) == expected_n:
            ds2 = ds.sel(time=~feb29)
            print(f"[{label}] Dropped {n_feb29} leap days (Feb 29) to match expected_n={expected_n}.")
            return ds2

    return ds

import pandas as pd
import cftime
import calendar as pycal
import xarray as xr

def _is_cftime_array(values) -> bool:
    return len(values) > 0 and isinstance(values[0], cftime.datetime)

def _infer_calendar(ds: xr.Dataset) -> str:
    # Prefer explicit CF metadata
    cal = ds["time"].encoding.get("calendar") or ds["time"].attrs.get("calendar")
    if cal:
        return cal

    # Fall back to cftime type
    t0 = ds["time"].values[0]
    if isinstance(t0, cftime.Datetime360Day):   return "360_day"
    if isinstance(t0, cftime.DatetimeNoLeap):   return "noleap"
    if isinstance(t0, cftime.DatetimeAllLeap):  return "all_leap"
    if isinstance(t0, cftime.DatetimeJulian):   return "julian"
    if isinstance(t0, cftime.DatetimeGregorian):return "gregorian"
    return "standard"

def _max_day_in_month(year: int, month: int, cal: str) -> int:
    cal = (cal or "").lower()
    if cal in ("360_day", "360"):
        return 30
    if cal in ("noleap", "365_day"):
        return 28 if month == 2 else pycal.monthrange(year, month)[1]
    if cal in ("all_leap", "366_day"):
        return 29 if month == 2 else pycal.monthrange(year, month)[1]
    # standard/gregorian/proleptic_gregorian/julian -> use Gregorian month lengths
    return pycal.monthrange(year, month)[1]

def _make_start_end_for_ds(ds: xr.Dataset, start_str: str, end_str: str):
    tvals = ds["time"].values
    s = pd.Timestamp(start_str)
    e = pd.Timestamp(end_str)

    if _is_cftime_array(tvals):
        t0 = tvals[0]
        CF = type(t0)
        has_yz = getattr(t0, "has_year_zero", False)

        cal = _infer_calendar(ds)

        # keep dataset time-of-day (often 12:00 in CMIP daily)
        hh = getattr(t0, "hour", 0)
        mm = getattr(t0, "minute", 0)
        ss = getattr(t0, "second", 0)

        # clamp days so they are valid for this calendar
        s_day = min(s.day, _max_day_in_month(s.year, s.month, cal))
        e_day = min(e.day, _max_day_in_month(e.year, e.month, cal))

        start = CF(s.year, s.month, s_day, hh, mm, ss, has_year_zero=has_yz)
        end   = CF(e.year, e.month, e_day, 23, 59, 59, has_year_zero=has_yz)
        return start, end

    else:
        start = pd.Timestamp(start_str)
        end   = pd.Timestamp(end_str) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return start, end
import os
import xarray as xr

def consistent_time_coordinate(ds_dict, member_id, selected_scenario, freq, reference_file_path=None):
    """
    Make time consistent.
    - For daily: slice to target window ONLY, do NOT align to a noleap reference.
    - For monthly: slice/pad against reference and then consis_time (if you still want that).
    """
    model_key = list(ds_dict.keys())[0]
    ds = ds_dict[model_key]
    label = f"{model_key} {member_id} {selected_scenario}"

    # target windows
    if selected_scenario in ("historical", "hist-noLu"):
        win_start, win_end = "1850-01-01", "2014-12-31"
    else:
        win_start, win_end = "2015-01-01", "2100-12-31"

    # Always do calendar-safe slicing to your desired window
    ds = slice_time_to_target(ds, start_str=win_start, end_str=win_end, label=label)

    # -------------------------
    # DAILY: stop here (native calendar)
    # -------------------------
    if freq == "day":
        # Do NOT pad to ref, do NOT call consis_time, do NOT drop Feb29.
        # Just warn if coverage is incomplete.
        start_obj, end_obj = _make_start_end_for_ds(ds, win_start, win_end)
        t0 = ds.time.values[0]
        t1 = ds.time.values[-1]

        if t0 > start_obj or t1 < end_obj:
            ds.attrs["time_coverage_warning"] = (
                f"{label}: daily dataset does not fully cover target window "
                f"{win_start}..{win_end}. start={t0}, end={t1}."
            )
            print(f"[{label}] WARNING: incomplete daily coverage. (See ds.attrs['time_coverage_warning'])")

        ds_dict[model_key] = ds
        return ds_dict

    # -------------------------
    # MONTHLY: keep your ref-based padding/alignment
    # -------------------------
    if reference_file_path is None:
        raise ValueError("reference_file_path must be provided for monthly alignment.")

    file_path = os.path.join(DATA_DIR, reference_file_path)
    print(f"Time coordinate reference file loaded from: {file_path}")
    ref_ds = xr.open_dataset(file_path)

    n_ds = ds.sizes.get("time")
    n_ref = ref_ds.sizes.get("time")
    expected = n_ref

    if n_ds is not None and n_ref is not None:
        if n_ds > n_ref:
            # slice again (should be no-op usually, but safe)
            ds = slice_time_to_target(ds, start_str=win_start, end_str=win_end, label=label)
        elif n_ds < n_ref:
            print(f"[{label}] ds shorter than ref ({n_ds} < {n_ref}). Padding with NaNs (monthly).")
            ds = pad_monthly_missing_with_nans(ds, target_start=win_start, target_end=win_end, label=label)

    ds_dict[model_key] = ds

    # IMPORTANT: only do this if consis_time is truly safe for your monthly workflow
    ds_dict = pro_dat.consis_time(ds_dict, ref_ds)

    return ds_dict


def test():
    """def consistent_time_coordinate(ds_dict, member_id, selected_scenario, freq, reference_file_path):

    Make the time coordinate consistent across datasets in a dict.
    
    # Step 2.1: Define reference time coordinate and load it to ref_ds
    file = reference_file_path
    file_path = os.path.join(DATA_DIR, file)
    print(f"Time coordinate reference file loaded from: {file_path}")
    ref_ds = xr.open_dataset(file_path)

    model_key = list(ds_dict.keys())[0]         
    ds = ds_dict[model_key]
    
    expected = ref_ds.sizes.get("time")
     
    n_ds = ds.sizes.get("time")
    n_ref = ref_ds.sizes.get("time")


    if selected_scenario not in ("historical", "hist-noLu"):
        # future window
        if n_ds > n_ref:
            ds = slice_time_to_target(
                ds,
                expected_n=expected,
                start_str="2015-01-01",
                end_str="2100-12-31",
                label=f"{model_key} {member_id} {selected_scenario}",
            )
        elif n_ds < n_ref:
            print(f"[{model_key} {member_id} {selected_scenario}] ds shorter than ref ({n_ds} < {n_ref}). Padding with NaNs.")
            if freq == "month":
                ds = pad_monthly_missing_with_nans(
                    ds,
                    target_start="2015-01-01",
                    target_end="2100-12-31",
                    label=f"{model_key} {member_id} {selected_scenario}",
                )
            elif freq == "day":
                ds = pad_daily_missing_with_nans(
                    ds,
                    target_start="2015-01-01",
                    target_end="2100-12-31",
                    label=f"{model_key} {member_id} {selected_scenario}",
                )
           
         
    else:
        # historical window
        if n_ds > n_ref:
            ds = slice_time_to_target(
                ds,
                expected_n=expected,
                start_str="1850-01-01",
                end_str="2014-12-31",
                label=f"{model_key} {member_id} {selected_scenario}",
            )
        elif n_ds < n_ref:
            print(f"[{model_key} {member_id} {selected_scenario}] ds shorter than ref ({n_ds} < {n_ref}). Padding with NaNs.")
            if freq == "month":
                ds = pad_monthly_missing_with_nans(
                    ds,
                    target_start="1850-01-01",
                    target_end="2014-12-31",
                    label=f"{model_key} {member_id} {selected_scenario}",
                )
            elif freq == "day":
                ds = pad_daily_missing_with_nans(
                    ds,
                    target_start="1850-01-01",
                    target_end="2014-12-31",
                    label=f"{model_key} {member_id} {selected_scenario}",
                )

    ds_dict[model_key] = ds

    # Step 2.2: Define reference time coordinate
    ds_dict = pro_dat.consis_time(ds_dict, ref_ds)

    return ds_dict
    """

def load_and_get_ensemble(DATA_DIR, data_state, experiments, DEFAULT_TEMP_RES, model_name, models, variables, ext=None):
    """
    Load multiple models/experiments, compute ensemble statistics, and save.
    """
    # Step 1.1: Load the datasets
    print("Loading datasets...")
    with ProgressBar():
        ds_dict_ = dask.compute(
            ld.load_multiple_models_and_experiments(
                DATA_DIR, data_state, experiments, DEFAULT_TEMP_RES, models, variables
            )
        )[0]

    # Compute ensemble mean
    ds_dict_output = {}
    for scenario in ds_dict_.keys():
        ds_dict_[scenario] = comp_stats.compute_ensemble_statistic(ds_dict_[scenario], 'mean')
        ds_dict_output[scenario] = {}
        ds_dict_output[scenario][model_name] = ds_dict_[scenario]['Ensemble mean']

    for experiment in experiments:
        print(experiment) 
        # Construct the output file path
        data_path = f"processed/{experiment}/{DEFAULT_TEMP_RES}/"
        file_path = os.path.join(DATA_DIR, data_path)
        print(f"Saving files to: {file_path}")
        # Save the processed datasets and remove any existing files at the target path
        if ext is not None:
            sd.save_files_with_member_id(ds_dict_output[experiment], file_path, f'ensmean_{ext}')
        else:
            sd.save_files_with_member_id(ds_dict_output[experiment], file_path, 'ensmean')

    print(f"{model_name} ensemble mean across {ext} members for {experiments[0]} {variables[0]} time step 0:")
    ds_dict_output[experiments[0]][model_name][variables[0]].isel(time=0).plot()
    plt.show()
    plt.close()
    
    return

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

    if specific_months_or_seasons:
        print(f'Selected period {start_year} to {end_year} for {time_selection_name}')
    else:
        print(f'Selected period {start_year} to {end_year}')

    return ds_dict_copy

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

def compute_bgws(ds_dict):
    """
    Computes the Blue Green Water Share (BGWS) for the given datasets.

    Parameters:
    ds_dict (dict): A dictionary of xarray datasets, potentially nested with experiments and models.

    Returns:
    dict: The input dictionary with the computed BGWS added, excluding any ensemble data.
    """
    def compute_bgws_for_ds(ds):
        bgws = ((ds['mrro_mean'] - ds['tran_mean']) / ds['pr_mean']) * 100
        
        # Replace infinite values with NaN
        bgws = xr.where(np.isinf(bgws), float('nan'), bgws)
        
        # Set all values above 100 and below -100 to 100/-100 as not more than 100% of incoming precipitation can be partitioned towards runoff/transpiration
        bgws = xr.where(bgws > 100, 100, bgws)
        bgws = xr.where(bgws < -100, -100, bgws)
        
        ds['bgws_mean'] = bgws
        ds['bgws_mean'].attrs = {'long_name': 'Blue-Green Water Share', 'units': '%'}
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

import numpy as np
import xarray as xr

def _safe_ratio(numer: xr.DataArray, denom: xr.DataArray, eps=1e-12):
    """
    Compute numer/denom safely:
    - avoids division by 0
    - returns NaN where denom <= eps or NaN
    """
    denom_ok = denom.where(denom > eps)
    return numer / denom_ok

def compute_partition_metrics(
    ds_dict,
    suffix="mean",                 # expects pr_mean, mrro_mean, tran_mean, ...
    eps=1e-12,
    clamp_bgws=True,
    bgws_clip=(-100, 100),
    compute_bgws=True,
    compute_ratios=True,
    variants=("tran", "evapo", "evspsbl"),  # which "ET-like" term to plug into BGWS
):
    """
    Adds BGWS variants and P-normalized ratios to datasets.

    Requires variables (with suffix):
      pr_{suffix}, mrro_{suffix},
      and (optional) tran_{suffix}, evapo_{suffix}, evspsbl_{suffix}

    Creates:
      - bgws_<term>_{suffix}  for each term in variants
      - r_over_p_{suffix}
      - et_over_p_{suffix}     (tran/pr)
      - e_over_p_{suffix}      (evapo/pr)
      - evap_over_p_{suffix}   (evspsbl/pr)  [ET/P]
    """
    pr_name   = f"pr_{suffix}"
    mrro_name = f"mrro_{suffix}"

    def compute_for_ds(ds: xr.Dataset, label=""):
        # --- required ---
        missing_req = [v for v in (pr_name, mrro_name) if v not in ds]
        if missing_req:
            print(f"[SKIP] {label}: missing required {missing_req}")
            return ds

        pr = ds[pr_name]
        mrro = ds[mrro_name]

        # --- ratios ---
        if compute_ratios:
            # R/P
            r_over_p = _safe_ratio(mrro, pr, eps=eps)
            ds[f"r_over_p_{suffix}"] = r_over_p
            ds[f"r_over_p_{suffix}"].attrs = {
                "long_name": "Runoff / Precipitation",
                "units": "1",
                "description": f"{mrro_name} / {pr_name}",
            }

            # ET-like ratios (only if present)
            if f"tran_{suffix}" in ds:
                et_over_p = _safe_ratio(ds[f"tran_{suffix}"], pr, eps=eps)
                ds[f"et_over_p_{suffix}"] = et_over_p
                ds[f"et_over_p_{suffix}"].attrs = {
                    "long_name": "Transpiration / Precipitation (Et/P)",
                    "units": "1",
                    "description": f"tran_{suffix} / {pr_name}",
                }

            if f"evapo_{suffix}" in ds:
                e_over_p = _safe_ratio(ds[f"evapo_{suffix}"], pr, eps=eps)
                ds[f"e_over_p_{suffix}"] = e_over_p
                ds[f"e_over_p_{suffix}"].attrs = {
                    "long_name": "Evaporation (non-transpiration) / Precipitation (E/P)",
                    "units": "1",
                    "description": f"evapo_{suffix} / {pr_name}",
                }

            if f"evspsbl_{suffix}" in ds:
                evap_over_p = _safe_ratio(ds[f"evspsbl_{suffix}"], pr, eps=eps)
                ds[f"evap_over_p_{suffix}"] = evap_over_p
                ds[f"evap_over_p_{suffix}"].attrs = {
                    "long_name": "Total Evapotranspiration / Precipitation (ET/P)",
                    "units": "1",
                    "description": f"evspsbl_{suffix} / {pr_name}",
                }

        # --- BGWS variants ---
        if compute_bgws:
            for term in variants:
                term_name = f"{term}_{suffix}"
                if term_name not in ds:
                    print(f"[BGWS SKIP] {label}: missing {term_name}")
                    continue

                bgws = _safe_ratio((mrro - ds[term_name]), pr, eps=eps) * 100.0

                # Replace inf with NaN (extra safety)
                bgws = xr.where(np.isinf(bgws), np.nan, bgws)

                if clamp_bgws:
                    lo, hi = bgws_clip
                    bgws = xr.where(bgws > hi, hi, bgws)
                    bgws = xr.where(bgws < lo, lo, bgws)

                out_name = f"bgws_{term}_{suffix}"
                ds[out_name] = bgws
                ds[out_name].attrs = {
                    "long_name": f"Blue-Green Water Share using {term}",
                    "units": "%",
                    "formula": f"(({mrro_name} - {term_name}) / {pr_name}) * 100",
                    "clip": f"{bgws_clip[0]}..{bgws_clip[1]}" if clamp_bgws else "none",
                }

        return ds

    # --- preserve your nested structure and skip ensemble keys ---
    ds_dict_clean = {}

    for key, value in ds_dict.items():
        if isinstance(value, dict):
            ds_dict_clean[key] = {}
            for model, ds in value.items():
                if model.startswith("Ensemble "):
                    print(f"Ignored ensemble data for {key} - {model}")
                    continue
                ds_dict_clean[key][model] = compute_for_ds(ds, label=f"{key}/{model}")
        else:
            if key.startswith("Ensemble "):
                print(f"Ignored ensemble data for {key}")
                continue
            ds_dict_clean[key] = compute_for_ds(value, label=str(key))

    return ds_dict_clean

    
def compute_diff_dict(ds_dict, reference_key, comparison_key, var_rel_change=None):
    """
    Computes absolute or relative differences between two scenarios in ds_dict.

    Rules:
    - Variables in `var_rel_change` are computed as relative change (%)
    - If var_rel_change == 'all', all shared variables are relative change (%)
    - Variable 'mrso' is ALWAYS computed as relative change (%)
    """
    ds_dict_change = {}

    ds_reference = ds_dict[reference_key]
    ds_comparison = ds_dict[comparison_key]

    # Remove ensemble keys
    keys_to_remove = [key for key in ds_reference.keys() if key.startswith("Ensemble ")]
    if keys_to_remove:
        print(f"Ensemble mean or median removed for keys: {keys_to_remove}")
        ds_reference = {k: v for k, v in ds_reference.items() if k not in keys_to_remove}
        ds_comparison = {k: v for k, v in ds_comparison.items() if k not in keys_to_remove}

    for model in ds_reference:
        if model not in ds_comparison:
            print(f"Skipping model '{model}' — missing in {comparison_key}")
            continue

        ref_ds = ds_reference[model]
        comp_ds = ds_comparison[model]

        common_vars = set(ref_ds.data_vars).intersection(comp_ds.data_vars)
        if not common_vars:
            print(f"Skipping model '{model}' — no shared variables")
            continue

        ds_change = xr.Dataset()

        # normalize var_rel_change input
        if var_rel_change == "all":
            rel_vars = set(common_vars)
        else:
            rel_vars = set(var_rel_change or [])

        for var in common_vars:
            # ALWAYS compute relative change for mrso
            force_relative = (var == "mrso")

            if force_relative or (var in rel_vars):
                denom = ref_ds[var].where(ref_ds[var] != 0)
                change = ((comp_ds[var] - ref_ds[var]) / denom) * 100
                change.attrs["units"] = "%"
                change.attrs["change_kind"] = "relative"
            else:
                change = comp_ds[var] - ref_ds[var]
                change.attrs["units"] = comp_ds[var].attrs.get("units", "")
                change.attrs["change_kind"] = "absolute"

            ds_change[var] = change

        ds_change.attrs = {
            "computed_from": f"{reference_key} vs {comparison_key}",
            "change_type": "mixed" if (var_rel_change not in [None, [], ()]) else "absolute_with_mrso_relative",
            "forced_relative_vars": ["mrso"],
        }

        ds_dict_change[model] = ds_change

    return ds_dict_change

def subdivide_ds_dict(
    ds_dict_base,
    ds_dict_change,
    base_id,
    change_id,
    variable,
    base_key=None,
):
    """
    Subdivide all datasets in ds_dict_change using regime masks derived from one base dataset.

    Parameters
    ----------
    ds_dict_base : dict or xr.Dataset
        Either:
        - a dict of datasets, in which case base_key must be provided, or
        - a single xr.Dataset directly
    ds_dict_change : dict
        Dictionary of datasets to which the same regime masks will be applied.
    base_id : str
        Label used in the subdivision coordinate names.
    change_id : str
        Output dictionary key.
    variable : str
        Variable used to define positive/negative regimes.
    base_key : str or None
        If ds_dict_base is a dict, select this key as the mask-defining dataset.

    Returns
    -------
    dict
        Nested dictionary: out[change_id][key] = subdivided dataset
    """
    import xarray as xr

    # choose the base dataset
    if isinstance(ds_dict_base, xr.Dataset):
        ds_base = ds_dict_base
    elif isinstance(ds_dict_base, dict):
        if base_key is None:
            raise ValueError("If ds_dict_base is a dict, you must provide base_key.")
        if base_key not in ds_dict_base:
            raise KeyError(f"base_key '{base_key}' not found in ds_dict_base.")
        ds_base = ds_dict_base[base_key]
    else:
        raise TypeError("ds_dict_base must be either a dict or an xr.Dataset.")

    subdivisions_masks = build_regime_masks(ds_base, base_id=base_id, variable=variable)

    ds_dict_change_sub = {change_id: {}}

    for key, ds_change in ds_dict_change.items():
        ds_dict_change_sub[change_id][key] = apply_regime_masks_to_dataset(
            ds_change, subdivisions_masks
        )

    print(
        f"Datasets subdivided based on historical state of {variable} "
        f"from base dataset '{base_key if base_key is not None else 'provided xr.Dataset'}'."
    )

    return ds_dict_change_sub

def subdivide_ds_by_regime(ds_base, ds_change, base_id, change_id, variable='bgws'):
    """
    Subdivide a single dataset based on historical data of a variable.
    
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
        dims=['subdivision', 'lat', 'lon'],
        coords={
            'subdivision': [f'Positive {base_id.capitalize()} {variable.upper()}', 
                            f'Negative {base_id.capitalize()} {variable.upper()}'],
            'lat': ds_base.lat,
            'lon': ds_base.lon,
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

    return expand_dataset


def build_regime_masks(ds_base, base_id, variable="bgws_tran_mean"):
    """
    Build positive/negative regime masks from one base dataset.
    """
    import numpy as np
    import xarray as xr

    if variable not in ds_base.data_vars:
        raise KeyError(f"Variable '{variable}' not found in ds_base.")

    base_var = ds_base[variable]

    mask_positive = xr.where(np.isfinite(base_var), base_var > 0, False)
    mask_negative = xr.where(np.isfinite(base_var), base_var < 0, False)

    subdivisions_masks = xr.concat(
        [mask_positive, mask_negative],
        dim="subdivision"
    ).assign_coords(
        subdivision=[
            f"Positive {base_id.capitalize()} {variable.upper()}",
            f"Negative {base_id.capitalize()} {variable.upper()}",
        ]
    )

    return subdivisions_masks

def apply_regime_masks_to_dataset(ds, subdivisions_masks):
    """
    Expand a dataset along subdivision and apply precomputed masks.
    """
    import xarray as xr

    # ensure compatible grid
    subdivisions_masks = subdivisions_masks.sel(lat=ds.lat, lon=ds.lon)

    expanded_vars = {}
    for name, var in ds.data_vars.items():
        expanded_var = var.expand_dims(subdivision=subdivisions_masks.subdivision)
        expanded_var = expanded_var.where(subdivisions_masks)
        expanded_vars[name] = expanded_var

    expanded_ds = xr.Dataset(
        expanded_vars,
        coords={**ds.coords, "subdivision": subdivisions_masks.subdivision}
    )
    return expanded_ds