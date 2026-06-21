"""
src/data_handling/compute_statistics.py

This script provides functions for data processing and statistical computation, 
including temporal and spatial statistics, ensemble statistics, and percentage changes.

Functions:
- `compute_statistic_per_model`: Compute a specified statistic (e.g., mean, std) for a dataset over a given dimension.
- `compute_temporal_statistic`: Compute a statistic over the temporal dimension (time or year).
- `compute_spatial_statistic`: Compute a statistic over the spatial dimension using weighted averaging.
- `compute_temporal_or_spatial_statistic`: Compute a statistic across datasets for a given dimension.
- `compute_yearly_sum`: Compute yearly sums while preserving NaN values.
- `compute_ensemble_statistic`: Compute ensemble-level statistics from multiple datasets.
- `compute_statistic_single`: Compute a single statistic for a dataset, optionally applying yearly means.
- `compute_yearly_means`: Compute yearly means for datasets.
- `compute_stats`: Compute yearly means and spatial statistics for datasets.
- `precompute_metrics`: Precompute correlation metrics (e.g., Pearson) for datasets.
- `compute_spatial_mean_with_subdivisions`: Compute spatial means for regions and subdivisions.
- `compute_area_percentage_changes`: Compute the percentage of area with positive/negative changes.
- `compute_flip_changes`: Compute grid-cell flips between positive and negative states.
- `compute_grid_cell_area`: Calculate the surface area of grid cells in a regular lat/lon grid.

Usage:
    Import this module to preprocess and analyze datasets for climate model analysis.

Author: Simon P. Heselschwerdt
Date: 2024-12-05
Dependencies: sys, os, xarray, pandas, numpy, multiprocessing, dask
"""

# ========== Imports ==========

import sys
import os
import xarray as xr
import pandas as pd
import copy
import numpy as np
import multiprocessing as mp
import dask
from dask.diagnostics import ProgressBar

# Define the full path to the directories containing utility scripts and configurations
data_handling_dir = '../src/data_handling'
config_file = '../src'

# Add directories to sys.path for importing custom modules
sys.path.append(data_handling_dir)
sys.path.append(config_file)

# Import custom utility functions and configurations
import load_data as load_dat
import process_data as pro_dat
from config import DATA_DIR, DEFAULT_EXPERIMENT, DEFAULT_TEMP_RES, DEFAULT_ACTIVITY_ID, DEFAULT_MEMBER_ID, DEFAULT_TABLE_ID, DEFAULT_MODEL, DEFAULT_VARIABLE

# ========== Statistic Computation Functions ==========

def compute_statistic_per_model(ds, dimension, statistic):
    """
    Computes a specified statistic for a single xarray dataset.

    Parameters:
    - ds: xarray Dataset to compute the statistic on.
    - dimension: The dimension to compute over, e.g., 'temporal' or 'spatial'.
    - statistic: The statistic to compute, e.g., 'mean', 'std', 'min', 'var', 'median'.

    Returns:
    - xarray Dataset with the computed statistic.
    """
    if dimension == "temporal":
        return compute_temporal_statistic(ds, statistic)
    elif dimension == "spatial":
        return compute_spatial_statistic(ds, statistic)
    else:
        raise ValueError(f"Invalid dimension '{dimension}' specified.")

def compute_temporal_statistic(ds, statistic):
    """
    Compute statistic over the temporal dimension (time or year) if available.

    Parameters:
    - ds: xarray Dataset to compute the statistic on.
    - statistic: The statistic to compute, e.g., 'mean', 'std', 'min', 'var', 'median'.

    Returns:
    - xarray Dataset with the computed statistic over the temporal dimension.
    """
    # Determine the dimension to compute the statistic over
    if 'time' in ds.dims:
        dimension = 'time'
    elif 'year' in ds.dims:
        dimension = 'year'
    else:
        raise ValueError("Neither 'time' nor 'year' dimension found in the dataset")

    # Compute the statistic
    stat_ds = getattr(ds, statistic)(dimension, keep_attrs=True, skipna=True)

    # Create an attribute called period to log the start and end year of the computed statistic if the dimension is 'time'
    if dimension == 'time':
        stat_ds.attrs['period'] = [str(ds.time.dt.year[0].values), str(ds.time.dt.year[-1].values)]
    
    return stat_ds

def compute_spatial_statistic(ds, statistic):
    """
    Compute statistic over spatial dimension.

    Parameters:
    - ds: xarray Dataset to compute the statistic on.
    - statistic: The statistic to compute, e.g., 'mean', 'std', 'min', 'var', 'median'.

    Returns:
    - xarray Dataset with the computed statistic over spatial dimension.
    """
    # Calculate weights based on latitude
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"

    # Log the weights for debugging and documentation purposes
    ds.attrs['weights'] = weights.values.tolist()
    
    # Apply weighted spatial averaging
    ds_weighted = ds.weighted(weights)
    
    # Compute the statistic on the weighted dataset
    stat_ds = getattr(ds_weighted, statistic)(("lon", "lat"), keep_attrs=True, skipna=True)
    
    return stat_ds

def compute_temporal_or_spatial_statistic(ds_dict, dimension, statistic):
    """
    Computes the specified statistic for each dataset in the dictionary.

    Parameters:
    - ds_dict: Dictionary of xarray datasets.
    - dimension: The dimension to compute over, e.g., 'temporal' or 'spatial'.
    - statistic: The statistic to compute, e.g., 'mean', 'std', 'min', 'max', 'var', 'median'.

    Returns:
    - Dictionary with computed statistic for each dataset.
    """
    # Check if function inputs are valid
    validate_inputs(ds_dict, dimension, statistic)

    # Compute statistic for each model using parallel processing
    with mp.Pool() as pool:
        results = pool.starmap(compute_statistic_per_model, [(ds, dimension, statistic) for ds in ds_dict.values()])

    return dict(zip(ds_dict.keys(), results))

def validate_inputs(ds_dict, dimension, statistic):
    if not isinstance(ds_dict, dict):
        raise TypeError("ds_dict must be a dictionary of xarray datasets.")
    if not all(isinstance(ds, xr.Dataset) for ds in ds_dict.values()):
        raise TypeError("All values in ds_dict must be xarray datasets.")
    if statistic not in ["mean", "std", "min", "max", "var", "median"]:
        raise ValueError(f"Invalid statistic '{statistic}' specified.")
    if dimension not in ["temporal", "spatial"]:
        raise ValueError(f"Invalid dimension '{dimension}' specified.")

def compute_yearly_sum(ds):
    """
    Compute the yearly sum while ignoring NaN values.

    Parameters:
    - ds: The dataset to compute the yearly sum from.

    Returns:
    - The dataset with yearly summed values, preserving NaN values for grid cells that were NaN in the original data.
    """
    if 'time' in ds.dims and 'time' in ds.coords:
        attrs = ds.attrs
        days = ds['time'].dt.days_in_month
        
        # Check if the unit is in 'day' or 'month' and replace it with 'year'
        for var in ds.data_vars:
            unit = ds[var].attrs.get('units', '')
            if 'day' in unit or 'month' in unit:
                new_unit = unit.replace('day', 'year').replace('month', 'year')
                ds[var].attrs['units'] = new_unit
            else:
                print(f"No yearly sum computed as variable '{var}' has units '{unit}'")
                return ds

        # Calculate the yearly sum, ignoring NaN values
        yearly_sum_ds = (ds * days).resample(time='AS').sum(dim='time', skipna=True)
        
        # Create a mask for NaN values in the original dataset
        nan_mask = ds.isnull().any(dim='time')
        
        # Apply the mask to the resulting dataset to preserve NaN values
        yearly_sum_ds = yearly_sum_ds.where(~nan_mask)
        
        yearly_sum_ds.attrs = attrs
        
        return yearly_sum_ds
    return ds

def compute_ensemble_statistic(ds_dict, statistic, output_name=None):
    """
    Computes the specified statistic for an ensemble of xarray datasets and adds it to the dictionary.
    
    Parameters:
    ds_dict (dict): A dictionary of xarray datasets, potentially nested with experiments and models.
    statistic (str): The statistic to compute.

    Returns:
    dict: The input dictionary with the ensemble statistic added.
    """
    def compute_ensemble_for_experiment(ds_experiment_dict, statistic, output_name=None):
        datasets_to_combine = {
            key: ds for key, ds in ds_experiment_dict.items()
            if key not in ['4 model ensemble mean', '5 model ensemble mean', '9 model ensemble mean']
        }

        excluded_from_start = [
            key for key in ds_experiment_dict.keys()
            if key in ['4 model ensemble mean', '5 model ensemble mean', '9 model ensemble mean']
        ]

        if excluded_from_start:
            print(f"Initial exclusion from ensemble ({statistic}): {', '.join(excluded_from_start)}")
        
        for key in datasets_to_combine:
            if 'member_id' in datasets_to_combine[key].coords:
                datasets_to_combine[key] = datasets_to_combine[key].drop_vars('member_id')
    
        # Use union instead of intersection
        all_vars = set.union(*(set(ds.data_vars) for ds in datasets_to_combine.values()))
        
        result = xr.Dataset()
        all_excluded = {}
        
        for var in all_vars:
            valid_datasets = {}
            excluded_models = []
            for model_name, ds in datasets_to_combine.items():
                if var in ds:
                    valid_datasets[model_name] = ds[var]
                else:
                    excluded_models.append(model_name)
    
            if len(valid_datasets) == 0:
                print(f"Skipping '{var}': not found in any datasets.")
                continue
    
            if excluded_models:
                print(f"Excluded {excluded_models} from ensemble statistic for variable '{var}' due to missing data.")
                all_excluded[var] = excluded_models
    
            try:
                combined = xr.concat(valid_datasets.values(), dim='ensemble')
                result[var] = getattr(combined, statistic)(dim='ensemble')
                result[var].attrs['excluded_models'] = excluded_models  # Optional: record in attrs
            except Exception as e:
                print(f"Failed to compute statistic '{statistic}' for variable '{var}': {e}")

        result.attrs['description'] = f'Ensemble {statistic}'
        result.attrs['computed_from'] = list(datasets_to_combine.keys())

        
        if all_excluded:
            summary_lines = []
            for var, models in all_excluded.items():
                summary_lines.append(f"{var}: excluded {', '.join(models)}")
            result.attrs['note'] = "Some models were excluded from specific variables:\n" + "\n".join(summary_lines)

        
        return result if result.data_vars else None


    ds_dict_result = {}

    # Check if the dictionary is nested with experiments
    for key, value in ds_dict.items():
        if isinstance(value, dict):  # Nested with experiments
            ds_experiment_result = compute_ensemble_for_experiment(value, statistic)
            if ds_experiment_result is not None:
                if output_name is None:
                    ds_dict_result[f'Ensemble {statistic} {key}'] = ds_experiment_result
                else:
                    ds_dict_result[f'{output_name} {statistic} {key}'] = ds_experiment_result
        else:  # Directly datasets
            ds_experiment_result = compute_ensemble_for_experiment(ds_dict, statistic)
            if ds_experiment_result is not None:
                if output_name is None:
                    ds_dict_result[f'Ensemble {statistic}'] = ds_experiment_result
                else:
                    ds_dict_result[f'{output_name} {statistic}'] = ds_experiment_result
            break
    
    print(f"Computed Ensemble {statistic} for all experiments.")
    return {**ds_dict, **ds_dict_result}

def compute_statistic_single(ds, statistic, dimension, yearly_mean=True):
    if dimension == "time":
        stat_ds = getattr(ds, statistic)("time", keep_attrs=True, skipna=True)
        stat_ds.attrs['period'] = [str(ds.time.dt.year[0].values), str(ds.time.dt.year[-1].values)]
        
    if dimension == "space":
        # Assign the period attribute before grouping by year
        ds.attrs['period'] = [str(ds.time.dt.year[0].values), str(ds.time.dt.year[-1].values)]
        
        if yearly_mean:
            ds = ds.groupby('time.year').mean('time', keep_attrs=True, skipna=True)
            ds.attrs['mean'] = 'yearly mean'
              
        #get the weights, apply on data, and compute statistic
        weights = np.cos(np.deg2rad(ds.lat))
        weights.name = "weights"
        ds_weighted = ds.weighted(weights)
        stat_ds = getattr(ds_weighted, statistic)(("lon", "lat"), keep_attrs=True, skipna=True)
    
    stat_ds.attrs['statistic'] = statistic
    stat_ds.attrs['statistic_dimension'] = dimension

    return stat_ds

def precompute_metrics(ds_dict, variables, metrics=['pearson']):
    # Initialize the results dictionary
    results_dict = {metric: {} for metric in metrics}
    
    for name, ds in ds_dict.items():
        # Create a DataFrame with all the variables
        df = pd.DataFrame({var: ds[var].values.flatten() for var in variables})
        
        # Define all pairs of variables
        pairs = list(permutations(variables, 2))  # <-- Change here
        args = [(df, var1, var2, metrics) for var1, var2 in pairs]

        # Use a multiprocessing pool to compute the metrics for all pairs
        with Pool() as p:
            results = p.map(compute_metrics_for_pair, args)
        
        # Store the results in the results_dict
        for var1, var2, metric_dict in results:
            for metric, value in metric_dict.items():
                # Ensure the keys exist in the dictionary
                results_dict[metric].setdefault(name, {}).setdefault(f'{var1}_{var2}', value)
    return results_dict

def compute_stats(ds_dict):
    """
    Compute yearly mean of each variable in the dataset.

    Parameters:
    ds_dict (dict): The input dictionary of xarray.Dataset.

    Returns:
    dict: A dictionary where the keys are the dataset names and the values are another dictionary.
          This inner dictionary has keys as variable names and values as DataArray of yearly means.
    """
    stats = {}
    for model, ds in ds_dict.items():
        # Compute the yearly mean
        yearly_ds = ds.resample(time='1Y').mean()

        stats[model] = {}
        for var in yearly_ds.data_vars:
            # Compute the spatial mean
            spatial_mean = compute_spatial_statistic(yearly_ds[var], 'mean')
            
            # Store the yearly mean values
            stats[model][var] = spatial_mean
    return stats

def compute_yearly_means(ds_dict):
    yearly_means_dict = {}

    # For each dataset, compute the yearly mean over the 'time', 'lat', and 'lon' dimensions
    for name, ds in ds_dict.items():  
        ds_yearly = ds.groupby('time.year').mean('time')    
        
        yearly_means_dict[name] = ds_yearly

    return yearly_means_dict

# ========== Utility Functions ==========

def compute_grid_cell_area(ds):
    """
    Calculate the surface area of each grid cell in km² for a regular lat/lon grid.

    Args:
        ds (xarray.Dataset): The dataset containing latitude ('lat') and longitude ('lon').

    Returns:
        xarray.DataArray: A 2D array of grid cell areas (km²) with the same lat/lon dimensions.
    """
    # Earth's radius in kilometers
    EARTH_RADIUS = 6371.0

    # Latitude and longitude spacing (assuming uniform grid)
    lat_spacing = np.abs(ds.lat.diff(dim='lat').mean().item())  # degrees
    lon_spacing = np.abs(ds.lon.diff(dim='lon').mean().item())  # degrees

    # Convert spacing to radians
    lat_spacing_rad = np.deg2rad(lat_spacing)
    lon_spacing_rad = np.deg2rad(lon_spacing)

    # Latitude weights (cosine of latitude)
    weights = np.cos(np.deg2rad(ds['lat']))

    # Calculate the area of each grid cell
    grid_cell_area = (
        (EARTH_RADIUS ** 2) * 
        lat_spacing_rad * 
        lon_spacing_rad * 
        weights
    )

    # Expand to 2D for all longitudes
    grid_area_2d = xr.DataArray(
        np.outer(grid_cell_area, np.ones(len(ds.lon))),
        dims=['lat', 'lon'],
        coords={'lat': ds.lat, 'lon': ds.lon}
    )

    return grid_area_2d


def apply_hydrological_mask_from_historical(ds_dict, ds_dict_hydro_mask=None, pr_thresh=0.05, tran_thresh=0.005, mrro_thresh=0.005):
    """
    Apply a hydrological activity mask to all datasets in a ds_dict,
    using the '4 Model Ensemble Mean' from the historical scenario to define the mask.

    Parameters:
    - ds_dict (dict): Dictionary where keys are scenarios (e.g., 'historical', 'ssp126')
                      and values are dicts of model names to xarray Datasets.
    - pr_thresh (float): Precipitation threshold in mm/day
    - tran_thresh (float): Transpiration threshold in mm/day
    - mrro_thresh (float): Runoff threshold in mm/day

    Returns:
    - masked_ds_dict (dict): Same structure, with all variables masked consistently
                             based on the ensemble mean of the historical scenario.
    """
    # Extract the ensemble mean dataset from historical scenario
    if ds_dict_hydro_mask is not None:
        try:
            ensemble_ds = ds_dict_hydro_mask['historical']['4 model ensemble mean']
        except KeyError:
            raise KeyError("Ensemble mean dataset ('4 model ensemble mean') not found in 'historical' scenario.")
    else:
        experiments = ["historical"]
        models = ["BCC-CSM2-MR_r1i1p1f1", "CMCC-ESM2_r1i1p1f1", "MIROC-ES2L_r1i1p1f2", "UKESM1-0-LL_ensmean"]
        variables=['pr', 'mrro', 'tran']
        season = None

        # Step 1.2: Load the datasets
        print("Historical data for mask needs to be loaded first...")
        with ProgressBar():
            ds_dict_hydro_mask = dask.compute(
                load_dat.load_period_mean(
                    DATA_DIR, 'processed', experiments, models, variables, specific_months_or_seasons=season
                )
            )[0]

        ds_dict_hydro_mask['historical'] = compute_ensemble_statistic(ds_dict_hydro_mask['historical'], 'mean', '4 model ensemble')
        ensemble_ds = ds_dict_hydro_mask['historical']['4 model ensemble mean']

    # Build the ensemble-based mask
    pr = ensemble_ds['pr']
    tran = ensemble_ds['tran']
    mrro = ensemble_ds['mrro']
    mask = (pr > pr_thresh) & (tran > tran_thresh) & (mrro > mrro_thresh)

    # Apply the mask to all scenarios and all models
    masked_ds_dict = {}

    for model_name, ds in ds_dict.items():
        if isinstance(ds, dict):
            raise TypeError(
                f"Expected xarray.Dataset for '{model_name}', but got dict. "
            )
        
        # Apply the ensemble mask to all variables in the dataset
        masked_vars = {
            var: ds[var].where(mask) for var in ds.data_vars
        }

        masked_ds = xr.Dataset(masked_vars, attrs=ds.attrs)

        # Ensure coordinates are preserved
        for coord in ds.coords:
            if coord not in masked_ds.coords:
                masked_ds = masked_ds.assign_coords({coord: ds[coord]})

        masked_ds_dict[model_name] = masked_ds

    return masked_ds_dict, ds_dict_hydro_mask

def compute_sustain_lulcc(models, ds_dict):
    ds_dict_sustain = {}
    # Compute differences for each model
    for model in models:
        # Get relevant datasets
        ssp126 = ds_dict['ssp126'][model]
        ssp370 = ds_dict['ssp370'][model]
        ssp126_370Lu = ds_dict['ssp126-ssp370Lu'][model]
        ssp370_126Lu = ds_dict['ssp370-ssp126Lu'][model]
    
        # Compute sustain_lulcc
        sustain_lulcc = 0.5 * (ssp126 - ssp126_370Lu) + 0.5 * (ssp370_126Lu - ssp370)
    
        # Save to output dictionary
        ds_dict_sustain[model] = sustain_lulcc

    return ds_dict_sustain

def data_sustain_lulcc(ds_dict_hydro_mask=None):
    # Step 1: Define the datasets
    experiments = ["ssp126", "ssp126-ssp370Lu", "ssp370", "ssp370-ssp126Lu"]
    models = ["BCC-CSM2-MR_r1i1p1f1", "CMCC-ESM2_r1i1p1f1", "MIROC-ES2L_r1i1p1f2", "UKESM1-0-LL_ensmean"]
    variables=['pr', 'mrro', 'tran', 'lai', 'gpp', 'mrsos', "evapo", "evspsbl", "pr_no_landmask", "evspsbl_no_landmask"]

    # Step 2: Load the datasets
    print("Loading period means...")
    with ProgressBar():
        ds_dict = dask.compute(
            load_dat.load_period_mean(
                DATA_DIR, 'processed', experiments, models, variables, specific_months_or_seasons=None
            )
        )[0]

  
    # Step 3: Compute bgws
    ds_dict = pro_dat.compute_bgws(ds_dict)

    # Step 4: Compute sustainable LULCC
    ds_dict_diff = {}
    ds_dict_diff = compute_sustain_lulcc(models, ds_dict)

    # Step 5: Compute MMM
    ensmean = compute_ensemble_statistic(ds_dict_diff, "mean", "4 model ensemble")
    ds_dict_diff["4 model ensemble mean"] = ensmean["4 model ensemble mean"]

    # Step 6: Apply flow mask
    masked_ds_dict_diff = {}
    masked_ds_dict_diff['sustain_lulcc'], ds_dict_hydro_mask  = apply_hydrological_mask_from_historical(ds_dict_diff, ds_dict_hydro_mask)

    return masked_ds_dict_diff, ds_dict_hydro_mask


def data_sustain_lulcc_seasonal(ds_dict_hydro_mask=None):
    # Step 1: Define the datasets
    experiments = ["ssp126", "ssp126-ssp370Lu", "ssp370", "ssp370-ssp126Lu"]
    models = ["BCC-CSM2-MR_r1i1p1f1", "CMCC-ESM2_r1i1p1f1", "MIROC-ES2L_r1i1p1f2", "UKESM1-0-LL_ensmean"]
    variables=['pr', 'mrro', 'tran', 'lai', 'gpp', 'mrsos', 'evapo', 'evspsbl']

    # Step 2: Load the datasets
    print("Loading period means...")
    with ProgressBar():
        ds_dict_monthly = dask.compute(
            load_dat.load_period_seasonal_clim(
                DATA_DIR, 'processed', experiments, models, variables
            )
        )[0]


    # Step 3: Compute monthly bgws
    ds_dict_monthly = pro_dat.compute_bgws(ds_dict_monthly)

    # Step 4: Compute sustainable LULCC
    ds_dict_diff_monthly = {}
    ds_dict_diff_monthly = compute_sustain_lulcc(models, ds_dict_monthly)

    # Step 5: Compute MMM 
    ensmean = compute_ensemble_statistic(ds_dict_diff_monthly, "mean", "4 model ensemble")
    ds_dict_diff_monthly["4 model ensemble mean"] = ensmean["4 model ensemble mean"]
    
    # Step 6: Compute MMM 
    masked_ds_dict_monthly = {}
    masked_ds_dict_monthly['sustain_lulcc'], _ = apply_hydrological_mask_from_historical(ds_dict_diff_monthly, ds_dict_hydro_mask)

    return masked_ds_dict_monthly

def data_historical_lulcc(ds_dict_hydro_mask=None):
    # Step 1: Define the datasets
    experiments = ["historical", "hist-noLu",]
    models = ["BCC-CSM2-MR_r1i1p1f1", "CMCC-ESM2_r1i1p1f1", "MIROC-ES2L_r1i1p1f2", "UKESM1-0-LL_ensmean","CNRM-ESM2-1_ensmean", "CanESM5_ensmean", "IPSL-CM6A-LR_ensmean", "EC-Earth3-Veg_r1i1p1f1", "CESM2_ensmean"]
    variables=['pr', 'mrro', 'tran', 'lai', 'gpp']

    # Step 2: Load the datasets
    print("Loading period means...")
    with ProgressBar():
        ds_dict = dask.compute(
            load_dat.load_period_mean(
                DATA_DIR, 'processed', experiments, models, variables, specific_months_or_seasons=None
            )
        )[0]

    # Step 3: Compute bgws
    ds_dict = pro_dat.compute_bgws(ds_dict)

    # Step 4: Compute historical LULCC
    ds_dict_diff = {}
    ds_dict_diff = pro_dat.compute_diff_dict(ds_dict, reference_key='hist-noLu', comparison_key='historical', var_rel_change=None)

    # Step 5: Compute MMMs
    core_models = [
    "BCC-CSM2-MR_r1i1p1f1",
    "CMCC-ESM2_r1i1p1f1",
    "MIROC-ES2L_r1i1p1f2",
    "UKESM1-0-LL_ensmean",
    ]
    
    add_models = [
        "CNRM-ESM2-1_ensmean",
        "CanESM5_ensmean",
        "IPSL-CM6A-LR_ensmean",
        "EC-Earth3-Veg_r1i1p1f1",
        "CESM2_ensmean",
    ]

    # 9-model MMM (all models)
    ens9 = compute_ensemble_statistic(ds_dict_diff, "mean", "9 model ensemble")
    ds_dict_diff["9 model ensemble mean"] = ens9["9 model ensemble mean"]
    
    # 4-core MMM
    core_dict = {m: ds_dict_diff[m] for m in core_models}
    ens4 = compute_ensemble_statistic(core_dict, "mean", "4 model core ensemble")
    ds_dict_diff["4 model ensemble mean"] = ens4["4 model core ensemble mean"]
    
    # 5-additional MMM
    add_dict = {m: ds_dict_diff[m] for m in add_models}
    ens5 = compute_ensemble_statistic(add_dict, "mean", "5 model additional ensemble")
    ds_dict_diff["5 model ensemble mean"] = ens5["5 model additional ensemble mean"]

    # Step 6: Apply flow mask
    masked_ds_dict_diff = {}
    masked_ds_dict_diff['historical-hist-noLu'], _ = apply_hydrological_mask_from_historical(ds_dict_diff, ds_dict_hydro_mask)

    return masked_ds_dict_diff


def luh2_data(ds_dict_hydro_mask):
    experiments = ["historical", "ssp126", "ssp370"]
    state_variables=[
        "primf",
        "primn",
        "secdf",
        "secdn",
        "secmb",
        "secma",
        "c3ann",
        "c4ann",
        "c3per",
        "c4per",
        "c3nfx",
        "pastr",
        "range"
    ]

    # Step 2: Load the datasets
    print("Loading period means...")
    with ProgressBar():
        ds_dict_luh2 = dask.compute(
            load_dat.load_period_mean_LUH2(
                DATA_DIR, 'processed', experiments, ["LUH2"], state_variables
            )
        )[0]

    # Step 3: Compute land use metrics
    for scenario in ds_dict_luh2.keys():
        ds_dict_luh2[scenario]['LUH2'] = compute_land_use_metrics(ds_dict_luh2[scenario]['LUH2'])

    # Step 4: Apply hydrological mask
    masked_ds_dict_luh2 = {}
    
    for scenario in ds_dict_luh2.keys():
        # Wrap LUH2 dataset so it looks like {model_name: Dataset}
        tmp_in = {"LUH2": ds_dict_luh2[scenario]["LUH2"]}
    
        tmp_out, _ = apply_hydrological_mask_from_historical(tmp_in, ds_dict_hydro_mask)
    
        # Unwrap back to scenario structure
        masked_ds_dict_luh2[scenario] = {"LUH2": tmp_out["LUH2"]}

    # Step 5: Compute differences (comparison ds - reference ds)
    ds_dict_luh2_diff = {}
    ds_dict_luh2_diff['ssp126-historical'] = pro_dat.compute_diff_dict(masked_ds_dict_luh2, reference_key='historical', comparison_key='ssp126', var_rel_change=None)
    ds_dict_luh2_diff['ssp370-historical'] = pro_dat.compute_diff_dict(masked_ds_dict_luh2, reference_key='historical', comparison_key='ssp370', var_rel_change=None)
    ds_dict_luh2_diff['sustain_lulcc'] = pro_dat.compute_diff_dict(masked_ds_dict_luh2, reference_key='ssp370', comparison_key='ssp126', var_rel_change=None)

    return ds_dict_luh2_diff

def compute_land_use_metrics(ds):
    """
    Compute new land-use variables and update the dataset.
    """
    new_vars = {
        "Forest": (ds["primf"] + ds["secdf"]) * 100,
        "Cropland": (ds["c3ann"] + ds["c4ann"] + ds["c3per"] + ds["c4per"] + ds["c3nfx"]) * 100,
        "Grazing Land": (ds["pastr"] + ds["range"]) * 100,
        "Natural Non-Forest": (ds["primn"] + ds["secdn"]) * 100
    }
    
    # Add new variables to the dataset
    ds = ds.assign(new_vars)
    return ds

def data_land_cover(ds_dict_hydro_mask):
    # Step 1: Define the datasets
    experiments = ["ssp126", "ssp126-ssp370Lu", "ssp370", "ssp370-ssp126Lu"]
    models = ["CMCC-ESM2_r1i1p1f1", "MIROC-ES2L_r1i1p1f2", "UKESM1-0-LL_ensmean",  "UKESM1-0-LL_r4i1p1f2"]
    variables=["fracLut", "treeFrac"]

    # Step 2: Load the datasets
    print("Loading period means...")
    with ProgressBar():
        ds_dict = dask.compute(
            load_dat.load_period_mean(
                DATA_DIR, 'processed', experiments, models, variables, specific_months_or_seasons=None
            )
        )[0]

    # Step 3: Compute sustainable LULCC
    core_models = [
        'CMCC-ESM2_r1i1p1f1',
        'MIROC-ES2L_r1i1p1f2',
        'UKESM1-0-LL_ensmean',
        'UKESM1-0-LL_r4i1p1f2'
    ]
    
    ds_dict_diff = {}
    ds_dict_diff['sustain_lulcc'] = {}
    ds_dict_diff['sustain_lulcc'] = compute_sustain_lulcc(core_models, ds_dict)

    # Step 4: Apply hydrological mask
    masked_ds_dict_diff = {}
    masked_ds_dict_diff['sustain_lulcc'], _ = apply_hydrological_mask_from_historical(ds_dict_diff['sustain_lulcc'], ds_dict_hydro_mask)

    return masked_ds_dict_diff

import numpy as np
import xarray as xr

def data_carbon_nitrogen_diagnostics(ds_dict_hydro_mask):
    # Step 1: Define the datasets
    experiments = ["ssp126", "ssp126-ssp370Lu", "ssp370", "ssp370-ssp126Lu"]
    models = ["BCC-CSM2-MR_r1i1p1f1", "MIROC-ES2L_r1i1p1f2", "UKESM1-0-LL_ensmean", "UKESM1-0-LL_r4i1p1f2"]
    variables=["lai", "cLeaf", "cRoot", "cVeg", "nVeg"]

    # Step 2: Load the datasets
    print("Loading period means...")
    with ProgressBar():
        ds_dict = dask.compute(
            load_dat.load_period_mean(
                DATA_DIR, 'processed', experiments, models, variables, specific_months_or_seasons=None
            )
        )[0]

    # Step 3: Compute diagnostics
    ds_dict = compute_diagnostics(ds_dict)
    
    # Step 4: Compute sustainable LULCC
    ds_dict_diff = {}
    ds_dict_diff['sustain_lulcc'] = {}
    ds_dict_diff['sustain_lulcc'] = compute_sustain_lulcc(models, ds_dict)

    # Step 5: Apply hydrological mask
    masked_ds_dict_diff = {}
    masked_ds_dict_diff['sustain_lulcc'], _ = apply_hydrological_mask_from_historical(ds_dict_diff['sustain_lulcc'], ds_dict_hydro_mask)

    return masked_ds_dict_diff

def compute_diagnostics(ds_dict):
    def safe_div(num, den):
        return xr.where(den != 0, num / den, np.nan)
        
    for experiment in ds_dict.keys():
        for model, ds in ds_dict[experiment].items():
            # --- cVeg/nVeg ---
            need = {'cVeg', 'nVeg'}
            if need.issubset(ds.data_vars):
                ds['cVeg_nVeg'] = safe_div(ds['cVeg'], ds['nVeg'])
                ds['cVeg_nVeg'].attrs.update(long_name='Vegetation C:N ratio', units='kgC kgN^-1')
            else:
                missing = need - set(ds.data_vars)
                print(f"[skip] {model}: missing {missing} for cVeg/nVeg")
    
            # --- lai/cLeaf ---
            need = {'lai', 'cLeaf'}
            if need.issubset(ds.data_vars):
                ds['lai_cLeaf'] = safe_div(ds['lai'], ds['cLeaf'])
                ds['lai_cLeaf'].attrs.update(long_name='Leaf area per leaf C', units='m^2 kgC^-1')
            else:
                missing = need - set(ds.data_vars)
                print(f"[skip] {model}: missing {missing} for lai/cLeaf")
    
            # --- cWood = cVeg - cLeaf - cRoot ---
            need = {'cVeg', 'cLeaf', 'cRoot'}
            if need.issubset(ds.data_vars):
                ds['cWood'] = ds['cVeg'] - ds['cLeaf'] - ds['cRoot']
                ds['cWood'].attrs.update(long_name='Woody carbon', units=ds['cVeg'].attrs.get('units', 'kgC m^-2'))
            else:
                missing = need - set(ds.data_vars)
                print(f"[skip] {model}: missing {missing} for cWood")
    
            # --- Carbon pool partitioning ratios (fractions of cVeg) ---
            # Require cVeg and each component present
            # f_leaf = cLeaf / cVeg; f_root = cRoot / cVeg; f_wood = cWood / cVeg
            if {'cVeg', 'cLeaf'}.issubset(ds.data_vars):
                ds['f_leaf'] = safe_div(ds['cLeaf'], ds['cVeg'])
                ds['f_leaf'].attrs.update(long_name='Leaf fraction of vegetation C', units='1')
            else:
                print(f"[skip] {model}: missing {{'cVeg','cLeaf'}} for f_leaf")
    
            if {'cVeg', 'cRoot'}.issubset(ds.data_vars):
                ds['f_root'] = safe_div(ds['cRoot'], ds['cVeg'])
                ds['f_root'].attrs.update(long_name='Root fraction of vegetation C', units='1')
            else:
                print(f"[skip] {model}: missing {{'cVeg','cRoot'}} for f_root")
    
            # f_wood can be computed if cWood exists or all three pools exist
            if 'cWood' in ds and 'cVeg' in ds.data_vars:
                ds['f_wood'] = safe_div(ds['cWood'], ds['cVeg'])
                ds['f_wood'].attrs.update(long_name='Wood fraction of vegetation C', units='1')
            elif {'cVeg', 'cLeaf', 'cRoot'}.issubset(ds.data_vars):
                tmp_cwood = ds['cVeg'] - ds['cLeaf'] - ds['cRoot']
                ds['f_wood'] = safe_div(tmp_cwood, ds['cVeg'])
                ds['f_wood'].attrs.update(long_name='Wood fraction of vegetation C (derived)', units='1')
            else:
                print(f"[skip] {model}: missing pools for f_wood")

    return ds_dict
