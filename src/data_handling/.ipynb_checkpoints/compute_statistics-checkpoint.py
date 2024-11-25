"""
src/data_handling/compute_statistics.py

This script provides functions to process data. Users can compute different statistics.

Functions:
- compute_statistic_per_model
- compute_temporal_statistic
- compute_spatial_statistic
- compute_temporal_or_spatial_statistic
- compute_yearly_sum
- compute_ensemble_statistic

Usage:
    Import this module in your scripts to process data for the analysis.
"""

import sys
import os
import xarray as xr
import pandas as pd
import copy
import numpy as np
import multiprocessing as mp
import dask
from dask.diagnostics import ProgressBar

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

def compute_ensemble_statistic(ds_dict, statistic):
    """
    Computes the specified statistic for an ensemble of xarray datasets and adds it to the dictionary.
    
    Parameters:
    ds_dict (dict): A dictionary of xarray datasets, potentially nested with experiments and models.
    statistic (str): The statistic to compute.

    Returns:
    dict: The input dictionary with the ensemble statistic added.
    """
    def compute_ensemble_for_experiment(ds_experiment_dict, statistic):
        # Exclude existing ensemble statistics from the computation
        datasets_to_combine = {key: ds for key, ds in ds_experiment_dict.items() if not key.startswith('Ensemble ')}
        
        # Drop 'member_id' coordinate if it exists in any of the datasets
        for ds_key in datasets_to_combine:
            if 'member_id' in datasets_to_combine[ds_key].coords:
                datasets_to_combine[ds_key] = datasets_to_combine[ds_key].drop_vars('member_id')
        
        combined = xr.concat(datasets_to_combine.values(), dim='ensemble')
        
        try:
            # Compute the desired ensemble statistic using getattr
            result = getattr(combined, statistic)(dim='ensemble')
        except AttributeError:
            print(f"Statistic '{statistic}' is not valid for xarray objects.")
            return None
        
        # Add attributes to the resulting dataset
        result.attrs['description'] = f'Ensemble {statistic}'
        result.attrs['computed_from'] = list(datasets_to_combine.keys())
        
        return result
    
    ds_dict_result = {}

    # Check if the dictionary is nested with experiments
    for key, value in ds_dict.items():
        if isinstance(value, dict):  # Nested with experiments
            ds_experiment_result = compute_ensemble_for_experiment(value, statistic)
            if ds_experiment_result is not None:
                ds_dict_result[f'Ensemble {statistic} {key}'] = ds_experiment_result
        else:  # Directly datasets
            ds_experiment_result = compute_ensemble_for_experiment(ds_dict, statistic)
            if ds_experiment_result is not None:
                ds_dict_result[f'Ensemble {statistic}'] = ds_experiment_result
            break
    
    print(f"Computed Ensemble {statistic} for all experiments.")
    return {**ds_dict, **ds_dict_result}

def calculate_spatial_mean(ds_dict):
    ds_dict_mean = {}
    for key, ds in ds_dict.items():
        attrs = ds.attrs
        for var in list(ds.data_vars.keys()):
            var_attrs = ds[var].attrs
            
            ds_dict_mean[key][var] = ds.mean(['lon', 'lat'])
            ds_dict_mean[key][var].attrs = var_attrs
        
        ds_dict_mean[key].attrs = attrs
        
    return ds_dict_mean

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
            spatial_mean = yearly_ds[var].mean(dim=['lat', 'lon'])
            
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

def compute_yearly_regional_means(ds_dict_region):
    yearly_means_dict = {}

    # For each dataset, compute the yearly mean over the 'time', 'lat', and 'lon' dimensions
    for region, ds_dict in ds_dict_region.items():
        yearly_means_dict[region] = {}
        for ds_name, ds in ds_dict.items():
            # Compute the yearly mean
            ds_yearly = ds.groupby('time.year').mean('time')
            
            # Create weights
            weights = np.cos(np.deg2rad(ds.lat))
            # Apply the weights and calculate the spatial mean
            ds_weighted = ds_yearly.weighted(weights)
            yearly_means_dict[region][ds_name] = ds_weighted.mean(('lat', 'lon'))

    return yearly_means_dict

def calculate_spatial_mean(ds_dict):
    ds_dict_mean = {}
    
    for key, ds in ds_dict.items():
        attrs = ds.attrs
        
        # Initialize a new Dataset for this key
        ds_dict_mean[key] = xr.Dataset()
        
        for var in list(ds.data_vars.keys()):
            var_attrs = ds[var].attrs
            
            ds_dict_mean[key][var] = ds[var].mean(['lon', 'lat'])
            ds_dict_mean[key][var].attrs = var_attrs
        
        ds_dict_mean[key].attrs = attrs
        
    return ds_dict_mean

def calculate_spatial_mean_std(ds_dict):
    ds_stats = {}
    
    for key, ds in ds_dict.items():
        attrs = ds.attrs
        
        # Initialize a new Dataset for spatial statistics
        ds_stats[key] = xr.Dataset()
        
        for var in list(ds.data_vars.keys()):
            var_attrs = ds[var].attrs
            
            # Calculate the mean and standard deviation, skipping NaN values
            ds_stats[key][f'{var}'] = ds[var].mean(['lon', 'lat'], skipna=True)
            ds_stats[key][f'{var}'].attrs = var_attrs
            
            # Use a minimum count of 2 for standard deviation to ensure degrees of freedom > 0
            ds_stats[key][f'{var}_std'] = ds[var].std(['lon', 'lat'], skipna=True)
            ds_stats[key][f'{var}_std'].attrs = var_attrs
        
        ds_stats[key].attrs = attrs
        
    return ds_stats

def compute_regional_means(ds_dict_region):
    """
    Computes the spatial mean for each region in the masked datasets using weighted averaging.

    Args:
        ds_dict_region (dict): A dictionary of xarray datasets organized by experiments and models,
                               where each dataset has a region dimension added.

    Returns:
        dict: A new dictionary where keys are the same as in the input dictionary,
              and each value is an xarray Dataset with the weighted spatial mean computed for each region.
    """
    
    ds_dict_region_mean = {}

    for experiment in ds_dict_region.keys():
        ds_dict_region_mean[experiment] = {}
        
        for ds_name, ds in ds_dict_region[experiment].items():
            ds_mean = xr.Dataset()  # Initiate an empty Dataset for the spatial means
            
            for var in ds:
                if 'region' in ds[var].dims:
                    # Compute the weighted spatial mean using compute_spatial_statistic function
                    spatial_mean = compute_spatial_statistic(ds[var], 'mean')
                    
                    # Add the spatial mean to the output Dataset
                    ds_mean[var] = spatial_mean
                    
                    # Preserve variable attributes
                    ds_mean[var].attrs = ds[var].attrs
            
            # Copy dataset attributes
            ds_mean.attrs.update(ds.attrs)
            
            # Add the modified dataset to the dictionary
            ds_dict_region_mean[experiment][ds_name] = ds_mean

    return ds_dict_region_mean

def compute_spatial_mean_with_subdivisions(ds_dict):
    """
    Computes the spatial mean for each region and subdivision in the datasets using weighted averaging.

    Args:
        ds_dict (dict): A dictionary of xarray datasets organized by experiments and models,
                        where each dataset has both region and subdivision dimensions added.

    Returns:
        dict: A new dictionary where keys are the same as in the input dictionary,
              and each value is an xarray Dataset with the weighted spatial mean computed
              for each region and subdivision.
    """
    
    ds_dict_mean = {}

    for experiment in ds_dict.keys():
        ds_dict_mean[experiment] = {}
        
        for ds_name, ds in ds_dict[experiment].items():
            ds_mean = xr.Dataset()  # Initiate an empty Dataset for the spatial means
            
            for var in ds:
                if 'region' in ds[var].dims and 'subdivision' in ds[var].dims:
                    # Compute the weighted spatial mean using compute_spatial_statistic function
                    spatial_mean = compute_spatial_statistic(ds[var], 'mean')
                    
                    # Add the spatial mean to the output Dataset
                    ds_mean[var] = spatial_mean
                    
                    # Preserve variable attributes
                    ds_mean[var].attrs = ds[var].attrs
            
            # Copy dataset attributes
            ds_mean.attrs.update(ds.attrs)
            
            # Add the modified dataset to the dictionary
            ds_dict_mean[experiment][ds_name] = ds_mean

    return ds_dict_mean
