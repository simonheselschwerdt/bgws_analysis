"""
src/data_handling/load_data.py

This script provides functions to load data in NetCDF format.
Users can select different Earth system models, variables, and temporal resolutions
such as monthly or seasonal data.

Functions:
- open_dataset: Load a NetCDF file into an xarray Dataset.
- open_models_variables: Load mutiple variables into the same dataset for one model.
- load_multiple_models_and_experiments: Load multiple models and experiments in dictionaries.
- load_period_mean: Load the period mean for multiple models and experiments in one dictionary.

Usage:
    Import this module in your scripts to load the preprocessed data for analysis.
"""

import os
import xarray as xr
import copy
import numpy as np
import multiprocessing as mp
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
import process_data as pro_dat
import compute_statistics as comp_stats
from xmip.preprocessing import correct_lon, correct_units, parse_lon_lat_bounds, maybe_convert_bounds_to_vertex, maybe_convert_vertex_to_bounds
import intake

import sys
config_dir = '../../src'
sys.path.append(config_dir)
from config import DATA_DIR

# Intake

def open_catalog(catalog_path):
    return intake.open_esm_datastore(catalog_path)

def pre_preprocessing(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocesses a CMIP6 dataset
    
    Parameters:
    ds (xr.Dataset): Input dataset
    
    Returns:
    xr.Dataset: Preprocessed dataset
    """
    
    def correct_coordinates(ds: xr.Dataset) -> xr.Dataset:
        """
        Corrects wrongly assigned data_vars to coordinates

        Parameters:
        ds (xr.Dataset): Input dataset

        Returns:
        xr.Dataset: Dataset with corrected coordinates
        """
        for co in ["lon", "lat"]:
            if co in ds.variables:
                ds = ds.set_coords(co)

        return ds.copy(deep=True)
 
    ds = correct_coordinates(ds)
    ds = correct_units(ds) 
    ds = parse_lon_lat_bounds(ds)
    ds = maybe_convert_bounds_to_vertex(ds)
    ds = maybe_convert_vertex_to_bounds(ds)
    return ds.copy(deep=True)

def drop_redundant(ds_dict, drop_list): 
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

def replace_coordinates(new_coords, replace_coords):
    """
    Helper funtion to replace coordinates before merging.
    
    Args:
        new_coords (xr dataset): A dictionary of xarray datasets which gives the new coordinates.
        replace_coords (xr dataset): A dictionary of xarray datasets which coordinates will be replaced.

    Returns:
        replace_coords (xr dataset): The replace dictionary with the new coordinates copied from new_coords.
    """
    
    for coord in ['lon', 'lat', 'time']:
        if not new_coords[coord].equals(replace_coords[coord]):
            replace_coords[coord] = new_coords[coord]
        else:
            pass
    
    return replace_coords

### CHECK BELOW FUNC IF REALLY NEEDED

def open_dataset(filepath):
    """
    Open dataset based on input filepath.

    Parameters:
    - filepath: Path to file with filename.

    Returns:
    - A dataset with loaded file.
    """
    ds = xr.open_dataset(filepath, chunks='auto')
    return ds

def open_models_variables(BASE_DIR, data_state, data_product, experiment, temp_res, model, variables):
    """
    Open dataset of one model for different variables and merge them into one common xarray dataset.

    Parameters:
    - BASE_DIR: Path to base directory to pass it to open function.
    - data_product: e.g. CMIP6 ERA5 CMIP5 CORDEX
    - experiment: e.g. historical ssp370 
    - temp_res: e.g. day month season year period
    - model: e.g. CAMS-CSM1-0 CESM2-WACCM CNRM-ESM2-1 GISS-E2-1-G MIROC-ES2L NorESM2-MM
    - variables: List of different variables e.g. [pr, lai, tas]

    Returns:
    - A dataset with all available variables of one model loaded in one xarray dataset.
    """
    filepaths = []
    valid_vars = []

    for var in variables:
        file = f'{data_state}/{data_product}/{experiment}/{temp_res}/{var}/{model}.nc'
        file_path = os.path.join(BASE_DIR, file)
        if os.path.exists(file_path):
            filepaths.append(file_path)
            valid_vars.append(var)
        else:
            print(f"No file found for variable '{var}' in model '{model}'.")

    datasets = [open_dataset(fp) for fp in filepaths]
    if datasets:
        ds = xr.merge(datasets)
        return ds, valid_vars
    else:
        return None, valid_vars

def load_multiple_models_and_experiments(BASE_DIR, data_state, data_product, experiments, temp_res, models, variables):
    """
    Open datasets for multiple models and experiments in one xarray dictionary.

    Parameters:
    - BASE_DIR: Path to base directory to pass it to open function.
    - data_product: Data product e.g. CMIP6 to pass it to open function.
    - experiments: List of experiments to load e.g. historical ssp370.
    - temp_res: e.g. day month season year period
    - models: List of models to load.
    - variables: List of variables to load.
    
    Returns:
    - A xarray dictionary containing the models with all variables for each experiment respectively.
    """

    # Initialize the dictionary to store datasets
    ds_dict = {}

    # Loop over each experiment and model name to load and merge datasets
    for experiment in experiments:
        ds_dict[experiment] = {}
        for model_name in models:
            try:
                ds, valid_vars = open_models_variables(BASE_DIR, data_state, data_product, experiment, temp_res, model_name, variables)
                if ds is not None:
                    ds_dict[experiment][model_name] = ds
                else:
                    print(f"No valid data found for model {model_name} in experiment {experiment} with variables {valid_vars}")
            except ValueError as e:
                print(f"Failed to load data for model {model_name} in experiment {experiment}: {e}")

    return ds_dict

def load_period_mean(BASE_DIR, data_state, data_product, experiments, models, variables):
    """
    Load data in different temporal resolutions and compute period means.

    Parameters:
    - BASE_DIR: Path to base directory to pass it to open function.
    - data_product: Data product e.g. CMIP6 to pass it to open function.
    - experiments: List of experiments to load e.g. historical ssp370.
    - models: List of models to load.
    - variables: List of variables to load.

    Returns:
    - A dictionary containing the models with all variables for each experiment respectively, with computed period means.
    """

    month_variables = ['tas', 'pr', 'vpd', 'evspsbl', 'evapo', 'tran', 'mrro', 'mrso', 'lai', 'gpp', 'wue', 'huss', 'ps']
    year_variable = 'RX5day'

    # Filter the variables based on the input
    month_variables = [var for var in month_variables if var in variables]
    year_variable = year_variable if year_variable in variables else None

    # Define periods for historical and ssp370 experiments
    periods = {
        'historical': (1985, 2014),
        'ssp370': (2071, 2100)
    }

    # Initialize dictionary to store datasets
    ds_dict = {}

    for experiment in experiments:
        ds_dict[experiment] = {}

        # Process month variables
        if month_variables:
            print(f"Loading 'month' resolution variables {month_variables} for experiment '{experiment}'...")
            ds_month = load_multiple_models_and_experiments(BASE_DIR, data_state, data_product, [experiment], 'month', models, month_variables)
            
            start_year, end_year = periods.get(experiment, (None, None))
            if start_year and end_year:
                if any(model in ds_month[experiment] for model in models):
                    print(f"Selecting period {start_year}-{end_year} for 'month' variables in experiment '{experiment}'...")
                    ds_month_selected = pro_dat.select_period(ds_month[experiment], start_year=start_year, end_year=end_year)

                    print(f"Computing period mean for 'month' variables in experiment '{experiment}'...")
                    ds_dict[experiment]['month'] = comp_stats.compute_temporal_or_spatial_statistic(ds_month_selected, 'temporal', 'mean')

        # Process year variable
        if year_variable:
            print(f"Loading 'year' resolution variable '{year_variable}' for experiment '{experiment}'...")
            ds_year = load_multiple_models_and_experiments(BASE_DIR, data_state, data_product, [experiment], 'year', models, [year_variable])

            print(f"Computing period mean for 'year' variable in experiment '{experiment}'...")
            ds_dict[experiment]['year'] = comp_stats.compute_temporal_or_spatial_statistic(ds_year[experiment], 'temporal', 'mean')

        # Merge all datasets for each model
        print(f"Merging all datasets for experiment '{experiment}'...")
        merged_dict = {}
        for model in models:
            model_datasets = []
            if 'month' in ds_dict[experiment] and model in ds_dict[experiment]['month']:
                model_datasets.extend([ds_dict[experiment]['month'][model][var] for var in month_variables if var in ds_dict[experiment]['month'][model]])
            if 'year' in ds_dict[experiment] and model in ds_dict[experiment]['year']:
                model_datasets.append(ds_dict[experiment]['year'][model][year_variable])
    
            if model_datasets:
                merged_ds = xr.merge(model_datasets)
                # Preserve the original model attributes
                if model in ds_month[experiment]:
                    merged_ds.attrs = ds_month[experiment][model].attrs
                merged_dict[model] = merged_ds

        ds_dict[experiment] = merged_dict

    return ds_dict
    
# Set Dask configuration to avoid large chunk creation
dask.config.set(**{'array.slicing.split_large_chunks': True})