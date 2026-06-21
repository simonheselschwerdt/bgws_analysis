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

def open_models_variables(BASE_DIR, data_state, experiment, temp_res, model, variables):
    """
    Open dataset of one model for different variables and merge them into one common xarray dataset.

    Parameters:
    - BASE_DIR: Path to base directory to pass it to open function.
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
        if model == 'LUH2':
            file = f'{data_state}/{experiment}/{temp_res}/{model}/{var}.nc'
        else:
            file = f'{data_state}/{experiment}/{temp_res}/{var}/{model}.nc'
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

def load_multiple_models_and_experiments(BASE_DIR, data_state, experiments, temp_res, models, variables):
    """
    Open datasets for multiple models and experiments in one xarray dictionary.

    Parameters:
    - BASE_DIR: Path to base directory to pass it to open function.
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
                ds, valid_vars = open_models_variables(BASE_DIR, data_state, experiment, temp_res, model_name, variables)
                if ds is not None:
                    ds_dict[experiment][model_name] = ds
                else:
                    print(f"No valid data found for model {model_name} in experiment {experiment} with variables {valid_vars}")
            except ValueError as e:
                print(f"Failed to load data for model {model_name} in experiment {experiment}: {e}")

    return ds_dict

def load_period_mean(BASE_DIR, data_state, experiments, models, variables, custom_periods=None, specific_months_or_seasons=None):
    """
    Load data in different temporal resolutions and compute period means.

    Parameters:
    - BASE_DIR: Path to base directory to pass it to open function.
    - data_state: State of the data to pass into the loader.
    - experiments: List of experiments to load e.g., historical, ssp370.
    - models: List of models to load.
    - variables: List of variables to load.
    - custom_periods: Optional dictionary to override default periods, e.g. {'historical': (1990, 2005)}

    Returns:
    - A dictionary containing the models with all variables for each experiment respectively, with computed period means.
    """
    month_variables = ['tas', 'pr', 'vpd', 'evspsbl', 'evapo', 'tran', 'mrro', 'mrso', 'mrsos', 'lai', 'gpp', 'wue', 'huss', 'ps', 'fracLut', 'treeFrac', 'cLeaf',  'cVeg', 'nVeg', 'cRoot', 'pr_no_landmask', 'evspsbl_no_landmask']
    year_variables = ['RX5day', 'rx5day_ratio']

    month_variables = [var for var in month_variables if var in variables]
    year_variables = [var for var in year_variables if var in variables]

    # Default periods (can be overridden by custom_periods)
    default_periods = {
        'historical': (1985, 2014),
        'hist-noLu': (1985, 2014),
        'ssp126': (2070, 2099),
        'ssp126-ssp370Lu': (2070, 2099),
        'ssp370': (2070, 2099),
        'ssp370-ssp126Lu': (2070, 2099),
        'ssp585': (2070, 2099)
    }

    ds_dict = {}

    for experiment in experiments:
        ds_dict[experiment] = {}

        # Get period (custom if provided, otherwise default)
        start_year, end_year = (custom_periods.get(experiment) 
                                if custom_periods and experiment in custom_periods 
                                else default_periods.get(experiment, (None, None)))

        # Process monthly variables
        if month_variables:
            print(f"Loading 'month' resolution variables {month_variables} for experiment '{experiment}'...")
            ds_month = load_multiple_models_and_experiments(BASE_DIR, data_state, [experiment], 'month', models, month_variables)

            if start_year and end_year:
                if any(model in ds_month[experiment] for model in models):
                    print(f"Selecting period {start_year}-{end_year} for 'month' variables in experiment '{experiment}'...")
                    ds_month_selected = pro_dat.select_period(ds_month[experiment], start_year=start_year, end_year=end_year, specific_months_or_seasons=specific_months_or_seasons)

                    print(f"Computing period mean for 'month' variables in experiment '{experiment}'...")
                    ds_dict[experiment]['month'] = comp_stats.compute_temporal_or_spatial_statistic(ds_month_selected, 'temporal', 'mean')

        # Process yearly variables
        if year_variables:
            print(f"Loading 'year' resolution variables {year_variables} for experiment '{experiment}'...")
            ds_year = load_multiple_models_and_experiments(BASE_DIR, data_state, [experiment], 'year', models, year_variables)

            if start_year and end_year:
                print(f"Selecting period {start_year}-{end_year} for 'year' variables in experiment '{experiment}'...")
                ds_year_selected = pro_dat.select_period(ds_year[experiment], start_year=start_year, end_year=end_year)

                print(f"Computing period mean for 'year' variables in experiment '{experiment}'...")
                ds_dict[experiment]['year'] = comp_stats.compute_temporal_or_spatial_statistic(ds_year_selected, 'temporal', 'mean')

        # Merge all datasets for each model
        print(f"Merging all datasets for experiment '{experiment}'...")
        merged_dict = {}
        for model in models:
            model_datasets = []
            if 'month' in ds_dict[experiment] and model in ds_dict[experiment]['month']:
                model_datasets.extend([ds_dict[experiment]['month'][model][var] for var in month_variables if var in ds_dict[experiment]['month'][model]])
            if 'year' in ds_dict[experiment] and model in ds_dict[experiment]['year']:
                model_datasets.extend([ds_dict[experiment]['year'][model][var] for var in year_variables if var in ds_dict[experiment]['year'][model]])

            if model_datasets:
                merged_ds = xr.merge(model_datasets)
                if model in ds_month.get(experiment, {}):
                    merged_ds.attrs = ds_month[experiment][model].attrs
                merged_dict[model] = merged_ds

        ds_dict[experiment] = merged_dict

    return ds_dict

def load_period_monthly_clim(BASE_DIR, data_state, experiments, models, variables, custom_periods=None):
    """
    Load data in different temporal resolutions and compute monthly climatologies.

    Parameters:
    - BASE_DIR: Path to base directory to pass it to open function.
    - data_state: State of the data to pass into the loader.
    - experiments: List of experiments to load e.g., historical, ssp370.
    - models: List of models to load.
    - variables: List of variables to load.
    - custom_periods: Optional dictionary to override default periods, e.g. {'historical': (1990, 2005)}

    Returns:
    - A dictionary containing the models with all variables for each experiment respectively, with computed period means or climatologies.
    """
    month_variables = ['tas', 'pr', 'vpd', 'evspsbl', 'evapo', 'tran', 'mrro', 'mrso', 'mrsos', 'lai', 'gpp', 'wue', 'huss', 'ps', 'fracLut', 'treeFrac', 'cLeaf',  'cVeg', 'nVeg', 'cRoot', 'pr_no_landmask', 'evspsbl_no_landmask']
    year_variables = ['RX5day', 'rx5day_ratio']

    month_variables = [var for var in month_variables if var in variables]
    year_variables = [var for var in year_variables if var in variables]

    default_periods = {
        'historical': (1985, 2014),
        'hist-noLu': (1985, 2014),
        'ssp126': (2070, 2099),
        'ssp126-ssp370Lu': (2070, 2099),
        'ssp370': (2070, 2099),
        'ssp370-ssp126Lu': (2070, 2099),
        'ssp585': (2070, 2099)
    }

    ds_dict = {}

    for experiment in experiments:
        ds_dict[experiment] = {}

        start_year, end_year = (custom_periods.get(experiment) 
                                if custom_periods and experiment in custom_periods 
                                else default_periods.get(experiment, (None, None)))

        # Process monthly variables
        if month_variables:
            print(f"Loading 'month' resolution variables {month_variables} for experiment '{experiment}'...")
            ds_month = load_multiple_models_and_experiments(BASE_DIR, data_state, [experiment], 'month', models, month_variables)

            if start_year and end_year:
                if any(model in ds_month[experiment] for model in models):
                    print(f"Selecting period {start_year}-{end_year} for 'month' variables in experiment '{experiment}'...")
                    selected = pro_dat.select_period(ds_month[experiment], start_year=start_year, end_year=end_year, specific_months_or_seasons=None)

                    # Compute monthly climatology
                    print("Computing monthly climatology (mean of each month across years)...")
                    climatology_dict = {}
                    for model, model_ds in selected.items():
                        monthly_means = {}
                        for var in month_variables:
                            if var in model_ds:
                                monthly_means[var] = model_ds[var].groupby('time.month').mean('time')
                        climatology_dict[model] = xr.Dataset(monthly_means)
                    ds_dict[experiment]['month'] = climatology_dict

        # Process yearly variables
        if year_variables:
            print(f"Loading 'year' resolution variables {year_variables} for experiment '{experiment}'...")
            ds_year = load_multiple_models_and_experiments(BASE_DIR, data_state, [experiment], 'year', models, year_variables)

            if start_year and end_year:
                print(f"Selecting period {start_year}-{end_year} for 'year' variables in experiment '{experiment}'...")
                ds_dict[experiment]['year'] = pro_dat.select_period(ds_year[experiment], start_year=start_year, end_year=end_year)

        # Merge all datasets for each model
        print(f"Merging all datasets for experiment '{experiment}'...")
        merged_dict = {}
        for model in models:
            model_datasets = []
            if 'month' in ds_dict[experiment] and model in ds_dict[experiment]['month']:
                model_datasets.extend([ds_dict[experiment]['month'][model][var] for var in month_variables if var in ds_dict[experiment]['month'][model]])
            if 'year' in ds_dict[experiment] and model in ds_dict[experiment]['year']:
                model_datasets.extend([ds_dict[experiment]['year'][model][var] for var in year_variables if var in ds_dict[experiment]['year'][model]])

            if model_datasets:
                merged_ds = xr.merge(model_datasets)
                if model in ds_month.get(experiment, {}):
                    merged_ds.attrs = ds_month[experiment][model].attrs
                merged_dict[model] = merged_ds

        ds_dict[experiment] = merged_dict

    return ds_dict


def load_period_seasonal_clim(BASE_DIR, data_state, experiments, models, variables, custom_periods=None):
    """
    Load data and compute seasonal climatologies (DJF, MAM, JJA, SON).

    Parameters:
    - BASE_DIR: Path to base directory to pass to open function.
    - data_state: State of the data to pass into the loader.
    - experiments: List of experiments to load e.g., historical, ssp370.
    - models: List of models to load.
    - variables: List of variables to load.
    - custom_periods: Optional dictionary to override default periods, e.g. {'historical': (1990, 2005)}

    Returns:
    - A dictionary containing seasonal climatologies for each model and experiment.
    """
    month_variables = ['tas', 'pr', 'vpd', 'evspsbl', 'evapo', 'tran', 'mrro', 'mrso', 'mrsos', 'lai', 'gpp', 'wue', 'huss', 'ps', 'fracLut', 'treeFrac', 'cLeaf',  'cVeg', 'nVeg', 'cRoot', 'pr_no_landmask', 'evspsbl_no_landmask']
    year_variables = ['RX5day', 'rx5day_ratio']

    month_variables = [var for var in month_variables if var in variables]
    year_variables = [var for var in year_variables if var in variables]

    default_periods = {
        'historical': (1985, 2014),
        'hist-noLu': (1985, 2014),
        'ssp126': (2070, 2099),
        'ssp126-ssp370Lu': (2070, 2099),
        'ssp370': (2070, 2099),
        'ssp370-ssp126Lu': (2070, 2099),
        'ssp585': (2070, 2099)
    }

    ds_dict = {}

    for experiment in experiments:
        ds_dict[experiment] = {}

        start_year, end_year = (custom_periods.get(experiment)
                                if custom_periods and experiment in custom_periods
                                else default_periods.get(experiment, (None, None)))

        # Load monthly variables
        if month_variables:
            print(f"Loading monthly resolution variables {month_variables} for experiment '{experiment}'...")
            ds_month = load_multiple_models_and_experiments(BASE_DIR, data_state, [experiment], 'month', models, month_variables)

            if start_year and end_year:
                if any(model in ds_month[experiment] for model in models):
                    print(f"Selecting period {start_year}-{end_year} for experiment '{experiment}'...")
                    selected = pro_dat.select_period(ds_month[experiment], start_year=start_year, end_year=end_year)

                    # Compute seasonal climatology
                    print("Computing seasonal climatology (DJF, MAM, JJA, SON)...")
                    seasonal_climatology_dict = {}
                    for model, model_ds in selected.items():
                        seasonal_means = {}
                        for var in month_variables:
                            if var in model_ds:
                                seasonal_means[var] = model_ds[var].groupby('time.season').mean('time')
                        seasonal_climatology_dict[model] = xr.Dataset(seasonal_means)

                    ds_dict[experiment]['season'] = seasonal_climatology_dict

        # Load yearly variables (if any)
        if year_variables:
            print(f"Loading yearly resolution variables {year_variables} for experiment '{experiment}'...")
            ds_year = load_multiple_models_and_experiments(BASE_DIR, data_state, [experiment], 'year', models, year_variables)

            if start_year and end_year:
                print(f"Selecting period {start_year}-{end_year} for yearly variables in experiment '{experiment}'...")
                ds_dict[experiment]['year'] = pro_dat.select_period(ds_year[experiment], start_year=start_year, end_year=end_year)

        # Merge datasets
        print(f"Merging all datasets for experiment '{experiment}'...")
        merged_dict = {}
        for model in models:
            model_datasets = []
            if 'season' in ds_dict[experiment] and model in ds_dict[experiment]['season']:
                model_datasets.extend([ds_dict[experiment]['season'][model][var]
                                       for var in month_variables
                                       if var in ds_dict[experiment]['season'][model]])
            if 'year' in ds_dict[experiment] and model in ds_dict[experiment]['year']:
                model_datasets.extend([ds_dict[experiment]['year'][model][var]
                                       for var in year_variables
                                       if var in ds_dict[experiment]['year'][model]])

            if model_datasets:
                merged_ds = xr.merge(model_datasets)
                if model in ds_month.get(experiment, {}):
                    merged_ds.attrs = ds_month[experiment][model].attrs
                merged_dict[model] = merged_ds

        ds_dict[experiment] = merged_dict

    return ds_dict

def load_period_seasonal_series(BASE_DIR, data_state, experiments, models, variables, custom_periods=None):
    """
    Load data and compute seasonal *time series* (DJF, MAM, JJA, SON) via quarterly means,
    not climatologies. Each variable keeps a 'time' axis at seasonal resolution and carries a
    'season' label and 'season_year' coordinate (DJF assigned to the Jan–Feb year).

    Returns:
      ds_dict[experiment][model] = xr.Dataset of requested vars with seasonal time series.
      (Yearly vars remain yearly, as before.)
    """
    import xarray as xr
    import numpy as np

    month_variables_all = ['tas','pr','vpd','evspsbl','evapo','tran','mrro','mrso','mrsos','lai','gpp','wue','huss','ps','fracLut','treeFrac','cLeaf','cVeg','nVeg','cRoot', 'pr_no_landmask', 'evspsbl_no_landmask']
    year_variables_all  = ['RX5day','rx5day_ratio']

    month_variables = [v for v in month_variables_all if v in variables]
    year_variables  = [v for v in year_variables_all  if v in variables]

    default_periods = {
        'historical': (1985, 2014),
        'hist-noLu': (1985, 2014),
        'ssp126': (2070, 2099),
        'ssp126-ssp370Lu': (2070, 2099),
        'ssp370': (2070, 2099),
        'ssp370-ssp126Lu': (2070, 2099),
        'ssp585': (2070, 2099),
    }

    ds_dict = {}

    # helper: make seasonal series (quarterly means, season labels, season_year)
    def to_seasonal_series(da):
        # Quarterly means with quarters starting in December → DJF, MAM, JJA, SON
        q = da.resample(time='QS-DEC').mean('time')

        # season label from quarter start month
        m = q['time'].dt.month
        season = xr.where(
            m == 12, 'DJF',
            xr.where(m == 3, 'MAM', xr.where(m == 6, 'JJA', 'SON'))
        )
        # assign DJF to the following year (Dec 1999 → 2000)
        season_year = q['time'].dt.year + xr.where(m == 12, 1, 0)

        q = q.assign_coords(season=('time', season.data),
                            season_year=('time', season_year.data))
        return q

    for experiment in experiments:
        ds_dict[experiment] = {}

        start_year, end_year = (
            custom_periods[experiment] if (custom_periods and experiment in custom_periods)
            else default_periods.get(experiment, (None, None))
        )

        # ---------- monthly → seasonal series ----------
        if month_variables:
            print(f"Loading monthly variables {month_variables} for '{experiment}'...")
            ds_month = load_multiple_models_and_experiments(
                BASE_DIR, data_state, [experiment], 'month', models, month_variables
            )

            if start_year and end_year and any(m in ds_month[experiment] for m in models):
                print(f"Selecting {start_year}-{end_year} for '{experiment}' (monthly)...")
                selected = pro_dat.select_period(ds_month[experiment],
                                                 start_year=start_year, end_year=end_year)

                print("Computing seasonal *time series* (DJF, MAM, JJA, SON)...")
                seasonal_dict = {}
                for model, model_ds in selected.items():
                    series = {}
                    for var in month_variables:
                        if var in model_ds:
                            series[var] = to_seasonal_series(model_ds[var])
                    seasonal_dict[model] = xr.Dataset(series)
                ds_month_seasonal = seasonal_dict
            else:
                ds_month_seasonal = {}

        # ---------- yearly variables ----------
        if year_variables:
            print(f"Loading yearly variables {year_variables} for '{experiment}'...")
            ds_year = load_multiple_models_and_experiments(
                BASE_DIR, data_state, [experiment], 'year', models, year_variables
            )
            if start_year and end_year:
                print(f"Selecting {start_year}-{end_year} for '{experiment}' (yearly)...")
                ds_year_sel = pro_dat.select_period(ds_year[experiment],
                                                    start_year=start_year, end_year=end_year)
            else:
                ds_year_sel = {}
        else:
            ds_year_sel = {}

        # ---------- merge per model ----------
        print(f"Merging datasets for '{experiment}'...")
        merged = {}
        for model in models:
            parts = []

            if month_variables and model in ds_month_seasonal:
                parts.extend([ds_month_seasonal[model][v]
                              for v in month_variables
                              if v in ds_month_seasonal[model]])

            if year_variables and model in ds_year_sel:
                parts.extend([ds_year_sel[model][v]
                              for v in year_variables
                              if v in ds_year_sel[model]])

            if parts:
                md = xr.merge(parts)
                # attach attrs from original monthly if available
                if month_variables and experiment in ds_month and model in ds_month[experiment]:
                    md.attrs = ds_month[experiment][model].attrs
                merged[model] = md

        ds_dict[experiment] = merged

    return ds_dict


def load_period(BASE_DIR, data_state, experiments, models, variables, custom_periods=None):
    """
    Load data in different temporal resolutions and compute period means.

    Parameters:
    - BASE_DIR: Path to base directory to pass it to open function.
    - data_state: State of the data to pass into the loader.
    - experiments: List of experiments to load e.g., historical, ssp370.
    - models: List of models to load.
    - variables: List of variables to load.
    - custom_periods: Optional dictionary to override default periods, e.g. {'historical': (1990, 2005)}

    Returns:
    - A dictionary containing the models with all variables for each experiment respectively, with computed period means.
    """
    month_variables = ['tas', 'pr', 'vpd', 'evspsbl', 'evapo', 'tran', 'mrro', 'mrso', 'mrsos', 'lai', 'gpp', 'wue', 'huss', 'ps', 'fracLut', 'treeFrac', 'cLeaf', 'cVeg', 'nVeg', 'cRoot', 'pr_no_landmask', 'evspsbl_no_landmask']
    year_variables = ['RX5day', 'rx5day_ratio']

    month_variables = [var for var in month_variables if var in variables]
    year_variables = [var for var in year_variables if var in variables]

    # Default periods (can be overridden by custom_periods)
    default_periods = {
        'historical': (1985, 2014),
        'hist-noLu': (1985, 2014),
        'ssp126': (2070, 2099),
        'ssp126-ssp370Lu': (2070, 2099),
        'ssp370': (2070, 2099),
        'ssp370-ssp126Lu': (2070, 2099),
        'ssp585': (2070, 2099)
    }

    ds_dict = {}

    for experiment in experiments:
        ds_dict[experiment] = {}

        # Get period (custom if provided, otherwise default)
        start_year, end_year = (custom_periods.get(experiment) 
                                if custom_periods and experiment in custom_periods 
                                else default_periods.get(experiment, (None, None)))

        # Process monthly variables
        if month_variables:
            print(f"Loading 'month' resolution variables {month_variables} for experiment '{experiment}'...")
            ds_month = load_multiple_models_and_experiments(BASE_DIR, data_state, [experiment], 'month', models, month_variables)

            if start_year and end_year:
                if any(model in ds_month[experiment] for model in models):
                    print(f"Selecting period {start_year}-{end_year} for 'month' variables in experiment '{experiment}'...")
                    ds_dict[experiment]['month'] = pro_dat.select_period(ds_month[experiment], start_year=start_year, end_year=end_year, specific_months_or_seasons=None)

        # Process yearly variables
        if year_variables:
            print(f"Loading 'year' resolution variables {year_variables} for experiment '{experiment}'...")
            ds_year = load_multiple_models_and_experiments(BASE_DIR, data_state, [experiment], 'year', models, year_variables)

            if start_year and end_year:
                print(f"Selecting period {start_year}-{end_year} for 'year' variables in experiment '{experiment}'...")
                ds_dict[experiment]['year'] = pro_dat.select_period(ds_year[experiment], start_year=start_year, end_year=end_year)

        # Merge all datasets for each model
        print(f"Merging all datasets for experiment '{experiment}'...")
        merged_dict = {}
        for model in models:
            model_datasets = []
            if 'month' in ds_dict[experiment] and model in ds_dict[experiment]['month']:
                model_datasets.extend([ds_dict[experiment]['month'][model][var] for var in month_variables if var in ds_dict[experiment]['month'][model]])
            if 'year' in ds_dict[experiment] and model in ds_dict[experiment]['year']:
                model_datasets.extend([ds_dict[experiment]['year'][model][var] for var in year_variables if var in ds_dict[experiment]['year'][model]])

            if model_datasets:
                merged_ds = xr.merge(model_datasets)
                if model in ds_month.get(experiment, {}):
                    merged_ds.attrs = ds_month[experiment][model].attrs
                merged_dict[model] = merged_ds

        ds_dict[experiment] = merged_dict

    return ds_dict


def load_period_mean_LUH2(BASE_DIR, data_state, experiments, models, variables):
    """
    Load data in different temporal resolutions and compute period means.

    Parameters:
    - BASE_DIR: Path to base directory to pass it to open function.
    - experiments: List of experiments to load e.g., historical, ssp370.
    - models: List of models to load.
    - variables: List of variables to load.

    Returns:
    - A dictionary containing the models with all variables for each experiment respectively, with computed period means.
    """

    # Define periods for historical and ssp370 experiments
    periods = {
        'historical': (1985, 2014),
        'ssp126': (2070, 2099),
        'ssp370': (2070, 2099),
        'ssp585': (2070, 2099)
    }

    # Initialize dictionary to store datasets
    ds_dict = {}

    for experiment in experiments:
        ds_dict[experiment] = {}

        print(f"Loading variables {variables} for experiment '{experiment}'...")
        ds_year = load_multiple_models_and_experiments(BASE_DIR, data_state, [experiment], 'year', models, variables)

        start_year, end_year = periods.get(experiment, (None, None))
        if start_year and end_year:
            print(f"Selecting period {start_year}-{end_year} for variables in experiment '{experiment}'...")
            ds_year_selected = pro_dat.select_period(ds_year[experiment], start_year=start_year, end_year=end_year)

            print(f"Computing period mean for variables in experiment '{experiment}'...")
            ds_dict[experiment]['year'] = comp_stats.compute_temporal_or_spatial_statistic(ds_year_selected, 'temporal', 'mean')

        # Merge all datasets for each model
        print(f"Merging all datasets for experiment '{experiment}'...")
        merged_dict = {}
        for model in models:
            model_datasets = []
            model_datasets.extend([ds_dict[experiment]['year'][model][var] for var in variables if var in ds_dict[experiment]['year'][model]])

            # Merge all datasets for this model
            if model_datasets:
                merged_ds = xr.merge(model_datasets)
                # Preserve the original model attributes
                #if model in ds_month.get(experiment, {}):
                #    merged_ds.attrs = ds_month[experiment][model].attrs
                merged_dict[model] = merged_ds

        ds_dict[experiment] = merged_dict

    return ds_dict

def load_cumulative_transitions_LUH2(BASE_DIR, data_state, experiments, models, variables, start_year=1850, end_year=2014):
    """
    Load LUH2 transition variables and compute cumulative transitions over a given period.

    Parameters:
    - BASE_DIR: Path to base directory
    - data_state: e.g., 'transitions'
    - experiments: List of experiments to load (e.g., ['historical'])
    - models: List of models to load (usually ['LUH2'])
    - variables: List of transition variables to sum (e.g., forest_to_cropland transitions)
    - start_year: Beginning of accumulation period
    - end_year: End of accumulation period

    Returns:
    - A dictionary with cumulative transition datasets per model per experiment
    """

    ds_dict = {}

    for experiment in experiments:
        ds_dict[experiment] = {}

        print(f"Loading transition variables {variables} for experiment '{experiment}'...")
        ds_year = load_multiple_models_and_experiments(BASE_DIR, data_state, [experiment], 'year', models, variables)

        print(f"Selecting period {start_year}-{end_year} for variables in experiment '{experiment}'...")
        ds_year_selected = pro_dat.select_period(ds_year[experiment], start_year=start_year, end_year=end_year)

        print(f"Computing cumulative sums for variables in experiment '{experiment}'...")
        merged_dict = {}
        for model in models:
            model_datasets = []
            for var in variables:
                if var in ds_year_selected[model]:
                    # Sum over time and multiply by 100 to get % of grid cell
                    cum = ds_year_selected[model][var].sum(dim="time", skipna=False) * 100
                    cum.name = var
                    model_datasets.append(cum)
            if model_datasets:
                merged_ds = xr.merge(model_datasets)
                merged_dict[model] = merged_ds

        ds_dict[experiment] = merged_dict

    return ds_dict

    
# Set Dask configuration to avoid large chunk creation
dask.config.set(**{'array.slicing.split_large_chunks': True})

def find_models_with_all_variables(cat, required_variables, required_experiments):
    matching_models = None

    for experiment in required_experiments:
        experiment_matching_models = None

        for var, table in required_variables.items():
            search_results = cat.search(variable_id=var, table_id=table, experiment_id=experiment)

            if search_results.df.empty:
                print(f"No data for variable '{var}' in experiment '{experiment}'")
                return set()

            models_with_var = set(search_results.df["source_id"].explode().unique())

            if experiment_matching_models is None:
                experiment_matching_models = models_with_var
            else:
                experiment_matching_models.intersection_update(models_with_var)

        if matching_models is None:
            matching_models = experiment_matching_models
        else:
            matching_models.intersection_update(experiment_matching_models)

    return matching_models or set()


def find_models_and_members(cat, required_variables, required_experiments, valid_models):
    model_member_map = {model: None for model in valid_models}

    for experiment in required_experiments:
        for var, table in required_variables.items():
            search_results = cat.search(variable_id=var, table_id=table, experiment_id=experiment)

            if search_results.df.empty:
                continue

            df = search_results.df.dropna(subset=["source_id", "member_id"])

            for model in valid_models:
                model_df = df[df["source_id"] == model]
                if model_df.empty:
                    continue

                members = set(model_df["member_id"].explode().unique())

                # Store members that exist across all experiments & variables
                if model_member_map[model] is None:
                    model_member_map[model] = members
                else:
                    model_member_map[model].intersection_update(members)

    # Remove models with no valid member left
    return {model: members for model, members in model_member_map.items() if members}

def load_model_data(cat, selected_model, selected_scenario, selected_vars, selected_member):
    """
    Loads data for a selected model, scenario, and specific variables.
    Loads each variable separately and merges them into one dataset.
    """
    ds_dict = {}
    dataset_parts = {}

    print(f"\nLoading data for model: {selected_model}, scenario: {selected_scenario}, member: {selected_member}")

    if selected_model == 'MIROC-ES2L' and selected_scenario == 'ssp126':
        for var in list(selected_vars.keys()):
            ds_dict[selected_model]=xr.open_dataset(f"/pool/data/CMIP6/data/ScenarioMIP/MIROC/MIROC-ES2L/ssp126/r1i1p1f2/{selected_vars[var]}/{var}/gn/v20190823/{var}_{selected_vars[var]}_MIROC-ES2L_ssp126_r1i1p1f2_gn_201501-210012.nc")

    else:
        for var, table in selected_vars.items():
            print(f"Searching for variable '{var}' in model '{selected_model}', scenario '{selected_scenario}'")
    
            # Search for the dataset
            search_results = cat.search(
                variable_id=var,
                table_id=table,
                experiment_id=selected_scenario,
                source_id=selected_model,
                member_id=selected_member
            )
    
            if search_results.df.empty:
                print(f"No data found for '{var}' in scenario '{selected_scenario}' for model '{selected_model}'. Skipping...")
                continue
    
            # Convert to xarray dataset
            with dask.config.set(use_cftime=True, decode_times=True):
                datasets = search_results.to_dataset_dict(add_measures=False)
    
            # Store dataset in dictionary
            if datasets:
                for key, ds in datasets.items():
                    dataset_parts[var] = ds
                    print(f"Loaded dataset for '{var}' with shape {ds[var].shape}")
    
        # Merge datasets into a single dataset per model
        if dataset_parts:
            ds_dict[selected_model] = xr.merge(dataset_parts.values(), compat="override")
            print(f"\nSuccessfully merged datasets for model '{selected_model}', scenario '{selected_scenario}', member '{selected_member}'.")
        else:
            print(f"No valid datasets found for model '{selected_model}', scenario '{selected_scenario}', member '{selected_member}'.")

    return ds_dict