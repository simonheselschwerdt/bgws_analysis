"""
src/data_handling/load_data.py

Funtions:
- open_dataset
- open_models_variables
- load_multiple_models_and_experiments
- load_period_mean_LUH2
- load_period_seasonal_clim

Author: Simon P. Heselschwerdt
Date: 2026-02-26
"""

import os
import xarray as xr
import copy
import numpy as np
import process_data as pro_dat
import compute_statistics as comp_stats
import dask
from dask.diagnostics import ProgressBar

import sys
config_dir = '../../src'
sys.path.append(config_dir)
from config import DATA_DIR

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
            file = f"{data_state}/{experiment}/{temp_res}/{model}/{var}.nc"
        else:
            file = f"{data_state}/{experiment}/{temp_res}/{var}/{model}.nc"
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

def temporal_stats_dict(ds_by_model: dict, stats=("mean",), dim="time", suffix=True):
    """
    Compute multiple temporal statistics along `dim` for each model dataset.

    Parameters
    ----------
    ds_by_model : dict[str, xr.Dataset]
    stats : iterable[str]
        any of: "mean", "std", "median"
    dim : str
        usually "time"
    suffix : bool
        if True -> output variables have suffix: <var>_<stat>
        if False -> keep same var names but you'd need separate dicts per stat

    Returns
    -------
    dict[str, xr.Dataset]
    """
    out = {}
    for model, ds in ds_by_model.items():
        pieces = []
        for stat in stats:
            if stat == "mean":
                ds_stat = ds.mean(dim=dim, skipna=True)
            elif stat == "std":
                # std over time, skipna
                ds_stat = ds.std(dim=dim, skipna=True)
            elif stat == "median":
                ds_stat = ds.median(dim=dim, skipna=True)
            else:
                raise ValueError(f"Unsupported stat: {stat}")

            if suffix:
                ds_stat = ds_stat.rename({v: f"{v}_{stat}" for v in ds_stat.data_vars})

            ds_stat.attrs = ds.attrs.copy()
            ds_stat.attrs["temporal_stat"] = stat
            pieces.append(ds_stat)

        out[model] = xr.merge(pieces, compat="override")

    return out

def load_period_stats(
    BASE_DIR,
    data_state,
    experiments,
    models,
    variables,
    custom_periods=None,
    specific_months_or_seasons=None,
    temporal_stats=("mean",),
    experiment_periods=None,   # <-- NEW
):
    """
    Load data and compute period stats.

    NEW:
      experiment_periods:
        dict[source_experiment] -> dict[output_key] = (start_year, end_year)
      This allows e.g. ssp126 to be loaded once but sliced into NF/FF outputs.
    """

    month_variables = [
        'tas', 'pr', 'vpd', 'evspsbl', 'evapo', 'tran', 'mrro', 'mrso', 'mrsos', 'lai', 'gpp', 'wue',
        'huss', 'ps', 'fracLut', 'treeFrac', 'cropFrac', 'cLeaf', 'cVeg', 'nVeg', 'cRoot',
        'pr_no_landmask', 'evspsbl_no_landmask', 'rsds', 'clt', "evspsblsoi", "evspsblveg", "evspsblpot"
    ]
    year_variables = ['RX5day', 'RX1day', 'rx5day_ratio']

    month_variables = [v for v in month_variables if v in variables]
    year_variables  = [v for v in year_variables if v in variables]

    default_periods = {
        'historical': (1985, 2014),
        'hist-noLu': (1985, 2014),
        'ssp126': (2070, 2099),
        'ssp126-ssp370Lu': (2070, 2099),
        'ssp370': (2070, 2099),
        'ssp370-ssp126Lu': (2070, 2099),
        'ssp585': (2070, 2099)
    }

    # ----------------------------
    # Build period plan
    # ----------------------------
    # If experiment_periods not provided, fall back to old behavior:
    # each experiment -> itself with one period.
    if experiment_periods is None:
        experiment_periods = {}
        for exp in experiments:
            start_year, end_year = (custom_periods.get(exp)
                                    if custom_periods and exp in custom_periods
                                    else default_periods.get(exp, (None, None)))
            experiment_periods[exp] = {exp: (start_year, end_year)}

    # Ensure we only load source experiments we actually need
    source_experiments = list(experiment_periods.keys())

    # ----------------------------
    # Load raw once per source experiment
    # ----------------------------
    ds_month_raw = {}
    ds_year_raw  = {}

    if month_variables:
        print(f"Loading 'month' vars {month_variables} for {source_experiments} ...")
        tmp = load_multiple_models_and_experiments(BASE_DIR, data_state, source_experiments, "month", models, month_variables)
        # tmp is expected as tmp[exp][model] = ds
        ds_month_raw = tmp

    if year_variables:
        print(f"Loading 'year' vars {year_variables} for {source_experiments} ...")
        tmp = load_multiple_models_and_experiments(BASE_DIR, data_state, source_experiments, "year", models, year_variables)
        ds_year_raw = tmp

    # ----------------------------
    # Produce output dict keyed by output_key (historical, ssp126_nf, ...)
    # ----------------------------
    out = {}

    for src_exp, out_periods in experiment_periods.items():
        for out_key, (start_year, end_year) in out_periods.items():

            out[out_key] = {}
            month_stats = {}
            year_stats  = {}

            # ---- month ----
            if month_variables and src_exp in ds_month_raw and ds_month_raw[src_exp]:
                if start_year and end_year:
                    ds_sel = pro_dat.select_period(
                        ds_month_raw[src_exp],
                        start_year=start_year, end_year=end_year,
                        specific_months_or_seasons=specific_months_or_seasons
                    )
                    month_stats = temporal_stats_dict(ds_sel, stats=temporal_stats, dim="time", suffix=True)

            # ---- year ----
            if year_variables and src_exp in ds_year_raw and ds_year_raw[src_exp]:
                if start_year and end_year:
                    ds_sel = pro_dat.select_period(
                        ds_year_raw[src_exp],
                        start_year=start_year, end_year=end_year
                    )
                    year_stats = temporal_stats_dict(ds_sel, stats=temporal_stats, dim="time", suffix=True)

            # ---- merge month+year per model ----
            for model in models:
                pieces = []
                if model in month_stats:
                    pieces.append(month_stats[model])
                if model in year_stats:
                    pieces.append(year_stats[model])

                if not pieces:
                    continue

                merged = xr.merge(pieces, compat="override")

                # attrs: prefer month raw attrs if present
                if month_variables and (src_exp in ds_month_raw) and (model in ds_month_raw[src_exp]):
                    merged.attrs = ds_month_raw[src_exp][model].attrs
                elif year_variables and (src_exp in ds_year_raw) and (model in ds_year_raw[src_exp]):
                    merged.attrs = ds_year_raw[src_exp][model].attrs

                out[out_key][model] = merged

    return out

def test():
    """
    def load_period_stats(BASE_DIR, data_state, experiments, models, variables, custom_periods=None, specific_months_or_seasons=None, temporal_stats=("mean",)):
        
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
        
        month_variables = ['tas', 'pr', 'vpd', 'evspsbl', 'evapo', 'tran', 'mrro', 'mrso', 'mrsos', 'lai', 'gpp', 'wue', 'huss', 'ps', 'fracLut', 'treeFrac', 'cLeaf',  'cVeg', 'nVeg', 'cRoot', 'pr_no_landmask', 'evspsbl_no_landmask',  'rsds', 'clt']
        year_variables = ['RX5day', 'RX1day', 'rx5day_ratio']
    
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
    
            start_year, end_year = (custom_periods.get(experiment)
                                    if custom_periods and experiment in custom_periods
                                    else default_periods.get(experiment, (None, None)))
    
            # -----------------
            # MONTH
            # -----------------
            ds_month = {}  # define so attrs copy won't crash later
            if month_variables:
                print(f"Loading 'month' resolution variables {month_variables} for experiment '{experiment}'...")
                ds_month = load_multiple_models_and_experiments(BASE_DIR, data_state, [experiment], "month", models, month_variables)
    
                if start_year and end_year and experiment in ds_month and ds_month[experiment]:
                    print(f"Selecting period {start_year}-{end_year} for 'month' variables in experiment '{experiment}'...")
                    ds_month_selected = pro_dat.select_period(
                        ds_month[experiment], start_year=start_year, end_year=end_year,
                        specific_months_or_seasons=specific_months_or_seasons
                    )
    
                    print(f"Computing temporal stats {temporal_stats} for 'month' variables in experiment '{experiment}'...")
                    # Option A: do it directly on selected data
                    ds_dict[experiment]["month"] = temporal_stats_dict(
                        ds_month_selected, stats=temporal_stats, dim="time", suffix=True
                    )
    
            # -----------------
            # YEAR
            # -----------------
            ds_year = {}
            if year_variables:
                print(f"Loading 'year' resolution variables {year_variables} for experiment '{experiment}'...")
                ds_year = load_multiple_models_and_experiments(BASE_DIR, data_state, [experiment], "year", models, year_variables)
    
                if start_year and end_year and experiment in ds_year and ds_year[experiment]:
                    print(f"Selecting period {start_year}-{end_year} for 'year' variables in experiment '{experiment}'...")
                    ds_year_selected = pro_dat.select_period(ds_year[experiment], start_year=start_year, end_year=end_year)
    
                    print(f"Computing temporal stats {temporal_stats} for 'year' variables in experiment '{experiment}'...")
                    ds_dict[experiment]["year"] = temporal_stats_dict(
                        ds_year_selected, stats=temporal_stats, dim="time", suffix=True
                    )
    
            # -----------------
            # MERGE per model
            # -----------------
            print(f"Merging all datasets for experiment '{experiment}'...")
            merged_dict = {}
            for model in models:
                model_datasets = []
    
                # month stats dataset already contains all variables with suffixes
                if "month" in ds_dict[experiment] and model in ds_dict[experiment]["month"]:
                    model_datasets.append(ds_dict[experiment]["month"][model])
    
                # year stats dataset
                if "year" in ds_dict[experiment] and model in ds_dict[experiment]["year"]:
                    model_datasets.append(ds_dict[experiment]["year"][model])
    
                if model_datasets:
                    merged_ds = xr.merge(model_datasets, compat="override")
    
                    # keep attrs (prefer month attrs if present, else year)
                    if experiment in ds_month and model in ds_month.get(experiment, {}):
                        merged_ds.attrs = ds_month[experiment][model].attrs
                    elif experiment in ds_year and model in ds_year.get(experiment, {}):
                        merged_ds.attrs = ds_year[experiment][model].attrs
    
                    merged_dict[model] = merged_ds
    
            ds_dict[experiment] = merged_dict
    
        return ds_dict
        """


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
                merged_dict[model] = merged_ds

        ds_dict[experiment] = merged_dict

    return ds_dict

def monthly_climatology_dict(ds_by_model: dict, stat="mean", dim="time", suffix="clim"):
    """
    Compute monthly climatology (12 months) for each model dataset.

    Produces variables like <var>_<suffix> with dimension 'month' (1..12).

    Parameters
    ----------
    ds_by_model : dict[str, xr.Dataset]
    stat : str
        "mean" or "median" (mean is typical)
    dim : str
        usually "time"
    suffix : str
        appended to var names, e.g. "clim" -> pr_clim

    Returns
    -------
    dict[str, xr.Dataset]
    """
    out = {}
    for model, ds in ds_by_model.items():
        if "time" not in ds.dims and "time" not in ds.coords:
            raise ValueError(f"{model}: no time coordinate found")

        if stat == "mean":
            ds_clim = ds.groupby("time.month").mean(dim=dim, skipna=True)
        elif stat == "median":
            ds_clim = ds.groupby("time.month").median(dim=dim, skipna=True)
        else:
            raise ValueError(f"Unsupported stat: {stat}")

        # Rename variables to carry suffix
        ds_clim = ds_clim.rename({v: f"{v}_{suffix}" for v in ds_clim.data_vars})

        # Keep attrs (optional)
        ds_clim.attrs = ds.attrs.copy()
        ds_clim.attrs["temporal_stat"] = stat
        ds_clim.attrs["climatology"] = "monthly"
        out[model] = ds_clim

    return out


def load_period_monthly_clim(
    BASE_DIR,
    data_state,
    experiments,
    models,
    variables,
    custom_periods=None,
    clim_stat="mean",
):
    """
    Load monthly-res variables, select the experiment period, and compute a 12-month climatology.

    Returns
    -------
    dict[experiment][model] -> xr.Dataset with variables named <var>_clim and dim 'month'
    """

    # Default periods (override with custom_periods if needed)
    default_periods = {
        "historical": (1985, 2014),
        "hist-noLu": (1985, 2014),
        "ssp126": (2070, 2099),
        "ssp370": (2070, 2099),
        "ssp585": (2070, 2099),
    }

    month_variables_all = [
        "tas", "pr", "vpd", "evspsbl", "evapo", "tran", "mrro", "mrso", "mrsos",
        "lai", "gpp", "wue", "huss", "ps", "rsds", "clt",
        "pr_no_landmask", "evspsbl_no_landmask", "evspsblsoi", "evspsblveg", "evspsblpot"
    ]
    month_variables = [v for v in month_variables_all if v in variables]

    ds_dict = {}
    for experiment in experiments:
        ds_dict[experiment] = {}

        start_year, end_year = (
            custom_periods.get(experiment)
            if custom_periods and experiment in custom_periods
            else default_periods.get(experiment, (None, None))
        )

        if not month_variables:
            print(f"[WARN] No monthly variables requested for experiment '{experiment}'.")
            continue

        print(f"Loading MONTH data for {experiment} vars={month_variables} ...")
        ds_month = load_multiple_models_and_experiments(
            BASE_DIR, data_state, [experiment], "month", models, month_variables
        )

        if start_year is None or end_year is None:
            raise ValueError(f"No period defined for experiment '{experiment}'")

        if experiment not in ds_month or not ds_month[experiment]:
            print(f"[WARN] No monthly datasets loaded for experiment '{experiment}'")
            continue

        print(f"Selecting period {start_year}-{end_year} for '{experiment}'...")
        ds_month_selected = pro_dat.select_period(
            ds_month[experiment],
            start_year=start_year,
            end_year=end_year,
            specific_months_or_seasons=None,  # keep all months
        )

        print(f"Computing monthly climatology ({clim_stat}) for '{experiment}'...")
        ds_clim = monthly_climatology_dict(
            ds_month_selected,
            stat=clim_stat,
            dim="time",
            suffix="clim",
        )

        ds_dict[experiment] = ds_clim

    return ds_dict


def compute_reference_partition_metrics(ds_ref, suffix):
    """
    Apply your compute_partition_metrics to reference dict, but ONLY with tran as ET-like term.
    """
    return pro_dat.compute_partition_metrics(
        ds_ref,
        suffix=suffix,
        variants=("tran",),          # IMPORTANT: only tran exists
        compute_bgws=True,
        compute_ratios=True,
    )



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
    month_variables = ['tas', 'pr', 'vpd', 'evspsbl', 'evapo', 'tran', 'mrro', 'mrso', 'mrsos', 'lai', 'gpp', 'wue', 'huss', 'ps', 'fracLut', 'treeFrac', 'cLeaf',  'cVeg', 'nVeg', 'cRoot', 'pr_no_landmask', 'evspsbl_no_landmask', "evspsblsoi", "evspsblveg", "evspsblpot"]
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