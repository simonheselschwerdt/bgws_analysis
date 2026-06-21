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
Date: 2026-02-26
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
            if key not in ['4 model ensemble mean', '5 model ensemble mean', '9 model ensemble mean', '11 model ensemble mean', '12 model ensemble mean',
                           '4 model ensemble median', '5 model ensemble median', '9 model ensemble median', '11 model ensemble median', '12 model ensemble median', '4 model ensemble std', '5 model ensemble std', '9 model ensemble std', '11 model ensemble std', '12 model ensemble std']                   
        }

        excluded_from_start = [
            key for key in ds_experiment_dict.keys()
            if key in ['4 model ensemble mean', '5 model ensemble mean', '9 model ensemble mean', '11 model ensemble mean', '12 model ensemble mean',
                       '4 model ensemble median', '5 model ensemble median', '9 model ensemble median', '11 model ensemble median', '12 model ensemble median', '4 model ensemble std', '5 model ensemble std', '9 model ensemble std', '11 model ensemble std', '12 model ensemble std']
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





# ========== Utility Functions ==========



def apply_hydrological_mask_from_historical(ds_dict, ds_dict_hydro_mask=None, pr_thresh=0.82, tran_thresh=0.1, mrro_thresh=0.05):
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
            ensemble_ds = ds_dict_hydro_mask['historical']['12 model ensemble mean']
        except KeyError:
            raise KeyError("Ensemble mean dataset ('12 model ensemble mean') not found in 'historical' scenario.")
    else:
        experiments = ["historical"]
        models = [
           'BCC-CSM2-MR_r1i1p1f1', 'CanESM5_r1i1p1f1', 'CESM2_r11i1p1f1', "CMCC-ESM2_r1i1p1f1", 'CNRM-ESM2-1_r1i1p1f2', "EC-Earth3-Veg_r1i1p1f1", 
           "GFDL-ESM4_r1i1p1f1",'IPSL-CM6A-LR_r1i1p1f1', 'MIROC-ES2L_r1i1p1f2', 'MPI-ESM1-2-LR_r11i1p1f1', 'NorESM2-MM_r1i1p1f1', 'UKESM1-0-LL_r1i1p1f2',
         ]
        variables=['pr', 'mrro', 'tran']
        season = None

        # Step 1.2: Load the datasets
        print("Historical data for mask needs to be loaded first...")
        with ProgressBar():
            ds_dict_hydro_mask = dask.compute(
                load_dat.load_period_stats(
                    DATA_DIR, 'processed', experiments, models, variables, specific_months_or_seasons=season, temporal_stats=("mean",),
                )
            )[0]
        
        ds_dict_hydro_mask['historical'] = compute_ensemble_statistic(ds_dict_hydro_mask['historical'], 'mean', '12 model ensemble')
        ensemble_ds = ds_dict_hydro_mask['historical']['12 model ensemble mean']

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

import xarray as xr
import numpy as np

def _resolve_var_name(ds: xr.Dataset, base: str) -> str:
    """
    Accept either 'pr' or 'pr_mean' (same for tran/mrro).
    """
    if base in ds.data_vars:
        return base
    cand = f"{base}_mean"
    if cand in ds.data_vars:
        return cand
    raise KeyError(f"Neither '{base}' nor '{cand}' found. Available: {list(ds.data_vars)[:15]}")

def build_hydro_mask_from_loaded_historical(
    hist_dict: dict[str, xr.Dataset],
    pr_thresh=0.822,
    tran_thresh=0.05,
    mrro_thresh=0.05,
    #lai_thresh=0.4,
):
    """
    Build a historical ensemble-mean mask for hydrologically active and vegetated land.
    Mask criteria:
      pr   > pr_thresh
      tran > tran_thresh
      mrro > mrro_thresh
      #lai  > lai_thresh
    """
    usable = {}
    for model, ds in hist_dict.items():
        try:
            prn = _resolve_var_name(ds, "pr")
            trn = _resolve_var_name(ds, "tran")
            rrn = _resolve_var_name(ds, "mrro")
     #       lain = _resolve_var_name(ds, "lai")
            usable[model] = ds[[prn, trn, rrn]]#, lain]]
        except KeyError as e:
            print(f"[MASK] skip {model}: {e}")

    if not usable:
        raise ValueError("Cannot build hydro mask: no historical model had pr+tran+mrro+lai.")

    aligned = xr.align(*usable.values(), join="inner")

    prn = _resolve_var_name(aligned[0], "pr")
    trn = _resolve_var_name(aligned[0], "tran")
    rrn = _resolve_var_name(aligned[0], "mrro")
    #lain = _resolve_var_name(aligned[0], "lai")

    pr_ens   = xr.concat([ds[prn] for ds in aligned], dim="model").mean("model", skipna=True)
    tran_ens = xr.concat([ds[trn] for ds in aligned], dim="model").mean("model", skipna=True)
    mrro_ens = xr.concat([ds[rrn] for ds in aligned], dim="model").mean("model", skipna=True)
    #lai_ens  = xr.concat([ds[lain] for ds in aligned], dim="model").mean("model", skipna=True)

    mask = (
        (pr_ens > pr_thresh)
        & (tran_ens > tran_thresh)
        & (mrro_ens > mrro_thresh)
     #   & (lai_ens > lai_thresh)
    )

    mask.name = "hydro_veg_mask"
    mask.attrs["description"] = (
        "Hydrological and vegetation mask from historical ensemble mean. "
        f"Thresholds: pr>{pr_thresh}, tran>{tran_thresh}, mrro>{mrro_thresh}.)"#, lai>{lai_thresh}."
    )
    return mask
    
def apply_hydro_mask_to_dict(ds_by_model: dict[str, xr.Dataset], mask: xr.DataArray):
    """
    Apply the same mask to all variables in each dataset.
    """
    out = {}
    for model, ds in ds_by_model.items():
        # Align mask and ds (safe even if some coords differ slightly)
        mask_a, ds_a = xr.align(mask, ds, join="inner")

        ds_masked = ds_a.where(mask_a)
        ds_masked.attrs = ds.attrs.copy()
        out[model] = ds_masked
    return out

import os
import shutil
import xarray as xr
import zarr
from numcodecs import Blosc

def _zarr_safe_key(s: str) -> str:
    # keep keys identical unless they contain '/'
    return s.replace("/", "_")

def save_nested_dict_zarr(ds_dict, store_path, hydro_mask=None, overwrite=True,
                          clevel=5, chunk_max=256, spatial_dim_candidates=("lat","lon","x","y")):
    import os, shutil, zarr
    import xarray as xr
    from numcodecs import Blosc

    if overwrite and os.path.exists(store_path):
        shutil.rmtree(store_path)

    compressor = Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.BITSHUFFLE)

    for exp, mdict in ds_dict.items():
        exp_g = _zarr_safe_key(exp)
        for model, ds in mdict.items():
            model_g = _zarr_safe_key(model)

            ds_to_write = ds.copy()
            ds_to_write.attrs["_orig_exp_key"] = exp
            ds_to_write.attrs["_orig_model_key"] = model

            for v in ds_to_write.data_vars:
                if str(ds_to_write[v].dtype).startswith("float"):
                    ds_to_write[v] = ds_to_write[v].astype("float32")

            chunks = {}
            for d in spatial_dim_candidates:
                if d in ds_to_write.dims:
                    chunks[d] = min(ds_to_write.sizes[d], chunk_max)
            if chunks:
                ds_to_write = ds_to_write.chunk(chunks)

            encoding = {v: {"compressor": compressor} for v in ds_to_write.data_vars}

            ds_to_write.to_zarr(
                store_path,
                group=f"{exp_g}/{model_g}",
                mode="a",
                consolidated=False,
                encoding=encoding,
            )

    if hydro_mask is not None:
        xr.Dataset({"hydro_mask": hydro_mask}).to_zarr(
            store_path, group="_meta", mode="a", consolidated=False
        )

    zarr.consolidate_metadata(store_path)


def load_nested_dict_zarr(store_path):
    import zarr
    import xarray as xr

    root = zarr.open_consolidated(store_path, mode="r")

    out = {}
    for exp_g in root.group_keys():
        if exp_g == "_meta":
            continue

        for model_g in root[exp_g].group_keys():
            ds = xr.open_zarr(store_path, group=f"{exp_g}/{model_g}", consolidated=True)

            exp = ds.attrs.get("_orig_exp_key", exp_g)
            model = ds.attrs.get("_orig_model_key", model_g)

            out.setdefault(exp, {})[model] = ds

    hydro_mask = None
    if "_meta" in root:
        hydro_mask = xr.open_zarr(store_path, group="_meta", consolidated=True)["hydro_mask"]

    return out, hydro_mask

def bgws_period_stats(ds_dict_hydro_mask=None):

    cache_dir = "/work/ch0636/g300115/phd_project/common/data/processed/cache_files_paper_1/" 
    cache_main = os.path.join(cache_dir, "bgws_masked.zarr")
    cache_diff = os.path.join(cache_dir, "bgws_changes.zarr")

    if os.path.exists(cache_main) and os.path.exists(cache_diff):
        masked_ds_dict, hydro_mask = load_nested_dict_zarr(cache_main)
        masked_ds_dict_diff, _ = load_nested_dict_zarr(cache_diff)
        return masked_ds_dict, masked_ds_dict_diff, hydro_mask
    
    # Step 1: Define the datasets
    experiments = ["historical", "ssp126", "ssp370"]
    models = ['BCC-CSM2-MR_r1i1p1f1', 'CanESM5_r1i1p1f1', 'CESM2_r11i1p1f1', "CMCC-ESM2_r1i1p1f1", 'CNRM-ESM2-1_r1i1p1f2', "EC-Earth3-Veg_r1i1p1f1", 
              "GFDL-ESM4_r1i1p1f1",'IPSL-CM6A-LR_r1i1p1f1', 'MIROC-ES2L_r1i1p1f2', 'MPI-ESM1-2-LR_r11i1p1f1', 'NorESM2-MM_r1i1p1f1', 'UKESM1-0-LL_r1i1p1f2',
             ]
    variables=['pr', 'mrro', 'tran', 'evapo', 'evspsbl', 'lai']# "pr_no_landmask", "evspsbl_no_landmask"]'evapo', , 'evspsbl'

    experiment_periods = {
        "historical": {"historical": (1985, 2014)},
        "ssp126": {"ssp126_nf": (2030, 2059), "ssp126_ff": (2070, 2099)},
        "ssp370": {"ssp370_nf": (2030, 2059), "ssp370_ff": (2070, 2099)},
    }
    
    with ProgressBar():
        ds_dict = dask.compute(
            load_dat.load_period_stats(
                DATA_DIR, "processed",
                experiments, models, variables,
                specific_months_or_seasons=None,
                temporal_stats=("mean",),
                experiment_periods=experiment_periods,   # <-- NEW
            )
        )[0]

    for exp, mdict in ds_dict.items():
        for model, ds in mdict.items():
            needed = {"pr_mean", "tran_mean", "mrro_mean"}
            if not needed.issubset(set(ds.data_vars)):
                print(f"[CHECK] {exp}/{model} missing {sorted(needed - set(ds.data_vars))}")


    # Step 3: Compute bgws
    ds_dict = pro_dat.compute_partition_metrics(ds_dict, suffix="mean", variants=("tran", "evapo", "evspsbl"),
                                                compute_bgws=True, compute_ratios=True,
                                                )


    # Step 4: Apply hydrological mask
    hydro_mask = build_hydro_mask_from_loaded_historical(
        ds_dict["historical"],
    )
    
    # Apply to all experiments
    masked_ds_dict = {}
    for exp in ds_dict.keys():
        masked_ds_dict[exp] = apply_hydro_mask_to_dict(ds_dict[exp], hydro_mask)
    
        ensmean = compute_ensemble_statistic(masked_ds_dict[exp], "mean", "12 model ensemble")
        masked_ds_dict[exp]["12 model ensemble mean"] = ensmean["12 model ensemble mean"]
    
        ensmedian = compute_ensemble_statistic(masked_ds_dict[exp], "median", "12 model ensemble")
        masked_ds_dict[exp]["12 model ensemble median"] = ensmedian["12 model ensemble median"]

        ensmedian = compute_ensemble_statistic(masked_ds_dict[exp], "std", "12 model ensemble")
        masked_ds_dict[exp]["12 model ensemble std"] = ensmedian["12 model ensemble std"]


    # Step 5: Compute differences (comparison ds - reference ds)
    masked_ds_dict_diff = {}
    for k in ["ssp126_nf", "ssp126_ff", "ssp370_nf", "ssp370_ff"]:
        masked_ds_dict_diff[f"{k}-historical"] = pro_dat.compute_diff_dict(
            masked_ds_dict,
            reference_key="historical",
            comparison_key=k,
            var_rel_change=None
        )

    masked_ds_dict_diff["ssp126_ff-ssp126_nf"] = pro_dat.compute_diff_dict(
    masked_ds_dict, reference_key="ssp126_nf", comparison_key="ssp126_ff", var_rel_change=None
    )
    masked_ds_dict_diff["ssp370_ff-ssp370_nf"] = pro_dat.compute_diff_dict(
        masked_ds_dict, reference_key="ssp370_nf", comparison_key="ssp370_ff", var_rel_change=None
    )

    # Step 5: Compute MMM for differences
    for exp in masked_ds_dict_diff.keys():
        ensmean = compute_ensemble_statistic(masked_ds_dict_diff[exp], "mean", "12 model ensemble")
        masked_ds_dict_diff[exp]["12 model ensemble mean"] = ensmean["12 model ensemble mean"]
    
        ensmedian = compute_ensemble_statistic(masked_ds_dict_diff[exp], "median", "12 model ensemble")
        masked_ds_dict_diff[exp]["12 model ensemble median"] = ensmedian["12 model ensemble median"]

        ensmedian = compute_ensemble_statistic(masked_ds_dict_diff[exp], "std", "12 model ensemble")
        masked_ds_dict_diff[exp]["12 model ensemble std"] = ensmedian["12 model ensemble std"]

    os.makedirs(cache_dir, exist_ok=True)
    save_nested_dict_zarr(masked_ds_dict, cache_main, hydro_mask=hydro_mask, overwrite=True)
    save_nested_dict_zarr(masked_ds_dict_diff, cache_diff, hydro_mask=hydro_mask, overwrite=True)

    return masked_ds_dict, masked_ds_dict_diff, hydro_mask

def selected_driver_landuse_period_stats(
    hydro_mask=None,
    nf_years=(2030, 2059),
    ff_years=(2070, 2099),
    hist_years=(1985, 2014),
    landuse_vars=("treeFrac", "cropFrac"),
):
    """
    Load the selected regression drivers + land-use fractions for the 9-model subset
    with treeFrac availability, compute bgws_tran_mean, changes, and ensemble statistics.

    Final variables retained in the returned datasets:
      - bgws_tran_mean
      - pr_mean
      - pr_seasonality
      - vpd_mean
      - vpd_seasonality
      - mrsos_mean
      - lai_mean
      - clt_mean
      - wue_mean
      - RX5day
      - <landuse_var>_mean for each variable in landuse_vars

    Parameters
    ----------
    hydro_mask : xr.DataArray or None
        Optional precomputed hydrological mask. If None, it is built internally
        from the historical data of the retained models.
    nf_years, ff_years, hist_years : tuple
        Period definitions.
    landuse_vars : tuple[str]
        Land-use variables to load, e.g. ("treeFrac",) or ("treeFrac", "cropFrac").

    Returns
    -------
    masked_ds_dict : dict
        Nested dictionary with masked period means and ensemble statistics.
    masked_ds_dict_diff : dict
        Nested dictionary with masked changes and ensemble statistics.
    hydro_mask : xr.DataArray
        The hydrological mask used.
    """

    import os
    import numpy as np
    import pandas as pd
    import xarray as xr
    import dask
    from dask.diagnostics import ProgressBar

    CACHE_MAIN = "/work/ch0636/g300115/phd_project/common/data/processed/cache_files_paper_1/selected_drivers_landuse_masked_9m.zarr"
    CACHE_DIFF = "/work/ch0636/g300115/phd_project/common/data/processed/cache_files_paper_1/selected_drivers_landuse_changes_9m.zarr"

    if os.path.exists(CACHE_MAIN) and os.path.exists(CACHE_DIFF):
        print("Cached files exist and are loaded")
        masked_ds_dict, hydro_mask_loaded = load_ds_dict_from_zarr(CACHE_MAIN)
        masked_ds_dict_diff, _ = load_ds_dict_from_zarr(CACHE_DIFF)
        return masked_ds_dict, masked_ds_dict_diff, hydro_mask_loaded

    # ------------------------------------------------------------------
    # Source experiments (on disk) + output-period keys
    # ------------------------------------------------------------------
    source_experiments = ["historical", "ssp370"] #"ssp126",

    experiment_periods = {
        "historical": {"historical": hist_years},
        #"ssp126": {"ssp126_nf": nf_years, "ssp126_ff": ff_years},
        "ssp370": {"ssp370_nf": nf_years, "ssp370_ff": ff_years},
    }

    out_keys = ["historical", "ssp370_nf", "ssp370_ff"] #"ssp126_nf", "ssp126_ff", 

    # 9-model subset with treeFrac availability for your exact member choices
    models = [
        "CanESM5_r1i1p1f1",
        "CESM2_r11i1p1f1",
        "CMCC-ESM2_r1i1p1f1",
        "CNRM-ESM2-1_r1i1p1f2",
        "EC-Earth3-Veg_r1i1p1f1",
        "GFDL-ESM4_r1i1p1f1",
        "IPSL-CM6A-LR_r1i1p1f1",
        "MPI-ESM1-2-LR_r11i1p1f1",
        "UKESM1-0-LL_r1i1p1f2",
    ]

    # ------------------------------------------------------------------
    # Variables to load
    # ------------------------------------------------------------------
    # Means loaded directly
    bgws_source_vars = ["pr", "mrro", "tran", "lai"]
    selected_driver_vars = ["wue", "vpd", "mrsos", "mrso", "clt", "treeFrac", "cropFrac"] + list(landuse_vars)

    # Seasonal diagnostics computed from raw monthly fields
    seasonality_raw_vars = ["pr", "vpd"]

    # Yearly extreme metric retained in the final model
    rx_vars = ["RX5day"]

    # Final variables retained in the returned datasets
    final_keep_vars = [
        "bgws_tran_mean",
        "pr_mean",
        "pr_seasonality",
        "vpd_mean",
        "vpd_seasonality",
        "mrsos_mean",
        "mrso_mean",
        "lai_mean",
        "clt_mean",
        "wue_mean",
        "RX5day",
    ] + [f"{v}_mean" for v in landuse_vars]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_spatial_only(ds):
        out = ds.copy()
        for v in list(out.data_vars):
            da = out[v]
            for dim in ("year", "time"):
                if dim in da.dims and da.sizes.get(dim, 0) > 1:
                    out[v] = da.mean(dim, keep_attrs=True)
        return out

    def _extract_years(values):
        v = np.asarray(values)
        if np.issubdtype(v.dtype, np.datetime64):
            return (v.astype("datetime64[Y]").astype(int) + 1970).astype(int)
        if np.issubdtype(v.dtype, np.integer):
            return v.astype(int)
        if np.issubdtype(v.dtype, np.floating):
            return np.round(v).astype(int)

        out = np.empty(v.shape, dtype=int)
        it = np.nditer(v, flags=["multi_index", "refs_ok"])
        for x in it:
            obj = x.item()
            if obj is None:
                out[it.multi_index] = -9999
                continue
            if isinstance(obj, pd.Timestamp):
                out[it.multi_index] = obj.year
                continue
            if hasattr(obj, "year"):
                out[it.multi_index] = int(obj.year)
                continue
            if isinstance(obj, str):
                try:
                    out[it.multi_index] = int(obj[:4])
                    continue
                except Exception:
                    pass
            try:
                out[it.multi_index] = pd.to_datetime(obj).year
            except Exception:
                raise TypeError(f"Cannot extract year from type {type(obj)} value={obj}")
        return out

    def _get_time_dim(ds):
        if "time" in ds.dims:
            return "time"
        if "year" in ds.dims:
            return "year"
        return None

    def _to_year_index(ds):
        tdim = _get_time_dim(ds)
        if tdim is None:
            return ds
        years = _extract_years(ds[tdim].values)
        out = ds.assign_coords(year=(tdim, years))
        if tdim != "year":
            out = out.swap_dims({tdim: "year"}).drop_vars(tdim)
        out = out.sortby("year")
        _, idx = np.unique(out["year"].values, return_index=True)
        return out.isel(year=idx)

    def _cut_to_period_by_year(ds, start_year, end_year):
        ds_y = _to_year_index(ds)
        if "year" in ds_y.dims:
            return ds_y.sel(year=slice(int(start_year), int(end_year)))
        return ds_y

    def _align_to_ref_years(ds_y, ref_years):
        if "year" not in ds_y.dims:
            return ds_y
        y0 = int(max(ds_y.year.min().item(), ref_years.min().item()))
        y1 = int(min(ds_y.year.max().item(), ref_years.max().item()))
        ds_y = ds_y.sel(year=slice(y0, y1))
        ref_overlap = ref_years.sel(year=slice(y0, y1))
        return ds_y.reindex(year=ref_overlap)

    def _subset_vars(ds, keep_vars):
        present = [v for v in keep_vars if v in ds.data_vars]
        return ds[present]

    # map output key -> (source experiment, start, end)
    outkey_to_src = {}
    for src, mapping in experiment_periods.items():
        for out_key, (y0, y1) in mapping.items():
            outkey_to_src[out_key] = (src, y0, y1)

    # ------------------------------------------------------------------
    # Step 1: Load all mean variables needed for final predictors + BGWS
    # ------------------------------------------------------------------
    print("Loading selected mean drivers + BGWS source variables (9-model subset)...")
    mean_vars = list(dict.fromkeys(bgws_source_vars + selected_driver_vars))

    with ProgressBar():
        ds_dict_means = dask.compute(
            load_dat.load_period_stats(
                DATA_DIR,
                "processed",
                source_experiments,
                models,
                mean_vars,
                specific_months_or_seasons=None,
                temporal_stats=("mean",),
                experiment_periods=experiment_periods,
            )
        )[0]

    # ------------------------------------------------------------------
    # Step 2: Compute BGWS partition metrics (adds bgws_tran_mean, ratios, etc.)
    # ------------------------------------------------------------------
    print("Computing BGWS partition metrics...")
    ds_dict_means = pro_dat.compute_partition_metrics(
        ds_dict_means,
        suffix="mean",
        variants=("tran",),
        compute_bgws=True,
        compute_ratios=True,
    )

    # ------------------------------------------------------------------
    # Step 3: Compute pr/vpd seasonality from raw monthly data
    # ------------------------------------------------------------------
    print("Loading monthly pr/vpd and computing seasonality...")
    with ProgressBar():
        ds_prvpd_raw = dask.compute(
            load_dat.load_multiple_models_and_experiments(
                DATA_DIR, "processed", source_experiments, "month", models, seasonality_raw_vars
            )
        )[0]

    seasonality = {k: {} for k in out_keys}
    for out_key in out_keys:
        src, y0, y1 = outkey_to_src[out_key]
        for model in models:
            ds_m = ds_prvpd_raw.get(src, {}).get(model, None)
            if ds_m is None:
                continue

            ds_sel_dict = pro_dat.select_period({model: ds_m}, start_year=y0, end_year=y1)
            ds_sel = ds_sel_dict.get(model, None)
            if ds_sel is None or "time" not in ds_sel.dims:
                continue

            ds_out = xr.Dataset()
            if "pr" in ds_sel:
                pr_clim = ds_sel["pr"].groupby("time.month").mean("time", skipna=True)
                ds_out["pr_seasonality"] = pr_clim.std("month", skipna=True).astype("float32")
            if "vpd" in ds_sel:
                vpd_clim = ds_sel["vpd"].groupby("time.month").mean("time", skipna=True)
                ds_out["vpd_seasonality"] = vpd_clim.std("month", skipna=True).astype("float32")

            if ds_out.data_vars:
                seasonality[out_key][model] = ds_out

    # ------------------------------------------------------------------
    # Step 4: Load RX5day (yearly) and compute period means
    # ------------------------------------------------------------------
    print("Loading RX5day (yearly) and computing period means...")
    ds_rx_by_var = {}
    for v in rx_vars:
        with ProgressBar():
            ds_rx_by_var[v] = dask.compute(
                load_dat.load_multiple_models_and_experiments(
                    DATA_DIR, "processed", source_experiments, "year", models, [v]
                )
            )[0]

    ds_dict_rx_means = {k: {} for k in out_keys}

    for out_key in out_keys:
        src, y0, y1 = outkey_to_src[out_key]

        # choose first available model as reference for yearly alignment
        ref_years = None
        for ref_model in models:
            ref_ds = None
            for v in rx_vars:
                ref_ds = ds_rx_by_var[v].get(src, {}).get(ref_model, None)
                if ref_ds is not None:
                    break
            if ref_ds is not None:
                ref_cut = _cut_to_period_by_year(ref_ds, y0, y1)
                if "year" in ref_cut.dims:
                    ref_years = ref_cut["year"]
                    break

        for model in models:
            rx_parts = []
            for v in rx_vars:
                ds_v = ds_rx_by_var[v].get(src, {}).get(model, None)
                if ds_v is None:
                    continue

                ds_v = _cut_to_period_by_year(ds_v, y0, y1)
                if ref_years is not None and "year" in ds_v.dims:
                    ds_v = _align_to_ref_years(ds_v, ref_years)
                rx_parts.append(ds_v)

            if not rx_parts:
                continue

            ds_rx_merged = xr.merge(rx_parts, compat="override")
            ds_rx_merged = _ensure_spatial_only(ds_rx_merged)
            ds_dict_rx_means[out_key][model] = ds_rx_merged

    # ------------------------------------------------------------------
    # Step 5: Merge mean variables + seasonality + RX5day
    # ------------------------------------------------------------------
    ds_dict_all = {k: {} for k in out_keys}
    for out_key in out_keys:
        for model in models:
            parts = []

            if model in ds_dict_means.get(out_key, {}):
                parts.append(ds_dict_means[out_key][model])

            if model in ds_dict_rx_means.get(out_key, {}):
                parts.append(ds_dict_rx_means[out_key][model])

            if model in seasonality.get(out_key, {}):
                parts.append(seasonality[out_key][model])

            if not parts:
                continue

            ds_dict_all[out_key][model] = xr.merge(parts, compat="override")

    # ------------------------------------------------------------------
    # Step 6: Keep only models with all selected variables across all periods
    # ------------------------------------------------------------------
    required_vars = set(final_keep_vars)
    valid_models = set(models)

    for out_key in out_keys:
        valid_here = set()
        for model in models:
            ds_m = ds_dict_all.get(out_key, {}).get(model, None)
            if ds_m is None:
                continue
            if required_vars.issubset(set(ds_m.data_vars)):
                valid_here.add(model)
            else:
                missing = sorted(required_vars - set(ds_m.data_vars))
                print(f"[CHECK] {out_key}/{model} missing {missing}")
        valid_models &= valid_here

    valid_models = [m for m in models if m in valid_models]

    if not valid_models:
        raise ValueError(
            "No common models remain after enforcing availability of all selected variables. "
            "If cropFrac is sparse, try landuse_vars=('treeFrac',)."
        )

    n_models = len(valid_models)
    ens_prefix = f"{n_models} model ensemble"

    print(f"Using {n_models} common models:")
    for m in valid_models:
        print(f" - {m}")

    for out_key in out_keys:
        ds_dict_all[out_key] = {m: ds_dict_all[out_key][m] for m in valid_models}

    # ------------------------------------------------------------------
    # Step 7: Build / apply hydrological mask
    # ------------------------------------------------------------------
    if hydro_mask is None:
        hydro_mask = build_hydro_mask_from_loaded_historical(
            ds_dict_all["historical"],
            pr_thresh=0.05,
            tran_thresh=0.005,
            mrro_thresh=0.005,
        )

    masked_ds_dict = {}
    for out_key in out_keys:
        masked_full = apply_hydro_mask_to_dict(ds_dict_all[out_key], hydro_mask)

        # keep only the final variables used in the regression / response
        masked_ds_dict[out_key] = {
            model: _subset_vars(ds, final_keep_vars)
            for model, ds in masked_full.items()
        }

        ensmean = compute_ensemble_statistic(masked_ds_dict[out_key], "mean", ens_prefix)
        masked_ds_dict[out_key][f"{ens_prefix} mean"] = ensmean[f"{ens_prefix} mean"]

        ensmedian = compute_ensemble_statistic(masked_ds_dict[out_key], "median", ens_prefix)
        masked_ds_dict[out_key][f"{ens_prefix} median"] = ensmedian[f"{ens_prefix} median"]

        ensstd = compute_ensemble_statistic(masked_ds_dict[out_key], "std", ens_prefix)
        masked_ds_dict[out_key][f"{ens_prefix} std"] = ensstd[f"{ens_prefix} std"]

    # ------------------------------------------------------------------
    # Step 8: Compute changes
    # ------------------------------------------------------------------
    masked_ds_dict_diff = {}

    for k in ["ssp370_nf", "ssp370_ff"]: #"ssp126_nf", "ssp126_ff", 
        masked_ds_dict_diff[f"{k}-historical"] = pro_dat.compute_diff_dict(
            masked_ds_dict,
            reference_key="historical",
            comparison_key=k,
            var_rel_change=None,
        )

    #masked_ds_dict_diff["ssp126_ff-ssp126_nf"] = pro_dat.compute_diff_dict(
    #    masked_ds_dict,
    #    reference_key="ssp126_nf",
    #    comparison_key="ssp126_ff",
    #    var_rel_change=None,
    #)

    masked_ds_dict_diff["ssp370_ff-ssp370_nf"] = pro_dat.compute_diff_dict(
        masked_ds_dict,
        reference_key="ssp370_nf",
        comparison_key="ssp370_ff",
        var_rel_change=None,
    )

    # ------------------------------------------------------------------
    # Step 9: Ensemble statistics for differences
    # ------------------------------------------------------------------
    for diff_key in list(masked_ds_dict_diff.keys()):
        ensmean = compute_ensemble_statistic(masked_ds_dict_diff[diff_key], "mean", ens_prefix)
        masked_ds_dict_diff[diff_key][f"{ens_prefix} mean"] = ensmean[f"{ens_prefix} mean"]

        ensmedian = compute_ensemble_statistic(masked_ds_dict_diff[diff_key], "median", ens_prefix)
        masked_ds_dict_diff[diff_key][f"{ens_prefix} median"] = ensmedian[f"{ens_prefix} median"]

        ensstd = compute_ensemble_statistic(masked_ds_dict_diff[diff_key], "std", ens_prefix)
        masked_ds_dict_diff[diff_key][f"{ens_prefix} std"] = ensstd[f"{ens_prefix} std"]

    # ------------------------------------------------------------------
    # Step 10: Cache + return
    # ------------------------------------------------------------------
    save_ds_dict_to_zarr(masked_ds_dict, CACHE_MAIN, hydro_mask=hydro_mask, overwrite=True)
    save_ds_dict_to_zarr(masked_ds_dict_diff, CACHE_DIFF, hydro_mask=hydro_mask, overwrite=True)

    return masked_ds_dict, masked_ds_dict_diff, hydro_mask

import os
import dask
from dask.diagnostics import ProgressBar

def bgws_period_monthly_clim_stats(
    hydro_mask=None,
    ds_dict_hydro_mask=None,
    DATA_DIR=None,

    experiments=("historical", "ssp126", "ssp370"),
    models=(
        'BCC-CSM2-MR_r1i1p1f1', 'CanESM5_r1i1p1f1', 'CESM2_r11i1p1f1', "CMCC-ESM2_r1i1p1f1", 'CNRM-ESM2-1_r1i1p1f2', "EC-Earth3-Veg_r1i1p1f1", 
        "GFDL-ESM4_r1i1p1f1",'IPSL-CM6A-LR_r1i1p1f1', 'MIROC-ES2L_r1i1p1f2', 'MPI-ESM1-2-LR_r11i1p1f1', 'NorESM2-MM_r1i1p1f1', 'UKESM1-0-LL_r1i1p1f2',
         
    ),
    variables=("pr", "mrro", "tran", "evapo", "evspsbl"),

    # ---- caching ----
    cache_dir=None,
    cache_main_name="bgws_period_monthly_clim_masked.zarr",
    cache_diff_name="bgws_period_monthly_clim_changes.zarr",
    use_cache=True,
    overwrite_cache=False,
    zarr_clevel=5,
    zarr_chunk_max=256,  # only used if your nested_dict_zarr saver supports it

    return_hydro_mask=True,
):
    """
    Loads monthly 12-month climatologies for BGWS inputs, computes partition metrics + BGWS,
    applies hydro mask, computes ensemble stats and scenario-historical differences,
    and (NEW) saves/loads results from Zarr caches.

    Returns
    -------
    masked_ds_dict : dict[exp][model] -> xr.Dataset (month, lat, lon, ...)
    masked_ds_dict_diff : dict[diff_key][model] -> xr.Dataset
    hydro_mask : xr.DataArray
    """

    # Allow using global DATA_DIR if caller keeps that pattern
    if DATA_DIR is None:
        DATA_DIR = globals().get("DATA_DIR", None)
    if DATA_DIR is None:
        raise ValueError("DATA_DIR must be provided (or exist as a global variable).")

    # Resolve cache helpers (same pattern as your other funcs)
    _load_cache = globals().get("load_nested_dict_zarr", None) or globals().get("load_ds_dict_from_zarr", None)
    _save_cache = globals().get("save_nested_dict_zarr", None) or globals().get("save_ds_dict_to_zarr", None)
    if _load_cache is None or _save_cache is None:
        raise RuntimeError(
            "Missing cache helpers. Need either (save/load)_nested_dict_zarr or save/load_ds_dict_from_zarr."
        )

    cache_main_path = None
    cache_diff_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_main_path = os.path.join(cache_dir, cache_main_name)
        cache_diff_path = os.path.join(cache_dir, cache_diff_name)

    # -----------------------------
    # Cache: load if available
    # -----------------------------
    if (
        use_cache
        and cache_main_path is not None and cache_diff_path is not None
        and os.path.exists(cache_main_path) and os.path.exists(cache_diff_path)
        and (not overwrite_cache)
    ):
        print(f"[CACHE] loading monthly clim masked:  {cache_main_path}")
        masked_ds_dict, hydro_cached = _load_cache(cache_main_path)

        print(f"[CACHE] loading monthly clim changes: {cache_diff_path}")
        masked_ds_dict_diff, _ = _load_cache(cache_diff_path)

        if return_hydro_mask:
            return masked_ds_dict, masked_ds_dict_diff, hydro_cached
        return masked_ds_dict, masked_ds_dict_diff

    # -----------------------------
    # Optional: grab hydro_mask from a passed tuple/dict if user gave it
    # -----------------------------
    if hydro_mask is None and ds_dict_hydro_mask is not None:
        # common pattern: (masked_ds_dict, masked_ds_dict_diff, hydro_mask)
        if isinstance(ds_dict_hydro_mask, tuple) and len(ds_dict_hydro_mask) >= 3:
            hydro_mask = ds_dict_hydro_mask[2]
        # or directly a DataArray
        elif hasattr(ds_dict_hydro_mask, "dims") and hasattr(ds_dict_hydro_mask, "values"):
            hydro_mask = ds_dict_hydro_mask

    # -----------------------------
    # 1) Load monthly climatologies
    # -----------------------------
    print("Loading monthly climatologies (12-month) ...")
    with ProgressBar():
        ds_dict_clim = dask.compute(
            load_dat.load_period_monthly_clim(
                DATA_DIR,
                "processed",
                list(experiments),
                list(models),
                list(variables),
                custom_periods=None,  # your defaults (1985-2014, 2070-2099)
                clim_stat="mean",
            )
        )[0]

    # quick check
    for exp, mdict in ds_dict_clim.items():
        for model, ds in mdict.items():
            needed = {"pr_clim", "tran_clim", "mrro_clim"}
            missing = sorted(needed - set(ds.data_vars))
            if missing:
                print(f"[CHECK] {exp}/{model} missing {missing}")

    # -----------------------------
    # 2) Compute partition metrics on climatology (ratios per month) + BGWS
    # -----------------------------
    ds_dict_clim = pro_dat.compute_partition_metrics(
        ds_dict_clim,
        suffix="clim",
        variants=("tran", "evapo", "evspsbl"),
        compute_bgws=True,
        compute_ratios=True,
    )

    # -----------------------------
    # 3) Build hydro mask if not supplied
    # -----------------------------
    if hydro_mask is None:
        print("[INFO] hydro_mask not provided; building from historical monthly climatology means.")
        hist_for_mask = {m: ds.mean("month", skipna=True) for m, ds in ds_dict_clim["historical"].items()}
        hydro_mask = build_hydro_mask_from_loaded_historical(
            hist_for_mask,
            pr_thresh=0.05,
            tran_thresh=0.005,
            mrro_thresh=0.005,
        )

    # -----------------------------
    # 4) Apply hydro mask + ensemble stats
    # -----------------------------
    masked_ds_dict = {}
    for exp in experiments:
        masked_ds_dict[exp] = apply_hydro_mask_to_dict(ds_dict_clim[exp], hydro_mask)

        ensmean = compute_ensemble_statistic(masked_ds_dict[exp], "mean", "12 model ensemble")
        masked_ds_dict[exp]["12 model ensemble mean"] = ensmean["12 model ensemble mean"]

        ensmedian = compute_ensemble_statistic(masked_ds_dict[exp], "median", "12 model ensemble")
        masked_ds_dict[exp]["12 model ensemble median"] = ensmedian["12 model ensemble median"]

    # -----------------------------
    # 5) Differences: future climatology - historical climatology (per month)
    # -----------------------------
    masked_ds_dict_diff = {}
    masked_ds_dict_diff["ssp126-historical"] = pro_dat.compute_diff_dict(
        masked_ds_dict,
        reference_key="historical",
        comparison_key="ssp126",
        var_rel_change=None,
    )
    masked_ds_dict_diff["ssp370-historical"] = pro_dat.compute_diff_dict(
        masked_ds_dict,
        reference_key="historical",
        comparison_key="ssp370",
        var_rel_change=None,
    )

    # Ensemble stats for differences
    for key in masked_ds_dict_diff.keys():
        ensmean = compute_ensemble_statistic(masked_ds_dict_diff[key], "mean", "12 model ensemble")
        masked_ds_dict_diff[key]["12 model ensemble mean"] = ensmean["12 model ensemble mean"]

        ensmedian = compute_ensemble_statistic(masked_ds_dict_diff[key], "median", "12 model ensemble")
        masked_ds_dict_diff[key]["12 model ensemble median"] = ensmedian["12 model ensemble median"]

    # -----------------------------
    # 6) Cache: save outputs
    # -----------------------------
    if cache_main_path is not None and cache_diff_path is not None:
        print(f"[CACHE] saving monthly clim masked:  {cache_main_path}")
        try:
            # nested_dict_zarr-style signature
            _save_cache(
                masked_ds_dict,
                cache_main_path,
                hydro_mask=hydro_mask,
                overwrite=True,
                clevel=zarr_clevel,
                chunk_max=zarr_chunk_max,
            )
        except TypeError:
            # ds_dict_to_zarr-style signature
            _save_cache(
                masked_ds_dict,
                cache_main_path,
                hydro_mask=hydro_mask,
                overwrite=True,
                clevel=zarr_clevel,
            )

        print(f"[CACHE] saving monthly clim changes: {cache_diff_path}")
        try:
            _save_cache(
                masked_ds_dict_diff,
                cache_diff_path,
                hydro_mask=hydro_mask,
                overwrite=True,
                clevel=zarr_clevel,
                chunk_max=zarr_chunk_max,
            )
        except TypeError:
            _save_cache(
                masked_ds_dict_diff,
                cache_diff_path,
                hydro_mask=hydro_mask,
                overwrite=True,
                clevel=zarr_clevel,
            )

    if return_hydro_mask:
        return masked_ds_dict, masked_ds_dict_diff, hydro_mask
    return masked_ds_dict, masked_ds_dict_diff


def load_reference_mean_and_clim(
    data_state="processed",
    experiment="historical",
    period=(1985, 2014),
    hydro_mask=None,
):
    """
    Convenience wrapper that loads:
      - ds_ref_mean: pr_mean/mrro_mean/tran_mean + derived ratios/bgws_tran_mean
      - ds_ref_clim: pr_clim/mrro_clim/tran_clim + derived ratios/bgws_tran_clim

    Optionally applies your existing hydro_mask (recommended, to match model analysis).
    """
    ref_models = ("OBS", "ERA5_land")
    variables = ("pr", "mrro", "tran")

    # ---- period means ----
    ds_ref_mean = load_reference_period_means(
        DATA_DIR,
        data_state=data_state,
        experiment=experiment,
        period=period,
        ref_models=ref_models,
        variables=variables,
        temporal_stats=("mean",),
    )
    ds_ref_mean = load_dat.compute_reference_partition_metrics(ds_ref_mean, suffix="mean")

    # ---- monthly climatology ----
    ds_ref_clim = load_reference_monthly_climatology(
        DATA_DIR,
        data_state=data_state,
        experiment=experiment,
        period=period,
        ref_models=ref_models,
        variables=variables,
        clim_stat="mean",
    )
    ds_ref_clim = load_dat.compute_reference_partition_metrics(ds_ref_clim, suffix="clim")

    # ---- optional: apply hydro mask (same mask as models) ----
    if hydro_mask is not None:
        ds_ref_mean[experiment] = apply_hydro_mask_to_dict(ds_ref_mean[experiment], hydro_mask)
        ds_ref_clim[experiment] = apply_hydro_mask_to_dict(ds_ref_clim[experiment], hydro_mask)

    return ds_ref_mean, ds_ref_clim

def load_reference_period_means(
    BASE_DIR,
    data_state="processed",
    experiment="historical",
    period=(1985, 2014),
    ref_models=("OBS", "ERA5_land"),
    variables=("pr", "mrro", "tran"),
    temporal_stats=("mean",),
):
    """
    Load period means for OBS and ERA5_land as "models", returning:
      ds_ref_mean[experiment][model] with variables like pr_mean, mrro_mean, tran_mean

    Uses your existing load_period_stats() under the hood.
    """
    custom_periods = {experiment: period}

    print(f"Loading reference PERIOD means for {experiment} {period} ...")
    with ProgressBar():
        ds_ref = dask.compute(
            load_dat.load_period_stats(
                BASE_DIR,
                data_state,
                experiments=[experiment],
                models=list(ref_models),
                variables=list(variables),
                custom_periods=custom_periods,
                specific_months_or_seasons=None,
                temporal_stats=temporal_stats,
            )
        )[0]

    # ds_ref has structure ds_ref[experiment][model]
    return ds_ref


def load_reference_monthly_climatology(
    BASE_DIR,
    data_state="processed",
    experiment="historical",
    period=(1985, 2014),
    ref_models=("OBS", "ERA5_land"),
    variables=("pr", "mrro", "tran"),
    clim_stat="mean",
):
    """
    Load monthly climatology (12 months) for OBS and ERA5_land as "models", returning:
      ds_ref_clim[experiment][model] with variables like pr_clim, mrro_clim, tran_clim (dim='month')

    Requires you have load_period_monthly_clim() implemented (as we discussed earlier).
    """
    custom_periods = {experiment: period}

    print(f"Loading reference MONTHLY climatology for {experiment} {period} ...")
    with ProgressBar():
        ds_ref = dask.compute(
            load_dat.load_period_monthly_clim(
                BASE_DIR,
                data_state,
                experiments=[experiment],
                models=list(ref_models),
                variables=list(variables),
                custom_periods=custom_periods,
                clim_stat=clim_stat,
            )
        )[0]

    return ds_ref


def load_period_means_for_models(
    experiments,
    models,
    variables,
    period_by_experiment,
    data_state="processed",
):
    """
    Generic loader for period means using your existing load_period_stats with custom periods.
    period_by_experiment: dict like {"historical": (1985, 1999)}
    """
    with ProgressBar():
        ds_dict = dask.compute(
            load_dat.load_period_stats(
                DATA_DIR,
                data_state,
                experiments=experiments,
                models=models,
                variables=variables,
                custom_periods=period_by_experiment,
                specific_months_or_seasons=None,
                temporal_stats=("mean",),
            )
        )[0]
    return ds_dict


def compute_change_dict(ds_early, ds_late, suffix="mean"):
    """
    ds_early and ds_late have structure:
      ds_early["historical"][model] -> xr.Dataset
      ds_late["historical"][model]  -> xr.Dataset
    Returns ds_change["historical"][model] = ds_late - ds_early
    """
    out = {"historical": {}}
    for model, dsL in ds_late["historical"].items():
        if model not in ds_early["historical"]:
            print(f"[WARN] missing early period for {model}, skipping")
            continue
        dsE = ds_early["historical"][model]
        out["historical"][model] = dsL - dsE
        out["historical"][model].attrs["change_periods"] = "late_minus_early"
        out["historical"][model].attrs["suffix"] = suffix
    return out


def load_historical_15yr_change_for_benchmark(
    hydro_mask=None,
    include_refs=True
):
    """
    Loads early (1985-1999) and late (2000-2014) means for historical.
    Computes partition metrics, applies mask, and returns:
      - ds_early, ds_late, ds_change  (each dict: ["historical"][model]->ds)
    """

    model_list= ['BCC-CSM2-MR_r1i1p1f1', 'CanESM5_r1i1p1f1', 'CESM2_r11i1p1f1', "CMCC-ESM2_r1i1p1f1", 'CNRM-ESM2-1_r1i1p1f2', "EC-Earth3-Veg_r1i1p1f1", 
              "GFDL-ESM4_r1i1p1f1",'IPSL-CM6A-LR_r1i1p1f1', 'MIROC-ES2L_r1i1p1f2', 'MPI-ESM1-2-LR_r11i1p1f1', 'NorESM2-MM_r1i1p1f1', 'UKESM1-0-LL_r1i1p1f2',
    ]
    
    experiment = "historical"
    experiments = [experiment]
    variables = ["pr", "mrro", "tran"]  # only what references have; OK for models too

    # Optionally include OBS and ERA5_land in the same workflow
    models = list(model_list)
    if include_refs:
        for ref in ["OBS", "ERA5_land"]:
            if ref not in models:
                models.append(ref)

    # 15-year windows
    early = {experiment: (1985, 1999)}
    late  = {experiment: (2000, 2014)}

    print("Loading historical early 15-yr means (1985-1999)...")
    ds_early = load_period_means_for_models(
        experiments, models, variables, early, data_state="processed"
    )

    print("Loading historical late 15-yr means (2000-2014)...")
    ds_late = load_period_means_for_models(
        experiments, models, variables, late, data_state="processed"
    )

    # Compute metrics for each period
    ds_early = pro_dat.compute_partition_metrics(
        ds_early, suffix="mean", variants=("tran",), compute_bgws=True, compute_ratios=True
    )
    ds_late = pro_dat.compute_partition_metrics(
        ds_late, suffix="mean", variants=("tran",), compute_bgws=True, compute_ratios=True
    )

    # Apply hydro mask if provided (recommended: use the SAME mask you used elsewhere)
    if hydro_mask is not None:
        ds_early[experiment] = apply_hydro_mask_to_dict(ds_early[experiment], hydro_mask)
        ds_late[experiment]  = apply_hydro_mask_to_dict(ds_late[experiment], hydro_mask)

    # Compute change = late - early
    ds_change = compute_change_dict(ds_early, ds_late, suffix="mean")

    # Compute ensemble mean/median
    ensmean = compute_ensemble_statistic(ds_change[experiment], "mean", "12 model ensemble")
    ds_change[experiment]["12 model ensemble mean"] = ensmean["12 model ensemble mean"]

    ensmedian = compute_ensemble_statistic(ds_change[experiment], "median", "12 model ensemble")
    ds_change[experiment]["12 model ensemble median"] = ensmedian["12 model ensemble median"]

    return ds_early, ds_late, ds_change


import os
import numpy as np
import xarray as xr
import pandas as pd
import dask
from dask.diagnostics import ProgressBar

def driver_period_stats(
    hydro_mask,
    nf_years=(2030, 2059),
    ff_years=(2070, 2099),
    hist_years=(1985, 2014),
):

    CACHE_MAIN = "/work/ch0636/g300115/phd_project/common/data/processed/cache_files_paper_1/period_stats_masked_nf_ff.zarr"
    CACHE_DIFF = "/work/ch0636/g300115/phd_project/common/data/processed/cache_files_paper_1/period_stats_changes_nf_ff.zarr"

    if os.path.exists(CACHE_MAIN) and os.path.exists(CACHE_DIFF):
        print("Cached files exist and are loaded")
        masked_ds_dict, hydro_mask_loaded = load_ds_dict_from_zarr(CACHE_MAIN)
        masked_ds_dict_diff, _ = load_ds_dict_from_zarr(CACHE_DIFF)
        return masked_ds_dict, masked_ds_dict_diff, hydro_mask_loaded

    # ------------------------------------------------------------------
    # Source experiments (on disk) + output-period keys
    # ------------------------------------------------------------------
    source_experiments = ["historical", "ssp126", "ssp370"]

    experiment_periods = {
        "historical": {"historical": hist_years},
        "ssp126": {"ssp126_nf": nf_years, "ssp126_ff": ff_years},
        "ssp370": {"ssp370_nf": nf_years, "ssp370_ff": ff_years},
    }

    out_keys = ["historical", "ssp126_nf", "ssp126_ff", "ssp370_nf", "ssp370_ff"]

     #'CMCC-CM2-SR5_r1i1p1f1', 
     #'CNRM-CM6-1_r1i1p1f2', 

    models = ['BCC-CSM2-MR_r1i1p1f1', 'CanESM5_r1i1p1f1', 'CESM2_r11i1p1f1', "CMCC-ESM2_r1i1p1f1", 'CNRM-ESM2-1_r1i1p1f2', "EC-Earth3-Veg_r1i1p1f1", 
              "GFDL-ESM4_r1i1p1f1",'IPSL-CM6A-LR_r1i1p1f1', 'MIROC-ES2L_r1i1p1f2', 'MPI-ESM1-2-LR_r11i1p1f1', 'NorESM2-MM_r1i1p1f1', 'UKESM1-0-LL_r1i1p1f2',
    ]
    
    ref_model = "BCC-CSM2-MR_r1i1p1f1"

    # monthly drivers (30y means)
    driver_vars = ["wue", "lai", "vpd", "mrsos", "mrso", "gpp", "clt", "rsds", "tas"]
    # yearly RX variables
    rx_vars = ["RX1day", "RX5day", "rx5day_ratio"]

    # ------------------------------------------------------------------
    # Helpers (reuse yours)
    # ------------------------------------------------------------------
    def _ensure_spatial_only(ds):
        out = ds.copy()
        for v in list(out.data_vars):
            da = out[v]
            for dim in ("year", "time"):
                if dim in da.dims and da.sizes.get(dim, 0) > 1:
                    out[v] = da.mean(dim, keep_attrs=True)
        return out

    def _extract_years(values):
        v = np.asarray(values)
        if np.issubdtype(v.dtype, np.datetime64):
            return (v.astype("datetime64[Y]").astype(int) + 1970).astype(int)
        if np.issubdtype(v.dtype, np.integer):
            return v.astype(int)
        if np.issubdtype(v.dtype, np.floating):
            return np.round(v).astype(int)

        out = np.empty(v.shape, dtype=int)
        it = np.nditer(v, flags=["multi_index", "refs_ok"])
        for x in it:
            obj = x.item()
            if obj is None:
                out[it.multi_index] = -9999
                continue
            if isinstance(obj, pd.Timestamp):
                out[it.multi_index] = obj.year
                continue
            if hasattr(obj, "year"):
                out[it.multi_index] = int(obj.year)
                continue
            if isinstance(obj, str):
                try:
                    out[it.multi_index] = int(obj[:4])
                    continue
                except Exception:
                    pass
            try:
                out[it.multi_index] = pd.to_datetime(obj).year
            except Exception:
                raise TypeError(f"Cannot extract year from type {type(obj)} value={obj}")
        return out

    def _get_time_dim(ds):
        if "time" in ds.dims:
            return "time"
        if "year" in ds.dims:
            return "year"
        return None

    def _to_year_index(ds):
        tdim = _get_time_dim(ds)
        if tdim is None:
            return ds
        years = _extract_years(ds[tdim].values)
        out = ds.assign_coords(year=(tdim, years))
        if tdim != "year":
            out = out.swap_dims({tdim: "year"}).drop_vars(tdim)
        out = out.sortby("year")
        _, idx = np.unique(out["year"].values, return_index=True)
        return out.isel(year=idx)

    def _cut_to_period_by_year(ds, start_year, end_year):
        ds_y = _to_year_index(ds)
        if "year" in ds_y.dims:
            return ds_y.sel(year=slice(int(start_year), int(end_year)))
        return ds_y

    def _align_to_ref_years(ds_y, ref_years):
        if "year" not in ds_y.dims:
            return ds_y
        y0 = int(max(ds_y.year.min().item(), ref_years.min().item()))
        y1 = int(min(ds_y.year.max().item(), ref_years.max().item()))
        ds_y = ds_y.sel(year=slice(y0, y1))
        ref_overlap = ref_years.sel(year=slice(y0, y1))
        return ds_y.reindex(year=ref_overlap)

    def _ensure_tas_celsius(ds):
        if "tas" not in ds:
            return ds
    
        da = ds["tas"]
        units = str(da.attrs.get("units", "")).strip().lower()
    
        # 1) Use metadata if available
        if units in ["k", "kelvin"]:
            ds["tas"] = da - 273.15
            ds["tas"].attrs["units"] = "°C"
            return ds
    
        if units in ["c", "degc", "celsius", "°c", "degrees_celsius"]:
            ds["tas"].attrs["units"] = "°C"
            return ds
    
        # 2) Fallback: infer from values
        vals = da.values
        finite_vals = vals[np.isfinite(vals)]
    
        if finite_vals.size == 0:
            return ds  # nothing to infer from
    
        vmin = np.nanmin(finite_vals)
        vmax = np.nanmax(finite_vals)
        vmean = np.nanmean(finite_vals)
    
        # Heuristic:
        # - Kelvin temperatures are usually around 200–330
        # - Celsius temperatures are usually around -80 to 60
        if vmean > 150 or vmax > 100:
            ds["tas"] = da - 273.15
            ds["tas"].attrs["units"] = "°C"
        else:
            ds["tas"].attrs["units"] = "°C"
    
        return ds

    # map output key -> (source experiment, start, end)
    outkey_to_src = {}
    for src, mapping in experiment_periods.items():
        for out_key, (y0, y1) in mapping.items():
            outkey_to_src[out_key] = (src, y0, y1)

    # ------------------------------------------------------------------
    # Step 1: Load driver period means via your upgraded load_period_stats
    # ------------------------------------------------------------------
    print("Loading driver 30-year means (NF/FF)...")
    with ProgressBar():
        ds_dict_driver_means = dask.compute(
            load_dat.load_period_stats(
                DATA_DIR,
                "processed",
                source_experiments,
                models,
                driver_vars,
                specific_months_or_seasons=None,
                temporal_stats=("mean",),
                experiment_periods=experiment_periods,   # <-- key change
            )
        )[0]
    # ds_dict_driver_means keys are now out_keys (historical, ssp126_nf, ...)

    # Convert tas to Celsius if needed
    for out_key in ds_dict_driver_means:
        for model in ds_dict_driver_means[out_key]:
            ds_dict_driver_means[out_key][model] = _ensure_tas_celsius(
                ds_dict_driver_means[out_key][model]
            )

    # ------------------------------------------------------------------
    # Step 2: Compute pr/vpd seasonality directly from raw monthly pr/vpd
    # ------------------------------------------------------------------
    print("Loading monthly pr/vpd and computing seasonality (NF/FF)...")
    with ProgressBar():
        ds_prvpd_raw = dask.compute(
            load_dat.load_multiple_models_and_experiments(
                DATA_DIR, "processed", source_experiments, "month", models, ["pr", "vpd"]
            )
        )[0]  # ds_prvpd_raw[src_exp][model] -> Dataset

    seasonality = {k: {} for k in out_keys}
    for out_key in out_keys:
        src, y0, y1 = outkey_to_src[out_key]
        for model in models:
            ds_m = ds_prvpd_raw.get(src, {}).get(model, None)
            if ds_m is None:
                continue

            # use your robust period selector on dict-of-ds
            ds_sel_dict = pro_dat.select_period({model: ds_m}, start_year=y0, end_year=y1)
            ds_sel = ds_sel_dict.get(model, None)
            if ds_sel is None or "time" not in ds_sel.dims:
                continue

            ds_out = xr.Dataset()
            if "pr" in ds_sel:
                pr_clim = ds_sel["pr"].groupby("time.month").mean("time", skipna=True)
                ds_out["pr_seasonality"] = pr_clim.std("month", skipna=True).astype("float32")
            if "vpd" in ds_sel:
                vpd_clim = ds_sel["vpd"].groupby("time.month").mean("time", skipna=True)
                ds_out["vpd_seasonality"] = vpd_clim.std("month", skipna=True).astype("float32")

            if ds_out.data_vars:
                seasonality[out_key][model] = ds_out

    # ------------------------------------------------------------------
    # Step 3: Load RX vars (yearly), compute period means per out_key safely
    # ------------------------------------------------------------------
    print("Loading RX1day/RX5day (yearly) and computing NF/FF means...")
    ds_rx_by_var = {}
    for v in rx_vars:
        with ProgressBar():
            ds_rx_by_var[v] = dask.compute(
                load_dat.load_multiple_models_and_experiments(
                    DATA_DIR, "processed", source_experiments, "year", models, [v]
                )
            )[0]

    ds_dict_rx_means = {k: {} for k in out_keys}

    for out_key in out_keys:
        src, y0, y1 = outkey_to_src[out_key]

        # reference years from BCC in this src+period
        ref_years = None
        ref_ds = None
        for v in rx_vars:
            ref_ds = ds_rx_by_var[v].get(src, {}).get(ref_model, None)
            if ref_ds is not None:
                break
        if ref_ds is not None:
            ref_cut = _cut_to_period_by_year(ref_ds, y0, y1)
            if "year" in ref_cut.dims:
                ref_years = ref_cut["year"]

        for model in models:
            rx_parts = []
            for v in rx_vars:
                ds_v = ds_rx_by_var[v].get(src, {}).get(model, None)
                if ds_v is None:
                    continue

                ds_v = _cut_to_period_by_year(ds_v, y0, y1)
                if ref_years is not None and "year" in ds_v.dims:
                    ds_v = _align_to_ref_years(ds_v, ref_years)
                rx_parts.append(ds_v)

            if not rx_parts:
                continue

            ds_rx_merged = xr.merge(rx_parts, compat="override")
            ds_rx_merged = _ensure_spatial_only(ds_rx_merged)
            ds_dict_rx_means[out_key][model] = ds_rx_merged

    # ------------------------------------------------------------------
    # Step 4: Merge drivers + RX + seasonality into one dataset per model/out_key
    # ------------------------------------------------------------------
    ds_dict_all = {k: {} for k in out_keys}
    for out_key in out_keys:
        for model in models:
            parts = []

            if model in ds_dict_driver_means.get(out_key, {}):
                parts.append(ds_dict_driver_means[out_key][model])

            if model in ds_dict_rx_means.get(out_key, {}):
                parts.append(ds_dict_rx_means[out_key][model])

            if model in seasonality.get(out_key, {}):
                parts.append(seasonality[out_key][model])

            if not parts:
                continue

            ds_dict_all[out_key][model] = xr.merge(parts, compat="override")

    # ------------------------------------------------------------------
    # Step 5: Apply hydrological mask + ensemble mean/median for each out_key
    # ------------------------------------------------------------------
    masked_ds_dict = {}
    for out_key in out_keys:
        masked_ds_dict[out_key] = apply_hydro_mask_to_dict(ds_dict_all[out_key], hydro_mask)

        ensmean = compute_ensemble_statistic(masked_ds_dict[out_key], "mean", "12 model ensemble")
        masked_ds_dict[out_key]["12 model ensemble mean"] = ensmean["12 model ensemble mean"]

        ensmedian = compute_ensemble_statistic(masked_ds_dict[out_key], "median", "12 model ensemble")
        masked_ds_dict[out_key]["12 model ensemble median"] = ensmedian["12 model ensemble median"]

        ensstd = compute_ensemble_statistic(masked_ds_dict[out_key], "std", "12 model ensemble")
        masked_ds_dict[out_key]["12 model ensemble std"] = ensstd["12 model ensemble std"]

    # ------------------------------------------------------------------
    # Step 6: Compute changes (vs historical, and FF-NF)
    # ------------------------------------------------------------------
    masked_ds_dict_diff = {}

    for k in ["ssp126_nf", "ssp126_ff", "ssp370_nf", "ssp370_ff"]:
        masked_ds_dict_diff[f"{k}-historical"] = pro_dat.compute_diff_dict(
            masked_ds_dict, reference_key="historical", comparison_key=k, var_rel_change=None
        )

    masked_ds_dict_diff["ssp126_ff-ssp126_nf"] = pro_dat.compute_diff_dict(
        masked_ds_dict, reference_key="ssp126_nf", comparison_key="ssp126_ff", var_rel_change=None
    )
    masked_ds_dict_diff["ssp370_ff-ssp370_nf"] = pro_dat.compute_diff_dict(
        masked_ds_dict, reference_key="ssp370_nf", comparison_key="ssp370_ff", var_rel_change=None
    )

    # ------------------------------------------------------------------
    # Step 7: Ensemble mean/median for diffs
    # ------------------------------------------------------------------
    for diff_key in list(masked_ds_dict_diff.keys()):
        ensmean = compute_ensemble_statistic(masked_ds_dict_diff[diff_key], "mean", "12 model ensemble")
        masked_ds_dict_diff[diff_key]["12 model ensemble mean"] = ensmean["12 model ensemble mean"]

        ensmedian = compute_ensemble_statistic(masked_ds_dict_diff[diff_key], "median", "12 model ensemble")
        masked_ds_dict_diff[diff_key]["12 model ensemble median"] = ensmedian["12 model ensemble median"]

        ensstd = compute_ensemble_statistic(masked_ds_dict_diff[diff_key], "std", "12 model ensemble")
        masked_ds_dict_diff[diff_key]["12 model ensemble std"] = ensstd["12 model ensemble std"]

    # ------------------------------------------------------------------
    # Step 8: Cache + return
    # ------------------------------------------------------------------
    save_ds_dict_to_zarr(masked_ds_dict, CACHE_MAIN, hydro_mask=hydro_mask, overwrite=True)
    save_ds_dict_to_zarr(masked_ds_dict_diff, CACHE_DIFF, hydro_mask=hydro_mask, overwrite=True)

    return masked_ds_dict, masked_ds_dict_diff, hydro_mask

import os
import re
import shutil
import xarray as xr
import zarr
from numcodecs import Blosc

def _safe_name(s: str) -> str:
    # safe for zarr group paths
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", s).strip("_")

def save_ds_dict_to_zarr(ds_dict, store_path, hydro_mask=None, overwrite=True,
                         chunk_hint=("lat", "lon"), clevel=5):
    """
    Save nested dict: ds_dict[exp][model] = xr.Dataset into one Zarr store
    using groups: /<exp>/<model>.
    Optionally writes hydro_mask into group '/_meta'.
    """
    if overwrite and os.path.exists(store_path):
        shutil.rmtree(store_path)

    compressor = Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.BITSHUFFLE)

    # Write each dataset into its own group
    for exp, models in ds_dict.items():
        exp_g = _safe_name(exp)
        for model, ds in models.items():
            model_g = _safe_name(model)

            # Make sure data are not object dtype + reduce size if you can
            ds_to_write = ds.copy()

            # optional: cast floats to float32 (big space saver)
            for v in ds_to_write.data_vars:
                if str(ds_to_write[v].dtype).startswith("float"):
                    ds_to_write[v] = ds_to_write[v].astype("float32")

            # optional chunking (only if dims exist)
            chunks = {}
            for d in chunk_hint:
                if d in ds_to_write.dims:
                    # pick a reasonable chunk size; adjust to your grid
                    chunks[d] = min(ds_to_write.sizes[d], 256)
            if chunks:
                ds_to_write = ds_to_write.chunk(chunks)

            encoding = {v: {"compressor": compressor} for v in ds_to_write.data_vars}

            ds_to_write.to_zarr(
                store_path,
                group=f"{exp_g}/{model_g}",
                mode="a",
                encoding=encoding,
                consolidated=False,
            )

    # write hydro_mask (optional)
    if hydro_mask is not None:
        meta = xr.Dataset({"hydro_mask": hydro_mask})
        meta.to_zarr(store_path, group="_meta", mode="a", consolidated=False)

    # consolidate metadata for faster opens
    zarr.consolidate_metadata(store_path)

def load_ds_dict_from_zarr(store_path):
    """
    Reconstruct nested dict from Zarr store with groups /<exp>/<model>.

    Backwards-compatibility fix:
      - renames "12_model_ensemble_mean"   -> "12 model ensemble mean"
      - renames "12_model_ensemble_median" -> "12 model ensemble median"
    """
    root = zarr.open_consolidated(store_path, mode="r")

    def _restore_ensemble_key(k: str) -> str:
        if k == "12_model_ensemble_mean":
            return "12 model ensemble mean"
        if k == "12_model_ensemble_median":
            return "12 model ensemble median"
        if k == "9_model_ensemble_mean":
            return "9 model ensemble mean"
        if k == "9_model_ensemble_median":
            return "9 model ensemble median"
        return k

    out = {}
    for exp in root.group_keys():
        if exp == "_meta":
            continue

        out[exp] = {}
        exp_grp = root[exp]

        for model in exp_grp.group_keys():
            model_key = _restore_ensemble_key(model)
            out[exp][model_key] = xr.open_zarr(
                store_path,
                group=f"{exp}/{model}",
                consolidated=True
            )

    hydro_mask = None
    if "_meta" in root:
        hydro_mask = xr.open_zarr(store_path, group="_meta", consolidated=True)["hydro_mask"]

    return out, hydro_mask


import gc
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import regionmask
import numpy as np


def _apply_hydro_mask(ds: xr.Dataset, hydro_mask: xr.DataArray) -> xr.Dataset:
    hm = hydro_mask.astype(bool)
    return ds.where(hm)


def _monthly_to_annual_mean(ds: xr.Dataset, varnames) -> xr.Dataset:
    ds = ds[list(varnames)]
    ann = ds.groupby("time.year").mean("time", keep_attrs=True).rename({"year": "time"})
    ann["time"] = ann["time"].astype(int)  # years as ints
    return ann


def _get_ar6_land_regions_and_indices(region_abbrevs):
    ar6 = regionmask.defined_regions.ar6.land
    abbrevs = list(ar6.abbrevs)

    missing = [ab for ab in region_abbrevs if ab not in abbrevs]
    if missing:
        raise ValueError(f"Unknown AR6 land abbrevs: {missing}. Check ar6.abbrevs.")

    idxs = [abbrevs.index(ab) for ab in region_abbrevs]
    return ar6, idxs


def _cell_area_weights(ds, lat_name="lat", area_var=None):
    if area_var is not None and area_var in ds:
        w = ds[area_var]
        return w / w.mean()
    lat = ds[lat_name]
    return np.cos(np.deg2rad(lat))


def _regional_mean_fluxes_ar6(
    ds_ann: xr.Dataset,
    region_abbrevs,
    lat_name="lat",
    lon_name="lon",
    area_var=None,
):
    ar6, idxs = _get_ar6_land_regions_and_indices(region_abbrevs)

    # Region mask (region, lat, lon)
    mask_3D = ar6.mask_3D(ds_ann, lon_name=lon_name, lat_name=lat_name)

    w = _cell_area_weights(ds_ann, lat_name=lat_name, area_var=area_var)

    reg_dsets = []
    for ab, ridx in zip(region_abbrevs, idxs):
        rmask = mask_3D.isel(region=ridx)
        rds = ds_ann.where(rmask)
        rmean = rds.weighted(w).mean(dim=(lat_name, lon_name), skipna=True)
        rmean = rmean.expand_dims(region=[ab])
        reg_dsets.append(rmean)

    out = xr.concat(reg_dsets, dim="region").assign_coords(region=region_abbrevs)
    return out


import os
import gc
import xarray as xr

# assumes these already exist in your environment:
# - save_nested_dict_zarr, load_nested_dict_zarr
# - load_dat.open_models_variables
# - _get_ar6_land_regions_and_indices
# - _apply_hydro_mask
# - _monthly_to_annual_mean
# - _cell_area_weights
# - pro_dat.compute_partition_metrics
# - compute_ensemble_statistic

def bgws_yearly_timeseries_ar6_streamed(
    DATA_DIR,
    hydro_mask,
    focus_regions=("WNA", "SES", "WCE", "SAS", "CAF", "EAU"),
    experiments=("historical", "ssp126", "ssp370"),
    models=None,
    variables=("pr", "mrro", "tran", "evapo", "evspsbl", "mrso"),
    hist_years=(1985, 2014),
    scen_years=(2015, 2100),
    area_var=None,
    compute_mmm=False,
    materialize_per_model=True,

    # ---- NEW: caching ----
    cache_dir=None,
    cache_name="bgws_yearly_timeseries_ar6.zarr",
    use_cache=True,
    overwrite_cache=False,
    zarr_clevel=5,
    zarr_chunk_max=256,

    # ---- optional: if you want the hydro_mask back when loading from cache ----
    return_hydro_mask=False,
):
    """
    Stream loads one (exp, model) at a time to avoid huge memory usage.
    Returns out[exp][model] datasets containing annual regional fluxes + BGWS metrics.

    NEW: Optional Zarr caching of the *final* output dict (including computed metrics and,
         optionally, MMM products). If cache exists, it is loaded and returned.
    """

    # -----------------------------
    # Cache: load if available
    # -----------------------------
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, cache_name)

    if use_cache and (cache_path is not None) and os.path.exists(cache_path) and (not overwrite_cache):
        print(f"[CACHE] loading yearly AR6 time series: {cache_path}")
        out_cached, hydro_cached = load_nested_dict_zarr(cache_path)
        if return_hydro_mask:
            return out_cached, hydro_cached
        return out_cached

    # -----------------------------
    # Defaults
    # -----------------------------
    if models is None:
        ['BCC-CSM2-MR_r1i1p1f1', 'CanESM5_r1i1p1f1', 'CESM2_r11i1p1f1', "CMCC-ESM2_r1i1p1f1", 'CNRM-ESM2-1_r1i1p1f2', "EC-Earth3-Veg_r1i1p1f1", 
         "GFDL-ESM4_r1i1p1f1",'IPSL-CM6A-LR_r1i1p1f1', 'MIROC-ES2L_r1i1p1f2', 'MPI-ESM1-2-LR_r11i1p1f1', 'NorESM2-MM_r1i1p1f1', 'UKESM1-0-LL_r1i1p1f2',
        ]

    out = {exp: {} for exp in experiments}

    # Pre-resolve AR6 region indices once
    ar6, idxs = _get_ar6_land_regions_and_indices(list(focus_regions))

    # Optional progress bar (only if available)
    try:
        from dask.diagnostics import ProgressBar
        _PB = ProgressBar
    except Exception:
        _PB = None

    for exp in experiments:
        print(f"\n=== Experiment: {exp} ===")
        for model_name in models:
            print(f" -> {model_name}")

            # 1) Load monthly for ONE model + experiment
            ds, valid_vars = load_dat.open_models_variables(
                DATA_DIR, "processed", exp, "month", model_name, list(variables)
            )
            if ds is None:
                print(f"    [SKIP] No valid data for {model_name} ({exp})")
                continue

            try:
                # 2) Time subset
                if exp == "historical":
                    ds = ds.sel(time=slice(f"{hist_years[0]}-01-01", f"{hist_years[1]}-12-31"))
                else:
                    ds = ds.sel(time=slice(f"{scen_years[0]}-01-01", f"{scen_years[1]}-12-31"))

                # 3) Hydro mask on grid
                ds = _apply_hydro_mask(ds, hydro_mask)

                # 4) Annual mean of RAW fluxes
                ds_ann = _monthly_to_annual_mean(ds, variables)

                # Rename to match compute_partition_metrics expectation
                ds_ann = ds_ann.rename({v: f"{v}_mean" for v in variables})

                # 5) Regional mean of annual fluxes (AR6 land)
                mask_3D = ar6.mask_3D(ds_ann)
                w = _cell_area_weights(ds_ann, lat_name="lat", area_var=area_var)

                reg_dsets = []
                for ab, ridx in zip(focus_regions, idxs):
                    rmask = mask_3D.isel(region=ridx)
                    rds = ds_ann.where(rmask)
                    rmean = rds.weighted(w).mean(dim=("lat", "lon"), skipna=True)
                    rmean = rmean.expand_dims(region=[ab])
                    reg_dsets.append(rmean)

                ds_regflux = xr.concat(reg_dsets, dim="region").assign_coords(region=list(focus_regions))

                # ---- OPTIONAL: storage change from soil moisture (mrso) ----
                # mrso_mean is storage S [mm] (annual mean). dS is year-to-year change in S [mm].
                if "mrso_mean" in ds_regflux.data_vars:
                    dS = ds_regflux["mrso_mean"].diff("time")
                    ds_regflux["dS_mean"] = dS.reindex(time=ds_regflux["time"])  # mm per year step

                if ("dS_mean" in ds_regflux) and ("pr_mean" in ds_regflux):
                    pr_yr = ds_regflux["pr_mean"] * 365
                    ds_regflux["dS_over_P_mean"] = (ds_regflux["dS_mean"] / pr_yr.where(pr_yr > 1e-6)) * 100.0

                if all(v in ds_regflux for v in ["pr_mean", "mrro_mean", "evapo_mean", "dS_mean"]):
                    pr_yr = ds_regflux["pr_mean"] * 365
                    r_yr  = ds_regflux["mrro_mean"] * 365
                    et_yr = ds_regflux["evapo_mean"] * 365
                    ds_regflux["wb_resid_mm_yr"] = pr_yr - r_yr - et_yr - ds_regflux["dS_mean"]
                    ds_regflux["wb_resid_over_P"] = (ds_regflux["wb_resid_mm_yr"] / pr_yr.where(pr_yr > 1e-6)) * 100.0

                # 6) Materialize small output now (recommended)
                if materialize_per_model and _PB is not None:
                    with _PB():
                        ds_regflux = ds_regflux.compute()
                elif materialize_per_model:
                    ds_regflux = ds_regflux.compute()

                out[exp][model_name] = ds_regflux

            finally:
                # 7) Close & free monthly dataset
                try:
                    ds.close()
                except Exception:
                    pass
                del ds
                gc.collect()

    # 8) Compute metrics on the annual regional fluxes
    out = pro_dat.compute_partition_metrics(
        out,
        suffix="mean",
        variants=("tran", "evapo", "evspsbl"),
        compute_bgws=True,
        compute_ratios=True,
    )

    # 9) Optional MMM stats across models (one model = one vote)
    if compute_mmm:
        for exp in experiments:
            ensmean = compute_ensemble_statistic(out[exp], "mean", "12 model ensemble")
            out[exp]["12 model ensemble mean"] = ensmean["12 model ensemble mean"]

            ensmedian = compute_ensemble_statistic(out[exp], "median", "12 model ensemble")
            out[exp]["12 model ensemble median"] = ensmedian["12 model ensemble median"]

    # -----------------------------
    # Cache: save final output
    # -----------------------------
    if cache_path is not None:
        print(f"[CACHE] saving yearly AR6 time series: {cache_path}")
        save_nested_dict_zarr(
            out,
            cache_path,
            hydro_mask=hydro_mask,
            overwrite=True,
            clevel=zarr_clevel,
            chunk_max=zarr_chunk_max,
        )

    if return_hydro_mask:
        return out, hydro_mask
    return out

def _subset_by_year(ds: xr.Dataset, y0: int, y1: int, time_dim: str = "time") -> xr.Dataset:
    """Calendar-safe subsetting using dt.year (works for cftime + datetime64)."""
    if ds is None:
        return ds
    if time_dim in ds.dims and time_dim in ds.coords:
        yrs = ds[time_dim].dt.year
        return ds.where((yrs >= int(y0)) & (yrs <= int(y1)), drop=True)
    return ds

def _to_int_year_dim(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert ds with a time-like dim into an integer 'year' dimension.
    - If 'time' exists: year = time.dt.year
    - If 'year' exists: ensure int
    Sorts and drops duplicate years.
    """
    if ds is None:
        return ds

    if "year" in ds.dims:
        y = ds["year"]
        if not np.issubdtype(y.dtype, np.integer):
            try:
                y = y.dt.year.astype(int)
            except Exception:
                y = y.astype(int)
        ds = ds.assign_coords(year=y.astype(int))

    elif "time" in ds.dims:
        # y is a DataArray with dim "time"
        y = ds["time"].dt.year.astype(int)
        # IMPORTANT: assign as DataArray (NOT as ("time", y))
        ds = ds.assign_coords(year=y)
        ds = ds.swap_dims({"time": "year"}).drop_vars("time")

    else:
        return ds

    ds = ds.sortby("year")
    _, idx = np.unique(ds["year"].values, return_index=True)
    ds = ds.isel(year=np.sort(idx))
    return ds


def _annual_seasonality_std_from_monthly(da_mon: xr.DataArray) -> xr.DataArray:
    """
    Year-by-year "seasonality" = std across the 12 monthly values within each year.
    (Matches your period-stats idea of std across months, but done per-year.)
    Returns a DataArray with dim 'time' = integer years.
    """
    seas = da_mon.groupby("time.year").std("time", skipna=True).rename({"year": "time"})
    seas["time"] = seas["time"].astype(int)
    return seas


def drivers_yearly_timeseries_ar6_streamed(
    DATA_DIR,
    hydro_mask,
    focus_regions=("WNA", "SES", "WCE", "SAS", "CAF", "EAU"),
    experiments=("historical", "ssp126", "ssp370"),
    models=None,
    drivers=(
        "pr_mean",
        "pr_seasonality",
        "RX5day",
        "rsds_mean",
        "vpd_mean",
        "mrso_mean",
        "mrsos_mean",
        "lai_mean",
        "wue_mean",
    ),
    hist_years=(1985, 2014),
    scen_years=(2015, 2100),
    area_var=None,
    compute_mmm=False,
    materialize_per_model=True,

    # ---- caching ----
    cache_dir=None,
    cache_name="drivers_yearly_timeseries_ar6.zarr",
    use_cache=True,
    overwrite_cache=False,
    zarr_clevel=5,
    zarr_chunk_max=256,

    return_hydro_mask=False,
):
    """
    Stream-load yearly AR6 regional time series for your predictor variables, analogous to
    bgws_yearly_timeseries_ar6_streamed(), and cache the nested dict to Zarr.

    - *_mean drivers: loaded monthly -> annual mean -> renamed to *_mean
    - pr_seasonality: computed per-year as std across monthly pr within each year
    - RX5day: LOADED (yearly) from disk (no computation)
    """

    # -----------------------------
    # Cache: load if available
    # -----------------------------
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, cache_name)

    # Use whichever cache helpers you have available
    _load_cache = globals().get("load_nested_dict_zarr", None) or globals().get("load_ds_dict_from_zarr", None)
    _save_cache = globals().get("save_nested_dict_zarr", None) or globals().get("save_ds_dict_to_zarr", None)
    if _load_cache is None or _save_cache is None:
        raise RuntimeError("Could not find cache helpers. Need either (save/load)_nested_dict_zarr or save/load_ds_dict_from_zarr.")

    if use_cache and (cache_path is not None) and os.path.exists(cache_path) and (not overwrite_cache):
        print(f"[CACHE] loading yearly AR6 driver time series: {cache_path}")
        out_cached, hydro_cached = _load_cache(cache_path)
        if return_hydro_mask:
            return out_cached, hydro_cached
        return out_cached

    ref_model = "BCC-CSM2-MR_r1i1p1f1"

    # -----------------------------
    # Defaults
    # -----------------------------
    if models is None:
        [
          'BCC-CSM2-MR_r1i1p1f1', 'CanESM5_r1i1p1f1', 'CESM2_r11i1p1f1', "CMCC-ESM2_r1i1p1f1", 'CNRM-ESM2-1_r1i1p1f2', "EC-Earth3-Veg_r1i1p1f1", 
          "GFDL-ESM4_r1i1p1f1",'IPSL-CM6A-LR_r1i1p1f1', 'MIROC-ES2L_r1i1p1f2', 'MPI-ESM1-2-LR_r11i1p1f1', 'NorESM2-MM_r1i1p1f1', 'UKESM1-0-LL_r1i1p1f2',
        ]

    out = {exp: {} for exp in experiments}

    # Pre-resolve AR6 region indices once
    ar6, idxs = _get_ar6_land_regions_and_indices(list(focus_regions))

    # Optional progress bar
    try:
        from dask.diagnostics import ProgressBar
        _PB = ProgressBar
    except Exception:
        _PB = None

    # -----------------------------
    # Parse requested drivers
    # -----------------------------
    drivers = list(drivers)

    need_pr_seas = "pr_seasonality" in drivers
    monthly_means = [d for d in drivers if d.endswith("_mean")]
    monthly_base = sorted({d.replace("_mean", "") for d in monthly_means} | ({"pr"} if need_pr_seas else set()))

    yearly_vars = []
    if "RX5day" in drivers:
        yearly_vars.append("RX5day")

    for exp in experiments:
        print(f"\n=== Experiment: {exp} ===")

        # determine years for this experiment
        if exp == "historical":
            y0, y1 = hist_years
        else:
            y0, y1 = scen_years
    
        # --- NEW: reference years from ref_model for yearly vars (RX5day) ---
        ref_years = None
        if yearly_vars and ref_model is not None:
            ds_ref_year, _ = load_dat.open_models_variables(
                DATA_DIR, "processed", exp, "year", ref_model, list(yearly_vars)
            )
            if ds_ref_year is not None:
                ds_ref_year = _subset_by_year(ds_ref_year, y0, y1, time_dim="time")
                ds_ref_year = _to_int_year_dim(ds_ref_year)
                if "year" in ds_ref_year.dims:
                    ref_years = ds_ref_year["year"]
                try:
                    ds_ref_year.close()
                except Exception:
                    pass
                del ds_ref_year
                
        for model_name in models:
            print(f" -> {model_name}")

            ds_mon = None
            ds_year = None
            try:
                # 1) Load monthly base vars needed
                valid_mon = []
                if monthly_base:
                    ds_mon, valid_mon = load_dat.open_models_variables(
                        DATA_DIR, "processed", exp, "month", model_name, list(monthly_base)
                    )

                # 2) Load yearly vars needed (e.g., RX5day)
                valid_year = []
                if yearly_vars:
                    ds_year, valid_year = load_dat.open_models_variables(
                        DATA_DIR, "processed", exp, "year", model_name, list(yearly_vars)
                    )

                if ds_mon is None and ds_year is None:
                    print(f"    [SKIP] No valid driver data for {model_name} ({exp})")
                    continue

                if ds_mon is not None:
                    # calendar-safe monthly subset (recommended)
                    ds_mon = _subset_by_year(ds_mon, y0, y1, time_dim="time")
                    ds_mon = _apply_hydro_mask(ds_mon, hydro_mask)
        
                if ds_year is not None:
                    # calendar-safe yearly subset
                    ds_year = _subset_by_year(ds_year, y0, y1, time_dim="time")
                    ds_year = _apply_hydro_mask(ds_year, hydro_mask)
        
                    # convert to integer year axis
                    ds_year = _to_int_year_dim(ds_year)
        
                    # align to reference model's years (prevents union/merge weirdness)
                    if ref_years is not None and "year" in ds_year.dims:
                        ds_year = ds_year.reindex(year=ref_years)
        
                    # finally rename year->time to match your annual ds_ann time (int years)
                    if "year" in ds_year.dims:
                        ds_year = ds_year.rename({"year": "time"})
                        ds_year["time"] = ds_year["time"].astype(int)

                # 4) Build annual driver dataset on grid
                parts = []

                # (a) annual means from monthly vars
                if ds_mon is not None and len(valid_mon) > 0:
                    ds_ann = _monthly_to_annual_mean(ds_mon, valid_mon)
                    ds_ann = ds_ann.rename({v: f"{v}_mean" for v in valid_mon})

                    keep = [d for d in monthly_means if d in ds_ann.data_vars]
                    if keep:
                        parts.append(ds_ann[keep])

                    # (b) pr_seasonality
                    if need_pr_seas and ("pr" in ds_mon.data_vars):
                        pr_seas = _annual_seasonality_std_from_monthly(ds_mon["pr"]).astype("float32")
                        parts.append(pr_seas.to_dataset(name="pr_seasonality"))

                # (c) yearly RX5day (loaded, not computed)
                if ds_year is not None and len(valid_year) > 0:
                    keep_y = [v for v in yearly_vars if v in ds_year.data_vars and v in drivers]
                    if keep_y:
                        parts.append(ds_year[keep_y])

                if not parts:
                    print(f"    [SKIP] No requested drivers available after processing for {model_name} ({exp})")
                    continue

                ds_drv = xr.merge(parts, compat="override")

                # 5) Regional mean (AR6 land)
                mask_3D = ar6.mask_3D(ds_drv)
                w = _cell_area_weights(ds_drv, lat_name="lat", area_var=area_var)

                reg_dsets = []
                for ab, ridx in zip(focus_regions, idxs):
                    rmask = mask_3D.isel(region=ridx)
                    rds = ds_drv.where(rmask)
                    rmean = rds.weighted(w).mean(dim=("lat", "lon"), skipna=True)
                    rmean = rmean.expand_dims(region=[ab])
                    reg_dsets.append(rmean)

                ds_reg = xr.concat(reg_dsets, dim="region").assign_coords(region=list(focus_regions))

                # 6) Materialize small output now
                if materialize_per_model and _PB is not None:
                    with _PB():
                        ds_reg = ds_reg.compute()
                elif materialize_per_model:
                    ds_reg = ds_reg.compute()

                out[exp][model_name] = ds_reg

            finally:
                for _ds in (ds_mon, ds_year):
                    if _ds is not None:
                        try:
                            _ds.close()
                        except Exception:
                            pass
                del ds_mon, ds_year
                gc.collect()

    # 7) Optional MMM stats across models
    if compute_mmm:
        for exp in experiments:
            ensmean = compute_ensemble_statistic(out[exp], "mean", "12 model ensemble")
            out[exp]["12 model ensemble mean"] = ensmean["12 model ensemble mean"]

            ensmedian = compute_ensemble_statistic(out[exp], "median", "12 model ensemble")
            out[exp]["12 model ensemble median"] = ensmedian["12 model ensemble median"]

    # -----------------------------
    # Cache: save final output
    # -----------------------------
    if cache_path is not None:
        print(f"[CACHE] saving yearly AR6 driver time series: {cache_path}")
        try:
            # nested_dict_zarr-style
            _save_cache(
                out,
                cache_path,
                hydro_mask=hydro_mask,
                overwrite=True,
                clevel=zarr_clevel,
                chunk_max=zarr_chunk_max,
            )
        except TypeError:
            # ds_dict_to_zarr-style (your posted helper)
            _save_cache(
                out,
                cache_path,
                hydro_mask=hydro_mask,
                overwrite=True,
                clevel=zarr_clevel,
            )

    if return_hydro_mask:
        return out, hydro_mask
    return out


import os
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from scipy.stats import kendalltau


def driver_eli_historical(
    hydro_mask,
    periods=None,
    min_valid_months=30,
):
    """
    Compute historical ELI for three periods and return a ds_dict-style object.

    Output keys by default:
      - historical_1985_1994
      - historical_1995_2004
      - historical_2005_2014

    Each dataset contains:
      - ELI
      - corr_sm_et
      - corr_rsds_et
      - n_valid_months
    """

    if periods is None:
        periods = {
            "historical_1985_1994": (1985, 1994),
            "historical_1995_2004": (1995, 2004),
            "historical_2005_2014": (2005, 2014),
        }

    CACHE_MAIN = "/work/ch0636/g300115/phd_project/common/data/processed/cache_files_paper_1/eli_historical_periods.zarr"

    if os.path.exists(CACHE_MAIN):
        print("Cached ELI file exists and is loaded")
        masked_ds_dict, hydro_mask_loaded = load_ds_dict_from_zarr(CACHE_MAIN)
        return masked_ds_dict, hydro_mask_loaded

    source_experiment = "historical"

    models = [
        "BCC-CSM2-MR_r1i1p1f1", "CanESM5_r1i1p1f1",
        "CESM2_r11i1p1f1", "CMCC-CM2-SR5_r1i1p1f1", "CNRM-CM6-1_r1i1p1f2",
        "CNRM-ESM2-1_r1i1p1f2",
        "IPSL-CM6A-LR_r1i1p1f1", "MIROC-ES2L_r1i1p1f2",
        "MPI-ESM1-2-LR_r11i1p1f1", "NorESM2-MM_r1i1p1f1", "UKESM1-0-LL_r1i1p1f2",
    ]

    eli_vars = ["mrso", "evspsbl", "rsds", "tas"]

    # ------------------------------------------------------------------
    # Load monthly data once
    # ------------------------------------------------------------------
    print("Loading monthly mrso, evspsbl, rsds, tas for historical...")
    with ProgressBar():
        ds_raw = dask.compute(
            load_dat.load_multiple_models_and_experiments(
                DATA_DIR,
                "processed",
                [source_experiment],
                "month",
                models,
                eli_vars,
            )
        )[0]

    # ------------------------------------------------------------------
    # Compute ELI for each model and period
    # ------------------------------------------------------------------
    ds_dict_eli = {k: {} for k in periods}

    for out_key, (y0, y1) in periods.items():
        print(f"Computing ELI for {out_key} ({y0}-{y1})")

        for model in models:
            ds_m = ds_raw.get(source_experiment, {}).get(model, None)
            if ds_m is None:
                continue

            ds_sel_dict = pro_dat.select_period({model: ds_m}, start_year=y0, end_year=y1)
            ds_sel = ds_sel_dict.get(model, None)

            if ds_sel is None or "time" not in ds_sel.dims:
                continue

            try:
                ds_eli = compute_eli_for_period(
                    ds_sel,
                    min_valid_months=min_valid_months,
                )

                # Compute here: result is already reduced to 2D fields,
                # so this is a good place to materialize.
                with ProgressBar():
                    ds_eli = ds_eli.compute()

                ds_dict_eli[out_key][model] = ds_eli

            except Exception as e:
                print(f"[WARN] Failed ELI for {model} in {out_key}: {e}")

        print(f"{out_key}: successful models = {len(ds_dict_eli[out_key])}")

    # ------------------------------------------------------------------
    # Apply hydrological mask + ensemble mean
    # ------------------------------------------------------------------
    masked_ds_dict = {}
    for out_key in periods:
        masked_ds_dict[out_key] = apply_hydro_mask_to_dict(ds_dict_eli[out_key], hydro_mask)

        if not masked_ds_dict[out_key]:
            print(f"[WARN] No successful ELI datasets for {out_key}; skipping ensemble statistics")
            continue

        ensmean = compute_ensemble_statistic(
            masked_ds_dict[out_key],
            "mean",
            "12 model ensemble"
        )

        if "12 model ensemble mean" in ensmean:
            with ProgressBar():
                masked_ds_dict[out_key]["12 model ensemble mean"] = (
                    ensmean["12 model ensemble mean"].compute()
                )

    # ------------------------------------------------------------------
    # Cache + return
    # ------------------------------------------------------------------
    save_ds_dict_to_zarr(masked_ds_dict, CACHE_MAIN, hydro_mask=hydro_mask, overwrite=True)

    return masked_ds_dict, hydro_mask


def compute_eli_for_period(ds, min_valid_months=30):
    """
    Compute ELI for one monthly dataset over one period.

    Expected variables:
      - mrso
      - evspsbl
      - rsds
      - tas
    """

    required = ["mrso", "evspsbl", "rsds", "tas"]
    missing = [v for v in required if v not in ds]
    if missing:
        raise ValueError(f"Missing variables: {missing}")

    ds = ds[required].sortby("time").chunk({"time": -1})

    sm = ds["mrso"]
    et = ds["evspsbl"]
    sw = ds["rsds"]
    tas = ds["tas"]

    # 1) detrend per period
    sm_dt = detrend_dim(sm, dim="time")
    et_dt = detrend_dim(et, dim="time")
    sw_dt = detrend_dim(sw, dim="time")

    # 2) remove mean seasonal cycle
    sm_anom = (sm_dt.groupby("time.month") - sm_dt.groupby("time.month").mean("time", skipna=True)).chunk({"time": -1})
    et_anom = (et_dt.groupby("time.month") - et_dt.groupby("time.month").mean("time", skipna=True)).chunk({"time": -1})
    sw_anom = (sw_dt.groupby("time.month") - sw_dt.groupby("time.month").mean("time", skipna=True)).chunk({"time": -1})

    # 3) warm-month mask (T >= 10 C)
    #tas_c = tas_to_celsius(tas)
    #warm = tas_c >= 10.0

    #sm_anom = sm_anom.where(warm).chunk({"time": -1})
    #et_anom = et_anom.where(warm).chunk({"time": -1})
    #sw_anom = sw_anom.where(warm).chunk({"time": -1})

    # 4) valid month count (all three variables available and warm)
    #valid = np.isfinite(sm_anom) & np.isfinite(et_anom) & np.isfinite(sw_anom)
    #n_valid = valid.sum("time")

    # 5) Kendall rank correlations
    corr_sm_et = kendall_corr(sm_anom, et_anom, dim="time")
    corr_sw_et = kendall_corr(sw_anom, et_anom, dim="time")

    # 6) ELI = cor(SM',ET') - cor(SWin',ET')
    eli = (corr_sm_et - corr_sw_et)#.where(n_valid >= min_valid_months)

    out = xr.Dataset(
        {
            "ELI": eli.astype("float32"),
            "corr_sm_et": corr_sm_et.astype("float32"),
            "corr_rsds_et": corr_sw_et.astype("float32"),
            #"n_valid_months": n_valid.astype("int16"),
        }
    )

    out["ELI"].attrs.update(
        {
            "long_name": "Ecosystem Limitation Index",
            "description": "ELI = Kendall corr(mrso_anom, evspsbl_anom) - Kendall corr(rsds_anom, evspsbl_anom); anomalies from detrended monthly series after removing mean seasonal cycle",#; warm months only (tas >= 10 degC)",
            "units": "1",
        }
    )
    out["corr_sm_et"].attrs.update(
        {
            "long_name": "Kendall correlation between soil moisture anomalies and ET anomalies",
            "units": "1",
        }
    )
    out["corr_rsds_et"].attrs.update(
        {
            "long_name": "Kendall correlation between incoming shortwave radiation anomalies and ET anomalies",
            "units": "1",
        }
    )
    #out["n_valid_months"].attrs.update(
    #    {
    #        "long_name": "Number of valid warm months used for ELI",
    #        "units": "months",
    #    }
    #)

    out.attrs["method"] = "Denissen-style ELI over one analysis period"
    #out.attrs["min_valid_months"] = int(min_valid_months)
    #out.attrs["temperature_threshold_degC"] = 10.0

    return out


def tas_to_celsius(tas):
    """
    Convert tas to degC if needed.
    """
    units = str(tas.attrs.get("units", "")).lower()

    if "k" in units and "deg" not in units and "c" not in units:
        return tas - 273.15

    # fallback heuristic
    try:
        tas_mean = float(tas.mean().compute())
    except Exception:
        tas_mean = np.nan

    if np.isfinite(tas_mean) and tas_mean > 100:
        return tas - 273.15

    return tas


def detrend_dim(da, dim="time"):
    """
    Remove linear trend along one dimension, allowing NaNs.
    """

    def _detrend_1d(y):
        y = np.asarray(y, dtype=np.float64)
        x = np.arange(y.size, dtype=np.float64)

        m = np.isfinite(y)
        if m.sum() < 2:
            return np.full_like(y, np.nan, dtype=np.float64)

        coef = np.polyfit(x[m], y[m], 1)
        trend = coef[0] * x + coef[1]

        out = y - trend
        out[~m] = np.nan
        return out

    return xr.apply_ufunc(
        _detrend_1d,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float64],
    )


def kendall_corr(da1, da2, dim="time"):
    """
    Kendall rank correlation along one dimension, allowing NaNs.
    """

    def _kendall_1d(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            return np.nan

        out = kendalltau(x[m], y[m], nan_policy="omit").correlation
        return np.nan if out is None else out

    return xr.apply_ufunc(
        _kendall_1d,
        da1,
        da2,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float64],
    )

def compute_spatial_mean_with_subdivisions(ds_dict):
    """
    Computes the spatial mean for subdivision in the datasets using weighted averaging.

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
                if 'subdivision' in ds[var].dims:
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

    print(f"Weighted spatial mean computed for each regime .")

    return ds_dict_mean

def compute_area_percentage_changes(ds_current, ds_change, variable):
    """
    Compute the percentage of land area with positive and negative changes for the given datasets.

    Parameters:
    ds_current (xarray.DataArray): Current dataset.
    ds_change (xarray.DataArray): Change dataset.
    variable (str): Variable name to use in the result keys.

    Returns:
    dict: Dictionary with percentages of positive and negative changes.
    """
    # Masks for positive and negative historical states
    mask_positive = ds_current['bgws_tran_mean'] > 0
    mask_negative = ds_current['bgws_tran_mean'] < 0

    ds_current = ds_current['bgws_tran_mean']
    ds_change = ds_change[variable]

    # Calculate grid cell areas for the current dataset
    cell_areas = compute_grid_cell_area(ds_current)

    # Subset the datasets based on these masks
    ds_change_positive = ds_change.where(mask_positive)
    ds_change_negative = ds_change.where(mask_negative)
    area_positive = cell_areas.where(mask_positive)
    area_negative = cell_areas.where(mask_negative)

    # Calculate area for positive historical state
    positive_change_pos_area = (area_positive * (ds_change_positive > 0)).sum().values.item()
    negative_change_pos_area = (area_positive * (ds_change_positive < 0)).sum().values.item()
    total_pos_area = area_positive.sum().values.item()

    # Calculate area for negative historical state
    positive_change_neg_area = (area_negative * (ds_change_negative > 0)).sum().values.item()
    negative_change_neg_area = (area_negative * (ds_change_negative < 0)).sum().values.item()
    total_neg_area = area_negative.sum().values.item()

    total_area = total_pos_area + total_neg_area

    result = {
        'Positive': {
            'Positive': f"{(positive_change_pos_area / total_area) * 100:.2f}%" if total_area > 0 else "0.00%",
            'Negative': f"{(negative_change_pos_area / total_area) * 100:.2f}%" if total_area > 0 else "0.00%",
        },
        'Negative': {
            'Positive': f"{(positive_change_neg_area / total_area) * 100:.2f}%" if total_area > 0 else "0.00%",
            'Negative': f"{(negative_change_neg_area / total_area) * 100:.2f}%" if total_area > 0 else "0.00%",
        }
    }

    return result

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

def compute_flip_changes(ds_current, ds_future, variable):
    """
    Compute the count and percentage of grid cells where the historical value was positive and is now negative and vice versa.

    Parameters:
    ds_current (xarray.DataArray): Current (historical) dataset.
    ds_future (xarray.DataArray): Future dataset.
    variable (str): Variable name to use in the result keys.

    Returns:
    dict: Dictionary with counts and percentages of flip changes.
    """

    # Compute grid cell areas (km²) directly using compute_grid_cell_area
    grid_cell_area = compute_grid_cell_area(ds_current)
    
    # Masks for positive and negative states
    mask_positive_current = ds_current > 0
    mask_negative_current = ds_current < 0

    mask_positive_future = ds_future > 0
    mask_negative_future = ds_future < 0

    # Calculate flips weighted by area
    pos_to_neg_flip_area = grid_cell_area.where(mask_positive_current & mask_negative_future).sum().values.item()
    neg_to_pos_flip_area = grid_cell_area.where(mask_negative_current & mask_positive_future).sum().values.item()

    # Calculate the total area (exclude NaNs)
    total_area = grid_cell_area.where(~ds_current.isnull()).sum().values.item()

    # Results as percentages of total land area
    result = {
        f'Positive to Negative (%)': f"{(pos_to_neg_flip_area / total_area) * 100:.2f}%",
        f'Negative to Positive (%)': f"{(neg_to_pos_flip_area / total_area) * 100:.2f}%"
    }

    return result

def combine_datasets(base_dict, driver_dict, skip_models=None):
    if skip_models is None:
        skip_models = set()

    combined = {}

    for exp in base_dict.keys():
        combined[exp] = base_dict[exp]

        for model in combined[exp].keys():
            if model in skip_models:
                continue

            for var in driver_dict[exp][model].data_vars:
                combined[exp][model][var] = driver_dict[exp][model][var]

    return combined

def load_and_combine_regression_data():
    """
    Load BGWS and driver period statistics, then combine them.

    Returns
    -------
    combined_dict : dict
        Combined historical / period statistics.
    combined_dict_diff : dict
        Combined difference statistics.
    hydro_mask : xr.DataArray or xr.Dataset
        Hydro mask used for driver statistics.
    """

    masked_ds_dict, masked_ds_dict_diff, hydro_mask = (
        bgws_period_stats()
    )

    masked_ds_dict_driver, masked_ds_dict_diff_driver, _ = (
        driver_period_stats(hydro_mask=hydro_mask)
    )

    combined_dict = combine_datasets(
        masked_ds_dict,
        masked_ds_dict_driver,
        skip_models={"12 model ensemble std", "OBS", "ERA5_land"}
    )

    combined_dict_diff = combine_datasets(
        masked_ds_dict_diff,
        masked_ds_dict_diff_driver,
        skip_models={"12 model ensemble std"}
    )

    return combined_dict, combined_dict_diff, hydro_mask