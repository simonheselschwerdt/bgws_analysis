"""
src/data_handling/import_data.py

Functions:
- open_dataset
- find_models_with_all_variables
- find_models_and_members
- import_model_data

Author: Simon P. Heselschwerdt
Date: 2026-02-26
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
from pathlib import Path
import sys

import intake_esgf

# Don't crash the program if a dataset can't be resolved to files
intake_esgf.conf["break_on_error"] = False

ROOT = Path.cwd().resolve().parent         
SRC  = ROOT / "src"
DATA_HANDLING = SRC / "data_handling"

sys.path.insert(0, str(DATA_HANDLING))
sys.path.insert(0, str(SRC))

import process_data as pro_dat
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

# Set Dask configuration to avoid large chunk creation
dask.config.set(**{'array.slicing.split_large_chunks': True})

def find_models_with_all_variables(cat, required_variables, required_experiments):
    """
    Return models that have *all* required variables for each required experiment.
    
    Parameters
    ----------
    cat : intake-esm catalog (already opened)
    required_variables : dict
        Mapping {variable_id: table_id} to check.
    required_experiments : list[str]
        Experiments (scenarios) to require.
    
    Returns
    -------
    set[str]
        Model (source_id) names that satisfy the availability checks.
    """
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
    """
    Return a mapping of models to valid member_ids available across variables/experiments.
    
    Parameters
    ----------
    cat : intake-esm catalog (already opened)
    required_variables : dict
        Mapping {variable_id: table_id} to check.
    required_experiments : list[str]
        Experiments (scenarios) to require.
    valid_models : iterable[str]
        Models to consider.
    
    Returns
    -------
    dict[str, set[str]]
        For each model, the set of member_id values that remain valid after intersection.
    """
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

import dask
import xarray as xr
import intake_esgf
from intake_esgf.catalog import MissingFileInformation

def not_important():
    """
    def import_model_data(cat, selected_model, selected_scenario, selected_vars, selected_member):
        ds_dict = {}
        dataset_parts = {}
    
        print(f"\nLoading data for model: {selected_model}, scenario: {selected_scenario}, member: {selected_member}")
    
        if selected_model == 'MIROC-ES2L' and selected_scenario == 'ssp126':
            for var in list(selected_vars.keys()):
                ds_dict[selected_model] = xr.open_dataset(
                    f"/pool/data/CMIP6/data/ScenarioMIP/MIROC/MIROC-ES2L/ssp126/r1i1p1f2/{selected_vars[var]}/{var}/gn/v20190823/"
                    f"{var}_{selected_vars[var]}_MIROC-ES2L_ssp126_r1i1p1f2_gn_201501-210012.nc"
                )
            return ds_dict
            
        elif selected_model == 'CAS-ESM2-0' and selected_scenario == 'historical':
            for var in list(selected_vars.keys()):
                ds_dict[selected_model] = xr.open_dataset(
                    f"/work/ik1017/CMIP6/data/CMIP6/CMIP/CAS/CAS-ESM2-0/historical/r1i1p1f1/{selected_vars[var]}/{var}/gn/v20200302/"
                    f"{var}_{selected_vars[var]}_CAS-ESM2-0_historical_r1i1p1f1_gn_185001-201412.nc"
                )
            return ds_dict
    
        elif selected_model == 'IPSL-CM6A-LR' and selected_scenario == 'ssp126':
            for var in list(selected_vars.keys()):
                ds_dict[selected_model] = xr.open_dataset(
                    f"/pool/data/CMIP6/data/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp126/r1i1p1f1/{selected_vars[var]}/{var}/gr/v20190903/"
                    f"{var}_{selected_vars[var]}_IPSL-CM6A-LR_ssp126_r1i1p1f1_gr_201501-210012.nc"
                )
            return ds_dict
        
        elif selected_model == 'BCC-CSM2-MR' and selected_scenario == 'ssp126' and selected_vars[list(selected_vars.keys())[0]] == 'day':
            for var in list(selected_vars.keys()):
                base = (f"/pool/data/CMIP6/data/ScenarioMIP/BCC/BCC-CSM2-MR/ssp126/r1i1p1f1/"
                        f"{selected_vars[var]}/{var}/gn/v20190315/")
                
                files = [
                    f"{base}{var}_{selected_vars[var]}_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20150101-20391231.nc",
                    f"{base}{var}_{selected_vars[var]}_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20400101-20641231.nc",
                    f"{base}{var}_{selected_vars[var]}_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20650101-20891231.nc",
                    f"{base}{var}_{selected_vars[var]}_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20900101-21001231.nc",
                ]
                
                ds_list = [xr.open_dataset(fp, decode_times=True, use_cftime=True) for fp in files]
                
                ds = xr.concat(ds_list, dim="time", data_vars="all", coords="minimal", compat="override")
                ds_dict[selected_model] = ds.sortby("time")
    
            return ds_dict
    
        for var, table in selected_vars.items():
            print(f"Searching for variable '{var}' in model '{selected_model}', scenario '{selected_scenario}'")
    
            search_results = cat.search(
                variable_id=var,
                table_id=table,
                experiment_id=selected_scenario,
                source_id=selected_model,
                member_id=selected_member
            )
    
            if search_results.df.empty:
                print(f"  ✗ No search results for {var}. Skipping.")
                continue
    
            try:
                with dask.config.set(use_cftime=True, decode_times=True):
                    datasets = search_results.to_dataset_dict(add_measures=False)
    
            except MissingFileInformation as e:
                # This is your case: search returns something, but no file URLs/paths available
                print(f"  ⚠ MissingFileInformation for {selected_model} {selected_scenario} {selected_member} {var} ({table}). Skipping this variable.")
                # optional: show the keys intake-esgf couldn't resolve
                try:
                    print(f"    Details: {e}")
                except Exception:
                    pass
                continue
    
            except Exception as e:
                print(f"  ⚠ Unexpected error loading {var}: {type(e).__name__}: {e}. Skipping this variable.")
                continue
    
            if datasets:
                # Often datasets has one entry, but keep your pattern
                for _, ds in datasets.items():
                    if var in ds:
                        dataset_parts[var] = ds
                        print(f"  ✓ Loaded {var} with shape {ds[var].shape}")
                    else:
                        print(f"  ⚠ Dataset loaded but missing variable {var}. Skipping.")
            else:
                print(f"  ⚠ No datasets returned for {var}. Skipping.")
    
        if dataset_parts:
            ds_dict[selected_model] = xr.merge(dataset_parts.values(), compat="override")
            print(f"✓ Merged {len(dataset_parts)} vars for {selected_model}, {selected_scenario}, {selected_member}.")
        else:
            print(f"✗ No valid datasets found for {selected_model}, {selected_scenario}, {selected_member}.")
    
        return ds_dict
    """
from pathlib import Path
import re
import pandas as pd
import numpy as np
import xarray as xr
import cftime
from functools import lru_cache

LOCAL_ROOT_CMIP = Path("/pool/data/CMIP6/data/CMIP")
LOCAL_ROOT_SCEN = Path("/pool/data/CMIP6/data/ScenarioMIP")

# If you have additional roots (e.g. /work/ik1017/...),
# add them here in order of preference:
EXTRA_LOCAL_ROOTS = []  # [Path("/work/ik1017/CMIP6/data/CMIP6/CMIP")]

GRID_PREFERENCE = ("gn", "gr", "gr1")  # pick first available


def activity_root(experiment_id: str) -> list[Path]:
    """Return candidate roots for a given experiment."""
    if experiment_id in ("historical", "hist-noLu"):
        roots = [LOCAL_ROOT_CMIP]
    else:
        roots = [LOCAL_ROOT_SCEN]
    return roots + EXTRA_LOCAL_ROOTS


def _version_key(v: str) -> int:
    # "v20190315" -> 20190315
    m = re.match(r"v(\d+)", v)
    return int(m.group(1)) if m else -1


def _parse_timerange_from_filename(name: str):
    """
    Parse trailing _YYYYMM-YYYYMM.nc or _YYYYMMDD-YYYYMMDD.nc from CMIP6 filenames.
    Returns (start_ts, end_ts) as pandas Timestamps, where end_ts is inclusive.
    """
    m = re.search(r"_(\d{6,8})-(\d{6,8})\.nc$", name)
    if not m:
        return None

    a, b = m.group(1), m.group(2)

    def to_start(s):
        if len(s) == 6:  # YYYYMM
            return pd.Timestamp(int(s[:4]), int(s[4:6]), 1)
        else:            # YYYYMMDD
            return pd.Timestamp(int(s[:4]), int(s[4:6]), int(s[6:8]))

    def to_end(s):
        if len(s) == 6:
            y, mo = int(s[:4]), int(s[4:6])
            # inclusive end-of-month
            return pd.Timestamp(y, mo, 1) + pd.offsets.MonthEnd(0)
        else:
            return pd.Timestamp(int(s[:4]), int(s[4:6]), int(s[6:8]))

    return to_start(a), to_end(b)


def _overlaps(file_start, file_end, target_start, target_end) -> bool:
    return (file_start <= target_end) and (file_end >= target_start)


@lru_cache(maxsize=4096)
def resolve_local_files(
    root: str,
    institution_id: str,
    source_id: str,
    experiment_id: str,
    member_id: str,
    table_id: str,
    variable_id: str,
):
    """
    Returns dict with: {"files": [...], "grid": "...", "version": "...", "base": Path}
    or None if not found.
    """
    root = Path(root)

    # /pool/data/CMIP6/data/{CMIP|ScenarioMIP}/{institution}/{model}/{experiment}/{member}/{table}/{var}/
    var_base = root / institution_id / source_id / experiment_id / member_id / table_id / variable_id
    if not var_base.exists():
        return None

    # grid label directory (gn/gr/...)
    grid_dirs = [p for p in var_base.iterdir() if p.is_dir()]
    if not grid_dirs:
        return None

    grid_dir = None
    for g in GRID_PREFERENCE:
        cand = var_base / g
        if cand.exists():
            grid_dir = cand
            break
    if grid_dir is None:
        # just take any
        grid_dir = grid_dirs[0]

    # version directories
    versions = sorted([p for p in grid_dir.iterdir() if p.is_dir() and p.name.startswith("v")],
                      key=lambda p: _version_key(p.name))
    if not versions:
        return None

    # choose newest by default (you can override with a "common version" logic later)
    vdir = versions[-1]
    nc_files = sorted(vdir.glob("*.nc"))
    if not nc_files:
        return None

    return {"files": nc_files, "grid": grid_dir.name, "version": vdir.name, "base": vdir}


def pick_files_for_window(nc_files, target_start: str, target_end: str):
    """Filter a list of files to those overlapping the target window."""
    ts = pd.Timestamp(target_start)
    te = pd.Timestamp(target_end)

    keep = []
    for fp in nc_files:
        tr = _parse_timerange_from_filename(fp.name)
        if tr is None:
            # If we can't parse range, keep it (better than missing)
            keep.append(fp)
            continue
        fs, fe = tr
        if _overlaps(fs, fe, ts, te):
            keep.append(fp)
    return keep


def open_local_cmip6_variable(
    institution_id: str,
    source_id: str,
    experiment_id: str,
    member_id: str,
    table_id: str,
    variable_id: str,
    target_start: str,
    target_end: str,
    chunks=None,
):
    """
    Try open local CMIP6 variable from pool.
    Returns (ds, meta) or (None, None).
    """
    for root in activity_root(experiment_id):
        info = resolve_local_files(
            str(root), institution_id, source_id, experiment_id, member_id, table_id, variable_id
        )
        if info is None:
            continue

        files = pick_files_for_window(info["files"], target_start, target_end)
        if not files:
            continue

        files = sorted(files)
        
        with xr.set_options(file_cache_maxsize=1):
            ds = xr.open_mfdataset(
                [str(f) for f in files],
                combine="by_coords",
                decode_times=True,
                use_cftime=True,
                parallel=False,                 # IMPORTANT
                engine="netcdf4",               # or "h5netcdf" if installed
                lock=dask.utils.SerializableLock(),  # IMPORTANT on shared FS
                #chunks=chunks if chunks is not None else "auto",
            )

        # Slice after open to avoid any spillover (e.g., projections to 2300)
        # Use inclusive end-of-day:
        #start = pd.Timestamp(target_start)
        #end = pd.Timestamp(target_end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

        # For cftime time, xarray accepts string slicing well in many cases:
        start_obj, end_obj = pro_dat._make_start_end_for_ds(ds, target_start, target_end)
        ds = ds.sel(time=slice(start_obj, end_obj))

        ds.attrs["local_cmip6_path"] = str(info["base"])
        ds.attrs["local_grid_label"] = info["grid"]
        ds.attrs["local_version"] = info["version"]
        return ds, info

    return None, None
    
def newest_common_version_across_vars(institution_id, source_id, experiment_id, member_id, selected_vars):
    """
    selected_vars: dict {var: table}
    returns a version string like 'v20190315' or None
    """
    roots = activity_root(experiment_id)

    per_var_versions = []
    for var, table in selected_vars.items():
        found_versions = set()
        for root in roots:
            base = Path(root) / institution_id / source_id / experiment_id / member_id / table / var
            if not base.exists():
                continue
            # all grids
            for gdir in base.iterdir():
                if not gdir.is_dir():
                    continue
                for vdir in gdir.iterdir():
                    if vdir.is_dir() and vdir.name.startswith("v"):
                        found_versions.add(vdir.name)
        if not found_versions:
            return None
        per_var_versions.append(found_versions)

    common = set.intersection(*per_var_versions) if per_var_versions else set()
    if not common:
        return None
    return max(common, key=_version_key)

def import_model_data(cat, selected_model, selected_scenario, selected_vars, selected_member):
    ds_dict = {}
    dataset_parts = {}

    # Decide target window
    if selected_scenario in ("historical", "hist-noLu"):
        target_start, target_end = "1850-01-01", "2014-12-31"
    else:
        target_start, target_end = "2015-01-01", "2100-12-31"

    # Get institution_id once (from catalog metadata)
    meta = cat.search(source_id=selected_model, experiment_id=selected_scenario).df
    if meta.empty or "institution_id" not in meta:
        institution_id = None
    else:
        institution_id = str(meta["institution_id"].iloc[0])

    for var, table in selected_vars.items():
        print(f"Loading {selected_model} {selected_scenario} {selected_member} {table}/{var}")

        # 1) Try local first (only if we have institution_id)
        if institution_id is not None:
            ds_local, info = open_local_cmip6_variable(
                institution_id=institution_id,
                source_id=selected_model,
                experiment_id=selected_scenario,
                member_id=selected_member,
                table_id=table,
                variable_id=var,
                target_start=target_start,
                target_end=target_end,
                chunks={"time": 365} if table == "day" else "auto",
            )
            if ds_local is not None and var in ds_local:
                dataset_parts[var] = ds_local
                print(f"  ✓ local ({info['grid']}/{info['version']}) files={len(info['files'])}")
                continue  # done for this var

        # 2) Fallback: intake-esgf
        print("  ↪ fallback to intake-esgf")
        search_results = cat.search(
            variable_id=var,
            table_id=table,
            experiment_id=selected_scenario,
            source_id=selected_model,
            member_id=selected_member
        )
        if search_results.df.empty:
            print(f"  ✗ No search results for {var}.")
            continue

        try:
            with dask.config.set(use_cftime=True, decode_times=True):
                datasets = search_results.to_dataset_dict(add_measures=False)
        except Exception as e:
            print(f"  ⚠ intake load failed for {var}: {type(e).__name__}: {e}")
            continue

        for _, ds in datasets.items():
            if var in ds:
                # slice down to target window (important if datasets go beyond 2100)
                start_obj, end_obj = pro_dat._make_start_end_for_ds(ds, target_start, target_end)
                ds = ds.sel(time=slice(start_obj, end_obj))
                dataset_parts[var] = ds
                print(f"  ✓ intake loaded {var}")
                break

    if dataset_parts:
        ds_dict[selected_model] = xr.merge(dataset_parts.values(), compat="override")
        print(f"✓ merged {len(dataset_parts)} vars for {selected_model} {selected_scenario} {selected_member}")
    else:
        print(f"✗ no valid data for {selected_model} {selected_scenario} {selected_member}")

    return ds_dict
