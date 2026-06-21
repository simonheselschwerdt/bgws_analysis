"""
Regression Analysis Script for Variable Importance Assessment
-------------------------------------------------------------
This script performs regression analysis using ElasticNet, hyperparameter optimization
with GridSearchCV, and computes variable importance using permutation methods.

Functions:
- scale_data: Scales features to the range [-1, 1].
- max_abs_scale: Helper function for scaling individual features.
- train_regression_model: Trains a regression model with GridSearchCV.
- test_train_evaluation: Evaluates model performance on training and test sets.
- compute_permutation_importance: Computes feature importances via permutations.
- regression_analysis: Orchestrates the full workflow for regression analysis.

Author: [Simon P. Heselschwerdt]
Date: [21.06.2026]
Dependencies: pandas, scikit-learn, xarray
"""

import numpy as np
import pandas as pd
import io

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

import shap
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.collections import PathCollection

import colormaps_and_utilities as col_uti


def make_regime_dataframe(ds, subdivision, response, predictors):
    vars_needed = [response] + predictors
    
    missing = [v for v in vars_needed if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Missing variables in dataset: {missing}")

    da = (
        ds[vars_needed]
        .to_array(dim="variable")
        .sel(subdivision=subdivision)
        .stack(sample=("lat", "lon"))
        .transpose("sample", "variable")
    )

    arr = da.values
    df = pd.DataFrame(arr, columns=vars_needed)

    # keep only complete rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").reset_index(drop=True)
    return df

def make_regime_dataframe_with_coords(ds, subdivision, response, predictors):
    vars_needed = [response] + predictors

    missing = [v for v in vars_needed if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Missing variables in dataset: {missing}")

    tmp = ds[vars_needed].sel(subdivision=subdivision)

    df = (
        tmp.to_array("variable")
        .stack(sample=("lat", "lon"))
        .transpose("sample", "variable")
        .to_pandas()
    )

    # recover coordinates from the stacked index
    df = df.reset_index()
    df = df.rename(columns={"level_0": "lat", "level_1": "lon"})

    # depending on xarray/pandas versions, lat/lon names may already exist
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("lat/lon were not preserved as expected.")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").reset_index(drop=True)
    return df

def add_spatial_blocks(df, lat_bin=10, lon_bin=20):
    out = df.copy()
    out["lat_block"] = np.floor(out["lat"] / lat_bin).astype(int)
    out["lon_block"] = np.floor(out["lon"] / lon_bin).astype(int)
    out["spatial_block"] = (
        out["lat_block"].astype(str) + "_" + out["lon_block"].astype(str)
    )
    return out

def nested_grouped_elasticnet_screening(X, y, groups, n_splits_outer=5, n_splits_inner=4):
    outer_cv = GroupKFold(n_splits=n_splits_outer)

    # alpha grid on log scale; l1_ratio from ridge-like to lasso-like
    param_grid = {
        "model__alpha": np.logspace(-3, 1, 20),
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
    }

    outer_results = []
    coef_table = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]
        g_train = groups.iloc[train_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=20000))
        ])

        inner_cv = GroupKFold(n_splits=n_splits_inner)

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="r2",
            cv=inner_cv.split(X_train, y_train, g_train),
            n_jobs=-1,
            refit=True,
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)

        best_alpha = search.best_params_["model__alpha"]
        best_l1 = search.best_params_["model__l1_ratio"]

        coefs = pd.Series(
            best_model.named_steps["model"].coef_,
            index=X.columns,
            name=f"fold_{fold}"
        )

        coef_table.append(coefs)

        selected = coefs[coefs != 0].index.tolist()

        outer_results.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "best_alpha": best_alpha,
            "best_l1_ratio": best_l1,
            "test_r2": test_r2,
            "n_selected": len(selected),
            "selected_predictors": selected,
        })

    outer_results = pd.DataFrame(outer_results)
    coef_table = pd.concat(coef_table, axis=1)

    return outer_results, coef_table

def nested_grouped_elasticnet_1se(X, y, groups, n_splits_outer=5, n_splits_inner=4):
    outer_cv = GroupKFold(n_splits=n_splits_outer)

    param_grid = {
        "model__alpha": np.logspace(-3, 1, 20),
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
    }

    outer_results = []
    coef_table = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]
        g_train = groups.iloc[train_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=20000))
        ])

        inner_cv = GroupKFold(n_splits=n_splits_inner)

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="r2",
            cv=inner_cv.split(X_train, y_train, g_train),
            n_jobs=-1,
            refit=False,
            return_train_score=False,
        )

        search.fit(X_train, y_train)

        cvres = pd.DataFrame(search.cv_results_)

        # best mean score
        best_idx = cvres["mean_test_score"].idxmax()
        best_mean = cvres.loc[best_idx, "mean_test_score"]
        best_std = cvres.loc[best_idx, "std_test_score"]

        # standard error across inner folds
        se = best_std / np.sqrt(n_splits_inner)

        # keep models within 1 standard error of the best
        eligible = cvres[cvres["mean_test_score"] >= best_mean - se].copy()

        # prefer simpler models:
        #   1) larger alpha
        #   2) larger l1_ratio
        eligible = eligible.sort_values(
            by=["param_model__alpha", "param_model__l1_ratio"],
            ascending=[False, False]
        )

        chosen = eligible.iloc[0]
        chosen_alpha = float(chosen["param_model__alpha"])
        chosen_l1 = float(chosen["param_model__l1_ratio"])

        # refit chosen model on full outer-train set
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=chosen_alpha, l1_ratio=chosen_l1, max_iter=20000))
        ])
        final_model.fit(X_train, y_train)

        y_pred = final_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)

        coefs = pd.Series(
            final_model.named_steps["model"].coef_,
            index=X.columns,
            name=f"fold_{fold}"
        )
        coef_table.append(coefs)

        selected = coefs[coefs != 0].index.tolist()

        outer_results.append({
            "fold": fold,
            "test_r2": test_r2,
            "best_inner_mean_r2": best_mean,
            "best_inner_se": se,
            "chosen_alpha": chosen_alpha,
            "chosen_l1_ratio": chosen_l1,
            "n_selected": len(selected),
            "selected_predictors": selected,
        })

    outer_results = pd.DataFrame(outer_results)
    coef_table = pd.concat(coef_table, axis=1)

    return outer_results, coef_table

def repeated_grouped_elasticnet_1se(
    X, y, groups,
    n_repeats=40,
    test_size=0.2,
    n_splits_inner=4,
    random_state=42,
):
    """
    Repeated group-wise train/test splits.
    Within each repeat:
      - split by spatial blocks
      - tune Elastic Net with grouped inner CV
      - apply 1SE rule
      - refit on full training groups
      - evaluate on held-out groups
    """

    param_grid = {
        "model__alpha": np.logspace(-3, 1, 20),
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
    }

    splitter = GroupShuffleSplit(
        n_splits=n_repeats,
        test_size=test_size,
        random_state=random_state,
    )

    coef_list = []
    results = []

    for rep, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]
        g_train = groups.iloc[train_idx]

        inner_cv = GroupKFold(n_splits=n_splits_inner)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=20000))
        ])

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="r2",
            cv=inner_cv.split(X_train, y_train, g_train),
            n_jobs=-1,
            refit=False,
            return_train_score=False,
        )
        search.fit(X_train, y_train)

        cvres = pd.DataFrame(search.cv_results_)

        best_idx = cvres["mean_test_score"].idxmax()
        best_mean = cvres.loc[best_idx, "mean_test_score"]
        best_std = cvres.loc[best_idx, "std_test_score"]
        se = best_std / np.sqrt(n_splits_inner)

        eligible = cvres[cvres["mean_test_score"] >= best_mean - se].copy()

        # choose the simplest eligible model:
        # larger alpha, then larger l1_ratio
        eligible = eligible.sort_values(
            by=["param_model__alpha", "param_model__l1_ratio"],
            ascending=[False, False],
        )
        chosen = eligible.iloc[0]

        chosen_alpha = float(chosen["param_model__alpha"])
        chosen_l1 = float(chosen["param_model__l1_ratio"])

        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=chosen_alpha, l1_ratio=chosen_l1, max_iter=20000))
        ])
        final_model.fit(X_train, y_train)

        y_pred = final_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)

        coefs = pd.Series(
            final_model.named_steps["model"].coef_,
            index=X.columns,
            name=f"rep_{rep}"
        )
        coef_list.append(coefs)

        results.append({
            "repeat": rep,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "test_r2": test_r2,
            "chosen_alpha": chosen_alpha,
            "chosen_l1_ratio": chosen_l1,
            "n_selected": int((coefs != 0).sum()),
        })

    coef_table = pd.concat(coef_list, axis=1)
    results = pd.DataFrame(results)

    nonzero = coef_table.replace(0, np.nan)

    summary = pd.DataFrame({
        "selection_frequency": (coef_table != 0).mean(axis=1),
        "sign_consistency": np.sign(nonzero).mean(axis=1),
        "mean_coef_all": coef_table.mean(axis=1),
        "median_coef_all": coef_table.median(axis=1),
        "mean_coef_nonzero": nonzero.mean(axis=1),
        "median_coef_nonzero": nonzero.median(axis=1),
    }).sort_values(
        ["selection_frequency", "median_coef_nonzero"],
        ascending=[False, False]
    )

    return results, coef_table, summary

def compare_common_sets(X, y, groups, model_sets, random_state=42):
    out = []

    for name, cols in model_sets.items():
        res, coefs, summary = repeated_grouped_elasticnet_1se(
            X[cols], y, groups,
            n_repeats=40,
            test_size=0.2,
            n_splits_inner=4,
            random_state=random_state,
        )

        out.append({
            "model": name,
            "n_predictors": len(cols),
            "mean_test_r2": res["test_r2"].mean(),
            "median_test_r2": res["test_r2"].median(),
            "sd_test_r2": res["test_r2"].std(),
            "mean_n_selected": res["n_selected"].mean(),
        })

    return pd.DataFrame(out).sort_values("mean_test_r2", ascending=False)

def repeated_grouped_importance_fixed_set(
    X, y, groups, predictors,
    n_repeats=40,
    test_size=0.2,
    n_splits_inner=4,
    random_state=42,
    n_perm=20,
):
    X = X[predictors].copy()

    param_grid = {
        "model__alpha": np.logspace(-3, 1, 20),
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
    }

    splitter = GroupShuffleSplit(
        n_splits=n_repeats,
        test_size=test_size,
        random_state=random_state,
    )

    rep_results = []
    coef_list = []
    perm_list = []

    for rep, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]
        g_train = groups.iloc[train_idx]

        inner_cv = GroupKFold(n_splits=n_splits_inner)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=20000))
        ])

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="r2",
            cv=inner_cv.split(X_train, y_train, g_train),
            n_jobs=-1,
            refit=False,
            return_train_score=False,
        )
        search.fit(X_train, y_train)

        cvres = pd.DataFrame(search.cv_results_)

        best_idx = cvres["mean_test_score"].idxmax()
        best_mean = cvres.loc[best_idx, "mean_test_score"]
        best_std = cvres.loc[best_idx, "std_test_score"]
        se = best_std / np.sqrt(n_splits_inner)

        eligible = cvres[cvres["mean_test_score"] >= best_mean - se].copy()
        eligible = eligible.sort_values(
            by=["param_model__alpha", "param_model__l1_ratio"],
            ascending=[False, False]
        )
        chosen = eligible.iloc[0]

        chosen_alpha = float(chosen["param_model__alpha"])
        chosen_l1 = float(chosen["param_model__l1_ratio"])

        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=chosen_alpha, l1_ratio=chosen_l1, max_iter=20000))
        ])
        final_model.fit(X_train, y_train)

        y_pred = final_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)

        coefs = pd.Series(
            final_model.named_steps["model"].coef_,
            index=predictors,
            name=f"rep_{rep}"
        )
        coef_list.append(coefs)

        perm = permutation_importance(
            final_model,
            X_test,
            y_test,
            n_repeats=n_perm,
            random_state=random_state + rep,
            scoring="r2",
            n_jobs=-1,
        )
        perm_imp = pd.Series(perm.importances_mean, index=predictors, name=f"rep_{rep}")
        perm_list.append(perm_imp)

        rep_results.append({
            "repeat": rep,
            "test_r2": test_r2,
            "chosen_alpha": chosen_alpha,
            "chosen_l1_ratio": chosen_l1,
            "n_selected": int((coefs != 0).sum()),
        })

    rep_results = pd.DataFrame(rep_results)
    coef_table = pd.concat(coef_list, axis=1)
    perm_table = pd.concat(perm_list, axis=1)

    coef_summary = pd.DataFrame({
        "selection_frequency": (coef_table != 0).mean(axis=1),
        "sign_consistency": np.sign(coef_table.replace(0, np.nan)).mean(axis=1),
        "mean_coef": coef_table.mean(axis=1),
        "median_coef": coef_table.median(axis=1),
    }).sort_values("mean_coef", ascending=False)

    perm_summary = pd.DataFrame({
        "mean_perm_importance": perm_table.mean(axis=1),
        "median_perm_importance": perm_table.median(axis=1),
        "sd_perm_importance": perm_table.std(axis=1),
    }).sort_values("mean_perm_importance", ascending=False)

    return rep_results, coef_table, coef_summary, perm_table, perm_summary

def summarize_importance_with_iqr(perm_table, coef_table):
    out = pd.DataFrame({
        "perm_mean": perm_table.mean(axis=1),
        "perm_median": perm_table.median(axis=1),
        "perm_q25": perm_table.quantile(0.25, axis=1),
        "perm_q75": perm_table.quantile(0.75, axis=1),
        "perm_p02_5": perm_table.quantile(0.025, axis=1),
        "perm_p97_5": perm_table.quantile(0.975, axis=1),
        "coef_median": coef_table.median(axis=1),
        "coef_p02_5": coef_table.quantile(0.025, axis=1),
        "coef_p97_5": coef_table.quantile(0.975, axis=1),
        "selection_frequency": (coef_table != 0).mean(axis=1),
        "sign_consistency": np.sign(coef_table.replace(0, np.nan)).mean(axis=1),
    }).sort_values("perm_median", ascending=False)
    return out

def get_individual_model_keys(ds_dict_change_sub, scenario_key):
    """
    Return only individual ESM keys, excluding ensemble summaries.
    """
    all_keys = list(ds_dict_change_sub[scenario_key].keys())
    return [
        k for k in all_keys
        if not k.startswith("12 model ensemble")
    ]


def analyze_fixed_model_across_ensemble(
    ds_dict_change_sub,
    scenario_key,
    predictors,
    response="bgws_tran_mean",
    subdivisions=None,
    lat_bin=10,
    lon_bin=20,
    n_repeats=20,
    test_size=0.2,
    n_splits_inner=4,
    random_state=42,
    n_perm=10,
):
    """
    Run the fixed predictor-set analysis for every individual ESM and both subdivisions.

    Requires these helper functions to already exist in your notebook:
      - make_regime_dataframe_with_coords
      - add_spatial_blocks
      - repeated_grouped_importance_fixed_set

    Returns
    -------
    per_model_long : pd.DataFrame
        One row per model x regime x predictor.
    perf_long : pd.DataFrame
        One row per model x regime with overall model performance.
    raw_outputs : dict
        Nested dictionary with the full outputs if you need them later.
    """
    model_keys = get_individual_model_keys(ds_dict_change_sub, scenario_key)

    # use ensemble mean only to read subdivision names if not supplied
    if subdivisions is None:
        ds_ref = ds_dict_change_sub[scenario_key]["12 model ensemble mean"]
        subdivisions = list(ds_ref.subdivision.values)

    per_model_rows = []
    perf_rows = []
    raw_outputs = {}

    for model_key in model_keys:
        ds_model = ds_dict_change_sub[scenario_key][model_key]
        raw_outputs[model_key] = {}

        for subdivision in subdivisions:
            df = make_regime_dataframe_with_coords(
                ds_model,
                subdivision=subdivision,
                response=response,
                predictors=predictors,
            )
            df = add_spatial_blocks(df, lat_bin=lat_bin, lon_bin=lon_bin)

            X = df[predictors].copy()
            y = df[response].copy()
            groups = df["spatial_block"].copy()

            rep_results, coef_table, coef_summary, perm_table, perm_summary = (
                repeated_grouped_importance_fixed_set(
                    X, y, groups,
                    predictors=predictors,
                    n_repeats=n_repeats,
                    test_size=test_size,
                    n_splits_inner=n_splits_inner,
                    random_state=random_state,
                    n_perm=n_perm,
                )
            )

            raw_outputs[model_key][subdivision] = {
                "rep_results": rep_results,
                "coef_table": coef_table,
                "coef_summary": coef_summary,
                "perm_table": perm_table,
                "perm_summary": perm_summary,
            }

            perf_rows.append({
                "model_key": model_key,
                "subdivision": subdivision,
                "mean_test_r2": rep_results["test_r2"].mean(),
                "median_test_r2": rep_results["test_r2"].median(),
                "sd_test_r2": rep_results["test_r2"].std(),
                "mean_n_selected": rep_results["n_selected"].mean(),
                "n": len(df),
            })

            for predictor in predictors:
                per_model_rows.append({
                    "model_key": model_key,
                    "subdivision": subdivision,
                    "predictor": predictor,
                    "mean_perm_importance": perm_table.loc[predictor].mean(),
                    "median_perm_importance": perm_table.loc[predictor].median(),
                    "coef_median": coef_table.loc[predictor].median(),
                    "coef_mean": coef_table.loc[predictor].mean(),
                    "selection_frequency": (coef_table.loc[predictor] != 0).mean(),
                    "sign_consistency": np.sign(
                        coef_table.loc[predictor].replace(0, np.nan)
                    ).mean(),
                })

    per_model_long = pd.DataFrame(per_model_rows)
    perf_long = pd.DataFrame(perf_rows)

    return per_model_long, perf_long, raw_outputs

def summarize_model_agreement(per_model_long, regime_name):
    """
    Summarize across individual ESMs for one regime.
    """
    tmp = per_model_long.loc[per_model_long["subdivision"] == regime_name].copy()

    tmp["rank_in_model"] = tmp.groupby("model_key")["mean_perm_importance"] \
                              .rank(ascending=False, method="min")

    out = tmp.groupby("predictor").agg(
        n_models=("model_key", "nunique"),
        esm_mean_perm=("mean_perm_importance", "mean"),
        esm_median_perm=("mean_perm_importance", "median"),
        esm_p25_perm=("mean_perm_importance", lambda s: s.quantile(0.25)),
        esm_p75_perm=("mean_perm_importance", lambda s: s.quantile(0.75)),
        positive_importance_fraction=("mean_perm_importance", lambda s: np.mean(s > 0)),
        top3_fraction=("rank_in_model", lambda s: np.mean(s <= 3)),
        same_sign_positive_fraction=("coef_median", lambda s: np.mean(s > 0)),
        same_sign_negative_fraction=("coef_median", lambda s: np.mean(s < 0)),
    )

    out["dominant_sign_fraction"] = out[[
        "same_sign_positive_fraction",
        "same_sign_negative_fraction"
    ]].max(axis=1)

    return out.sort_values("esm_median_perm", ascending=False)

import numpy as np
import pandas as pd
import xarray as xr

def compute_clipped_standardized_context(df, predictors, clip=1.0):
    """
    Clipped standardized spatial mean change for each predictor in one regime:
    (mean / std) clipped to [-clip, clip].
    """
    z = (df[predictors].mean() / df[predictors].std(ddof=0)).replace([np.inf, -np.inf], np.nan)
    return z.clip(-clip, clip).rename("context_z")


def compute_per_model_context_long(ds_dict_change_sub, scenario_key, predictors, clip=1.0):
    """
    Compute clipped standardized spatial mean change per model x regime x predictor
    from the subdivided dataset dictionary.

    Returns
    -------
    pd.DataFrame with columns:
      model_key, subdivision, predictor, context_z
    """
    rows = []

    for model_key, ds in ds_dict_change_sub[scenario_key].items():
        # skip precomputed std fields
        if "std" in model_key:
            continue

        for subdivision in ds.subdivision.values:
            for pred in predictors:
                if pred not in ds.data_vars:
                    continue

                da = ds[pred].sel(subdivision=subdivision)
                vals = da.values
                vals = vals[np.isfinite(vals)]

                if vals.size < 2:
                    ctx = np.nan
                else:
                    sd = np.std(vals, ddof=0)
                    if sd == 0 or not np.isfinite(sd):
                        ctx = np.nan
                    else:
                        ctx = np.mean(vals) / sd
                        ctx = np.clip(ctx, -clip, clip)

                rows.append({
                    "model_key": model_key,
                    "subdivision": subdivision,
                    "predictor": pred,
                    "context_z": ctx,
                })

    return pd.DataFrame(rows)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import blended_transform_factory
import matplotlib.patheffects as pe

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

context_cmap = plt.get_cmap("PuOr_r")

context_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

def compute_standardized_context(df, predictors):
    """
    Standardized regime-mean predictor change:
    spatial mean / spatial std for each predictor.
    """
    z = (df[predictors].mean() / df[predictors].std(ddof=0)).replace([np.inf, -np.inf], np.nan)
    return z.rename("context_z")


def _prepare_panel_df(
    main_summary,
    per_model_long,
    perf_long,
    per_model_context_long,
    regime_name,
    predictor_display_names=None,
    context_series=None,
    r2_threshold=0.3,
):
    perf_sub = perf_long.loc[
        (perf_long["subdivision"] == regime_name) &
        (perf_long["mean_test_r2"] > r2_threshold)
    ].copy()

    keep_models = set(perf_sub["model_key"])

    tmp = per_model_long.loc[
        (per_model_long["subdivision"] == regime_name) &
        (per_model_long["model_key"].isin(keep_models))
    ].copy()

    if per_model_context_long is not None:
        tmp = tmp.merge(
            per_model_context_long,
            on=["model_key", "subdivision", "predictor"],
            how="left"
        )

    tmp["rank_in_model"] = tmp.groupby("model_key")["median_perm_importance"] \
                              .rank(ascending=False, method="min")

    agreement = tmp.groupby("predictor").agg(
        n_models=("model_key", "nunique"),
        top3_fraction=("rank_in_model", lambda s: np.mean(s <= 3)),
        positive_importance_fraction=("median_perm_importance", lambda s: np.mean(s > 0)),
    )

    plot_df = main_summary.copy().join(agreement, how="left")

    if context_series is not None:
        plot_df = plot_df.join(context_series.rename("context_z"), how="left")

    plot_df = plot_df.sort_values("perm_median", ascending=True)

    labels = []
    for pred, row in plot_df.iterrows():
        lab = predictor_display_names.get(pred, pred) if predictor_display_names is not None else pred

        n_models = int(row["n_models"]) if pd.notna(row["n_models"]) else 0
        n_top3 = int(round(row["top3_fraction"] * n_models)) if n_models else 0
        n_pos = int(round(row["positive_importance_fraction"] * n_models)) if n_models else 0

        if n_models > 0 and n_top3 >= max(1, int(np.ceil(0.5 * n_models))):
            star = r"$^{**}$"
        elif n_models > 0 and n_pos >= max(1, int(np.ceil(0.8 * n_models))):
            star = r"$^{*}$"
        else:
            star = ""

        labels.append(lab + star)

    plot_df["display_label"] = labels
    return plot_df, tmp, len(keep_models)


def _plot_driver_panel(
    ax,
    plot_df,
    tmp,
    panel_label,
    title_text,
    title_facecolor,
    r2_text,
    n_text,
    context_norm,
    context_cmap,
    max_abs_context,
    show_ylabels=True,
    interval="iqr",   # "iqr" or "95"
    ticklabel_size=17,
    label_size=19,
    title_size=20,
    point_size=120,
    ind_alpha=0.45,
    beta_x_axes=0.02,
):
    predictors = plot_df.index.tolist()
    y = np.arange(len(predictors))
    rng = np.random.default_rng(42)

    # individual ESM dots
    for i, pred in enumerate(predictors):
        sub = tmp.loc[tmp["predictor"] == pred].copy()
        if len(sub) == 0:
            continue

        vals = sub["median_perm_importance"].values
        ctxs = sub["context_z"].values if "context_z" in sub.columns else np.full(len(sub), np.nan)
        jitter = rng.uniform(-0.13, 0.13, size=len(sub))

        colors = []
        for ctx in ctxs:
            if pd.isna(ctx):
                colors.append((0.75, 0.75, 0.75, ind_alpha))
            else:
                ctx_normed = np.clip(ctx / max_abs_context, -1.0, 1.0)
                r, g, b, _ = context_cmap(context_norm(ctx_normed))
                colors.append((r, g, b, ind_alpha))

        ax.scatter(
            vals,
            np.full(len(sub), i) + jitter,
            s=80,
            c=colors,
            edgecolors="black",
            linewidths=1,
            zorder=1,
        )

    # ensemble-mean square + interval
    for i, pred in enumerate(predictors):
        row = plot_df.loc[pred]
        x = row["perm_median"]

        if interval == "iqr" and "perm_q25" in row.index and "perm_q75" in row.index:
            xlo = row["perm_q25"]
            xhi = row["perm_q75"]
        else:
            xlo = row["perm_p02_5"]
            xhi = row["perm_p97_5"]

        ctx = row["context_z"] if "context_z" in row.index else np.nan
        ctx_normed = np.clip(ctx / max_abs_context, -1.0, 1.0) if pd.notna(ctx) else np.nan
        color = context_cmap(context_norm(ctx_normed)) if pd.notna(ctx_normed) else (0.2, 0.2, 0.2, 1.0)

        # IQR / uncertainty interval with white halo + end caps
        interval_color = "0.35"
        halo_color = "white"
        interval_lw = 3.2
        halo_lw = 7.0
        cap_halfheight = 0.10

        line = ax.hlines(i, xlo, xhi, color=interval_color, lw=interval_lw, zorder=3)
        line.set_path_effects([pe.Stroke(linewidth=5.0, foreground=halo_color), pe.Normal()])
        
        caps = ax.vlines([xlo, xhi],
                         i - cap_halfheight, i + cap_halfheight,
                         color=interval_color, lw=interval_lw, zorder=3)
        caps.set_path_effects([pe.Stroke(linewidth=halo_lw, foreground=halo_color), pe.Normal()])

        # colored square
        ax.scatter(
            x, i,
            s=point_size,
            marker="s",
            color=color,
            edgecolor="black",
            linewidth=0.9,
            zorder=4,
        )

    ax.axvline(0, color="0.55", linestyle="--", lw=1.2, zorder=0)

    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(plot_df["display_label"], fontsize=ticklabel_size, color="black")
    else:
        ax.set_yticklabels([])

    # sign symbol next to variable label (superscript-like)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    
    # x-position slightly left of plotting area; y-offset slightly upward for superscript look
    sign_x_axes = -0.07
    sign_y_offset = -0.015   # negative = visually upward because y-axis is inverted
    
    for i, pred in enumerate(predictors):
        s = plot_df.loc[pred, "sign_consistency"]
        if s >= 0.9:
            symbol = "▲"
            color = (40/255, 125/255, 210/255)   # dark blue
        elif s <= -0.9:
            symbol = "▼"
            color = (15/255, 115/255, 15/255)   # dark green
    
        ax.text(
            sign_x_axes, i + sign_y_offset,
            symbol,
            transform=trans,
            ha="left", va="center",
            fontsize=ticklabel_size + 5,
            color=color,
            fontweight="bold",
            zorder=6,
            clip_on=False,
        )

    ax.set_xlabel(r"Permutation importance ($\Delta R^2$)", fontsize=label_size)
    ax.tick_params(axis="x", labelsize=ticklabel_size)
    ax.tick_params(axis="y", length=0, pad=18)

    # panel label farther left
    ax.text(
        -0.12, 1.015, panel_label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=title_size + 2,
        fontweight="bold",
    )

    # left-aligned title box
    ax.text(
        0.02, 1.015, title_text,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=title_size,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.22", facecolor=title_facecolor, edgecolor="none"),
    )

    # performance text
    ax.text(
        0.97, 0.02,
        f"$R^2$ = {r2_text}\n$n$ = {n_text}",
        ha="right", va="bottom",
        transform=ax.transAxes,
        fontsize=ticklabel_size,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.9),
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_driver_importance_two_panel(
    blue_summary,
    green_summary,
    per_model_long,
    perf_long,
    per_model_context_long,
    blue_regime_name,
    green_regime_name,
    predictor_display_names=None,
    blue_context=None,
    green_context=None,
    blue_r2=None,
    green_r2=None,
    blue_n=None,
    green_n=None,
    figsize=(17.2, 7.6),
    ticklabel_size=17,
    label_size=19,
    title_size=20,
    point_size=120,
    xpad_frac=0.06,
    context_cmap_name=context_cmap,
    r2_threshold=0.3,
    ind_alpha=0.45,
    interval="iqr",

    dpi=300,
    filetype="png",
    savepath=None,
):
    all_context = pd.concat([
        blue_context.rename("context_z") if blue_context is not None else pd.Series(dtype=float),
        green_context.rename("context_z") if green_context is not None else pd.Series(dtype=float),
        per_model_context_long["context_z"] if per_model_context_long is not None and "context_z" in per_model_context_long else pd.Series(dtype=float),
    ])
    
    # symmetric normalization to [-1, 1]
    max_abs_context = float(np.nanmax(np.abs(all_context.values))) if len(all_context) else 1.0
    max_abs_context = max(max_abs_context, 1e-12)
    
    context_cmap = plt.get_cmap(context_cmap_name)
    context_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    blue_df, blue_tmp, blue_n_models_shown = _prepare_panel_df(
        blue_summary,
        per_model_long,
        perf_long,
        per_model_context_long,
        blue_regime_name,
        predictor_display_names=predictor_display_names,
        context_series=blue_context,
        r2_threshold=r2_threshold,
    )

    green_df, green_tmp, green_n_models_shown = _prepare_panel_df(
        green_summary,
        per_model_long,
        perf_long,
        per_model_context_long,
        green_regime_name,
        predictor_display_names=predictor_display_names,
        context_series=green_context,
        r2_threshold=r2_threshold,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    _plot_driver_panel(
        axes[0], blue_df, blue_tmp,
        panel_label="(a)",
        title_text="Historical blue water regime",
        title_facecolor=(40/255, 125/255, 210/255),
        r2_text=f"{blue_r2:.2f}" if blue_r2 is not None else "NA",
        n_text=f"{blue_n}" if blue_n is not None else "NA",
        context_norm=context_norm,
        context_cmap=context_cmap,
        show_ylabels=True,
        interval=interval,
        ticklabel_size=ticklabel_size,
        label_size=label_size,
        title_size=title_size,
        point_size=point_size,
        ind_alpha=ind_alpha,
        beta_x_axes=0.02,
        max_abs_context=max_abs_context,
    )

    _plot_driver_panel(
        axes[1], green_df, green_tmp,
        panel_label="(b)",
        title_text="Historical green water regime",
        title_facecolor=(15/255, 115/255, 15/255),
        r2_text=f"{green_r2:.2f}" if green_r2 is not None else "NA",
        n_text=f"{green_n}" if green_n is not None else "NA",
        context_norm=context_norm,
        context_cmap=context_cmap,
        show_ylabels=True,
        interval=interval,
        ticklabel_size=ticklabel_size,
        label_size=label_size,
        title_size=title_size,
        point_size=point_size,
        ind_alpha=ind_alpha,
        beta_x_axes=0.02,
        max_abs_context=max_abs_context,
    )

    # shared x limits with extra left room for β column
    xmin = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
    xmax = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
    dx = xmax - xmin
    xmin = -0.05 * max(xmax, 0.5)   # reserve visible negative space for β labels
    xmax = xmax + xpad_frac * dx
    axes[0].set_xlim(xmin, xmax)
    axes[1].set_xlim(xmin, xmax)

    # legend
    interval_label = (
        "Uncertainty range (IQR)"
        if interval == "iqr"
        else "Uncertainty range (95%)"
    )
    legend_handles = [
        Line2D(
            [0], [0],
            marker='s', color='none', markerfacecolor='0.75',
            markeredgecolor='black', markeredgewidth=1,
            markersize=14, alpha=ind_alpha,
            label='Ensemble mean\nimportance'
        ),
        Line2D(
            [0], [0],
            marker='o', color='none', markerfacecolor='0.75',
            markeredgecolor='black', markeredgewidth=0.6,
            markersize=14, alpha=ind_alpha,
            label=rf'Indiv. ESM importance'
        ),
        Line2D(
            [0, 1], [0, 0],
            color='0.35', lw=3.5,
            label=interval_label
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.895, 0.75),   # just right of colorbar
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=19,
        ncol=1,
        handlelength=0.7,
        handletextpad=0.8,
        borderaxespad=0.0,
    )

    # vertical colorbar on right
    sm = ScalarMappable(norm=context_norm, cmap=context_cmap)
    sm.set_array([])
    
    cax = fig.add_axes([0.8, 0.27, 0.018, 0.56])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Normalised predictor change", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    # note under legend
    fig.text(
        0.91, 0.65,
        r"$^{**}$ top-3 in ≥50% of ESMs" "\n"
        r"  $^{*}$ consistently important" "\n"
        r"     in ≥80% of ESMs" "\n",
        transform=fig.transFigure,
        ha="left",
        va="top",
        fontsize=19,
    )

        # blue upward triangle
    fig.text(
        0.91, 0.5,
        "▲",
        transform=fig.transFigure,
        ha="left",
        va="top",
        fontsize=22,
        color=(40/255, 125/255, 210/255),
        fontweight="bold",
    )
    
    fig.text(
        0.925, 0.5,
        " pos. regression coeff.",
        transform=fig.transFigure,
        ha="left",
        va="top",
        fontsize=19,
    )
    
    # green downward triangle
    fig.text(
        0.91, 0.45,
        "▼",
        transform=fig.transFigure,
        ha="left",
        va="top",
        fontsize=22,
        color=(15/255, 115/255, 15/255),
        fontweight="bold",
    )
    
    fig.text(
        0.925, 0.45,
        " neg. regression coeff.",
        transform=fig.transFigure,
        ha="left",
        va="top",
        fontsize=19,
    )

    fig.subplots_adjust(left=0.18, right=0.78, top=0.88, bottom=0.20, wspace=0.4)

    if savepath:
        fname = f"fig_3_{dpi}dpi_updated.{filetype}"
        col_uti.save_fig(fig, savepath, fname, dpi=dpi)
        print(f"Figure saved under {savepath}{fname}")

    return fig, axes

def tune_grouped_rf_once(
    X,
    y,
    groups,
    predictors,
    n_splits_inner=4,
    random_state=42,
    param_grid=None,
    n_jobs=1,
):
    X = X[predictors].copy()

    if param_grid is None:
        param_grid = {
            "n_estimators": [200],
            "max_depth": [8, 16],
            "min_samples_leaf": [3, 5],
            "min_samples_split": [2, 10],
            "max_features": [0.5, 1.0],
        }

    inner_cv = GroupKFold(n_splits=n_splits_inner)

    rf = RandomForestRegressor(
        random_state=random_state,
        bootstrap=True,
        n_jobs=1,
    )

    search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="r2",
        cv=inner_cv.split(X, y, groups),
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )
    search.fit(X, y)

    return search.best_params_, pd.DataFrame(search.cv_results_).sort_values("rank_test_score")

def repeated_grouped_rf_shap_fixed_params(
    X,
    y,
    groups,
    predictors,
    rf_params,
    n_repeats=12,
    test_size=0.2,
    random_state=42,
    n_perm=8,
    max_shap_samples=400,
    store_all_shap=False,
):
    """
    Faster RF + SHAP workflow:
    - fixed RF hyperparameters
    - repeated grouped holdout splits
    - held-out permutation importance
    - held-out SHAP on a small subsample
    """

    X = X[predictors].copy()

    splitter = GroupShuffleSplit(
        n_splits=n_repeats,
        test_size=test_size,
        random_state=random_state,
    )

    rep_results = []
    perm_list = []
    shap_list = []

    shap_values_frames = []
    shap_feature_frames = []

    for rep, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        print(f"Repeat {rep}/{n_repeats}")

        X_train = X.iloc[train_idx].copy()
        X_test  = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test  = y.iloc[test_idx].copy()

        model = RandomForestRegressor(
            **rf_params,
            random_state=random_state + rep,
            bootstrap=True,
            n_jobs=1,   # safer; avoid worker crashes
        )
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # permutation importance on held-out test data
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=n_perm,
            random_state=random_state + rep,
            scoring="r2",
            n_jobs=1,
        )
        perm_imp = pd.Series(
            perm.importances_mean,
            index=predictors,
            name=f"rep_{rep}"
        )
        perm_list.append(perm_imp)

        # SHAP on a small held-out subsample
        if max_shap_samples is not None and len(X_test) > max_shap_samples:
            X_shap = X_test.sample(max_shap_samples, random_state=random_state + rep)
        else:
            X_shap = X_test.copy()

        explainer = shap.TreeExplainer(model)

        try:
            shap_values = explainer.shap_values(X_shap, check_additivity=False)
        except TypeError:
            shap_values = explainer.shap_values(X_shap)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        shap_abs = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=predictors,
            name=f"rep_{rep}"
        )
        shap_list.append(shap_abs)

        if store_all_shap:
            shap_df = pd.DataFrame(shap_values, columns=predictors)
            shap_df["repeat"] = rep
            shap_values_frames.append(shap_df)

            feat_df = X_shap.reset_index(drop=True).copy()
            feat_df["repeat"] = rep
            shap_feature_frames.append(feat_df)

        rep_results.append({
            "repeat": rep,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "overfit_gap": train_r2 - test_r2,
        })

    rep_results = pd.DataFrame(rep_results)
    perm_table = pd.concat(perm_list, axis=1)
    shap_table = pd.concat(shap_list, axis=1)

    perm_summary = pd.DataFrame({
        "perm_mean": perm_table.mean(axis=1),
        "perm_median": perm_table.median(axis=1),
        "perm_q25": perm_table.quantile(0.25, axis=1),
        "perm_q75": perm_table.quantile(0.75, axis=1),
    }).sort_values("perm_median", ascending=False)

    shap_summary = pd.DataFrame({
        "shap_mean": shap_table.mean(axis=1),
        "shap_median": shap_table.median(axis=1),
        "shap_q25": shap_table.quantile(0.25, axis=1),
        "shap_q75": shap_table.quantile(0.75, axis=1),
    }).sort_values("shap_median", ascending=False)

    if store_all_shap:
        shap_values_concat = pd.concat(shap_values_frames, axis=0, ignore_index=True)
        shap_features_concat = pd.concat(shap_feature_frames, axis=0, ignore_index=True)
    else:
        shap_values_concat = None
        shap_features_concat = None

    return (
        rep_results,
        perm_table,
        perm_summary,
        shap_table,
        shap_summary,
        shap_values_concat,
        shap_features_concat,
    )

def fit_rf_one_grouped_repeat_with_shap(
    X,
    y,
    groups,
    predictors,
    rf_params,
    repeat_to_use,
    n_repeats_total=10,
    test_size=0.2,
    random_state=42,
    max_shap_samples=500,
):
    """
    Recreate one specific GroupShuffleSplit repeat and compute SHAP
    on the held-out test set for that repeat.
    """
    X = X[predictors].copy()

    splitter = GroupShuffleSplit(
        n_splits=n_repeats_total,
        test_size=test_size,
        random_state=random_state,
    )

    splits = list(splitter.split(X, y, groups))
    train_idx, test_idx = splits[repeat_to_use - 1]   # repeat numbers start at 1

    X_train = X.iloc[train_idx].copy()
    X_test  = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test  = y.iloc[test_idx].copy()

    model = RandomForestRegressor(
        **rf_params,
        random_state=random_state + repeat_to_use,
        bootstrap=True,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))

    if max_shap_samples is not None and len(X_test) > max_shap_samples:
        X_shap = X_test.sample(max_shap_samples, random_state=random_state + repeat_to_use)
    else:
        X_shap = X_test.copy()

    explainer = shap.TreeExplainer(model)

    try:
        shap_values = explainer.shap_values(X_shap, check_additivity=False)
    except TypeError:
        shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    shap_df = pd.DataFrame(shap_values, columns=predictors)
    X_shap_df = X_shap.reset_index(drop=True).copy()

    out = {
        "model": model,
        "repeat": repeat_to_use,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_shap": X_shap_df,
        "shap_values": shap_df,
    }
    return out

def choose_representative_repeat(rep_results):
    """
    Pick the repeat whose test R2 is closest to the median test R2.
    """
    med = rep_results["test_r2"].median()
    idx = (rep_results["test_r2"] - med).abs().idxmin()
    return int(rep_results.loc[idx, "repeat"])

def plot_rf_shap_beeswarm(
    shap_df,
    X_shap_df,
    predictors,
    display_names=None,
    title=None,
    max_display=None,
    figsize=(8, 5.5),
):
    feature_names = [display_names.get(p, p) if display_names is not None else p for p in predictors]

    plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_df[predictors].values,
        features=X_shap_df[predictors],
        feature_names=feature_names,
        show=False,
        max_display=max_display if max_display is not None else len(predictors),
    )
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def choose_three_representative_repeats(rep_results):
    q_targets = rep_results["test_r2"].quantile([0.25, 0.50, 0.75]).values
    chosen = []
    tmp = rep_results.copy()

    for q in q_targets:
        idx = (tmp["test_r2"] - q).abs().idxmin()
        chosen.append(int(tmp.loc[idx, "repeat"]))
        tmp = tmp.drop(idx)

    return chosen

def render_shap_summaryplot_image(
    shap_df,
    X_df,
    importance_series,
    display_names,
    figsize=(11.5, 5.6),
    dpi=220,
    cmap_name="coolwarm",
):
    """
    Render a SHAP summary dot plot as a standalone image and return it as a PIL image.
    """
    ordered = importance_series.sort_values(ascending=False).index.tolist()

    # keep strongest predictor at top
    ordered_for_plot = ordered

    feat_names = [
        f"{display_names.get(p, p)} ({importance_series.loc[p]:.2f})"
        for p in ordered_for_plot
    ]

    X_plot = _signed_norm_to_m11(X_df[ordered_for_plot].copy())
    S_plot = shap_df[ordered_for_plot].copy()

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    plt.sca(ax)
    shap.summary_plot(
        S_plot.values,
        X_plot,
        feature_names=feat_names,
        plot_type="dot",
        max_display=len(ordered_for_plot),
        sort=False,
        show=False,
        color_bar=False,
    )

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=-1, vmax=1)

    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            coll.set_cmap(cmap)
            coll.set_norm(norm)
            coll.set_clim(-1, 1)

    ax.set_xlabel("SHAP value (impact on model output)", fontsize=22)
    ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=22, length=0, pad=4)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.axvline(0, color="0.6", lw=1.2, zorder=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf).convert("RGBA")

def plot_rf_shap_two_panel_composite_row(
    blue_shap_out,
    green_shap_out,
    blue_importance,
    green_importance,
    display_names,
    blue_r2,
    green_r2,
    blue_n,
    green_n,
    shap_figsize=(11.5, 5.6),
    shap_dpi=220,
    final_figsize=(23, 7.8),
    cmap_name="coolwarm",
    cbar_label="Normalised predictor change",
):
    """
    Make a clean one-row two-panel figure by rendering each SHAP summary plot separately
    and composing them into one final figure.
    """
    blue_img = render_shap_summaryplot_image(
        shap_df=blue_shap_out["shap_values"],
        X_df=blue_shap_out["X_shap"],
        importance_series=blue_importance,
        display_names=display_names,
        figsize=shap_figsize,
        dpi=shap_dpi,
        cmap_name=cmap_name,
    )

    green_img = render_shap_summaryplot_image(
        shap_df=green_shap_out["shap_values"],
        X_df=green_shap_out["X_shap"],
        importance_series=green_importance,
        display_names=display_names,
        figsize=shap_figsize,
        dpi=shap_dpi,
        cmap_name=cmap_name,
    )

    fig = plt.figure(figsize=final_figsize)
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        left=0.04, right=0.88, top=0.92, bottom=0.12, wspace=0.10
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.imshow(blue_img)
    ax2.imshow(green_img)

    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # panel labels
    ax1.text(
        0.00, 1.02, "(a)",
        transform=ax1.transAxes,
        ha="left", va="bottom",
        fontsize=21, fontweight="bold",
    )
    ax2.text(
        0.00, 1.02, "(b)",
        transform=ax2.transAxes,
        ha="left", va="bottom",
        fontsize=21, fontweight="bold",
    )

    # title boxes
    ax1.text(
        0.10, 1.02, "Blue water regime",
        transform=ax1.transAxes,
        ha="left", va="bottom",
        fontsize=19, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.22",
                  facecolor=(40/255, 125/255, 210/255),
                  edgecolor="none"),
    )
    ax2.text(
        0.10, 1.02, "Green water regime",
        transform=ax2.transAxes,
        ha="left", va="bottom",
        fontsize=19, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.22",
                  facecolor=(15/255, 115/255, 15/255),
                  edgecolor="none"),
    )

    # R2 + n boxes
    ax1.text(
        0.98, 0.2,
        f"$R^2$ = {blue_r2:.2f}\n$n$ = {blue_n}",
        transform=ax1.transAxes,
        ha="right", va="bottom",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.9),
    )
    ax2.text(
        0.98, 0.2,
        f"$R^2$ = {green_r2:.2f}\n$n$ = {green_n}",
        transform=ax2.transAxes,
        ha="right", va="bottom",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.9),
    )

    # shared colorbar
    cmap = plt.get_cmap(cmap_name)
    sm = ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cmap)
    sm.set_array([])

    cax = fig.add_axes([0.90, 0.28, 0.012, 0.52])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["Low", "0", "High"])

    return fig, (ax1, ax2)


def get_ensemble_mean_key_generic(ds_dict_change_sub, scenario_key):
    keys = list(ds_dict_change_sub[scenario_key].keys())
    ens_keys = [k for k in keys if "ensemble mean" in k.lower()]
    if len(ens_keys) == 0:
        raise KeyError(f"No ensemble mean key found for {scenario_key}.")
    if len(ens_keys) > 1:
        print(f"Multiple ensemble mean keys found, using: {ens_keys[0]}")
    return ens_keys[0]

def get_individual_model_keys_generic(ds_dict_change_sub, scenario_key):
    keys = list(ds_dict_change_sub[scenario_key].keys())
    return [
        k for k in keys
        if ("ensemble mean" not in k.lower())
        and ("ensemble std" not in k.lower())
        and ("std" not in k.lower())
    ]

def analyze_fixed_model_across_ensemble_generic(
    ds_dict_change_sub,
    scenario_key,
    predictors,
    response="bgws_tran_mean",
    subdivisions=None,
    ensemble_mean_key=None,
    lat_bin=10,
    lon_bin=20,
    n_repeats=20,
    test_size=0.2,
    n_splits_inner=4,
    random_state=42,
    n_perm=10,
):
    model_keys = get_individual_model_keys_generic(ds_dict_change_sub, scenario_key)

    if ensemble_mean_key is None:
        ensemble_mean_key = get_ensemble_mean_key_generic(ds_dict_change_sub, scenario_key)

    if subdivisions is None:
        ds_ref = ds_dict_change_sub[scenario_key][ensemble_mean_key]
        subdivisions = list(ds_ref.subdivision.values)

    per_model_rows = []
    perf_rows = []
    raw_outputs = {}

    for model_key in model_keys:
        ds_model = ds_dict_change_sub[scenario_key][model_key]
        raw_outputs[model_key] = {}

        for subdivision in subdivisions:
            df = make_regime_dataframe_with_coords(
                ds_model,
                subdivision=subdivision,
                response=response,
                predictors=predictors,
            )
            df = add_spatial_blocks(df, lat_bin=lat_bin, lon_bin=lon_bin)

            X = df[predictors].copy()
            y = df[response].copy()
            groups = df["spatial_block"].copy()

            rep_results, coef_table, coef_summary, perm_table, perm_summary = (
                repeated_grouped_importance_fixed_set(
                    X, y, groups,
                    predictors=predictors,
                    n_repeats=n_repeats,
                    test_size=test_size,
                    n_splits_inner=n_splits_inner,
                    random_state=random_state,
                    n_perm=n_perm,
                )
            )

            raw_outputs[model_key][subdivision] = {
                "rep_results": rep_results,
                "coef_table": coef_table,
                "coef_summary": coef_summary,
                "perm_table": perm_table,
                "perm_summary": perm_summary,
            }

            perf_rows.append({
                "model_key": model_key,
                "subdivision": subdivision,
                "mean_test_r2": rep_results["test_r2"].mean(),
                "median_test_r2": rep_results["test_r2"].median(),
                "sd_test_r2": rep_results["test_r2"].std(),
                "mean_n_selected": rep_results["n_selected"].mean(),
                "n": len(df),
            })

            for predictor in predictors:
                per_model_rows.append({
                    "model_key": model_key,
                    "subdivision": subdivision,
                    "predictor": predictor,
                    "mean_perm_importance": perm_table.loc[predictor].mean(),
                    "median_perm_importance": perm_table.loc[predictor].median(),
                    "coef_median": coef_table.loc[predictor].median(),
                    "coef_mean": coef_table.loc[predictor].mean(),
                    "selection_frequency": (coef_table.loc[predictor] != 0).mean(),
                    "sign_consistency": np.sign(
                        coef_table.loc[predictor].replace(0, np.nan)
                    ).mean(),
                })

    per_model_long = pd.DataFrame(per_model_rows)
    perf_long = pd.DataFrame(perf_rows)

    return per_model_long, perf_long, raw_outputs


def compute_per_model_context_long_generic(ds_dict_change_sub, scenario_key, predictors, clip=1.0):
    rows = []

    model_keys = get_individual_model_keys_generic(ds_dict_change_sub, scenario_key)

    for model_key in model_keys:
        ds = ds_dict_change_sub[scenario_key][model_key]

        for subdivision in ds.subdivision.values:
            for pred in predictors:
                if pred not in ds.data_vars:
                    continue

                da = ds[pred].sel(subdivision=subdivision)
                vals = da.values
                vals = vals[np.isfinite(vals)]

                if vals.size < 2:
                    ctx = np.nan
                else:
                    sd = np.std(vals, ddof=0)
                    if sd == 0 or not np.isfinite(sd):
                        ctx = np.nan
                    else:
                        ctx = np.mean(vals) / sd
                        ctx = np.clip(ctx, -clip, clip)

                rows.append({
                    "model_key": model_key,
                    "subdivision": subdivision,
                    "predictor": pred,
                    "context_z": ctx,
                })

    return pd.DataFrame(rows)

def run_landuse_driver_figure(
    ds_dict_change_sub_landuse,
    scenario_key,
    ensemble_mean_key,
    predictors,
    predictor_display_names,
    response="bgws_tran_mean",
    lat_bin=10,
    lon_bin=20,
    n_repeats_main=40,
    n_repeats_esm=20,
    test_size=0.2,
    n_splits_inner=4,
    random_state=42,
    n_perm_main=20,
    n_perm_esm=10,
    r2_threshold=0.3,
    figsize=(17.2, 7.6),
    ticklabel_size=20,
    label_size=21,
    title_size=22,
    point_size=300,
    ind_alpha=0.45,
    interval="iqr",
):
    ds_ens = ds_dict_change_sub_landuse[scenario_key][ensemble_mean_key]
    subdivision_names = list(ds_ens.subdivision.values)

    blue_regime_name = subdivision_names[0]
    green_regime_name = subdivision_names[1]

    # ensemble-mean dataframes
    df_blue = make_regime_dataframe_with_coords(
        ds_ens,
        subdivision=blue_regime_name,
        response=response,
        predictors=predictors,
    )
    df_green = make_regime_dataframe_with_coords(
        ds_ens,
        subdivision=green_regime_name,
        response=response,
        predictors=predictors,
    )

    df_blue = add_spatial_blocks(df_blue, lat_bin=lat_bin, lon_bin=lon_bin)
    df_green = add_spatial_blocks(df_green, lat_bin=lat_bin, lon_bin=lon_bin)

    X_blue = df_blue[predictors].copy()
    y_blue = df_blue[response].copy()
    groups_blue = df_blue["spatial_block"].copy()

    X_green = df_green[predictors].copy()
    y_green = df_green[response].copy()
    groups_green = df_green["spatial_block"].copy()

    # ensemble-mean fixed-model importance
    blue_results, blue_coefs, blue_coef_summary, blue_perm, blue_perm_summary = (
        repeated_grouped_importance_fixed_set(
            X_blue, y_blue, groups_blue,
            predictors=predictors,
            n_repeats=n_repeats_main,
            test_size=test_size,
            n_splits_inner=n_splits_inner,
            random_state=random_state,
            n_perm=n_perm_main,
        )
    )

    green_results, green_coefs, green_coef_summary, green_perm, green_perm_summary = (
        repeated_grouped_importance_fixed_set(
            X_green, y_green, groups_green,
            predictors=predictors,
            n_repeats=n_repeats_main,
            test_size=test_size,
            n_splits_inner=n_splits_inner,
            random_state=random_state,
            n_perm=n_perm_main,
        )
    )

    blue_summary = summarize_importance_with_iqr(blue_perm, blue_coefs)
    green_summary = summarize_importance_with_iqr(green_perm, green_coefs)

    # per-ESM robustness
    per_model_long, perf_long, raw_outputs = analyze_fixed_model_across_ensemble_generic(
        ds_dict_change_sub=ds_dict_change_sub_landuse,
        scenario_key=scenario_key,
        predictors=predictors,
        response=response,
        subdivisions=subdivision_names,
        ensemble_mean_key=ensemble_mean_key,
        lat_bin=lat_bin,
        lon_bin=lon_bin,
        n_repeats=n_repeats_esm,
        test_size=test_size,
        n_splits_inner=n_splits_inner,
        random_state=random_state,
        n_perm=n_perm_esm,
    )

    # context for color
    blue_context = compute_clipped_standardized_context(df_blue, predictors, clip=1.0)
    green_context = compute_clipped_standardized_context(df_green, predictors, clip=1.0)

    per_model_context_long = compute_per_model_context_long_generic(
        ds_dict_change_sub=ds_dict_change_sub_landuse,
        scenario_key=scenario_key,
        predictors=predictors,
        clip=1.0,
    )

    fig, axes = plot_driver_importance_two_panel(
        blue_summary=blue_summary.loc[predictors],
        green_summary=green_summary.loc[predictors],
        per_model_long=per_model_long,
        perf_long=perf_long,
        per_model_context_long=per_model_context_long,
        blue_regime_name=blue_regime_name,
        green_regime_name=green_regime_name,
        predictor_display_names=predictor_display_names,
        blue_context=blue_context,
        green_context=green_context,
        blue_r2=blue_results["test_r2"].mean(),
        green_r2=green_results["test_r2"].mean(),
        blue_n=len(df_blue),
        green_n=len(df_green),
        figsize=figsize,
        ticklabel_size=ticklabel_size,
        label_size=label_size,
        title_size=title_size,
        point_size=point_size,
        r2_threshold=r2_threshold,
        ind_alpha=ind_alpha,
        interval=interval,
    )

    outputs = {
        "fig": fig,
        "axes": axes,
        "blue_results": blue_results,
        "green_results": green_results,
        "blue_summary": blue_summary,
        "green_summary": green_summary,
        "per_model_long": per_model_long,
        "perf_long": perf_long,
        "raw_outputs": raw_outputs,
        "df_blue": df_blue,
        "df_green": df_green,
    }
    return outputs

def build_intermodel_sign_agreement_mask(
    ds_dict_change_sub,
    scenario_key,
    var="bgws_tran_mean",
    min_agreement=0.80,
    min_models=None,
):
    """
    Build a confidence mask based on inter-model sign agreement
    for the projected change in `var`.

    Returns an xr.Dataset with:
      - agreement_fraction
      - n_valid_models
      - confidence_mask
    """
    model_keys = get_individual_model_keys(ds_dict_change_sub, scenario_key)

    if min_models is None:
        min_models = int(np.ceil(0.75 * len(model_keys)))  # conservative default

    sign_list = []

    for model_key in model_keys:
        da = ds_dict_change_sub[scenario_key][model_key][var]

        sign_da = xr.where(da > 0, 1,
                   xr.where(da < 0, -1, np.nan))

        sign_da = sign_da.expand_dims(model_key=[model_key])
        sign_list.append(sign_da)

    sign_stack = xr.concat(sign_list, dim="model_key")

    pos_fraction = (sign_stack == 1).mean("model_key", skipna=True)
    neg_fraction = (sign_stack == -1).mean("model_key", skipna=True)
    agreement_fraction = xr.concat(
        [pos_fraction, neg_fraction],
        dim=pd.Index(["pos", "neg"], name="sign_type")
    ).max("sign_type")

    n_valid_models = sign_stack.notnull().sum("model_key")

    confidence_mask = (
        (agreement_fraction >= min_agreement) &
        (n_valid_models >= min_models)
    )

    return xr.Dataset({
        "agreement_fraction": agreement_fraction,
        "n_valid_models": n_valid_models,
        "confidence_mask": confidence_mask,
    })

def apply_subdivision_mask_to_scenario(ds_dict_change_sub, scenario_key, mask_da):
    """
    Apply a common boolean mask (subdivision x lat x lon) to every dataset
    in one scenario of ds_dict_change_sub.
    """
    out = {k: dict(v) for k, v in ds_dict_change_sub.items()}
    out[scenario_key] = {
        key: ds.where(mask_da)
        for key, ds in ds_dict_change_sub[scenario_key].items()
    }
    return out

def run_fixed_model_for_subdivided_ds(
    ds,
    predictors,
    response="bgws_tran_mean",
    lat_bin=10,
    lon_bin=20,
    n_repeats=40,
    test_size=0.2,
    n_splits_inner=4,
    random_state=42,
    n_perm=20,
):
    """
    Reuse your existing helpers on one subdivided dataset
    (e.g. ensemble mean, masked or unmasked).
    """
    subdivisions = list(ds.subdivision.values)
    outputs = {}

    for subdivision in subdivisions:
        df = make_regime_dataframe_with_coords(
            ds,
            subdivision=subdivision,
            response=response,
            predictors=predictors,
        )
        df = add_spatial_blocks(df, lat_bin=lat_bin, lon_bin=lon_bin)

        X = df[predictors].copy()
        y = df[response].copy()
        groups = df["spatial_block"].copy()

        rep_results, coef_table, coef_summary, perm_table, perm_summary = (
            repeated_grouped_importance_fixed_set(
                X, y, groups,
                predictors=predictors,
                n_repeats=n_repeats,
                test_size=test_size,
                n_splits_inner=n_splits_inner,
                random_state=random_state,
                n_perm=n_perm,
            )
        )

        outputs[subdivision] = {
            "df": df,
            "rep_results": rep_results,
            "coef_table": coef_table,
            "coef_summary": coef_summary,
            "perm_table": perm_table,
            "perm_summary": perm_summary,
            "summary_iqr": summarize_importance_with_iqr(perm_table, coef_table),
        }

    return outputs

def _signed_norm_to_m11(Xdf, eps=1e-12):
    Xn = Xdf.copy()
    for c in Xn.columns:
        v = Xn[c].to_numpy(dtype=float)
        m = np.nanmax(np.abs(v))
        Xn[c] = 0.0 if (not np.isfinite(m) or m < eps) else (Xn[c] / m)
    return Xn.clip(-1, 1)

def summarize_importance_with_ci(perm_table, coef_table):
    out = pd.DataFrame({
        "perm_mean": perm_table.mean(axis=1),
        "perm_median": perm_table.median(axis=1),
        "perm_p02_5": perm_table.quantile(0.025, axis=1),
        "perm_p97_5": perm_table.quantile(0.975, axis=1),
        "coef_median": coef_table.median(axis=1),
        "coef_p02_5": coef_table.quantile(0.025, axis=1),
        "coef_p97_5": coef_table.quantile(0.975, axis=1),
        "selection_frequency": (coef_table != 0).mean(axis=1),
        "sign_consistency": np.sign(coef_table.replace(0, np.nan)).mean(axis=1),
    }).sort_values("perm_mean", ascending=False)
    return out