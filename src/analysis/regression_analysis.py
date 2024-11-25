"""
src/analysis/regression_analysis.py

This script provides functions to perform a regression analysis.

Functions:
- get_default_param_grid
- get_model_class
- scale_data
- max_scale
- train_regression_model
- test_train_evaluation
- evaluate_regression_model
- compute_permutation_importance
- compute_shap_values
- regression_analysis
- regression_analysis_subdivision


Usage:
    Import this module in your scripts to perform a regression analysis.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import shap
from sklearn.preprocessing import StandardScaler

def get_default_param_grid(regression_type):
    """
    Get the default parameter grid for the specified regression type.

    Parameters:
    - regression_type (str): The type of regression ('ridge', 'lasso', 'elasticnet').

    Returns:
    - dict: Parameter grid for the specified regression type.
    """
    param_grids = {
        'ridge': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        'lasso': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        'elasticnet': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.2, 0.5, 0.8]}
    }
    if regression_type not in param_grids:
        raise ValueError(f"Unknown regression type: {regression_type}")
    return param_grids[regression_type]

def get_model_class(regression_type):
    """
    Get the model class for the specified regression type.

    Parameters:
    - regression_type (str): The type of regression ('ridge', 'lasso', 'elasticnet').

    Returns:
    - class: Model class corresponding to the specified regression type.
    """
    model_classes = {
        'ridge': Ridge,
        'lasso': Lasso,
        'elasticnet': ElasticNet
    }
    if regression_type not in model_classes:
        raise ValueError(f"Unknown regression type: {regression_type}")
    return model_classes[regression_type]

def scale_data(X, method='std'):
    scaled_data = pd.DataFrame(index=X.index)  # Initialize an empty DataFrame to store scaled data
    scaler_data = {}  # Dictionary to store scaling parameters for each column
    
    for column in X.columns:
        # Select the column data
        col_data = X[[column]]

        if method == 'std':
            scaler = StandardScaler()
            scaled_data[column] = scaler.fit_transform(col_data).flatten()  # Flatten is used to avoid shape mismatch
            scaler_data[column] = {'mean': scaler.mean_[0], 'std': scaler.scale_[0]}

        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data[column] = scaler.fit_transform(col_data).flatten()
            scaler_data[column] = {'min': scaler.data_min_[0], 'max': scaler.data_max_[0]}

        elif method == 'max':
            # Call the max_scale function to handle the scaling
            scaled_col, max_val = max_abs_scale(col_data)
            scaled_data[column] = scaled_col[column]
            scaler_data[column] = {'max': max_val}

        elif method is None:
            scaled_data[column] = col_data  # Directly assign the original data if no scaling is applied

        else:
            raise ValueError('Unsupported scaling method. Choose "std", "minmax", or "max".')

    return scaled_data, scaler_data

def max_abs_scale(data):
    """
    Scales each feature in the DataFrame to the range [-1, 1] based on its maximum absolute value.

    Parameters:
    - data (pd.DataFrame): Input DataFrame with numerical features.

    Returns:
    - scaled_data (pd.DataFrame): DataFrame where each feature is scaled to [-1, 1].
    - scale_factors (pd.Series): The maximum absolute value used for scaling each column.
    """
    # Compute the maximum absolute value for each column
    max_abs_val = data.abs().max(axis=0)
    
    # Scale each column by dividing by the maximum absolute value
    scaled_data = data.div(max_abs_val)

    return scaled_data, max_abs_val

def train_regression_model(X_train, y_train, regression_type, param_grid, 
                           cv_strategy, scoring):
    """
    Train a regression model using Grid Search with cross-validation.

    Parameters:
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data target.
    - regression_type (str): The regression model type to be used ('ridge', 'lasso', 'elasticnet').
    - param_grid (dict): Parameter grid for hyperparameter tuning.
    - cv_strategy (int or cross-validation generator, optional): Cross-validation strategy.
    - scoring (str, optional): Scoring metric for Grid Search.

    Returns:
    - estimator: Best estimator found by Grid Search.
    """
    model_class = get_model_class(regression_type)
    model_instance = model_class()
    
    grid_search = GridSearchCV(model_instance, param_grid, cv=cv_strategy, scoring=scoring, n_jobs=-1) # n_jobs=-1, you ensure that the GridSearchCV will use all available CPU cores to perform the grid search
    grid_search.fit(X_train, y_train) 
    
    best_model = grid_search.best_estimator_

    return best_model

def test_train_evaluation(model, X_train, y_train, X_test, y_test, scoring):
    """
    Evaluate the model on both the training and test sets.

    Parameters:
    - model (estimator): The trained model to be evaluated.
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data target.
    - X_test (array-like): Test data features.
    - y_test (array-like): Test data target.
    - scoring (str, optional): Scoring metric for Grid Search, default is 'r2'.

    Returns:
    - dict: Dictionary containing MSE and R2 scores for both training and test sets.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    if scoring == 'r2':
        train = r2_score(y_train, y_train_pred)
        test = r2_score(y_test, y_test_pred)
    elif scoring == 'mse':
        train = mean_squared_error(y_train, y_train_pred)
        test = mean_squared_error(y_test, y_test_pred)
    else:
        raise ValueError(f"Unknown scoring type: {scoring}")
    
    
    return {
        f'{scoring.upper()} Train': train,
        f'{scoring.upper()} Test': test
    }

def evaluate_regression_model(regression_model, X_train, X_test, y_train, y_test, 
                              evaluation_method, cv_strategy, scoring):
    """
    Evaluate a regression model using either cross-validation or a separate test set.

    Parameters:
    - regression_model: Trained model
    - X_train (array-like): Training data features.
    - X_test (array-like): Test data features.
    - y_train (array-like): Training data target.
    - y_test (array-like): Test data target.
    - evaluation_method (str, optional): Method of evaluation ('test', 'cv' or 'test_and_cv'), default is 'test'.
    - cv_strategy (int or cross-validation generator, optional): Cross-validation strategy, default is 5-fold.
    - scoring (str, optional): Scoring metric for Grid Search, default is 'r2'.

    Returns:
    - dict: Dictionary containing evaluation metrics.
    """
    if evaluation_method == 'test':
        # Evaluate on test set
        performance = test_train_evaluation(regression_model, X_train, y_train, X_test, y_test, scoring)
        
    elif evaluation_method == 'cv':
        # Evaluate using cross-validation
        cv_scores = cross_val_score(regression_model, np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)), cv=cv_strategy, scoring=scoring)
        performance = {'CV Mean Score': np.mean(cv_scores), 'CV Std Dev': np.std(cv_scores)}

    elif evaluation_method == 'test_and_cv':
        # Evaluate on test set
        performance = test_train_evaluation(regression_model, X_train, y_train, X_test, y_test, scoring)

        # Evaluate using cross-validation
        cv_scores = cross_val_score(regression_model, np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)), cv=cv_strategy, scoring=scoring)
        
        # Update the performance dictionary with CV results
        performance.update({
            'CV Mean Score': np.mean(cv_scores),
            'CV Std Dev': np.std(cv_scores)
        })
    
    else:
        raise ValueError("Unknown evaluation method: choose 'test' or 'cv'")
    
    return performance

def compute_permutation_importance(regression_model, X_train, X_test, y_train, y_test, n_permutations=20):
    """
    Compute permutation importance for the model on both training and test data.

    Parameters:
    - regression_model: Trained regression model.
    - X_train (array-like): Training data features.
    - X_test (array-like): Test data features.
    - y_train (array-like): Training data target.
    - y_test (array-like): Test data target.
    - n_permutations (int, optional): Number of permutations for permutation importance, default is 20.

    Returns:
    - dict: Dictionary containing permutation importances for training and test data.
    """
    importances_test = permutation_importance(regression_model, X_test, y_test, n_repeats=n_permutations, random_state=42, scoring='r2').importances
    importances_train = permutation_importance(regression_model, X_train, y_train, n_repeats=n_permutations, random_state=42, scoring='r2').importances

    return {
        'Permutation Importance Testing Data': importances_test,
        'Permutation Importance Training Data': importances_train
    }


def compute_shap_values(model, X_train, X_test):
    """
    Compute SHAP values for a given model and dataset.

    Parameters:
    - model: Trained regression model.
    - X_train (array-like): Training data features.
    - X_test (array-like): Test data features.

    Returns:
    - dict: Dictionary containing SHAP values for training and test data.
    """
    # Use the appropriate SHAP explainer based on the model type
    if isinstance(model, (Ridge, Lasso, ElasticNet)):
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.Explainer(model, X_train)
    
    shap_values_train = explainer(X_train)
    shap_values_test = explainer(X_test)
    
    return {
        'SHAP Values Training Data': shap_values_train,
        'SHAP Values Testing Data': shap_values_test
    }

def regression_analysis(ds, predictor_vars, predictant, scaling_method, test_size,
                        regression_type, param_grid, grid_cell_threshold, 
                        cv_folds, scoring, evaluation_method, 
                        variable_importance_method, n_permutations):
    """
    Perform regression analysis on a given dataset.

    Parameters:
    - ds (xarray.Dataset): The dataset containing the data for analysis.
    - predictor_vars (list): List of predictor variable names.
    - predictant (str): Name of the target variable.
    - scaling_method (str): Name of scaling method, default is 'max'.
    - test_size (float): Proportion of the dataset to include in the test split.
    - regression_type (str): Type of regression ('ridge', 'lasso', 'elasticnet').
    - param_grid (dict): Parameter grid for hyperparameter tuning.
    - grid_cell_threshold (int): Minimum number of grid cells required for analysis.
    - cv_folds (int): Number of folds for cross-validation.
    - scoring (str): Scoring metric for Grid Search.
    - evaluation_method (str, optional): Method of evaluation ('test', 'cv' or 'test_and_cv'), default is 'test'.
    - variable_importance_method (str): Method for variable importance analysis.
    - n_permutations (int): Number of permutations for permutation importance.

    Returns:
    - dict: Dictionary containing the regression model, performance metrics, and permutation importances.
            None if the number of grid cells is below the threshold.
    """
    
    # Create pandas DataFrame from xarray dataset, dropping rows with NaNs in predictor and predictant columns
    df = ds.to_dataframe().dropna(subset=predictor_vars + [predictant])

    # Perform regression analysis only if the number of grid cells exceeds the threshold
    if len(df) > grid_cell_threshold:
        print(f'Regression analysis based on {len(df)} grid cells')
        
        X = df[predictor_vars]
        y = df[predictant]

        # Scale the predictors data
        X_scaled,_ = scale_data(X, method=scaling_method)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        
        # Train the regression model using Grid Search with cross-validation
        regression_model = train_regression_model(X_train, y_train, regression_type, param_grid, cv_folds, scoring)
        # Evaluate the model performance
        performance = evaluate_regression_model(regression_model, X_train, X_test, y_train, y_test, evaluation_method, cv_folds, scoring)

        if variable_importance_method == 'PI':
            # Compute permutation importance for the model
            variable_importance = compute_permutation_importance(regression_model, X_train, X_test, y_train, y_test, n_permutations)
        elif variable_importance_method == 'SHAP':
            # Compute SHAP values
            variable_importance = compute_shap_values(regression_model, X_train, X_test)
                                                                       
        return {
            'regression_model': regression_model, 
            'performance': performance,
            'variable_importance': variable_importance,
            'n': len(df)
        }

    # Print a message if the number of grid cells is below the threshold
    print('No regression analysis as grid cells < threshold ')
    
    return None

def regression_analysis_regions_subdivisions(ds_change, predictor_vars, predictant, scaling_method='max', 
                                    test_size = 0.3, regression_type='elasticnet', 
                                    param_grid=None, grid_cell_threshold=50, 
                                    cv_folds=5, scoring='r2', evaluation_method='test', 
                                    variable_importance_method='PI', n_permutations=20,
                                    selected_region=None, selected_subdivision=None):
    """
    Perform regression analysis on specified regions and subdivisions, or all if none specified.

    Parameters:
    - ds_change (xarray.Dataset): Dataset containing the data.
    - predictor_vars (list): List of predictor variable names.
    - predictant (str): Name of the predictant variable.
    - scaling_method (str): Name of scaling method, default is 'max'.
    - test_size (int): Size of test dataset, default is 0.3.
    - regression_type (str): Type of regression ('ridge', 'lasso', 'elasticnet').
    - param_grid (dict, optional): Parameter grid for hyperparameter tuning, default is None.
    - grid_cell_threshold (int, optional): Minimum number of grid cells required for analysis, default is 50.
    - cv_folds (int, optional): Number of folds for cross-validation, default is 5.
    - scoring (str, optional): Scoring metric for Grid Search, default is 'r2'.
    - evaluation_method (str, optional): Method of evaluation ('test', 'cv' or 'test_and_cv'), default is 'test'.
    - variable_importance_method (str): Method for variable importance analysis.
    - n_permutations (int, optional): Number of permutations for permutation importance, default is 20.
    - selected_region (str, optional): Specific region to analyze, default is None (analyze all).
    - selected_subdivision (str, optional): Specific subdivision to analyze, default is None (analyze all).

    Returns:
    - dict: Dictionary containing analysis results for regions and subdivisions.
    """
    if not param_grid:
        param_grid = get_default_param_grid(regression_type)
    
    results = {}

    # Determine regions to iterate over
    regions = selected_region if selected_region else ds_change.region.values

    for region in regions:
        # Get the name of the region
        region_name = ds_change.names.sel(region=region).values

        if f'{region_name}' not in results:
            results[f'{region_name}'] = {}

        # Determine subdivisions to iterate over
        subdivisions = [selected_subdivision] if selected_subdivision else ds_change.subdivision.values

        for subdivision in subdivisions:
            ds = ds_change.sel(region=region, subdivision=subdivision)

            print(f'Performing regression analysis for {region_name} - {subdivision}')

            model_results = regression_analysis(ds, predictor_vars, predictant, scaling_method, test_size,
                                                regression_type, param_grid, grid_cell_threshold, 
                                                cv_folds, scoring, evaluation_method, 
                                                variable_importance_method, n_permutations)

            if model_results:
                results[f'{region_name}'][f'{subdivision}'] = model_results

    return results

def regression_analysis_regions(ds_change, predictor_vars, predictant, scaling_method='max', 
                                    test_size = 0.3, regression_type='elasticnet', 
                                    param_grid=None, grid_cell_threshold=50, 
                                    cv_folds=5, scoring='r2', evaluation_method='test', 
                                    variable_importance_method='PI', n_permutations=20,
                                    selected_region=None):
    """
    Perform regression analysis on specified regions, or all if none specified.

    Parameters:
    - ds_change (xarray.Dataset): Dataset containing the data.
    - predictor_vars (list): List of predictor variable names.
    - predictant (str): Name of the predictant variable.
    - scaling_method (str): Name of scaling method, default is 'max'.
    - test_size (int): Size of test dataset, default is 0.3.
    - regression_type (str): Type of regression ('ridge', 'lasso', 'elasticnet').
    - param_grid (dict, optional): Parameter grid for hyperparameter tuning, default is None.
    - grid_cell_threshold (int, optional): Minimum number of grid cells required for analysis, default is 50.
    - cv_folds (int, optional): Number of folds for cross-validation, default is 5.
    - scoring (str, optional): Scoring metric for Grid Search, default is 'r2'.
    - evaluation_method (str, optional): Method of evaluation ('test', 'cv' or 'test_and_cv'), default is 'test'.
    - variable_importance_method (str): Method for variable importance analysis.
    - n_permutations (int, optional): Number of permutations for permutation importance, default is 20.
    - selected_region (str, optional): Specific region to analyze, default is None (analyze all).

    Returns:
    - dict: Dictionary containing analysis results for regions and subdivisions.
    """
    if not param_grid:
        param_grid = get_default_param_grid(regression_type)
    
    results = {}

    # Determine regions to iterate over
    regions = selected_region if selected_region else ds_change.region.values

    for region in regions:
        # Get the name of the region
        region_name = ds_change.names.sel(region=region).values

        if f'{region_name}' not in results:
            results[f'{region_name}'] = {}

        ds = ds_change.sel(region=region)

        print(f'Performing regression analysis for {region_name}')

        model_results = regression_analysis(ds, predictor_vars, predictant, scaling_method, test_size,
                                            regression_type, param_grid, grid_cell_threshold, 
                                            cv_folds, scoring, evaluation_method, 
                                            variable_importance_method, n_permutations)

        if model_results:
            results[f'{region_name}'] = model_results

    return results
