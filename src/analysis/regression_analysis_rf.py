"""
Regression Analysis Script for Variable Importance Assessment
-------------------------------------------------------------
This script performs regression analysis using random forest, hyperparameter optimization
with GridSearchCV, and computes variable importance using SHAP (SHapley Additive exPlanations). It also computes permutation importance for comparison.

Functions:
- scale_data: Scales features to the range [-1, 1].
- max_abs_scale: Helper function for scaling individual features.
- test_train_evaluation: Evaluates model performance on training and test sets.
- train_random_forest_model: Train random forest model.
- compute_permutation_importance_rf: Computes feature importances via permutations.
- compute_random_forest_importance: Computes feature importances based on mean decrease in impurity.
- compute_shap_values: Computes feature importances via permutations.
- random_forest_analysis: Orchestrates the full workflow for regression analysis.
- select_best_model_with_overfitting_check: Find hyperparameter selection with hightest R2 and R2 difference below 10% comparing train and test data.
- random_forest_analysis_with_overfitting_check: Orchestrates hyperparameter selection.

Author: [Simon P. Heselschwerdt]
Date: [2025-02-06]
Dependencies: pandas, scikit-learn, xarray, shap
"""

import xarray as xr
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

import shap

def scale_data(X):
    """
    Scale each feature in the DataFrame to the range [-1, 1] using max-absolute scaling.

    Parameters:
    - X (pd.DataFrame): Input DataFrame with features to be scaled.

    Returns:
    - scaled_data (pd.DataFrame): Scaled features as a DataFrame.
    - scaler_data (dict): Scaling parameters (max values) for each feature.
    """
    scaled_data = pd.DataFrame(index=X.index)
    scaler_data = {}
    
    for column in X.columns:
        col_data = X[[column]]
        scaled_col, max_val = max_abs_scale(col_data)
        scaled_data[column] = scaled_col[column]
        scaler_data[column] = {'max': max_val}
        
    return scaled_data, scaler_data

def max_abs_scale(data):
    """
    Scale each column of a DataFrame to the range [-1, 1] based on maximum absolute value.

    Parameters:
    - data (pd.DataFrame): Input DataFrame with numerical features.

    Returns:
    - scaled_data (pd.DataFrame): DataFrame with scaled features.
    - max_abs_val (pd.Series): Maximum absolute values for scaling.
    """
    max_abs_val = data.abs().max(axis=0)
    scaled_data = data.div(max_abs_val)
    return scaled_data, max_abs_val

def test_train_evaluation(model, X_train, y_train, X_test, y_test):
    """
    Evaluate a trained model on both training and test sets using R2 score.

    Parameters:
    - model (ElasticNet): Trained regression model.
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data target.
    - X_test (array-like): Test data features.
    - y_test (array-like): Test data target.

    Returns:
    - dict: R2 scores for training and test sets.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return {
        'R2 Train': r2_score(y_train, y_train_pred),
        'R2 Test': r2_score(y_test, y_test_pred)
    }

def train_random_forest_model(X_train, y_train, param_grid, cv_folds=None):
    """
    Train a Random Forest regression model using the provided hyperparameters.

    Parameters:
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data target.
    - param_grid (dict): Hyperparameters for the Random Forest model.

    Returns:
    - best_model (RandomForestRegressor): Trained Random Forest model.
    """
    # Initialize the Random Forest model
    rf_model = RandomForestRegressor(**param_grid, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    return rf_model

def compute_permutation_importance_rf(model, X_train, X_test, y_train, y_test, n_permutations=20):
    """
    Compute permutation importance for Random Forest on training and test datasets.

    Parameters:
    - model (RandomForestRegressor): Trained Random Forest model.
    - X_train, X_test (array-like): Training and test data features.
    - y_train, y_test (array-like): Training and test data targets.
    - n_permutations (int): Number of permutations for importance computation.

    Returns:
    - dict: Permutation importances for training and test datasets.
    """
    importances_test = permutation_importance(model, X_test, y_test, n_repeats=n_permutations, random_state=42, scoring='r2').importances
    importances_train = permutation_importance(model, X_train, y_train, n_repeats=n_permutations, random_state=42, scoring='r2').importances
    return {
        'Permutation Importance Testing Data': importances_test,
        'Permutation Importance Training Data': importances_train
    }

def compute_random_forest_importance(model, feature_names):
    """
    Compute feature importance directly from the Random Forest model.

    Parameters:
    - model (RandomForestRegressor): Trained Random Forest model.
    - feature_names (list): Names of the features.

    Returns:
    - pd.DataFrame: DataFrame with features and their importances.
    """
    importances = model.feature_importances_
    return pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)


def compute_shap_values(model, X_train, X_test):
    """
    Compute SHAP values for a trained model.

    Parameters:
    - model: Trained Random Forest model or any compatible model for SHAP.
    - X_train (pd.DataFrame): Training data features.
    - X_test (pd.DataFrame): Test data features.

    Returns:
    - shap_values_train: SHAP values for training data.
    - shap_values_test: SHAP values for test data.
    - explainer: SHAP explainer object for additional analysis.
    """
    # Initialize the SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for training and test sets
    shap_values_train = explainer.shap_values(X_train)
    shap_values_test = explainer.shap_values(X_test)
    
    return shap_values_train, shap_values_test, explainer

def random_forest_analysis(ds, predictor_vars, predictant, test_size, param_grid, n_permutations, shap=False):
    """
    Perform Random Forest regression analysis with hyperparameter tuning using GridSearchCV,
    compute SHAP values and variable importances.

    Parameters:
    - ds (xarray.Dataset): Dataset containing predictors and target variable.
    - predictor_vars (list): List of predictor variable names.
    - predictant (str): Target variable name.
    - test_size (float): Proportion of data to use for the test split.
    - param_grid (dict): Hyperparameters for Random Forest.
    - n_permutations (int): Number of permutations for importance analysis.

    Returns:
    - dict: Results including the model, performance metrics, SHAP values, and variable importances.
    """
    # Prepare data: Drop rows with NaNs and extract predictors/target
    df = ds.to_dataframe().dropna(subset=predictor_vars + [predictant])
    X, y = df[predictor_vars], df[predictant]
    
    # Scale predictors using max-absolute scaling
    X_scaled, _ = scale_data(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    
    # Train the Random Forest model with GridSearchCV for hyperparameter tuning
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Print the best parameters
    print(f"Best hyperparameters for Random Forest: {best_params}")
    
    # Evaluate model performance on training and testing datasets
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    performance = {
        'R2 Train': r2_score(y_train, y_train_pred),
        'R2 Test': r2_score(y_test, y_test_pred)
    }
    
    # Compute feature importances directly from the model
    feature_importances = compute_random_forest_importance(best_model, predictor_vars)
    
    # Compute permutation importance for predictors
    variable_importance = compute_permutation_importance_rf(best_model, X_train, X_test, y_train, y_test, n_permutations)

    if shap:
        # Compute SHAP values for training and test sets
        shap_values_train, shap_values_test, explainer = compute_shap_values(best_model, X_train, X_test)
        
        return {
            'regression_model': best_model,
            'performance': performance,
            'feature_importances': feature_importances,
            'variable_importance': variable_importance,
            'shap_values_train': shap_values_train,
            'shap_values_test': shap_values_test,
            'shap_explainer': explainer,
            'n': len(df),
            'X_train': X_train,  # Include X_train in results
            'X_test': X_test     # Include X_test in results
        }
    else:
        return {
            'regression_model': best_model,
            'performance': performance,
            'feature_importances': feature_importances,
            'variable_importance': variable_importance,
            'n': len(df),
            'X_train': X_train,  # Include X_train in results
            'X_test': X_test     # Include X_test in results
        }

def select_best_model_with_overfitting_check(grid_search, train_r2, test_r2, overfit_threshold=0.10):
    """
    Select the best model based on test performance while ensuring it doesn't overfit beyond the specified threshold.

    Parameters:
    - grid_search: The fitted GridSearchCV object containing models and scores.
    - train_r2 (list): List of training R² scores corresponding to the models.
    - test_r2 (list): List of testing R² scores corresponding to the models.
    - overfit_threshold (float): Maximum allowed difference between training and testing R² scores.

    Returns:
    - best_model: The best estimator from GridSearchCV satisfying performance and overfitting criteria.
    - best_params: The hyperparameters of the selected model.
    - best_index: The index of the best model in the GridSearch results.
    """
    best_index = None
    best_test_r2 = -float('inf')
    best_model = None

    # Retrieve all hyperparameters
    grid_search_results = grid_search.cv_results_['params']

    # Loop through each model and evaluate the criteria
    for idx, (train_score, test_score) in enumerate(zip(train_r2, test_r2)):
        if train_score - test_score <= overfit_threshold:  # Ensure it meets overfitting criteria
            if test_score > best_test_r2:  # Choose the best test performance under the constraint
                best_test_r2 = test_score
                best_index = idx

    if best_index is not None:
        best_params = grid_search_results[best_index]  # Retrieve best hyperparameters
        best_model = grid_search.best_estimator_  # Get the best trained model

        print(f"Selected Model Hyperparameters: {best_params}")

    else:
        print("No suitable model found within the overfitting threshold.")

    return best_model, best_params, best_index


def random_forest_analysis_with_overfitting_check(ds, predictor_vars, predictant, test_size, param_grid, overfit_threshold=0.10):
    """
    Perform Random Forest regression analysis and ensure no overfitting beyond a defined threshold.

    Parameters:
    - ds (xarray.Dataset): Dataset containing predictors and target variable.
    - predictor_vars (list): List of predictor variable names.
    - predictant (str): Target variable name.
    - test_size (float): Proportion of data to use for the test split.
    - param_grid (dict): Hyperparameters for Random Forest.
    - overfit_threshold (float): Maximum allowed difference between training and testing R² scores.
    - n_permutations (int): Number of permutations for importance analysis.

    Returns:
    - dict: Results including the model, performance metrics, SHAP values, and variable importances.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    # Prepare data: Drop rows with NaNs and extract predictors/target
    df = ds.to_dataframe().dropna(subset=predictor_vars + [predictant])
    X, y = df[predictor_vars], df[predictant]
    
    # Scale predictors using max-absolute scaling
    X_scaled, _ = scale_data(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    
    # Initialize GridSearchCV for Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='r2', n_jobs=-1, return_train_score=True)
    grid_search.fit(X_train, y_train)

    # Retrieve training and testing R² scores
    train_r2_scores = grid_search.cv_results_['mean_train_score']
    test_r2_scores = grid_search.cv_results_['mean_test_score']
    best_model, best_params, best_index = select_best_model_with_overfitting_check(
        grid_search=grid_search,
        train_r2=train_r2_scores,
        test_r2=test_r2_scores,
        overfit_threshold=overfit_threshold
    )

    if best_model is None:
        raise ValueError("No model satisfies the overfitting threshold. Consider relaxing the threshold or tuning hyperparameters.")

    # Use best estimator from GridSearchCV
    best_model_instance = grid_search.best_estimator_
    best_model_instance.fit(X_train, y_train)

    # Evaluate model performance on the full training and testing datasets
    y_train_pred = best_model_instance.predict(X_train)
    y_test_pred = best_model_instance.predict(X_test)
    performance = {
        'R2 Train': r2_score(y_train, y_train_pred),
        'R2 Test': r2_score(y_test, y_test_pred)
    }
    
    
    return {
        'regression_model': best_model_instance,
        'performance': performance,
        'n': len(df),
        'X_train': X_train,  # Include X_train in results
        'X_test': X_test     # Include X_test in results
    }