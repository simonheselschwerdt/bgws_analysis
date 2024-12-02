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
Date: [2024-11-28]
Dependencies: pandas, scikit-learn, xarray
"""

import xarray as xr
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

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

def train_regression_model(X_train, y_train, param_grid, cv_folds):
    """
    Train an ElasticNet regression model using Grid Search with cross-validation.

    Parameters:
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data target.
    - param_grid (dict): Parameter grid for hyperparameter tuning.
    - cv_folds (int): Number of folds for cross-validation.

    Returns:
    - best_model (ElasticNet): Best ElasticNet model identified by Grid Search.
    """
    elastic_net_model = ElasticNet()
    grid_search = GridSearchCV(estimator=elastic_net_model, param_grid=param_grid, cv=cv_folds, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

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

def compute_permutation_importance(model, X_train, X_test, y_train, y_test, n_permutations=20):
    """
    Compute permutation importance for training and test datasets.

    Parameters:
    - model (ElasticNet): Trained regression model.
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

def regression_analysis(ds, predictor_vars, predictant, test_size, param_grid, cv_folds, n_permutations):
    """
    Perform regression analysis using ElasticNet, GridSearchCV, and permutation importance.

    Parameters:
    - ds (xarray.Dataset): Dataset containing predictors and target variable.
    - predictor_vars (list): List of predictor variable names.
    - predictant (str): Target variable name.
    - test_size (float): Proportion of data to use for the test split.
    - param_grid (dict): Grid of hyperparameters for ElasticNet.
    - cv_folds (int): Number of folds for cross-validation.
    - n_permutations (int): Number of permutations for importance analysis.

    Returns:
    - dict: Results including the model, performance metrics, and variable importances.
    """
    # Prepare data: Drop rows with NaNs and extract predictors/target
    df = ds.to_dataframe().dropna(subset=predictor_vars + [predictant])
    X, y = df[predictor_vars], df[predictant]
    
    # Scale predictors using max-absolute scaling
    X_scaled, _ = scale_data(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    
    # Train the regression model with hyperparameter tuning
    regression_model = train_regression_model(X_train, y_train, param_grid, cv_folds)
    
    # Evaluate model performance on training and testing datasets
    performance = test_train_evaluation(regression_model, X_train, y_train, X_test, y_test)

    # Compute permutation importance for predictors
    variable_importance = compute_permutation_importance(regression_model, X_train, X_test, y_train, y_test, n_permutations)
    
    return {
        'regression_model': regression_model,
        'performance': performance,
        'variable_importance': variable_importance,
        'n': len(df)
    }