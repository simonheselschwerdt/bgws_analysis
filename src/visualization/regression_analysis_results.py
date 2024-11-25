"""
src/visualization/regression_analysis_results.py

This script provides functions to visualize the results of the regression analysis.

Functions:
- 

Usage:
    Import this module in your scripts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tables
import seaborn as sns
import matplotlib.colors as mcolors
import colormaps as colmap
from matplotlib.colors import TwoSlopeNorm
import shap

def process_result(result, r2_train_scores, r2_test_scores, cv_mean_scores):
    """
    Processes a single result entry to extract performance metrics.

    Parameters:
    - result (dict): The result dictionary containing performance metrics.
    - r2_train_scores (list): List to append R2 scores for training data.
    - r2_test_scores (list): List to append R2 scores for testing data.
    - cv_mean_scores (list): List to append cross-validation mean scores.

    Returns:
    - None
    """
    if isinstance(result, dict) and 'performance' in result:
        performance = result['performance']
        
        # Handle test/train results
        if 'R2 Train' in performance:
            r2_train_scores.append(performance['R2 Train'])
        if 'R2 Test' in performance:
            r2_test_scores.append(performance['R2 Test'])
        
        # Handle cross-validation results
        if 'CV Mean Score' in performance:
            cv_mean_scores.append(performance['CV Mean Score'])

def compute_mean_r2_regions(results):
    """
    Computes the mean R2 scores or cross-validation scores for results without subdivisions.

    Parameters:
    - results (dict): Dictionary containing the regression analysis results without subdivisions.

    Returns:
    - mean_r2_train (float or None): Mean R2 score for training data, if available.
    - mean_r2_test (float or None): Mean R2 score for testing data, if available.
    - mean_cv_score (float or None): Mean cross-validation score, if available.
    """
    r2_train_scores = []
    r2_test_scores = []
    cv_mean_scores = []

    for region_name, result in results.items():
        process_result(result, r2_train_scores, r2_test_scores, cv_mean_scores)

    mean_r2_train = np.mean(r2_train_scores) if r2_train_scores else None
    mean_r2_test = np.mean(r2_test_scores) if r2_test_scores else None
    mean_cv_score = np.mean(cv_mean_scores) if cv_mean_scores else None

    # Print the results accordingly
    if mean_r2_train is not None and mean_r2_test is not None:
        print(f"Mean R2 for Training Data: {mean_r2_train:.2f}")
        print(f"Mean R2 for Testing Data: {mean_r2_test:.2f}")
    if mean_cv_score is not None:
        print(f"Mean CV Score: {mean_cv_score:.2f}")

    return mean_r2_train, mean_r2_test, mean_cv_score


import numpy as np

def compute_mean_r2_regions_subdivisions(results, include_global=True):
    """
    Computes the mean R2 scores or cross-validation scores for results with subdivisions.

    Parameters:
    - results (dict): Dictionary containing the regression analysis results with subdivisions.
    - include_global (bool): Whether to include the 'Global' region in the mean calculations. Default is True.

    Returns:
    - mean_r2_train (float or None): Mean R2 score for training data, if available.
    - mean_r2_test (float or None): Mean R2 score for testing data, if available.
    - mean_cv_score (float or None): Mean cross-validation score, if available.
    """
    r2_train_scores = []
    r2_test_scores = []
    cv_mean_scores = []

    for region_name, content in results.items():
        # Skip the 'Global' region if include_global is False
        if not include_global and region_name == 'Global':
            continue
        
        # Process each subdivision
        if isinstance(content, dict):
            for key, value in content.items():
                # If the value is a dictionary with performance, it's a result
                if isinstance(value, dict) and 'performance' in value:
                    process_result(value, r2_train_scores, r2_test_scores, cv_mean_scores)
                elif isinstance(value, dict):
                    # Otherwise, it's another level of subdivision
                    for sub_key, sub_value in value.items():
                        process_result(sub_value, r2_train_scores, r2_test_scores, cv_mean_scores)

    mean_r2_train = np.mean(r2_train_scores) if r2_train_scores else None
    mean_r2_test = np.mean(r2_test_scores) if r2_test_scores else None
    mean_cv_score = np.mean(cv_mean_scores) if cv_mean_scores else None

    # Print the results accordingly
    if mean_r2_train is not None and mean_r2_test is not None:
        print(f"Mean R2 for Training Data: {mean_r2_train:.2f}")
        print(f"Mean R2 for Testing Data: {mean_r2_test:.2f}")
    if mean_cv_score is not None:
        print(f"Mean CV Score: {mean_cv_score:.2f}")

    return mean_r2_train, mean_r2_test, mean_cv_score

def compute_mean_r2_regions_subdivisions(results, include_global=True):
    """
    Computes the mean R2 scores or cross-validation scores for results with subdivisions.

    Parameters:
    - results (dict): Dictionary containing the regression analysis results with subdivisions.
    - include_global (bool): Whether to include the 'Global' region in the mean calculations. Default is True.

    Returns:
    - mean_r2_train (float or None): Mean R2 score for training data, if available.
    - mean_r2_test (float or None): Mean R2 score for testing data, if available.
    - mean_cv_score (float or None): Mean cross-validation score, if available.
    """
    r2_train_scores = []
    r2_test_scores = []
    cv_mean_scores = []

    for region_name, content in results.items():
        # Skip the 'Global' region if include_global is False
        if not include_global and region_name == 'Global':
            continue
        
        # Process each subdivision
        if isinstance(content, dict):
            for key, value in content.items():
                # If the value is a dictionary with performance, it's a result
                if isinstance(value, dict) and 'performance' in value:
                    process_result(value, r2_train_scores, r2_test_scores, cv_mean_scores)
                elif isinstance(value, dict):
                    # Otherwise, it's another level of subdivision
                    for sub_key, sub_value in value.items():
                        process_result(sub_value, r2_train_scores, r2_test_scores, cv_mean_scores)

    mean_r2_train = np.mean(r2_train_scores) if r2_train_scores else None
    mean_r2_test = np.mean(r2_test_scores) if r2_test_scores else None
    mean_cv_score = np.mean(cv_mean_scores) if cv_mean_scores else None

    # Print the results accordingly
    if mean_r2_train is not None and mean_r2_test is not None:
        print(f"Mean R2 for Training Data: {mean_r2_train:.2f}")
        print(f"Mean R2 for Testing Data: {mean_r2_test:.2f}")
    if mean_cv_score is not None:
        print(f"Mean CV Score: {mean_cv_score:.2f}")

    return mean_r2_train, mean_r2_test, mean_cv_score

def create_results_dataframe_regions(results, save_dir=None, filename=None):
    """
    Creates a DataFrame from results, handling regional data,
    and sorts it based on the R2 test score or CV score.

    Parameters:
    - results (dict): Dictionary containing the regression analysis results.
    - save_dir (str, optional): Directory to save the DataFrame. Defaults to None.
    - filename (str, optional): Filename to save the DataFrame. Defaults to None.

    Returns:
    - df (pd.DataFrame): DataFrame with the results sorted by performance.
    """
    data = []
    columns = ['Region', 'alpha', 'l1_ratio', 'n']
    
    for region_name, result in results.items():
        if 'regression_model' not in result:
            continue  # Skip if regression_model is not present

        best_params = result['regression_model'].get_params()
        alpha = round(best_params.get('alpha', None), 2)
        l1_ratio = round(best_params.get('l1_ratio', None), 2) if 'l1_ratio' in best_params else None
        n = result.get('n', None)

        performance = result['performance']
        row = [region_name, alpha, l1_ratio, n]

        # Append R2 Train if present
        if 'R2 Train' in performance:
            r2_train = round(performance['R2 Train'], 2)
            row.append(r2_train)
            if 'R2 Train' not in columns:
                columns.append('R2 Train')

        # Append R2 Test if present
        if 'R2 Test' in performance:
            r2_test = round(performance['R2 Test'], 2)
            row.append(r2_test)
            if 'R2 Test' not in columns:
                columns.append('R2 Test')

        # Append CV Mean Score if present
        if 'CV Mean Score' in performance:
            cv_mean_score = round(performance['CV Mean Score'], 2)
            row.append(cv_mean_score)
            if 'CV Mean Score' not in columns:
                columns.append('CV Mean Score')

        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Sort by 'CV Mean Score' if available, else by 'R2 Test'
    if 'CV Mean Score' in df.columns and not df['CV Mean Score'].isnull().all():
        df = df.sort_values(by='CV Mean Score', ascending=False).reset_index(drop=True)
    elif 'R2 Test' in df.columns:
        df = df.sort_values(by='R2 Test', ascending=False).reset_index(drop=True)

    # Save the DataFrame if save_dir and filename are provided
    if save_dir and filename:
        tables.save_tabel(df, save_dir, filename)
        
    return df

def create_results_dataframe_regions_subdivisions(results, save_dir=None, filename=None):
    """
    Creates a DataFrame from results, handling regional subdivision data,
    and sorts it based on the R2 test score or CV score.

    Parameters:
    - results (dict): Dictionary containing the regression analysis results.
    - save_dir (str, optional): Directory to save the DataFrame. Defaults to None.
    - filename (str, optional): Filename to save the DataFrame. Defaults to None.

    Returns:
    - df (pd.DataFrame): DataFrame with the results sorted by performance.
    """
    data = []
    columns = ['Region', 'alpha', 'l1_ratio', 'n']
    
    for region_name, subdivisions in results.items():
        for subdivision, result in subdivisions.items():
            if 'regression_model' not in result:
                continue  # Skip if regression_model is not present

            best_params = result['regression_model'].get_params()
            alpha = round(best_params.get('alpha', None), 2)
            l1_ratio = round(best_params.get('l1_ratio', None), 2) if 'l1_ratio' in best_params else None
            n = result.get('n', None)

            performance = result['performance']
            row = [f'{subdivision} - {region_name}', alpha, l1_ratio, n]

            # Append R2 Train if present
            if 'R2 Train' in performance:
                r2_train = round(performance['R2 Train'], 2)
                row.append(r2_train)
                if 'R2 Train' not in columns:
                    columns.append('R2 Train')

            # Append R2 Test if present
            if 'R2 Test' in performance:
                r2_test = round(performance['R2 Test'], 2)
                row.append(r2_test)
                if 'R2 Test' not in columns:
                    columns.append('R2 Test')

            # Append CV Mean Score if present
            if 'CV Mean Score' in performance:
                cv_mean_score = round(performance['CV Mean Score'], 2)
                row.append(cv_mean_score)
                if 'CV Mean Score' not in columns:
                    columns.append('CV Mean Score')

            data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Sort by 'CV Mean Score' if available, else by 'R2 Test'
    if 'CV Mean Score' in df.columns and not df['CV Mean Score'].isnull().all():
        df = df.sort_values(by='CV Mean Score', ascending=False).reset_index(drop=True)
    elif 'R2 Test' in df.columns:
        df = df.sort_values(by='R2 Test', ascending=False).reset_index(drop=True)

    # Save the DataFrame if save_dir and filename are provided
    if save_dir and filename:
        tables.save_tabel(df, save_dir, filename)

    return df

def plot_performance_scores_regions(results, score_type, plot_type='test', save_path=None):
    """
    Plots performance scores for regions based on the specified plot type (test or CV).

    Parameters:
    - results (dict): Dictionary containing the regression analysis results.
    - score_type (str): Type of score to plot (e.g., 'R2', 'MSE').
    - plot_type (str): Type of evaluation to plot ('test' or 'cv'). Defaults to 'test'.
    - save_path (str, optional): Path to save the plot. Defaults to None.

    Returns:
    - None
    """
    data = []
    evaluation_method = None  # Initialize to None
    
    for region_name, result in results.items():
        if 'regression_model' not in result:
            continue  # Skip if regression_model is not present
    
        performance = result['performance']
        if plot_type == 'test' and 'R2 Train' in performance and 'R2 Test' in performance:
            evaluation_method = 'test'
            r2_train = round(performance['R2 Train'], 2)
            r2_test = round(performance['R2 Test'], 2)
            row = [region_name, r2_train, r2_test]
        elif plot_type == 'cv' and 'CV Mean Score' in performance:
            evaluation_method = 'cv'
            cv_mean_score = round(performance['CV Mean Score'], 2)
            row = [region_name, cv_mean_score]
        else:
            continue  # Skip if the requested plot type score is not available
    
        data.append(row)
        
    # Convert data to DataFrame
    if plot_type == 'test':
        df = pd.DataFrame(data, columns=['Region', 'R2 Train', 'R2 Test'])
    else:
        df = pd.DataFrame(data, columns=['Region', 'CV Mean Score'])

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Bar width and spacing
    bar_width = 0.35
    region_names = df['Region'].unique()
    num_regions = len(region_names)
    
    # Define colors for bars
    train_color = (60/255, 145/255, 230/255)  # Blue for Train
    test_color = (255/255, 127/255, 14/255)   # Orange for Test
    cv_color = (44/255, 160/255, 44/255)      # Green for CV Mean Score

    # Set up bar positions
    indices = np.arange(num_regions)
    
    for i, region in enumerate(region_names):
        region_data = df[df['Region'] == region]
        if plot_type == 'test':
            r2_train = region_data['R2 Train'].values[0]
            r2_test = region_data['R2 Test'].values[0]
            ax.bar(i - bar_width/2, r2_train, bar_width, label='Train R2' if i == 0 else "", color=train_color)
            ax.bar(i + bar_width/2, r2_test, bar_width, label='Test R2' if i == 0 else "", color=test_color)
        else:
            cv_mean_score = region_data['CV Mean Score'].values[0]
            ax.bar(i, cv_mean_score, bar_width, label='CV Mean Score' if i == 0 else "", color=cv_color)

    # Add red dashed line at 0.7 for test plots (if applicable)
    ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Performance Threshold 0.7')

    # Set labels and title
    ax.set_xlabel('Regions', fontsize=14)
    ax.set_ylabel(score_type.upper(), fontsize=14)
    ax.set_xticks(indices)
    ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=14)
    
    # Increase font size of y-axis labels
    ax.tick_params(axis='y', labelsize=14)

    # Add legend and grid
    ax.legend(fontsize=14)  # Increase font size of legend
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path:
        filename = f'{plot_type}_{score_type}_regions_comparison.pdf'
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.show()


def plot_performance_scores_regions_subdivisions(results, score_type, plot_type='test', save_path=None):
    """
    Plots performance scores for regions and subdivisions based on the specified plot type (train, test, or CV).

    Parameters:
    - results (dict): Dictionary containing the regression analysis results.
    - score_type (str): Type of score to plot (e.g., 'R2', 'MSE').
    - plot_type (str): Type of evaluation to plot ('train', 'test', or 'cv'). Defaults to 'test'.
    - save_path (str, optional): Path to save the plot. Defaults to None.

    Returns:
    - None
    """
    data = []
    
    for region_name, subdivisions in results.items():
        for subdivision, result in subdivisions.items():
            if 'regression_model' not in result:
                continue  # Skip if regression_model is not present
    
            performance = result['performance']
            if plot_type == 'train' and 'R2 Train' in performance:
                r2_train = round(performance['R2 Train'], 2)
                row = [region_name, subdivision, r2_train]
            elif plot_type == 'test' and 'R2 Test' in performance:
                r2_test = round(performance['R2 Test'], 2)
                row = [region_name, subdivision, r2_test]
            elif plot_type == 'cv' and 'CV Mean Score' in performance:
                cv_mean_score = round(performance['CV Mean Score'], 2)
                row = [region_name, subdivision, cv_mean_score]
            else:
                continue  # Skip if the requested plot type score is not available
    
            data.append(row)

    # Determine the score column name based on the plot type
    score_column = {
        'train': 'R2 Train',
        'test': 'R2 Test',
        'cv': 'CV Mean Score'
    }.get(plot_type, 'R2 Test')  # Default to 'R2 Test' if plot_type is not valid

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=['Region', 'Subdivision', score_column])

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Bar width and spacing
    bar_width = 0.35
    region_names = df['Region'].unique()
    num_regions = len(region_names)
    
    # Define colors for subdivisions
    colors = {'Positive Historical BGWS': (60/255, 145/255, 230/255), 
              'Negative Historical BGWS': (34/255, 139/255, 34/255)}
    
    # Set up bar positions
    indices = np.arange(num_regions)
    
    # Initialize bar containers
    bars1 = []
    bars2 = []

    for i, region in enumerate(region_names):
        region_data = df[df['Region'] == region]
        pos_data = region_data[region_data['Subdivision'] == 'Positive Historical BGWS']
        neg_data = region_data[region_data['Subdivision'] == 'Negative Historical BGWS']
        
        if not pos_data.empty:
            pos_score = pos_data[score_column].values[0]
            bars1.append(ax.bar(i - bar_width/2, pos_score, bar_width, label='Positive Historical BGWS' if i == 0 else "", color=colors['Positive Historical BGWS']))
        
        if not neg_data.empty:
            neg_score = neg_data[score_column].values[0]
            bars2.append(ax.bar(i + bar_width/2, neg_score, bar_width, label='Negative Historical BGWS' if i == 0 else "", color=colors['Negative Historical BGWS']))

    # Add red dashed line at 0.7
    ax.axhline(y=0.66, color='red', linestyle='--', linewidth=2, label='Performance Threshold 0.66')

    # Set labels and title
    ax.set_ylabel(score_type.upper(), fontsize=14)

    # Define the regions to be marked in red
    regions_to_mark = ["E.North-America", "N.South-America", "E.Asia", 
                       "Central-Africa", "W.Southern-Africa", "E.Australia", "Global"]
    
    # Set x-tick labels and mark specific regions in red
    ax.set_xticks(indices)
    ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=18)
    
    # Modify the tick label colors
    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if label.get_text() in regions_to_mark:
            label.set_color('red')

#    ax.set_xticks(indices)
#    ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=14)
    
    # Increase font size of y-axis labels
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylabel(r'$R^2$', fontsize=20)


    # Add legend and grid
    ax.legend(fontsize=20)  # Increase font size of legend
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path:
        filename = f'{plot_type}_{score_type}_regions_subregions_comparison.pdf'
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.show()

def map_colors_to_display_names(base_colors, predictor_vars):
    """
    Maps custom colors to the formatted display variable names.

    Parameters:
    - base_colors (dict): Dictionary with original variable names as keys and color names as values.
    - predictor_vars (list): List of predictor variable names.

    Returns:
    - dict: Dictionary with formatted display names as keys and colors as values.
    """
    display_names = prepare_display_variables(predictor_vars)
    display_colors = {display_names[var]: base_colors[var] for var in predictor_vars if var in base_colors}
    return display_colors

def prepare_display_variables(variables):
    var_map = {
        'tas': ('T', '°C'),
        'vpd': ('VPD', 'hPa'),
        'gpp': ('GPP', r'\frac{\frac{gC}{m^2}}{day}'),  
        'pr': ('P', r'\frac{mm}{day}'),
        'mrro': ('R', r'\frac{mm}{day}'),
        'evspsbl': ('ET', r'\frac{mm}{day}'),
        'tran': ('Tran', r'\frac{mm}{day}'),
        'evapo': ('E', r'\frac{mm}{day}'),
        'lai': ('LAI', r'\frac{m^2}{m^2}'),
        'mrso': ('SM', '\%'),
        'rgtr': ('P/T', r'\frac{GPP}{T}'),
        'et_partitioning': ('EP', r'\frac{E-Tran}{ET}'),
        'gsl': ('GSL', 'months'),
        'RX5day': ('RX5day', 'mm'),
        'wue': ('WUE', r'\frac{GPP}{Tran}'),
        'bgws': ('BGWS', r'\frac{R-Tran}{P}')
    }
    display_variables = {}
    for var in variables:
        if var in var_map:
            abbreviation, units = var_map[var]
            # Enclose units in \left[ and \right] for automatic sizing
            display_variables[var] = f"${{\Delta\, \mathrm{{\it{{{abbreviation}}}}}}}$"
        else:
            print(f"Variable '{var}' not found in var_map.")
            display_variables[var] = var  # Or handle this case as appropriate
    return display_variables

def plot_region_permutation_importances(results, predictor_vars, importance_type='test',save_path=None):
    """
    Plots permutation importances for region-level results.
    
    Parameters:
    - results (dict): Dictionary containing the region-level regression analysis results.
    - predictor_vars (list): List of predictor variable names.
    - importance_type (str): The type of permutation importance to plot ('test' or 'train'). Defaults to 'test'.
    - save_path (optional): Path to save the figures. Defaults to None.
    
    Returns:
    - None
    """
    var_names = prepare_display_variables(predictor_vars)
    
    for region, data in results.items():
        # Check if 'variable_importance' exists in the data
        if 'variable_importance' not in data:
            print(f"Skipping {region} as it does not have variable importance data.")
            continue

        importance_key = f'Permutation Importance {"Testing" if importance_type == "test" else "Training"} Data'
        # Check if the specific importance_key exists
        if importance_key not in data['variable_importance']:
            print(f"Skipping {region} as it does not have {importance_key}.")
            continue

        var_importance_df = pd.DataFrame(data['variable_importance'][importance_key].T, columns=predictor_vars)
        var_importance_df.rename(columns=var_names, inplace=True)

        # Assuming you have a trained model and feature names from X_train
        coef_df = get_coefficients(data['regression_model'], predictor_vars, var_names)
            
        title = f'{region}'

        # Automatically find the correct performance metric based on importance_type
        performance_metric = next((key for key in data['performance'].keys() if importance_type.capitalize() in key), None)
        if not performance_metric:
            print(f"Skipping {region} as it does not have a matching performance metric for {importance_type}.")
            continue

        performance_value = round(data['performance'][performance_metric], 2)
        if 'R2' in performance_metric:
            performance_metric = 'R²'

        n = data['n']

        # Save the plot if a save path is provided
        if save_path:
            savepath = os.path.join(save_path, region)

        plot_permutation_importance(var_importance_df, coef_df, title, performance_value, n, importance_type, performance_metric, savepath)


def plot_region_subdivision_permutation_importances(results, predictor_vars, importance_type='test',save_path=None):
    """
    Plots permutation importances for region-subdivision results.
    
    Parameters:
    - results (dict): Dictionary containing the region-subdivision regression analysis results.
    - predictor_vars (list): List of predictor variable names.
    - importance_type (str): The type of permutation importance to plot ('test' or 'train'). Defaults to 'test'.
    - save_path (optional): Path to save the figures. Defaults to None.
    
    Returns:
    - None
    """
    var_names = prepare_display_variables(predictor_vars)
    
    for region, subregions in results.items():
        for subregion, data in subregions.items():
            importance_key = f'Permutation Importance {"Testing" if importance_type == "test" else "Training"} Data'
            if importance_key not in data['variable_importance']:
                print(f"Skipping {region} - {subregion} as it does not have {importance_key}.")
                continue

            var_importance_df = pd.DataFrame(data['variable_importance'][importance_key].T, columns=predictor_vars)
            var_importance_df.rename(columns=var_names, inplace=True)

            # Assuming you have a trained model and feature names from X_train
            coef_df = get_coefficients(data['regression_model'], predictor_vars, var_names)

            # Automatically find the correct performance metric based on importance_type
            performance_metric = next((key for key in data['performance'].keys() if importance_type.capitalize() in key), None)
            if not performance_metric:
                print(f"Skipping {region} - {subregion} as it does not have a matching performance metric for {importance_type}.")
                continue

            performance_value = round(data['performance']['CV Mean Score'], 2)
            if 'R2' in performance_metric:
                performance_metric = 'R²'

            n = data['n']

            # Save the plot if a save path is provided
            if save_path:
                savepath = os.path.join(save_path, region)
                plot_permutation_importance(var_importance_df, coef_df, subregion, performance_value, n, importance_type, performance_metric, savepath)
            else:
                plot_permutation_importance(var_importance_df, coef_df, subregion, performance_value, n, importance_type, performance_metric)


def permut_colormap(df_merged):
    # Create a colormap and normalization
    deep_blue = (60/255, 145/255, 230/255)
    deep_green = (34/255, 139/255, 34/255)

    coeff_min = df_merged['Coefficient'].min()
    coeff_max = df_merged['Coefficient'].max()
    coeff_abs_max = max(abs(coeff_min), abs(coeff_max))

    # Create the diverging colormap
    cmap = colmap.create_diverging_colormap(deep_blue, deep_green)
    
    # Create normalization based on min and max coefficient values
    norm = TwoSlopeNorm(vmin=-coeff_abs_max*0.7, vcenter=0, vmax=coeff_abs_max*0.7)

    return cmap, norm

def create_palette_from_colormap(bgws_cmap, bgws_cmap_norm, coefficients):
    # Generate colors for each coefficient
    return [bgws_cmap(bgws_cmap_norm(coeff)) for coeff in coefficients]

def plot_permutation_importance(df_importance, df_coefficients, title, performance_metric_value, n, importance_type, performance_metric, save_path=None):
    """
    Plot permutation importance with directional insight from regression coefficients.

    Parameters:
    - df_importance (pd.DataFrame): DataFrame containing the permutation importance data.
    - df_coefficients (pd.DataFrame): DataFrame containing regression coefficients for the features.
    - title (str): Title for the plot.
    - performance_metric_value (float): The performance metric value to display.
    - n (int): Number of data points to display.
    - performance_metric (str): The name of the performance metric to display.
    - importance_type (str): The type of permutation importance to plot ('test' or 'train'). Defaults to 'test'.
    - save_path (str, optional): Path to save the figure.

    Returns:
    - None
    """
    
    # Transform the df_importance from wide to long format for easier plotting
    df_long = df_importance.melt(value_vars=df_importance.columns, var_name='Feature', value_name='Importance')
    
    # Merge importance data with coefficients
    df_merged = pd.merge(df_long, df_coefficients, on='Feature', how='inner')

    df_merged = df_merged.sort_values(by='Importance', ascending=False)

    bgws_cmap, bgws_cmap_norm = permut_colormap(df_merged)

    # Generate the color palette based on coefficients
    palette = create_palette_from_colormap(bgws_cmap, bgws_cmap_norm, df_merged['Coefficient'].unique())

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='Importance', y='Feature', data=df_merged, ax=ax, palette=palette) 
        
    ax.axvline(0, color='grey', linestyle='--')
    
    # Set the title and performance metric text
    ax.text(0.95, 0.01, f'{performance_metric}: {performance_metric_value}\nn={n}',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=20)
    
    ax.set_xlabel('Decrease in R² Score', fontsize=24)
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=20) 
    
    # Add a colorbar without ticks
    sm = plt.cm.ScalarMappable(cmap=bgws_cmap, norm=bgws_cmap_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Regression Coefficient', fontsize=24)

    # Set custom ticks: the lower end and upper end of the color scale
    cbar.set_ticks([bgws_cmap_norm.vmin, 0, bgws_cmap_norm.vmax])
    
    # Set custom tick labels
    cbar.set_ticklabels(['\u2212', '0', '+'], fontsize=20)

    # Calculate the min and max importance values for each feature
    importance_min_max = df_merged.groupby('Feature')['Importance'].agg(['min', 'max']).reset_index()

    # Remove duplicate entries based on 'Feature' and keep the first occurrence
    df_unique = df_merged.drop_duplicates(subset='Feature')

    # Merge the min, max, and upper_25_threshold values back into df_unique for easier access
    df_unique = pd.merge(df_unique, importance_min_max, on='Feature')

    # Calculate the range of importance values
    importance_min = df_merged['Importance'].min()
    importance_max = df_merged['Importance'].max()
    importance_range = importance_max - importance_min
    
    # Define the threshold for the upper 25% of the range
    upper_25_threshold = importance_min + 0.75 * importance_range
    
    # Get the x-axis limits based on the box plot area only (ignoring the colorbar)
    x_min, x_max = ax.get_xlim()
    
    # Loop through each unique box and add regression coefficient text next to it
    for pos, row in enumerate(df_unique.itertuples()):
        # Clean the feature name by removing the dollar signs
        feature = row.Feature
        latex_feature = feature.replace("$", "")  # Clean LaTeX feature name
        importance = row.Importance
        importance_min = row.min
        coefficient = row.Coefficient
        
        # Create the coefficient text in the format 'β_Feature' with cleaned feature name
        coeff_text = f'$\\beta_{{{latex_feature}}} = {coefficient:.2f}$'
        
        # Set a small offset for placing the text within or near the box plot
        text_offset = 0.02 * (x_max - x_min)
        
        # Determine the position for the text based on whether importance is in the upper 25% of the range
        if importance >= upper_25_threshold:
            # If importance is in the upper 25% of the range, place the text on the left
            text_x = importance_min - text_offset
            ha = 'right'
        else:
            # Otherwise, place the text on the right
            text_x = importance + text_offset
            ha = 'left'
    
        # Place the text inside the plot, ensuring it doesn't overwrite the plot elements
        ax.text(text_x, pos, coeff_text, va='center', ha=ha, fontsize=22, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        filename = f'{title}_{importance_type}.pdf'
        filepath = os.path.join(save_path, filename)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved at: {filepath}")
    
    plt.show()


def get_coefficients(best_model, feature_names, var_names):
    """
    Extract regression coefficients from a trained model, round them to two decimals,
    and align them with LaTeX formatted feature names, ensuring that zero values do not display a negative sign.

    Parameters:
    - best_model: Trained regression model.
    - feature_names: List of feature names used in the model training.
    - var_names: Dictionary mapping simple feature names to LaTeX formatted names.

    Returns:
    - DataFrame containing feature names and their corresponding coefficients,
      rounded to two decimal places and adjusted for zero values.
    """
    try:
        # Extract coefficients
        coefficients = best_model.coef_

        # Round coefficients to two decimals and adjust for zero values
        rounded_coefficients = [round(coef, 2) if round(coef, 2) != 0 else 0 for coef in coefficients]
        
        # Create a DataFrame mapping feature names to rounded coefficients
        coef_df = pd.DataFrame(data={'Feature': feature_names, 'Coefficient': rounded_coefficients})

        # Rename the 'Feature' column values according to the LaTeX formatted names
        coef_df['Feature'] = coef_df['Feature'].map(var_names)
        
        return coef_df
    
    except AttributeError:
        # If the model does not have coefficients (e.g., tree-based models)
        return "This model does not support extraction of coefficients."

def plot_region_shap(
    results,
    predictor_vars,
    importance_type='test',
    plot_type = 'dot',
    save_path=None
):
    """
    Plots permutation importances for region-subdivision results.
    
    Parameters:
    - results (dict): Dictionary containing the region-subdivision regression analysis results.
    - predictor_vars (list): List of predictor variable names.
    - importance_type (str): The type of permutation importance to plot ('test' or 'train'). Defaults to 'test'.
    - plot_type (str): Style of plot ('dot', 'violine', or 'bar'). Default is 'dot'.
    - save_path (optional): Path to save the figures. Defaults to None.
    
    Returns:
    - None
    """

    var_names = list(prepare_display_variables(predictor_vars).values())

    for region, data in results.items():
        importance_key = f'SHAP Values {"Testing" if importance_type == "test" else "Training"} Data'
        if importance_key not in data['variable_importance']:
            print(f"Skipping {region} as it does not have {importance_key}.")
            continue

        shap_values = data['variable_importance'][importance_key].values
        feature_values = data['variable_importance'][importance_key].data
        
        title = f'{region}'

        # Automatically find the correct performance metric based on importance_type
        performance_metric = next((key for key in data['performance'].keys() if importance_type.capitalize() in key), None)
        if not performance_metric:
            print(f"Skipping {region} as it does not have a matching performance metric for {importance_type}.")
            continue

        performance_value = round(data['performance'][performance_metric], 2)
        if 'R2' in performance_metric:
            performance_metric = 'R²'

        n = data['n']

         # Save the plot if a save path is provided
        if save_path:
            savepath = os.path.join(save_path, region)

        # Call the function to plot the SHAP values
        plot_shap_values(shap_values, feature_values, title, importance_type, var_names, len(predictor_vars), plot_type, performance_metric, performance_value, n, savepath)

def plot_region_subdivision_shap(
    results,
    predictor_vars,
    importance_type='test',
    plot_type = 'dot',
    save_path=None
):
    """
    Plots permutation importances for region-subdivision results.
    
    Parameters:
    - results (dict): Dictionary containing the region-subdivision regression analysis results.
    - predictor_vars (list): List of predictor variable names.
    - importance_type (str): The type of permutation importance to plot ('test' or 'train'). Defaults to 'test'.
    - plot_type (str): Style of plot ('dot', 'violine', or 'bar'). Default is 'dot'.
    - save_path (optional): Path to save the figures. Defaults to None.
    
    Returns:
    - None
    """

    var_names = list(prepare_display_variables(predictor_vars).values())

    for region, subregions in results.items():
        for subregion, data in subregions.items():
            importance_key = f'SHAP Values {"Testing" if importance_type == "test" else "Training"} Data'
            if importance_key not in data['variable_importance']:
                print(f"Skipping {region} - {subregion} as it does not have {importance_key}.")
                continue

            shap_values = data['variable_importance'][importance_key].values
            feature_values = data['variable_importance'][importance_key].data
            
            title = f'{region} - {subregion}'

            # Automatically find the correct performance metric based on importance_type
            performance_metric = next((key for key in data['performance'].keys() if importance_type.capitalize() in key), None)
            if not performance_metric:
                print(f"Skipping {region} - {subregion} as it does not have a matching performance metric for {importance_type}.")
                continue

            performance_value = round(data['performance']['CV Mean Score'], 2)
            if 'R2' in performance_metric:
                performance_metric = 'R²'

            n = data['n']

            # Save the plot if a save path is provided
            if save_path:
                savepath = os.path.join(save_path, region)

            # Call the function to plot the SHAP values
            plot_shap_values(shap_values, feature_values, title, importance_type, var_names, len(predictor_vars), plot_type, performance_metric, performance_value, n, savepath)

def plot_shap_values(shap_values, feature_values, title, importance_type, feature_names, max_display=10, plot_type='bar', performance_metric="", performance_value="", n="", save_path=None):
    """
    Plot SHAP values for model explanations, adding performance metric and sample size as annotations.

    Parameters:
    - shap_values: SHAP values calculated for the model.
    - feature_values: DataFrame of the input features corresponding to the SHAP values.
    - title (str): Title for the plot.
    - importance_type (str): The type of permutation importance to plot ('test' or 'train'). Defaults to 'test'.
    - feature_names: List of feature names.
    - max_display (int): Maximum number of top features to display.
    - plot_type (str): Type of plot ('bar', 'dot', 'violin', etc.).
    - performance_metric (str): Metric used to measure model performance.
    - performance_value (str): Value of the performance metric.
    - n (int): Sample size.
    - save_path (str, optional): Path to save the figure.

    Returns:
    - None
    """
    
    # Convert feature data to DataFrame if not already one, ensuring alignment with feature_names
    if not isinstance(feature_values, pd.DataFrame):
        features = pd.DataFrame(feature_values, columns=feature_names)
        
    # Ensure the use of an active figure with subplots for direct annotation
    fig, ax = plt.subplots()
    if plot_type == 'bar':
        shap.summary_plot(shap_values, features, plot_type="bar", max_display=max_display, feature_names=feature_names, show=False, sort=True)
    elif plot_type == 'dot':
        shap.summary_plot(shap_values, features, plot_type="dot", max_display=max_display, feature_names=feature_names, show=False, sort=True)
    elif plot_type == 'violin':
        shap.summary_plot(shap_values, features, plot_type="violin", max_display=max_display, feature_names=feature_names, show=False, sort=True)
    else:
        raise ValueError("Unsupported plot type. Use 'bar', 'dot', or 'violin'.")

    # Add text for performance and sample size directly to the current axes
    ax.text(0.95, 0.01, f"{performance_metric}: {performance_value}\nSample size: {n}",
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=12)

    # Save the plot if a save path is provided
    if save_path:
        safe_title = title.replace(' ', '_').replace('-', '_')
        filename = f'{safe_title}_{importance_type}_{plot_type}.pdf'
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved at: {filepath}")
    
    plt.show()