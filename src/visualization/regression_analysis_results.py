"""
Visualization of Regression Analysis Results
--------------------------------------------
This script provides functionality to process and visualize the results of 
regression analysis, including model performance metrics, variable importance, 
permutation importance, and SHAP value summaries.

Functions:
- process_results: Extracts key components (performance metrics, variable importance, sample size) from regression results.
- create_results_dataframe: Creates a summary DataFrame of regression results.
- map_colors_to_display_names: Maps specific colors to formatted display variable names.
- prepare_display_variables: Prepares LaTeX-formatted variable names for visualization.
- permut_colormap: Generates a diverging colormap for permutation importance plots.
- create_palette_from_colormap: Creates a custom palette from a colormap based on coefficients.
- get_coefficients: Extracts coefficients or feature importances from regression models.
- get_model_coefficients_or_importances: Retrieves either coefficients or feature importances for different models.
- plot_permutation_importance: Visualizes permutation importance with regression coefficient overlay.
- plot_shap_summary: Creates SHAP summary plots with mean SHAP values and model performance annotations.

Usage: Import this module and call the appropriate function for processing or visualizing regression results.

Author: [Simon P. Heselschwerdt]
Date: [2025-06-02]
Dependencies:pandas, matplotlib, seaborn, numpy, shap
"""
# ========== Imports ==========

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import shap

# ========== Import Custom Functions ==========

import colormaps_and_utilities as col_uti

# ========== Functions ==========

def process_results(results):
    """
    Extracts key components of regression results.

    Parameters:
    - results (dict): Results dictionary containing keys such as 'performance',
                      'variable_importance', and 'n'.

    Returns:
    - dict: Processed components of the results:
        - performance_metrics: Dictionary with R2 scores for train and test sets.
        - variable_importance: Dictionary with permutation importance arrays.
        - n: Number of data points.
    """
    if not isinstance(results, dict):
        raise ValueError("Invalid results format. Expected a dictionary.")

    return {
        "performance_metrics": results.get("performance", {}),
        "variable_importance": results.get("variable_importance", {}),
        "n": results.get("n", None),
    }


def create_results_dataframe(results, predictor_vars):
    """
    Creates a DataFrame summarizing regression results.

    Parameters:
    - results (dict): Dictionary containing regression results.
    - predictor_vars (list): List of predictor variable names.

    Returns:
    - pd.DataFrame: DataFrame summarizing the results.
    """
    processed = process_results(results)
    performance = processed["performance_metrics"]
    variable_importance = processed["variable_importance"]

    # Extract performance metrics
    r2_train = performance.get("R2 Train")
    r2_test = performance.get("R2 Test")
    n = processed["n"]

    # Summarize variable importance
    importance_test = variable_importance.get("Permutation Importance Testing Data", [])
    importance_train = variable_importance.get("Permutation Importance Training Data", [])

    data = {
        "R2 Train": [r2_train],
        "R2 Test": [r2_test],
        "N": [n],
        **{
            f"Importance {var} Test": [np.mean(importance_test[idx]) if idx < len(importance_test) else np.nan]
            for idx, var in enumerate(predictor_vars)
        },
        **{
            f"Importance {var} Train": [np.mean(importance_train[idx]) if idx < len(importance_train) else np.nan]
            for idx, var in enumerate(predictor_vars)
        },
    }

    return pd.DataFrame(data)

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
        'pr': ('P', r'\frac{mm}{day}'),
        'mrro': ('R', r'\frac{mm}{day}'),
        'tran': ('Tran', r'\frac{mm}{day}'),
        'bgws': ('BGWS', r'%'),
        'RX5day': ('RX5day', 'mm'),
        'evapo': ('E', r'\frac{mm}{day}'),
        'evspsbl': ('ET', r'\frac{mm}{day}'),
        'vpd': ('VPD', 'hPa'),
        'mrso': ('SM', '\%'),
        'lai': ('LAI', r'\frac{m^2}{m^2}'),
        'gpp': ('GPP', r'\frac{\frac{gC}{m^2}}{day}'),  
        'wue': ('WUE', r'\frac{GPP}{Tran}')
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

def permut_colormap(df_merged):
    # Create a colormap and normalization
    deep_blue = (60/255, 145/255, 230/255)
    deep_green = (34/255, 139/255, 34/255)

    coeff_min = df_merged['Coefficient'].min()
    coeff_max = df_merged['Coefficient'].max()
    coeff_abs_max = max(abs(coeff_min), abs(coeff_max))

    # Create the diverging colormap
    cmap = col_uti.create_diverging_colormap(deep_blue, deep_green)
    
    # Create normalization based on min and max coefficient values
    norm = TwoSlopeNorm(vmin=-coeff_abs_max*0.7, vcenter=0, vmax=coeff_abs_max*0.7)

    return cmap, norm

def create_palette_from_colormap(bgws_cmap, bgws_cmap_norm, coefficients):
    # Generate colors for each coefficient
    return [bgws_cmap(bgws_cmap_norm(coeff)) for coeff in coefficients]

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
    if hasattr(best_model, 'coef_'):
        # Extract coefficients
        coefficients = best_model.coef_

        # Round coefficients to two decimals and adjust for zero values
        rounded_coefficients = [round(coef, 2) if round(coef, 2) != 0 else 0 for coef in coefficients]
        
        # Create a DataFrame mapping feature names to rounded coefficients
        coef_df = pd.DataFrame(data={'Feature': feature_names, 'Coefficient': rounded_coefficients})

        # Rename the 'Feature' column values according to the LaTeX formatted names
        coef_df['Feature'] = coef_df['Feature'].map(var_names)

        return coef_df
    
    elif hasattr(best_model, 'feature_importances_'):
        
        # For tree-based models like RandomForestRegressor
        importances = best_model.feature_importances_

        rounded_importances = [round(imp, 2) if round(imp, 2) != 0 else 0 for imp in importances]

        # Create a DataFrame mapping feature names to rounded coefficients
        imp_df = pd.DataFrame(data={'Feature': feature_names, 'Coefficient': rounded_importances})

        imp_df['Feature'] = imp_df['Feature'].map(var_names)

        return imp_df

def get_model_coefficients_or_importances(model, predictor_vars):
    """
    Retrieve coefficients or feature importances from a trained model.

    Parameters:
    - model: The trained regression model (e.g., ElasticNet, RandomForestRegressor).
    - predictor_vars (list): List of predictor variable names.

    Returns:
    - pd.DataFrame: A DataFrame containing features and their corresponding coefficients or importances.
    """
    if hasattr(model, 'coef_'):
        # For linear models like ElasticNet, Ridge, Lasso, etc.
        coefficients = model.coef_
        return pd.DataFrame({
            'Feature': predictor_vars,
            'Coefficient': coefficients
        }).sort_values(by='Coefficient', ascending=False)
    
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models like RandomForestRegressor
        importances = model.feature_importances_
        return pd.DataFrame({
            'Feature': predictor_vars,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
    
    else:
        raise AttributeError("The model does not support coefficients or feature importances.")

def plot_permutation_importance(results, predictor_vars, regime, importance_type='test', save_path=None):
    """
    Plot permutation importance with directional insight from regression coefficients.
    
    Parameters:
    - results (dict): Dictionary containing the region-level regression analysis results.
    - predictor_vars (list): List of predictor variable names.
    - regime: Description or name of the regime for labeling the plot.
    - importance_type (str): The type of permutation importance to plot ('test' or 'train'). Defaults to 'test'.
    - save_path (optional): Path to save the figures. Defaults to None.
    
    Returns:
    - None
    """
    # Prepare LaTeX display names for the predictor variables
    var_names = prepare_display_variables(predictor_vars)

    # Determine the key for the permutation importance data (either 'Testing' or 'Training')
    importance_key = f'Permutation Importance {"Testing" if importance_type == "test" else "Training"} Data'

    # Create a DataFrame for permutation importance and rename columns for display
    var_importance_df = pd.DataFrame(results['variable_importance'][importance_key].T, columns=predictor_vars)
    var_importance_df.rename(columns=var_names, inplace=True)

    # Extract regression coefficients and map them to the LaTeX display variable names
    coef_df = get_coefficients(results['regression_model'], predictor_vars, var_names)
            
    # Determine the performance metric and round its value
    performance_metric = next((key for key in results['performance'].keys() if importance_type.capitalize() in key), None)
    performance_value = round(results['performance'][performance_metric], 2)
    performance_metric = 'R²'

    # Extract the number of data points used in the regression analysis
    n = results['n']
    
    # Convert importance DataFrame to long format, merge with coefficients and sort by importance values in descending order
    df_long = var_importance_df.melt(value_vars=var_importance_df.columns, var_name='Feature', value_name='Importance')
    df_merged = pd.merge(df_long, coef_df, on='Feature', how='inner')

    # Calculate mean importance for each feature
    mean_importance = df_merged.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
    
    # Order the features by mean importance
    df_merged['Feature'] = pd.Categorical(
        df_merged['Feature'],
        categories=mean_importance.index,
        ordered=True
    )
    
    # Sort the DataFrame based on the ordered Feature column
    df_merged = df_merged.sort_values(by=['Feature', 'Importance'], ascending=[True, False])

    # Generate colormap and normalization for coefficients
    bgws_cmap, bgws_cmap_norm = permut_colormap(df_merged)
    palette = create_palette_from_colormap(bgws_cmap, bgws_cmap_norm, df_merged['Coefficient'].unique())

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='Importance', y='Feature', data=df_merged, ax=ax, palette=palette, hue='Feature', legend=False)

    # Add a vertical line at x=0 for reference
    ax.axvline(0, color='grey', linestyle='--')
    
    # Add text displaying the performance metric and the number of data points
    ax.text(0.95, 0.01, f'{performance_metric}: {performance_value}\nn={n}',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=24)

    # Customize axes
    ax.set_xlabel('Decrease in R² Score', fontsize=28)
    ax.set_ylabel('') # No label for y-axis
    ax.tick_params(axis='both', which='major', labelsize=24) 
    
    # Add colorbar for coefficients
    sm = plt.cm.ScalarMappable(cmap=bgws_cmap, norm=bgws_cmap_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Regression Coefficient', fontsize=28)
    cbar.set_ticks([bgws_cmap_norm.vmin, 0, bgws_cmap_norm.vmax])
    cbar.set_ticklabels(['\u2212', '0', '+'], fontsize=24)

    # Calculate minimum and maximum
    importance_min_max = df_merged.groupby('Feature', observed=False)['Importance'].agg(['min', 'max']).reset_index()

    #importance_min_max = df_merged.groupby('Feature')['Importance'].agg(['min', 'max']).reset_index()

    # Remove duplicate entries based on 'Feature' and merge the min, max, and upper_25_threshold values back into df_unique for easier access
    df_unique = df_merged.drop_duplicates(subset='Feature')
    df_unique = pd.merge(df_unique, importance_min_max, on='Feature')

    # Calculate the range of importance values
    importance_min = df_merged['Importance'].min()
    importance_max = df_merged['Importance'].max()
    importance_range = importance_max - importance_min
    
    # Define the threshold for the upper 25% of the range
    upper_25_threshold = importance_min + 0.75 * importance_range
    
    # Get the x-axis limits based on the box plot area only (ignoring the colorbar)
    x_min, x_max = ax.get_xlim()
    
    # Annotate regression coefficients next to their respective boxplots
    for pos, row in enumerate(df_unique.itertuples()):
        # Clean the feature name by removing the dollar signs
        feature = row.Feature
        latex_feature = feature.replace("$", "")  # Remove LaTeX formatting
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

    # Adjust plot layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        filename = f'{regime}_{importance_type}.pdf'
        filepath = os.path.join(save_path, filename)
        os.makedirs(save_path, exist_ok=True) 
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved at: {filepath}")
  
    plt.show()

def plot_shap_summary(results, X, predictor_vars, test_train, save_path=None, title=None):
    """
    Plot SHAP summary plot for the SHAP values of a model, with mean SHAP values (with sign) included, 
    and displays R² and n in the bottom-right corner.

    Parameters:
    - shap_values (np.array): SHAP values for the dataset (either training or test set).
    - X (pd.DataFrame): The corresponding feature dataset used to compute SHAP values.
    - predictor_vars (list): List of predictor variable names.
    - r2 (float, optional): R² value to display in the plot. Defaults to None.
    - n (int, optional): Number of data points to display in the plot. Defaults to None.
    - save_path (str, optional): Path to save the plot. Defaults to None.
    - title (str, optional): Title of the plot. Defaults to None.

    Returns:
    - None: Displays or saves the SHAP summary plot.
    """
    shap_values = results[f'shap_values_{test_train.lower()}']
    
    # Ensure SHAP values and input dataset align
    assert shap_values.shape[1] == len(predictor_vars), "SHAP values must align with predictor variables."

    # Compute mean absolute SHAP values
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Prepare variable names with a Delta (Δ) symbol
    var_names = {
        'RX5day': r'$\Delta$RX5day',
        'pr': r'$\Delta$P',
        'lai': r'$\Delta$LAI',
        'wue': r'$\Delta$WUE',
        'vpd': r'$\Delta$VPD',
        'mrso': r'$\Delta$SM'
    }

    # Sort features by mean absolute SHAP value
    sorted_indices = np.argsort(mean_abs_shap_values)[::-1]
    sorted_vars = [predictor_vars[i] for i in sorted_indices]
    sorted_mean_abs = mean_abs_shap_values[sorted_indices]
    
    # Map sorted variable names to their prepared names
    prepared_sorted_vars = [var_names.get(var, var) for var in sorted_vars]
    
    # Prepare display variable names with mean SHAP values
    display_vars_with_mean = {
        var: f"{prepared_var} ({mean:.2f})"
        for var, prepared_var, mean in zip(sorted_vars, prepared_sorted_vars, sorted_mean_abs)
    }

    # Rename columns in the input data for display
    X_display = X.copy()
    X_display.columns = [display_vars_with_mean.get(col, col) for col in X.columns]

    # Set font for Matplotlib
    plt.rcParams['font.family'] = 'Nimbus Sans'

    # Create SHAP summary plot
    plt.figure(figsize=(10, 6))
    ax = shap.summary_plot(shap_values, X_display, plot_type="dot", show=False)

    from decimal import Decimal, ROUND_HALF_UP
    # Correctly round r2 using Decimal
    r2 = float(Decimal(results['performance'][f"R2 {test_train}"]).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    
    # Extract the number of data points used in the regression analysis
    n = results['n']
    # Add R² and n to the bottom-right corner of the plot
    r2_text = f"R²: {r2:.2f}\n" if r2 is not None else ""
    n_text = f"n={n}" if n is not None else ""
    annotation = r2_text + n_text
    plt.text(
        0.95, 0.01, annotation,
        verticalalignment='bottom', horizontalalignment='right',
        transform=plt.gca().transAxes,
        color='black', fontsize=12
    )

    # Save the plot if a path is specified
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, f"{title or 'shap_summary'}.pdf"),
            dpi=300,
            bbox_inches="tight"
        )

    # Show the plot
    plt.show()