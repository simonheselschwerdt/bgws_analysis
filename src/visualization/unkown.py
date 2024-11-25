def plot_statistic(ds_dict, variable, cbar_min=0, cbar_max=0.75, cmap='viridis', log_scale=False, add_regression=True, show_plot=True, save_fig=False, file_format='png', smooth_window=1):
    """
    Plots a map of the specified statistic of the given variable for each dataset in the dictionary.

    Args:
        ds_dict (dict): A dictionary of xarray datasets, where each key is the name of the dataset
            and each value is the dataset itself.
        variable (str): The name of the variable to plot.
        cbar_min (float): A value to set vmin by multiplying with the variables minimum value across the dataset. Default is 0.
        cbar_max (float): A value to set vmax by multiplying with the variables maximum value across the dataset. Default is 0.75.
        cmap (str): The name of the colormap to use for the plot. Default is 'viridis'.
        log_scale (bool): If True, plot the data on a log scale. Default is False.
        add_regression (bool): If Ture, compute regression and plot it. Default is True.
        show_plot (bool): If True, display the plot in addition to saving it. Default is True.
        save_fig (bool): If True, save the figure to a file. Default is False.
        file_format (str): The format of the saved figure. Default is 'png'.
        smooth_window (int): Window size for rolling mean smoothing. Default is 1 (no smoothing).

    Returns:
        str: The file path where the figure was saved.
    """

    # Check the validity of input arguments
    if not isinstance(ds_dict, dict):
        raise TypeError("ds_dict must be a dictionary of xarray datasets.")
    if not all(isinstance(ds, xr.Dataset) for ds in ds_dict.values()):
        raise TypeError("All values in ds_dict must be xarray datasets.")
    if not isinstance(variable, str):
        raise TypeError('variable must be a string.')
    if smooth_window < 1:
        raise ValueError("smooth_window must be a positive integer.")
   
    # Dictionary to store plot titles for each statistic
    titles = {"mean": "Mean", "std": "Standard deviation", "min": "Minimum", "max": "Maximum", "var": "Variability", "median": "Median", "time": "Time", "space": "Space"}
    
    # Calculate vmin and vmax
    temp_dim_ds = xr.concat([ds[variable] for ds in ds_dict.values() if variable in ds], dim='temp_dim', coords='minimal')
    vmin = round(float(temp_dim_ds.min())) * cbar_min
    vmax = round(float(temp_dim_ds.max()), -int(math.floor(math.log10(abs(float(temp_dim_ds.max())))))) * cbar_max
    
    # Get the statistic of the datasets
    statistic = str(ds_dict[list(ds_dict.keys())[0]].attrs['statistic'])
    
    if ds_dict[list(ds_dict.keys())[0]].statistic_dimension == 'time':
       # Create a figure with subplots for each dataset
        n_datasets_with_var = sum([1 for ds in ds_dict.values() if variable in ds])
        n_rows = (n_datasets_with_var + 1) // 2
        n_cols = 1 if n_datasets_with_var == 1 else 2

        with plt.style.context('bmh'):
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=[12 * n_cols, 6 * n_rows])

        # Ensure axes is a 2D array for consistent indexing
        if n_datasets_with_var == 1:
            axes = np.array([[axes]])
        else:
            axes = axes.reshape((n_rows, n_cols))
        
        subplot_counter = 0

        # Loop over datasets and plot the requested statistic
        for i, (name, ds) in enumerate(ds_dict.items()):
            if variable not in ds:
                print(f"Variable '{variable}' not found in dataset '{name}', skipping.")
                continue


            # plot the variable in subplot
            row = subplot_counter // 2
            col = subplot_counter % 2
            subplot_counter += 1

            ax = axes[row, col]
            data_to_plot = ds[variable] #.where(ds[variable] >= 0)
            if log_scale:
                data_to_plot = np.log10(data_to_plot)
            data_to_plot.plot(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, extend='max')
            ax.set_title(name)
            ax.set_xlabel('longitude')  # set x-axis labels
            ax.set_ylabel('latitude')  # set y-axis labels

        # Remove unused axes for odd number of datasets
        if n_datasets_with_var % 2 == 1:
            fig.delaxes(axes[-1, -1])


    elif ds_dict[list(ds_dict.keys())[0]].statistic_dimension == 'space':
        
        # Create consistent time coordinate using the first time coordinate for all following models   
        ds_names = list(ds_dict.keys())

        for i in range(len(ds_names) - 1):
            current_ds = ds_dict[ds_names[i]]
            next_ds = ds_dict[ds_names[i + 1]]

            if (current_ds['time'] != next_ds['time']).all():
                next_ds['time'] = current_ds.time
        
        fig, ax = plt.subplots(figsize=(30, 15))
        
        # Initialize a list to store the DataArrays from each dataset
        data_list = []

        # Loop over datasets and plot the requested statistic in a single figure
        for i, (name, ds) in enumerate(ds_dict.items()):
            if variable not in ds:
                print(f"Variable '{variable}' not found in dataset '{name}', skipping.")
                continue


            data_to_plot = ds[variable].squeeze()
            
            # Apply log-scale
            if log_scale:
                data_to_plot = np.log10(data_to_plot)

            # Apply smoothing
            if smooth_window > 1:
                data_to_plot = data_to_plot.rolling(time=smooth_window, center=True).mean()
            
            data_list.append(data_to_plot)  # Append the transformed data to the list
                
            # Plot the data and get the color of the line
            data_lines = data_to_plot.plot.line(x='time', ax=ax, label=None)
            data_color = data_lines[0].get_color()
            
            if add_regression:
                # Create regression model
                trend = data_to_plot.polyfit(dim='time', deg=1)

                # Fit regression model to data and add it to plot 
                regression_line = xr.polyval(ds['time'], trend).squeeze()
                regression_line['polyfit_coefficients'].plot(x='time', ax=ax, color=data_color)
                
                # Calculate percentage of change
                first_value = regression_line['polyfit_coefficients'].isel(time=0).item()
                last_value = regression_line['polyfit_coefficients'].isel(time=-1).item()
                percentage_change = ((last_value - first_value) / first_value) * 100

                # Add the percentage change to the legend label
                ax.plot([], [], color=data_color, label=f"{name} ({percentage_change:.2f}%)")


        # Set the x and y axis labels
        ax.set_xlabel('Year')
        if log_scale:
            ax.set_ylabel(f"{ds_dict[list(ds_dict.keys())[0]][variable].long_name} [{ds_dict[list(ds_dict.keys())[0]][variable].units}] - log-scale")
        else:
            ax.set_ylabel(f"{ds_dict[list(ds_dict.keys())[0]][variable].long_name} [{ds_dict[list(ds_dict.keys())[0]][variable].units}]")

        # Calculate the ensemble mean
        ensemble_mean = xr.concat(data_list, dim='dataset').mean(dim='dataset')
        
        if add_regression:

                # Create regression model
                trend = ensemble_mean.polyfit(dim='time', deg=1)

                # Fit regression model to data and add it to plot 
                regression_line = xr.polyval(ds['time'], trend).squeeze()
                regression_line['polyfit_coefficients'].plot(x='time', ax=ax, color=data_color)
                
                # Calculate percentage of change
                first_value = regression_line['polyfit_coefficients'].isel(time=0).item()
                last_value = regression_line['polyfit_coefficients'].isel(time=-1).item()
                ens_percentage_change = ((last_value - first_value) / first_value) * 100
            
        # Plot the ensemble mean with a different line style and/or color
        ensemble_mean.plot.line(x='time', ax=ax, linestyle='--', color='black', label=f"Ensemble mean ({ens_percentage_change:.2f}%)") 

        # Add a legend
        ax.legend()
        ax.grid(True)

    else:
        raise ValueError(f"Invalid value '{ds_dict[list(ds_dict.keys())[0]].statistic_dimension}' for 'statistic_dimension'. Allowed values are 'time' and 'space'.")

    # Set figure title with first and last year of dataset
    if smooth_window>1:
        fig.suptitle(f"{titles[ds_dict[list(ds_dict.keys())[0]].statistic_dimension]} {titles[statistic]} of {ds_dict[list(ds_dict.keys())[0]][variable].attrs['long_name']} ({ds_dict[list(ds_dict.keys())[0]].attrs['period'][0]} - {ds_dict[list(ds_dict.keys())[0]].attrs['period'][1]}) ({smooth_window}-year running mean)", fontsize=20, y=1.0)
    else:
        fig.suptitle(f"{titles[ds_dict[list(ds_dict.keys())[0]].statistic_dimension]} {titles[statistic]} of {ds_dict[list(ds_dict.keys())[0]][variable].attrs['long_name']} ({ds_dict[list(ds_dict.keys())[0]].attrs['period'][0]} - {ds_dict[list(ds_dict.keys())[0]].attrs['period'][1]})", fontsize=20, y=1.0)

    # adjust layout and save plot to file
    fig.tight_layout()

    if show_plot:
        plt.show()


    if save_fig:
        savepath = os.path.join('../..', 'results', 'CMIP6', ds.experiment_id, ds.statistic_dimension, ds.statistic)
        os.makedirs(savepath, exist_ok=True)
        
        if smooth_window>1:
            if log_scale:
                filename = f'{ds.statistic_dimension}.{ds.statistic}.{variable}.{ds.experiment_id}.{smooth_window}-year_running_mean.log_scale.{file_format}'
            else:
                filename = f'{ds.statistic_dimension}.{ds.statistic}.{variable}.{ds.experiment_id}.{smooth_window}-year_running_mean.{file_format}'
        else:
            if log_scale:
                filename = f'{ds.statistic_dimension}.{ds.statistic}.{variable}.{ds.experiment_id}.log_scale.{file_format}'
            else:
                filename = f'_{ds.statistic_dimension}.{ds.statistic}.{variable}.{ds.experiment_id}.{file_format}'
              
        
        filepath = os.path.join(savepath, filename)
        fig.savefig(filepath, dpi=300)
    else:
        filepath = 'Figure not saved. If you want to save the figure add save_fig=True to the function call'

    return filepath