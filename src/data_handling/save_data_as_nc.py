# Save data as netCDF

import dask
import os
import xarray as xr

def save_files(ds_dict, save_path):
    """
    Save files as netCDF.

    Parameters:
    - ds_dict: Dictionary of xarray datasets or a single dataset.
    - save_path: Base path where data will be saved. Each variable will be saved in a separate directory under {model_name}.nc.

    Returns:
    - None
    """
    for model, ds in ds_dict.items():
        for var in ds:
            # Variable to keep
            variable_to_keep = var
            dimensions_to_keep = {'time', 'lat', 'lon'}
            coordinates_to_keep = {'time', 'lat', 'lon'}

            # Add depth to dimensions and coordinates if it exists
            if any('depth' in ds[var].dims for var in ds.variables):
                dimensions_to_keep.add('depth')
                coordinates_to_keep.add('depth')

            # Create a new dataset with only the desired variable
            ds_var = ds[[variable_to_keep]]

            # Ensure slicing is applied only to valid dimensions
            valid_dims = dimensions_to_keep.intersection(ds_var.dims)
            ds_var = ds_var.isel({dim: slice(None) for dim in valid_dims})

            # Set the desired coordinates, ensuring compatibility
            coords_to_set = set(ds_var.variables).intersection(coordinates_to_keep)
            ds_var = ds_var.set_coords(list(coords_to_set))

            # Define variable and file name and final path
            var_dir = os.path.join(save_path, var)
            file_name = f'{model}.nc'
            final_path = os.path.join(var_dir, file_name)

            # Check if path exists and create path if not
            os.makedirs(var_dir, exist_ok=True)

            # Remove existing file corresponding to the model
            if os.path.exists(final_path):
                os.remove(final_path)
                print(f"File {final_path} removed")

            # Save to netcdf file
            with dask.config.set(scheduler='threads'):
                try:
                    ds_var.to_netcdf(final_path)
                    print(f"File saved at: {final_path}")
                except ValueError as e:
                    print(f"Failed to save {final_path} due to: {e}")
    return
