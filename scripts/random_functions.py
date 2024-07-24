def find_last_datapoint(ds_dict, variable):
    
    data = ds_dict[list(ds_dict.keys())[5]][variable]

    # Reverse the time dimension (start from the end)
    data = data.isel(time=slice(None, None, -1))

    # Find the first time index with non-NaN values
    non_nan_time_index = data.notnull().any(dim=["lat", "lon"]).argmax().values

    # Find the first time index with non-zero values
    non_zero_time_index = (data != 0).any(dim=["lat", "lon"]).argmax().values

    # Find the maximum of both time indices to get the first time index with actual values
    last_actual_values_time_index = max(non_nan_time_index, non_zero_time_index)

    # Convert it to index from start of the data
    last_actual_values_time_index = len(data.time) - 1 - last_actual_values_time_index

    last_actual_values_time = data.time.isel(time=last_actual_values_time_index).values

    print("Last time index with actual values:", last_actual_values_time_index)
    print("Last time with actual values:", last_actual_values_time)