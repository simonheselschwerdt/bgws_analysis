Please apply the scripts in the following order:

1. 1_CMIP6_consistent_time.ipynb
2. 2_CMIP6_regrid.ipynb
3. 3_CMIP6_landmask.ipynb
4. 4_CMIP6_slice_antartica.ipynb
5. 5_CMIP6_convert_units.ipynb

These scripts are only needed if...

... you want to compute vpd/wue or sm at 1 and 2 m:
    - CMIP_compute_vpd.ipynb
    - CMIP_compute_wue.ipynb
    - CMIP6_compute_1_2_m_soil_moisture.ipynb
    
    
... you have missing or redundant timesteps in your data...
    - CMIP6_handle_missing_or_redund_time.ipynb
    
... you have want to compute 1 and 2 m sm and your tsl and mrsol data has different depth layers and different names for it:
    - CMIP6_converting_solay.ipynb
    - CMIP6_interpolate_mrsol_tsl_depth.ipynb