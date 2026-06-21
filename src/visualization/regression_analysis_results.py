# Step 4.1: (you already ran regressions and have results_bw_regime, results_gw_regime)
# Step 4.2: selected_variables & yaxis_limits are defined as you wrote

fig = figure_perm_and_parallel(
    results_bw_regime=results_bw_regime,
    results_gw_regime=results_gw_regime,
    predictor_vars=predictor_vars,
    ddict_change_sub_mean=ds_dict_change_sub_mean,
    ddict_sub_mean=ds_dict_sub_mean,
    selected_variables=selected_variables,
    yaxis_limits=yaxis_limits,
    dpi=300,
    filetype="pdf",
    savepath="/your/path/here/"
)
