
ball_strike_model_train_cutoff = '2020-01-01'
pitch_model_train_cutoff = '2020-01-01'
event_model_train_cutoff = '2023-01-01'

xgboost_ball_strike_pred_columns = ['zone', 'pitch_type', 'plate_z', 'plate_x', 'pfx_x', 'pfx_z', 'on_edge_left', 'on_edge_right', 'on_edge_top', 'on_edge_bot', 'on_edge_inside', 'on_edge_outside', 'on_low_inside_corner', 'on_low_outside_corner', 'on_high_inside_corner', 'on_high_outside_corner', 'on_edge', 'on_corner']

# i dont get to look at anything besides
xgboost_benchmark_columns = ['stand', 'p_throws', 'balls', 'strikes', 'count_value', 
 'strike_value', 'ball_value']

xgboost_pred_columns = [
 'release_speed',
 'release_pos_x',
 'release_pos_z',
 'zone',
 'stand',
 'p_throws',
 'balls',
 'strikes',
 'pfx_x',
 'pfx_z',
 'plate_x',
 'plate_z',
 'effective_speed',
 'release_spin_rate',
 'release_extension',
 'spin_axis',
 'tilt',
 'spin_efficiency',
 'on_edge_left',
 'on_edge_right',
 'on_edge_top',
 'on_edge_bot',
 'on_edge_inside',
 'on_edge_outside',
 'on_low_inside_corner',
 'on_low_outside_corner',
 'on_high_inside_corner',
 'on_high_outside_corner',
 'on_edge',
 'on_corner',
 'count_value',
 'strike_value',
 'ball_value',
 'strike_probability',
 'release_speed_max',
 'portion_of_max_velo',
 'velo_percentile_for_pitcher',
 'armside_horz_break',
 'armside_tilt',
 'tilt_off_fb',
 'horz_break_off_fb',
 'vert_break_off_fb',
 'k_pitch_type_adj',
 'pitch_type',
 'count_woba',
 'zone_woba',
 'pitch_woba']


xgboost_pred_columns_stuff_only = [x for x in xgboost_pred_columns 
                           if 'plate' not in x and 'zone' not in x and 'strike_probability' not in x
                            and '_pos_' not in x and 'edge' not in x and 'corner' not in x]

xgboost_pred_columns_cmd_only = [x for x in xgboost_pred_columns 
                           if 'velo' not in x and 'speed' not in x 
                           and 'pfx' not in x and 'break' not in x and 'spin' not in x
                           and 'tilt' not in x]

xgboost_target_column = 'delta_run_expectancy'
xgboost_categorical_columns = ['zone','k_pitch_type_adj','pitch_type']
xgboost_categorical_columns_cmd_only = ['zone','k_pitch_type_adj','pitch_type']
xgboost_categorical_columns_stuff_only = ['k_pitch_type_adj','pitch_type']

