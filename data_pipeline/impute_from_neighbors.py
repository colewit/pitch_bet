
from scipy.spatial.distance import pdist, squareform

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def get_distance_matrix(vals1, vals2, weights1, weights2):
    """Converts a list of values to a distance matrix using itertools.product and a distance function.
    
    Args:
      vals: A list of values.
      distance_func: A function that calculates the distance between two values.
    
    Returns:
      A 2D NumPy array representing the distance matrix.
    """
    def euclidean_distance(val1, val2):
        return np.sqrt((val1 - val2) ** 2)
        
    # Generate all combinations of pairs from the list
    
    weights_arr1 = np.array(weights1)
    weights_arr2 = np.array(weights2)
    weights_arr1[np.isnan(weights_arr1)] = 0
    weights_arr2[np.isnan(weights_arr2)] = 0
    weight_matrix = weights_arr1[:, np.newaxis] + weights_arr2[np.newaxis, :]

    vals_arr1 = np.array(vals1)
    vals_arr2 = np.array(vals2)
    vals_arr1[np.isnan(vals_arr1)] = 0
    vals_arr2[np.isnan(vals_arr2)] = 0

    distance_matrix = euclidean_distance(vals_arr1[:, np.newaxis], vals_arr2[ np.newaxis,:])

    
    return distance_matrix * weight_matrix



def dist_between_pitchers(df, pitch_distance_df, pitch_types):

    weighted_dist = np.zeros((df.shape[0], df.shape[0]))
    for pt1 in pitch_types:

        
        wd_for_pitch = []
        
        for pt2 in pitch_types:

            if pt1[:2] in ['CU','SL'] and pt2[:2] in ['FF','FF','SI', 'CH']:
                continue
            elif pt2[:2] in ['CU','SL'] and pt1[:2] in ['FF','FF','SI', 'CH']:
                continue
            elif pt1[:2] == 'CH' and pt2[:2] not in ['SI', 'CH']:
                continue
            elif pt2[:2] == 'CH' and pt1[:2] not in ['SI', 'CH']:
                continue
            
            curr_dist = np.zeros((df.shape[0],df.shape[0]))
            
            for column in ['pred_pitch_value_cmd_RA9_pitch_type',
                   'pred_pitch_value_stuff_RA9_pitch_type',
                   'empirical_pitch_value_RA9_pitch_type']:
               
                curr_dist += get_distance_matrix(df[f'{column}_{pt1}'],
                                              df[f'{column}_{pt2}'],
                                              df[f'is_{pt1}'],
                                              df[f'is_{pt2}'])


            curr_dist *= 1+pitch_distance_df.loc[pt1, pt2]
            wd_for_pitch.append(curr_dist)

        stacked_matrices = np.stack(wd_for_pitch, axis=0)
        # Calculate the minimum along the new axis
        min_matrix = np.min(stacked_matrices, axis=0)
        
        weighted_dist += min_matrix
        

    return weighted_dist


def get_neighbors_pitcher(train_data, select_column, val_columns, meta_columns, pitch_distance_df, pitch_types):


    
    data = train_data[val_columns + meta_columns]
        
    data = data.sort_values(['game_date', 'team_at_bat_number'])
    data['month'] = data.game_date.dt.month
    data['year'] = data.game_date.dt.year
    query_df = data.groupby([select_column,'year', 'month']).first().reset_index()
    
    data['year'] = data.game_date.dt.year
    
    k = 10
    month_df = {}
    for month, qdf in query_df.groupby(['year','month']):
        
        num_rows = len(qdf)

        distance_matrix =  dist_between_pitchers(qdf, pitch_distance_df, pitch_types)
        knn = NearestNeighbors(n_neighbors=k)
    
        # Fit the model on the data
        knn.fit(distance_matrix)
        
        # Find the k nearest neighbors
        distances, indices = knn.kneighbors(distance_matrix)

        d={}
        for (player, idx_set) in zip(qdf[select_column], indices):
            d[player] = qdf.iloc[idx_set][select_column].values
    
        month_df[month]=d
    return month_df
 
def get_neighbors_batter(data, select_column, val_columns, meta_columns):
    
    '''
    find the most similar batters up thru the current month
    '''
    def fill_na_with_percentile(data, columns, percentile):
        for col in columns:
            perc_value = np.nanpercentile(data[col], percentile)
            data[col].fillna(perc_value, inplace=True)
        return data 
        
    greater_is_better_cols = [x for x in val_columns if 'strikeout' not in x]
    less_is_better_cols = [x for x in val_columns if 'strikeout' in x]

    fill_na_with_percentile(data, greater_is_better_cols, 5)
    
    # Fill less is better columns with 95th percentile
    fill_na_with_percentile(data, less_is_better_cols, 95)
    
    data = data[val_columns + meta_columns].dropna()
    
    data = data.sort_values(['game_date', 'team_at_bat_number'])
    data['month'] = data.game_date.dt.month
    data['year'] = data.game_date.dt.year
    query_df = data.groupby([select_column,'month', 'year']).first().reset_index()
    
    scaler = MinMaxScaler()
    
    k = 10
    
    month_df = {}
    for grp, qdf in query_df.groupby(['year','month']):

        query_vals = scaler.fit_transform(qdf[val_columns])
        
        pca=PCA(4)
        
        query_vals = pca.fit_transform(query_vals)
        
        knn = NearestNeighbors(n_neighbors=k)
    
        # Fit the model on the data
        knn.fit(query_vals)
        
        # Find the k nearest neighbors
        distances, indices = knn.kneighbors(query_vals)

        d={}
        for (player, idx_set) in zip(qdf[select_column], indices):
            d[player] = qdf.iloc[idx_set][select_column].values
    
        month_df[grp]=d
    return month_df


def impute_from_neighbors(train_data, train_meta, train_pitches):

    meta=['game_date','pitcher','player_name','time_thru_the_order','team_at_bat_number','batter']
    batter_vals =  ['estimated_iso_mean',
                'estimated_ba_using_speedangle_batter','triple_batter','single_batter',
                 'estimated_iso_95th',
                 'estimated_iso_75th',
                 'estimated_iso_25th',
                 'launch_angle_mean',
                 'launch_speed_mean',
                 'launch_speed_max',
                 'launch_speed_95th',
                 'launch_speed_75th',
                 'launch_speed_25th',
                 'pct_barrel',
                 'pct_solid_contact',
                 'pct_exit_velo_above_95_mph',
                 'pct_exit_velo_above_100_mph',
                 'pct_homerun_launch_angle',
                 'pct_hit_distance_above_350',
                 'pct_hit_distance_above_375',
                 'pct_hit_distance_above_400',
                 'pct_hit_distance_above_420',
                 'estimated_iso_mean_regressed',
                 'estimated_iso_95th_regressed',
                 'estimated_iso_75th_regressed',
                 'estimated_iso_25th_regressed',
                 'launch_angle_mean_regressed',
                 'launch_speed_mean_regressed',
                 'launch_speed_max_regressed',
                 'launch_speed_95th_regressed',
                 'launch_speed_75th_regressed',
                 'launch_speed_25th_regressed',
                 'pct_barrel_regressed',
                 'pct_solid_contact_regressed',
                 'pct_exit_velo_above_95_mph_regressed',
                 'pct_exit_velo_above_100_mph_regressed',
                 'pct_homerun_launch_angle_regressed',
                 'pct_hit_distance_above_350_regressed',
                 'pct_hit_distance_above_375_regressed',
                 'pct_hit_distance_above_400_regressed',
                 'pct_hit_distance_above_420_regressed',
                 'strikeout_batter','walk_batter','homerun_batter',
                 'called_strike_batter',
                 'whiff_batter',
                 'homerun_batter',
                 'walk_batter',
                 'strikeout_batter',
                 'estimated_woba_using_speedangle_batter',
                 'called_strike_batter_regressed',
                 'whiff_batter_regressed',
                 'homerun_batter_regressed',
                 'walk_batter_regressed',
                 'strikeout_batter_regressed',
                 'estimated_woba_using_speedangle_batter_regressed']

    pitcher_cols_to_fill = [
         'pred_pitch_value_RA9',
         'pred_pitch_value_cmd_RA9',
         'pred_pitch_value_stuff_RA9',
         'empirical_pitch_value_RA9',
         'pred_pitch_value_RA9_regressed',
         'pred_pitch_value_cmd_RA9_regressed',
         'pred_pitch_value_stuff_RA9_regressed',
         'empirical_pitch_value_RA9_regressed',
         'called_strike_pitcher',
         'whiff_pitcher',
         'homerun_pitcher',
         'walk_pitcher',
         'strikeout_pitcher',
         'estimated_woba_using_speedangle_pitcher']

    train_data = pd.concat([train_data.reset_index(), train_meta.reset_index()], axis = 1)

    print('before we have', train_pitches.k_pitch_type_adj.unique())
    train_pitches.k_pitch_type_adj = [None if x in ['nan','None'] else x for x in train_pitches.k_pitch_type_adj]

    
    pitches = train_pitches[['k_pitch_type_adj','effective_speed',
                       'release_spin_rate','armside_horz_break','pfx_z']]

    print(pitches)
    pitches = pitches.dropna()
    
    scaler = MinMaxScaler()
    pitches[['effective_speed', 'release_spin_rate','armside_horz_break','pfx_z']]\
        = scaler.fit_transform(
            pitches[['effective_speed', 'release_spin_rate','armside_horz_break','pfx_z']])

    pitch_types = pitches.k_pitch_type_adj.unique()

    print('putch types are', pitch_types)
    pitches = pitches.groupby('k_pitch_type_adj').agg('mean')
    
    distances = pdist(pitches, metric='euclidean')
    
    # Convert to a square distance matrix
    distance_matrix = squareform(distances)
    
    distance_df = pd.DataFrame(distance_matrix, index=pitches.index, columns=pitches.index)


    pitcher_vals = [x for x in train_data.columns if '_pitcher' in x or 'pitch_type' in x or 'is_'==x[:3]]

    pitcher_d = get_neighbors_pitcher(train_data,'pitcher', pitcher_vals, meta, distance_df, pitch_types)
    batter_d = get_neighbors_batter(train_data,'batter', batter_vals, meta)


    train_data_cp = train_data.copy(deep=True).sort_values(['game_date', 'team_at_bat_number'])
    
    df_batter = {k:[] for k in batter_vals}
    df_batter['date']=[]
    df_batter['batter'] = []
    
    
    
    for (year, month), d in batter_d.items():
    
        df = train_data[train_data_cp.game_date <=f'{year}-{month}-01']\
            .groupby('batter').last().reset_index()
        for batter, neighbors in d.items():
            sub_df = df[df.batter.isin(neighbors)]
            vals = sub_df[batter_vals].mean().to_dict()
            for k,v in vals.items():
                df_batter[k].append(v)
    
            df_batter['batter'].append(batter)
            df_batter['date'].append(f'{year}-{month}-01')

    train_data_cp = train_data.copy(deep=True).sort_values(['game_date', 'team_at_bat_number'])
    
    df_pitcher = {k:[] for k in pitcher_cols_to_fill}
    df_pitcher['date']=[]
    df_pitcher['pitcher'] = []
    
    
    
    for (year, month), d in pitcher_d.items():
    
        df = train_data[train_data_cp.game_date <=f'{year}-{month}-01']\
            .groupby('pitcher').last().reset_index()
        for pitcher, neighbors in d.items():
            sub_df = df[df.pitcher.isin(neighbors)]
            vals = sub_df[pitcher_cols_to_fill].mean().to_dict()
            for k,v in vals.items():
                df_pitcher[k].append(v)
    
            df_pitcher['pitcher'].append(pitcher)
            df_pitcher['date'].append(f'{year}-{month}-01')
        
        
    df_pitcher = pd.DataFrame(df_pitcher).rename(columns={'date':'game_date'})
    df_pitcher['game_date'] = pd.to_datetime(df_pitcher.game_date)
    
    df_batter = pd.DataFrame(df_batter).rename(columns={'date':'game_date'})
    df_batter['game_date'] = pd.to_datetime(df_batter.game_date)
    
    merged = pd.merge_asof(
        train_data,
        df_batter,
        by = ['batter'], on='game_date', direction = 'backward')

    merged = pd.merge_asof(
        merged,
        df_pitcher,
        by = ['pitcher'], on='game_date', direction = 'backward')

    
        
    dupe_cols = [x for x in merged.columns if x[-2:]=='_x' or x[-2:]=='_y']
    for col in [x for x in merged.columns if x[-2:]=='_x']:
        merged[col.replace('_x','')] = merged[col].combine_first(merged[col.replace('_x','_y')])
    merged = merged.drop(columns = dupe_cols)

    merged = merged[train_data.columns].drop(columns = train_meta.columns)
    return merged

if __name__ == '__main__':

    
    train_data = d['x_train']
    train_meta= d['x_meta_train'][['player_name','pitcher','batter','game_date','batting_team']]

    test_data = d['x_test']
    test_meta= d['x_meta_test'][['player_name','pitcher','batter','game_date','batting_team']]

    train_pitches = pd.read_parquet('data_with_pitch_values.parquet')
    
    train_data = impute_from_neighbors(train_data, train_meta, train_pitches)
    test_data = impute_from_neighbors(test_data, test_meta, train_pitches)

    d['x_train'] = train_data
    d['x_test'] = test_data
    with open('data_for_ordinal_imputed.pkl','wb') as f:
        
        pickle.dump(d,f)
