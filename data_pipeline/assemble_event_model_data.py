import pandas as pd
import numpy as np

import pickle
from sklearn.preprocessing import OneHotEncoder

from data_pipeline.aggregate_pitch_values import past_rolling_mean
from config import pitch_model_train_cutoff, event_model_train_cutoff

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def one_hot_encode_data(one_hot_encoder, data, categorical_columns):

    if categorical_columns != []:
        one_hot_data = one_hot_encoder.transform(data[categorical_columns])
    
        # Convert encoded data to DataFrame for better visualization
        one_hot_data_df = pd.DataFrame(one_hot_data.toarray(),
                                    columns=one_hot_encoder\
                                           .get_feature_names_out(categorical_columns))

        data = pd.concat([data.drop(columns = categorical_columns).reset_index(drop=True),
                          one_hot_data_df], axis = 1)
    return data

def impute_nas(train_data):
     # whiff_ fill with 1th perc
    # HR 99th
    # fewer samples for estimated_woba_using_speedangle
    # called_strike_, pred_pitch_value_, empirical_pitch_value
    # pred_pitch_value_stuff_next_3_
    # outcome...batter
    
    imputed_train_data = train_data.copy(deep=True)
    
    whiff_columns = [x for x in train_data.columns if 'whiff_' in x and 'batter' not in x]
    imputed_train_data[whiff_columns] = imputed_train_data[whiff_columns]\
        .fillna(np.nanpercentile(imputed_train_data.whiff_pitcher, 10))
    
    homerun_columns = [x for x in train_data.columns if 'homerun_' in x and 'batter' not in x]
    imputed_train_data[homerun_columns] = imputed_train_data[homerun_columns]\
        .fillna(np.nanpercentile(imputed_train_data.homerun_pitcher, 90))
    
    pitch_value_columns = [x for x in train_data.columns if 'pitch_value_' in x and 'batter' not in x]
    imputed_train_data[pitch_value_columns] = imputed_train_data[pitch_value_columns]\
        .fillna(np.nanpercentile(imputed_train_data.pred_pitch_value_RA9, 90))
    
    
    imputed_train_data['woba_pitcher'] = imputed_train_data['woba_pitcher']\
        .fillna(np.nanpercentile(imputed_train_data['woba_pitcher'],90))
    
    called_strike_columns = [x for x in train_data.columns if 'called_strike_' in x and 'batter' not in x]
    imputed_train_data[called_strike_columns] = imputed_train_data[called_strike_columns]\
        .fillna(np.nanpercentile(imputed_train_data.called_strike_pitcher, 10))
    
    imputed_train_data = imputed_train_data.sort_values(['game_date','lineup_slot'])
    imputed_train_data['backup_next_batter_woba'] = imputed_train_data\
        .groupby(['game_date','lineup_slot', 'batting_team']).next_batter_woba\
        .transform('first')
    
    imputed_train_data['next_batter_woba'] = imputed_train_data['next_batter_woba'].combine_first(
        imputed_train_data['backup_next_batter_woba'])
    
    imputed_train_data = imputed_train_data.drop(columns = 'backup_next_batter_woba')
    
    called_strike_columns = [x for x in train_data.columns if 'called_strike_' in x and 'batter' in x]
    imputed_train_data[called_strike_columns] = imputed_train_data[called_strike_columns]\
        .fillna(np.nanpercentile(imputed_train_data.called_strike_batter, 50))
    
    whiff_columns = [x for x in train_data.columns if 'whiff_' in x and 'batter' in x]
    imputed_train_data[whiff_columns] = imputed_train_data[whiff_columns]\
        .fillna(np.nanpercentile(imputed_train_data.whiff_batter, 70))
    
    homerun_columns = [x for x in train_data.columns if 'homerun_' in x and 'batter' in x]
    imputed_train_data[homerun_columns] = imputed_train_data[homerun_columns]\
        .fillna(np.nanpercentile(imputed_train_data.homerun_batter, 30))
    
    strikeout_columns = [x for x in train_data.columns if 'strikeout_' in x and 'batter' in x]
    imputed_train_data[strikeout_columns] = imputed_train_data[strikeout_columns]\
        .fillna(np.nanpercentile(imputed_train_data.strikeout_batter, 70))
    
    woba_columns = [x for x in train_data.columns if 'woba_' in x and 'batter' in x]
    imputed_train_data[woba_columns] = imputed_train_data[woba_columns]\
        .fillna(np.nanpercentile(imputed_train_data.woba_batter, 30))
    
    
    walk_columns = [x for x in train_data.columns if 'walk_' in x and 'batter' in x]
    imputed_train_data[walk_columns] = imputed_train_data[walk_columns]\
        .fillna(np.nanpercentile(imputed_train_data.walk_batter, 30))
    return imputed_train_data

# Define a function to calculate cumulative sum excluding the current row
def cumulative_sum_up_to_not_including_row(column):
    return column.cumsum().shift(1)

def add_state_data(train_data):
    
    train_data = train_data.sort_values(['game_date', 'team_at_bat_number'])

    
    train_data['homerun'] = (train_data.label == 5)
    train_data['double'] = (train_data.label == 4)
    train_data['single'] = (train_data.label == 3)
    train_data['walk'] = (train_data.label == 2)
    train_data['strikeout'] = (train_data.label == 0)
    
    train_data['linear_weight'] = (
        0.55 * (train_data.label == 2) + 
        0.7 * (train_data.label == 3) + 
        (train_data.label == 4) + 
        (train_data.label == 5) * 1.65 - 
        0.26
    )

    train_data['on_1b'] = (~train_data['on_1b'].isna()).astype(int)
    train_data['on_2b'] = (~train_data['on_2b'].isna()).astype(int)
    train_data['on_3b'] = (~train_data['on_3b'].isna()).astype(int)
    
    
    # Calculate cumulative sum for each category and each inning
    categories = ['strikeout', 'walk', 'homerun', 'double', 'single', 'linear_weight']
    for category in categories:
        train_data[f'{category}s_so_far'] = \
            train_data.groupby(['pitcher', 'game_date'])[category]\
            .transform(cumulative_sum_up_to_not_including_row)
        
        train_data[f'{category}s_this_inning'] = \
            train_data.groupby(['pitcher', 'game_date', 'inning'])[category]\
            .transform(cumulative_sum_up_to_not_including_row)
    
    # Calculate rolling sum for each category
    for category in categories:
        
        train_data[f'{category}s_last_3'] = train_data.groupby(['pitcher', 'game_date'])[category]\
            .transform(lambda x: x.rolling(window=3, min_periods=3).sum().shift(1))
        
        train_data[f'{category}s_last_9'] = train_data.groupby(['pitcher', 'game_date'])[category]\
            .transform(lambda x: x.rolling(window=9, min_periods=9).sum().shift(1))
    
    categories = ['strikeout', 'walk', 'homerun', 'double', 'single', 'linear_weight']
    for category in categories:
        
        train_data[f'matchup_{category}s_today'] = \
            train_data.groupby(['pitcher', 'game_date', 'batter'])[category]\
            .transform(cumulative_sum_up_to_not_including_row)
        
        train_data[f'batter_{category}s_today'] = \
            train_data.groupby(['game_date', 'batter'])[category]\
            .transform(cumulative_sum_up_to_not_including_row)

    return train_data

def dist_between_pitchers(df, pitch_distance_df):

    weighted_dist = np.zeros((df.shape[0], df.shape[0]))

        
    for pt1 in ['CH', 'CU', 'FF_1', 'FF_2','SI', 'SL']:

        
        wd_for_pitch = []
        
        for pt2 in ['CH', 'CU', 'FF_1', 'FF_2','SI', 'SL']:

            if pt1 in ['CU','SL'] and pt2 in ['FF_1','FF_2','SI', 'CH']:
                continue
            elif pt2 in ['CU','SL'] and pt1 in ['FF_1','FF_2','SI', 'CH']:
                continue
            elif pt1 == 'CH' and pt2 not in ['SI', 'CH']:
                continue
            elif pt2 == 'CH' and pt1 not in ['SI', 'CH']:
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

    


def get_neighbors_pitcher(select_column, val_columns, meta_columns, pitch_distance_df):
    
    data = train_data[val_columns + meta_columns]
        
    data = data.sort_values(['game_date', 'team_at_bat_number'])
    data['month'] = data.game_date.dt.month
    data['year'] = data.game_date.dt.year
    query_df = data.groupby([select_column,'year', 'month']).first().reset_index()
    
    data['year'] = data.game_date.dt.year
    
    k = 5
    month_df = {}
    for month, qdf in query_df.groupby(['year','month']):
        
        num_rows = len(qdf)

        distance_matrix =  dist_between_pitchers(qdf, pitch_distance_df)
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
 
def get_neighbors_batter(select_column, val_columns, meta_columns):
    
    '''
    find the most similar batters up thru the current month
    '''
    
    
    data = train_data[val_columns + meta_columns].dropna()
    
    data = data.sort_values(['game_date', 'team_at_bat_number'])
    data['month'] = data.game_date.dt.month
    data['year'] = data.game_date.dt.year
    query_df = data.groupby([select_column,'month', 'year']).first().reset_index()
    
    scaler = MinMaxScaler()
    
    k = 5
    
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

def make_neighbor_df(matchup_df, pitcher_d, batter_d):
    l = []

    # so example is june 1 for G Cole
    # I go and find all the matchups where
    # the pitcher is a neighbor of GC and I make sure the date of the matchup
    # occurs before the first date of the neighbord.
    for (year, month), sub_d  in pitcher_d.items():
        for pitcher, grp in sub_d.items():
        
            
            d = matchup_df[np.logical_and(
                matchup_df.pitcher.isin(grp),
                matchup_df.date < f'{year}-{month}-1')].copy(deep=True)
            d['grp_month'] = month
            d['grp_year'] = year
            d['grp_pitcher'] = pitcher
            l.append(d)

    # so pitcher df for Cole now has matchups prior to June 1, it has grp month 6
    # and group year 2023 and grp pitcher which is Cole's Id
    pitcher_df = pd.concat(l)
    
    
    l = []
    for (year, month), sub_d  in batter_d.items():
        for batter, grp in sub_d.items():
    
            d = matchup_df[np.logical_and(
                matchup_df.batter.isin(grp),
                matchup_df.date < f'{year}-{month}-1')].copy(deep=True)
            d['grp_month'] = month
            d['grp_year'] = year
            d['grp_batter'] = batter
            l.append(d)
    batter_df = pd.concat(l)
    return batter_df, pitcher_df




def score_matchups(batter_df, pitcher_df):

    # find the woba in all at bats that have occurred prior
    def agg_bdf(bdf, pitcher_denom):
    
        if bdf.empty:
            return np.nan, 0
            
        pitcher_ratio = bdf.woba_pitcher/pitcher_denom
        batter_ratio = bdf.batter_ratio
        
        weights = batter_ratio * pitcher_ratio
        woba = (weights*bdf.at_bat_woba).sum()/bdf.num_at_bats.sum()
        num_ab = bdf.num_at_bats.sum()
        return woba, num_ab

    d = {'batter':[], 'matchup_woba':[], 'pitcher':[], 'grp_matchup_num_ab':[],
     'grp_matchup_woba':[], 'matchup_num_ab':[], 'month':[],'year':[]}


    # now we go thru Cole, and we get his id and month, year = 6,2023
    for (pitcher, month, year), pdf in tqdm.tqdm(pitcher_df.groupby(['grp_pitcher', 'grp_month','grp_year'])):
    
        pitcher_denom = pdf['grp_pitcher_woba'].iloc[0]

        # filter batter df down to the df where month = june and year = 2023
        filtered_batter_df = batter_df[
            
            np.logical_and(batter_df.pitcher.isin(pdf.pitcher.unique()),
               
               np.logical_and(
                   batter_df.grp_batter.isin(unique_batters),
                   
                   np.logical_and(
                       batter_df.grp_month==month,
                       batter_df.grp_year==year)))]

        # add print / assert here to verify that game_date of the matchup is less than month, year
        # (in this example less than june 1 2023
    
        
        for idx, (batter, bdf) in enumerate(filtered_batter_df.groupby('grp_batter')):
    
            
            d['batter'].append(batter)
            d['pitcher'].append(pitcher)
            d['month'].append(month)
            d['year'].append(year)
            
            
            woba_grp, num_ab_grp = agg_bdf(bdf, pitcher_denom)
            d['grp_matchup_woba'].append(woba_grp)
            d['grp_matchup_num_ab'].append(num_ab_grp)
    
            woba, num_ab = agg_bdf(bdf[bdf.is_representative_batter], pitcher_denom)
            d['matchup_woba'].append(woba)
            d['matchup_num_ab'].append(num_ab)


    matchup_lookup = pd.DataFrame(d)
    matchup_lookup['month'] = np.where(matchup_lookup.month ==4, 10, matchup_lookup.month -1)
    matchup_lookup['year'] = np.where(matchup_lookup.month ==4, matchup_lookup.year-1, matchup_lookup.year)
    
    matchup_lookup['game_date'] = pd.to_datetime([y+'-'+m for y,m in 
                              zip(matchup_lookup.year.astype(str),matchup_lookup.month.astype(str))])
    
    matchup_lookup.pitcher = matchup_lookup.pitcher.astype(int)
    matchup_lookup.batter = matchup_lookup.batter.astype(int)
    return matchup_lookup


def setup_batter_pitcher_matchups(train_data, pitches):

        
    pitches = pitches[['k_pitch_type_adj','effective_speed',
                       'release_spin_rate','armside_horz_break','pfx_z']]
    
    scaler = MinMaxScaler()
    pitches[['effective_speed', 'release_spin_rate','armside_horz_break','pfx_z']]\
        = scaler.fit_transform(
            pitches[['effective_speed', 'release_spin_rate','armside_horz_break','pfx_z']])
    
    pitches = pitches.groupby('k_pitch_type_adj').agg('mean')#.reset_index()
    
    distances = pdist(pitches, metric='euclidean')
    
    # Convert to a square distance matrix
    distance_matrix = squareform(distances)
    
    # Convert the distance matrix to a DataFrame
    distance_df = pd.DataFrame(distance_matrix, index=pitches.index, columns=pitches.index)

    meta=['game_date','pitcher','label','player_name','time_thru_the_order','batter']

    pitcher_vals = [x for x in train_data.columns if '_pitcher' in x or 'pitch_type' in x or 'is_'==x[:3]]
    batter_vals = [x for x in train_data.columns if 'batter' in x and 'regressed' in x] + ['woba_batter']
    
    pitcher_d = get_neighbors_pitcher('pitcher', pitcher_vals, meta, distance_df)
    batter_d = get_neighbors_batter('batter', batter_vals, meta)
    train_data['year'] = train_data.game_date.dt.year
    train_data['month'] = train_data.game_date.dt.month

    # how matchup went down in a given month between every batter, pitcher combo
    matchup_df = train_data.groupby(['batter','pitcher','month', 'year'])\
        .agg(at_bat_woba=('at_bat_woba', 'sum'),
             num_at_bats=('at_bat_woba',len),
             woba_batter=('woba_batter','last'),
             woba_pitcher=('woba_pitcher','last')).reset_index()
    
    matchup_df['date'] = [y+'-'+x for x,y in matchup_df[['month','year']].astype(str).values]
    matchup_df['date'] = pd.to_datetime(matchup_df.date)

    batter_df, pitcher_df = make_neighbor_df(matchup_df, pitcher_d, batter_d)

    def f(x):
        try:
            return x[~x.isna()].iloc[0]
        except:
            return np.nan
            
    batter_df['is_representative_batter'] = \
        batter_df.batter==batter_df.grp_batter
    
    batter_df['grp_batter_woba'] = np.where(
        batter_df['is_representative_batter'],
        batter_df.woba_batter,
        np.nan)
    
    batter_df['grp_batter_woba'] = batter_df\
        .groupby('grp_batter').grp_batter_woba.transform('mean')
    
    batter_df['grp_batter_woba'] = batter_df\
        .groupby('grp_batter').grp_batter_woba.transform(lambda x: x.fillna(np.nanmean(x)))
    
    batter_df['grp_mean_batter_woba'] = batter_df\
        .groupby('grp_batter').woba_batter.transform(f)
    
    
    batter_df['batter_ratio'] = batter_df['grp_mean_batter_woba'] /\
        batter_df['grp_batter_woba']
    
    
    
    pitcher_df['is_representative_pitcher'] = \
        pitcher_df.pitcher==pitcher_df.grp_pitcher
    
    pitcher_df['grp_pitcher_woba'] = np.where(
        pitcher_df['is_representative_pitcher'],
        pitcher_df.woba_pitcher,
        np.nan)
    
    pitcher_df['grp_pitcher_woba'] = pitcher_df\
        .groupby('grp_pitcher').grp_pitcher_woba.transform(f)
    
    pitcher_df['grp_pitcher_woba'] = pitcher_df\
        .groupby('grp_pitcher').grp_pitcher_woba.transform(lambda x: x.fillna(np.nanmean(x)))
    
    pitcher_df['grp_mean_pitcher_woba'] = pitcher_df\
        .groupby('grp_pitcher').woba_pitcher.transform('mean')
    
    
    pitcher_df['pitcher_ratio'] = pitcher_df['woba_pitcher']\
        .combine_first(pitcher_df.grp_mean_pitcher_woba)/pitcher_df['grp_pitcher_woba']

    return score_matchups(batter_df, pitcher_df)

def prepare_batter_contact_data(train_pitches):
    
    batter_contact_quality = train_pitches\
        [train_pitches.end_of_at_bat][
        ['team_at_bat_number','launch_speed','launch_angle',
         'launch_speed_angle','batter','game_date','hit_distance_sc',
         'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle']]
    
    
    # need to change these to cummean
    batter_contact_quality = batter_contact_quality.sort_values(['game_date','team_at_bat_number'])
    
    batter_contact_quality['estimated_iso'] = \
        batter_contact_quality.estimated_woba_using_speedangle - batter_contact_quality.estimated_ba_using_speedangle
    
    batter_contact_quality['estimated_iso_mean'] = batter_contact_quality.groupby(['batter'])\
        .estimated_iso.transform(lambda x:x.rolling(700, min_periods = 100).mean().shift(1))
    
    batter_contact_quality['estimated_iso_95th'] = batter_contact_quality.groupby(['batter'])\
        .estimated_iso.transform(lambda x:x.rolling(700, min_periods = 100).quantile(.95).shift(1))
    
    batter_contact_quality['estimated_iso_75th'] = batter_contact_quality.groupby(['batter'])\
        .estimated_iso.transform(lambda x:x.rolling(700, min_periods = 100).quantile(.75).shift(1))
    
    batter_contact_quality['estimated_iso_25th'] = batter_contact_quality.groupby(['batter'])\
        .estimated_iso.transform(lambda x:x.rolling(700, min_periods = 100).quantile(.25).shift(1))
    
    
    batter_contact_quality['launch_angle_mean'] = batter_contact_quality.groupby(['batter'])\
        .launch_angle.transform(lambda x:x.rolling(700, min_periods = 100).mean().shift(1))
    
    batter_contact_quality['launch_speed_mean'] = batter_contact_quality.groupby(['batter'])\
        .launch_speed.transform(lambda x:x.rolling(700, min_periods = 100).mean().shift(1))
    batter_contact_quality['launch_speed_max'] = batter_contact_quality.groupby(['batter'])\
        .launch_speed.transform(lambda x:x.rolling(700, min_periods = 100).max().shift(1))
    
    batter_contact_quality['launch_speed_95th'] = batter_contact_quality.groupby(['batter'])\
        .launch_speed.transform(lambda x:x.rolling(700, min_periods = 100).quantile(.95).shift(1))
    
    batter_contact_quality['launch_speed_75th'] = batter_contact_quality.groupby(['batter'])\
        .launch_speed.transform(lambda x:x.rolling(700, min_periods = 100).quantile(.75).shift(1))
    
    batter_contact_quality['launch_speed_25th'] = batter_contact_quality.groupby(['batter'])\
        .launch_speed.transform(lambda x:x.rolling(700, min_periods = 100).quantile(.25).shift(1))
    
    
    batter_contact_quality['is_barrel'] = batter_contact_quality.launch_speed_angle == 6
    batter_contact_quality['is_solid_contact'] = batter_contact_quality.launch_speed_angle == 5
    batter_contact_quality['exit_velo_above_95_mph'] = batter_contact_quality.launch_speed > 95
    batter_contact_quality['exit_velo_above_100_mph'] = batter_contact_quality.launch_speed > 100
    
    batter_contact_quality['homerun_launch_angle'] = batter_contact_quality.launch_angle.between(25,35)
    
    batter_contact_quality['hit_distance_above_350'] = batter_contact_quality.hit_distance_sc > 350
    batter_contact_quality['hit_distance_above_375'] = batter_contact_quality.hit_distance_sc > 375
    batter_contact_quality['hit_distance_above_400'] = batter_contact_quality.hit_distance_sc > 400
    batter_contact_quality['hit_distance_above_420'] = batter_contact_quality.hit_distance_sc > 420
    
    
    batter_contact_quality['pct_barrel'] = batter_contact_quality.groupby(['batter'])\
        .is_barrel.transform(lambda x: x.rolling(700,100).mean().shift(1))
    
    batter_contact_quality['pct_solid_contact'] = batter_contact_quality.groupby(['batter'])\
        .is_solid_contact.transform(lambda x: x.rolling(700,100).mean().shift(1))
    
    batter_contact_quality['pct_exit_velo_above_95_mph'] = batter_contact_quality.groupby(['batter'])\
        .exit_velo_above_95_mph.transform(lambda x: x.rolling(700,100).mean().shift(1))
    
    batter_contact_quality['pct_exit_velo_above_100_mph'] = batter_contact_quality.groupby(['batter'])\
        .exit_velo_above_100_mph.transform(lambda x: x.rolling(700,100).mean().shift(1))
    
    batter_contact_quality['pct_homerun_launch_angle'] = batter_contact_quality.groupby(['batter'])\
        .homerun_launch_angle.transform(lambda x: x.rolling(700,100).mean().shift(1))
    
    batter_contact_quality['num_barrel'] = batter_contact_quality.groupby(['batter'])\
        .is_barrel.transform(lambda x: x.rolling(700,100).sum().shift(1))
    
    batter_contact_quality['num_solid_contact'] = batter_contact_quality.groupby(['batter'])\
        .is_solid_contact.transform(lambda x: x.rolling(700,100).sum().shift(1))
    
    batter_contact_quality['num_exit_velo_above_95_mph'] = batter_contact_quality.groupby(['batter'])\
        .exit_velo_above_95_mph.transform(lambda x: x.rolling(700,100).sum().shift(1))
    
    batter_contact_quality['num_exit_velo_above_100_mph'] = batter_contact_quality.groupby(['batter'])\
        .exit_velo_above_100_mph.transform(lambda x: x.rolling(700,100).sum().shift(1))
    
    batter_contact_quality['num_homerun_launch_angle'] = batter_contact_quality.groupby(['batter'])\
        .homerun_launch_angle.transform(lambda x: x.rolling(700,100).sum().shift(1))
    
    
    batter_contact_quality['pct_hit_distance_above_350'] = batter_contact_quality.groupby(['batter'])\
        .hit_distance_above_350.transform(lambda x: x.rolling(700,100).mean().shift(1))
    batter_contact_quality['num_hit_distance_above_350'] = batter_contact_quality.groupby(['batter'])\
        .hit_distance_above_350.transform(lambda x: x.rolling(700,100).sum().shift(1))
    
    batter_contact_quality['pct_hit_distance_above_375'] = batter_contact_quality.groupby(['batter'])\
        .hit_distance_above_375.transform(lambda x: x.rolling(700,100).mean().shift(1))
    batter_contact_quality['num_hit_distance_above_375'] = batter_contact_quality.groupby(['batter'])\
        .hit_distance_above_375.transform(lambda x: x.rolling(700,100).sum().shift(1))
    
    batter_contact_quality['pct_hit_distance_above_400'] = batter_contact_quality.groupby(['batter'])\
        .hit_distance_above_400.transform(lambda x: x.rolling(700,100).mean().shift(1))
    batter_contact_quality['num_hit_distance_above_400'] = batter_contact_quality.groupby(['batter'])\
        .hit_distance_above_400.transform(lambda x: x.rolling(700,100).sum().shift(1))
    
    batter_contact_quality['pct_hit_distance_above_420'] = batter_contact_quality.groupby(['batter'])\
        .hit_distance_above_420.transform(lambda x: x.rolling(700,100).mean().shift(1))
    batter_contact_quality['num_hit_distance_above_420'] = batter_contact_quality.groupby(['batter'])\
        .hit_distance_above_420.transform(lambda x: x.rolling(700,100).sum().shift(1))

    contact_quality_cols =[x for x in batter_contact_quality.columns if 
     'num' in x or 'pct' in x or 'mean' in x or 'max' in x or '5th' in x or '0th' in x]
    
    batter_contact_quality['dummy'] = 1
    batter_contact_quality['cumulative_at_bats'] = \
        batter_contact_quality.groupby('batter').dummy.transform('cumsum')
    
    batter_contact_quality['league_average_at_bats'] = 700 - batter_contact_quality['cumulative_at_bats']

    # in the future think about if you want the contact quality to be out of 
    # only abs that end in play.
    
    l = []
    for column in contact_quality_cols:
        if column=='team_at_bat_number':
            continue
        batter_contact_quality[column+'_regressed'] = \
            (batter_contact_quality['league_average_at_bats']*batter_contact_quality[column].mean() + \
            batter_contact_quality['cumulative_at_bats']*batter_contact_quality[column])
        
        batter_contact_quality[column+'_regressed'] /= 700
        l.append(column+'_regressed')
    contact_quality_cols += l
    return batter_contact_quality, contact_quality_cols

def prepare_batter_outcome_data(train_pitches):
    
    # prep values for rolling window
    roll_columns = ['called_strike', 'whiff', 'homerun', 'single','double','triple',
                    'walk', 'strikeout', 'estimated_woba_using_speedangle', 
                    'estimated_ba_using_speedangle']
    
    sort_columns = ['game_date','at_bat_number']
    group_columns = ['batter']
    columns_to_keep = ['walk','field_out','strikeout',
                       'single','double','homerun','triple',
                       'pitcher','pitcher_at_bat_number']
    window_size = 700  # Number of pitches thrown
    min_period = 200
    
    # find rolling performance of batter up to current date
    
    batter_outcome_df = past_rolling_mean(train_pitches[train_pitches.end_of_at_bat],
                                       roll_columns,
                                       group_columns = group_columns,
                                       sort_columns= sort_columns,
                                       columns_to_keep = columns_to_keep,
                                       window = window_size,
                                       min_periods = min_period,
                                       drop_duplicates = False,
                                       suffix = "_batter")

    
    batter_outcome_df = batter_outcome_df.sort_values(['game_date','at_bat_number'])
    batter_outcome_df['dummy'] = 1
    batter_outcome_df['cumulative_at_bats'] = batter_outcome_df.groupby('batter').dummy.transform('cumsum')
    batter_outcome_df.drop(columns='dummy', inplace=True)

    # make a column of regressed performance by throwing in league avg abs till total abs are 700
    num_abs_league_average = (700 - batter_outcome_df.cumulative_at_bats).clip(0)
    for col in batter_outcome_df.columns:
        if '_batter' in col:
    
            league_avg_performance = num_abs_league_average*batter_outcome_df[col].median() 
            batter_performance = batter_outcome_df.cumulative_at_bats*batter_outcome_df[col]
    
            total_abs = num_abs_league_average + batter_outcome_df.cumulative_at_bats
            batter_outcome_df[col+'_regressed'] = (league_avg_performance + batter_performance) /total_abs
    
    # add label column to batter data for ab but count double and triple the same for now
    batter_outcome_df['double_or_triple'] = batter_outcome_df[['double','triple']].max(axis=1)
    
    batter_outcome_df['label'] = np.argmax(
        batter_outcome_df[['strikeout','field_out','walk','single', 'double_or_triple','homerun']],axis=1)

    batter_outcome_columns = [x for x in batter_outcome_df.columns if '_batter' in x]
    return batter_outcome_df, batter_outcome_columns
    
def assemble_event_model_data(pitch_values, train_pitches):
    
    train_pitches = train_pitches[train_pitches.game_date>=pitch_model_train_cutoff]

    train_pitches = train_pitches.sort_values(['game_date','at_bat_number','pitch_number'])

    train_pitches['at_bat_change'] = \
        (train_pitches.groupby(['pitcher','game_date']).at_bat_number.diff().fillna(0)!=0)
    
    train_pitches['pitcher_at_bat_number'] = train_pitches\
        .groupby(['pitcher','game_date']).at_bat_change.transform('cumsum') + 1

    pitch_values = pitch_values.merge(
        train_pitches[train_pitches.end_of_at_bat][
            ['batter','player_name', 'game_date','pitcher_at_bat_number', 'pitcher']],
        how = 'left', 
         on = ['pitcher','player_name', 'pitcher_at_bat_number','game_date'])

    # edit time thru order in pitcher vals
    cols = ['pitcher', 'player_name','game_date','pitcher_at_bat_number','time_thru_the_order','batter']
    
    cols += [x for x in pitch_values.columns if 'whiff' in x or 'called_strike' in x 
             or 'strikeout' in x or 'homerun' in x or 'woba' in x or 'walk' in x or '_RA9' in x]
    
    cols += [ 'is_CU','is_SL','is_FF_2','is_FF_1','is_SI','is_CH']
    pitch_values['time_thru_the_order'] = pitch_values.pitcher_at_bat_number//9 + 1
    pitch_values = pitch_values[cols]
    
    train_pitches.strikeout = np.where(train_pitches.end_of_at_bat, train_pitches.strikeout, np.nan)
    train_pitches.walk = np.where(train_pitches.end_of_at_bat, train_pitches.walk, np.nan)
    train_pitches.homerun = np.where(train_pitches.end_of_at_bat, train_pitches.homerun, np.nan)
    train_pitches.single = np.where(train_pitches.end_of_at_bat, train_pitches.single, np.nan)
    train_pitches.double = np.where(train_pitches.end_of_at_bat, train_pitches.double, np.nan)
    train_pitches.triple = np.where(train_pitches.end_of_at_bat, train_pitches.triple, np.nan)
    train_pitches['field_out'] = \
        1 - train_pitches[['walk','homerun','strikeout','single','double','triple']].max(axis=1)


    train_pitches['pitch_type'] = train_pitches['pitch_type'].astype(str)
    
    train_pitches = train_pitches.sort_values(['game_date','at_bat_number','pitch_number'])
    
    train_pitches['at_bat_change'] = \
        (train_pitches.groupby(['pitcher','game_date']).at_bat_number.diff().fillna(0)!=0)
    
    train_pitches['pitcher_at_bat_number'] = train_pitches\
        .groupby(['pitcher','game_date']).at_bat_change.transform('cumsum') + 1
    
    train_pitches['team_at_bat_number'] = train_pitches\
        .groupby(['inning_topbot','game_date','home_team']).at_bat_change.transform('cumsum') + 1


    # add more state info to the game
    train_pitches['batting_score'] = np.where(
        train_pitches.inning_topbot=='Top', 
        train_pitches.away_score,
        train_pitches.home_score)
    
    train_pitches['batting_team'] = np.where(
        train_pitches.inning_topbot=='Top', 
        train_pitches.away_team,
        train_pitches.home_team)
    
    train_pitches['lineup_slot'] = 1 + (train_pitches.team_at_bat_number-1)%9
    train_pitches['lineup_slot'] = train_pitches.groupby(['batter','game_date'])\
        .lineup_slot.transform('first')

    train_pitches['whiff'] = train_pitches.description.str.contains('swinging_strike')
    
    train_pitches['called_strike'] = train_pitches.description=='called_strike'

    batter_contact_quality, contact_quality_cols = prepare_batter_contact_data(train_pitches)

    batter_outcome_df, batter_outcome_columns = prepare_batter_outcome_data(train_pitches)


    at_bat_cols = [x for x in pitch_values.columns if 'last_9' in x or 'next_3' in x]
    pitcher_meta = ['pitcher','game_date', 'pitcher_at_bat_number', 'batter']
    batter_meta = ['label','pitcher','batter','game_date','pitcher_at_bat_number'] 
    
    summary_stat_cols = [x for x in pitch_values.columns 
                  if x not in at_bat_cols
                  and x not in batter_outcome_columns
                  and x != 'batter' and x != 'pitcher_at_bat_number']
    
    
    
    train_data = pitch_values[at_bat_cols + pitcher_meta ]\
        .merge(batter_outcome_df[batter_outcome_columns + batter_meta],
               how ='left', on = pitcher_meta)
    
    train_data = train_data.merge(
        train_pitches[train_pitches.end_of_at_bat][['inning_topbot','inning','on_1b','on_2b','on_3b',
                 'game_date','pitcher_at_bat_number','pitcher','batter',
                 'batting_team','batting_score','team_at_bat_number',
                 'lineup_slot']].drop_duplicates(),
        how = 'left', on = ['game_date','pitcher_at_bat_number','pitcher','batter'])

    pitch_values = pitch_values.sort_values(['game_date','pitcher_at_bat_number'])
    
    # rolling stats at the start of first ab are what you enter the game with
    summary_stat_before_first_pitch_df = \
        pitch_values[summary_stat_cols].groupby(['pitcher','game_date']).first().reset_index()
    
    train_data = train_data\
        .merge(summary_stat_before_first_pitch_df, how ='left', on = ['pitcher','game_date'])
    
    meta_cols = ['label', 'player_name','pitcher','batter', 'game_date', 'pitcher_at_bat_number'] 
    
    label = 'label'
    ordered_cols = meta_cols+[x for x in train_data.columns if x not in meta_cols]
    train_data=train_data[ordered_cols]

    columns_to_drop = [x for x in train_data.columns if 'empirical' in x 
                       and ('last_9' in x or 'next_3' in x)]
    
    train_data[label] = train_data[label].astype('category')
    
    train_data = train_data.drop(columns = columns_to_drop)
    pred_columns = [x for x in train_data.columns if x!=label and x not in meta_cols]
    pred_columns += ['pitcher_at_bat_number']


    # probably have to do this bc i have to pare down to just a single row per AB
    train_data = train_data.drop_duplicates(meta_cols)
    
    # get rid of na player names
    train_data = train_data\
        .dropna(subset=['player_name'])
    
    train_data = add_state_data(train_data)
    
    train_data = train_data.sort_values(['game_date','team_at_bat_number'])
    train_data = train_data.reset_index()
    
    # get last ab against pitcher and last ab from batter pitcher matchup
    train_data['batter_last_ab'] = train_data.groupby(['pitcher', 'game_date', 'batter']).label.shift(1)    
    train_data['last_ab'] = train_data.groupby(['game_date','pitcher']).label.shift(1)

    ## train_data = train_data.sort_values(['game_date', 'team_at_bat_number'])
    train_data['at_bat_woba'] = 1.2*(
        .55*(train_data.label==2) + .7*(train_data.label==3) +\
        1.0*(train_data.label==4) + 1.65*(train_data.label==5))
    
    
    # these should technically be changed to not include at bat
    train_data['woba_batter'] = train_data.groupby(['batter'])\
        .at_bat_woba.transform(lambda x:x.rolling(window=700, min_periods=300).mean())
    train_data['woba_batter'] = train_data.groupby(['batter']).woba_batter.shift(1)
    
    
    train_data['next_batter_woba'] = train_data\
        .groupby(['game_date','inning_topbot', 'batting_team']).woba_batter.shift(-1)
    
    train_data['woba_pitcher'] = train_data.groupby(['pitcher'])\
        .at_bat_woba.transform(lambda x:x.rolling(window=700, min_periods=300).mean())
    train_data['woba_pitcher'] = train_data.groupby(['pitcher']).woba_pitcher.shift(1)
    
    train_data = train_data.dropna(subset='label')

    if False:
    
        matchup_lookup = setup_batter_pitcher_matchups(train_data, pitches)
        matchup_lookup.to_parquet('matchup_lookup.parquet')
    else:
        matchup_lookup = pd.read_parquet('matchup_lookup.parquet')

    print('shape before', train_data.shape)
    train_data = train_data.dropna(subset=['pitcher','batter'])
    print('shape after', train_data.shape)
    print(train_data.game_date.min(), train_data.game_date.max())
    
    matchup_lookup.pitcher = matchup_lookup.pitcher.astype(int)
    matchup_lookup.batter = matchup_lookup.batter.astype(int)
    train_data.pitcher = train_data.pitcher.astype(int)
    train_data.batter = train_data.batter.astype(int)
    
    matchup_lookup['game_date'] = pd.to_datetime([str(y)+'-'+str(m) 
                                    for y, m in zip(matchup_lookup.year, matchup_lookup.month)])
    train_data = pd.merge_asof(
        train_data,
        matchup_lookup.drop(columns = ['month','year']).sort_values('game_date'), 
        by = ['batter','pitcher'],
        on='game_date',
        direction='backward')
    
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    
    one_hot_encoder.fit(train_data.head(100000)[['last_ab', 'batter_last_ab']].dropna())

    train_data = train_data.drop(columns = [x for x in contact_quality_cols 
                                            if x !='team_at_bat_number' and x in train_data.columns])
    train_data = train_data.drop(columns = [x for x in train_data.columns
                                            if 'matchup' in x and 'today' in x])
    batter_contact_quality = batter_contact_quality.groupby(['game_date','batter']).first().reset_index()
    
    train_data = train_data.merge(
        batter_contact_quality[contact_quality_cols + ['game_date', 'batter']]\
            .drop(columns = 'team_at_bat_number'),
        how = 'left', on = ['game_date', 'batter'])

    train_data['homerun_intensity'] = train_data['homerun_pitcher'].fillna(0) + \
        train_data.homerun_batter.fillna(0)
    train_data['walk_intensity'] = train_data['walk_pitcher'].fillna(0) + \
        train_data.walk_batter.fillna(0)
    train_data['strikeout_intensity'] = train_data['strikeout_pitcher'].fillna(0) + \
        train_data.strikeout_batter.fillna(0)
    
    for column in contact_quality_cols:
        train_data[column] =  pd.to_numeric(train_data[column], errors='coerce')
        
    train_data = train_data.drop(columns =  
                                 [x for x in train_data.columns if 'last_9' in x
                                   and 'pred' in x]+
                                 ['homerun','double', 'single','walk',
                                  'strikeout','linear_weight', 'at_bat_woba'])

    train_data['time_thru_the_order'] = train_data.pitcher_at_bat_number//9 + 1

    xtrain = train_data[train_data.game_date<=event_model_train_cutoff]
    xtest = train_data[train_data.game_date>event_model_train_cutoff]
    train_label = xtrain.label
    test_label = xtest.label
    
    categorical_columns=['last_ab','batter_last_ab']
    meta_columns = ['player_name','pitcher','batter', 'game_date','inning_topbot','batting_team']
    
    pred_columns = [x for x in train_data.columns 
                    if x!='label' and x not in meta_columns and x!='index']
    
    xtrain_enc = one_hot_encode_data(one_hot_encoder, xtrain[pred_columns],
                                 categorical_columns)
    xtest_enc = one_hot_encode_data(one_hot_encoder, xtest[pred_columns], 
                                 categorical_columns)

    x_meta_train = xtrain[meta_columns+['team_at_bat_number', 'inning']]
    x_meta_test = xtest[meta_columns+['team_at_bat_number', 'inning']]

    adaptive_label_data = train_data[['label']+ ['game_date','team_at_bat_number','pitcher','batter']]\
        .merge(
        train_pitches[train_pitches.end_of_at_bat][['game_date','team_at_bat_number','pitcher','batter',
                      'estimated_woba_using_speedangle','hit_distance_sc',
                        'launch_angle','launch_speed',
                       'end_of_at_bat', 'estimated_ba_using_speedangle']].drop_duplicates(),
        how = 'left', on = ['game_date','team_at_bat_number','pitcher','batter'])

    adaptive_label_data = adaptive_label_data\
        .dropna(subset=['label', 'launch_angle']).sort_values('label')
    
    adaptive_label_train = adaptive_label_data[adaptive_label_data.game_date<=event_model_train_cutoff]
    adaptive_label_test = adaptive_label_data[adaptive_label_data.game_date>event_model_train_cutoff]

    data_dict = {'x_train':xtrain_enc, 'y_train':train_label, 'adaptive_label_train':adaptive_label_train,
                     'x_test':xtest_enc, 'y_test':test_label, 'adaptive_label_test':adaptive_label_test,
                     'x_meta_train':x_meta_train, 'x_meta_test':x_meta_test}

    return data_dict



if __name__ == '__main__':


    pitch_values = pd.read_parquet('rolling_pitch_value.parquet')
    train_pitches = pd.read_parquet('data_with_pitch_values.parquet')
    data_dict = assemble_event_model_data(pitch_values, train_pitches)

    with open('data_for_ordinal.pkl','wb') as f:
        pickle.dump(data_dict,f)
