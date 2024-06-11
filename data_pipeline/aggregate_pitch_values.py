
import pandas as pd
import numpy as np


def get_RA_df(data_with_pitch_values, agg=False):
    
    data_with_pitch_values['post_batting_score'] = np.where(data_with_pitch_values.inning_topbot=='Bot', 
                                                    data_with_pitch_values.post_home_score,
                                                     data_with_pitch_values.post_away_score)
    
    data_with_pitch_values['batting_score'] = np.where(data_with_pitch_values.inning_topbot=='Bot', 
                                               data_with_pitch_values.home_score,
                                               data_with_pitch_values.away_score)
    
    RA_df = data_with_pitch_values[data_with_pitch_values.end_of_at_bat].groupby(['pitcher', 'game_date'])\
        .agg({'post_batting_score':'max', 'batting_score':'min', 'out':'sum',
             'inning':'min'}).reset_index()
    
    RA_df['runs'] = RA_df.post_batting_score - RA_df.batting_score
    
    RA_df['RA9'] = (RA_df.runs * (27/RA_df.out.clip(1))).clip(0,9)
    RA_df = RA_df[RA_df.inning==1]
    
    
    RA_df = RA_df[~np.logical_and(RA_df.runs < 2, RA_df.out < 15)]

    # RA range should be between the most elite of levels for any reliever.
    # This may clip values of extreme individual pitches like a Uehara splitter
    # but we are worried about starters for this project and there's
    # not a meaningful amount of starters with a pitch in the long run that'll have a sub 2 RA/9 against
    RA_df = RA_df.groupby('pitcher').agg({'runs':'sum','out':'sum'}).reset_index()
    RA_df = RA_df[RA_df.out>3*60]
    RA_df['RA9'] = (RA_df.runs * (27/RA_df.out.clip(1))).clip(0,9)
    
    RA_df['RA9_percentile'] = RA_df.RA9.rank(pct=True)
    return RA_df


def future_rolling_mean(df, columns_to_roll, group_columns, sort_columns,
                        columns_to_keep, window, min_periods=1, suffix='', drop_duplicates=True):

    df = df.sort_values(sort_columns)
    '''
    df[[x+suffix for x in columns_to_roll]] = df.iloc[::-1].groupby(group_columns, sort=False)[columns_to_roll]\
            .rolling(window, min_periods=min_periods)\
            .mean()\
            .iloc[::-1].reset_index(drop=True)

    '''
    df[[x+suffix for x in columns_to_roll]] = df.groupby(group_columns, sort=False)[columns_to_roll].transform(
        lambda x: x.iloc[::-1].rolling(window=window, min_periods=min_periods).mean().iloc[::-1])
    
    rolled_columns = [x+suffix for x in columns_to_roll]
    
    if not isinstance(group_columns, list):
        group_columns = [group_columns]
    
    df[rolled_columns] = df[rolled_columns + group_columns ].groupby(group_columns).shift(-1)
    
    col_subset = group_columns + sort_columns + columns_to_keep

    print('subset', col_subset)
    print(df, 'before')

    print(rolled_columns, 'are rolled')
    if drop_duplicates:
        df = df.drop_duplicates(col_subset)
    
    return df[col_subset+rolled_columns]
    
def past_rolling_mean(df, columns_to_roll, group_columns, sort_columns,
                      columns_to_keep, window, min_periods=1, suffix='', drop_duplicates=True):

    df = df.sort_values(sort_columns)
    
    if not isinstance(group_columns, list):
        group_columns = [group_columns]
    '''
    df[[x+suffix for x in columns_to_roll]] = df.groupby(group_columns, sort=False)[columns_to_roll]\
            .rolling(window, min_periods=min_periods)\
            .mean().reset_index(drop=True)
    '''
    
    df[[x+suffix for x in columns_to_roll]] = df.groupby(group_columns, sort=False)[columns_to_roll].transform(
        lambda x: x.rolling(window=window, min_periods=min_periods).mean())

    rolled_columns = [x+suffix for x in columns_to_roll]
        
    df[rolled_columns] = df[rolled_columns + group_columns ].groupby(group_columns).shift(1)

    col_subset = group_columns + sort_columns + columns_to_keep
    if drop_duplicates:
        df = df.drop_duplicates(col_subset)

    
    return df[col_subset+rolled_columns]

def convert_columns_to_RA9_scale(pitcher_df, 
                                RA_df, 
                                target_cols, 
                                pivot_column = None,
                                regress_to_mean=True,
                                pivot_alias='',
                                sorting_columns=['player_name', 'game_date']):
    
    for target_col in target_cols:

        pitcher_df['RA9_percentile'] = pitcher_df[target_col].rank(pct=True)
        placeholder = -1
        pitcher_df['RA9_percentile'] = pitcher_df['RA9_percentile'].fillna(placeholder)
        
        pitcher_df = pd.merge_asof(pitcher_df.sort_values('RA9_percentile'),
                      RA_df[['RA9','RA9_percentile']].sort_values('RA9_percentile').drop_duplicates(),
                      on = 'RA9_percentile',
                     direction='nearest')

        indices = pitcher_df.RA9_percentile == placeholder
        pitcher_df.loc[indices, 'RA9_percentile'] = np.nan
        pitcher_df.loc[indices, 'RA9'] = np.nan
        pitcher_df = pitcher_df.rename(columns={"RA9":target_col+'_RA9'})
        

    pitcher_df = pitcher_df.sort_values(sorting_columns)
        
    if pivot_column is not None:
        pivot_df = pitcher_df.pivot_table(index=['pitcher','game_date', 'player_name'],
                                  columns=pivot_column,
                                  values=[target_col+'_RA9' for target_col in target_cols],
                                  aggfunc='mean')
        
        # Rename columns
        
        new_columns = []
        for col in pivot_df.columns.levels[0]:
            for num in pivot_df.columns.levels[1]:
                new_columns.append(f'{col}_{pivot_alias}_{num}')
        
        pivot_df.columns = new_columns
        pivot_df = pivot_df.reset_index()
        return pivot_df
    
    if regress_to_mean:

        
        pitcher_df = pitcher_df.reset_index()
        for target_col in target_cols:
            
            pitcher_df[target_col+'_RA9_regressed'] = \
                (pitcher_df[target_col+'_RA9'] + 4.8 *  pitcher_df['multiplier']) / (1+pitcher_df['multiplier'])

            #pitcher_df = pitcher_df.drop(columns = target_col)
            
    return pitcher_df.reset_index()

def clean_and_convert_to_RA9(value_df, data_with_pitch_values,
                             sorting_columns, pivot_column, pivot_alias,
                             regress_to_mean = False, suffix=''):
    

    rename_dict = {f'dre_above_average_{suffix.strip('_')}'.strip('_'):'empirical_pitch_value',
     f'pred_dre_above_average_{suffix.strip('_')}'.strip('_'):'pred_pitch_value',
     f'pred_dre_above_average_cmd_{suffix.strip('_')}'.strip('_'):'pred_pitch_value_cmd',
     f'pred_dre_above_average_stuff_{suffix.strip('_')}'.strip('_'):'pred_pitch_value_stuff'}

    value_df = value_df.rename(columns = rename_dict) 
    
    agg_RA = pivot_column is None
    RA_df = get_RA_df(data_with_pitch_values, agg=agg_RA)

    pitcher_data = convert_columns_to_RA9_scale(value_df,
                                                RA_df,
                                                target_cols = ['pred_pitch_value',
                                                               'pred_pitch_value_cmd',
                                                               'pred_pitch_value_stuff',
                                                               'empirical_pitch_value'],
                                                pivot_column=pivot_column,
                                                pivot_alias=pivot_alias,
                                                regress_to_mean = regress_to_mean,
                                                sorting_columns = sorting_columns)

    
    rename_dict={x:x.replace('_RA9',f'{suffix}_RA9') for x in ['pred_pitch_value_RA9',
            'pred_pitch_value_cmd_RA9',
            'pred_pitch_value_stuff_RA9',
            'empirical_pitch_value_RA9']}
    
    
    pitcher_data = pitcher_data.rename(columns = rename_dict)

    # setting up cols just so its in the order i want
    cols = ['game_date','pitcher','player_name'] + \
        [x for x in pitcher_data.columns if '_RA9' in x]
    
    cols += [x for x in pitcher_data.columns if x not in cols]
    return pitcher_data[cols]

def get_innings_df(inning_df):
    
    # Sort by game_date and at_bat_number
    inning_df = inning_df.sort_values(['game_date', 'at_bat_number'])

    # Define a rolling window function
    def rolling_sum_last_year(df, window='365D'):
        df = df.set_index('game_date').sort_index()
        df['out_cumsum'] = df['out'].rolling(window, closed='right').sum()
        df['runs_cumsum'] = df['runs'].rolling(window, closed='right').sum()
        return df.reset_index()  # Ensure game_date remains a column
    
    # Apply the rolling window function for each pitcher
    inning_df = inning_df.groupby('pitcher', group_keys=False).apply(rolling_sum_last_year)
    
    # Drop rows where the rolling window might not be fully populated (e.g., first year of data)
    inning_df.dropna(subset=['out_cumsum', 'runs_cumsum'], inplace=True)
    
    # Get the last row per pitcher and game_date
    inning_df = inning_df.groupby(['pitcher', 'game_date']).last().reset_index()
    
    inning_df['innings'] = inning_df['out_cumsum'] / 3
    inning_df['multiplier'] = 180 / inning_df['innings'].clip(0, 180) - 1

    inning_df = inning_df.sort_values('game_date')
    inning_df['innings'] = inning_df.groupby('pitcher').innings.shift(1)
    inning_df['multiplier'] = inning_df.groupby('multiplier').innings.shift(1)

    return inning_df

# Rest of your code using inning_df = g(inning_df.copy())


def aggregate_pitch_values(data_with_pitch_values):

    def fill_with_first_non_na(series):
        try:
            first_non_na = series.dropna().iloc[0]
            return series.fillna(first_non_na)
        except:
            return series
    
    data_with_pitch_values['player_name'] = \
        data_with_pitch_values.groupby('pitcher')['player_name']\
        .transform(fill_with_first_non_na)

    data_with_pitch_values.strikeout = np.where(data_with_pitch_values.end_of_at_bat, data_with_pitch_values.strikeout, np.nan)
    data_with_pitch_values.walk = np.where(data_with_pitch_values.end_of_at_bat, data_with_pitch_values.walk, np.nan)
    data_with_pitch_values.homerun = np.where(data_with_pitch_values.end_of_at_bat, data_with_pitch_values.homerun, np.nan)

    data_with_pitch_values['k_pitch_type_adj'] = data_with_pitch_values['k_pitch_type_adj'].astype(str)
    data_with_pitch_values['pitch_type'] = data_with_pitch_values['pitch_type'].astype(str)
    data_with_pitch_values['season'] = data_with_pitch_values.game_date.dt.year
    data_with_pitch_values = data_with_pitch_values.sort_values('game_date')

    pitch_usage_cols = []
    for pt in data_with_pitch_values.k_pitch_type_adj.unique():
        data_with_pitch_values[f'is_{pt}'] = (data_with_pitch_values.k_pitch_type_adj==pt).astype(int)
        pitch_usage_cols.append(f'is_{pt}')

    data_with_pitch_values['dummy'] = 1
    
    data_with_pitch_values['pitches_this_season'] = data_with_pitch_values.groupby(['pitcher','season']).dummy.transform('cumsum')
    data_with_pitch_values['pitches_this_season_pitch_type'] = data_with_pitch_values\
        .groupby(['pitcher','season', 'k_pitch_type_adj']).dummy.transform('cumsum')


    data_with_pitch_values = data_with_pitch_values.sort_values(['pitcher','game_date','at_bat_number','pitch_number'])
    data_with_pitch_values['at_bat_change'] = \
        (data_with_pitch_values.groupby(['pitcher','game_date']).at_bat_number.diff().fillna(0)!=0)
    
    data_with_pitch_values['pitcher_at_bat_number'] = data_with_pitch_values\
        .groupby(['pitcher','game_date']).at_bat_change.transform('cumsum') + 1

    '''
    first are rate stats for outcomes of a pitcher's at bats over their last 600 batters faced
    '''
    
    roll_columns = ['homerun', 'walk', 'strikeout', 'estimated_woba_using_speedangle']
    
    sort_columns = ['game_date','pitcher_at_bat_number']
    group_columns = ['pitcher']
    columns_to_keep = ['player_name']
    window_size = 600  # Number of batters faced
    min_period = 50

    print('before it is', data_with_pitch_values[data_with_pitch_values.end_of_at_bat][roll_columns])
    outcomes_pitcher_agg = past_rolling_mean(data_with_pitch_values[data_with_pitch_values.end_of_at_bat],
                      roll_columns, group_columns, sort_columns,
                      columns_to_keep, window=window_size, min_periods=min_period, suffix='_pitcher')

    print(outcomes_pitcher_agg.dropna(subset='strikeout_pitcher'), 'is outcome df')
    # get performance over the last 600 at bats up tto day before game. We will get one
    # row per at bat in the previous set up so we keep only the most recent at bat
    outcomes_pitcher_agg = outcomes_pitcher_agg.groupby(['pitcher','game_date']).last().reset_index()

    '''
    now we record the per pitch stat values (as opposed to stats that only come
    at the end of an AB above
    '''
    

    # what i want to add here is rolling forward performance on all days up until today
    # on pred / empirical pitch value of hitters n thru n+ 3 or 9 batters

    # first roll on just next 9 batters for the same game. So take at bat outcomes
    # but do next 9 batters and then aggregate it sorted by game_date
    # by cummean with ['pitcher_at_bat_number']. Then shift back one


    
    roll_columns = ['dre_above_average', 'pred_dre_above_average',
                    'pred_dre_above_average_cmd','pred_dre_above_average_stuff',
                    'delta_run_expectancy','pred_delta_run_expectancy',
                    'pred_delta_run_expectancy_cmd','pred_delta_run_expectancy_stuff',
                    'pred_benchmark']
    
    at_bat_outcomes_df = data_with_pitch_values\
        .groupby(['game_date','pitcher','player_name','batter','pitcher_at_bat_number'])\
        .agg({x:'mean' for x in roll_columns}).reset_index()

    window_size = 3
    min_periods = 1
    suffix = '_next_3_batters'
    group_columns = ['pitcher', 'game_date']
    sort_columns = ['pitcher_at_bat_number']
    columns_to_keep=['player_name']
    rolling_value_next_time_thru_order = \
        future_rolling_mean(at_bat_outcomes_df, columns_to_roll = roll_columns, 
                            group_columns = group_columns, 
                            sort_columns = sort_columns,
                            columns_to_keep = columns_to_keep,
                            window = window_size, min_periods=1, suffix = suffix)
    

    # sort by game date
    roll_columns = [x+suffix for x in roll_columns]

    # now make sure we change it to next 3 batters on average over prior games
    rolling_value_next_time_thru_order = rolling_value_next_time_thru_order.sort_values('game_date')

    # find cumulative mean of next_3_batters columns up to current date
    rolling_value_next_time_thru_order[roll_columns] = \
        rolling_value_next_time_thru_order\
        .groupby(['pitcher','pitcher_at_bat_number'])[roll_columns].transform(lambda x: x.expanding().mean())

    # shift back by one so that the cummean doesnt include current date
    rolling_value_next_time_thru_order[roll_columns] = \
        rolling_value_next_time_thru_order\
        .groupby(['pitcher','pitcher_at_bat_number'])[roll_columns].shift(1)

    # now we want value over the pitcher's last 3,000 pitchers.
    # this will act as along term horizon for a pitchers effectiveness to date
    roll_columns = ['dre_above_average', 'pred_dre_above_average',
                    'pred_dre_above_average_cmd','pred_dre_above_average_stuff',
                    'delta_run_expectancy','pred_delta_run_expectancy',
                    'pred_delta_run_expectancy_cmd','pred_delta_run_expectancy_stuff',
                    'pred_benchmark', 'called_strike', 'whiff'] + pitch_usage_cols
    
    sort_columns = ['game_date', 'pitcher_at_bat_number']
    group_columns = ['pitcher']
    columns_to_keep = ['player_name', 'pitches_this_season']
    window_size = 3000  # Number of pitches thrown
    min_period = 500

    rolling_value_per_pitch = past_rolling_mean(data_with_pitch_values,
                      roll_columns, group_columns, sort_columns,
                      columns_to_keep, window=window_size, min_periods=min_period, suffix='')

    
    rolling_value_per_pitch = rolling_value_per_pitch.rename(
        columns = {'called_strike':'called_strike_pitcher',
                   'whiff':'whiff_pitcher'})
    
    '''
    
    Now we have a df on value by pitch type
    '''
    
    roll_columns = ['called_strike', 'whiff',
                    'homerun','estimated_woba_using_speedangle',
                    'dre_above_average', 'pred_dre_above_average',
                    'pred_dre_above_average_cmd','pred_dre_above_average_stuff']
    
    sort_columns = ['game_date','pitcher_at_bat_number']
    group_columns = ['pitcher', 'k_pitch_type_adj']
    columns_to_keep = ['player_name', 'pitch_type', 'pitches_this_season_pitch_type']
    window_size = 1000  # Number of pitches thrown
    min_period = 100
    
    pitch_df = data_with_pitch_values.copy(deep=True)
    pitch_df['homerun'] = pitch_df.homerun.fillna(0)
    pitch_df = pitch_df[pitch_df.k_pitch_type_adj!='nan']

    rolling_value_by_pitch_type = past_rolling_mean(pitch_df,
                      roll_columns, group_columns, sort_columns,
                      columns_to_keep, window=window_size, min_periods=min_period, suffix='')\
        .drop_duplicates(['game_date','k_pitch_type_adj','pitcher'])

    
    value_by_pitch_type_pivot = rolling_value_by_pitch_type.pivot_table(
                                      index=['pitcher', 'player_name', 'game_date'],
                                      columns='k_pitch_type_adj',
                                      values=['called_strike','whiff','homerun',
                                             'estimated_woba_using_speedangle'],
                                      aggfunc='mean',
                                      fill_value=np.nan).reset_index()
    
    value_by_pitch_type_pivot.columns = ['_'.join(x).strip('_') for x in value_by_pitch_type_pivot.columns]

    inning_df = data_with_pitch_values[data_with_pitch_values.end_of_at_bat].copy(deep=True)

    inning_df = get_innings_df(inning_df)
    
    pitch_use_df = data_with_pitch_values[['pitcher','k_pitch_type_adj','game_date', 'at_bat_number']].copy(deep=True)
    pitch_use_df = pitch_use_df.sort_values(['game_date','at_bat_number'])
    
    pitch_use_df['dummy'] = 1
    pitch_use_df['num_pitches'] = pitch_use_df\
        .groupby(['pitcher', 'k_pitch_type_adj']).dummy.transform('cumsum')
    
    pitch_use_df = pitch_use_df.groupby(['pitcher','game_date','k_pitch_type_adj']).last().reset_index()
    
    pitch_use_df['pitch_multiplier'] = (500 / pitch_use_df.num_pitches).clip(1) - 1
    
    pitcher_value_over_past_season = rolling_value_per_pitch.groupby(['pitcher','game_date'])\
        .agg('last')\
        .merge(inning_df[['pitcher','multiplier', 'game_date']],
               how = 'left', on =['game_date', 'pitcher'])
    
    
    rolling_value_next_time_thru_order = clean_and_convert_to_RA9(rolling_value_next_time_thru_order,
                          data_with_pitch_values,
                          suffix='_next_3_batters',
                          pivot_column=None,
                          pivot_alias=None,
                          sorting_columns = ['pitcher','game_date', 'pitcher_at_bat_number'])


    regressed_pitcher_data = clean_and_convert_to_RA9(pitcher_value_over_past_season,
                          data_with_pitch_values,
                          suffix='',
                          pivot_column=None,
                          pivot_alias=None,
                          regress_to_mean = True,
                          sorting_columns = ['pitcher','game_date', 'pitcher_at_bat_number'])

    rolling_value_by_pitch_type = clean_and_convert_to_RA9(rolling_value_by_pitch_type,
                            data_with_pitch_values,
                            pivot_column='k_pitch_type_adj',
                            pivot_alias='pitch_type',
                            sorting_columns = ['pitcher','game_date', 'pitcher_at_bat_number'])

    all_rolling_pitch_data  = \
        rolling_value_next_time_thru_order.merge(regressed_pitcher_data[['game_date','pitcher'] + \
               [x for x in regressed_pitcher_data.columns
                if x not in rolling_value_next_time_thru_order.columns]],
               how = 'left', on = ['game_date','pitcher'])
    
    all_rolling_pitch_data = all_rolling_pitch_data  \
        .merge(rolling_value_by_pitch_type [['game_date','pitcher'] + \
               [x for x in rolling_value_by_pitch_type.columns
                if x not in all_rolling_pitch_data.columns]],
               how = 'left', on = ['game_date','pitcher'])

    all_rolling_pitch_data = all_rolling_pitch_data.merge(value_by_pitch_type_pivot[
        [x for x in value_by_pitch_type_pivot.columns 
         if x not in all_rolling_pitch_data.columns] + ['pitcher','game_date']], 
        how='left', on = ['pitcher','game_date'])

    all_rolling_pitch_data = all_rolling_pitch_data.merge(outcomes_pitcher_agg[
        [x for x in outcomes_pitcher_agg.columns 
         if x not in all_rolling_pitch_data.columns] + ['pitcher','game_date']], 
        how='left', on = ['pitcher','game_date'])
    
    return all_rolling_pitch_data

if __name__ == '__main__':
    
    data_with_pitch_values = pd.read_parquet('data_with_pitch_values.parquet')
    all_rolling_pitch_data = aggregate_pitch_values(data_with_pitch_values)
