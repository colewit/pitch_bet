

from config import xgboost_pred_columns_stuff_only
from engineer_features import predict_xgboost
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_RA_df(all_pitches_df, agg=False):
    
    all_pitches_df['post_batting_score'] = np.where(all_pitches_df.inning_topbot=='Bot', 
                                                    all_pitches_df.post_home_score,
                                                     all_pitches_df.post_away_score)
    
    all_pitches_df['batting_score'] = np.where(all_pitches_df.inning_topbot=='Bot', 
                                               all_pitches_df.home_score,
                                               all_pitches_df.away_score)
    
    RA_df = all_pitches_df[all_pitches_df.end_of_at_bat].groupby(['pitcher', 'game_date'])\
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



def build_rolling_df(df, 
                     roll_columns,
                     sort_columns, 
                     group_columns,
                     merge_columns, 
                     columns_to_keep,
                     window_size,
                     min_period,
                     roll_index = 'game_date',
                     drop_duplicates = True,
                     ascending=True,
                     suffix = ""):
    
    df = df.sort_values(sort_columns, ascending = ascending)

    def roll(df, meta_columns):

        df_tmp = df[roll_columns].rolling(window=window_size, min_periods = min_period).mean()
        if 'game_date' in df.columns:
            df_tmp['game_date'] = df.game_date

        for x in meta_columns:
            df_tmp[x] = df[x]

        return df_tmp

    # Group by pitcher and calculate rolling mean for specified columns

    meta = [ x for x in ['game_date', 'pitcher_at_bat_number', 'player_name'] if x not in group_columns]
    rolled_data = df \
        .groupby(group_columns, sort=False)[roll_columns+meta]\
        .apply(roll, meta_columns = meta).reset_index()
    

    rename_keys = {x:x+suffix for x in roll_columns}

    rolled_data = rolled_data.rename(columns=rename_keys)
    rolled_data = rolled_data.groupby(merge_columns).agg('last').reset_index()

    rolled_data = df[merge_columns]\
        .merge(
               rolled_data[list(rename_keys.values()) + merge_columns],
        how = 'left', on = merge_columns)

    col_subset = list(set(merge_columns + columns_to_keep + sort_columns))

    by_cols = [x for x in merge_columns if x != roll_index]

    rolled_data = rolled_data.sort_values(roll_index)
    df = df.sort_values(roll_index)

    if drop_duplicates:
        df = df[col_subset].drop_duplicates()
    else:
        df = df[col_subset]

    # if it is not a time column make sure they are both floats
    if not pd.api.types.is_datetime64_any_dtype(df[roll_index]):
        rolled_data[roll_index] = rolled_data[roll_index].astype(float)
        df[roll_index] = df[roll_index].astype(float)

    
    
    rolled_data = pd.merge_asof(df,
                              rolled_data.dropna(subset=roll_index), 
                              by=by_cols, on=roll_index, direction='backward')

    # move roll columns back by one roll index so that it is the rolled column up to
    # but not including the current roll index. For example if ERA rolled on game date,
    # then you want rolling era up to but not including the game we are evaluating

    rolled_data[list(rename_keys.values())] = \
            rolled_data.groupby(group_columns)[list(rename_keys.values())].shift(1)

    return rolled_data
    
def plot_pdp(xg_model, df, name, date, pred_columns, categorical_columns,
             cols_to_check_against, pitch_type=None):



    if pitch_type is not None:
        df=df[df.pitch_type==pitch_type]

    dfp = df[np.logical_and(df.player_name==name,
                                   df.game_date<=date)].sample(1)

    df = df[df.p_throws == dfp.p_throws.iloc[-1]]
    batter_d = {}
    for x in cols_to_check_against:
        batter_d[x] = np.nanpercentile(df[x], range(5,105,10))

    rep_batter = pd.DataFrame(batter_d)

    dfp = pd.concat([dfp]*len(rep_batter))

    for col in cols_to_check_against:
        df_cp=dfp.copy(deep=True)

        df_cp[col] = rep_batter[col].values

        
        preds = predict_xgboost(model, 
                                df_cp, 
                                pred_columns,
                                categorical_columns)
    
        print('Showing PDP for column', col)
        plt.plot(df_cp[col], preds.reshape(-1))
        plt.show()

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

def clean_and_convert_to_RA9(value_df, all_pitches_df,
                             sorting_columns, pivot_column, pivot_alias,
                             regress_to_mean = False, suffix=''):
    

    rename_dict = {f'dre_above_average_{suffix.strip('_')}'.strip('_'):'empirical_pitch_value',
     f'pred_dre_above_average_{suffix.strip('_')}'.strip('_'):'pred_pitch_value',
     f'pred_dre_above_average_cmd_{suffix.strip('_')}'.strip('_'):'pred_pitch_value_cmd',
     f'pred_dre_above_average_stuff_{suffix.strip('_')}'.strip('_'):'pred_pitch_value_stuff'}

    value_df = value_df.rename(columns = rename_dict) 
    
    agg_RA = pivot_column is None
    RA_df = get_RA_df(all_pitches_df, agg=agg_RA)

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



if __name__ == '__main__':

    def fill_with_first_non_na(series):
        try:
            first_non_na = series.dropna().iloc[0]
            return series.fillna(first_non_na)
        except:
            return series
        

    
    all_pitches_df = pd.read_parquet('df_current_season.parquet')
    all_pitches_df['player_name'] = \
        all_pitches_df.groupby('pitcher')['player_name']\
        .transform(fill_with_first_non_na)

    all_pitches_df.strikeout = np.where(all_pitches_df.end_of_at_bat, all_pitches_df.strikeout, np.nan)
    all_pitches_df.walk = np.where(all_pitches_df.end_of_at_bat, all_pitches_df.walk, np.nan)
    all_pitches_df.homerun = np.where(all_pitches_df.end_of_at_bat, all_pitches_df.homerun, np.nan)

    all_pitches_df['k_pitch_type_adj'] = all_pitches_df['k_pitch_type_adj'].astype(str)
    all_pitches_df['pitch_type'] = all_pitches_df['pitch_type'].astype(str)
    all_pitches_df['season'] = all_pitches_df.game_date.dt.year
    all_pitches_df = all_pitches_df.sort_values('game_date')

    pitch_usage_cols = []
    for pt in all_pitches_df.k_pitch_type_adj.unique():
        all_pitches_df[f'is_{pt}'] = (all_pitches_df.k_pitch_type_adj==pt).astype(int)
        pitch_usage_cols.append(f'is_{pt}')

    print(all_pitches_df[pitch_usage_cols])
    all_pitches_df['dummy'] = 1
    
    all_pitches_df['pitches_this_season'] = all_pitches_df.groupby(['pitcher','season']).dummy.transform('cumsum')
    all_pitches_df['pitches_this_season_pitch_type'] = all_pitches_df\
        .groupby(['pitcher','season', 'k_pitch_type_adj']).dummy.transform('cumsum')


    all_pitches_df = all_pitches_df.sort_values(['pitcher','game_date','at_bat_number','pitch_number'])
    all_pitches_df['at_bat_change'] = \
        (all_pitches_df.groupby(['pitcher','game_date']).at_bat_number.diff().fillna(0)!=0)
    
    all_pitches_df['pitcher_at_bat_number'] = all_pitches_df\
        .groupby(['pitcher','game_date']).at_bat_change.transform('cumsum') + 1

    print(all_pitches_df[np.logical_and(
        all_pitches_df.player_name=='Iglesias, Raisel', all_pitches_df.end_of_at_bat)].shape, 'is shape')


    '''
    first are rate stats for outcomes of a pitcher's at bats over their last 600 batters faced
    '''
    
    roll_columns = ['homerun', 'walk', 'strikeout', 'estimated_woba_using_speedangle']
    
    sort_columns = ['game_date','pitcher','pitcher_at_bat_number']
    group_columns = ['pitcher']
    merge_columns = ['game_date','pitcher']
    columns_to_keep = ['player_name']
    window_size = 600  # Number of batters faced
    min_period = 50
    roll_index = 'game_date'
    outcomes_pitcher_agg = build_rolling_df(all_pitches_df[all_pitches_df.end_of_at_bat], 
                                            roll_columns,
                                            sort_columns,
                                            group_columns,
                                            merge_columns,
                                            window_size=window_size,
                                            min_period=min_period,
                                            roll_index = roll_index,
                                            columns_to_keep=columns_to_keep,
                                            suffix = '_pitcher')

    # get performance over the last 600 at bats up tto day before game. We will get one
    # row per at bat in the previous set up so we keep only the most recent at bat
    outcomes_pitcher_agg = outcomes_pitcher_agg.groupby(['pitcher','game_date']).last().reset_index()

    '''
    now we record the per pitch stat values (as opposed to stats that only come
    at the end of an AB above
    '''
    
    roll_columns = ['dre_above_average', 'pred_dre_above_average',
                    'pred_dre_above_average_cmd','pred_dre_above_average_stuff',
                    'delta_run_expectancy','pred_delta_run_expectancy',
                    'pred_delta_run_expectancy_cmd','pred_delta_run_expectancy_stuff',
                    'pred_benchmark']
    
    sort_columns = ['game_date','pitcher', 'pitcher_at_bat_number']
    group_columns = ['pitcher', 'game_date']
    merge_columns = ['game_date','pitcher','pitcher_at_bat_number']
    columns_to_keep = ['player_name', 'time_thru_the_order', 'batter']
    window_size = 9  # Number of pitches thrown
    min_period = 3
    roll_index = 'pitcher_at_bat_number'
    
    
    
    at_bat_outcomes_df = all_pitches_df\
        .groupby(['game_date','pitcher','player_name','batter',
                  'time_thru_the_order','pitcher_at_bat_number'])\
        .agg({x:'mean' for x in roll_columns}).reset_index()
    
    rolling_value_last_time_thru_order = build_rolling_df(at_bat_outcomes_df,
                                                          roll_columns,
                                                          sort_columns,
                                                          group_columns,
                                                          merge_columns,
                                                          window_size=window_size,
                                                          columns_to_keep=columns_to_keep,
                                                          min_period=min_period,
                                                          roll_index = 'pitcher_at_bat_number',
                                                          suffix = '_last_9_batters')
    

    # what i want to add here is rolling forward performance on all days up until today
    # on pred / empirical pitch value of hitters n thru n+ 3 or 9 batters

    # first roll on just next 9 batters for the same game. So take at bat outcomes
    # but do next 9 batters and then aggregate it sorted by game_date
    # by cummean with ['pitcher_at_bat_number']. Then shift back one

    window_size = 3
    suffix = '_next_3_batters'
    rolling_value_next_time_thru_order = build_rolling_df(at_bat_outcomes_df, 
                                                          roll_columns,
                                                          sort_columns, 
                                                          group_columns, 
                                                          merge_columns,
                                                          window_size=window_size,
                                                          columns_to_keep=columns_to_keep,
                                                          min_period=min_period,
                                                          ascending = True,
                                                          roll_index = 'pitcher_at_bat_number',
                                                          suffix = suffix)

    # sort by game date
    roll_columns = [x+suffix for x in roll_columns]
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
    
    sort_columns = ['game_date','pitcher', 'pitcher_at_bat_number']
    group_columns = ['pitcher']
    merge_columns = ['game_date','pitcher']
    columns_to_keep = ['player_name', 'pitches_this_season', 'time_thru_the_order','pitcher_at_bat_number']
    window_size = 3000  # Number of pitches thrown
    min_period = 500
    
    rolling_value_per_pitch = build_rolling_df(all_pitches_df,
                                               roll_columns,
                                               sort_columns, 
                                               group_columns, 
                                               merge_columns,
                                               window_size=window_size,
                                               columns_to_keep=columns_to_keep,
                                               min_period=min_period,
                                               suffix = '')

    print(rolling_value_per_pitch[pitch_usage_cols])
    
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
    
    sort_columns = ['game_date','pitcher', 'pitcher_at_bat_number']
    group_columns = ['pitcher', 'k_pitch_type_adj']
    merge_columns = ['game_date','pitcher', 'k_pitch_type_adj']
    columns_to_keep = ['player_name', 'pitch_type', 'pitches_this_season_pitch_type']
    window_size = 1000  # Number of pitches thrown
    min_period = 200
    
    pitch_df = all_pitches_df.copy(deep=True)
    pitch_df['homerun'] = pitch_df.homerun.fillna(0)
    pitch_df = pitch_df[pitch_df.k_pitch_type_adj!='nan']
    
    rolling_value_by_pitch_type = build_rolling_df(pitch_df, 
                                       roll_columns,
                                       sort_columns, 
                                       group_columns=group_columns, 
                                       merge_columns=merge_columns,
                                       window_size = window_size,
                                       min_period=min_period,
                                       columns_to_keep=columns_to_keep,
                                       suffix = '')\
        .drop_duplicates(['game_date','k_pitch_type_adj','pitcher'])


    
    value_by_pitch_type_pivot = rolling_value_by_pitch_type.pivot_table(
                                      index=['pitcher', 'player_name', 'game_date'],
                                      columns='k_pitch_type_adj',
                                      values=['called_strike','whiff','homerun',
                                             'estimated_woba_using_speedangle'],
                                      aggfunc='mean',
                                      fill_value=np.nan).reset_index()
    
    value_by_pitch_type_pivot.columns = ['_'.join(x).strip('_') for x in value_by_pitch_type_pivot.columns]

    inning_df = all_pitches_df[all_pitches_df.end_of_at_bat].copy(deep=True)
    
    inning_df = inning_df.sort_values(['game_date','at_bat_number'])
    inning_df['out'] = inning_df.groupby('pitcher').out.transform('cumsum')
    inning_df['runs'] = inning_df.groupby('pitcher').runs.transform('cumsum')
    inning_df = inning_df.groupby(['pitcher','game_date']).last().reset_index()
    
    
    inning_df['innings'] = inning_df.out/3
    inning_df['multiplier'] = 180 / inning_df.innings.clip(0,180) - 1
    
    pitch_use_df = all_pitches_df[['pitcher','k_pitch_type_adj','game_date', 'at_bat_number']].copy(deep=True)
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
    
    
    rolling_value_last_time_thru_order = clean_and_convert_to_RA9(rolling_value_last_time_thru_order,
                          all_pitches_df,
                          suffix='_last_9_batters',
                          pivot_column=None,
                          pivot_alias=None,
                          sorting_columns = ['pitcher','game_date', 'pitcher_at_bat_number'])

    rolling_value_next_time_thru_order = clean_and_convert_to_RA9(rolling_value_next_time_thru_order,
                          all_pitches_df,
                          suffix='_next_3_batters',
                          pivot_column=None,
                          pivot_alias=None,
                          sorting_columns = ['pitcher','game_date', 'pitcher_at_bat_number'])


    regressed_pitcher_data = clean_and_convert_to_RA9(pitcher_value_over_past_season,
                          all_pitches_df,
                          suffix='',
                          pivot_column=None,
                          pivot_alias=None,
                          regress_to_mean = True,
                          sorting_columns = ['pitcher','game_date', 'pitcher_at_bat_number'])

    rolling_value_by_pitch_type = clean_and_convert_to_RA9(rolling_value_by_pitch_type,
                            all_pitches_df,
                            pivot_column='k_pitch_type_adj',
                            pivot_alias='pitch_type',
                            sorting_columns = ['pitcher','game_date', 'pitcher_at_bat_number'])

    engineered_pitch_data = rolling_value_last_time_thru_order \
        .merge(rolling_value_by_pitch_type[['game_date','pitcher'] + \
               [x for x in rolling_value_by_pitch_type.columns if '_RA9' in x]],
               how = 'left', on = ['game_date','pitcher'])

    
    engineered_pitch_data = engineered_pitch_data \
        .merge(regressed_pitcher_data[['game_date','pitcher'] + \
               [x for x in regressed_pitcher_data.columns
                if x not in engineered_pitch_data.columns]],
               how = 'left', on = ['game_date','pitcher'])

    engineered_pitch_data = engineered_pitch_data \
        .merge(rolling_value_next_time_thru_order[['game_date','pitcher','pitcher_at_bat_number'] + \
               [x for x in rolling_value_next_time_thru_order.columns if '_RA9' in x]],
               how = 'left', on = ['game_date','pitcher','pitcher_at_bat_number'])

    engineered_pitch_data = engineered_pitch_data.merge(value_by_pitch_type_pivot[
        [x for x in value_by_pitch_type_pivot.columns 
         if x not in engineered_pitch_data.columns] + ['pitcher','game_date']], 
        how='left', on = ['pitcher','game_date'])

    engineered_pitch_data = engineered_pitch_data.merge(outcomes_pitcher_agg[
        [x for x in outcomes_pitcher_agg.columns 
         if x not in engineered_pitch_data.columns] + ['pitcher','game_date']], 
        how='left', on = ['pitcher','game_date'])

    
    engineered_pitch_data.to_parquet('train_data_pitch_values.parquet')

