from model_scripts.score_pitches import train_xgboost
import pickle
import pandas as pd
import numpy as np
from config import pitch_model_train_cutoff, event_model_train_cutoff
import os


def prep_data_for_pitcher_pull(train_pitches):

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
        
    data_for_pitcher_pull = train_pitches[train_pitches.end_of_at_bat].copy(deep=True)
    
    data_for_pitcher_pull = data_for_pitcher_pull.sort_values(
        ['game_date','pitcher','pitcher_at_bat_number'])
    
    
    data_for_pitcher_pull['is_start'] = data_for_pitcher_pull \
        .groupby(['pitcher','game_date']).inning.transform('min') == 1
    
    data_for_pitcher_pull['month'] = data_for_pitcher_pull.game_date.dt.month
    data_for_pitcher_pull['day'] = data_for_pitcher_pull.game_date.dt.day
    
    #data_for_pitcher_pull = data_for_pitcher_pull[np.logical_or(
    #    data_for_pitcher_pull.month>4, data_for_pitcher_pull.day > 10)]
    
    data_for_pitcher_pull = data_for_pitcher_pull[data_for_pitcher_pull.is_start]
    
    data_for_pitcher_pull['runs_this_inning'] = data_for_pitcher_pull \
        .groupby(['pitcher','inning', 'game_date']).runs.transform('cumsum')
    
    data_for_pitcher_pull['runs_this_game'] = data_for_pitcher_pull \
        .groupby(['pitcher', 'game_date']).runs.transform('cumsum')
    
    data_for_pitcher_pull['outs_recorded_this_game'] = data_for_pitcher_pull \
        .groupby(['pitcher', 'game_date']).out.transform('cumsum').clip(0,27)
    
    data_for_pitcher_pull['outs_this_inning'] = data_for_pitcher_pull \
        .groupby(['pitcher','inning', 'game_date']).out.transform('cumsum').clip(0,3)
    
    data_for_pitcher_pull['homeruns_this_inning'] = data_for_pitcher_pull \
        .groupby(['pitcher','inning', 'game_date']).homerun.transform('cumsum')
    
    data_for_pitcher_pull['homeruns_this_game'] = data_for_pitcher_pull \
        .groupby(['pitcher','game_date']).homerun.transform('cumsum')
    
    data_for_pitcher_pull['hits_this_inning'] = data_for_pitcher_pull \
        .groupby(['pitcher','inning', 'game_date']).hit.transform('cumsum')
    
    data_for_pitcher_pull['hits_this_game'] = data_for_pitcher_pull \
        .groupby(['pitcher','game_date']).hit.transform('cumsum')
    
    data_for_pitcher_pull['walks_this_game'] = data_for_pitcher_pull \
        .groupby(['pitcher','game_date']).walk.transform('cumsum')
    
    data_for_pitcher_pull['walks_this_inning'] = data_for_pitcher_pull \
        .groupby(['pitcher','inning', 'game_date']).walk.transform('cumsum')
    
    data_for_pitcher_pull['strikeouts_this_game'] = data_for_pitcher_pull \
        .groupby(['pitcher','game_date']).strikeout.transform('cumsum')
    
    data_for_pitcher_pull['strikeouts_this_inning'] = data_for_pitcher_pull \
        .groupby(['pitcher','inning', 'game_date']).strikeout.transform('cumsum')
    
    
    data_for_pitcher_pull = data_for_pitcher_pull[['inning','pitcher','player_name','runs',
                            'homerun','walk','strikeout','hit',
                           'out', 'pitcher_at_bat_number','game_date','batter'] + \
        [x for x in data_for_pitcher_pull if 'this_game' in x or 'this_inning' in x]]
    
    
    data_for_pitcher_pull['total_outs_recorded_by_end_of_game'] = data_for_pitcher_pull\
        .groupby(['pitcher','game_date']).outs_recorded_this_game.transform('max')
    
    df_for_perc = data_for_pitcher_pull.drop_duplicates(['game_date','pitcher']).copy(deep=True)
    
    for perc in [.05, .25,.5, .75, .95]:
        
        cumulative_percentile_by_group = df_for_perc\
            .groupby('pitcher')['total_outs_recorded_by_end_of_game'].expanding().quantile(perc)
        
        # Add the result as a new column
        df_for_perc[f'total_outs_recorded_{int(perc*100)}_perc'] = \
            cumulative_percentile_by_group.reset_index(level=0, drop=True)
        
    perc_columns = [x for x in df_for_perc.columns if '_perc' in x]
    data_for_pitcher_pull = data_for_pitcher_pull\
        .merge(df_for_perc[perc_columns + ['game_date','pitcher']],
                                how = 'left', on=['game_date','pitcher'])
    
    data_for_pitcher_pull['max_outs_recorded_by_end_of_game'] = data_for_pitcher_pull\
        .groupby(['pitcher']).total_outs_recorded_by_end_of_game.transform(np.maximum.accumulate)
    
    data_for_pitcher_pull['min_outs_recorded_by_end_of_game'] = data_for_pitcher_pull\
        .groupby(['pitcher']).total_outs_recorded_by_end_of_game.transform(np.minimum.accumulate)
    
    
    cumulative_mean_func = lambda x: x.cumsum() / range(1, len(x) + 1)
    data_for_pitcher_pull['mean_recorded_by_end_of_game'] = data_for_pitcher_pull \
        .groupby(['pitcher','inning']).total_outs_recorded_by_end_of_game.transform(cumulative_mean_func)
    
    
    data_for_pitcher_pull.drop(columns='total_outs_recorded_by_end_of_game', inplace=True)
    data_for_pitcher_pull['pulled'] = data_for_pitcher_pull.groupby([
        'pitcher','game_date']).pitcher_at_bat_number.transform(lambda x: x == max(x))

    return data_for_pitcher_pull

def train_pitcher_pull_model(train_pitches):
    

    data_for_pitcher_pull = prep_data_for_pitcher_pull(train_pitches)
    meta = ['pitcher','player_name','game_date','batter']
    target = 'pulled'
    
    pull_pitcher_cols = [x for x in data_for_pitcher_pull.columns if x not in meta+[target]]


    data_for_pitcher_pull_train = \
        data_for_pitcher_pull[
            data_for_pitcher_pull.game_date.between(pitch_model_train_cutoff, event_model_train_cutoff)]
    
    xg_model_pull = train_xgboost(data_for_pitcher_pull_train,
                  target_column='pulled',
                  pred_columns=pull_pitcher_cols,
                  categorical_columns=None,
                  multiclass=False)

    return xg_model_pull, data_for_pitcher_pull

if __name__ == '__main__':

    run_name = 'pilot'
    train_pitches = pd.read_parquet(f'intermediate_data_files/{run_name}/data_with_pitch_values.parquet')

    model, data_for_pitcher_pull = train_pitcher_pull_model(train_pitches)
    data_for_pitcher_pull.to_parquet(f'intermediate_data_files/{run_name}/data_for_pitcher_pull.parquet')
    
    if not os.path.isdir(f'pull_pitcher_models/{run_name}/'):
        os.makedirs(f'pull_pitcher_models/{run_name}/')
    with open(f'pull_pitcher_models/{run_name}/model.pkl','wb') as f:
        pickle.dump(model, f)

    