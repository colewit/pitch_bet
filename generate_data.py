
from score_pitches import predict_with_pitch_value_models
from aggregate_pitch_values import aggregate_pitch_values
from assemble_event_model_data import assemble_event_model_data
from engineer_features import engineer_features
from impute_from_neighbors import impute_from_neighbors

from pybaseball import statcast
import pandas as pd
import os
from datetime import timedelta
import pickle

pd.options.display.max_columns = 100
pd.set_option('future.no_silent_downcasting', True)


def update_statcast_year(year, overwrite=False, cache=True):

    year = str(pd.to_datetime('today').year)
    end_date = str(pd.to_datetime('today').date())

    if overwrite:
        start_date = f'{year}-01-01'
        curr_year_data = pd.DataFrame()
    else:
        try:
            curr_year_data = pd.read_parquet(f'pitches_by_year/pitches_{year}.parquet')
            start_date = curr_year_data.game_date.max() + timedelta(1)
            curr_year_data = curr_year_data[curr_year_data.game_date<start_date]
    
        
        except:
            start_date = f'{year}-01-01'
            curr_year_data = pd.DataFrame()

    new_data = statcast(start_dt=start_date, end_dt=end_date)
    curr_year_data = pd.concat([curr_year_data.reset_index(drop=True), new_data])
    curr_year_data.to_parquet(f'pitches_by_season/pitches_{year}.parquet')
    return curr_year_data
    

def pull_statcast_years(year_start, year_end, overwrite=False, cache=True):

    l=[]
    for year in range(year_start, year_end):
        df=update_statcast_year(year, overwrite=False)
        l.append(df)
    return pd.concat(l)

def load_statcast_years(year_start, year_end):
    l=[]
    for year in range(year_start, year_end):
        df=pd.read_parquet(f'pitches_by_season/pitches_{year}.parquet')
        l.append(df)
    return pd.concat(l)

def coerce_numeric(df):
    if 'index' in df.columns:
        df=df.drop(columns='index')
    for col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].dtype == object and df[col].str.endswith('%').any():
                # Handle string percentages
                df[col] = pd.to_numeric(df[col].str.rstrip('%')).div(100)
            else:
                # Coerce other columns to numeric if possible
                df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

if __name__=='__main__':

    run_name = 'pilot'
    if not os.path.isdir(f'intermediate_data_files/{run_name}'):
        os.makedirs(f'intermediate_data_files/{run_name}')
    
    #_ = update_statcast_year(2024)

    #df = load_statcast_years(year_start=2015, year_end=2024)

    #df = engineer_features(df)
    #df.to_parquet(f'intermediate_data_files/{run_name}/engineered_features.parquet')
    #df = pd.read_parquet(f'intermediate_data_files/{run_name}/engineered_features.parquet')
    #data_with_pitch_values = predict_with_pitch_value_models(df)
    #data_with_pitch_values.to_parquet(f'intermediate_data_files/{run_name}/data_with_pitch_values.parquet')
    
    data_with_pitch_values = pd.read_parquet('data_with_pitch_values.parquet')
    
    #rolling_pitch_value = aggregate_pitch_values(data_with_pitch_values)
    #rolling_pitch_value.to_parquet(f'intermediate_data_files/{run_name}/rolling_pitch_value.parquet')

    rolling_pitch_value = pd.read_parquet(f'intermediate_data_files/{run_name}/rolling_pitch_value.parquet')
    
    #data_dict = assemble_event_model_data(rolling_pitch_value, data_with_pitch_values)
    #with open(f'intermediate_data_files/{run_name}/data_for_ordinal.pkl','wb') as f:
    #    pickle.dump(data_dict,f)

    with open(f'intermediate_data_files/{run_name}/data_for_ordinal.pkl','rb') as f:
        data_dict=pickle.load(f)
    train_data = data_dict['x_train']
    train_meta= data_dict['x_meta_train'][['player_name','pitcher','batter','game_date','batting_team']]

    test_data = data_dict['x_test']
    test_meta= data_dict['x_meta_test'][['player_name','pitcher','batter','game_date','batting_team']]
    
    train_data = impute_from_neighbors(train_data, train_meta, data_with_pitch_values)
    test_data = impute_from_neighbors(test_data, test_meta, data_with_pitch_values)

    data_dict['x_train'] = train_data
    data_dict['x_test'] = test_data
    
    with open(f'intermediate_data_files/{run_name}/data_for_ordinal_imputed.pkl','wb') as f:    
        pickle.dump(data_dict,f)


    fg_pitchers = pd.read_csv('pitcher_projections.csv')
    fg_batters = pd.read_csv('batter_projections.csv')


    train_data['Season'] = train_meta.game_date.dt.year
    train_data['pitcher'] = train_meta.pitcher
    train_data['batter'] = train_meta.batter

    test_data['Season'] = test_meta.game_date.dt.year
    test_data['pitcher'] = test_meta.pitcher
    test_data['batter'] = test_meta.batter

    fg_pitchers  = fg_pitchers .rename(columns={'MLBAMID':'pitcher'})
    cols = [x for x in fg_pitchers .columns if 'steamer' in x or 'zip' in x or 'Loc' in x or 'Stf' in x or 'bot' in x]
    
    train_data = train_data.merge(fg_pitchers [cols + ['Season','pitcher']].drop_duplicates(['Season','pitcher']),
                 how = 'left', on = ['Season','pitcher'])
    test_data = test_data.merge(fg_pitchers [cols + ['Season','pitcher']].drop_duplicates(['Season','pitcher']),
                 how = 'left', on = ['Season','pitcher'])

    cols = [x for x in fg_batters.columns if 'zip' in x]

    train_data = train_data.merge(fg_batters [cols + ['Season','batter']].drop_duplicates(['Season','batter']),
                 how = 'left', on = ['Season','batter'])
    test_data = test_data.merge(fg_batters [cols + ['Season','batter']].drop_duplicates(['Season','batter']),
                 how = 'left', on = ['Season','batter'])


    train_data = coerce_numeric(train_data)
    test_data = coerce_numeric(test_data)

    data_dict['x_train'] = train_data.drop(columns = ['batter','pitcher','Season'])
    data_dict['x_test'] = test_data.drop(columns = ['batter','pitcher','Season'])
    with open('data_for_ordinal_fangraphs.pkl','wb') as f:
        pickle.dump(data_dict, f)

