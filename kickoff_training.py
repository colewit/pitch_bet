
from model_scripts.adaptive_labels import fit_model_for_adaptive_labels
from model_scripts.cheng_booster import run_grid_search
from model_scripts.score_pitches import predict_with_pitch_value_models, train_pitch_value_models

from data_pipeline.aggregate_pitch_values import aggregate_pitch_values
from data_pipeline.assemble_event_model_data import assemble_event_model_data
from data_pipeline.engineer_features import engineer_features
from data_pipeline.impute_from_neighbors import impute_from_neighbors

from config import event_model_train_cutoff

from pybaseball import statcast
import pandas as pd
import os
from datetime import timedelta
import numpy as np
import pickle


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
    for year in range(year_start, year_end+1):
        df=update_statcast_year(year, overwrite=False)
        l.append(df)
    return pd.concat(l)

def load_statcast_years(year_start, year_end):
    l=[]
    for year in range(year_start, year_end+1):
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
    '''
    if False:
        df = load_statcast_years(year_start=2015, year_end=2024)
    
    
        df, k_pitch_type_map = engineer_features(df,
                                                 pitch_type_pipeline_path=f'pitch_type_models/{run_name}', 
                                                 fit_pipeline=True)
        
        k_pitch_type_map.to_parquet(f'intermediate_data_files/{run_name}/k_pitch_type_map.parquet')
        df.to_parquet(f'intermediate_data_files/{run_name}/engineered_features.parquet')
    
        
        df = pd.read_parquet(f'intermediate_data_files/{run_name}/engineered_features.parquet')
        df = df[df.game_date<='2024-01-01']
    
        if not os.path.isdir(f'pitch_value_models/{run_name}'):
            os.makedirs(f'pitch_value_models/{run_name}')
            
        train_pitch_value_models(df, path=f'pitch_value_models/{run_name}')
        data_with_pitch_values = predict_with_pitch_value_models(df, path=f'pitch_value_models/{run_name}')
        data_with_pitch_values.to_parquet(f'intermediate_data_files/{run_name}/data_with_pitch_values.parquet')

    data_with_pitch_values=pd.read_parquet(f'intermediate_data_files/{run_name}/data_with_pitch_values.parquet')
    rolling_pitch_value = aggregate_pitch_values(data_with_pitch_values)
    rolling_pitch_value.to_parquet(f'intermediate_data_files/{run_name}/rolling_pitch_value.parquet')
    
    
    #rolling_pitch_value=pd.read_parquet(f'intermediate_data_files/{run_name}/rolling_pitch_value.parquet')

    print('assembling event model data')
    data_dict = assemble_event_model_data(rolling_pitch_value, data_with_pitch_values)
    
    with open(f'intermediate_data_files/{run_name}/data_for_ordinal.pkl','wb') as f:
        pickle.dump(data_dict,f)

    train_data = data_dict['X']
    train_meta= data_dict['X_meta'][['player_name','pitcher','batter','game_date','batting_team']]
    

    if False:
        with open(f'intermediate_data_files/{run_name}/data_for_ordinal.pkl','rb') as f:
            data_dict = pickle.load(f)
    
        train_data = data_dict['X']
        train_meta= data_dict['X_meta'][['player_name','pitcher','batter','game_date','batting_team']]
        data_with_pitch_values=pd.read_parquet(f'intermediate_data_files/{run_name}/data_with_pitch_values.parquet')


    print('imputing from neighbors')
    train_data = impute_from_neighbors(train_data, train_meta, data_with_pitch_values)

    data_dict['X'] = train_data

    with open(f'intermediate_data_files/{run_name}/data_for_ordinal_imputed.pkl','wb') as f:    
        pickle.dump(data_dict,f)

    if False:
        with open(f'intermediate_data_files/{run_name}/data_for_ordinal_imputed.pkl','rb') as f:    
            data_dict = pickle.load(f)

        train_data = data_dict['X']
        train_meta= data_dict['X_meta'][['player_name','pitcher','batter','game_date','batting_team']]

    print('adding fangraphs data')
    fg_pitchers = pd.read_csv('pitcher_projections.csv')
    fg_batters = pd.read_csv('batter_projections.csv')


    train_data['Season'] = train_meta.game_date.dt.year
    train_data['pitcher'] = train_meta.pitcher
    train_data['batter'] = train_meta.batter

    fg_pitchers  = fg_pitchers .rename(columns={'MLBAMID':'pitcher'})
    cols = [x for x in fg_pitchers .columns if 'steamer' in x or 'zip' in x or 'Loc' in x or 'Stf' in x or 'bot' in x]
    
    train_data = train_data.merge(fg_pitchers [cols + ['Season','pitcher']].drop_duplicates(['Season','pitcher']),
                 how = 'left', on = ['Season','pitcher'])

    cols = [x for x in fg_batters.columns if 'zip' in x]

    train_data = train_data.merge(fg_batters [cols + ['Season','batter']].drop_duplicates(['Season','batter']),
                 how = 'left', on = ['Season','batter'])

    train_data = coerce_numeric(train_data)

    data_dict['X'] = train_data.drop(columns = ['batter','pitcher','Season'])

    with open(f'intermediate_data_files/{run_name}/data_for_ordinal_fangraphs.pkl','wb') as f:
        pickle.dump(data_dict, f)
    '''
    if True:
        with open(f'intermediate_data_files/{run_name}/data_for_ordinal_fangraphs.pkl', 'rb') as f:
            data_dict = pickle.load(f)

    print('training adaptive label model')
    if not os.path.isdir(f'adaptive_label_models/{run_name}/'):
        os.makedirs(f'adaptive_label_models/{run_name}/')

    adaptive_label_data = data_dict['adaptive_label_data'].dropna()
    adaptive_label_train_data = adaptive_label_data[adaptive_label_data.game_date<=event_model_train_cutoff]
    
    # aggregate the data
    adaptive_label_model_pipeline = fit_model_for_adaptive_labels(adaptive_label_train_data)
    
    # Save the fitted model to a file
    with open(f'adaptive_label_models/{run_name}/model.pkl', 'wb') as f:
        pickle.dump(adaptive_label_model_pipeline, f)


    if not os.path.isdir(f'booster_event_models/{run_name}'):
        os.makedirs(f'booster_event_models/{run_name}')

    print('running grid search for event model')
    run_grid_search(data_dict,
                        train_cutoff=event_model_train_cutoff,
                        boost_rounds=300,
                        max_depth=4,
                        patience=10,
                        cheng_labels=False, 
                        adaptive_labels=True,
                        corn_loss=True,
                        model_folder = f'booster_event_models/{run_name}',
                        adaptive_label_model_pipeline=adaptive_label_model_pipeline)

