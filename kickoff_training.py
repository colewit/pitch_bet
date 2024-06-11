
from adaptive_labels import fit_model_for_adaptive_labels
from cheng_booster import run_grid_search
import pickle

from score_pitches import predict_with_pitch_value_models, train_pitch_value_models
from aggregate_pitch_values import aggregate_pitch_values
from assemble_event_model_data import assemble_event_model_data
from engineer_features import engineer_features
from impute_from_neighbors import impute_from_neighbors

from pybaseball import statcast
import pandas as pd
import os
from datetime import timedelta
import numpy as np

if __name__=='__main__':


    run_name = 'pilot'
    
    #df = load_statcast_years(year_start=2015, year_end=2024)
    #df = engineer_features(df)
    #df.to_parquet(f'intermediate_data_files/{run_name}/engineered_features.parquet')
    
    df = pd.read_parquet(f'intermediate_data_files/{run_name}/engineered_features.parquet')
    df = df[df.game_date<='2024-01-01']

    if not os.path.isdir(f'pitch_value_models/{run_name}'):
        os.makedirs(f'pitch_value_models/{run_name}')
        
    data_with_pitch_values = train_pitch_value_models(df, path=f'pitch_value_models/{run_name}')
    data_with_pitch_values.to_parquet(f'intermediate_data_files/{run_name}/data_with_pitch_values.parquet')
    
    rolling_pitch_value = aggregate_pitch_values(data_with_pitch_values)
    rolling_pitch_value.to_parquet(f'intermediate_data_files/{run_name}/rolling_pitch_value.parquet')
    
    data_dict = assemble_event_model_data(rolling_pitch_value, data_with_pitch_values)
    
    with open(f'intermediate_data_files/{run_name}/data_for_ordinal.pkl','wb') as f:
        pickle.dump(data_dict,f)

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
        
    #with open('data_for_ordinal_fangraphs.pkl', 'rb') as f:
    #    data_dict = pickle.load(f)
    if not os.path.isdir(f'adaptive_label_models/{run_name}/'):
        os.makedirs(f'adaptive_label_models/{run_name}/')

    if train_adaptive_label_model:
        train_data = data_dict['adaptive_label_train'].dropna()
        # aggregate the data
        adaptive_label_model_pipeline = fit_model_for_adaptive_labels(train_data)
        
        # Save the fitted model to a file
        with open(f'adaptive_label_models/{run_name}/model.pkl', 'wb') as f:
            pickle.dump(adaptive_label_model_pipeline, f)
    #else:
    #    with open('label_models/model.pkl', 'rb') as f:
    #        adaptive_label_model_pipeline = pickle.load(f)

    if not os.path.isdir(f'booster_event_models/{run_name}'):
        os.makedirs(f'booster_event_models/{run_name}')

    run_grid_search(data_dict,
                        boost_rounds=300,
                        max_depth=4,
                        patience=10,
                        cheng_labels=False, 
                        adaptive_labels=True,
                        corn_loss=True,
                        model_folder = f'booster_event_models/{run_name}',
                        adaptive_label_model_pipeline=adaptive_label_model_pipeline)

