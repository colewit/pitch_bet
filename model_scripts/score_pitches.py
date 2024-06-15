import pandas as pd
import numpy as np
import xgboost as xgb

import pickle
from config import xgboost_target_column, xgboost_categorical_columns, xgboost_pred_columns
from config import xgboost_ball_strike_pred_columns
from config import xgboost_pred_columns_stuff_only, xgboost_pred_columns_cmd_only, xgboost_benchmark_columns
from config import pitch_model_train_cutoff
from config import ball_strike_model_train_cutoff

def grid_search(X_train, y_train, X_test, y_test, classification=False, multiclass=False):

    # Initialize list to store CV results

    xgb_f = xgb.XGBClassifier if classification else xgb.XGBRegressor

    if multiclass:
        
        objective = 'multi:softprob' 
        eval_metric = 'mlogloss'
    elif classification:
        objective = 'binary:logistic'
        eval_metric = 'auc'
    else:
        objective = 'reg:squarederror'
        eval_metric = 'rmse'

    
    # Loop over parameter combinations
    if not classification:
        y_train = y_train.astype(float) 
    
    best_loss = np.inf
    best_model = None
    for max_depth in [5,7,10]:
        for n_estimators in [100,200]:
        # Initialize XGBoost classifier with logistic objective
            xgb_model = xgb_f(objective=objective,
                              n_estimators=n_estimators, 
                              max_depth=max_depth,
                              early_stopping_rounds=10,
                              eval_metric=eval_metric)

            eval_set = [(X_test, y_test)]
            xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose = 0)

            if xgb_model.best_score < best_loss:
                best_model = xgb_model
                best_loss = xgb_model.best_score
        
            # Delete the model to free up memory
            del xgb_model

    return best_model


def train_xgboost(data, target_column, pred_columns, 
                  categorical_columns=None,
                  test_prop=.5, 
                  classification = True,
                  multiclass = False):

    data = data.copy(deep=True)
  
    data = data[pred_columns+[target_column]]

    if categorical_columns is not None:
        data = pd.get_dummies(data, columns=categorical_columns)

    data = data[[x for x in data.columns if '_nan'!= x[-4:] and '_None' != x[-5:]]]
    # Assuming df is your DataFrame
    # First, identify columns with object dtype
    object_columns = data.select_dtypes(include=['object']).columns

    # Convert object columns to numeric
    data[object_columns] = data[object_columns].apply(pd.to_numeric, errors='coerce')
    
    # Now, all object columns have been converted to numeric types (integers)

    test_N = int(data.shape[0]*test_prop)
    test_indices = np.random.choice(data.shape[0], test_N)
    train_indices = np.setdiff1d(range(data.shape[0]), test_indices)
    
    train_df = data.iloc[train_indices]
    test_df = data.iloc[test_indices]
 
    xg_model = grid_search(
                     train_df.drop(columns = target_column),
                     train_df[target_column],
                     test_df.drop(columns = target_column),
                     test_df[target_column],
                     classification=classification,
                     multiclass = multiclass)

    preds = xg_model.predict(test_df.drop(columns = target_column))

    if classification:
        accuracy = (preds == test_df[target_column]).mean()
        print("accuracy out of sample is", accuracy)
    else:
        accuracy = abs(preds - test_df[target_column]).mean()
        print("MAE out of sample is", accuracy)
    return xg_model
    
def predict_xgboost(xg_model, data, pred_columns, categorical_columns, proba=False, binary_proba=True):

    data = data.copy(deep=True)
    data = data[pred_columns]
    data = pd.get_dummies(data, columns=categorical_columns)

    
    for column in xg_model.feature_names_in_:
        if column not in data.columns and 'pitch_type' in column:
            data[column] = 0
        elif column not in data.columns:
            raise Exception("Error: New data is missing column", column)
            
    data = data[xg_model.feature_names_in_]
    object_columns = data.select_dtypes(include=['object']).columns

    # Convert object columns to numeric
    data[object_columns] = data[object_columns].apply(pd.to_numeric, errors='coerce')

    if proba:
        preds = xg_model.predict_proba(data)
        if binary_proba:
            preds = preds[:,1]
    else:
        preds = xg_model.predict(data)
    
    return preds

def train_pitch_value_models(df, path, load_ball_strike=True):

    categorical_columns = ["zone", "pitch_type"]
    
    if load_ball_strike:
        with open(f'{path}/model_ball_strike.pkl','rb') as f:
            ball_strike_model = pickle.load(f)
    else:
        
        df['pitch_type'] = df.pitch_type.astype('category')    
        called_data = df[
            np.logical_and(df.taken, df.game_date<ball_strike_model_train_cutoff)]\
            .dropna(subset=xgboost_ball_strike_pred_columns + categorical_columns)
        
        ball_strike_model = train_xgboost(called_data, 'strike', xgboost_ball_strike_pred_columns, categorical_columns)
        
        with open(f'{path}/model_ball_strike.pkl','wb') as f:
            pickle.dump(ball_strike_model, f)

        
    df['pitch_type'] = df.pitch_type.astype(str)

    df['strike_probability'] = predict_xgboost(ball_strike_model, df, xgboost_ball_strike_pred_columns, 
                                               categorical_columns=categorical_columns, proba=True)
        
        
    df['pitch_type'] = df.pitch_type.astype(str)

    print('about to train models with columns')
    print(list(xgboost_pred_columns))
    
    df['pitch_type'] = df.pitch_type.astype('category')
    
    
    data = df[['game_date']+xgboost_pred_columns+[xgboost_target_column]]
    train_data = data[data.game_date <= pitch_model_train_cutoff]

    xg_model_benchmark = train_xgboost(train_data, 
                             xgboost_target_column, 
                             xgboost_benchmark_columns, 
                             categorical_columns = [],
                             classification=False)
    
        
    with open(f'{path}/benchmark_model.pkl','wb') as f:
        pickle.dump(xg_model_benchmark, f)
        
    xg_model = train_xgboost(train_data, 
                             xgboost_target_column, 
                             xgboost_pred_columns, 
                             xgboost_categorical_columns,
                             classification=False)

    
    with open(f'{path}/model_value.pkl','wb') as f:
        pickle.dump(xg_model, f)

    xg_model_stuff = train_xgboost(train_data, 
                                   xgboost_target_column, 
                                   xgboost_pred_columns_stuff_only, 
                                   categorical_columns = ['k_pitch_type_adj','pitch_type'],
                                   classification=False)
    

    with open(f'{path}/model_stuff.pkl','wb') as f:
        pickle.dump(xg_model_stuff, f)


    xg_model_cmd = train_xgboost(train_data,
                                 xgboost_target_column, 
                                 xgboost_pred_columns_cmd_only, 
                                 xgboost_categorical_columns, 
                                 classification=False)
    
    
    with open(f'{path}/model_cmd.pkl','wb') as f:
        pickle.dump(xg_model_cmd, f)


def predict_with_pitch_value_models(df, path):

    '''
    categorical_columns = ["zone", "pitch_type"]
    df['pitch_type'] = df.pitch_type.astype('category')
    called_data = df[
        np.logical_and(df.taken, df.game_date<ball_strike_model_train_cutoff)]\
        .dropna(subset=xgboost_ball_strike_pred_columns + categorical_columns)

    xg_model = train_xgboost(called_data.sample(50000), 'strike', xgboost_ball_strike_pred_columns, categorical_columns)
    with open('pitch_value_models/model_ball_strike.pkl','wb') as f:
        pickle.dump(xg_model, f)
        
    df['pitch_type'] = df.pitch_type.astype(str)
    '''
    
    with open(f'{path}/model_ball_strike.pkl','rb') as f:
        ball_strike_model = pickle.load(f)
            
    with open(f'{path}/model_cmd.pkl','rb') as f:
        xg_model_cmd = pickle.load( f)
        
    with open(f'{path}/model_stuff.pkl','rb') as f:
        xg_model_stuff = pickle.load( f)
        
    with open(f'{path}/model_value.pkl','rb') as f:
        xg_model = pickle.load( f)
    
    with open(f'{path}/benchmark_model.pkl','rb') as f:
        xg_model_benchmark = pickle.load( f)


    for col in xgboost_categorical_columns:
        df[col] = df[col].astype('category')


    df['strike_probability'] = predict_xgboost(ball_strike_model, 
                                               df,
                                               xgboost_ball_strike_pred_columns, 
                                               categorical_columns=["zone", "pitch_type"],
                                               proba=True)
    
    df_with_pitch_values = df[df.game_date>pitch_model_train_cutoff]
    
    pd.options.display.max_rows = 300
    print(df_with_pitch_values.dtypes)

    df_with_pitch_values['pred_delta_run_expectancy'] = predict_xgboost(xg_model, 
                                                      df_with_pitch_values, 
                                                      xgboost_pred_columns,
                                                      xgboost_categorical_columns)
    
    df_with_pitch_values['pred_delta_run_expectancy_cmd'] = predict_xgboost(xg_model_cmd, 
                                                      df_with_pitch_values, 
                                                      xgboost_pred_columns_cmd_only,
                                                      xgboost_categorical_columns)

    df_with_pitch_values['pred_delta_run_expectancy_stuff'] = predict_xgboost(xg_model_stuff, 
                                                      df_with_pitch_values, 
                                                      xgboost_pred_columns_stuff_only,
                                                      ['k_pitch_type_adj','pitch_type'])

    df_with_pitch_values['pred_benchmark'] = predict_xgboost(xg_model_benchmark, 
                                                      df_with_pitch_values, 
                                                      xgboost_benchmark_columns,
                                                      [])
    
    df_with_pitch_values['dre_above_average'] = df_with_pitch_values['delta_run_expectancy'] - \
        df_with_pitch_values['pred_benchmark']
    
    df_with_pitch_values['pred_dre_above_average'] = df_with_pitch_values['pred_delta_run_expectancy'] - \
        df_with_pitch_values['pred_benchmark']
    
    df_with_pitch_values['pred_dre_above_average_stuff'] = \
        df_with_pitch_values['pred_delta_run_expectancy_stuff'] - \
        df_with_pitch_values['pred_benchmark']
    
    df_with_pitch_values['pred_dre_above_average_cmd'] = df_with_pitch_values['pred_delta_run_expectancy_cmd'] - \
        df_with_pitch_values['pred_benchmark']
    
    df_with_pitch_values['strike_value'] = np.where(df_with_pitch_values.strike_value.isna(),
                                          -.26-df_with_pitch_values.count_value_away_from_average,
                                          df_with_pitch_values.strike_value)
    
    df_with_pitch_values['whiff'] = df_with_pitch_values.description.str.contains('swinging_strike')
    df_with_pitch_values['ground_ball'] = np.where(
        df_with_pitch_values.launch_angle.isna(), np.nan,
        df_with_pitch_values.bb_type=='ground_ball')
    
    df_with_pitch_values['barrel'] = np.where(
        df_with_pitch_values.launch_angle.isna(), np.nan,
            df_with_pitch_values.launch_speed_angle==6)

    df_with_pitch_values['called_strike'] = df_with_pitch_values.description=='called_strike'
    return df_with_pitch_values

if __name__ == '__main__':

    df = pd.read_parquet('engineered_data.parquet')

    if False:

        train_pitch_value_models(df)
        
    df_with_pitch_values = predict_with_pitch_value_models(df)

    df_with_pitch_values.to_parquet('data_with_pitch_values.parquet')
    

