
import pickle

from pitch_wrangling import pitch_physics, pitcher_specific_metrics
from pitch_wrangling import find_sz_edges,plot_strike_zone
from pitch_wrangling import get_events, get_linear_weights

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

from config import xgboost_target_column, xgboost_categorical_columns, xgboost_pred_columns
from config import xgboost_pred_columns_stuff_only, xgboost_pred_columns_cmd_only, xgboost_benchmark_columns
from config import pitch_model_train_cutoff
    
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
    
    if proba:
        preds = xg_model.predict_proba(data)
        if binary_proba:
            preds = preds[:,1]
    else:
        preds = xg_model.predict(data)
    
    return preds



def find_k_pitch_types(df, n_clusters=6, use_pca=True, n_components=4):


    pitch_group_columns = ['velo_percentile_for_pitcher','release_spin_rate',
                           'armside_horz_break','pfx_z', 'armside_tilt', 'spin_efficiency']
    pitch_types = df[pitch_group_columns].dropna()
    
    # Run K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    if use_pca:
        # Standardize the features (important for PCA)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pitch_types)
        
        # Initialize PCA with the number of components you want to retain
        pca = PCA(n_components=n_components)
        
        # Perform PCA
        pca.fit(scaled_data)

        # Transform the data into the new feature space
        pca_data = pca.transform(scaled_data)
        
        # Optionally, you can examine the explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        print("Explained Variance Ratio:", explained_variance_ratio)

        kmeans.fit(pca_data)
    
    else:
        kmeans.fit(scaled_data)

    # Print cluster labels
    print("Cluster labels:", kmeans.labels_)
    
    # (Optional) Print cluster centers after standardization
    print("Cluster centers (after standardization):")
    print(kmeans.cluster_centers_)

    if use_pca:
        df['est_pitch_type'] = kmeans.predict(
            pca.transform(scaler.transform(df[pitch_group_columns].fillna(0))))
    else:
        df['est_pitch_type'] = kmeans.predict(
            scaler.transform(df[pitch_group_columns].fillna(0)))

    # align the models est pitch type with the most frequent label statcast gives those pitches
    k_pitch_types = df.groupby('est_pitch_type')\
        .agg(k_pitch_type =('pitch_type',pd.Series.mode)).reset_index()
    
    k_pitch_types['count'] = k_pitch_types.\
        groupby('k_pitch_type').est_pitch_type.transform(lambda x:range(1,1+len(x)))
    
    k_pitch_types['total_count'] = k_pitch_types.\
        groupby('k_pitch_type')['count'].transform('max')

    # rename so that if two clusters are FF we get FF_1 and FF_2
    k_pitch_types['k_pitch_type'] = np.where(k_pitch_types.total_count>1, 
                              k_pitch_types.k_pitch_type.astype(str) +'_'+ k_pitch_types['count'].astype(str),
                              k_pitch_types.k_pitch_type.astype(str))

    df = df.merge(k_pitch_types[['k_pitch_type','est_pitch_type']],
         how = 'left', on = 'est_pitch_type').drop(columns = 'est_pitch_type')

    df['k_pitch_type_adj'] = df.groupby(['pitch_type', 'pitcher'])\
        .k_pitch_type.transform(lambda x: x.mode().iloc[0])

    return df



def get_context_stats(df, at_bat_cap = 400):
    
    df['batter_zone_woba'] = 1.2*df\
        .groupby(['batter','zone'])\
        .linear_weight_if_at_bat_over.transform('mean')
    
    df['batter_pitch_woba'] = 1.2*df\
        .groupby(['k_pitch_type_adj'])\
        .linear_weight_if_at_bat_over.transform('mean')


    
    df['batter_count_woba'] = 1.2*df\
        .groupby(['batter','balls', 'strikes']) \
        .linear_weight_if_at_bat_over.transform('mean')
    
    df['batter_zone_sample_size'] = df\
        .groupby(['batter','zone']).end_of_at_bat.transform('sum')
    
    df['batter_pitch_sample_size'] = df\
        .groupby(['batter','k_pitch_type_adj']).end_of_at_bat.transform('sum')
    
    df['batter_count_sample_size'] = df\
        .groupby(['batter','balls', 'strikes']).end_of_at_bat.transform('sum')

    df['batter_sample_size'] = df\
        .groupby(['batter']).end_of_at_bat.transform('sum')
    

    df['count_woba'] = 1.2*df\
    .groupby(['balls', 'strikes']) \
    .linear_weight_if_at_bat_over.transform('mean')
    
    df['zone_woba'] = 1.2*df\
        .groupby(['zone']) \
        .linear_weight_if_at_bat_over.transform('mean')

    df['pitch_woba'] = 1.2*df\
        .groupby(['k_pitch_type_adj']) \
        .linear_weight_if_at_bat_over.transform('mean')

    df['batter_woba'] = 1.2*df\
        .groupby(['batter']) \
        .linear_weight_if_at_bat_over.transform('mean')
    
    df['batter_count_sample_portion'] = (np.sqrt(df['batter_count_sample_size'])/np.sqrt(at_bat_cap)).clip(0,1)
    df['batter_count_woba'] = df['batter_count_sample_portion']*df['batter_count_woba'] + \
        (1-df['batter_count_sample_portion'])*df.count_woba

    df['batter_sample_portion'] = (np.sqrt(df['batter_sample_size'])/np.sqrt(at_bat_cap)).clip(0,1)
    df['batter_woba'] = df['batter_sample_portion']*df['batter_woba'] + \
        (1-df['batter_sample_portion'])*.310
    
    df['batter_zone_sample_portion'] = (np.sqrt(df['batter_zone_sample_size'])/np.sqrt(at_bat_cap)).clip(0,1)
    df['batter_zone_woba'] = df['batter_zone_sample_portion']*df['batter_zone_woba'] + \
        (1-df['batter_zone_sample_portion'])*df.zone_woba
    
    df['batter_pitch_sample_portion'] = (np.sqrt(df['batter_pitch_sample_size'])/np.sqrt(at_bat_cap)).clip(0,1)
    df['batter_pitch_woba'] = df['batter_pitch_sample_portion']*df['batter_pitch_woba'] + \
        (1-df['batter_pitch_sample_portion'])*df.pitch_woba


    # Assuming df is your DataFrame
    pivot_df = df.pivot_table(index='batter', columns='zone', 
                              values='batter_zone_woba', aggfunc='first').reset_index()
    
    # Rename columns
    pivot_df.columns.name = None  # Remove the name of the columns index
    pivot_df.columns = ['batter'] + [f'batter_zone_{col}_woba' for col in pivot_df.columns[1:]]
    
    # Display the pivoted DataFrame
    pivot_df.head()
    
    df = df.merge(pivot_df, how = 'left', on = 'batter')

    pivot_df = df.pivot_table(index='batter', columns='k_pitch_type_adj', 
                              values='batter_pitch_woba', aggfunc='first').reset_index()

    # Rename columns
    pivot_df.columns.name = None  # Remove the name of the columns index
    pivot_df.columns = ['batter'] + [f'batter_pitch_{col}_woba' for col in pivot_df.columns[1:]]
    
    # Display the pivoted DataFrame
    pivot_df.head()
    
    df = df.merge(pivot_df, how = 'left', on = 'batter')
    df = df.drop(columns = [x for x in df.columns if '_sample_' in x])
    df['count_value_away_from_average'] = df.count_value - df.count_value.mean()
    return df

if __name__=='__main__':

    if True:
        
        df = pd.read_parquet('pitches_by_year')
    
        df = pitch_physics(df)
        df = find_sz_edges(df, edge_tolerance = 2/12)
    
        df = get_events(df)
        df = get_linear_weights(df)
    
        df = pitcher_specific_metrics(df)
    
        df['taken'] = np.logical_or(df.description == 'called_strike', df.ball)
    
        categorical_columns = ["zone", "pitch_type"] 
        target_column = 'strike'
        pred_columns = categorical_columns + ['plate_z','plate_x', 'pfx_x','pfx_z'] +\
            [x for x in df.columns if 'edge' in x or 'corner' in x]
    
        df['pitch_type'] = df.pitch_type.astype('category')
        
        # dont need a huge sample as this feature is a pretty easy one to learn
        called_data = df[
            np.logical_and(df.taken, df.game_date<pitch_model_train_cutoff)]\
            .sample(50000).dropna(subset=pred_columns)

        xg_model = train_xgboost(called_data, target_column, pred_columns, categorical_columns)
        
        
        
        
        df['strike_probability'] = predict_xgboost(xg_model, df, pred_columns, 
                                                   categorical_columns, proba=True)
        
        
        df['pitch_type'] = df.pitch_type.astype(str)
        
        df = find_k_pitch_types(df, n_clusters=6, use_pca=True, n_components=4)
        df = get_context_stats(df)
        
        df['delta_run_expectancy'] = np.where(
            df.end_of_at_bat,
            df.linear_weight - df.count_value_away_from_average,
            df.linear_weight)
        
        df.stand = np.where(df.stand=='L',1,0)
        df.p_throws = np.where(df.p_throws=='L',1,0)
        
        df.to_parquet('engineered_data.parquet')


    else:
        df = pd.read_parquet('engineered_data.parquet')
        df.stand = np.where(df.stand=='L',1,0)
        df.p_throws = np.where(df.p_throws=='L',1,0)




    if False:

        print('about to train models with columns')

        df['pitch_type'] = df.pitch_type.astype('category')

        data = df[['game_date']+xgboost_pred_columns+[xgboost_target_column]]
        train_data = data[data.game_date < pitch_model_train_cutoff]

        xg_model_benchmark = train_xgboost(train_data, 
                                 xgboost_target_column, 
                                 xgboost_benchmark_columns, 
                                 categorical_columns = [],
                                 classification=False)
    
        
        with open('benchmark_model.pkl','wb') as f:
            pickle.dump(xg_model_benchmark, f)
            
        xg_model = train_xgboost(train_data, 
                                 xgboost_target_column, 
                                 xgboost_pred_columns, 
                                 xgboost_categorical_columns,
                                 classification=False)
    
        
        with open('model_value.pkl','wb') as f:
            pickle.dump(xg_model, f)
    
        xg_model_stuff = train_xgboost(train_data, 
                                       xgboost_target_column, 
                                       xgboost_pred_columns_stuff_only, 
                                       categorical_columns = ['k_pitch_type_adj','pitch_type'],
                                       classification=False)
        
    
        with open('model_stuff.pkl','wb') as f:
            pickle.dump(xg_model_stuff, f)
    
    
        xg_model_cmd = train_xgboost(train_data,
                                     xgboost_target_column, 
                                     xgboost_pred_columns_cmd_only, 
                                     xgboost_categorical_columns, 
                                     classification=False)
        
        
        with open('model_cmd.pkl','wb') as f:
            pickle.dump(xg_model_cmd, f)

    else:
        with open('model_cmd.pkl','rb') as f:
            xg_model_cmd = pickle.load( f)
            
        with open('model_stuff.pkl','rb') as f:
            xg_model_stuff = pickle.load( f)
            
        with open('model_value.pkl','rb') as f:
            xg_model = pickle.load( f)
        
        with open('benchmark_model.pkl','rb') as f:
            xg_model_benchmark = pickle.load( f)

    
    for col in xgboost_categorical_columns:
        df[col] = df[col].astype('category')
        
    train_data = df[df.game_date>pd.to_datetime(pitch_model_train_cutoff)]

    train_data['pred_delta_run_expectancy'] = predict_xgboost(xg_model, 
                                                      train_data, 
                                                      xgboost_pred_columns,
                                                      xgboost_categorical_columns)
    
    train_data['pred_delta_run_expectancy_cmd'] = predict_xgboost(xg_model_cmd, 
                                                      train_data, 
                                                      xgboost_pred_columns_cmd_only,
                                                      xgboost_categorical_columns)

    train_data['pred_delta_run_expectancy_stuff'] = predict_xgboost(xg_model_stuff, 
                                                      train_data, 
                                                      xgboost_pred_columns_stuff_only,
                                                      ['k_pitch_type_adj','pitch_type'])

    train_data['pred_benchmark'] = predict_xgboost(xg_model_benchmark, 
                                                      train_data, 
                                                      xgboost_benchmark_columns,
                                                      [])
    
    train_data['dre_above_average'] = \
        train_data['delta_run_expectancy'] - \
        train_data['pred_benchmark']
    
    train_data['pred_dre_above_average'] = \
        train_data['pred_delta_run_expectancy'] - \
        train_data['pred_benchmark']
    
    train_data['pred_dre_above_average_stuff'] = \
        train_data['pred_delta_run_expectancy_stuff'] - \
        train_data['pred_benchmark']
    
    train_data['pred_dre_above_average_cmd'] = \
        train_data['pred_delta_run_expectancy_cmd'] - \
        train_data['pred_benchmark']
    
    train_data['strike_value'] = np.where(train_data.strike_value.isna(),
                                          -.26-train_data.count_value_away_from_average,
                                          train_data.strike_value)
    
    train_data['whiff'] = train_data.description.str.contains('swinging_strike')
    train_data['ground_ball'] = np.where(
        train_data.launch_angle.isna(), np.nan,
        train_data.bb_type=='ground_ball')
    
    train_data['barrel'] = np.where(
        train_data.launch_angle.isna(), np.nan,
            train_data.launch_speed_angle==6)

    train_data['called_strike'] = train_data.description=='called_strike'


    train_data.to_parquet('train_data.parquet')
    
 
