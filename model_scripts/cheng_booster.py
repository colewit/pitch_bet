
import pandas as pd
import numpy as np
import dill as pickle
import os
import joblib
import time
import gc
import sys
import psutil

import traceback
from model_scripts.ordinal_losses import create_corn_obj_xgb, create_corn_eval_xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, make_scorer
from sklearn.pipeline import Pipeline

import xgboost as xgb

def cheng_labels(y_label, n_classes):
    """
    Encode ordinal labels using cumulative binary vectors.
    
    Args:
        y_label: A 1D numpy array of shape (num_samples,) containing the ordinal labels.
        n_classes: The number of ordinal classes.
    
    Returns:
        encoded_labels: A 2D numpy array of shape (num_samples, n_classes) containing the encoded labels.
    """
    # Initialize an array for the encoded labels
    encoded_labels = np.zeros((len(y_label), n_classes))
    
    # Fill the encoded labels
    for i in range(n_classes):
        encoded_labels[:, i] = (y_label > i).astype(int)
    
    return encoded_labels



def sample_events_multinomial(df):
    # Check if all values are NaN

    df['strikeout'] = 0
    df['walk'] = 0
    outcomes = ['strikeout','fieldout','walk','single','double','homerun']
    na_indices = np.where(df['homerun'].isna())[0]

    probs_array = df[outcomes].fillna(0).values
    
    # Compute the cumulative sum of probabilities for each row
    cumsum_probs = np.cumsum(probs_array, axis=1)
    
    # Generate uniform random numbers for each row
    random_numbers = np.random.rand(df.shape[0], 1)
    
    # Vectorized sampling: Find the first index where the cumulative sum exceeds the random number
    sampled_labels = (random_numbers < cumsum_probs).argmax(axis=1).astype(float)

    sampled_labels[na_indices] = np.nan
    
    return sampled_labels

def get_cheng_labels(y_label, n_classes):
    """
    Encode ordinal labels using cumulative binary vectors.
    
    Args:
        y_label: A 1D numpy array of shape (num_samples,) containing the ordinal labels.
        n_classes: The number of ordinal classes.
    
    Returns:
        encoded_labels: A 2D numpy array of shape (num_samples, n_classes) containing the encoded labels.
    """
    # Initialize an array for the encoded labels
    encoded_labels = np.zeros((len(y_label), n_classes-1))
    
    # Fill the encoded labels
    for i in range(n_classes-1):
        encoded_labels[:, i] = (y_label > i).astype(int)
    
    return encoded_labels
    

    
def calculate_class_weights(y_train):
    """
    Calculates class weights based on inverse class frequency.
    
    Args:
      y_train (np.ndarray): Array of true class labels.
    
    Returns:
      np.ndarray: Array of class weights.
    """
    class_counts = np.bincount(y_train)
    total_count = np.sum(class_counts)
    class_weights = total_count / class_counts
    class_weights = class_weights / np.sum(class_weights)
    return class_weights


def run_grid_search(data_dict,
                    train_cutoff,
                    boost_rounds=300,
                    max_depth=4,
                    patience=10,
                    cheng_labels=False, 
                    corn_loss=True,
                    adaptive_labels=True,
                    adaptive_label_model_pipeline=None,
                    model_folder = 'saved_boosters'):
    
    X_meta = data_dict['X_meta']
    X = data_dict['X'].astype(float)
    label = data_dict['Y'].astype(int)
    

    columns = [x for x in X.columns if 'index' not in x and x!='month' and x!='year']
    X = X[columns]
    
    num_features = X.shape[1]
    num_classes = len(np.unique(label))


    train_indices = np.where(X_meta.game_date<=train_cutoff)[0]
    test_indices = np.where(X_meta.game_date>train_cutoff)[0]
    
    X_test = X.iloc[test_indices]
    X = X.iloc[train_indices]
    
    X_meta_test = X_meta.iloc[test_indices]
    X_meta = X_meta.iloc[train_indices]

    train_label = np.array(label)[train_indices]
    test_label = np.array(label)[test_indices]
    
    if cheng_labels:    
        y = get_cheng_labels(train_label.astype(int), num_classes)
    else:
        y = train_label.astype(int)
  

    def to_float(x):
        return x.float()

    best_val_loss = np.inf

    if adaptive_labels:

        outcomes = ['fieldout', 'single', 'double', 'homerun']
        features = ['estimated_woba_using_speedangle', 'hit_distance_sc',
                    'launch_angle', 'launch_speed', 'estimated_ba_using_speedangle']

        meta = ['game_date','team_at_bat_number','pitcher','batter']

        print({k:v.shape for k,v in data_dict.items()})
        X_adaptive_labels = data_dict['adaptive_label_data'][features+meta]
        X_adaptive_labels = X_adaptive_labels.dropna()   
        
        proba_df = pd.DataFrame(
            adaptive_label_model_pipeline.predict_proba(X_adaptive_labels[features]),
            columns=outcomes)
        
        proba_df = pd.concat([proba_df, X_adaptive_labels[meta].reset_index(drop=True)], axis = 1)

        full_proba_df = X_meta.merge(proba_df, 
                     how = 'left', 
                     on = ['game_date','team_at_bat_number','pitcher','batter'])

        X_train, X_final_val, y_train, y_final_val, X_meta_train, X_meta_val, proba_df_train, proba_df_val = \
            train_test_split(X, y, X_meta, full_proba_df, test_size=0.2, random_state=42)
    else:
        X_train, X_final_val, y_train, y_final_val, X_meta_train, X_meta_val = \
            train_test_split(X, y, X_meta, test_size=0.2, random_state=42)

    for max_depth in [3,4,5,6,8,10]:
        for lr in [.005, .01, .05, .1]:
    
            
            try:
                print("running for LR", lr)
    
                bootstrapped_labels_train = sample_events_multinomial(proba_df_train)
                bootstrapped_labels_val = sample_events_multinomial(proba_df_val)
                
                labels_arr_train = np.where(np.isnan(bootstrapped_labels_train),
                                      y_train, bootstrapped_labels_train)
                
                labels_arr_val = np.where(np.isnan(bootstrapped_labels_val),
                                      y_final_val, bootstrapped_labels_val)
    
    
                print(X_train.shape, labels_arr_train.shape)
                print(X_final_val.shape, labels_arr_val.shape)
                print(X_test.shape, test_label.shape)
                
                dtrain = xgb.DMatrix(X_train, label=labels_arr_train)
                dval = xgb.DMatrix(X_final_val, label=labels_arr_val)
                dtest = xgb.DMatrix(X_test, label = test_label)
                
                param = {
                    'max_depth': max_depth,
                    'eta': lr,
                    'objective': 'multi:softprob',
                    'num_class': num_classes-1,
                    'disable_default_eval_metric':True
                }
    
                evallist = [(dval, 'eval')]#, (dtest, 'test')]
    
                custom_loss_fn = create_corn_obj_xgb(labels_arr_train.astype(int), num_classes)
        
                custom_eval_fn = create_corn_eval_xgb(labels_arr_val.astype(int), num_classes)
    
                evals_result = {}
                bst = xgb.train(param, dtrain, boost_rounds, evallist,
                                early_stopping_rounds=patience, 
                                obj=custom_loss_fn,
                                custom_metric=custom_eval_fn,
                                evals_result=evals_result)
    
                val_loss = bst.best_score
    
                #test_loss = evals_result['test']['corn_loss']
                
                # Save the model
                model_hist = {
                    'model':bst,
                    'learning_rate': lr,
                    'max_depth': max_depth,
                    'valid_loss': val_loss
                }
    
                model_hist_name = f'best_model_lr_{lr}_max_depth_{max_depth}_history.pkl'
                model_hist_path = os.path.join(model_folder.rstrip('/'), model_hist_name)
                with open(model_hist_path, 'wb') as f:
                    pickle.dump(model_hist, f)
            
                if val_loss < best_val_loss:
        
                    best_val_loss = val_loss
    
                    model_hist_name = f'best_model_history.pkl'
                    model_hist_path = os.path.join(model_folder.rstrip('/'), model_hist_name)
    
                    # Save the fully trained model
                    with open(model_hist_path, 'wb') as f:
                        pickle.dump(model_hist, f)
      
    
            except:
                
                traceback.print_exc()
                continue
            
    print('training of boosters completed')

if __name__ == '__main__':

    with open('data_for_ordinal_fangraphs.pkl','rb') as f:
        data_dict = pickle.load(f)

    with open('label_models/model.pkl','rb') as f:
        adaptive_label_model_pipeline = pickle.load(f)

    run_grid_search(data_dict,
                        boost_rounds=300,
                        max_depth=4,
                        patience=10,
                        cheng_labels=False, 
                        adaptive_labels=True,
                        corn_loss=True,
                        model_folder = 'saved_boosters',
                        adaptive_label_model_pipeline=adaptive_label_model_pipeline)
