
import pandas as pd
import numpy as np

from prepare_pitch_values import build_rolling_df
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

def setup_adaptive_smoothing_model(imputed_train_data, 
                                   one_hot_encoder,
                                   num_classes=6,
                                   alpha = 1,
                                   smoothing_prior = None,
                                   categorical_columns=['last_ab','batter_last_ab'],
                                   meta_columns = ['player_name','pitcher','batter', 'game_date']):
    

    if smoothing_prior is None:
        smoothing_prior = np.array([.02]*num_classes)
        
    xtrain = imputed_train_data[imputed_train_data.game_date<='2023-06-01']
    xtest = imputed_train_data[imputed_train_data.game_date>'2023-06-01']
    train_label = xtrain.label
    test_label = xtest.label
    
    pred_columns = [x for x in imputed_train_data.columns if x!=label and x not in meta_columns]
    #pred_columns = [x for x in pred_columns
    #               if 'last' not in x and 'matchup' not in x
    #               and 'this_inning' not in x and 'today' not in x]
    
    
    xtrain_enc = one_hot_encode_data(one_hot_encoder, xtrain[pred_columns],
                                 categorical_columns)
    xtest_enc = one_hot_encode_data(one_hot_encoder, xtest[pred_columns], 
                                 categorical_columns)
        
    y_train_smoothed = empirical_label_smoothing( train_label.astype(int),
                                                 smoothing_prior, alpha, num_classes)
    
    y_test_smoothed = empirical_label_smoothing( test_label.astype(int),
                                                 smoothing_prior, alpha, num_classes)
    
    dtrain = xgb.DMatrix(xtrain_enc, label=train_label)
    dtest = xgb.DMatrix(xtest_enc, label = test_label)
    
    custom_loss_fn = create_custom_loss(train_label,prior=smoothing_prior, alpha=alpha )
    custom_eval_fn = create_custom_eval_metric(y_test_smoothed,prior=smoothing_prior, alpha=alpha )
    
    # Set parameters
    model_params = {
      'objective': 'multi:softprob',  # Use custom objective function
      'max_depth': 4,
      'eta': 0.07,
        'num_class': num_classes
    }
    
    #model = XGBClassifier(**model_params)
    bst = xgb.train(model_params, dtrain, num_boost_round=500, 
                     evals=[(dtest, 'test')],
                    early_stopping_rounds=5,
                    obj=custom_loss_fn,
                    custom_metric=custom_eval_fn)
    return bst

def create_custom_eval_metric(y_smoothed, alpha, prior):
    def eval_metric(preds, dtest):
        """
        Custom evaluation metric for XGBoost.
        
        :param preds: Predictions from the model.
        :return: Tuple (metric_name, value, higher_better).
        """
        
        num_classes = y_smoothed.shape[1]
        preds = preds.reshape(-1, num_classes)
        preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        preds /= np.sum(preds, axis=1, keepdims=True)
        
        # Cross-entropy loss with smoothed labels
        loss = -np.sum(y_smoothed * np.log(preds + 1e-7)) / preds.shape[0]
        
        return 'custom_loss', loss

    prior = prior/alpha
    return eval_metric

def empirical_label_smoothing( y_label, prior, alpha, n_classes):
    
    prior = prior/alpha
    # Apply label smoothing (replace with your preferred smoothing logic)
    smoothed_labels = (1 - sum(prior)) * np.eye(n_classes)[y_label] + prior[np.newaxis,:]
    return smoothed_labels

def create_custom_loss(y_train, prior, alpha):
    """
    Creates a custom loss function with closure to access y_train.
    
    Args:
      y_train: Training labels (numpy array).
    
    Returns:
      A custom loss function that can access y_train.
    """
    def custom_loss_smoothed_ce(y_pred, dtrain):
        """
        Custom objective function with label smoothing for XGBoost.
        
        Args:
            y_pred: Predicted probabilities from the model (numpy array).
            dtrain: XGBoost DMatrix containing training features (not used for labels).
        
        Returns:
            Loss value (float) and gradient (numpy array).
        """

        # Access y_train from the closure (without needing a global variable)
        n_classes = len(np.unique(y_train))
        
        #alpha = dtrain.params.get('alpha', 0.1)  # Example default value
        # Apply label smoothing
        smoothed_labels = empirical_label_smoothing( y_train, prior, alpha, n_classes)


        loss = -np.sum(smoothed_labels * np.log(y_pred[np.newaxis,:] + 1e-7)) / len(y_train)
        
        preds = y_pred.reshape(-1, num_classes)
        
        # Compute the softmax of the predictions
        preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        preds /= np.sum(preds, axis=1, keepdims=True)

        # Calculate cross-entropy loss (replace with efficient vectorized implementation)
        #loss = -np.sum(smoothed_labels * np.log(preds + 1e-7)) / len(y_train)
        
        # Compute the gradients and hessians
        grad = preds - smoothed_labels
        hess = preds * (1 - preds)

        return grad.flatten(), hess.flatten()

    y_train = y_train.astype(int)
    prior = prior/alpha
    return custom_loss_smoothed_ce



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
    
    train_data = train_data.sort_values(['game_date', 'inning','pitcher_at_bat_number'])

    
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

if __name__ == '__main__':


    all_pitch_data = pd.read_parquet('pilot_data.parquet')
    cols = ['pitcher', 'player_name','game_date','pitcher_at_bat_number','time_thru_the_order','batter']

    cols += [x for x in all_pitch_data.columns if 'whiff' in x or 'called_strike' in x 
             or 'strikeout' in x or 'homerun' in x or 'woba' in x or 'walk' in x or '_RA9' in x]
    
    cols += [ 'is_CU','is_SL','is_FF_2','is_FF_1','is_SI','is_CH']
    all_pitch_data = all_pitch_data[cols]

    df_2023 = pd.read_parquet('df_current_season.parquet')

    df_2023['k_pitch_type_adj'] = df_2023['k_pitch_type_adj'].astype(str)
    df_2023.strikeout = np.where(df_2023.end_of_at_bat, df_2023.strikeout, np.nan)
    df_2023.walk = np.where(df_2023.end_of_at_bat, df_2023.walk, np.nan)
    df_2023.homerun = np.where(df_2023.end_of_at_bat, df_2023.homerun, np.nan)
    df_2023.single = np.where(df_2023.end_of_at_bat, df_2023.single, np.nan)
    df_2023.double = np.where(df_2023.end_of_at_bat, df_2023.double, np.nan)
    df_2023.triple = np.where(df_2023.end_of_at_bat, df_2023.triple, np.nan)
    df_2023['field_out'] = 1 - df_2023[['walk','homerun','strikeout','single','double','triple']].max(axis=1)
    
    
    
    df_2023['k_pitch_type_adj'] = df_2023['k_pitch_type_adj'].astype(str)
    df_2023['pitch_type'] = df_2023['pitch_type'].astype(str)
    
    df_2023 = df_2023.sort_values(['pitcher','game_date','at_bat_number','pitch_number'])
    df_2023['at_bat_change'] = (df_2023.groupby(['pitcher','game_date']).at_bat_number.diff().fillna(0)!=0)

    df_2023['pitcher_at_bat_number'] = df_2023\
        .groupby(['pitcher','game_date']).at_bat_change.transform('cumsum') + 1

    df_2023['team_at_bat_number'] = df_2023\
        .groupby(['inning_topbot','game_date','home_team']).at_bat_change.transform('cumsum') + 1
    
    # add more state info to the game
    df_2023['batting_score'] = np.where(
        df_2023.inning_topbot=='Top', 
        df_2023.away_score,
        df_2023.home_score)
    
    df_2023['batting_team'] = np.where(
        df_2023.inning_topbot=='Top', 
        df_2023.away_team,
        df_2023.home_team)

    df_2023['lineup_slot'] = 1 + (df_2023.team_at_bat_number-1)%9
    df_2023['lineup_slot'] = df_2023.groupby(['batter','game_date'])\
        .lineup_slot.transform('first')


    # prep values for rolling window
    roll_columns = ['called_strike', 'whiff', 'homerun',
                    'walk', 'strikeout', 'estimated_woba_using_speedangle']
    
    sort_columns = ['game_date','batter','at_bat_number']
    group_columns = ['batter']
    merge_columns = ['batter','game_date']
    columns_to_keep = ['walk','field_out','strikeout',
                       'single','double','homerun','triple', 'pitcher','pitcher_at_bat_number']
    window_size = 700  # Number of pitches thrown
    min_period = 200

    # find rolling performance of batter up to current date
    batter_by_game = build_rolling_df(df_2023[df_2023.end_of_at_bat], 
                         roll_columns,
                         sort_columns, 
                         group_columns,
                         merge_columns, 
                         columns_to_keep,
                         window_size,
                         min_period,
                         drop_duplicates = False,
                         suffix = "_batter")

    batter_by_game = batter_by_game.sort_values(['game_date','at_bat_number'])
    batter_by_game['dummy'] = 1
    batter_by_game['cumulative_at_bats'] = batter_by_game.groupby('batter').dummy.transform('cumsum')
    batter_by_game.drop(columns='dummy', inplace=True)

    # make a column of regressed performance by throwing in league avg abs till total abs are 700
    num_abs_league_average = (700 - batter_by_game.cumulative_at_bats).clip(0)
    for col in batter_by_game.columns:
        if '_batter' in col:
    
            league_avg_performance = num_abs_league_average*batter_by_game[col].median() 
            batter_performance = batter_by_game.cumulative_at_bats*batter_by_game[col]
    
            total_abs = num_abs_league_average + batter_by_game.cumulative_at_bats
            batter_by_game[col+'_regressed'] = (league_avg_performance + batter_performance) /total_abs

    # add label column to batter data for ab but count double and triple the same for now
    batter_by_game['double_or_triple'] = batter_by_game[['double','triple']].max(axis=1)

    batter_by_game['label'] = np.argmax(batter_by_game[['strikeout','field_out','walk','single',
                                              'double_or_triple','homerun']],axis=1)

    
    train_batter_cols = [x for x in batter_by_game.columns if '_batter' in x]
    
    at_bat_cols = [x for x in all_pitch_data.columns if 'last_9' in x or 'next_3' in x]
    
    pitcher_meta = ['pitcher','game_date', 'pitcher_at_bat_number', 'batter']
    batter_meta = ['label','pitcher','batter','game_date','pitcher_at_bat_number'] 
    
    summary_stat_cols = [x for x in all_pitch_data.columns 
                  if x not in at_bat_cols
                  and x not in train_batter_cols
                  and x != 'batter' and x != 'pitcher_at_bat_number']
    
    
    
    train_data = all_pitch_data[at_bat_cols + pitcher_meta ]\
        .merge(batter_by_game[train_batter_cols + batter_meta],
               how ='left', on = pitcher_meta)
    
    train_data = train_data.merge(
        df_2023[['inning_topbot','inning','on_1b','on_2b','on_3b',
                 'game_date','pitcher_at_bat_number','pitcher',
                 'team_at_bat_number','batting_team','batting_score',
                 'lineup_slot']].drop_duplicates(),
        how = 'left', on = ['game_date','pitcher_at_bat_number','pitcher'])

    all_pitch_data = all_pitch_data.sort_values(['game_date','pitcher_at_bat_number'])

    # rolling stats at the start of first ab are what you enter the game with
    summary_stat_before_first_pitch_df = \
        all_pitch_data[summary_stat_cols].groupby(['pitcher','game_date']).first().reset_index()
    
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

    print(train_data.shape, 'before')
    # probably have to do this bc i have to pare down to just a single row per AB
    train_data = train_data.drop_duplicates(meta_cols)
    print(train_data.shape, 'after')

    # get rid of na player names
    train_data = train_data\
        .dropna(subset=['player_name'])

    train_data = add_state_data(train_data)

    train_data = train_data.sort_values(['game_date','team_at_bat_number'])
    train_data = train_data.reset_index()

    # get last ab against pitcher and last ab from batter pitcher matchup
    train_data['batter_last_ab'] = train_data.groupby(['pitcher', 'game_date', 'batter']).label.shift(1)    
    train_data['last_ab'] = train_data.groupby(['game_date','pitcher']).label.shift(1)

    train_data = train_data.sort_values(['game_date', 'team_at_bat_number'])
    train_data['at_bat_woba'] = 1.2*(
        .55*(train_data.label==2) + .7*(train_data.label==3) +\
        1.0*(train_data.label==4) + 1.65*(train_data.label==5))
    
    # these should technically be changed to not include at bat
    train_data['woba_batter'] = train_data.groupby(['batter'])\
        .at_bat_woba.transform(lambda x:x.rolling(window=700, min_periods=50).mean())
    train_data['woba_batter'] = train_data.groupby(['game_date','batter']).woba_batter.shift(1)

    
    train_data['next_batter_woba'] = train_data.groupby(['game_date','inning_topbot']).woba_batter.shift(-1)

    train_data['woba_pitcher'] = train_data.groupby(['pitcher'])\
        .at_bat_woba.transform(lambda x:x.rolling(window=700, min_periods=50).mean())
    train_data['woba_pitcher'] = train_data.groupby(['game_date','pitcher']).woba_pitcher.shift(1)

    train_data = train_data.dropna(subset='label')
    
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

    one_hot_encoder.fit(train_data.head(100000)[['last_ab', 'batter_last_ab']].dropna())
    imputed_train_data = impute_nas(train_data)
    
    model = setup_adaptive_smoothing_model(imputed_train_data, 
                                           one_hot_encoder,
                                           num_classes=6,
                                           alpha = 1,
                                           smoothing_prior = None,
                                           categorical_columns=['last_ab','batter_last_ab'])

