import numpy as np
import pandas as pd
from assemble_event_model_data import *
import pickle
from config import pitch_model_train_cutoff
import xgboost as xgb

import matplotlib.pyplot as plt
import time
from ordinal_losses import sigmoid
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager
import pickle
import time
import traceback
from functools import partial


def predict_xgboost(xg_model, data, pred_columns,
                    categorical_columns, one_hot_encoder=None,
                    proba=False, binary_proba=True, platt_models=None):

    data = data[pred_columns]

    if categorical_columns!=[]:
        one_hot_data = one_hot_encoder.transform(data[categorical_columns])
    
        # Convert encoded data to DataFrame for better visualization
        one_hot_data_df = pd.DataFrame(one_hot_data.toarray(),
                                    columns=one_hot_encoder\
                                           .get_feature_names_out(categorical_columns))
    
        data = pd.concat([data.drop(columns = categorical_columns).reset_index(),
                          one_hot_data_df], axis = 1)
        

    
    data = data[xg_model.feature_names].astype(float)

    dmat = xgb.DMatrix(data)
    if proba:
        #preds = xg_model.predict_proba(data)
        preds = xg_model.predict(dmat)
        if binary_proba:
            preds = preds[:,1]

        if platt_models is not None:
            preds = platt_scaling(data, preds, platt_models)
    else:
        preds = xg_model.predict(dmat)
    
    return preds


def predict_xgboost_ordinal(xg_model, data):

    data = data[xg_model.feature_names].astype(float)

    dmat = xgb.DMatrix(data)
    
    logits = xg_model.predict(dmat, output_margin=True)
    preds = sigmoid(logits)
    preds = np.cumprod(preds, axis=1)
    
    probas = np.zeros_like(preds)
    
    # Compute the probabilities for each row
    for row in range(preds.shape[0]):
        # Probability for class 0
        probas[row, 0] = 1 - preds[row, 0]
        # Probabilities for subsequent classes
        for i in range(1, preds.shape[1]):
            probas[row, i] = preds[row, i - 1] - preds[row, i]
    # add homer proba
    probas =np.hstack([probas, 1- probas.sum(axis=1).reshape(-1,1)])
    
    return probas

def platt_scaling(data, probas, linear_model):
 
    probas = pd.DataFrame(probas, columns = ['strikeout','fieldout','walk','single','double','homerun'])
    lr_train = pd.concat([data.reset_index(drop=True), probas], axis = 1)
    
    linear_model.fit(lr_train, xtrain.label)
    
class Game(object):

    def __init__(self, n_games):

        self.n_games = n_games
        self.reset_state()
        self.pull_pitcher_model = None

    def setup_event_model(self,event_model,pred_columns, 
                          categorical_columns, one_hot_encoder, platt_models):

        self.model = event_model
        self.one_hot_encoder = one_hot_encoder
        self.pred_columns = pred_columns
        self.categorical_columns = categorical_columns
        self.platt_models=platt_models

    def predict_event(self, pitcher_data, platt_models):

        event_probas = predict_xgboost_ordinal(self.model,
                                       pitcher_data)
        return event_probas
        
    def setup_pull_pitcher_model(self,
                                 pull_pitcher_model,
                                 most_recent_game_df,
                                 pull_pitcher_columns):
        
        static_cols = ['total_outs_recorded_5_perc',
                       'total_outs_recorded_25_perc', 'total_outs_recorded_50_perc',
                       'total_outs_recorded_75_perc', 'total_outs_recorded_95_perc',
                       'max_outs_recorded_by_end_of_game', 'min_outs_recorded_by_end_of_game',
                       'mean_recorded_by_end_of_game']
        d={}
        for col in static_cols:
            d[col] = most_recent_game_df[col].iloc[0]

        self.static_columns = static_cols
        self.pull_pitcher_static = d
        self.pull_pitcher_columns = pull_pitcher_columns
        self.pull_pitcher_model = pull_pitcher_model
        self.pull_pitcher_data = np.zeros((self.n_games, 36, len(self.pull_pitcher_columns)))
    
    def update_state_for_event_model(self, batter):

        df = pd.DataFrame(self.all_event_dicts)
        

        df['linear_weight'] = df.walk*.55 + df.single*.7 + \
            1.0*df.double + 1.26*df.triple + df.homerun*1.65 - .26
        
        batter_df = df[df.batter==batter]
        
        df['curr_inning'] = df.groupby('game_idx').inning.transform(lambda x : x == x.max())
        inning_df = df[df.curr_inning]
      
        
        state_data = {}

        state_data['game_idx'] = range(self.n_games)
        state_data['inning'] = self.inning#[game_idx]
        state_data['outs_when_up'] = self.outs#[game_idx]
        state_data['on_1b'] = self.runner_on_1b#[game_idx]
        state_data['on_2b'] = self.runner_on_2b#[game_idx]
        state_data['on_3b'] = self.runner_on_3b#[game_idx]

        state_data = pd.DataFrame(state_data)
        
        categories = ['strikeout', 'walk', 'single','double', 'homerun',  'linear_weight']
        if batter is None:
            for category in categories:
                state_data[f'{category}s_so_far'] = np.nan
                state_data[f'{category}s_this_inning'] = np.nan
    
            # Calculate rolling sum for each category
            for category in categories:
                state_data[f'{category}s_last_3'] = np.nan
                state_data[f'{category}s_last_9'] = np.nan
    
            for category in categories:
                state_data[f'batter_{category}s_today'] = np.nan
       
            state_data['batter_last_ab'] = np.nan  
            state_data['last_ab'] = np.nan

        # Calculate cumulative sum for each category and each inning
        else:

            game_state_df = df.groupby('game_idx').agg(
                strikeouts_so_far = ('strikeout','sum'),
                batting_score = ('runs','sum'),
                walks_so_far = ('walk','sum'),
                singles_so_far = ('single','sum'),
                doubles_so_far = ('double','sum'),
                homeruns_so_far = ('homerun','sum'),
                linear_weights_so_far = ('linear_weight','sum'),
                last_ab = ('event', 'last')).reset_index()

            last_3_state_df = df.groupby('game_idx').tail(3).reset_index()
            if last_3_state_df.shape[0] == self.n_games * 3:
        
                last_3_state_df = last_3_state_df.groupby('game_idx').agg(
                    strikeouts_last_3 = ('strikeout','sum'),
                    walks_last_3 = ('walk','sum'),
                    singles_last_3 = ('single','sum'),
                    doubles_last_3 = ('double','sum'),
                    homeruns_last_3= ('homerun','sum'),
                    linear_weights_last_3 = ('linear_weight','sum')).reset_index()
            else:
                d={}
                d['game_idx'] = range(self.n_games)
                for k in categories:
                    d[k+'_last_3'] = np.nan
                last_3_state_df = pd.DataFrame(d)


            
            last_9_state_df = df.groupby('game_idx').tail(9).reset_index()

            if last_9_state_df.shape[0] == self.n_games * 9:
                last_9_state_df = last_9_state_df.groupby('game_idx').agg(
                    strikeouts_last_9 = ('strikeout','sum'),
                    walks_last_9 = ('walk','sum'),
                    singles_last_9 = ('single','sum'),
                    doubles_last_9 = ('double','sum'),
                    homeruns_last_9 = ('homerun','sum'),
                    linear_weights_last_9 = ('linear_weight','sum')).reset_index()
            else:
                d={}
                d['game_idx'] = range(self.n_games)
                for k in categories:
                    d[k+'_last_9'] = np.nan
                last_9_state_df = pd.DataFrame(d)
                
            inning_state_df = inning_df.groupby('game_idx').agg(
                strikeouts_this_inning = ('strikeout','sum'),
                walks_this_inning = ('walk','sum'),
                singles_this_inning = ('single','sum'),
                doubles_this_inning = ('double','sum'),
                homeruns_this_inning = ('homerun','sum'),
                linear_weights_this_inning = ('linear_weight','sum')).reset_index()
                


            if batter_df.empty:
 
                d={}
                d['batter_last_ab'] = np.nan
                d['game_idx'] = range(self.n_games)
                
                for k in categories:
                    d[f'batter_{k}s_today'] = np.nan
                    
                batter_df = pd.DataFrame(d)
                
            else:

                batter_df = batter_df.groupby('game_idx').agg(
                    batter_last_ab = ('event', 'last'),
                    batter_strikeouts_today= ('strikeout','sum'),
                    batter_walks_today= ('walk','sum'),
                    batter_singles_today= ('single','sum'),
                    batter_doubles_today = ('double','sum'),
                    batter_homeruns_today = ('homerun','sum'),
                    batter_linear_weights_today= ('linear_weight','sum')).reset_index()


            state_data = state_data.merge(batter_df, how = 'left', on='game_idx')
            state_data = state_data.merge(game_state_df, how = 'left', on='game_idx')
            state_data = state_data.merge(last_3_state_df, how = 'left', on='game_idx')
            state_data = state_data.merge(last_9_state_df, how = 'left', on='game_idx')
            
        #print("STATE DATA")
        #display(state_data.head())
        return state_data
        
    def reset_state(self):

        self.all_event_dicts = {
            'walk':[],
            'out':[],
            'strikeout':[],
            'single':[],
            'double':[],
            'triple':[],
            'homerun':[],
            'batter':[],
            'event':[],
            'runs':[],
            'inning':[],
            'game_idx':[]
            
        }

        
        self.inning=np.ones(self.n_games)
        self.outs = np.zeros(self.n_games)
        self.outs_recorded = np.zeros(self.n_games)

        self.runner_on_1b = np.zeros(self.n_games).astype(bool)
        self.runner_on_2b = np.zeros(self.n_games).astype(bool)
        self.runner_on_3b = np.zeros(self.n_games).astype(bool)
        
        self.runs = np.zeros(self.n_games)
        self.event_runs = np.zeros(self.n_games)
 
        self.state_df = self.update_state_for_event_model(None)

    def end_inning(self, end_inning_indices):
        self.runner_on_1b[end_inning_indices] = np.zeros(self.n_games)[end_inning_indices]
        self.runner_on_2b[end_inning_indices] = np.zeros(self.n_games)[end_inning_indices]
        self.runner_on_3b[end_inning_indices] = np.zeros(self.n_games)[end_inning_indices]
        self.outs[end_inning_indices] = np.zeros(self.n_games)[end_inning_indices]

    def record_field_out(self, out_indices):

        self.outs[out_indices]+=1
        self.outs_recorded[out_indices]+=1
        self.inning[out_indices] = 1 + self.outs_recorded[out_indices]//3

        score_from_third = np.random.uniform(0, 1, size=self.n_games)>.6
        
        self.event_runs[out_indices] = np.logical_and(
            self.outs<2, np.logical_and(self.runner_on_3b, score_from_third))[out_indices]

        self.runner_on_3b[out_indices] = np.logical_and(self.runner_on_3b,
                                                        ~score_from_third)[out_indices]

        moves_to_third = (np.logical_and(self.runner_on_2b, ~self.runner_on_3b) \
            * (np.random.uniform(0,1, size=self.n_games)>.8)).astype(bool)
        
        self.runner_on_3b[out_indices] = np.logical_or(self.runner_on_3b, 
                                                       moves_to_third)[out_indices]
        
        self.runner_on_2b[out_indices] = np.logical_and(self.runner_on_2b, 
                                                        ~moves_to_third)[out_indices]
        

        
        end_inning_indices = np.logical_and(out_indices, self.outs==3)
        self.end_inning(end_inning_indices)

    def record_strike_out(self, strikeout_indices):

        self.outs[strikeout_indices]+=1
        self.outs_recorded[strikeout_indices]+=1
        self.inning[strikeout_indices] = 1 + self.outs_recorded[strikeout_indices]//3

        end_inning_indices = np.logical_and(strikeout_indices, self.outs==3)
        self.end_inning(end_inning_indices)

    def record_single(self, single_indices):

        first_and_second = np.logical_and(self.runner_on_2b, self.runner_on_1b)

        score_from_second = (np.random.uniform(0,1,size=self.n_games)>.2)
        first_to_third = (np.random.uniform(0,1,size=self.n_games)>.4)
        first_to_third = np.logical_and(first_to_third, 
                                        ~np.logical_and(~score_from_second, first_and_second))
        
        self.event_runs[single_indices] += self.runner_on_3b[single_indices]
        self.event_runs[single_indices] += \
            (self.runner_on_2b*score_from_second)[single_indices]
        
        self.runner_on_3b[single_indices] = \
            ((self.runner_on_1b * first_to_third + \
            self.runner_on_2b * ~score_from_second)[single_indices]).astype(bool)

        
        self.runner_on_2b[single_indices] = np.logical_and(self.runner_on_1b,
                                                           ~first_to_third)[single_indices]
        self.runner_on_1b[single_indices] = 1
 
    def record_double(self, double_indices):

        score_from_first = (np.random.uniform(0,1,size=self.n_games)>.2)
        self.event_runs[double_indices] += self.runner_on_3b[double_indices]
        self.event_runs[double_indices] += self.runner_on_2b[double_indices]
        self.event_runs[double_indices] += \
            (self.runner_on_1b*score_from_first)[double_indices]

        self.runner_on_3b[double_indices] = (self.runner_on_1b* ~score_from_first)[double_indices]
        self.runner_on_2b[double_indices] = 1
        self.runner_on_1b[double_indices] = 0
        
    def record_homerun(self, homerun_indices):


        self.event_runs[homerun_indices] = \
            (1+self.runner_on_1b+self.runner_on_2b+self.runner_on_3b)[homerun_indices]
        
        self.runner_on_3b[homerun_indices]=0
        self.runner_on_2b[homerun_indices]=0
        self.runner_on_1b[homerun_indices]=0
 
    def record_walk(self,walk_indices):#game_idx):


        bases_loaded = \
            np.logical_and(self.runner_on_3b, np.logical_and(self.runner_on_2b, self.runner_on_1b))

        first_and_second = \
            np.logical_and(~self.runner_on_3b, np.logical_and(self.runner_on_2b, self.runner_on_1b))

        self.event_runs[walk_indices] = bases_loaded.astype(int)[walk_indices]

    
        # if theres runners on first and second, now theres a runner on third
        self.runner_on_3b[walk_indices] = np.logical_or(self.runner_on_3b, first_and_second)[walk_indices]
        self.runner_on_2b[walk_indices] = np.logical_or(self.runner_on_2b, self.runner_on_1b)[walk_indices]
        self.runner_on_1b[walk_indices] = np.ones(self.n_games)[walk_indices]
        
    
    def pull_from_game_check(self):

        if self.pull_pitcher_model:
            return self.pull_from_game_check_with_model()
        else:
            raise Exception('dont use rules anymore')
            return self.pull_from_game_check_with_rules()
    
    def pull_from_game_check_with_model(self):

        df = pd.DataFrame(self.all_event_dicts)
   
        df['hit'] = df[['single','double','homerun']].max(axis =1)
        df['curr_inning'] = df.groupby('game_idx').inning.transform(lambda x:x==x.max())
        inning_df = df[df.curr_inning]

        df=df.groupby('game_idx')\
            .agg(inning=('inning','max'),
                 runs=('runs','last'),
                 homerun=('homerun','last'),
                 pitcher_at_bat_number=('inning',len),
                 walk=('walk','last'),
                 single=('single','last'),
                 double=('double','last'),
                 strikeout=('strikeout','last'),
                 hit=('hit','last'),
                 out=('out','last'),
                 runs_this_game=('runs','sum'),
                 outs_recorded_this_game=('out','sum'),
                 hits_this_game=('hit','sum'),
                 homeruns_this_game=('homerun','sum'),
                 doubles_this_game=('double','sum'),
                singles_this_game=('single','sum'),
                walks_this_game=('walk','sum'),
                strikeouts_this_game=('strikeout','sum')).reset_index()

        
        inning_df=inning_df.groupby(['game_idx']).agg(
                runs_this_inning=('runs','sum'),
                 homeruns_this_inning=('homerun','sum'),
                 outs_this_inning=('out','sum'),
                 hits_this_inning=('hit','sum'),
                 doubles_this_inning=('double','sum'),
                singles_this_inning=('single','sum'),
                walks_this_inning=('walk','sum'),
                strikeouts_this_inning=('strikeout','sum')).reset_index()

        df = df.merge(inning_df[[x for x in inning_df if 'inning' in x or x=='game_idx']],
                 how = 'left', on = 'game_idx')

        s =time.time()
        columns = ['inning','runs','homerun','walk','strikeout','hit','out',
                   'pitcher_at_bat_number','runs_this_inning','runs_this_game',
                   'outs_this_inning','outs_recorded_this_game','homeruns_this_inning',
                   'homeruns_this_game','hits_this_inning', 'hits_this_game',
                   'walks_this_inning','walks_this_game','strikeouts_this_inning',
                   'strikeouts_this_game']


        for k,v in self.pull_pitcher_static.items():
            df[k] = v

        return df[columns + self.static_columns].values

    def pull_from_game_check_with_rules(self):
        runners = self.hits + self.walks
  
        if self.inning > 8:
            return True
        elif self.inning == 8 and (self.runs > 1 or runners > 6):
            return True
        elif self.inning == 7 and (self.runs > 2 or runners > 8):
            return True
        elif (self.inning == 5 or self.inning == 6) and (self.runs > 3 or runners > 9):
            return True
        elif self.inning == 4 and (self.runs > 5 or runners > 9):
            return True
        elif (self.inning == 3 or self.inning == 2) and self.runs > 5:
            return True
        elif self.inning == 1 and self.runs > 6:
            return True
            
        return False
        
    def print_game_state(self, event):

        if self.runner_on_1b and self.runner_on_2b and self.runner_on_3b:
            
            runners_str = 'bases loaded'
        elif self.runner_on_1b and self.runner_on_3b:
            runners_str = 'runners on first and third'
        elif self.runner_on_2b and self.runner_on_3b:
            runners_str = 'runners on second and third'
        elif self.runner_on_1b and self.runner_on_2b:
            runners_str = 'runners on first and second'
        elif self.runner_on_3b:
            runners_str = 'a runner on third'
        elif self.runner_on_2b:
            runners_str = 'a runner on second'
        elif self.runner_on_1b:
            runners_str = 'a runner on first'
        else:
            runners_str = 'bases empty'
            
        event_name = ['strikeout','field_out','walk','single','double','homerun'][event]
        des = f'''
            {event_name}. Now {self.runs} have been scored. We are in inning {self.inning} with
            {self.outs} outs and {runners_str}.'''

        print(' '.join(des.split()))
        
    def simulate(self, sim_arr, columns):

        self.reset_state()
        s1 = time.time()

        s = time.time()
        for batter in range(sim_arr.shape[0]):


            pitcher_data = sim_arr[batter]
            pitcher_data = pd.DataFrame(pitcher_data, columns = columns)
            pitcher_data = pitcher_data.drop(columns = \
                                [x for x in self.state_df.columns if x in pitcher_data.columns])

            
            pitcher_data['game_idx'] = range(self.n_games)

            pitcher_data = pitcher_data.merge(self.state_df, how = 'left', on = 'game_idx')

            event_probas = self.predict_event(pitcher_data, self.platt_models)

            #print('probas are', event_probas)
            events = []
            for probas in event_probas:
                event = np.random.multinomial(1, probas).argmax()
                events.append(event)

            homerun_indices = np.where(np.array(events)==5, True, False)
            double_indices = np.where(np.array(events)==4, True, False)
            single_indices = np.where(np.array(events)==3, True, False)
            walk_indices = np.where(np.array(events)==2, True, False)
            field_out_indices = np.where(np.array(events)==1, True, False)
            strikeout_indices = np.where(np.array(events)==0, True, False)
            
            self.all_event_dicts['batter'] += self.n_games*[batter]
            self.all_event_dicts['inning'] += list(self.inning)
            self.all_event_dicts['event'] += list(events)
            self.all_event_dicts['game_idx'] += list(range(self.n_games))

            
            self.record_strike_out(strikeout_indices)
            self.record_field_out(field_out_indices)
            self.record_walk(walk_indices)
            self.record_single(single_indices)
            self.record_double(double_indices)
            self.record_homerun(homerun_indices)

            out_indices = np.logical_or(field_out_indices, strikeout_indices)

            self.all_event_dicts['runs']+=list(self.event_runs)
            self.all_event_dicts['out']+=list(out_indices.astype(int))
            self.all_event_dicts['strikeout']+=list(strikeout_indices.astype(int))
            self.all_event_dicts['walk']+=list(walk_indices.astype(int))
            self.all_event_dicts['single']+=list(single_indices.astype(int))
            self.all_event_dicts['double']+=list(double_indices.astype(int))
            self.all_event_dicts['triple']+=[0]*self.n_games
            self.all_event_dicts['homerun']+=list(homerun_indices.astype(int))

            self.event_runs = np.zeros(self.n_games)

            #for game_idx, event in enumerate(events):    
            self.state_df = self.update_state_for_event_model((batter + 1) % 9)
                
            pull_arr = self.pull_from_game_check_with_model()
            self.pull_pitcher_data[:,batter, :] = pull_arr

        print('game', time.time()-s1)

        l=[]

        column_names =['runs','strikeouts','hits','homeruns','walks','outs_recorded']
        d={k:[] for k in  column_names}

        pull_pitcher_data = self.pull_pitcher_data.reshape(
            self.pull_pitcher_data.shape[0]*self.pull_pitcher_data.shape[1], -1)

        pull_probas = self.pull_pitcher_model.predict_proba(pull_pitcher_data)[:,1].round(2)
        random_numbers = np.random.uniform(0,1,len(pull_probas))
        pull_indices = pull_probas > random_numbers
        pull_indices = \
            pull_indices.reshape(self.pull_pitcher_data.shape[0],self.pull_pitcher_data.shape[1])
        
        pull_by_game = np.argmax(pull_indices, axis = 1)

        #print(pull_by_game)
        df = pd.DataFrame(self.all_event_dicts)
        df['hit'] = df[['single','double','triple','homerun']].max(axis=1)

        l = []
        for game_idx, pull_idx in enumerate(pull_by_game):
            sub_df = df[df.game_idx==game_idx].iloc[:pull_idx]
            l.append(sub_df)

        df = pd.concat(l)
        df = df[['runs','strikeout','hit','homerun','walk','out', 'game_idx']]\
            .rename(columns={'strikeout':'strikeouts','hit':'hits','homerun':'homeruns',
                             'walk':'walks','out':'outs_recorded'})\
            .groupby('game_idx')\
            .agg('sum')

        return df



def super_simple_simulation(pred_columns,
                            batter_columns,
                            df_features,
                            matchup_lookup,
                            df_meta, 
                            game_date,
                            pitcher, 
                            opposing_team,
                            game_sim_obj,
                            n_samples=10000):


    s1 = time.time()
    # can pull this out and do upfront for a lot of sims
    team_df = df_meta[np.logical_or(
        df_meta.home_team==opposing_team,
        df_meta.away_team==opposing_team)]
    
    team_df = team_df.sort_values('pitcher_at_bat_number')

    # closest game date available
    batter_game_date = team_df[team_df.game_date <= game_date].game_date.max()
    team_df = team_df[team_df.game_date==batter_game_date]

    # if home team is LAD then they hit in the bottom of the inning
    opposing_home = team_df.home_team.iloc[0] == opposing_team
    if opposing_home:
        team_df = team_df[team_df.inning_topbot=='Bot']
    else:
        team_df = team_df[team_df.inning_topbot=='Top']
        
    team_df = team_df.sort_values(['inning','pitcher_at_bat_number'])\
        .drop_duplicates('batter').head(9)

    lineup = team_df.batter.values

    #print(lineup)
    matchup_columns = [x for x in df_features.columns if 'matchup_' in x]


    df_features = df_features.sort_values(['team_at_bat_number', 'game_date'])
    lineup_df = df_features[
        np.logical_and(df_features.batter.isin(lineup),
                       df_features.game_date==batter_game_date)]\
        .drop_duplicates('batter').head(9)

    #display(lineup_df)
    # Sort the DataFrame based on the order of batters in the lineup list

    missing_batters = [x for x in lineup if x not in lineup_df.batter.values]
    if missing_batters != []:
        missing_batters_df = pd.concat([lineup_df.iloc[:1].copy(deep=True)]*len(missing_batters))
        missing_batters_df[batter_columns] = np.nan
        missing_batters_df['batter'] = missing_batters
        lineup_df = pd.concat([lineup_df, missing_batters_df])

    lineup_df = lineup_df.set_index('batter').loc[lineup].reset_index()

    lineup_df = lineup_df[batter_columns + ['batter']]
    
    pitcher_df = df_features[df_features.player_name==pitcher]

    n_at_bats = 4
    
    pitcher_game_date = pitcher_df[pitcher_df.game_date <=game_date].game_date.max()


    prior_pitcher_df = pitcher_df[pitcher_df.game_date <= pitcher_game_date]
    next_3_df = prior_pitcher_df[[x for x in pitcher_df.columns if 'next_3' in x] + ['game_date','pitcher_at_bat_number']]
    next_3_df = next_3_df.sort_values('game_date')
    next_3_df = \
        next_3_df.groupby('pitcher_at_bat_number').last().reset_index().sort_values('pitcher_at_bat_number')

    next_3_df = next_3_df.iloc[:n_at_bats*9]
    rows_to_add = n_at_bats*9 - next_3_df.shape[0]

    if rows_to_add > 0:
        # Create a DataFrame with the rows_to_add rows, filled with NaN
        new_rows = pd.DataFrame(np.nan, index=range(rows_to_add), columns=next_3_df.columns)
        
        # Concatenate the original DataFrame with the new rows DataFrame
        next_3_df = pd.concat([next_3_df, new_rows], ignore_index=True)

    pitcher_df = pitcher_df[pitcher_df.game_date == pitcher_game_date].iloc[:1]
    pitcher_df = pitcher_df.drop(columns = batter_columns+matchup_columns)
    pitcher_df = pitcher_df\
        .drop_duplicates('pitcher_at_bat_number')\
        .sort_values('pitcher_at_bat_number')

    pitcher_df = pitcher_df.reset_index().sort_values('pitcher_at_bat_number')
    pitcher_df = pd.DataFrame(np.tile(pitcher_df.values, (9,1)),
                                  columns=pitcher_df.columns)
    pitcher_df['batter'] = list(lineup)

    pitcher_df = pd.DataFrame(np.tile(pitcher_df.values, (n_at_bats,1)),
                                  columns=pitcher_df.columns)

    for x in [x for x in pitcher_df.columns if 'next_3' in x]:
        pitcher_df[x] = next_3_df[x]

    pitcher_df['pitcher_at_bat_number'] = range(1,pitcher_df.shape[0]+1)
    pitcher_df['team_at_bat_number'] = range(1,pitcher_df.shape[0]+1)
    pitcher_df['lineup_slot'] = 1+np.array(range(0,pitcher_df.shape[0])) % 9
    pitcher_df['time_thru_the_order'] = np.array(pitcher_df.pitcher_at_bat_number) //9 + 1

    pitcher_df['batter'] = pitcher_df['batter'].astype(int)
    pitcher_df['pitcher'] = pitcher_df['pitcher'].astype(int)

    matchup_lookup['batter'] = matchup_lookup['batter'].astype(int)
    matchup_lookup['pitcher'] = matchup_lookup['pitcher'].astype(int)
    pitcher_df = pd.merge_asof(pitcher_df, matchup_lookup.sort_values('game_date'),
                  on = 'game_date',
                  direction='backward',
                  by = ['batter','pitcher'])

    pitcher_df_rep = pd.DataFrame(np.tile(pitcher_df.values, (n_samples,1)),
                                  columns=pitcher_df.columns)
    pitcher_df_rep['game_number'] = sum([[i]*9*n_at_bats for i in range(n_samples)], [])

    at_bat_df = pitcher_df_rep.merge( lineup_df, how = 'left', on = 'batter')

    at_bat_df['homerun_intensity'] = at_bat_df['homerun_batter'].fillna(0) +\
        at_bat_df['homerun_pitcher'].fillna(0)
    
    at_bat_df['walk_intensity'] = at_bat_df['walk_batter'].fillna(0) +\
        at_bat_df['walk_pitcher'].fillna(0)
    
    at_bat_df['strikeout_intensity'] = at_bat_df['strikeout_batter'].fillna(0) +\
        at_bat_df['strikeout_pitcher'].fillna(0)
    
    at_bat_df = at_bat_df.sort_values(['pitcher_at_bat_number', 'game_number'])

    df_columns = at_bat_df.columns
    at_bat_df = at_bat_df.values.reshape(9*n_at_bats, n_samples, -1)

    
    d = game_sim_obj.simulate(at_bat_df, df_columns)

    
    print('total time is', time.time()-s1)
    return pd.DataFrame(d)

def plot_player_distr(sims):
    
    strikeouts = sims['strikeouts']
    homeruns = sims['homeruns']
    walks = sims['walks']
    runs = sims['runs']
    
    # Determine common bin range
    max_value = max(strikeouts.max(), homeruns.max(), walks.max(), runs.max())
    min_value = min(strikeouts.min(), homeruns.min(), walks.min(), runs.min())
    
    bins = range(int(min_value), int(max_value) + 1)

    
    plt.hist(sims['strikeouts'], bins=bins, color='g', alpha = .3, edgecolor='white')
    plt.tight_layout()
    plt.xlabel("strikeouts")
    plt.show()
    #plt.hist(sims['double'], color='y', alpha = .3)
    plt.hist(sims['homeruns'], bins=bins, color='r', alpha = .3, edgecolor='white')
    plt.tight_layout()
    plt.xlabel("homeruns")
    plt.show()
    plt.hist(sims['walks'], bins=bins, color='y', alpha = .3, edgecolor='white')
    plt.xlabel("walks")
    plt.tight_layout()
    plt.show()
    plt.hist(sims['runs'], bins=bins, color='black', alpha = .3, edgecolor='white')
    plt.xlabel("runs")
    plt.tight_layout()
    plt.show()

def process_row(row, game_sim_obj, 
                xg_model_pull,
                data_for_pitcher_pull, 
                train_data,
                matchup_lookup,
                train_pitches,
                dynamic_columns,
                batter_columns,
                n_samples = 100):
 
    try:
        curr_pitcher_pull = data_for_pitcher_pull[np.logical_and(
            data_for_pitcher_pull.player_name == row['player_name'],
            data_for_pitcher_pull.game_date == row['game_date'])]

        curr_pitcher_pull = curr_pitcher_pull.iloc[0:1]
        if curr_pitcher_pull.empty:
            raise Exception("No game found")

     
        meta = ['pitcher','player_name','game_date','batter']
        target = 'pulled'
        pull_pitcher_cols = [x for x in data_for_pitcher_pull.columns if x not in meta+[target]]
        game_sim_obj.setup_pull_pitcher_model(xg_model_pull,
                                              most_recent_game_df=curr_pitcher_pull,
                                              pull_pitcher_columns=pull_pitcher_cols)

        sim = super_simple_simulation(
            dynamic_columns,
            batter_columns,
            train_data,
            matchup_lookup,
            train_pitches,
            game_date=row['game_date'],
            pitcher=row['player_name'],
            opposing_team=row['opposing_team'],
            game_sim_obj=game_sim_obj,
            n_samples=n_samples)

        sim['player_name'] = row['player_name']
        sim['game_date'] = row['game_date']
        sim['opposing_team'] = row['opposing_team']
        sim['strikeout_label'] = row['strikeout']
        return sim
    except:
        print('couldnt find for player', row['player_name'], 'and date', row['game_date'])
        traceback.print_exc()
        return None

def callback(result):
    with lock:
        sims.append(result)
        with open('sims.pkl', 'wb') as f:
            pickle.dump(sims, f)
            
meta_columns = ['player_name','pitcher','batter', 'game_date','inning_topbot','batting_team']

additional_batter_cols = ['estimated_iso_mean',
 'estimated_iso_95th',
 'estimated_iso_75th',
 'estimated_iso_25th',
 'launch_angle_mean',
 'launch_speed_mean',
 'launch_speed_max',
 'launch_speed_95th',
 'launch_speed_75th',
 'launch_speed_25th',
 'pct_barrel',
 'pct_solid_contact',
 'pct_exit_velo_above_95_mph',
 'pct_exit_velo_above_100_mph',
 'pct_homerun_launch_angle',
 'num_barrel',
 'num_solid_contact',
 'num_exit_velo_above_95_mph',
 'num_exit_velo_above_100_mph',
 'num_homerun_launch_angle',
 'pct_hit_distance_above_350',
 'num_hit_distance_above_350',
 'pct_hit_distance_above_375',
 'num_hit_distance_above_375',
 'pct_hit_distance_above_400',
 'num_hit_distance_above_400',
 'pct_hit_distance_above_420',
 'num_hit_distance_above_420',
 'estimated_iso_mean_regressed',
 'estimated_iso_95th_regressed',
 'estimated_iso_75th_regressed',
 'estimated_iso_25th_regressed',
 'launch_angle_mean_regressed',
 'launch_speed_mean_regressed',
 'launch_speed_max_regressed',
 'launch_speed_95th_regressed',
 'launch_speed_75th_regressed',
 'launch_speed_25th_regressed',
 'pct_barrel_regressed',
 'pct_solid_contact_regressed',
 'pct_exit_velo_above_95_mph_regressed',
 'pct_exit_velo_above_100_mph_regressed',
 'pct_homerun_launch_angle_regressed',
 'num_barrel_regressed',
 'num_solid_contact_regressed',
 'num_exit_velo_above_95_mph_regressed',
 'num_exit_velo_above_100_mph_regressed',
 'num_homerun_launch_angle_regressed',
 'pct_hit_distance_above_350_regressed',
 'num_hit_distance_above_350_regressed',
 'pct_hit_distance_above_375_regressed',
 'num_hit_distance_above_375_regressed',
 'pct_hit_distance_above_400_regressed',
 'num_hit_distance_above_400_regressed',
 'pct_hit_distance_above_420_regressed',
 'num_hit_distance_above_420_regressed']

state_columns = [
        'inning',
       'on_1b', 'on_2b', 'on_3b', 'outs_when_up', 'strikeouts_so_far',
       'strikeouts_this_inning', 'walks_so_far', 'walks_this_inning',
       'homeruns_so_far', 'homeruns_this_inning', 'doubles_so_far',
       'doubles_this_inning', 'singles_so_far', 'singles_this_inning',
       'linear_weights_so_far', 'linear_weights_this_inning',
       'strikeouts_last_3', 'strikeouts_last_9', 'walks_last_3',
       'walks_last_9', 'homeruns_last_3', 'homeruns_last_9', 'doubles_last_3',
       'doubles_last_9', 'singles_last_3', 'singles_last_9',
       'linear_weights_last_3', 'linear_weights_last_9',
       'batter_strikeouts_today',
       'batter_strikeouts_against_this_pitcher_total', 'batter_walks_today',
       'batter_walks_against_this_pitcher_total', 'batter_homeruns_today',
       'batter_homeruns_against_this_pitcher_total', 'batter_doubles_today',
       'batter_doubles_against_this_pitcher_total', 'batter_singles_today',
       'batter_singles_against_this_pitcher_total',
       'batter_linear_weights_today',
       'batter_linear_weights_against_this_pitcher_total',
       'num_matchups_batter_pitcher', 'batter_last_ab', 'last_ab']


if __name__ == '__main__':
    
    train_pitches = pd.read_parquet('data_with_pitch_values.parquet')
    train_pitches = train_pitches[train_pitches.game_date>=pitch_model_train_cutoff]
    
    train_pitches = train_pitches.sort_values(['game_date','at_bat_number','pitch_number'])
    
    train_pitches['at_bat_change'] = \
        (train_pitches.groupby(['pitcher','game_date']).at_bat_number.diff().fillna(0)!=0)
    
    train_pitches['pitcher_at_bat_number'] = train_pitches\
        .groupby(['pitcher','game_date']).at_bat_change.transform('cumsum') + 1
    
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
    
    from engineer_features import train_xgboost
    meta = ['pitcher','player_name','game_date','batter']
    target = 'pulled'
    
    pull_pitcher_cols = [x for x in data_for_pitcher_pull.columns if x not in meta+[target]]
    X_pulled_train = data_for_pitcher_pull[data_for_pitcher_pull.game_date<='2023-06-01']
    X_pulled_test = data_for_pitcher_pull[data_for_pitcher_pull.game_date>='2023-06-01']
    
    
    xg_model_pull = train_xgboost(data_for_pitcher_pull,
                  target_column='pulled',
                  pred_columns=pull_pitcher_cols,
                  categorical_columns=None,
                  multiclass=False)

    
    with open('data_for_ordinal_fangraphs.pkl','rb') as f:
        d = pickle.load(f)
    
    meta_cols =['player_name','pitcher','batter','game_date','batting_team']
    xtrain_enc = d['x_train']
    xmeta = d['x_meta_train'][meta_cols]
    train_data = pd.concat([xmeta.reset_index(drop=True), xtrain_enc.reset_index(drop=True)], axis = 1)

    
    with open('booster_event_models/best_model_history.pkl','rb') as f:
        d = pickle.load(f)
    
    xg_model = d['model']
    
    matchup_lookup = pd.read_parquet('matchup_lookup.parquet')
    
    matchup_lookup.pitcher = matchup_lookup.pitcher.astype(int)
    matchup_lookup.batter = matchup_lookup.batter.astype(int)
    train_data.pitcher = train_data.pitcher.astype(int)
    train_data.batter = train_data.batter.astype(int)
    
    matchup_lookup['game_date'] = pd.to_datetime([str(y)+'-'+str(m) 
                                    for y, m in zip(matchup_lookup.year, matchup_lookup.month)])
    
    
    
    pred_columns = [x for x in train_data.columns 
                        if x!='label' and x not in meta_columns and x!='index']
    
    dynamic_columns = [x for x in pred_columns if x not in state_columns]
    
    agg_batter_columns = [x for x in train_data.columns 
                          if ('_batter' in x and '_batters' not in x) and 'matchup' not in x]
    batter_columns = [x for x in additional_batter_cols if x!='team_at_bat_number']+agg_batter_columns
    
    
    game_map = train_pitches

    game_map['min_inning'] = game_map.groupby(['game_date','pitcher']).inning.transform('min')
    game_map = game_map[game_map.min_inning == 1]

    outcomes = ['walk', 'strikeout', 'homerun', 'runs', 'out']
    game_map = game_map[
        ['game_date', 'home_team', 'away_team', 'inning_topbot','player_name']+outcomes]\
        .groupby(['game_date','home_team','away_team','player_name','inning_topbot'])\
        .agg('sum').reset_index()
    
    
    game_map['opposing_team'] = np.where(game_map.inning_topbot=='Top',
                                         game_map.away_team,
                                         game_map.home_team)
    
    
    game_map = game_map[game_map.game_date>='2022-01-01']
    game_map = game_map[game_map.game_date>='2023-06-18']
    #game_map = game_map[np.logical_or(game_map.game_date <= '2023-01-01',
    #                                  game_map.game_date>='2023-04-10')]

    n_samples = 100
    game_sim_obj = Game(n_samples)
    game_sim_obj.setup_event_model(xg_model,
                               pred_columns=pred_columns,
                               categorical_columns=['last_ab', 'batter_last_ab'],
                               one_hot_encoder=None, platt_models=None)
    

    process_row_partial = partial(process_row, game_sim_obj=game_sim_obj,
                                 xg_model_pull=xg_model_pull,
                                 data_for_pitcher_pull=data_for_pitcher_pull,
                                 train_data=train_data, matchup_lookup=matchup_lookup,
                                 train_pitches=train_pitches, dynamic_columns=dynamic_columns,
                                 batter_columns=batter_columns)

    all_results = []
    #with Pool(32) as p:
    chunk_size = 10
    for idx_start in range(0, game_map.shape[0], chunk_size):

        print('chunk',idx_start)
        idx_end = idx_start+chunk_size
        chunk = game_map.iloc[idx_start:idx_end]

        results = []
        for idx, x in chunk.iterrows():
            print(x)
            results.append(process_row_partial(x))

        #rows = [row for idx, row in chunk.iterrows()]
        #results = p.map(process_row_partial, rows)
            
        all_results += results
        print('done with chunk',idx_start)
        with open('sims2.pkl','wb') as f:
            pickle.dump(all_results,f)
    
    print(f'Processed simulations')
    