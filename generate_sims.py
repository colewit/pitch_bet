import numpy as np
import pandas as pd

import copy
import time
import sys
import pickle
import traceback
from functools import partial

import matplotlib.pyplot as plt
import xgboost as xgb

from pitcher_pull_model import prep_data_for_pitcher_pull
from model_scripts.ordinal_losses import sigmoid
from data_pipeline.assemble_event_model_data import *
from config import pitch_model_train_cutoff

pd.set_option('future.no_silent_downcasting', True)

pd.options.display.max_columns = 1000


def predict_xgboost(xg_model, data, pred_columns,
                    categorical_columns, one_hot_encoder=None,
                    proba=False, binary_proba=True):

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
    else:
        preds = xg_model.predict(dmat)
    
    return preds


def predict_xgboost_ordinal(xg_model, data):

    data = data[xg_model.feature_names].astype(float).values

    # faster for this purpose than predict as we dont need to convert to dmatrix
    # margin assures raw logits
    logits = xg_model.inplace_predict(data, predict_type='margin')

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

    # adds homer proba
    probas =np.hstack([probas, 1- probas.sum(axis=1).reshape(-1,1)])
    return probas
    
class Game(object):

    def __init__(self, n_games):

        self.n_games = n_games
        self.reset_state()
        self.pull_pitcher_model = None

    def setup_event_model(self,event_model):

        self.model = event_model

    def predict_event(self, pitcher_data):

        event_probas = predict_xgboost_ordinal(self.model, pitcher_data)
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
    

        
    def reset_state(self):

        self.curr_inning_start = np.zeros(self.n_games)
        self.inning=np.ones(self.n_games)
        self.outs = np.zeros(self.n_games)
        self.outs_recorded = np.zeros(self.n_games)

        self.runner_on_1b = np.zeros(self.n_games).astype(bool)
        self.runner_on_2b = np.zeros(self.n_games).astype(bool)
        self.runner_on_3b = np.zeros(self.n_games).astype(bool)
        
        
        self.event_runs = np.zeros(self.n_games)
 
        self.state_df = self.update_state_for_event_model(None)

        self.runs_arr = np.zeros((self.n_games, 36)).astype(int)
        self.events_arr = np.zeros((self.n_games, 36)).astype(int)
        self.linear_weights_arr = np.zeros((self.n_games, 36))


    def end_inning(self, end_inning_indices):
        self.runner_on_1b[end_inning_indices] = np.zeros(self.n_games)[end_inning_indices]
        self.runner_on_2b[end_inning_indices] = np.zeros(self.n_games)[end_inning_indices]
        self.runner_on_3b[end_inning_indices] = np.zeros(self.n_games)[end_inning_indices]
        self.outs[end_inning_indices] = np.zeros(self.n_games)[end_inning_indices]
        self.curr_inning_start[end_inning_indices] = self.batter+1

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
 
    def record_walk(self,walk_indices):


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
        
        def zero_out_before_inning(events, start_of_curr_inning, fill):
            # Create a mask for each row where the indices are to be zeroed

            events = copy.deepcopy(events)
            m = events.shape[0]
            n = events.shape[1]
            
            rows = np.arange(m)[:, np.newaxis]  # Shape (m, 1)
            cols = np.arange(n)  # Shape (n,)
            mask = cols < np.array(start_of_curr_inning)[rows]  # Broadcasting L to shape (m, n)
        
            # Apply the mask to zero out elements in A
            events[mask] = fill
            return events
            
        columns = ['strikeouts_this_game','fieldouts_this_game','walks_this_game',
                   'singles_this_game','doubles_this_game','homeruns_this_game']
        columns_last = ['strikeout','out','walk', 'single','double','homerun']


        counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=6),
                                     axis=1, arr=self.events_arr[:,:self.batter+1])
        counts_last = np.apply_along_axis(lambda x: np.bincount(x, minlength=6),
                                     axis=1, arr=self.events_arr[:,self.batter:self.batter+1])
        
        df = pd.DataFrame(counts, columns = columns)
        df_last = pd.DataFrame(counts_last, columns = columns_last)

        df['runs_this_game'] = self.runs_arr[:, :self.batter+1].sum(axis =1)
        df['hits_this_game'] = df[['singles_this_game','doubles_this_game','homeruns_this_game']].sum(axis =1)
        df['outs_recorded_this_game'] = df[['fieldouts_this_game','strikeouts_this_game']].sum(axis =1)
        df['inning'] = self.inning
        df['runs'] = self.event_runs

        df_last['out'] = df_last['strikeout']+df_last['out']
        df_last['hit'] = df_last[['single','double','homerun']].sum(axis =1)
        df=pd.concat([df.reset_index(drop=True),df_last.reset_index(drop=True)], axis =1)

        inning_events_arr = zero_out_before_inning(self.events_arr,
                                                   start_of_curr_inning=self.curr_inning_start, fill=6)
        
        columns = ['strikeouts_this_inning','fieldouts_this_inning','walks_this_inning',
                   'singles_this_inning','doubles_this_inning','homeruns_this_inning']

        # count of 6s is a dummy count. we get rid of it
        counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=7),
                                         axis=1, arr=inning_events_arr[:,:self.batter+1])
        counts = counts[:,:-1]
            
        inning_df = pd.DataFrame(counts, columns = columns)  
        
        inning_df['hits_this_inning'] = inning_df[
            ['singles_this_inning','doubles_this_inning','homeruns_this_inning']].sum(axis =1)
        
        inning_df['outs_this_inning'] = inning_df['fieldouts_this_inning']+inning_df['strikeouts_this_inning']

        # zero out entries that arent from this inning.
        inning_runs_arr = zero_out_before_inning(self.runs_arr, 
                                                 start_of_curr_inning=self.curr_inning_start, fill=0)

        inning_df['runs_this_inning'] = inning_runs_arr.sum(axis = 1)
        
        df=pd.concat([df.reset_index(drop=True), inning_df.reset_index(drop=True)], axis =1)
        df['pitcher_at_bat_number'] = self.batter
        
        s =time.time()
        columns = ['inning','runs','homerun','walk','strikeout','hit','out',
                   'pitcher_at_bat_number','runs_this_inning','runs_this_game',
                   'outs_this_inning','outs_recorded_this_game','homeruns_this_inning',
                   'homeruns_this_game','hits_this_inning', 'hits_this_game',
                   'walks_this_inning','walks_this_game','strikeouts_this_inning',
                   'strikeouts_this_game']


        for k,v in self.pull_pitcher_static.items():
            df[k] = v

        return df[columns + self.static_columns]
    
    def update_state_for_event_model(self, batter):

        def zero_out_before_inning(events, start_of_curr_inning):
            # Create a mask for each row where the indices are to be zeroed

            events = copy.deepcopy(events)
            m = events.shape[0]
            n = events.shape[1]
            
            rows = np.arange(m)[:, np.newaxis]  # Shape (m, 1)
            cols = np.arange(n)  # Shape (n,)
            mask = cols < np.array(start_of_curr_inning)[rows]  # Broadcasting L to shape (m, n)
        
            # Apply the mask to zero out elements in A
            events[mask] = 6
            return events

        state_data = {}

        state_data['game_idx'] = range(self.n_games)
        state_data['inning'] = self.inning
        state_data['outs_when_up'] = self.outs
        state_data['on_1b'] = self.runner_on_1b
        state_data['on_2b'] = self.runner_on_2b
        state_data['on_3b'] = self.runner_on_3b

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

            columns = ['strikeouts_so_far','fieldouts_so_far','walks_so_far',
                       'singles_so_far','doubles_so_far','homeruns_so_far']
            
            counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=6),
                                         axis=1, arr=self.events_arr[:,:batter])
            
            game_state_df = pd.DataFrame(counts, columns = columns)

            game_state_df['batting_score'] = self.runs_arr[:, :batter].sum(axis = 1)
            game_state_df['last_ab'] = self.events_arr[:,batter-1]
            game_state_df['linear_weights_so_far'] = self.linear_weights_arr[:, :batter].sum(axis = 1)

            if batter >= 2:

                columns = ['strikeouts_last_3','fieldouts_last_3','walks_last_3',
                       'singles_last_3','doubles_last_3','homeruns_last_3']
            
                counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=6),
                                             axis=1, arr=self.events_arr[:,(batter) -3: (batter)])
                
                last_3_state_df = pd.DataFrame(counts, columns = columns)
  
            else:
                d={}
                for k in categories:
                    d[k+'_last_3'] = [np.nan]*self.n_games
                last_3_state_df = pd.DataFrame(d)
                last_3_state_df['linear_weights_last_3'] = self\
                    .linear_weights_arr[:,(batter) -3: (batter)]\
                    .sum(axis = 1)

            if batter >= 8:

                columns = ['strikeouts_last_9','fieldouts_last_9','walks_last_9',
                       'singles_last_9','doubles_last_9','homeruns_last_9']
            
                counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=6),
                                             axis=1, arr=self.events_arr[:,(batter) -9: (batter)])
                
                last_9_state_df = pd.DataFrame(counts, columns = columns)
                last_9_state_df['linear_weights_last_9'] = self\
                    .linear_weights_arr[:,(batter) -9: (batter)]\
                    .sum(axis = 1)

            else:
                d={}
                for k in categories:
                    d[k+'_last_9'] = [np.nan]*self.n_games
                last_9_state_df = pd.DataFrame(d)

            
            inning_events_arr = zero_out_before_inning(self.events_arr, start_of_curr_inning=self.curr_inning_start)
            columns = ['strikeouts_this_inning','fieldouts_this_inning','walks_this_inning',
                       'singles_this_inning','doubles_this_inning','homeruns_this_inning']

            # count of 6s is a dummy count. we get rid of it
            counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=7),
                                             axis=1, arr=inning_events_arr[:,:batter])
            counts = counts[:,:-1]
                
            inning_state_df = pd.DataFrame(counts, columns = columns)
            inning_state_df['linear_weights_this_inning'] = .55*inning_state_df.walks_this_inning+ \
                .7*inning_state_df.singles_this_inning+1*inning_state_df.doubles_this_inning+\
                1.65*inning_state_df.homeruns_this_inning
            
            if batter < 9:
 
                d={}
                d['batter_last_ab'] = [np.nan]*self.n_games

                for k in categories:
                    d[f'batter_{k}s_today'] = [np.nan]*self.n_games
                    
                batter_df = pd.DataFrame(d)
                
            else:
                batter_last_ab = self.events_arr[:, batter - 9]

                indices = list(range(batter % 9, batter, 9))
           
                columns = ['batter_strikeouts_today','batter_fieldouts_today','batter_walks_today',
                           'batter_singles_today','batter_doubles_today','batter_homeruns_today']
                
                counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=6),
                                                 axis=1, arr=self.events_arr[:,indices])
                    
                batter_df = pd.DataFrame(counts, columns = columns)
                batter_df['batter_last_ab'] = batter_last_ab
                batter_df['batter_linear_weights_today'] = self.linear_weights_arr[:, indices].sum(axis = 1)


            state_data = pd.concat([state_data.reset_index(drop=True), 
                       batter_df.reset_index(drop=True),
                       game_state_df.reset_index(drop=True),
                       inning_state_df.reset_index(drop=True),
                       last_3_state_df.reset_index(drop=True),
                       last_9_state_df.reset_index(drop=True)], axis = 1)

        return state_data   
        
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

        N = sim_arr.shape[0]
        for batter in range(N):

            s = time.time()
            self.batter = batter
            
            pitcher_data = sim_arr[batter]
            pitcher_data = pd.DataFrame(pitcher_data, columns = columns)
            pitcher_data = pitcher_data.drop(columns = \
                                [x for x in self.state_df.columns if x in pitcher_data.columns])


            pitcher_data = pd.concat([pitcher_data.reset_index(drop=True), 
                                      self.state_df.reset_index(drop=True)], axis = 1)

            event_probas = self.predict_event(pitcher_data)

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

            self.record_strike_out(strikeout_indices)
            self.record_field_out(field_out_indices)
            self.record_walk(walk_indices)
            self.record_single(single_indices)
            self.record_double(double_indices)
            self.record_homerun(homerun_indices)

            self.runs_arr[:,batter] = self.event_runs
            
            self.events_arr[:, batter] = np.array(events).astype(int)
            self.linear_weights_arr[:, batter] = \
                .55*walk_indices+.7*single_indices+1*double_indices+1.65*homerun_indices

            self.state_df = self.update_state_for_event_model(batter + 1)
            pull_row = self.pull_from_game_check_with_model()

            self.event_runs = np.zeros(self.n_games)
            self.pull_pitcher_data[:,batter, :] = pull_row.values
            
        print('game', time.time()-s1)

         
        s=time.time()


        pull_pitcher_data = self.pull_pitcher_data.reshape(
            self.pull_pitcher_data.shape[0]*self.pull_pitcher_data.shape[1], -1)

        pull_probas = self.pull_pitcher_model.predict_proba(pull_pitcher_data)[:,1].round(2)
        random_numbers = np.random.uniform(0,1,len(pull_probas))
        pull_indices = pull_probas > random_numbers
        pull_indices = \
            pull_indices.reshape(self.pull_pitcher_data.shape[0],self.pull_pitcher_data.shape[1])
        
        pull_by_game = np.argmax(pull_indices, axis = 1)

        def zero_out_after_game_end(events, end_of_game):
            # Create a mask for each row where the indices are to be zeroed

            events = copy.deepcopy(events)
            m = events.shape[0]
            n = events.shape[1]
            
            rows = np.arange(m)[:, np.newaxis]  # Shape (m, 1)
            cols = np.arange(n)  # Shape (n,)
            mask = cols > np.array(end_of_game)[rows]  # Broadcasting L to shape (m, n)
        
            # Apply the mask to zero out elements in A
            events[mask] = 0
            return events

        end_of_game_arr = zero_out_after_game_end(self.events_arr, pull_by_game)
        columns = ['strikeouts','fieldouts','walks', 'singles','doubles','homeruns']

        # count of 6s is a dummy count. we get rid of it
        counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=6),
                                             axis=1, arr=end_of_game_arr[:,:batter])
                
        df = pd.DataFrame(counts, columns = columns)
        df['hits'] = df[['singles','doubles','homeruns']].sum(axis=1)
        df['outs_recorded'] = df.fieldouts+df.strikeouts

        
        return df

def coerce_numeric(df):
    def safe_to_numeric(series):
        try:
            return pd.to_numeric(series)
        except (ValueError, TypeError):
            return series
        
    if 'index' in df.columns:
        df=df.drop(columns='index')
    for col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].dtype == object and df[col].astype(str).str.endswith('%').any():
                # Handle string percentages
                df[col] = pd.to_numeric(df[col].astype(str).str.rstrip('%')).div(100)
            else:
                # Coerce other columns to numeric if possible
                df[col] = safe_to_numeric(df[col])
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
    next_3_df = prior_pitcher_df[[x for x in pitcher_df.columns if 'next_3' in x] + \
        ['game_date','pitcher_at_bat_number']]
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

    at_bat_df = coerce_numeric(at_bat_df)

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
        time.sleep(3)
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

    run_name = '2024'
    model_run_name = 'pilot'

    train_pitches = pd.read_parquet(f'intermediate_data_files/{run_name}/data_with_pitch_values.parquet')
    
    data_for_pitcher_pull = prep_data_for_pitcher_pull(train_pitches)
    
    

    
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
        
    print(data_for_pitcher_pull.game_date.max(), 'is pull date max')
    print(train_pitches.game_date.max(), 'is train pitches date max')
    with open(f'pull_pitcher_models/{model_run_name}/model.pkl','rb') as f:
        xg_model_pull = pickle.load(f)

    
    with open(f'intermediate_data_files/{run_name}/data_for_ordinal_fangraphs.pkl','rb') as f:
        d = pickle.load(f)
    
    meta_cols =['player_name','pitcher','batter','game_date','batting_team']
    xtrain_enc = d['X']
    xmeta = d['X_meta'][meta_cols]
    train_data = pd.concat([xmeta.reset_index(drop=True), xtrain_enc.reset_index(drop=True)], axis = 1)

    
    with open(f'booster_event_models/{model_run_name}/best_model_history.pkl','rb') as f:
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
    
    

    game_map = game_map[game_map.game_date>='2024-01-01']
    #game_map = game_map[np.logical_or(game_map.game_date <= '2023-01-01',
    #                                  game_map.game_date>='2023-04-10')]

    n_samples = 100
    game_sim_obj = Game(n_samples)
    game_sim_obj.setup_event_model(xg_model)
    

    process_row_partial = partial(process_row, game_sim_obj=game_sim_obj,
                                 xg_model_pull=xg_model_pull,
                                 data_for_pitcher_pull=data_for_pitcher_pull,
                                 train_data=train_data, matchup_lookup=matchup_lookup,
                                 train_pitches=train_pitches, dynamic_columns=dynamic_columns,
                                 batter_columns=batter_columns)

    all_results = []
    chunk_size = 10
    for idx_start in range(0, game_map.shape[0], chunk_size):

        print('chunk',idx_start)
        idx_end = idx_start+chunk_size
        chunk = game_map.iloc[idx_start:idx_end]

        results = []
        for idx, x in chunk.iterrows():
            print(x)
            results.append(process_row_partial(x))

        all_results += results
        print('done with chunk',idx_start)
        with open('sims_2024.pkl','wb') as f:
            pickle.dump(all_results,f)
    
    print(f'Processed simulations')
    