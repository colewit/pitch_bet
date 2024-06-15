
import pandas as pd
import numpy as np
import dill as pickle
import os
import joblib
import time
import gc
import sys
import psutil

from spacecutter.models import OrdinalLogisticModel
from spacecutter.losses import CumulativeLinkLoss
from spacecutter.callbacks import AscensionCallback

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from skorch.callbacks import EarlyStopping

from skorch.callbacks import Callback, ProgressBar, Checkpoint
from skorch.net import NeuralNet
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


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



class CustomMinMaxScaler:
    def __init__(self, fill_value=-1):
        #self.scaler = MinMaxScaler()
        self.fill_value = fill_value
        self.already_fit = False

    def fit(self, X):

        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0) 

        self.already_fit = True

    def transform(self, X):

        X = (X - self.means) / self.stds
        # Restore NaNs in the scaled data

        if isinstance(X, np.ndarray):
            mask = np.isnan(X)
            X[mask] = self.fill_value
            
        elif isinstance(X, torch.Tensor):
            X = torch.nan_to_num(X, nan=-1)
        
        
        return X

    def fit_transform(self, X, y=None):

        if not self.already_fit:
            self.fit(X)

        return self.transform(X)

def weighted_cross_entropy_with_distances_proba(class_distances):
    """
    Compute the weighted cross-entropy loss with distances between classes using predicted probabilities.

    Args:
    y_true (np.ndarray): True labels.
    y_pred_proba (np.ndarray): Predicted probabilities.
    class_distances (np.ndarray): Distance matrix between classes.

    Returns:
    float: Weighted cross-entropy loss.
    """

    def scorer(y_true, y_pred):
    # Convert y_true to tensor

        y_true = torch.tensor(np.array(y_true).astype(int), dtype=torch.int).squeeze()

        # Convert y_pred_proba to tensor
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        arr = y_pred[range(len(y_true)), y_true] + 1e-6

        # Calculate the cross-entropy loss for each class pair
        cross_entropy_loss = -torch.log(arr)

        # Calculate the weights based on the distances and predicted probabilities
        weights = torch.zeros_like(cross_entropy_loss)
        for i in range(len(y_true)):
            for j in range(y_pred.shape[1]):
                weights[i] += class_distances[y_true[i], j] * y_pred[i, j]

        # Compute the weighted loss
        weighted_loss = cross_entropy_loss * weights
        # Return the mean weighted loss
        return weighted_loss.mean().item()
        
    return scorer

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_prob=0.5):
        super(MyModel, self).__init__()
        
        # Create a list to hold the layers
        layers = []

        # Add the first layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size
        
        # Add the final layer (output layer)
        layers.append(nn.Linear(prev_size, 1))
        
        # Combine all layers into a sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Define the forward pass through the model

        return self.network(x)

      
class WeightedCELossTracker(Callback):

    
    
    def __init__(self, class_distances, X_train, y_train, scaler,
                 patience=5, best_loss = None, wait = 0, epoch = 0):
        
        super(WeightedCELossTracker, self).__init__()


        self.class_distances = class_distances
        self.patience = patience
        self.best_loss = best_loss
        self.wait = wait
        self.X_train = scaler.transform(X_train).float().detach().numpy()

        self.y_train = y_train
        self.epoch = epoch
    
    def print_memory_usage(self):
        # Get all objects in memory

        mem_info = psutil.virtual_memory()
        total_memory = mem_info.total / (1024 ** 3)  # Convert to GB
        available_memory = mem_info.available / (1024 ** 3)  # Convert to GB
        used_memory = mem_info.used / (1024 ** 3)  # Convert to GB

        print(f"Total memory: {total_memory:.2f} GB")
        print(f"Available memory: {available_memory:.2f} GB")
        print(f"Used memory: {used_memory:.2f} GB")

        all_objects = gc.get_objects()
        # Create a list of tuples (object, size)
        object_sizes = [(obj, sys.getsizeof(obj)) for obj in all_objects]
        # Sort objects by size in descending order
        sorted_objects = sorted(object_sizes, key=lambda x: x[1], reverse=True)
        # Print the top 10 largest objects
        print("Top 10 objects by size:")
        for obj, size in sorted_objects[:10]:
            print(f"Object type: {type(obj)}, Size: {size/1e9} gigabytes")
        
    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):

        s = time.time()
        self.loss_func = weighted_cross_entropy_with_distances_proba(self.class_distances)

        params = self.get_params()

        loss = 0
        batch_size = 2048  # Adjust batch size as needed
        y_pred = np.empty((0, num_classes))
        with torch.no_grad():
          for i in range(0, len(self.X_train), batch_size):
              batch_X = self.X_train[i:i+batch_size]
              batch_pred = net.predict_proba(batch_X)
              y_pred = np.concatenate((y_pred, batch_pred))

        # Calculate the loss on the validation fold
        loss = self.loss_func(self.y_train, y_pred)

        # Early stopping logic
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0  # Reset wait counter on improvement

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True  # Stop training if patience is reached

        print(f"Epoch: {self.epoch}, Val Loss: {loss}, Best Loss: {self.best_loss}, Epochs Since Improvement: {self.wait}, Time: {time.time() -s} seconds")
        net.history.record('val_loss', loss)
        self.epoch +=1 
        gc.collect()
        #self.print_memory_usage()

import torch.nn.functional as F


def coral_loss(logits, levels, imp):
    val =  -torch.sum((torch.log(torch.sigmoid(logits))*levels + 
             torch.log(1 - torch.sigmoid(logits))*(1-levels))*imp,
           dim=1)
    return torch.mean(val) 
    
if __name__ == '__main__':

    with open('data_for_ordinal.pkl','rb') as f:
        d = pickle.load(f)

    xtrain_enc = d['x'].astype(float)
    train_label = d['y'].astype(int)

    xtrain_enc_values = xtrain_enc.values

    # Convert to PyTorch tensors
    X = torch.as_tensor(xtrain_enc_values.astype(np.float32)).float()
    y = train_label.astype(int).values.reshape(-1, 1)

    num_features = xtrain_enc.shape[1]
    
    predictor = MyModel(num_features, hidden_sizes=[512, 256, 128], dropout_prob=.5)
    
    num_classes = len(np.unique(train_label))
    
    X_train, X_final_val, y_train, y_final_val = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)  
    cv_folds = []
    for train_index, val_index in cv.split(X_train):
        cv_folds.append((train_index, val_index))

    linear_weights = [-.3, -.2, .55, .7, 1, 1.65]
    distance_matrix = np.zeros((len(linear_weights), len(linear_weights)))

    for i in range(num_classes):
        for j in range(num_classes):
            distance_matrix[i, j] = np.abs(linear_weights[i] - linear_weights[j]) + .1

    def to_float(x):
        return x.float()

    
    ce_scorer = weighted_cross_entropy_with_distances_proba(distance_matrix)
        
    scoring = make_scorer(ce_scorer, greater_is_better=False)

    best_loss = np.inf
    best_fold_loss = np.inf
    max_fold_epochs = 100
    max_full_epochs = 300
    batch_size = 256
    patience = 10
    for lr in np.logspace(-5, -2, 4):

        
        try:
            print("running for LR", lr)
    
            # Track average validation loss across folds
            avg_val_loss = 0.0
    
            # Train and evaluate the model on each fold
            for fold, (train_index, val_index) in enumerate(cv_folds):
                print(f"Fold {fold + 1}/{len(cv_folds)}")
                
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
                custom_scaler = CustomMinMaxScaler(fill_value=-1)
                custom_scaler.fit(X_train_fold)
                
                custom_callback = WeightedCELossTracker(X_train = X_val_fold,
                                                        y_train = y_val_fold.reshape(-1,1),
                                                        scaler = custom_scaler,
                                                        class_distances = distance_matrix,
                                                        patience = patience)

                #CumulativeLinkLoss,

                class_weights = calculate_class_weights(y_train_fold.reshape(-1))
                skorch_model = NeuralNet(
                    module=OrdinalLogisticModel,
                    module__predictor=predictor,
                    module__num_classes=num_classes,
                    criterion=CumulativeLinkLoss(class_weights=class_weights),
                    max_epochs=max_fold_epochs,
                    optimizer=torch.optim.Adam,
                    lr=lr,
                    batch_size=batch_size,
                    optimizer__weight_decay=0.0,
                    device='cpu',
                    callbacks=[
                        ('ascension', AscensionCallback()),
                        custom_callback,
                        Checkpoint(monitor='val_loss',
                                   f_params=f'checkpoints/best_model_lr_{lr}.pt',
                                   f_history=f'checkpoints/best_model_history_lr_{lr}.json'),
                        EarlyStopping(patience=patience, monitor='val_loss')
                    ],
                    train_split=None,
                    verbose=0,
                )
    
                pipeline = Pipeline(steps=[
                    ('scaler', custom_scaler),
                    ('caster', FunctionTransformer(to_float)),
                    ('nn', skorch_model)
                ])
                # Train the model on the training data for this fold
                pipeline.fit(X_train_fold, y_train_fold)
    
                # Evaluate the model on the validation data for this fold in batches
                
                val_loss_fold = pipeline.named_steps['nn'].history[-1]['val_loss']
               
                print(f"Validation Loss (Fold {fold + 1}): {val_loss_fold}")
    
                avg_val_loss += val_loss_fold
    
            avg_val_loss /= len(cv_folds)
            print(f"Average Validation Loss: {avg_val_loss}")
            
            # Save the dictionary to a pickle file
            model_filename = f'best_model_lr_{lr}.pkl'
            
            model_path = os.path.join('saved_models', model_filename)
    
            # remove callbacks from pipeline bc they make pickling harder
            pipeline.named_steps['nn'].callbacks = pipeline.named_steps['nn'].callbacks[:1]
            # Save the model
            model_hist = {
                'pipeline':pipeline,
                'state_dict_path': model_path,
                'learning_rate': lr,
                'batch_size': batch_size, 
                'val_loss': avg_val_loss
            }
    
            pipeline.named_steps['nn'].save_params(f_params=model_path)
            model_hist_name = f'best_model_lr_{lr}_history.pkl'
            model_hist_path = os.path.join('saved_models', model_hist_name)
            with open(model_hist_path, 'wb') as f:
                pickle.dump(model_hist, f)
        
            if avg_val_loss < best_fold_loss:
    
                best_fold_loss = avg_val_loss
    
                custom_scaler = CustomMinMaxScaler(fill_value=-1)
                custom_scaler.fit(X_train)
                print("Fitting best model")
                custom_callback = WeightedCELossTracker(X_train = X_final_val,
                                                        y_train = y_final_val,
                                                        scaler = custom_scaler,
                                                        class_distances = distance_matrix,
                                                        patience = patience)
                
                
                
                skorch_model = NeuralNet(
                    module=OrdinalLogisticModel,
                    module__predictor=predictor,
                    module__num_classes=num_classes,
                    criterion=CumulativeLinkLoss,
                    max_epochs=max_full_epochs,
                    optimizer=torch.optim.Adam,
                    lr=lr,
                    batch_size=batch_size,
                    optimizer__weight_decay=0.0,
                    device='cpu',
                    callbacks=[
                        ('ascension', AscensionCallback()),
                        custom_callback,
                        Checkpoint(monitor='val_loss',
                                   f_params=f'checkpoints/best_model.pt',
                                   f_history=f'checkpoints/best_model_history.json'),
                        EarlyStopping(patience=patience, monitor='val_loss')
                    ],
                    train_split=None,
                    verbose=0,
                )
                
                skorch_model.initialize()
                # Reload the best model with the saved state dict
                skorch_model.load_params(f_params=model_path)
    
                pipeline = Pipeline(steps=[
                    ('scaler', custom_scaler),
                    ('caster', FunctionTransformer(to_float)),
                    ('nn', skorch_model)
                ])
    
                # Train the reloaded best model on the full training set
                pipeline.fit(X_train, y_train)
            
                loss = pipeline.named_steps['nn'].history[-1]['val_loss']
                
    
                if loss < best_loss:
    
                    
                    model_filename = f'best_model.pt'
                
                    model_path = os.path.join('saved_models', model_filename)
                    model_hist_name = f'best_model_history.pkl'
                    model_hist_path = os.path.join('saved_models', model_hist_name)
        
                    pipeline.named_steps['nn'].callbacks = pipeline.named_steps['nn'].callbacks[:1]
                    model_hist = {
                        'pipeline':pipeline,
                        'state_dict_path': model_path,
                        'learning_rate': lr,
                        'batch_size': batch_size, 
                        'val_loss': loss
                    }
                    # Save the fully trained model
                    with open(model_hist_path, 'wb') as f:
                        pickle.dump(model_hist, f)
                    pipeline.named_steps['nn'].save_params(f_params=model_path)
    
                    best_loss = loss
        except:
            import traceback
            traceback.print_exc()
            continue
    
    print("Training completed.")

