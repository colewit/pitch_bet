
import pandas as pd
import numpy as np
import dill as pickle
import os
import joblib
import time
import gc
import sys
import psutil

from ordinal_losses import CornLoss
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
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

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

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_classes,
                 cheng_labels=True, coral_loss=False, corn_loss=True, dropout_prob=0.5):
        super(MyModel, self).__init__()

        self.cheng_labels = cheng_labels
        self.coral_loss = coral_loss
        self.corn_loss = corn_loss
        # Create a list to hold the layers
        layers = []
        
        # Add the first layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers = self.add_block(layers, prev_size, hidden_size, dropout_prob)
            prev_size = hidden_size
        
        # Add the final layer (output layer)

        if self.coral_loss:
            self.fc = nn.Linear(prev_size, 1, bias=False)
            self.linear_1_bias = torch.nn.Parameter(torch.zeros(n_classes-1).float())

        elif self.cheng_labels or self.corn_loss:
            self.fc = nn.Linear(prev_size, n_classes-1)
        else:
            
            self.fc = nn.Linear(prev_size, n_classes)
        
        # Assign layers to the class attribute
        self.layers = nn.ModuleList(layers)

    def add_block(self, layers, inp, out, dropout):
        linear = torch.nn.Linear(inp, out, bias=False)
        relu = torch.nn.LeakyReLU()
        drop = torch.nn.Dropout(dropout)
        batch_norm = torch.nn.BatchNorm1d(out)
        layers += [linear, relu, drop, batch_norm]
        return layers
        
        
    def get_logits(self,x):
        out = x
        for layer in self.layers:
            out = layer(out)
            
        logits = self.fc(out)
        return logits
        
    def forward(self, x):
        # Define the forward pass through the model
        logits = self.get_logits(x)
        
        if self.coral_loss:
            return logits + self.linear_1_bias
        elif self.cheng_labels or self.corn_loss:
            #return F.sigmoid(logits)
            return logits
        else:
            return F.softmax(logits, dim=1)


    def predict(self, x):
        # Prediction function
        preds = self.forward(x)
        if self.coral_loss or self.cheng_labels:
            preds = F.sigmoid(preds)
            
        elif self.corn_loss:
            preds = F.sigmoid(preds)
            preds = torch.cumprod(preds, dim=1)
        return preds
        
    def predict_proba(self, x):
        return self.predict(x)

class ValidationTracker(Callback):
    """
    Tracks the Binary Cross-Entropy (BCE) loss on the validation set during training
    and records it in the network's history for early stopping.
    
    Args:
      X_val (torch.Tensor): Validation set features.
      y_val (torch.Tensor): Validation set labels.
    """

    def __init__(self, X_val, y_val, scaler, loss_func):
        super(ValidationTracker, self).__init__()

        self.X_val= scaler.transform(X_val).float()
        self.y_val = y_val
        self.loss_func = loss_func
        
    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        """
        Calculates, prints, and records the BCE loss on the validation set at the end of each epoch.
        
        Args:
            net (skorch.Net): The neural network model.
            dataset_train (skorch.dataset.Dataset): Training dataset.
            dataset_valid (skorch.dataset.Dataset): Validation dataset.
            **kwargs: Additional arguments passed by skorch.
        """
        with torch.no_grad():
          # Get predictions on the validation set

            y_pred = net.module_.predict(self.X_val)
            loss = self.loss_func(y_pred, self.y_val)


        # Record the validation loss in the network's history
        net.history.record('valid_loss', loss)
        
class CoralLoss(nn.Module):
    """
    Module form of cumulative_link_loss() loss function

    Parameters
    ----------
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.

    """

    def __init__(self) -> None:
        super().__init__()


    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        
        val = (-torch.sum((F.logsigmoid(y_pred)*y_true
                          + (F.logsigmoid(y_pred) - y_pred)*(1-y_true)),
               dim=1))
        return torch.mean(val)

    
if __name__ == '__main__':

    with open('data_for_ordinal_imputed.pkl','rb') as f:
        d = pickle.load(f)

    x_meta = d['x_meta_train']
    xtrain_enc = d['x_train'].astype(float)
    train_label = d['y_train'].astype(int)

    num_features = xtrain_enc.shape[1]
    num_classes = len(np.unique(train_label))

    xtrain_enc_values = xtrain_enc.values

    # Convert to PyTorch tensors
    X = torch.as_tensor(xtrain_enc_values.astype(np.float32)).float()

    cheng_labels = False
    coral_loss = False
    corn_loss = True
    adaptive_labels = True
    
    if cheng_labels:    
        y = get_cheng_labels(train_label.astype(int), num_classes)
        y = torch.as_tensor(y.astype(np.float32)).float()
    else:
        y = train_label.astype(int)
        y = torch.as_tensor(y).long()
    
    hidden_sizes=[512, 512, 256, 64]
    
    
    X_train, X_final_val, y_train, y_final_val, x_meta_train, x_meta_test = \
        train_test_split(X, y, x_meta, test_size=0.2, random_state=42)

    def to_float(x):
        return x.float()

    best_val_loss = np.inf
    max_train_epochs = 300
    max_test_epochs = 300
    batch_size = 512
    patience = 10
    loss_func = torch.nn.BCELoss(reduction='mean') if cheng_labels else \
        torch.nn.CrossEntropyLoss(reduction='mean') 

    if coral_loss:
        loss_func = CoralLoss()
    elif corn_loss:
        loss_func = CornLoss(num_classes)

    if adaptive_labels:
        proba_df = pd.read_parquet('adaptive_labels.parquet')

        proba_df = x_meta_train.merge(proba_df, 
                     how = 'left', 
                     on = ['game_date','team_at_bat_number','pitcher','batter'])

    print(x_meta.shape, xtrain_enc.shape, train_label.shape, proba_df.shape)
    
    for lr in np.logspace(-7, -2, 6):

        
        try:
            print("running for LR", lr)
            
            # Track average validation loss across folds
            avg_val_loss = 0.0

            custom_scaler = CustomMinMaxScaler(fill_value=-1)
            custom_scaler.fit(X_train)

            val_tracker = ValidationTracker(X_final_val, y_final_val,
                                               scaler = custom_scaler, loss_func=loss_func)
            callbacks = [
                    val_tracker,
                    Checkpoint(monitor='valid_loss',
                                f_params=f'checkpoints/best_model_lr_{lr}.pt',
                                f_history=f'checkpoints/best_model_history_lr_{lr}.json'),
                    EarlyStopping(patience=patience, monitor='valid_loss')
                ]
            
            skorch_model = NeuralNet(
                module = MyModel,
                module__hidden_sizes = hidden_sizes,
                module__cheng_labels = cheng_labels,
                module__coral_loss = coral_loss,
                module__corn_loss = corn_loss,
                module__input_size = num_features,
                module__n_classes=num_classes,  # Number of output classes
                criterion=loss_func,  
                max_epochs=max_train_epochs,
                optimizer=torch.optim.Adam,
                lr=lr,
                batch_size=batch_size,
                optimizer__weight_decay=0.0,
                device='cpu',  # Adjust device if needed
                callbacks=callbacks,
                train_split=None,
                verbose=2,
            )
            pipeline = Pipeline(steps=[
                ('scaler', custom_scaler),
                ('caster', FunctionTransformer(to_float)),
                ('nn', skorch_model)
            ])

            bootstrapped_labels = sample_events_multinomial(proba_df)
            labels_arr = np.where(np.isnan(bootstrapped_labels),
                                  y_train, bootstrapped_labels)
            
            # Train the model on the training data for this fold
            pipeline.fit(X_train.float(), labels_arr)

            
            # Evaluate the model on the validation data for this fold in batches
                
            val_loss = pipeline.named_steps['nn'].history[-1]['valid_loss']
          
            # Save the model
            model_hist = {
                'pipeline':pipeline,
                'learning_rate': lr,
                'batch_size': batch_size, 
                'valid_loss': avg_val_loss
            }

            model_hist_name = f'best_model_lr_{lr}_history.pkl'
            model_hist_path = os.path.join('saved_models', model_hist_name)
            with open(model_hist_path, 'wb') as f:
                pickle.dump(model_hist, f)
        
            if val_loss < best_val_loss:
    
                best_val_loss = val_loss

                model_hist_name = f'best_model_history.pkl'
                model_hist_path = os.path.join('saved_models', model_hist_name)
    
                model_hist = {
                    'pipeline':pipeline,
                    'learning_rate': lr,
                    'batch_size': batch_size, 
                    'val_loss': val_loss
                }
                # Save the fully trained model
                with open(model_hist_path, 'wb') as f:
                    pickle.dump(model_hist, f)
  

        except:
            import traceback
            traceback.print_exc()
            continue
    
    print("Training completed.")

