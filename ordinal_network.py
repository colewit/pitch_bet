from spacecutter.losses import CumulativeLinkLoss
from spacecutter.callbacks import AscensionCallback
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, make_scorer
from sklearn.pipeline import Pipeline
from skorch.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from skorch.callbacks import Callback, ProgressBar, Checkpoint
from skorch.net import NeuralNet
from sklearn.model_selection import KFold
import numpy as np
import pickle
import torch
import torch.nn as nn
from spacecutter.models import OrdinalLogisticModel
import os
import joblib

class GridSearchCVWithSaving(GridSearchCV):
    def _fit(self, X, y, groups=None, parameter_iterable=None):
        super()._fit(X, y, groups, parameter_iterable)
        # Create a directory to save models if it doesn't exist
        os.makedirs('saved_models', exist_ok=True)
        for i, (params, (train, test)) in enumerate(self.cv_results_['params']):
            # Retrieve the best estimator for each parameter combination
            estimator = self.best_estimator_
            # Get the skorch model
            skorch_net = estimator.named_steps['net']
            # Create a unique filename based on the parameters
            filename = f'saved_models/model_{i}_' + '_'.join([f'{k}={v}' for k, v in params.items()]) + '.joblib'
            # Save the model's state dictionary (weights)
            torch.save(skorch_net.module_.state_dict(), filename)
            print(f'Saved model weights with parameters: {params} to {filename}')


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
        y_true = torch.tensor(np.array(y_true).astype(int), dtype=torch.int)
     
        # Convert y_pred_proba to tensor
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

        # Calculate the cross-entropy loss for each class pair
        cross_entropy_loss = -torch.log(y_pred[range(len(y_true)), y_true] + 1e-6)
        
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

    
    
    def __init__(self, cv_folds, class_distances, X_train, y_train,
                 patience=5, best_loss = None, wait = 0, cv_losses = [], epoch = 0):
        
        super(WeightedCELossTracker, self).__init__()

        
        self.cv_folds = cv_folds
        self.class_distances = class_distances
        self.patience = patience
        self.best_loss = best_loss
        self.wait = wait
        self.cv_losses = cv_losses
        self.X_train = X_train
        self.y_train = y_train
        self.epoch = epoch

    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):

        self.loss_func = weighted_cross_entropy_with_distances_proba(self.class_distances)

        # Access validation fold for this epoch (assuming folds are sequential)
        fold_idx = self.epoch % len(self.cv_folds)
        train_mask, val_mask = self.cv_folds[fold_idx]
        y_true_fold = self.y_train[val_mask]

        params = self.get_params()
    
        # Iterate over batches of data
        batch_size = 256  # Adjust batch size as needed
        y_pred_fold = np.empty((0, num_classes))
        with torch.no_grad():
          for i in range(0, len(self.X_train), batch_size):
            batch_X = self.X_train[i:i+batch_size]
            batch_y_pred = net.predict_proba(batch_X)
            y_pred_fold = np.concatenate((y_pred_fold, batch_y_pred))

          
        # Calculate the loss on the validation fold
        loss = self.loss_func(y_true_fold, y_pred_fold)

        self.cv_losses.append(loss)

        # Early stopping logic
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0  # Reset wait counter on improvement

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True  # Stop training if patience is reached

        
        print(f"Epoch: {self.epoch}, Val Loss: {loss}, Best Loss: {self.best_loss}")
        net.history.record('val_loss', loss)
        self.epoch +=1 
        
if __name__ == '__main__':

    with open('data_for_ordinal.pkl','rb') as f:
        d = pickle.load(f)

    xtrain_enc = d['x']
    train_label = d['y'].astype(int)

    num_features = xtrain_enc.shape[1]
    
    predictor = MyModel(num_features, hidden_sizes=[128, 128, 128], dropout_prob=.5)
    
    num_classes = len(np.unique(train_label))
    model = OrdinalLogisticModel(predictor, num_classes)
    
    X_tensor = torch.as_tensor(xtrain_enc.fillna(-1).values.astype(np.float32))
    
    predictor_output = predictor(X_tensor).detach()
    model_output = model(X_tensor).detach()

    
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)  
    cv_folds = []
    for train_index, val_index in cv.split(X_tensor[:100]):
        cv_folds.append((train_index, val_index))

    linear_weights = [-.3, -.2, .55, .7, 1, 1.65]
    distance_matrix = np.zeros((len(linear_weights), len(linear_weights)))

    for i in range(num_classes):
        for j in range(num_classes):
            distance_matrix[i, j] = np.abs(linear_weights[i] - linear_weights[j]) + .1

    custom_callback = WeightedCELossTracker(cv_folds=cv_folds, X_train = X_tensor,
                                            y_train = train_label,
                                            class_distances = distance_matrix, patience = 5)
    
    skorch_model = NeuralNet(
        module=OrdinalLogisticModel,
        module__predictor=predictor,
        module__num_classes=num_classes,
        criterion=CumulativeLinkLoss,
        max_epochs=2,
        optimizer=torch.optim.Adam,
        lr = .1,
        batch_size = 256,
        optimizer__weight_decay=0.0,
        device='cpu',
        callbacks=[
            ('ascension', AscensionCallback()),
            custom_callback,
            Checkpoint(monitor='val_loss', f_params='best_model.pt')
        ],
        train_split=None,
        verbose=0,)
    
    def to_float(x):
        return x.astype(np.float32)
    
    pipeline = Pipeline([
        ('caster', FunctionTransformer(to_float)),
        ('net', skorch_model)
    ])
    
    param_grid = {
        'net__max_epochs': np.array([100]).astype(int),
        'net__lr': np.logspace(-5, -1, 4)
    }

    ce_scorer = weighted_cross_entropy_with_distances_proba(distance_matrix)
    
    scoring = make_scorer(ce_scorer, greater_is_better=False)
    
    sc_grid_search = GridSearchCVWithSaving(
        pipeline, param_grid, scoring=scoring,
        n_jobs=None, cv= cv_folds, verbose=2
    )
    
    y_train = train_label.astype(int).values.reshape(-1, 1)
    sc_grid_search.fit(xtrain_enc.fillna(-1).values[:100], y_train[:100])

    with open('ordinal_model.pkl','wb') as f:
        pickle.dump(sc_grid_search, f)

