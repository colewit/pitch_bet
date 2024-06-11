import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# add sampling function to adaptive labels file
def fit_model_for_adaptive_labels(train_data):
    
    outcomes = ['fieldout', 'single', 'double', 'homerun']
    pred_columns = ['estimated_woba_using_speedangle', 'hit_distance_sc',
                    'launch_angle', 'launch_speed', 'estimated_ba_using_speedangle']
    
    label_sample_data = train_data[pred_columns + ['label']]
    
    # Define the features and target
    X = label_sample_data.drop(columns='label').values
    y = label_sample_data['label'].values.astype(int)
    
    # Create a pipeline with Min-Max scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('logistic_regression', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
    ])
    
    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    
    # Print the accuracy for each fold and the mean accuracy
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"Standard Deviation of Accuracy: {cv_scores.std():.4f}")

    pipeline.fit(X, y)
    return pipeline
    
if __name__ == '__main__':
    
    with open('data_for_ordinal.pkl', 'rb') as f:
        data = pickle.load(f)
    
    train_data = data['adaptive_label_train'].dropna()
    # Prepare the data
    pipeline = fit_model_for_adaptive_labels(train_data)
    
    # Save the fitted model to a file
    with open('label_models/model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    # Predict probabilities on the full dataset
    proba_df = pd.DataFrame(pipeline.predict_proba(X), columns=outcomes)
    proba_df = pd.concat([train_data.reset_index(), proba_df], axis=1)
    
    # Save the probabilities dataframe to a parquet file
    proba_df.to_parquet('adaptive_labels.parquet')
