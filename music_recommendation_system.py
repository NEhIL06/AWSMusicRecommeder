import pandas as pd
import numpy as np
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Initialize AWS session and SageMaker session
session = boto3.Session(region_name='ap-south-1')
sagemaker_session = sagemaker.Session(boto_session=session)

# Specify IAM role ARN
role = 'arn:aws:iam::463224263437:role/SageMakerExecutionRole'

# S3 bucket details
bucket = 'spotify-recommendation-system-463224263437'
prefix = 'spotify-data'

# Step 1: Data Preprocessing
def preprocess_data(data_path):
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Select relevant features
    features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 
                'valence', 'speechiness', 'track_popularity', 'instrumentalness', 
                'acousticness', 'playlist_genre']
    df = df[features]
    
    # Encode categorical variable (playlist_genre)
    label_encoder = LabelEncoder()
    df['playlist_genre'] = label_encoder.fit_transform(df['playlist_genre'])
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 
                         'valence', 'speechiness', 'track_popularity', 'instrumentalness', 
                         'acousticness']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Save preprocessor objects
    joblib.dump(label_encoder, 'label_encoder.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return df

# Step 2: Prepare data for training
def prepare_training_data(df):
    # Split features and target
    X = df.drop('playlist_genre', axis=1)
    y = df['playlist_genre']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save train and test data to CSV
    train_data = pd.concat([y_train, X_train], axis=1)
    test_data = pd.concat([y_test, X_test], axis=1)
    
    train_data.to_csv('train.csv', index=False)
    test_data.to_csv('test.csv', index=False)
    
    # Upload to S3
    sagemaker_session.upload_data('train.csv', bucket=bucket, key_prefix=f'{prefix}/train')
    sagemaker_session.upload_data('test.csv', bucket=bucket, key_prefix=f'{prefix}/test')
    
    return f's3://{bucket}/{prefix}/train/train.csv', f's3://{bucket}/{prefix}/test/test.csv'

# Step 3: Train model using SageMaker
def train_model(train_s3_uri, test_s3_uri):
    # Define SKLearn estimator
    sklearn_estimator = SKLearn(
        entry_point='train_script.py',
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        framework_version='0.23-1',
        py_version='py3',
        sagemaker_session=sagemaker_session,
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 5
        }
    )
    
    # Define input channels
    inputs = {
        'train': TrainingInput(train_s3_uri, content_type='csv'),
        'test': TrainingInput(test_s3_uri, content_type='csv')
    }
    
    # Start training
    sklearn_estimator.fit(inputs)
    
    return sklearn_estimator

# Step 4: Deploy model to SageMaker endpoint
def deploy_model(estimator):
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',
        endpoint_name='spotify-recommendation-endpoint'
    )
    return predictor

if __name__ == '__main__':
    # Assuming dataset is downloaded locally as 'spotify_data.csv'
    df = preprocess_data('spotify_dataset.csv')
    train_s3_uri, test_s3_uri = prepare_training_data(df)
    estimator = train_model(train_s3_uri, test_s3_uri)
    predictor = deploy_model(estimator)