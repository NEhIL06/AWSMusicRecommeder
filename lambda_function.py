import json
import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def lambda_handler(event, context):
    # Initialize SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime', region_name='ap-south-1')
    
    # Load preprocessor objects
    label_encoder = joblib.load('label_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # Parse input data
    body = json.loads(event['body'])
    input_data = pd.DataFrame([{
        'energy': body['energy'],
        'tempo': body['tempo'],
        'danceability': body['danceability'],
        'loudness': body['loudness'],
        'liveness': body['liveness'],
        'valence': body['valence'],
        'speechiness': body['speechiness'],
        'track_popularity': body['track_popularity'],
        'instrumentalness': body['instrumentalness'],
        'acousticness': body['acousticness']
    }])
    
    # Preprocess input
    numerical_features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 
                         'valence', 'speechiness', 'track_popularity', 'instrumentalness', 
                         'acousticness']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])
    
    # Convert to CSV for SageMaker endpoint
    csv_data = input_data.to_csv(index=False, header=False)
    
    # Invoke SageMaker endpoint
    response = runtime.invoke_endpoint(
        EndpointName='spotify-recommendation-endpoint',
        ContentType='text/csv',
        Body=csv_data
    )
    
    # Parse response
    prediction = json.loads(response['Body'].read().decode())
    genre = label_encoder.inverse_transform([int(prediction[0])])[0]
    
    return {
        'statusCode': 200,
        'body': json.dumps({'recommended_genre': genre})
    }