import json
import joblib
import pandas as pd
import pickle
import os

# Global variable to store model_dir
MODEL_DIR = None

def model_fn(model_dir):
    global MODEL_DIR
    MODEL_DIR = model_dir  # Store model_dir globally
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb'))
        genre_encoder = pickle.load(open(os.path.join(MODEL_DIR, 'genre_encoder.pkl'), 'rb'))
        subgenre_encoder = pickle.load(open(os.path.join(MODEL_DIR, 'subgenre_encoder.pkl'), 'rb'))

        df = pd.DataFrame([data])
        df['playlist_genre'] = genre_encoder.transform(df['playlist_genre'])
        df['playlist_subgenre'] = subgenre_encoder.transform(df['playlist_subgenre'])
        numerical_features = [
            'energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence',
            'speechiness', 'track_popularity', 'instrumentalness', 'mode', 'key',
            'duration_ms', 'acousticness'
        ]
        df[numerical_features] = scaler.transform(df[numerical_features])
        return df
    raise ValueError('Unsupported content type')

def predict_fn(input_data, model):
    features = [
        'energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence',
        'speechiness', 'track_popularity', 'instrumentalness', 'mode', 'key',
        'duration_ms', 'acousticness', 'playlist_genre', 'playlist_subgenre'
    ]
    _, indices = model.kneighbors(input_data[features])  # Use _ for unused distances
    return indices.tolist()

def output_fn(prediction, content_type):
    return json.dumps({'recommended_indices': prediction})