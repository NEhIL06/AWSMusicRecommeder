import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
import os
import argparse

def train_model(input_path, output_path):
    df = pd.read_csv(input_path)
    features = [
        'energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence',
        'speechiness', 'track_popularity', 'instrumentalness', 'mode', 'key',
        'duration_ms', 'acousticness', 'playlist_genre', 'playlist_subgenre'
    ]
    X = df[features]
    model = NearestNeighbors(n_neighbors=10, algorithm='auto')
    model.fit(X)
    joblib.dump(model, os.path.join(output_path, 'model.joblib'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()
    train_model(f'{args.input_dir}/train_data.csv', args.output_dir)