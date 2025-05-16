import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv('spotify_data/spotify_data.csv')

# Select relevant features
numerical_features = [
    'energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence',
    'speechiness', 'track_popularity', 'instrumentalness', 'mode', 'key',
    'duration_ms', 'acousticness'
]
categorical_features = ['playlist_genre', 'playlist_subgenre']
metadata_features = ['track_id', 'track_name', 'track_artist', 'track_album_name']

# Handle missing values
df = df[numerical_features + categorical_features + metadata_features].dropna()

# Encode categorical features
genre_encoder = LabelEncoder()
subgenre_encoder = LabelEncoder()
df['playlist_genre'] = genre_encoder.fit_transform(df['playlist_genre'])
df['playlist_subgenre'] = subgenre_encoder.fit_transform(df['playlist_subgenre'])

# Normalize numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv('spotify_data/train_data.csv', index=False)
test_df.to_csv('spotify_data/test_data.csv', index=False)

# Save encoders and scaler
with open('spotify_data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('spotify_data/genre_encoder.pkl', 'wb') as f:
    pickle.dump(genre_encoder, f)
with open('spotify_data/subgenre_encoder.pkl', 'wb') as f:
    pickle.dump(subgenre_encoder, f)