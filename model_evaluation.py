import pandas as pd
from sklearn.neighbors import NearestNeighbors

test_df = pd.read_csv('spotify_data/test_data.csv')
train_df = pd.read_csv('spotify_data/train_data.csv')
features = [
    'energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence',
    'speechiness', 'track_popularity', 'instrumentalness', 'mode', 'key',
    'duration_ms', 'acousticness', 'playlist_genre', 'playlist_subgenre'
]
model = NearestNeighbors(n_neighbors=10)
model.fit(train_df[features])
precisions = []
for i in range(len(test_df)):
    distances, indices = model.kneighbors([test_df[features].iloc[i]])
    predicted = train_df.iloc[indices[0]]['track_id']
    true = test_df.iloc[i]['track_id']
    precision = 1 if true in predicted.values else 0
    precisions.append(precision)
print(f'Precision@10: {sum(precisions) / len(precisions)}')