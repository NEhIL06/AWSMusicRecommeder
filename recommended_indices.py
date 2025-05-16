import pandas as pd
df = pd.read_csv('spotify_data/spotify_data.csv')
recommended_indices = [901, 1230, 21, 1047, 939, 923, 16, 677, 1098, 76]  # Replace with API output
recommended_songs = df.iloc[recommended_indices][['track_name', 'track_artist', 'track_album_name']]
print(recommended_songs)