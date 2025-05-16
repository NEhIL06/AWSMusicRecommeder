import requests

url = "https://92jrvxbxr8.execute-api.ap-south-1.amazonaws.com/prod/recommend"
payload = {
    "energy": 0.5, "tempo": 120, "danceability": 0.7, "loudness": -5,
    "liveness": 0.2, "valence": 0.6, "speechiness": 0.1,
    "track_popularity": 50, "instrumentalness": 0.0, "mode": 1,
    "key": 5, "duration_ms": 200000, "acousticness": 0.3,
    "playlist_genre": "pop", "playlist_subgenre": "pop"
}
headers = {"Content-Type": "application/json"}
response = requests.post(url, json=payload, headers=headers)
print(response.json())