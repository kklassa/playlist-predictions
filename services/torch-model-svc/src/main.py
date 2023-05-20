from typing import List, Any
from fastapi import FastAPI
import pickle
import torch
import pandas as pd
import numpy as np

from model import NNPredictor


app = FastAPI(body_limit=2**24)


@app.on_event("startup")
def load_model():
    global le_track, le_artist, scaler, model

    with open('./artifacts/preprocessing/le_track.pkl', 'rb') as f:
        le_track = pickle.load(f)
    with open('./artifacts/preprocessing/le_artist.pkl', 'rb') as f:
        le_artist = pickle.load(f)
    with open('./artifacts/preprocessing/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    model = NNPredictor(18, 128).float()
    model.load_state_dict(torch.load('./artifacts/models/nn_regressor.pth'))
    model.eval() 


@app.get("/")
async def hello():
    return {
        "message": "hello",
        "model": "nn-regressor-v1"
        }


@app.post("/prediction", response_model=List[float])
async def predict(input: List[List[Any]]) -> List[float]:
    data = await load_data(input)
    data = await preprocess_data(data)
    predictions = await predict_next_week_plays(data)
    return predictions


@app.post("/top-playlist", response_model=List[str])
async def generate_top_playlist(input: List[List[Any]]) -> List[str]:
    data = await load_data(input)
    preprocessed_data = await preprocess_data(data)
    predictions = await predict_next_week_plays(preprocessed_data)
    predicted_plays = np.ceil(predictions).astype(int)
    indices = np.argsort(predicted_plays)[-50:]
    predicted_tracks = data.iloc[indices]['id_track'].values
    return predicted_tracks.tolist()


async def predict_next_week_plays(data: pd.DataFrame) -> List[float]:
    predictions = model.forward(torch.from_numpy(data.values).float()).detach().numpy().flatten()
    return predictions.tolist()


async def load_data(data: List[List[Any]]) -> pd.DataFrame:
    labels = ['id_track', 'popularity', 'duration_ms', 'explicit', 'id_artist', 'danceability', 
              'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
              'liveness', 'valence', 'tempo', 'artist_popularity', 'track_plays', 'artist_plays']
    df = pd.DataFrame(data, columns=labels)
    return df


async def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    features_to_normalize = ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 
                            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'artist_popularity', 
                            'track_plays', 'artist_plays']
    processed_data = data.copy()
    processed_data['id_track'] = le_track.transform(data['id_track'])
    processed_data['id_artist'] = le_artist.transform(data['id_artist'])
    processed_data[features_to_normalize] = scaler.transform(data[features_to_normalize])
    return processed_data
   