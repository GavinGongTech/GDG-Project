from fastapi import FastAPI
from pydantic import BaseModel

from .model import train_and_save_model, predict_from_features

app = FastAPI(title="Win/Loss Prediction API")


class TrainResponse(BaseModel):
    test_accuracy: float


class PredictRequest(BaseModel):
    point_diff: float
    h2h_winrate_before: float
    weather_scaled: float  # between -1 and 1
    home_advantage: int    # 1 = home, 0 = away
    injuries: float
    prev_point_diff: float


class PredictResponse(BaseModel):
    win_probability: float


@app.post("/train", response_model=TrainResponse)
def train():
    """Train model from data in archive.zip."""
    acc = train_and_save_model()
    return TrainResponse(test_accuracy=acc)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Return win probability for a single matchup."""
    proba = predict_from_features(req.dict())
    return PredictResponse(win_probability=proba)