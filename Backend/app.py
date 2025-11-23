from fastapi import FastAPI
from pydantic import BaseModel

from .model import train_and_save_model, predict_from_features

app = FastAPI(title="Win/Loss Prediction API")


class TrainResponse(BaseModel):
    test_accuracy: float


class PredictRequest(BaseModel):
    points_diff: float
    mov: float
    total_yards: float
    turnovers: float
    score_pct: float
    turnover_pct: float
    exp_pts_tot: float


class PredictResponse(BaseModel):
    win_probability: float

@app.get("/")
def root():
    return {"message": "Win/Loss Prediction API is running. See /docs for usage."}

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