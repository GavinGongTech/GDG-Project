# Backend/model.py
from pathlib import Path
from typing import Dict

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_utils import load_raw_data, add_features, train_test_split_features

MODEL_PATH = Path(__file__).parent / "win_loss_model.pkl"


def train_and_save_model() -> float:
    """Train the model from scratch and save it. Returns test accuracy."""
    df = load_raw_data()
    df = add_features(df)
    X_train, X_test, y_train, y_test = train_test_split_features(df)

    # Simple numeric pipeline: scale features, logistic regression
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} with test accuracy={acc:.3f}")
    return acc


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train it first by calling train_and_save_model()."
        )
    return joblib.load(MODEL_PATH)


def predict_from_features(features: Dict[str, float]) -> float:
    """
    features: dict with keys:
      - point_diff
      - h2h_winrate_before
      - weather_scaled
      - home_advantage
      - injuries
      - prev_point_diff

    Returns: probability of win (float 0–1).
    """
    model = load_model()

    ordered_features = [
        features["point_diff"],
        features["h2h_winrate_before"],
        features["weather_scaled"],
        features["home_advantage"],
        features["injuries"],
        features["prev_point_diff"],
    ]

    import numpy as np

    X = np.array(ordered_features).reshape(1, -1)
    proba = model.predict_proba(X)[0, 1]  # P(win)
    return float(proba)