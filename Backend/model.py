# Backend/model.py
import logging
import sys
from pathlib import Path
from typing import Dict

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_utils import load_raw_data, add_features, train_test_split_features

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "win_loss_model.pkl"


def train_and_save_model() -> float:
    """Train the model from scratch and save it. Returns test accuracy."""
    logger.info("[train_and_save_model] Starting model training...")
    
    try:
        logger.info("Loading raw data...")
        df = load_raw_data()
        logger.info(f"Raw data loaded. Shape: {df.shape}")
        
        logger.info("Adding features...")
        df = add_features(df)
        logger.info(f"Features added. Shape: {df.shape}")
        
        logger.info("Splitting into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split_features(df)
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Train labels distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test labels distribution: {y_test.value_counts().to_dict()}")

        logger.info("Creating model pipeline...")
        # Simple numeric pipeline: scale features, logistic regression
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        )

        logger.info("Training model...")
        clf.fit(X_train, y_train)
        logger.info("Model training complete")
        
        logger.info("Making predictions on test set...")
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"Test accuracy: {acc:.4f}")

        logger.info(f"Saving model to {MODEL_PATH}...")
        joblib.dump(clf, MODEL_PATH)
        logger.info(f"Model saved successfully to {MODEL_PATH}")
        
        return acc
    except Exception as e:
        logger.error(f"Error in train_and_save_model: {str(e)}", exc_info=True)
        raise


def load_model():
    logger.info(f"[load_model] Attempting to load model from {MODEL_PATH}")
    
    if not MODEL_PATH.exists():
        error_msg = f"Model file not found at {MODEL_PATH}. Train it first by calling train_and_save_model()."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info("Loading model from disk...")
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
        logger.info(f"Model type: {type(model)}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise


def predict_from_features(features: Dict[str, float]) -> float:
    """
    features: dict with keys:
      - points_diff
      - mov
      - total_yards
      - turnovers
      - score_pct
      - turnover_pct
      - exp_pts_tot

    Returns: probability of win (float 0–1).
    """
    logger.info("[predict_from_features] Starting prediction...")
    logger.info(f"Input features: {features}")
    
    try:
        model = load_model()
        logger.debug("Model loaded for prediction")

        # Validate all required features are present
        required_features = [
            "points_diff", "mov", "total_yards", "turnovers",
            "score_pct", "turnover_pct", "exp_pts_tot"
        ]
        missing_features = [f for f in required_features if f not in features]
        if missing_features:
            error_msg = f"Missing required features: {missing_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        ordered_features = [
            features["points_diff"],
            features["mov"],
            features["total_yards"],
            features["turnovers"],
            features["score_pct"],
            features["turnover_pct"],
            features["exp_pts_tot"],
        ]
        logger.debug(f"Ordered features array: {ordered_features}")

        import numpy as np

        X = np.array(ordered_features).reshape(1, -1)
        logger.debug(f"Feature matrix shape: {X.shape}")
        logger.debug(f"Feature matrix: {X}")
        
        logger.info("Running model prediction...")
        proba = model.predict_proba(X)
        logger.debug(f"Raw prediction probabilities: {proba}")
        
        win_probability = float(proba[0, 1])  # P(winning season)
        logger.info(f"Win probability: {win_probability:.4f}")
        
        return win_probability
        
    except Exception as e:
        logger.error(f"Error in predict_from_features: {str(e)}", exc_info=True)
        raise