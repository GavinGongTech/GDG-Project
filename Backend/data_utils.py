import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ZIP_PATH = Path(__file__).parent / "archive.zip"


def load_raw_data() -> pd.DataFrame:
    """Load the main CSV inside archive.zip as a pandas DataFrame."""
    if not DATA_ZIP_PATH.exists():
        raise FileNotFoundError(f"archive.zip not found at {DATA_ZIP_PATH}")

    with zipfile.ZipFile(DATA_ZIP_PATH, "r") as z:
        csv_files = [name for name in z.namelist() if name.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError("No CSV file found inside archive.zip")
        csv_name = csv_files[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use existing season-level stats as features and create a binary target:
    1 = winning season (win_loss_perc > 0.5), 0 = otherwise.
    """
    df = df.copy()

    # Target: winning season or not
    df["winning_season"] = (df["win_loss_perc"] > 0.5).astype(int)

    # You can engineer extra features if you want; for now we'll just keep
    # the raw numeric columns we care about.
    return df


def train_test_split_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train/test split returning X_train, X_test, y_train, y_test.
    We use a small set of intuitive numeric features.
    """
    TARGET_COL = "winning_season"

    feature_cols = [
        "points_diff",            # point differential
        "mov",                    # margin of victory
        "total_yards",
        "turnovers",
        "score_pct",
        "turnover_pct",
        "exp_pts_tot",
    ]

    df_clean = df.dropna(subset=feature_cols + [TARGET_COL])

    X = df_clean[feature_cols]
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test