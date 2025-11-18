import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ZIP_PATH = Path(__file__).parent / "archive.zip"  # adjust if needed


def load_raw_data() -> pd.DataFrame:
    """Load the main CSV inside archive.zip as a pandas DataFrame."""
    if not DATA_ZIP_PATH.exists():
        raise FileNotFoundError(f"archive.zip not found at {DATA_ZIP_PATH}")

    with zipfile.ZipFile(DATA_ZIP_PATH, "r") as z:
        # If you know the file name inside the zip, replace this with that string.
        # For now, take the first CSV in the archive.
        csv_files = [name for name in z.namelist() if name.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError("No CSV file found inside archive.zip")
        csv_name = csv_files[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the features listed in your spec.

    You MUST adjust the column names marked with TODOs to match your dataset.
    """
    df = df.copy()

    # === Example column assumptions (CHANGE THESE) ===
    # TODO: change these to your real column names
    TEAM_SCORE_COL = "team_score"
    OPP_SCORE_COL = "opponent_score"
    HOME_TEAM_COL = "is_home"          # 1 if home, 0 if away
    WEATHER_COL = "weather_rating"     # some numeric [-1,1] or similar
    INJURIES_COL = "num_injured"       # number of injured key players
    TEAM_NAME_COL = "team"
    OPP_NAME_COL = "opponent"
    TARGET_COL = "won_game"            # 1 if team won, 0 if lost

    # --- Point differential (current game) ---
    df["point_diff"] = df[TEAM_SCORE_COL] - df[OPP_SCORE_COL]

    # --- Conditional H2H (head-to-head) ---
    # Simple example: cumulative win rate vs. this opponent *before* this game.
    # This is a bit advanced, so we’ll approximate with groupby transform.
    df = df.sort_values("date")  # TODO: ensure you have a date column
    df["h2h_wins_before"] = (
        df.groupby([TEAM_NAME_COL, OPP_NAME_COL])[TARGET_COL]
        .apply(lambda s: s.shift().cumsum())
        .reset_index(level=[0, 1], drop=True)
    )
    df["h2h_games_before"] = (
        df.groupby([TEAM_NAME_COL, OPP_NAME_COL]).cumcount()
    )
    df["h2h_winrate_before"] = df["h2h_wins_before"] / df["h2h_games_before"].clip(lower=1)

    # --- Weather already assumed numeric in [-1,1] ---
    df["weather_scaled"] = df[WEATHER_COL]  # if not, you can rescale later

    # --- Home advantage ---
    df["home_advantage"] = df[HOME_TEAM_COL].astype(int)

    # --- Injuries (already numeric) ---
    df["injuries"] = df[INJURIES_COL]

    # --- Momentum / confidence (previous game’s score diff) ---
    df["prev_point_diff"] = (
        df.groupby(TEAM_NAME_COL)["point_diff"]
        .shift()
        .fillna(0)
    )

    # Drop rows where target is NaN (e.g., very first games if needed)
    df = df.dropna(subset=[TARGET_COL])

    return df


def train_test_split_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create train/test split returning X_train, X_test, y_train, y_test."""
    # TODO: ensure this matches your actual feature list & target column.
    TARGET_COL = "won_game"

    feature_cols = [
        "point_diff",
        "h2h_winrate_before",
        "weather_scaled",
        "home_advantage",
        "injuries",
        "prev_point_diff",
        # You can add more numeric features here
    ]

    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test