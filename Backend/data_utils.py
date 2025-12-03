import logging
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DATA_ZIP_PATH = Path(__file__).parent / "archive.zip"


def load_raw_data() -> pd.DataFrame:
    """Load the main CSV inside archive.zip as a pandas DataFrame."""
    logger.info(f"[load_raw_data] Loading data from {DATA_ZIP_PATH}")
    
    if not DATA_ZIP_PATH.exists():
        error_msg = f"archive.zip not found at {DATA_ZIP_PATH}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Archive file exists: {DATA_ZIP_PATH}")

    try:
        with zipfile.ZipFile(DATA_ZIP_PATH, "r") as z:
            csv_files = [name for name in z.namelist() if name.lower().endswith(".csv")]
            logger.info(f"Found {len(csv_files)} CSV file(s) in archive: {csv_files}")
            
            if not csv_files:
                error_msg = "No CSV file found inside archive.zip"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            csv_name = csv_files[0]
            logger.info(f"Loading CSV: {csv_name}")
            
            with z.open(csv_name) as f:
                df = pd.read_csv(f)
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.debug(f"Columns: {list(df.columns)}")
            logger.debug(f"First few rows:\n{df.head()}")
            
            return df
    except Exception as e:
        logger.error(f"Error loading raw data: {str(e)}", exc_info=True)
        raise


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use existing season-level stats as features and create a binary target:
    1 = winning season (win_loss_perc > 0.5), 0 = otherwise.
    """
    logger.info("[add_features] Adding features to dataframe...")
    df = df.copy()

    # Target: winning season or not
    if "win_loss_perc" not in df.columns:
        logger.warning("'win_loss_perc' column not found. Checking alternatives...")
        logger.debug(f"Available columns: {list(df.columns)}")
        # Try to find similar column
        win_cols = [col for col in df.columns if 'win' in col.lower() or 'loss' in col.lower() or 'perc' in col.lower()]
        logger.debug(f"Found potential win/loss columns: {win_cols}")
        if not win_cols:
            logger.error("Could not find win_loss_perc or similar column")
            raise ValueError("Required column 'win_loss_perc' not found in data")
    
    df["winning_season"] = (df["win_loss_perc"] > 0.5).astype(int)
    winning_count = df["winning_season"].sum()
    logger.info(f"Created target column 'winning_season'. Winning seasons: {winning_count}/{len(df)}")

    return df


def train_test_split_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train/test split returning X_train, X_test, y_train, y_test.
    We use a small set of intuitive numeric features.
    """
    logger.info("[train_test_split_features] Preparing train/test split...")
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
    
    logger.info(f"Required feature columns: {feature_cols}")
    logger.info(f"Available columns in dataframe: {list(df.columns)}")
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required feature columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Original dataframe shape: {df.shape}")
    df_clean = df.dropna(subset=feature_cols + [TARGET_COL])
    dropped = len(df) - len(df_clean)
    logger.info(f"Dropped {dropped} rows with missing values. Clean dataframe shape: {df_clean.shape}")

    X = df_clean[feature_cols]
    y = df_clean[TARGET_COL]
    
    logger.info(f"Feature matrix X shape: {X.shape}")
    logger.info(f"Target vector y shape: {y.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"Feature statistics:\n{X.describe()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test