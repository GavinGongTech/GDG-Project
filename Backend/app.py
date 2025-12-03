import logging
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .model import train_and_save_model, predict_from_features
from .data_utils import load_raw_data

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backend.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Win/Loss Prediction API")

logger.info("="*50)
logger.info("STARTING BACKEND APPLICATION")
logger.info("="*50)

# Add CORS middleware to allow frontend requests from any localhost port
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",  # Allow any localhost port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS middleware configured - allowing localhost:3000, localhost:3001, and any localhost port via regex")


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


class MatchupRequest(BaseModel):
    team1: str
    team2: str


class MatchupResponse(BaseModel):
    winner: str
    team1_win_probability: float
    team2_win_probability: float
    team1_stats: dict
    team2_stats: dict
    matchup_features: dict
    feature_comparisons: dict


@app.get("/")
def root():
    logger.info("GET / - Root endpoint called")
    return {"message": "Win/Loss Prediction API is running. See /docs for usage."}

@app.post("/train", response_model=TrainResponse)
def train():
    """Train model from data in archive.zip."""
    logger.info("="*50)
    logger.info("POST /train - Training model started")
    logger.info("="*50)
    try:
        acc = train_and_save_model()
        logger.info(f"Model training completed with accuracy: {acc:.4f}")
        return TrainResponse(test_accuracy=acc)
    except Exception as e:
        logger.error(f"Error training model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Return win probability for a single matchup."""
    logger.info("="*50)
    logger.info("POST /predict - Prediction request received")
    logger.info(f"Request features: {req.dict()}")
    logger.info("="*50)
    try:
        proba = predict_from_features(req.dict())
        logger.info(f"Prediction completed: win_probability={proba:.4f}")
        return PredictResponse(win_probability=proba)
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def get_team_stats_from_data(team_abbr: str):
    """Get average/most recent stats for a team from the archive data."""
    logger.info(f"[get_team_stats_from_data] Looking up stats for team: {team_abbr}")
    
    try:
        logger.debug("Loading raw data...")
        df = load_raw_data()
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.debug(f"Available columns: {list(df.columns)[:10]}...")
        
        # Try to find team in data - check common column names
        team_cols = [col for col in df.columns if any(
            word in col.lower() for word in ['team', 'abbr', 'name', 'franchise']
        )]
        logger.debug(f"Found potential team columns: {team_cols}")
        
        if not team_cols:
            logger.warning("No team column found, using average stats for all teams")
            feature_cols = [
                "points_diff", "mov", "total_yards", "turnovers",
                "score_pct", "turnover_pct", "exp_pts_tot"
            ]
            available_cols = [col for col in feature_cols if col in df.columns]
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing feature columns: {missing_cols}")
            if available_cols:
                avg_stats = df[available_cols].mean().to_dict()
                logger.info(f"Using average stats: {avg_stats}")
                return avg_stats
        
        # Team abbreviation to full name mapping
        team_abbr_map = {
            'ARI': ['Arizona Cardinals', 'Cardinals'],
            'ATL': ['Atlanta Falcons', 'Falcons'],
            'BAL': ['Baltimore Ravens', 'Ravens'],
            'BUF': ['Buffalo Bills', 'Bills'],
            'CAR': ['Carolina Panthers', 'Panthers'],
            'CHI': ['Chicago Bears', 'Bears'],
            'CIN': ['Cincinnati Bengals', 'Bengals'],
            'CLE': ['Cleveland Browns', 'Browns'],
            'DAL': ['Dallas Cowboys', 'Cowboys'],
            'DEN': ['Denver Broncos', 'Broncos'],
            'DET': ['Detroit Lions', 'Lions'],
            'GB': ['Green Bay Packers', 'Packers'],
            'HOU': ['Houston Texans', 'Texans'],
            'IND': ['Indianapolis Colts', 'Colts'],
            'JAX': ['Jacksonville Jaguars', 'Jaguars'],
            'KC': ['Kansas City Chiefs', 'Chiefs'],
            'LV': ['Las Vegas Raiders', 'Oakland Raiders', 'Raiders'],
            'LAC': ['Los Angeles Chargers', 'San Diego Chargers', 'Chargers'],
            'LAR': ['Los Angeles Rams', 'St. Louis Rams', 'Rams'],
            'MIA': ['Miami Dolphins', 'Dolphins'],
            'MIN': ['Minnesota Vikings', 'Vikings'],
            'NE': ['New England Patriots', 'Patriots'],
            'NO': ['New Orleans Saints', 'Saints'],
            'NYG': ['New York Giants', 'Giants'],
            'NYJ': ['New York Jets', 'Jets'],
            'PHI': ['Philadelphia Eagles', 'Eagles'],
            'PIT': ['Pittsburgh Steelers', 'Steelers'],
            'SF': ['San Francisco 49ers', '49ers'],
            'SEA': ['Seattle Seahawks', 'Seahawks'],
            'TB': ['Tampa Bay Buccaneers', 'Buccaneers'],
            'TEN': ['Tennessee Titans', 'Titans'],
            'WAS': ['Washington Commanders', 'Washington Football Team', 'Washington Redskins', 'Commanders']
        }
        
        # Try to match team abbreviation
        logger.debug(f"Attempting to match team abbreviation '{team_abbr}' in columns: {team_cols}")
        
        # Get possible team name matches
        possible_names = team_abbr_map.get(team_abbr.upper(), [])
        possible_names.append(team_abbr.upper())  # Also try the abbreviation itself
        
        logger.debug(f"Possible team name matches for {team_abbr}: {possible_names}")
        
        for col in team_cols:
            try:
                # Try exact match first, then contains
                for name_match in possible_names:
                    # Try exact match (case insensitive)
                    exact_mask = df[col].astype(str).str.upper().str.strip() == name_match.upper()
                    if exact_mask.any():
                        team_data = df[exact_mask]
                        logger.info(f"Found exact match for '{team_abbr}' as '{name_match}' in column {col}: {len(team_data)} rows")
                        break
                    else:
                        # Try contains match
                        contains_mask = df[col].astype(str).str.upper().str.contains(name_match.upper(), na=False, regex=False)
                        if contains_mask.any():
                            team_data = df[contains_mask]
                            logger.info(f"Found contains match for '{team_abbr}' as '{name_match}' in column {col}: {len(team_data)} rows")
                            break
                else:
                    continue  # No match found for this column, try next
                
                # If we found matches, extract stats
                feature_cols = [
                    "points_diff", "mov", "total_yards", "turnovers",
                    "score_pct", "turnover_pct", "exp_pts_tot"
                ]
                available_cols = [col for col in feature_cols if col in team_data.columns]
                missing_cols = [col for col in feature_cols if col not in team_data.columns]
                
                if missing_cols:
                    logger.warning(f"Missing feature columns for team {team_abbr}: {missing_cols}")
                
                if available_cols:
                    # Use average of all seasons for more stable stats
                    stats = team_data[available_cols].mean().to_dict()
                    logger.info(f"Team {team_abbr} stats (averaged over {len(team_data)} seasons): {stats}")
                    return stats
                    
            except Exception as e:
                logger.warning(f"Error checking column {col}: {e}")
                continue
        
        logger.warning(f"No match found for team {team_abbr}, using average stats")
        # Fallback: return average stats
        feature_cols = [
            "points_diff", "mov", "total_yards", "turnovers",
            "score_pct", "turnover_pct", "exp_pts_tot"
        ]
        available_cols = [col for col in feature_cols if col in df.columns]
        if available_cols:
            avg_stats = df[available_cols].mean().to_dict()
            logger.info(f"Using dataset average stats: {avg_stats}")
            return avg_stats
            
    except Exception as e:
        logger.error(f"Error loading team stats for {team_abbr}: {str(e)}", exc_info=True)
    
    # Ultimate fallback: return default/neutral stats
    default_stats = {
        "points_diff": 0.0,
        "mov": 0.0,
        "total_yards": 350.0,
        "turnovers": 1.5,
        "score_pct": 0.5,
        "turnover_pct": 0.5,
        "exp_pts_tot": 0.0,
    }
    logger.warning(f"Using default/neutral stats for {team_abbr}: {default_stats}")
    return default_stats


@app.post("/predict-matchup", response_model=MatchupResponse)
def predict_matchup(req: MatchupRequest):
    """
    Predict winner between two teams using their names/abbreviations.
    Fetches team statistics from archive data and compares predictions.
    """
    logger.info("="*50)
    logger.info("POST /predict-matchup - Matchup prediction requested")
    logger.info(f"Team 1: {req.team1}")
    logger.info(f"Team 2: {req.team2}")
    logger.info("="*50)
    
    try:
        logger.info("Fetching stats for Team 1...")
        team1_stats = get_team_stats_from_data(req.team1)
        logger.info(f"Team 1 stats retrieved: {team1_stats}")
        
        logger.info("Fetching stats for Team 2...")
        team2_stats = get_team_stats_from_data(req.team2)
        logger.info(f"Team 2 stats retrieved: {team2_stats}")
        
        # Calculate relative features for team1 vs team2
        logger.info("Calculating matchup features...")
        matchup_features = {
            "points_diff": team1_stats.get("points_diff", 0) - team2_stats.get("points_diff", 0),
            "mov": team1_stats.get("mov", 0) - team2_stats.get("mov", 0),
            "total_yards": team1_stats.get("total_yards", 350) - team2_stats.get("total_yards", 350),
            "turnovers": team2_stats.get("turnovers", 1.5) - team1_stats.get("turnovers", 1.5),  # Negative is better
            "score_pct": team1_stats.get("score_pct", 0.5) - team2_stats.get("score_pct", 0.5),
            "turnover_pct": team2_stats.get("turnover_pct", 0.5) - team1_stats.get("turnover_pct", 0.5),  # Negative is better
            "exp_pts_tot": team1_stats.get("exp_pts_tot", 0) - team2_stats.get("exp_pts_tot", 0),
        }
        logger.info(f"Matchup features calculated: {matchup_features}")
        
        # Create feature comparisons for display
        feature_comparisons = {
            "points_diff": {
                "team1": team1_stats.get("points_diff", 0),
                "team2": team2_stats.get("points_diff", 0),
                "advantage": req.team1 if team1_stats.get("points_diff", 0) > team2_stats.get("points_diff", 0) else req.team2
            },
            "mov": {
                "team1": team1_stats.get("mov", 0),
                "team2": team2_stats.get("mov", 0),
                "advantage": req.team1 if team1_stats.get("mov", 0) > team2_stats.get("mov", 0) else req.team2
            },
            "total_yards": {
                "team1": team1_stats.get("total_yards", 350),
                "team2": team2_stats.get("total_yards", 350),
                "advantage": req.team1 if team1_stats.get("total_yards", 350) > team2_stats.get("total_yards", 350) else req.team2
            },
            "turnovers": {
                "team1": team1_stats.get("turnovers", 1.5),
                "team2": team2_stats.get("turnovers", 1.5),
                "advantage": req.team2 if team1_stats.get("turnovers", 1.5) > team2_stats.get("turnovers", 1.5) else req.team1  # Lower is better
            },
            "score_pct": {
                "team1": team1_stats.get("score_pct", 0.5),
                "team2": team2_stats.get("score_pct", 0.5),
                "advantage": req.team1 if team1_stats.get("score_pct", 0.5) > team2_stats.get("score_pct", 0.5) else req.team2
            },
            "turnover_pct": {
                "team1": team1_stats.get("turnover_pct", 0.5),
                "team2": team2_stats.get("turnover_pct", 0.5),
                "advantage": req.team2 if team1_stats.get("turnover_pct", 0.5) > team2_stats.get("turnover_pct", 0.5) else req.team1  # Lower is better
            },
            "exp_pts_tot": {
                "team1": team1_stats.get("exp_pts_tot", 0),
                "team2": team2_stats.get("exp_pts_tot", 0),
                "advantage": req.team1 if team1_stats.get("exp_pts_tot", 0) > team2_stats.get("exp_pts_tot", 0) else req.team2
            }
        }
        
        # Get win probability for team1
        logger.info("Calling predict_from_features...")
        team1_prob = predict_from_features(matchup_features)
        team2_prob = 1.0 - team1_prob
        
        logger.info(f"Team 1 ({req.team1}) win probability: {team1_prob:.4f}")
        logger.info(f"Team 2 ({req.team2}) win probability: {team2_prob:.4f}")
        
        winner = req.team1 if team1_prob > team2_prob else req.team2
        logger.info(f"Predicted winner: {winner}")
        
        result = MatchupResponse(
            winner=winner,
            team1_win_probability=team1_prob,
            team2_win_probability=team2_prob,
            team1_stats=team1_stats,
            team2_stats=team2_stats,
            matchup_features=matchup_features,
            feature_comparisons=feature_comparisons
        )
        logger.info("="*50)
        logger.info("PREDICTION COMPLETE")
        logger.info(f"Result: {result.dict()}")
        logger.info("="*50)
        
        return result
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        logger.error("Model needs to be trained first. Call /train endpoint.")
        raise HTTPException(
            status_code=404,
            detail="Model not found. Please train the model first by calling /train endpoint."
        )
    except Exception as e:
        logger.error(f"Error in predict_matchup: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")