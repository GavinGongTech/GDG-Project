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

app = FastAPI(title="Wagerly NFL Prediction API")

# Hardcoded team stats (based on recent NFL season averages)
# Format: points_diff, mov, total_yards, turnovers, score_pct, turnover_pct, exp_pts_tot
TEAM_STATS = {
    "ARI": {"points_diff": -85, "mov": -5.0, "total_yards": 5400, "turnovers": 24, "score_pct": 34.2, "turnover_pct": 12.5, "exp_pts_tot": 18.5, "power": 42},
    "ATL": {"points_diff": 20, "mov": 1.2, "total_yards": 5800, "turnovers": 18, "score_pct": 38.5, "turnover_pct": 9.8, "exp_pts_tot": 22.3, "power": 55},
    "BAL": {"points_diff": 180, "mov": 10.6, "total_yards": 6200, "turnovers": 12, "score_pct": 45.2, "turnover_pct": 6.5, "exp_pts_tot": 30.5, "power": 88},
    "BUF": {"points_diff": 145, "mov": 8.5, "total_yards": 6000, "turnovers": 14, "score_pct": 42.8, "turnover_pct": 7.2, "exp_pts_tot": 28.2, "power": 82},
    "CAR": {"points_diff": -150, "mov": -8.8, "total_yards": 4800, "turnovers": 28, "score_pct": 28.5, "turnover_pct": 15.2, "exp_pts_tot": 14.8, "power": 28},
    "CHI": {"points_diff": -60, "mov": -3.5, "total_yards": 5200, "turnovers": 22, "score_pct": 32.5, "turnover_pct": 11.8, "exp_pts_tot": 17.5, "power": 45},
    "CIN": {"points_diff": 55, "mov": 3.2, "total_yards": 5700, "turnovers": 16, "score_pct": 40.2, "turnover_pct": 8.5, "exp_pts_tot": 24.5, "power": 68},
    "CLE": {"points_diff": -40, "mov": -2.4, "total_yards": 5100, "turnovers": 20, "score_pct": 33.8, "turnover_pct": 10.5, "exp_pts_tot": 18.2, "power": 48},
    "DAL": {"points_diff": 90, "mov": 5.3, "total_yards": 5900, "turnovers": 15, "score_pct": 41.5, "turnover_pct": 8.0, "exp_pts_tot": 26.8, "power": 72},
    "DEN": {"points_diff": 25, "mov": 1.5, "total_yards": 5400, "turnovers": 19, "score_pct": 36.2, "turnover_pct": 10.2, "exp_pts_tot": 20.5, "power": 52},
    "DET": {"points_diff": 160, "mov": 9.4, "total_yards": 6300, "turnovers": 11, "score_pct": 46.5, "turnover_pct": 5.8, "exp_pts_tot": 31.2, "power": 90},
    "GB": {"points_diff": 75, "mov": 4.4, "total_yards": 5650, "turnovers": 17, "score_pct": 39.8, "turnover_pct": 9.0, "exp_pts_tot": 24.0, "power": 70},
    "HOU": {"points_diff": 65, "mov": 3.8, "total_yards": 5550, "turnovers": 18, "score_pct": 38.2, "turnover_pct": 9.5, "exp_pts_tot": 23.5, "power": 65},
    "IND": {"points_diff": -30, "mov": -1.8, "total_yards": 5300, "turnovers": 21, "score_pct": 34.5, "turnover_pct": 11.0, "exp_pts_tot": 19.2, "power": 50},
    "JAX": {"points_diff": -45, "mov": -2.6, "total_yards": 5250, "turnovers": 22, "score_pct": 33.2, "turnover_pct": 11.5, "exp_pts_tot": 18.8, "power": 47},
    "KC": {"points_diff": 190, "mov": 11.2, "total_yards": 6100, "turnovers": 10, "score_pct": 47.5, "turnover_pct": 5.2, "exp_pts_tot": 32.0, "power": 95},
    "LV": {"points_diff": -70, "mov": -4.1, "total_yards": 5150, "turnovers": 23, "score_pct": 31.8, "turnover_pct": 12.0, "exp_pts_tot": 17.0, "power": 40},
    "LAC": {"points_diff": 50, "mov": 2.9, "total_yards": 5600, "turnovers": 17, "score_pct": 39.0, "turnover_pct": 9.2, "exp_pts_tot": 23.0, "power": 62},
    "LAR": {"points_diff": 45, "mov": 2.6, "total_yards": 5500, "turnovers": 18, "score_pct": 38.0, "turnover_pct": 9.6, "exp_pts_tot": 22.5, "power": 60},
    "MIA": {"points_diff": 80, "mov": 4.7, "total_yards": 6050, "turnovers": 16, "score_pct": 41.0, "turnover_pct": 8.2, "exp_pts_tot": 26.0, "power": 73},
    "MIN": {"points_diff": 70, "mov": 4.1, "total_yards": 5700, "turnovers": 16, "score_pct": 40.5, "turnover_pct": 8.5, "exp_pts_tot": 25.0, "power": 71},
    "NE": {"points_diff": -100, "mov": -5.9, "total_yards": 4900, "turnovers": 26, "score_pct": 29.5, "turnover_pct": 14.0, "exp_pts_tot": 15.5, "power": 35},
    "NO": {"points_diff": 15, "mov": 0.9, "total_yards": 5450, "turnovers": 19, "score_pct": 37.0, "turnover_pct": 10.0, "exp_pts_tot": 21.0, "power": 54},
    "NYG": {"points_diff": -120, "mov": -7.1, "total_yards": 4750, "turnovers": 27, "score_pct": 28.0, "turnover_pct": 14.5, "exp_pts_tot": 14.2, "power": 30},
    "NYJ": {"points_diff": -55, "mov": -3.2, "total_yards": 5100, "turnovers": 21, "score_pct": 32.8, "turnover_pct": 11.2, "exp_pts_tot": 17.8, "power": 44},
    "PHI": {"points_diff": 130, "mov": 7.6, "total_yards": 6000, "turnovers": 13, "score_pct": 43.5, "turnover_pct": 6.8, "exp_pts_tot": 28.8, "power": 85},
    "PIT": {"points_diff": 35, "mov": 2.1, "total_yards": 5350, "turnovers": 18, "score_pct": 37.5, "turnover_pct": 9.5, "exp_pts_tot": 21.5, "power": 58},
    "SF": {"points_diff": 170, "mov": 10.0, "total_yards": 6250, "turnovers": 11, "score_pct": 46.0, "turnover_pct": 5.5, "exp_pts_tot": 30.8, "power": 92},
    "SEA": {"points_diff": 40, "mov": 2.4, "total_yards": 5550, "turnovers": 19, "score_pct": 37.8, "turnover_pct": 10.0, "exp_pts_tot": 22.0, "power": 59},
    "TB": {"points_diff": 60, "mov": 3.5, "total_yards": 5650, "turnovers": 17, "score_pct": 39.5, "turnover_pct": 9.0, "exp_pts_tot": 24.2, "power": 66},
    "TEN": {"points_diff": -90, "mov": -5.3, "total_yards": 5000, "turnovers": 25, "score_pct": 30.5, "turnover_pct": 13.5, "exp_pts_tot": 16.0, "power": 38},
    "WAS": {"points_diff": 85, "mov": 5.0, "total_yards": 5750, "turnovers": 15, "score_pct": 40.8, "turnover_pct": 8.0, "exp_pts_tot": 25.5, "power": 74},
}

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders"
}

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
        "*",  # Also allow all origins for flexibility
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",  # Allow any localhost port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS middleware configured - allowing localhost:3000, localhost:3001, and any localhost port via regex")

# Pydantic models for API requests/responses
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


class TeamStats(BaseModel):
    points_diff: float
    mov: float
    total_yards: float
    turnovers: float
    score_pct: float
    turnover_pct: float
    exp_pts_tot: float
    power: float


class MatchupRequest(BaseModel):
    team1: str  # Team abbreviation (e.g., "KC")
    team2: str  # Team abbreviation (e.g., "SF")


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
    return {"message": "Wagerly NFL Prediction API is running!", "docs": "/docs"}


@app.get("/teams")
def get_teams():
    """Get all available teams with their stats."""
    logger.info("GET /teams - Teams list requested")
    return {abbr: {"name": TEAM_NAMES[abbr], **stats} for abbr, stats in TEAM_STATS.items()}


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
