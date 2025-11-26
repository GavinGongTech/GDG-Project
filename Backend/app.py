from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Wagerly NFL Prediction API")

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class MatchupRequest(BaseModel):
    team1: str  # Team abbreviation (e.g., "KC")
    team2: str  # Team abbreviation (e.g., "SF")


class TeamStats(BaseModel):
    points_diff: float
    mov: float
    total_yards: float
    turnovers: float
    score_pct: float
    turnover_pct: float
    exp_pts_tot: float
    power: float


class MatchupResponse(BaseModel):
    team1: str
    team2: str
    team1_name: str
    team2_name: str
    team1_stats: TeamStats
    team2_stats: TeamStats
    winner: str
    winner_name: str
    win_probability: float
    power_diff: float


@app.get("/")
def root():
    return {"message": "Wagerly NFL Prediction API is running!", "docs": "/docs"}


@app.get("/teams")
def get_teams():
    """Get all available teams with their stats."""
    return {abbr: {"name": TEAM_NAMES[abbr], **stats} for abbr, stats in TEAM_STATS.items()}


@app.post("/predict-matchup", response_model=MatchupResponse)
def predict_matchup(req: MatchupRequest):
    """Predict the winner of a matchup between two teams."""
    import math
    
    team1 = req.team1.upper()
    team2 = req.team2.upper()
    
    if team1 not in TEAM_STATS:
        return {"error": f"Unknown team: {team1}"}
    if team2 not in TEAM_STATS:
        return {"error": f"Unknown team: {team2}"}
    
    stats1 = TEAM_STATS[team1]
    stats2 = TEAM_STATS[team2]
    
    # Calculate win probability based on power ratings
    power1 = stats1["power"]
    power2 = stats2["power"]
    
    # Use a logistic function to convert power difference to probability
    power_diff = power1 - power2
    win_prob_team1 = 1 / (1 + math.exp(-power_diff / 15))
    
    # Determine winner
    if win_prob_team1 >= 0.5:
        winner = team1
        winner_name = TEAM_NAMES[team1]
        win_probability = win_prob_team1
    else:
        winner = team2
        winner_name = TEAM_NAMES[team2]
        win_probability = 1 - win_prob_team1
    
    return MatchupResponse(
        team1=team1,
        team2=team2,
        team1_name=TEAM_NAMES[team1],
        team2_name=TEAM_NAMES[team2],
        team1_stats=TeamStats(**{k: v for k, v in stats1.items()}),
        team2_stats=TeamStats(**{k: v for k, v in stats2.items()}),
        winner=winner,
        winner_name=winner_name,
        win_probability=round(win_probability * 100, 1),
        power_diff=round(power_diff, 1)
    )
