# Wagerly - NFL Outcome Predictor

A web application that predicts NFL game outcomes using team performance metrics and statistical modeling.

## What It Does

Wagerly allows users to select two NFL teams and predicts the winner of a hypothetical matchup. The app displays:
- **Predicted winner** with win probability percentage
- **Detailed stats breakdown** comparing both teams across 8 key metrics
- **Power advantage** showing which team has the statistical edge

## How It Works

### Frontend
- Built with vanilla HTML/CSS/JavaScript
- Served via Vite dev server
- Mortal Kombat-style team selection interface
- Calls the FastAPI backend for predictions

### Backend
- **FastAPI** Python server
- RESTful API with CORS enabled
- Hardcoded season statistics for all 32 NFL teams

---

## The Prediction Model

### Team Metrics

Each team has the following stats (based on recent NFL season data):

| Metric | Description |
|--------|-------------|
| `power` | Overall team power rating (0-100 scale) |
| `points_diff` | Season point differential |
| `mov` | Margin of Victory (average) |
| `total_yards` | Total offensive yards |
| `turnovers` | Total turnovers committed |
| `score_pct` | Scoring percentage |
| `turnover_pct` | Turnover percentage |
| `exp_pts_tot` | Expected points total |

### The Math: Logistic Probability Function

The win probability is calculated using a **logistic (sigmoid) function** based on the power rating difference between teams:

```
P(Team1 wins) = 1 / (1 + e^(-power_diff / 15))
```

Where:
- `power_diff = Team1_power - Team2_power`
- `e` is Euler's number (~2.718)
- `15` is a scaling factor that controls how quickly probability changes with power difference

#### Why Logistic Function?

The logistic function is ideal for probability because:
1. **Output is always between 0 and 1** (valid probability)
2. **Symmetric** - a +20 power advantage gives the same edge as -20 gives the opponent
3. **Diminishing returns** - going from 80 to 90 power matters less than 40 to 50
4. **Industry standard** - used in sports betting and ELO rating systems

#### Example Calculations

| Power Diff | Win Probability |
|------------|-----------------|
| 0 | 50.0% |
| +10 | 66.0% |
| +20 | 79.0% |
| +30 | 88.1% |
| +40 | 93.5% |
| -10 | 34.0% |
| -20 | 21.0% |

### Power Ratings

The `power` rating (0-100) is the primary factor in predictions. It's a composite score derived from:
- Point differential
- Margin of victory
- Offensive production (yards)
- Turnover efficiency
- Scoring efficiency

**Example power ratings:**
- Kansas City Chiefs: 95 (elite)
- San Francisco 49ers: 92
- Detroit Lions: 90
- Carolina Panthers: 28 (rebuilding)

---

## Running the App

### Prerequisites
- Node.js (for frontend)
- Python 3.11+ (for backend)
- pip packages: `fastapi`, `uvicorn`, `pydantic`

### Start the Backend
```bash
cd Backend
uvicorn app:app --reload --port 8000
```

### Start the Frontend
```bash
npm install
npm run dev
```

Then open `http://localhost:3000` in your browser.

---

## API Endpoints

### `GET /`
Health check - returns API status

### `GET /teams`
Returns all 32 NFL teams with their stats

### `POST /predict-matchup`
Predicts the winner of a matchup

**Request:**
```json
{
  "team1": "KC",
  "team2": "SF"
}
```

**Response:**
```json
{
  "team1": "KC",
  "team2": "SF",
  "team1_name": "Kansas City Chiefs",
  "team2_name": "San Francisco 49ers",
  "team1_stats": { ... },
  "team2_stats": { ... },
  "winner": "KC",
  "winner_name": "Kansas City Chiefs",
  "win_probability": 54.7,
  "power_diff": 3.0
}
```

---

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript, Vite
- **Backend:** Python, FastAPI, Pydantic
- **Styling:** Custom CSS with CSS variables

---

## Team

GDG AI/ML Collab Group @ NYU

