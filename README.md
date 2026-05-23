# 🏏 IPL Analytics & ML Project

A complete end-to-end machine learning project on IPL ball-by-ball data — from raw data ingestion to predictive models for match winners, season awards, and playoff qualification.

Built for a data science / ML learning session, mixing **engaging EDA** with **production-style ML pipelines**.

---

## 📦 What's in this project

### Data pipeline
| File | Purpose |
|---|---|
| `get_ipl_data.py` | Downloads ball-by-ball data from [Cricsheet](https://cricsheet.org/), cleans team names, engineers features (cumulative score, run rate, phase, etc.), and saves 3 CSVs ready for analysis |

### Jupyter Notebooks

| Notebook | What it does | Audience |
|---|---|---|
| `ipl_wow_facts.ipynb` | 🎤 10 hook-style "did you know?" stats with interactive Plotly charts. Perfect for opening a session. | Beginners / general fans |
| `ipl_eda_insights.ipynb` | 📊 Deep exploratory analysis — powerplay impact, venue patterns, chase win % by required run rate, head-to-head matrix | Mixed audience |
| `ipl_win_predictor.ipynb` | 🤖 Trains 3 ML models (Logistic Regression, Random Forest, Gradient Boosting) to predict live chase win probability ball-by-ball. Includes calibration check and live match replay. | Technical |
| `ipl_awards_predictor.ipynb` | 🏆 Predicts Orange Cap, Purple Cap, Most 4s, Most 6s — both for the current season (projection) and next season (ML model trained on per-player-per-season history) | Technical |
| `ipl_playoff_predictor.ipynb` | 🎯 Predicts the 4th playoff qualifier using Monte Carlo simulation (10K runs) + deterministic "what-if" scenario enumeration with NRR tiebreaker margins | Technical |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn plotly joblib requests
```

### 2. Download the data (one time, ~30 seconds)

```bash
python get_ipl_data.py
```

This creates an `ipl_data/` folder with:
- `matches.csv` — one row per match (~1,225 rows)
- `deliveries.csv` — one row per ball (~291,000 rows)
- `deliveries_features.csv` — deliveries + engineered features ready for ML

### 3. Open the notebooks

```bash
jupyter notebook
```

Recommended order for a session:
1. **Start:** `ipl_wow_facts.ipynb` — warm up the audience with surprising stats
2. **Explore:** `ipl_eda_insights.ipynb` — build intuition with deeper EDA
3. **Predict:** `ipl_win_predictor.ipynb` — show how ML turns intuition into predictions
4. **Apply:** `ipl_awards_predictor.ipynb` and `ipl_playoff_predictor.ipynb` — use the model thinking on real season questions

---

## 📂 Project Structure

```
ipl-ml-project/
├── README.md
├── get_ipl_data.py                  # data fetcher + feature engineering
├── ipl_data/                        # generated CSVs (after running script)
│   ├── matches.csv
│   ├── deliveries.csv
│   └── deliveries_features.csv
├── models/                          # saved ML models (after training)
│   └── ipl_win_predictor.pkl
└── notebooks/
    ├── ipl_wow_facts.ipynb
    ├── ipl_eda_insights.ipynb
    ├── ipl_win_predictor.ipynb
    ├── ipl_awards_predictor.ipynb
    └── ipl_playoff_predictor.ipynb
```

---

## 🔍 Notebook deep-dive

### `ipl_wow_facts.ipynb` — Session opener
10 counter-intuitive facts, each formatted as: **🎤 Ask audience → 🥁 build-up → 🎯 reveal viz → 💡 takeaway**. Topics include the toss myth, run-rate explosion, most dangerous over, dot-ball reality, six-hitting kings, dismissal patterns, lowest defended totals, powerplay premium, and more.

### `ipl_eda_insights.ipynb` — Live-commentary stats
12 sections covering: phase-level summaries, powerplay runs → win %, wickets impact, runs×wickets heatmap, death overs, per-team powerplay profiles, chase win % by required run rate, venue insights (batter's paradise vs bowler's grave), toss impact, ball-by-ball win-probability curves, head-to-head matrix.

### `ipl_win_predictor.ipynb` — The core ML model
- **Target:** chase win probability (innings 2)
- **Features:** runs left, balls left, wickets in hand, current/required run rate, recent runs/wickets in last 30 balls, batting team, bowling team, venue
- **Critical:** train/test split by `match_id` (not random rows) to prevent leakage
- **Models:** Logistic Regression, Random Forest, Gradient Boosting compared by AUC, log-loss, Brier score
- **Calibration check:** does the model say "70%" when teams actually win 70% of the time?
- **Bonus:** ball-by-ball match replay showing the WP curve with wicket markers

### `ipl_awards_predictor.ipynb` — Player-level forecasts
For each award (Orange Cap / Purple Cap / Most 4s / Most 6s):
1. **Current-season:** live leaderboard with projection based on matches played
2. **Next-season:** ML model trained on per-player-per-season history with features like weighted-average last 3 seasons, career mean/max, experience

### `ipl_playoff_predictor.ipynb` — Who grabs 4th place?
- Builds live points table from completed matches (W/L/NR/Points/NRR)
- Uses your hand-entered remaining fixtures list (no schedule guessing)
- **Mathematical lock/eliminate check:** identifies teams whose fate is already decided
- **Monte Carlo (10K sims)** on contested matches only
- **What-if scenario enumeration:** for 4 remaining matches = 16 scenarios, prints hard requirements per team like *"DC must beat KKR AND PBKS must beat LSG"*
- **NRR margin needed:** estimates win margin needed for tiebreaker

---

## 🛠️ How `get_ipl_data.py` works

Downloads the Cricsheet IPL zip (~10MB), reads every match CSV + info file, and produces:

**Cleaning steps:**
- Normalizes team names (Delhi Daredevils → Delhi Capitals, Kings XI Punjab → Punjab Kings, Royal Challengers Bangalore → Royal Challengers Bengaluru). Without this, the same franchise appears as different teams across seasons and breaks every model.
- Fills NaN extras with 0 (a "not a wide" ball is 0, not missing)

**Engineered per-ball features:**
- `is_wide`, `is_noball`, `is_bye`, `is_legbye`, `is_wicket` (0/1 flags)
- `is_four`, `is_six`, `is_legal_delivery`
- `ball_outcome` — one-word summary like `FOUR`, `SIX`, `WICKET (caught)`, `wide +2`
- `current_score`, `current_wickets` — running totals within each innings
- `balls_remaining`, `current_run_rate`
- `runs_last_30_balls`, `wickets_last_30_balls` — rolling momentum features
- `phase` — `powerplay` / `middle` / `death`

---

## 🎯 Modeling decisions & gotchas

1. **Split by match_id, not by row.** If you split rows randomly, balls from the same match leak across train and test, inflating accuracy. The notebooks always split by `match_id`.

2. **Team name normalization is mandatory.** IPL franchises rebrand. If your model treats "Punjab Kings" and "Kings XI Punjab" as separate teams, you halve the data per team and learn nothing useful.

3. **Wickets credited to bowler vs not.** Run-outs aren't bowler's wickets. The awards notebook filters bowler credit to: bowled, caught, lbw, caught and bowled, stumped, hit wicket.

4. **Calibration > accuracy** for probability models. A model that says "80%" should actually win ~80% of the time. The win predictor includes a calibration plot.

5. **Time-aware validation for awards.** Train on seasons 1..N-1, validate on season N. Random k-fold leaks future info into past predictions.

---

## 🔌 Data source

All data from [**Cricsheet**](https://cricsheet.org/) — an open-source cricket data repository maintained by Stephen Rushe. Used under their CC-BY-4.0-equivalent licence. If you publish anything based on this work, credit Cricsheet.

---

## 📈 Possible extensions

- **Player-level features** in the win predictor (currently only team-level): striker career SR, bowler economy at this venue, etc.
- **First-innings score predictor** as a companion to the chase model
- **Streamlit/Gradio dashboard** wrapping `predict_win()` in a slider UI
- **Player-against-bowler matchups** (e.g., Rohit's stats vs left-arm pace)
- **Auction value prediction** based on previous-season performance

---

## 🙏 Credits

- **Data:** [Cricsheet](https://cricsheet.org/)
- **Built with:** pandas, numpy, scikit-learn, plotly, jupyter
