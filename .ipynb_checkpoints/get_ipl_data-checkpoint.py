"""
IPL Ball-by-Ball Data Fetcher (Cricsheet) -- CLEAR PER-BALL OUTCOME VERSION
---------------------------------------------------------------------------
Produces 3 CSVs:
  1. matches.csv             - one row per match
  2. deliveries.csv          - one row per ball with CLEAN outcome columns
  3. deliveries_features.csv - deliveries + ML features

Every ball has simple, readable columns showing exactly what happened:
runs, is_wide, is_noball, is_bye, is_legbye, is_wicket, is_four, is_six,
and a one-word ball_outcome summary.
"""

import pandas as pd
import numpy as np
import zipfile
import requests
import io
import csv
import os

OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

URL = "https://cricsheet.org/downloads/ipl_csv2.zip"

TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}


def clean_team(name):
    if pd.isna(name):
        return name
    return TEAM_NAME_MAP.get(name, name)


# ------------------------------------------------------------------
# 1. Download
# ------------------------------------------------------------------
print("Downloading Cricsheet IPL zip ...")
resp = requests.get(URL, timeout=60)
resp.raise_for_status()
zf = zipfile.ZipFile(io.BytesIO(resp.content))
all_files = zf.namelist()


# ------------------------------------------------------------------
# 2. Ball-by-ball
# ------------------------------------------------------------------
match_files = [f for f in all_files
               if f.endswith(".csv") and "_info" not in f and "README" not in f]

print(f"Reading {len(match_files)} match files ...")
frames = []
for f in match_files:
    try:
        d = pd.read_csv(zf.open(f))
        d["match_id"] = f.replace(".csv", "")
        frames.append(d)
    except Exception as e:
        print("  skip", f, e)

deliveries = pd.concat(frames, ignore_index=True)

# Clean team names
deliveries["batting_team"] = deliveries["batting_team"].apply(clean_team)
deliveries["bowling_team"] = deliveries["bowling_team"].apply(clean_team)


# ------------------------------------------------------------------
# 3. CLEAN PER-BALL OUTCOME COLUMNS  <-- THE FIX YOU ASKED FOR
# ------------------------------------------------------------------
# Cricsheet leaves extras columns as NaN when not applicable. Fill with 0.
for col in ["runs_off_bat", "extras", "wides", "noballs", "byes", "legbyes", "penalty"]:
    if col in deliveries.columns:
        deliveries[col] = deliveries[col].fillna(0)

# Simple 0/1 flag columns
deliveries["is_wide"]    = (deliveries["wides"]   > 0).astype(int)
deliveries["is_noball"]  = (deliveries["noballs"] > 0).astype(int)
deliveries["is_bye"]     = (deliveries["byes"]    > 0).astype(int)
deliveries["is_legbye"]  = (deliveries["legbyes"] > 0).astype(int)
deliveries["is_wicket"]  = deliveries["wicket_type"].notna().astype(int)

# Total runs scored on this ball (batter runs + all extras)
deliveries["total_runs"] = deliveries["runs_off_bat"] + deliveries["extras"]

# Boundaries off the bat
deliveries["is_four"] = (deliveries["runs_off_bat"] == 4).astype(int)
deliveries["is_six"]  = (deliveries["runs_off_bat"] == 6).astype(int)

# Legal delivery (wide/no-ball doesn't count as a ball faced)
deliveries["is_legal_delivery"] = ((deliveries["is_wide"] == 0) &
                                   (deliveries["is_noball"] == 0)).astype(int)


# Human-readable outcome
def ball_outcome(row):
    if row["is_wicket"] == 1:
        return f"WICKET ({row['wicket_type']})"
    if row["is_wide"] == 1:
        return f"wide +{int(row['total_runs'])}"
    if row["is_noball"] == 1:
        return f"no-ball +{int(row['total_runs'])}"
    if row["is_six"] == 1:
        return "SIX"
    if row["is_four"] == 1:
        return "FOUR"
    if row["is_bye"] == 1:
        return f"bye +{int(row['byes'])}"
    if row["is_legbye"] == 1:
        return f"legbye +{int(row['legbyes'])}"
    r = int(row["runs_off_bat"])
    return f"{r} run" + ("s" if r != 1 else "")


deliveries["ball_outcome"] = deliveries.apply(ball_outcome, axis=1)

print(f"\nDeliveries shape: {deliveries.shape}")

# ------------------------------------------------------------------
# 4. Print a sample so you SEE the outcome columns
# ------------------------------------------------------------------
print("\n--- SAMPLE: first 12 balls of first match ---")
first_match = deliveries["match_id"].iloc[0]
sample = deliveries[deliveries["match_id"] == first_match].head(12)
print(sample[["ball", "striker", "bowler", "runs_off_bat", "extras",
              "is_wide", "is_noball", "is_wicket", "total_runs",
              "ball_outcome"]].to_string(index=False))


# ------------------------------------------------------------------
# 5. Match info
# ------------------------------------------------------------------
info_files = [f for f in all_files if "_info.csv" in f]
print(f"\nReading {len(info_files)} info files ...")

matches = []
for f in info_files:
    match_id = f.replace("_info.csv", "")
    raw = zf.open(f).read().decode("utf-8")
    reader = csv.reader(raw.splitlines())
    info = {"match_id": match_id}
    teams, poms = [], []
    for row in reader:
        if len(row) < 2:
            continue
        key = row[1]
        val = row[2] if len(row) > 2 else None
        if key == "team":
            teams.append(val)
        elif key in ("winner", "venue", "city", "toss_winner", "toss_decision",
                     "date", "season", "match_type", "outcome", "method",
                     "result", "winner_runs", "winner_wickets", "eliminator"):
            info[key] = val
        elif key == "player_of_match":
            poms.append(val)
    if len(teams) >= 2:
        info["team1"] = clean_team(teams[0])
        info["team2"] = clean_team(teams[1])
    info["winner"]      = clean_team(info.get("winner"))
    info["toss_winner"] = clean_team(info.get("toss_winner"))
    info["player_of_match"] = ", ".join(poms) if poms else None
    matches.append(info)

matches_df = pd.DataFrame(matches)
print(f"Matches shape: {matches_df.shape}")


# ------------------------------------------------------------------
# 6. ML feature engineering
# ------------------------------------------------------------------
print("Engineering ML features ...")
df = deliveries.sort_values(["match_id", "innings", "ball"]).reset_index(drop=True)
g = df.groupby(["match_id", "innings"], sort=False)

df["current_score"]   = g["total_runs"].cumsum()
df["current_wickets"] = g["is_wicket"].cumsum()

df["over_num"]     = df["ball"].astype(float).apply(lambda x: int(x))
df["ball_in_over"] = (df["ball"].astype(float) * 10).round().astype(int) % 10
df["legal_balls_bowled"] = g["is_legal_delivery"].cumsum()
df["balls_remaining"]    = (120 - df["legal_balls_bowled"]).clip(lower=0)

df["current_run_rate"] = (df["current_score"].astype(float) /
                          df["legal_balls_bowled"].replace(0, np.nan) * 6)

df["runs_last_30_balls"]    = g["total_runs"].transform(
    lambda s: s.rolling(30, min_periods=1).sum())
df["wickets_last_30_balls"] = g["is_wicket"].transform(
    lambda s: s.rolling(30, min_periods=1).sum())

df["phase"] = pd.cut(df["over_num"],
                    bins=[-1, 5, 14, 20],
                    labels=["powerplay", "middle", "death"])

df = df.merge(
    matches_df[["match_id", "season", "venue", "city",
                "toss_winner", "toss_decision", "winner"]],
    on="match_id", how="left", suffixes=("", "_m"))

print(f"Feature df shape: {df.shape}")


# ------------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------------
paths = {
    "matches":    os.path.join(OUT_DIR, "matches.csv"),
    "deliveries": os.path.join(OUT_DIR, "deliveries.csv"),
    "features":   os.path.join(OUT_DIR, "deliveries_features.csv"),
}
matches_df.to_csv(paths["matches"], index=False)
deliveries.to_csv(paths["deliveries"], index=False)
df.to_csv(paths["features"], index=False)

print("\nSaved:")
for name, p in paths.items():
    print(f"  {name:<11} {p}  ({os.path.getsize(p)/1024/1024:.1f} MB)")