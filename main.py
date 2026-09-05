import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import load_data, engineer_features, FEATURE_COLUMNS, TOP_DIVISION
from models import ScaledLogisticModel
from simulate import run_simulations, summarize_simulations

DATA_PATH = "data/matches.csv"
MODEL_PATH = "models/epl_predictive_model.pkl"
PRED_MATCHES_PATH = "predictions/epl_simulated_matches.csv"
PRED_TABLE_PATH = "predictions/epl_season_projection.csv"
N_FUTURE_SEASONS = 3
N_SIMULATIONS = 40
MODEL_C = 0.8
TEAMS_PER_SEASON = 20

MIN_TRAINING_SEASONS = 2

def rolling_origin_folds(top_flight):
    seasons = sorted(top_flight['season'].unique())
    folds = []
    for season in seasons[MIN_TRAINING_SEASONS:]:
        season_matches = top_flight[top_flight['season'] == season]
        if len(season_matches) < 20:
            continue
        season_start = season_matches['match_date'].min()
        folds.append((f"{season}-{str(season + 1)[-2:]} full", season_start, season_matches.index))
        midpoint = pd.Timestamp(year=season + 1, month=1, day=1)
        second_half = season_matches[season_matches['match_date'] >= midpoint]
        if len(second_half) >= 40:
            folds.append((f"{season}-{str(season + 1)[-2:]} 2nd half", midpoint, second_half.index))
    return folds

def build_legacy_features(top_flight):
    frame = top_flight.sort_values('match_date', kind='mergesort').copy()
    encoder = LabelEncoder()
    encoder.fit(pd.concat([frame['home_team_name'], frame['away_team_name']]))
    frame['home_enc'] = encoder.transform(frame['home_team_name'])
    frame['away_enc'] = encoder.transform(frame['away_team_name'])

    home_part = frame[['match_date', 'home_team_name', 'result']].copy()
    home_part.columns = ['match_date', 'team', 'result']
    home_part['is_home'] = 1
    home_part['win'] = (home_part['result'] == 'home team win').astype(float)
    home_part['orig_index'] = frame.index
    away_part = frame[['match_date', 'away_team_name', 'result']].copy()
    away_part.columns = ['match_date', 'team', 'result']
    away_part['is_home'] = 0
    away_part['win'] = (away_part['result'] == 'away team win').astype(float)
    away_part['orig_index'] = frame.index
    long_df = pd.concat([home_part, away_part], ignore_index=True)
    long_df = long_df.sort_values(['team', 'match_date'], kind='mergesort')
    long_df['last3'] = long_df.groupby('team')['win'].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0.0)
    frame['home_last3'] = long_df[long_df['is_home'] == 1].set_index('orig_index')['last3']
    frame['away_last3'] = long_df[long_df['is_home'] == 0].set_index('orig_index')['last3']
    return frame

def evaluate_models(df, top_flight):
    folds = rolling_origin_folds(top_flight)
    if not folds:
        print("Not enough seasons for a held-out evaluation, skipping comparison")
        return

    legacy = build_legacy_features(top_flight)
    legacy_features = ['home_enc', 'away_enc', 'home_last3', 'away_last3']

    totals = {'n': 0, 'majority': 0.0, 'legacy_acc': 0.0, 'legacy_ll': 0.0, 'new_acc': 0.0, 'new_ll': 0.0}
    print(f"\nRolling-origin evaluation ({len(folds)} held-out windows, each trained only on earlier matches)")
    for name, cutoff_date, test_index in folds:
        test = top_flight.loc[test_index]
        train = df[df['match_date'] < cutoff_date]
        train_top = top_flight[top_flight['match_date'] < cutoff_date]
        if len(train_top) < 200 or len(test) < 20:
            continue

        majority_class = train_top['result'].value_counts().idxmax()
        majority_acc = (test['result'] == majority_class).mean()

        legacy_train = legacy[legacy['match_date'] < cutoff_date]
        legacy_test = legacy.loc[test_index]
        legacy_model = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)
        legacy_model.fit(legacy_train[legacy_features], legacy_train['result'])
        legacy_pred = legacy_model.predict(legacy_test[legacy_features])
        legacy_acc = accuracy_score(legacy_test['result'], legacy_pred)
        legacy_ll = log_loss(legacy_test['result'], legacy_model.predict_proba(legacy_test[legacy_features]), labels=legacy_model.classes_)

        model = ScaledLogisticModel(C=MODEL_C)
        model.fit(train[FEATURE_COLUMNS], train['result'])
        proba = model.predict_proba(test[FEATURE_COLUMNS])
        new_acc = accuracy_score(test['result'], model.predict(test[FEATURE_COLUMNS]))
        new_ll = log_loss(test['result'], proba, labels=model.classes_)

        n = len(test)
        totals['n'] += n
        totals['majority'] += majority_acc * n
        totals['legacy_acc'] += legacy_acc * n
        totals['legacy_ll'] += legacy_ll * n
        totals['new_acc'] += new_acc * n
        totals['new_ll'] += new_ll * n
        print(f"  {name:24s} n={n:3d}  baseline {majority_acc:.3f}  original {legacy_acc:.3f}  current {new_acc:.3f}")

    n = totals['n']
    print(f"\n  Weighted over {n:,} held-out matches")
    print(f"    Always-predict-home-win baseline : accuracy {totals['majority']/n:.4f}")
    print(f"    Original model (team id + form)  : accuracy {totals['legacy_acc']/n:.4f}  log loss {totals['legacy_ll']/n:.4f}")
    print(f"    Current model                    : accuracy {totals['new_acc']/n:.4f}  log loss {totals['new_ll']/n:.4f}")
    print(f"    Improvement over original        : {(totals['new_acc'] - totals['legacy_acc'])/n*100:+.2f} percentage points\n")

def main():
    print("Starting Premier League Predictor")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: data file not found at {DATA_PATH}")
        print("Run 'python update_data.py --rebuild' to download the dataset first")
        return

    df = load_data(DATA_PATH)
    print(f"Loaded {len(df):,} matches from {DATA_PATH} ({df['match_date'].min().date()} to {df['match_date'].max().date()})")

    df, elo_ratings = engineer_features(df)
    top_flight = df[df['division'] == TOP_DIVISION]
    print(f"{len(top_flight):,} {TOP_DIVISION} matches, {len(df):,} matches in total across four divisions")

    evaluate_models(df, top_flight)

    print("Training final model on every division and season available")
    final_model = ScaledLogisticModel(C=MODEL_C)
    final_model.fit(df[FEATURE_COLUMNS], df['result'])
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    current_season = int(top_flight['season'].max())
    season_matches = top_flight[top_flight['season'] == current_season]
    current_teams = sorted(set(season_matches['home_team_name']) | set(season_matches['away_team_name']))
    played = len(season_matches)
    total_fixtures = len(current_teams) * (len(current_teams) - 1)
    season_label = f"{current_season}-{str(current_season + 1)[-2:]}"

    if len(current_teams) != TEAMS_PER_SEASON:
        print(f"WARNING: found {len(current_teams)} teams in {season_label}, expected {TEAMS_PER_SEASON}")

    if played < total_fixtures:
        print(f"\n{season_label} is in progress: {played} of {total_fixtures} matches played")
        print(f"Simulating the remaining {total_fixtures - played} fixtures {N_SIMULATIONS} times, "
              f"then {N_FUTURE_SEASONS} further seasons")
    else:
        print(f"\n{season_label} is complete, simulating {N_FUTURE_SEASONS} future seasons {N_SIMULATIONS} times")

    matches_df, table_df = run_simulations(
        final_model, df, current_season, current_teams, elo_ratings,
        n_future_seasons=N_FUTURE_SEASONS, n_simulations=N_SIMULATIONS,
    )
    projection = summarize_simulations(table_df, current_teams, N_SIMULATIONS)

    os.makedirs("predictions", exist_ok=True)
    matches_df.to_csv(PRED_MATCHES_PATH, index=False)
    projection.to_csv(PRED_TABLE_PATH, index=False)

    headline_season = current_season if played < total_fixtures else current_season + 1
    headline = projection[projection['Season'] == headline_season]
    label = f"{headline_season}-{str(headline_season + 1)[-2:]}"
    print(f"\nProjected final {label} table (average of {N_SIMULATIONS} simulations)")
    display = headline[['Rank', 'Team', 'Points', 'Wins', 'Draws', 'Losses', 'GoalDiff', 'TitleProb', 'Top4Prob', 'RelegationProb']]
    print(display.to_string(index=False))

    print(f"\nSaved simulated matches to {PRED_MATCHES_PATH}")
    print(f"Saved projected standings to {PRED_TABLE_PATH}")
    print("All done")

if __name__ == "__main__":
    main()
