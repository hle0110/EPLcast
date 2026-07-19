import os
import sys
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import load_data, engineer_features, FEATURE_COLUMNS
from models import ScaledLogisticModel
from simulate import simulate_seasons, summarize_simulations

DATA_PATH = "data/matches.csv"
MODEL_PATH = "models/epl_predictive_model.pkl"
PRED_MATCHES_PATH = "predictions/epl_next5years_matches.csv"
PRED_TABLE_PATH = "predictions/epl_top10_next5years.csv"
N_FUTURE_SEASONS = 5
N_SIMULATIONS = 30
TRAIN_CUTOFF_SEASON = 2023

def build_legacy_features(df_pl):
    df_pl = df_pl.sort_values(['season', 'round_num']).reset_index(drop=True)
    encoder = LabelEncoder()
    encoder.fit(pd.concat([df_pl['home_team_name'], df_pl['away_team_name']]))
    df_pl['home_enc'] = encoder.transform(df_pl['home_team_name'])
    df_pl['away_enc'] = encoder.transform(df_pl['away_team_name'])

    home_part = df_pl[['season', 'round_num', 'home_team_name', 'result']].copy()
    home_part.columns = ['season', 'round_num', 'team', 'result']
    home_part['is_home'] = 1
    home_part['win'] = (home_part['result'] == 'home team win').astype(float)
    away_part = df_pl[['season', 'round_num', 'away_team_name', 'result']].copy()
    away_part.columns = ['season', 'round_num', 'team', 'result']
    away_part['is_home'] = 0
    away_part['win'] = (away_part['result'] == 'away team win').astype(float)
    long_df = pd.concat([home_part, away_part], ignore_index=True)
    long_df['orig_index'] = pd.concat([pd.Series(df_pl.index), pd.Series(df_pl.index)], ignore_index=True)
    long_df = long_df.sort_values(['team', 'season', 'round_num'])
    long_df['last3'] = long_df.groupby('team')['win'].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0.0)
    home_last3 = long_df[long_df['is_home'] == 1].set_index('orig_index')['last3']
    away_last3 = long_df[long_df['is_home'] == 0].set_index('orig_index')['last3']
    df_pl['home_last3'] = df_pl.index.map(home_last3)
    df_pl['away_last3'] = df_pl.index.map(away_last3)
    return df_pl

def evaluate_models(pl):
    train = pl[pl['season'] <= TRAIN_CUTOFF_SEASON].copy()
    test = pl[pl['season'] > TRAIN_CUTOFF_SEASON].copy()
    if len(test) == 0:
        print("Not enough recent seasons for a held-out evaluation split, skipping comparison")
        return

    y_train, y_test = train['result'], test['result']
    majority = y_train.value_counts().idxmax()
    maj_acc = (y_test == majority).mean()

    legacy_all = build_legacy_features(pd.concat([train, test], ignore_index=True))
    legacy_train = legacy_all[legacy_all['season'] <= TRAIN_CUTOFF_SEASON]
    legacy_test = legacy_all[legacy_all['season'] > TRAIN_CUTOFF_SEASON]
    legacy_features = ['home_enc', 'away_enc', 'home_last3', 'away_last3']
    legacy_model = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)
    legacy_model.fit(legacy_train[legacy_features], legacy_train['result'])
    legacy_pred = legacy_model.predict(legacy_test[legacy_features])
    legacy_proba = legacy_model.predict_proba(legacy_test[legacy_features])
    legacy_acc = accuracy_score(legacy_test['result'], legacy_pred)
    legacy_ll = log_loss(legacy_test['result'], legacy_proba, labels=legacy_model.classes_)

    new_model = ScaledLogisticModel(C=0.5)
    new_model.fit(train[FEATURE_COLUMNS], y_train)
    new_pred = new_model.predict(test[FEATURE_COLUMNS])
    new_proba = new_model.predict_proba(test[FEATURE_COLUMNS])
    new_acc = accuracy_score(y_test, new_pred)
    new_ll = log_loss(y_test, new_proba, labels=new_model.classes_)

    print(f"\nHeld-out evaluation on seasons after {TRAIN_CUTOFF_SEASON} ({len(test)} matches)")
    print(f"  Always-predict-majority-class baseline : accuracy {maj_acc:.4f}")
    print(f"  Original model (team id + last-3 form)  : accuracy {legacy_acc:.4f}  log loss {legacy_ll:.4f}")
    print(f"  New model (elo + form + h2h, logistic regression)   : accuracy {new_acc:.4f}  log loss {new_ll:.4f}")
    gain = (new_acc - legacy_acc) * 100
    print(f"  Accuracy improvement: {gain:+.2f} percentage points\n")

def main():
    print("Starting Premier League Predictor")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: data file not found at {DATA_PATH}")
        return

    df = load_data(DATA_PATH)
    print(f"Loaded {len(df):,} historical matches from {DATA_PATH}")

    df, elo_ratings = engineer_features(df)
    print("Engineered elo, form, head-to-head and promotion features")

    pl = df[df['division'] == 'Premier League'].reset_index(drop=True)
    print(f"{len(pl):,} Premier League matches available for training/evaluation")

    evaluate_models(pl)

    print("Training final ensemble model on all available Premier League history")
    final_model = ScaledLogisticModel(C=0.5)
    final_model.fit(pl[FEATURE_COLUMNS], pl['result'])

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    latest_season = pl['season'].max()
    current_teams = sorted(
        set(pl[pl['season'] == latest_season]['home_team_name'])
        | set(pl[pl['season'] == latest_season]['away_team_name'])
    )
    print(f"Simulating from season {latest_season + 1} using the {len(current_teams)} teams from the {latest_season}-{str(latest_season+1)[-2:]} season")

    matches_df, table_df = simulate_seasons(
        final_model, current_teams, elo_ratings, pl,
        n_seasons=N_FUTURE_SEASONS, n_simulations=N_SIMULATIONS,
        start_season=int(latest_season) + 1,
    )
    top10 = summarize_simulations(table_df, current_teams, N_SIMULATIONS)

    os.makedirs("predictions", exist_ok=True)
    matches_df.to_csv(PRED_MATCHES_PATH, index=False)
    top10.to_csv(PRED_TABLE_PATH, index=False)

    print(f"\nPredicted top 10 for the {latest_season + 1}-{str(latest_season+2)[-2:]} season "
          f"(averaged over {N_SIMULATIONS} Monte Carlo simulations)")
    print(top10[top10['Season'] == latest_season + 1].to_string(index=False))

    print(f"\nSaved match-level sample to {PRED_MATCHES_PATH}")
    print(f"Saved simulated standings to {PRED_TABLE_PATH}")
    print("All done")

if __name__ == "__main__":
    main()
