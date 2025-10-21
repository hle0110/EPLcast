import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from itertools import combinations

# progress bar
def progress_bar(iterable, prefix="", size=40):
    total = len(iterable)
    for i, item in enumerate(iterable):
        percent = (i+1)/total
        bar = "█"*int(percent*size) + "-"*(size-int(percent*size))
        print(f"\r{prefix} |{bar}| {i+1}/{total}", end="")
        yield item
    print()

print("🚀 Starting Premier League Predictor...")

# 1. Load data
data_path = "data/matches.csv"
if not os.path.exists(data_path):
    print(f"❌ ERROR: Data file not found at {data_path}")
    exit()
df = pd.read_csv(data_path, low_memory=False)
print(f"✅ Loaded {len(df):,} matches from {data_path}")

# 2. Clean and encode
df.dropna(subset=['home_team_name','away_team_name','home_team_score','away_team_score','result'], inplace=True)
encoder = LabelEncoder()
df['home_enc'] = encoder.fit_transform(df['home_team_name'])
df['away_enc'] = encoder.transform(df['away_team_name'])
print("✅ Team names encoded")

# 3. Last 3 match win rate
df = df.sort_values(['season','match_id']).reset_index(drop=True)
df['home_last3'] = 0.0
df['away_last3'] = 0.0
def last3_win_rate(team, row_index, df_stats):
    matches = df_stats[(df_stats['home_team_name']==team)|(df_stats['away_team_name']==team)]
    matches = matches.loc[:row_index-1]
    if len(matches)==0: return 0.0
    wins=0
    for _, r in matches.tail(3).iterrows():
        res = str(r['result']).strip().lower()
        if r['home_team_name']==team and res=='home team win': wins+=1
        elif r['away_team_name']==team and res=='away team win': wins+=1
    return wins/3

print("📊 Calculating rolling win rates...")
for idx, row in enumerate(df.itertuples(), 1):
    df.at[row.Index,'home_last3'] = last3_win_rate(row.home_team_name, row.Index, df)
    df.at[row.Index,'away_last3'] = last3_win_rate(row.away_team_name, row.Index, df)
    if idx % 5000 == 0:
        print(f"  Processed {idx}/{len(df)} matches", end="\r")
print("\n✅ Rolling win rates done")

# 4. Train model
features = ['home_enc','away_enc','home_last3','away_last3']
X = df[features]
y = df['result']
print("🌲 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)
model.fit(X,y)
print("✅ Model trained successfully")

# 5. Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model,"models/epl_predictive_model.pkl")
print("💾 Model saved to models/epl_predictive_model.pkl")

# 6. Simulate 5 future seasons
os.makedirs("predictions", exist_ok=True)
def simulate_season(df_stats, model, encoder, season_year, max_matches=20000):
    teams = df_stats['home_team_name'].unique()
    fixtures = []
    for home, away in combinations(teams, 2):
        fixtures.append({'home_team':home,'away_team':away})
        fixtures.append({'home_team':away,'away_team':home})
    fixtures = pd.DataFrame(fixtures)
    if len(fixtures) > max_matches:
        fixtures = fixtures.sample(max_matches, random_state=42).reset_index(drop=True)

    season_stats = pd.DataFrame({
        'Team': teams,'Points': 0,'Played': 0,'Wins': 0,'Draws': 0,'Losses': 0,
        'GoalsFor': 0,'GoalsAgainst': 0,'GoalDiff': 0
    }).set_index('Team')
    match_predictions = []

    for idx, row in enumerate(fixtures.itertuples(),1):
        home, away = row.home_team, row.away_team
        home_last3 = last3_win_rate(home, df_stats.index.max()+1, df_stats)
        away_last3 = last3_win_rate(away, df_stats.index.max()+1, df_stats)
        home_enc = encoder.transform([home])[0]
        away_enc = encoder.transform([away])[0]

        X_pred = pd.DataFrame({'home_enc':[home_enc],'away_enc':[away_enc],
                               'home_last3':[home_last3],'away_last3':[away_last3]})
        pred = model.predict(X_pred)[0]

        if pred=='home team win':
            home_goals = np.random.randint(2,5)
            away_goals = np.random.randint(0,home_goals)
            season_stats.at[home,'Points'] += 3
            season_stats.at[home,'Wins'] += 1
            season_stats.at[away,'Losses'] += 1
        elif pred=='away team win':
            away_goals = np.random.randint(2,5)
            home_goals = np.random.randint(0,away_goals)
            season_stats.at[away,'Points'] += 3
            season_stats.at[away,'Wins'] += 1
            season_stats.at[home,'Losses'] += 1
        else:
            home_goals = away_goals = np.random.randint(0,3)
            season_stats.at[home,'Points'] += 1
            season_stats.at[away,'Points'] += 1
            season_stats.at[home,'Draws'] += 1
            season_stats.at[away,'Draws'] += 1

        # update goals
        season_stats.at[home,'GoalsFor'] += home_goals
        season_stats.at[home,'GoalsAgainst'] += away_goals
        season_stats.at[home,'GoalDiff'] = season_stats.at[home,'GoalsFor'] - season_stats.at[home,'GoalsAgainst']
        season_stats.at[away,'GoalsFor'] += away_goals
        season_stats.at[away,'GoalsAgainst'] += home_goals
        season_stats.at[away,'GoalDiff'] = season_stats.at[away,'GoalsFor'] - season_stats.at[away,'GoalsAgainst']

        match_predictions.append({'season':season_year,'home_team':home,'away_team':away,
                                  'predicted_result':pred,'home_goals':home_goals,'away_goals':away_goals})

        new_row = {'home_team_name':home,'away_team_name':away,'result':pred,'season':season_year}
        df_stats = pd.concat([df_stats,pd.DataFrame([new_row])],ignore_index=True)

        if idx % 1000 == 0:
            print(f"  Simulated {idx}/{len(fixtures)} matches for season {season_year}", end="\r")

    top10 = season_stats.sort_values(by=['Points','GoalDiff','GoalsFor'],ascending=False).head(10)
    top10.reset_index(inplace=True)
    top10['Season'] = season_year
    top10['Rank'] = top10.index + 1
    top10 = top10[['Season','Rank','Team','Points','Wins','Draws','Losses','GoalsFor','GoalsAgainst','GoalDiff']]
    return pd.DataFrame(match_predictions), top10, df_stats

# run simulation
print("⚽ Simulating next 5 seasons...")
future_df = df.copy()
all_matches, all_top10 = [], []
current_season = df['season'].max()+1

for i in range(5):
    matches, top10, future_df = simulate_season(future_df, model, encoder, current_season, max_matches=20000)
    all_matches.append(matches)
    all_top10.append(top10)
    print(f"🏆 Finished season {current_season}")
    current_season += 1

# save CSV
matches_df = pd.concat(all_matches, ignore_index=True)
top10_df = pd.concat(all_top10, ignore_index=True)
matches_df.to_csv("predictions/epl_next5years_matches.csv", index=False)
top10_df.to_csv("predictions/epl_top10_next5years.csv", index=False)

print("\n✅ Sample Predictions:")
print(matches_df.head(10))
print("\n🏅 Top 10 for first predicted season:")
print(all_top10[0])
print("\n💾 Files saved in predictions/")
print("✅ All done!")