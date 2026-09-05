import pandas as pd
import numpy as np
import re

BASE_ELO = 1500.0
ELO_K = 24.0
HOME_ADVANTAGE = 60.0
ELO_MARGIN_MULTIPLIER = 0.35
FORM_WINDOW = 9
H2H_WINDOW = 3
TOP_DIVISION = 'Premier League'

STAT_PAIRS = [
    ('home_team_score', 'away_team_score', 'gf', 'ga', 1.35),
    ('home_shots_on_target', 'away_shots_on_target', 'sot_for', 'sot_against', 4.5),
]

FEATURE_COLUMNS = [
    'elo_diff',
    'home_form_points', 'away_form_points',
    'home_form_gf', 'home_form_ga',
    'away_form_gf', 'away_form_ga',
    'home_form_sot_for', 'home_form_sot_against',
    'away_form_sot_for', 'away_form_sot_against',
    'h2h_home_rate', 'home_prev_tier', 'away_prev_tier',
]

def extract_round(match_id):
    m = re.search(r'(\d+)$', str(match_id))
    return int(m.group(1)) if m else 0

def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=['home_team_name', 'away_team_name', 'home_team_score', 'away_team_score', 'result']).copy()
    df['home_team_score'] = df['home_team_score'].astype(int)
    df['away_team_score'] = df['away_team_score'].astype(int)
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    df['round_num'] = df['match_id'].apply(extract_round)
    df = df.sort_values(['match_date', 'tier', 'match_id'], kind='mergesort').reset_index(drop=True)
    return df

def compute_elo(df, k=ELO_K, home_adv=HOME_ADVANTAGE, margin_mult=ELO_MARGIN_MULTIPLIER):
    ratings = {}
    home_pre = np.empty(len(df))
    away_pre = np.empty(len(df))
    for i, row in enumerate(df.itertuples()):
        h, a = row.home_team_name, row.away_team_name
        rh = ratings.get(h, BASE_ELO)
        ra = ratings.get(a, BASE_ELO)
        home_pre[i] = rh
        away_pre[i] = ra
        expected_home = 1.0 / (1.0 + 10.0 ** (-((rh + home_adv) - ra) / 400.0))
        hs, as_ = row.home_team_score, row.away_team_score
        actual_home = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        k_eff = k * (1.0 + margin_mult * np.log1p(abs(hs - as_)))
        delta = k_eff * (actual_home - expected_home)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    df = df.copy()
    df['home_elo'] = home_pre
    df['away_elo'] = away_pre
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    return df, ratings

def _long_frame(df):
    rename_home = {'home_team_name': 'team'}
    rename_away = {'away_team_name': 'team'}
    for home_col, away_col, for_name, against_name, _ in STAT_PAIRS:
        rename_home[home_col] = for_name
        rename_home[away_col] = against_name
        rename_away[away_col] = for_name
        rename_away[home_col] = against_name
    source_cols = [p[0] for p in STAT_PAIRS] + [p[1] for p in STAT_PAIRS]
    home_part = df[['season', 'match_date', 'home_team_name'] + source_cols].rename(columns=rename_home)
    home_part['orig_index'] = df.index
    home_part['is_home'] = 1
    away_part = df[['season', 'match_date', 'away_team_name'] + source_cols].rename(columns=rename_away)
    away_part['orig_index'] = df.index
    away_part['is_home'] = 0
    long_df = pd.concat([home_part, away_part], ignore_index=True)
    long_df['points'] = np.where(long_df['gf'] > long_df['ga'], 3.0, np.where(long_df['gf'] == long_df['ga'], 1.0, 0.0))
    return long_df.sort_values(['team', 'match_date'], kind='mergesort').reset_index(drop=True)

def compute_rolling_form(df, window=FORM_WINDOW):
    long_df = _long_frame(df)
    fill_values = {'points': 1.35}
    for home_col, away_col, for_name, against_name, default in STAT_PAIRS:
        fill_values[for_name] = default
        fill_values[against_name] = default
    stat_cols = list(fill_values.keys())
    grouped = long_df.groupby('team', group_keys=False)
    for col in stat_cols:
        long_df[f'form_{col}'] = grouped[col].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long_df = long_df.fillna({f'form_{k}': v for k, v in fill_values.items()})
    form_cols = [f'form_{c}' for c in stat_cols]
    home_rows = long_df[long_df['is_home'] == 1].set_index('orig_index')[form_cols]
    away_rows = long_df[long_df['is_home'] == 0].set_index('orig_index')[form_cols]
    home_rows.columns = [f'home_{c}' for c in form_cols]
    away_rows.columns = [f'away_{c}' for c in form_cols]
    return df.join(home_rows).join(away_rows)

def compute_h2h(df, window=H2H_WINDOW):
    df = df.copy()
    df['pair_key'] = [tuple(sorted((h, a))) for h, a in zip(df['home_team_name'], df['away_team_name'])]
    df['home_win_flag'] = (df['result'] == 'home team win').astype(float)
    values = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby('pair_key'):
        ordered = group.sort_values('match_date', kind='mergesort')
        values.loc[ordered.index] = ordered['home_win_flag'].shift(1).rolling(window, min_periods=1).mean().values
    df['h2h_home_rate'] = values.fillna(0.45)
    return df.drop(columns=['pair_key', 'home_win_flag'])

def season_tier_lookup(df):
    home = df[['season', 'home_team_name', 'tier']].rename(columns={'home_team_name': 'team'})
    away = df[['season', 'away_team_name', 'tier']].rename(columns={'away_team_name': 'team'})
    combined = pd.concat([home, away], ignore_index=True)
    return combined.groupby(['team', 'season'])['tier'].min().reset_index()

def compute_prev_tier(df):
    season_tier = season_tier_lookup(df)
    season_tier['season'] = season_tier['season'] + 1
    season_tier = season_tier.rename(columns={'tier': 'prev_tier'})
    df = df.merge(season_tier, left_on=['home_team_name', 'season'], right_on=['team', 'season'], how='left')
    df = df.drop(columns=['team']).rename(columns={'prev_tier': 'home_prev_tier'})
    df = df.merge(season_tier, left_on=['away_team_name', 'season'], right_on=['team', 'season'], how='left')
    df = df.drop(columns=['team']).rename(columns={'prev_tier': 'away_prev_tier'})
    df['home_prev_tier'] = df['home_prev_tier'].fillna(5.0)
    df['away_prev_tier'] = df['away_prev_tier'].fillna(5.0)
    return df

def engineer_features(df):
    df, elo_ratings = compute_elo(df)
    df = compute_rolling_form(df)
    df = compute_h2h(df)
    df = compute_prev_tier(df)
    return df, elo_ratings

def team_prev_tier_map(df, season):
    season_tier = season_tier_lookup(df)
    previous = season_tier[season_tier['season'] == season - 1]
    return dict(zip(previous['team'], previous['tier'].astype(float)))
