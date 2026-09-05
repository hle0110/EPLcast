import datetime
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from team_name_map import canonical_name

DATA_PATH = "data/matches.csv"
FIRST_SEASON = 2021
DIVISIONS = {
    'E0': ('Premier League', 1),
    'E1': ('Championship', 2),
    'E2': ('League One', 3),
    'E3': ('League Two', 4),
}
COLUMNS = [
    'key_id', 'season_id', 'season', 'tier', 'division', 'match_id', 'match_name',
    'match_date', 'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name',
    'score', 'home_team_score', 'away_team_score', 'home_team_score_margin',
    'away_team_score_margin', 'result', 'home_team_win', 'away_team_win', 'draw',
    'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
    'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
    'home_yellow', 'away_yellow', 'home_red', 'away_red',
]
STAT_COLUMNS = {
    'home_shots': 'HS', 'away_shots': 'AS',
    'home_shots_on_target': 'HST', 'away_shots_on_target': 'AST',
    'home_corners': 'HC', 'away_corners': 'AC',
    'home_fouls': 'HF', 'away_fouls': 'AF',
    'home_yellow': 'HY', 'away_yellow': 'AY',
    'home_red': 'HR', 'away_red': 'AR',
}

def season_code(season_year):
    return f"{season_year % 100:02d}{(season_year + 1) % 100:02d}"

def current_season_year(today=None):
    today = today or datetime.date.today()
    return today.year if today.month >= 7 else today.year - 1

def result_of(hg, ag):
    if hg > ag:
        return 'home team win'
    if ag > hg:
        return 'away team win'
    return 'draw'

def fetch_division(div_code, season_year, quiet=False):
    url = f"https://www.football-data.co.uk/mmz4281/{season_code(season_year)}/{div_code}.csv"
    try:
        raw = pd.read_csv(url)
    except Exception as exc:
        if not quiet:
            print(f"  no data for {div_code} {season_year}-{str(season_year+1)[-2:]} ({exc})")
        return None
    raw = raw.dropna(subset=['FTHG', 'FTAG', 'HomeTeam', 'AwayTeam'])
    if len(raw) == 0:
        return None
    raw['parsed_date'] = pd.to_datetime(raw['Date'], dayfirst=True, errors='coerce')
    raw = raw.sort_values('parsed_date', kind='mergesort').reset_index(drop=True)
    division, tier = DIVISIONS[div_code]
    rows = []
    for i, r in raw.iterrows():
        home = canonical_name(r['HomeTeam'])
        away = canonical_name(r['AwayTeam'])
        hg, ag = int(r['FTHG']), int(r['FTAG'])
        res = result_of(hg, ag)
        row = {
            'season_id': f'S-{season_year}-{tier}',
            'season': season_year,
            'tier': tier,
            'division': division,
            'match_id': f'M-{season_year}-{tier}-{i+1:03d}',
            'match_name': f'{home} vs {away}',
            'match_date': r['parsed_date'].date().isoformat() if pd.notna(r['parsed_date']) else '',
            'home_team_name': home,
            'away_team_name': away,
            'score': f'{hg}-{ag}',
            'home_team_score': hg,
            'away_team_score': ag,
            'home_team_score_margin': hg - ag,
            'away_team_score_margin': ag - hg,
            'result': res,
            'home_team_win': 1 if res == 'home team win' else 0,
            'away_team_win': 1 if res == 'away team win' else 0,
            'draw': 1 if res == 'draw' else 0,
        }
        for out_col, src_col in STAT_COLUMNS.items():
            value = r.get(src_col)
            row[out_col] = int(value) if pd.notna(value) else 0
        rows.append(row)
    return pd.DataFrame(rows)

def fetch_season(season_year, quiet=False):
    frames = []
    for div_code in DIVISIONS:
        frame = fetch_division(div_code, season_year, quiet=quiet)
        if frame is not None:
            frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def assign_ids(df):
    df = df.sort_values(['season', 'tier', 'match_id'], kind='mergesort').reset_index(drop=True)
    names = sorted(set(df['home_team_name']) | set(df['away_team_name']))
    team_ids = {name: f'T-{i+1:03d}' for i, name in enumerate(names)}
    df['home_team_id'] = df['home_team_name'].map(team_ids)
    df['away_team_id'] = df['away_team_name'].map(team_ids)
    df['key_id'] = range(1, len(df) + 1)
    return df[COLUMNS]

def rebuild_dataset(path=DATA_PATH, first_season=FIRST_SEASON, last_season=None):
    last_season = last_season or current_season_year()
    frames = []
    for year in range(first_season, last_season + 1):
        season = fetch_season(year, quiet=True)
        if season is None:
            print(f"  {year}-{str(year+1)[-2:]}: no data available yet")
            continue
        print(f"  {year}-{str(year+1)[-2:]}: {len(season)} matches")
        frames.append(season)
    if not frames:
        print("ERROR: no data could be downloaded")
        return False
    full = assign_ids(pd.concat(frames, ignore_index=True))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    full.to_csv(path, index=False)
    print(f"Wrote {len(full):,} matches to {path}")
    return True

def update_dataset(path=DATA_PATH, season_year=None):
    season_year = season_year or current_season_year()
    label = f"{season_year}-{str(season_year + 1)[-2:]}"
    print(f"Checking for new {label} results")

    if not os.path.exists(path):
        print(f"{path} not found, rebuilding the full dataset from source")
        return rebuild_dataset(path)

    existing = pd.read_csv(path, low_memory=False)
    fresh = fetch_season(season_year)
    if fresh is None:
        print(f"No completed {label} matches published yet, nothing to update")
        return False

    before = len(existing[existing['season'] == season_year])
    kept = existing[existing['season'] != season_year]
    combined = assign_ids(pd.concat([kept, fresh], ignore_index=True))
    after = len(fresh)
    combined.to_csv(path, index=False)
    print(f"{label} matches on file: {before} -> {after} ({after - before:+d})")
    return after != before

if __name__ == "__main__":
    if '--rebuild' in sys.argv:
        ok = rebuild_dataset()
        sys.exit(0 if ok else 1)
    changed = update_dataset()
    sys.exit(0 if changed else 1)
