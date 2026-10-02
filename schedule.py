import json
import os
import sys
import time
import urllib.request
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from team_name_map import canonical_name

SCHEDULE_PATH = "data/fixtures.csv"
SCHEDULE_URL = "https://fixturedownload.com/feed/json/epl-{season}"
FETCH_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10
TEAMS_PER_SEASON = 20
SCHEDULE_ALIASES = {
    'Man Utd': 'Manchester United',
    'Spurs': 'Tottenham Hotspur',
}
SCHEDULE_COLUMNS = ['season', 'round', 'kickoff_utc', 'home_team_name', 'away_team_name']


def schedule_name(name):
    name = str(name).strip()
    return SCHEDULE_ALIASES.get(name, canonical_name(name))


def parse_schedule(entries, season):
    rows = []
    for entry in entries:
        rows.append({
            'season': season,
            'round': int(entry['RoundNumber']),
            'kickoff_utc': pd.to_datetime(entry['DateUtc'], utc=True).strftime('%Y-%m-%d %H:%M'),
            'home_team_name': schedule_name(entry['HomeTeam']),
            'away_team_name': schedule_name(entry['AwayTeam']),
        })
    frame = pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
    return frame.sort_values(['kickoff_utc', 'home_team_name'], kind='mergesort').reset_index(drop=True)


def schedule_problems(frame):
    problems = []
    teams = set(frame['home_team_name']) | set(frame['away_team_name'])
    expected = TEAMS_PER_SEASON * (TEAMS_PER_SEASON - 1)
    if len(teams) != TEAMS_PER_SEASON:
        problems.append(f"{len(teams)} clubs instead of {TEAMS_PER_SEASON}")
    if len(frame) != expected:
        problems.append(f"{len(frame)} fixtures instead of {expected}")
    if frame.duplicated(subset=['home_team_name', 'away_team_name']).any():
        problems.append("duplicate fixtures")
    if (frame['home_team_name'] == frame['away_team_name']).any():
        problems.append("a club playing itself")
    return problems


def fetch_schedule(season):
    url = SCHEDULE_URL.format(season=season)
    last_error = None
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'EPLcast'})
            with urllib.request.urlopen(request, timeout=30) as response:
                return parse_schedule(json.loads(response.read().decode('utf-8')), season)
        except Exception as exc:
            last_error = exc
        if attempt < FETCH_ATTEMPTS:
            time.sleep(RETRY_DELAY_SECONDS)
    print(f"  fixture list unreachable after {FETCH_ATTEMPTS} attempts ({last_error})")
    return None


def update_schedule(season, path=SCHEDULE_PATH):
    fresh = fetch_schedule(season)
    if fresh is None:
        print("  keeping the fixture list already on file")
        return False
    problems = schedule_problems(fresh)
    if problems:
        print(f"  fixture list rejected ({', '.join(problems)}), keeping the one on file")
        return False
    if os.path.exists(path):
        existing = pd.read_csv(path)
        others = existing[existing['season'] != season]
        fresh = pd.concat([others, fresh], ignore_index=True)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fresh.to_csv(path, index=False)
    print(f"  fixture list saved to {path}")
    return True


def load_schedule(season, teams, path=SCHEDULE_PATH):
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    frame = frame[frame['season'] == season].reset_index(drop=True)
    if len(frame) == 0 or schedule_problems(frame):
        return None
    if set(frame['home_team_name']) | set(frame['away_team_name']) != set(teams):
        return None
    frame['kickoff_utc'] = pd.to_datetime(frame['kickoff_utc'], utc=True)
    return frame
