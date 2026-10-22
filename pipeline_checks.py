import datetime
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schedule import SCHEDULE_PATH, TEAMS_PER_SEASON, schedule_problems
from update_data import DATA_PATH, current_season_year
from main import infer_season_clubs

TOP_DIVISION = 'Premier League'
FULL_SEASON = TEAMS_PER_SEASON * (TEAMS_PER_SEASON - 1)
MAX_RESULT_AGE_DAYS = 9
SOURCE_LAG_DAYS = 2
MIN_UNREPORTED_FIXTURES = 3


def season_label(season):
    return f"{season}-{str(season + 1)[-2:]}"


def fixture_list_problems(fixtures, matches, season):
    if fixtures is None or len(fixtures) == 0:
        return ["the fixture list is missing or empty"]
    current = fixtures[fixtures['season'] == season]
    if len(current) == 0:
        return [f"the fixture list has no {season_label(season)} fixtures, it is stale"]
    problems = [f"{season_label(season)} fixture list: {problem}" for problem in schedule_problems(current)]
    top = matches[(matches['division'] == TOP_DIVISION) & (matches['season'] == season)]
    played = set(top['home_team_name']) | set(top['away_team_name'])
    if len(played) > TEAMS_PER_SEASON:
        problems.append(f"{len(played)} clubs have {season_label(season)} results instead of at most {TEAMS_PER_SEASON}")
    if len(top) > FULL_SEASON:
        problems.append(f"{len(top)} {season_label(season)} results instead of at most {FULL_SEASON}")
    listed = set(current['home_team_name']) | set(current['away_team_name'])
    unknown = sorted(played - listed)
    if unknown:
        problems.append(f"clubs with results but missing from the fixture list: {', '.join(unknown)}")
    expected = expected_clubs(matches, season)
    if expected is not None and expected != listed:
        missing = sorted(expected - listed)
        extra = sorted(listed - expected)
        problems.append(f"{season_label(season)} fixture list clubs do not match the clubs in the results data "
                        f"(expected but missing: {', '.join(missing) or 'none'}; "
                        f"listed but not expected: {', '.join(extra) or 'none'}); add the spelling to SCHEDULE_ALIASES")
    return problems


def expected_clubs(matches, season):
    top = matches[matches['division'] == TOP_DIVISION]
    appeared, inferred, _ = infer_season_clubs(matches, top, season)
    for clubs in (appeared, inferred):
        if len(clubs) == TEAMS_PER_SEASON:
            return clubs
    return None


def stale_results_problem(matches, fixtures, now):
    top = matches[matches['division'] == TOP_DIVISION]
    if len(top) == 0:
        return "no Premier League results on file"
    newest = pd.to_datetime(top['match_date']).max().normalize()
    today = now.normalize()
    age = (today - newest).days
    if age <= MAX_RESULT_AGE_DAYS or fixtures is None or len(fixtures) == 0:
        return None
    kickoffs = pd.to_datetime(fixtures['kickoff_utc'], utc=True).dt.tz_localize(None)
    due = kickoffs[(kickoffs.dt.normalize() > newest) & (kickoffs <= now - datetime.timedelta(days=SOURCE_LAG_DAYS))]
    if len(due) < MIN_UNREPORTED_FIXTURES:
        return None
    return (f"the newest Premier League result is {newest.date()}, {age} days old, "
            f"but {len(due)} fixtures kicked off after it; the results source looks stale")


def run_checks(data_path=DATA_PATH, schedule_path=SCHEDULE_PATH, now=None):
    now = now or pd.Timestamp.now(tz='UTC').tz_localize(None)
    season = current_season_year(now.date())
    matches = pd.read_csv(data_path, low_memory=False)
    fixtures = pd.read_csv(schedule_path) if os.path.exists(schedule_path) else None
    problems = fixture_list_problems(fixtures, matches, season)
    stale = stale_results_problem(matches, fixtures, now)
    if stale:
        problems.append(stale)
    return problems


def main():
    problems = run_checks()
    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        print(f"Data health check failed with {len(problems)} problem(s)")
        return 1
    print("Data health check passed: fixture list complete, results up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
