import os
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dashboard
import features
import simulate
import update_data

DATA_PATH = os.path.join(ROOT, 'data', 'matches.csv')
PROJECTION_PATH = os.path.join(ROOT, 'predictions', 'epl_season_projection.csv')
TEAMS_PER_SEASON = 20


@pytest.fixture(scope='module')
def matches():
    return pd.read_csv(DATA_PATH, low_memory=False)


@pytest.fixture(scope='module')
def projection():
    if not os.path.exists(PROJECTION_PATH):
        pytest.skip('projection not generated yet')
    return pd.read_csv(PROJECTION_PATH)


def test_no_duplicate_fixtures(matches):
    duplicates = matches.duplicated(subset=['season', 'tier', 'home_team_name', 'away_team_name'])
    assert duplicates.sum() == 0


def test_no_missing_values(matches):
    assert matches.isna().sum().sum() == 0


def test_score_column_matches_goals(matches):
    rebuilt = matches['home_team_score'].astype(str) + '-' + matches['away_team_score'].astype(str)
    assert (matches['score'] == rebuilt).all()


def test_result_labels_are_consistent(matches):
    home_wins = matches[matches['result'] == 'home team win']
    away_wins = matches[matches['result'] == 'away team win']
    draws = matches[matches['result'] == 'draw']
    assert (home_wins['home_team_score'] > home_wins['away_team_score']).all()
    assert (away_wins['away_team_score'] > away_wins['home_team_score']).all()
    assert (draws['home_team_score'] == draws['away_team_score']).all()


def test_key_ids_are_sequential(matches):
    assert list(matches['key_id']) == list(range(1, len(matches) + 1))


def test_completed_seasons_have_full_fixture_lists(matches):
    top = matches[matches['division'] == 'Premier League']
    current = top['season'].max()
    for season, group in top[top['season'] < current].groupby('season'):
        assert len(group) == 380, f'season {season} has {len(group)} matches'


def test_result_of_helper():
    assert update_data.result_of(2, 1) == 'home team win'
    assert update_data.result_of(0, 3) == 'away team win'
    assert update_data.result_of(1, 1) == 'draw'


def test_season_code_formatting():
    assert update_data.season_code(2026) == '2627'
    assert update_data.season_code(1999) == '9900'


def test_current_season_year_boundaries():
    import datetime
    assert update_data.current_season_year(datetime.date(2026, 8, 20)) == 2026
    assert update_data.current_season_year(datetime.date(2027, 5, 20)) == 2026
    assert update_data.current_season_year(datetime.date(2027, 7, 1)) == 2027


def test_exit_code_constants_are_distinct():
    codes = {update_data.STATUS_UPDATED, update_data.STATUS_UNCHANGED, update_data.STATUS_SOURCE_ERROR}
    assert codes == {0, 1, 2}


def test_rolling_form_does_not_leak_current_match():
    rows = []
    for index, goals in enumerate([1, 2, 3, 4], start=1):
        rows.append({
            'season': 2026, 'tier': 1, 'division': 'Premier League',
            'match_id': f'M-2026-1-{index:03d}', 'match_date': f'2026-08-0{index}',
            'home_team_name': 'Alpha', 'away_team_name': f'Team{index}',
            'home_team_score': goals, 'away_team_score': 0,
            'home_shots_on_target': goals + 2, 'away_shots_on_target': 1,
            'result': 'home team win',
        })
    frame = pd.DataFrame(rows)
    frame['match_date'] = pd.to_datetime(frame['match_date'])
    enriched = features.compute_rolling_form(frame, window=5)
    assert enriched.iloc[0]['home_form_gf'] == pytest.approx(1.35)
    assert enriched.iloc[1]['home_form_gf'] == pytest.approx(1.0)
    assert enriched.iloc[3]['home_form_gf'] == pytest.approx(2.0)


def test_remaining_fixtures_excludes_played_games():
    teams = ['A', 'B', 'C']
    played = pd.DataFrame([{'home_team_name': 'A', 'away_team_name': 'B'}])
    remaining = simulate.remaining_fixtures(played, teams)
    assert ('A', 'B') not in remaining
    assert ('B', 'A') in remaining
    assert len(remaining) == len(teams) * (len(teams) - 1) - 1


def test_round_robin_is_a_complete_double_schedule():
    teams = [f'T{i}' for i in range(20)]
    rounds = simulate.round_robin_schedule(teams, np.random.default_rng(0))
    fixtures = [pair for rnd in rounds for pair in rnd]
    assert len(fixtures) == 380
    assert len(set(fixtures)) == 380
    for rnd in rounds:
        appearing = [team for pair in rnd for team in pair]
        assert len(appearing) == len(set(appearing))


def test_table_from_results_counts_points_correctly():
    teams = ['A', 'B']
    played = pd.DataFrame([
        {'home_team_name': 'A', 'away_team_name': 'B', 'home_team_score': 2, 'away_team_score': 0},
        {'home_team_name': 'B', 'away_team_name': 'A', 'home_team_score': 1, 'away_team_score': 1},
    ])
    table = simulate.table_from_results(played, teams)
    assert table['A']['Points'] == 4
    assert table['B']['Points'] == 1
    assert table['A']['GoalsFor'] == 3
    assert table['A']['Played'] == 2


def test_projection_has_one_row_per_team(projection):
    for season, group in projection.groupby('Season'):
        assert len(group) == TEAMS_PER_SEASON, f'season {season} has {len(group)} teams'


def test_projection_probabilities_sum_correctly(projection):
    for season, group in projection.groupby('Season'):
        assert group['TitleProb'].sum() == pytest.approx(1.0, abs=0.02)
        assert group['Top4Prob'].sum() == pytest.approx(4.0, abs=0.05)
        assert group['RelegationProb'].sum() == pytest.approx(3.0, abs=0.05)


def test_projection_covers_a_full_season(projection):
    played = projection['Wins'] + projection['Draws'] + projection['Losses']
    assert played.round(1).eq(38.0).all()


def test_projected_goal_difference_nets_out(projection):
    for season, group in projection.groupby('Season'):
        assert abs(group['GoalDiff'].sum()) < 1.0


def test_dashboard_renders_every_team(projection):
    season = projection['Season'].min()
    page = dashboard.build_page(projection, season, {
        'status_line': 'test render',
        'played': 40,
        'total': 380,
        'simulations': 40,
        'accuracy': '53.3%',
        'last_match_date': pd.Timestamp('2026-09-14'),
        'standings': None,
        'form': None,
        'history': None,
    })
    assert page.count('<tr class=') == TEAMS_PER_SEASON
    assert '{' not in page.split('<style>')[0]
    assert page.rstrip().endswith('</html>')
