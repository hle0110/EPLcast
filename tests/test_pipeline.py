import datetime
import os
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dashboard
import features
import insights
import main
import models
import pipeline_checks
import schedule
import simulate
import site_builder
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
    required = [c for c in matches.columns if c not in update_data.OPTIONAL_COLUMNS]
    assert matches[required].isna().sum().sum() == 0


def test_closing_odds_are_stored_and_sensible(matches):
    priced = matches[matches['season'] >= 2019]
    odds = priced[['odds_home', 'odds_draw', 'odds_away']]
    assert odds.notna().all().all()
    assert (odds > 1.0).all().all()
    overround = (1.0 / odds).sum(axis=1)
    assert overround.between(0.9, 1.25).all()
    assert overround[priced['division'] == 'Premier League'].between(1.0, 1.15).all()


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


def test_history_only_records_when_new_matches_are_played(tmp_path, monkeypatch):
    path = tmp_path / 'history.csv'
    projection = pd.DataFrame([
        {'Season': 2026, 'Team': 'Alpha', 'Rank': 1, 'Points': 80.0,
         'TitleProb': 0.7, 'Top4Prob': 1.0, 'RelegationProb': 0.0},
        {'Season': 2026, 'Team': 'Beta', 'Rank': 2, 'Points': 70.0,
         'TitleProb': 0.3, 'Top4Prob': 1.0, 'RelegationProb': 0.0},
    ])

    class FixedDate(datetime.date):
        current = datetime.date(2026, 9, 15)

        @classmethod
        def today(cls):
            return cls.current

    monkeypatch.setattr(main.datetime, 'date', FixedDate)

    main.record_history(projection, 2026, 40, path=str(path))
    after_first = path.read_text()

    FixedDate.current = datetime.date(2026, 9, 18)
    main.record_history(projection, 2026, 40, path=str(path))
    assert path.read_text() == after_first

    main.record_history(projection, 2026, 50, path=str(path))
    stored = pd.read_csv(path)
    assert sorted(stored['matches_played'].unique()) == [40, 50]
    assert sorted(stored['recorded_on'].unique()) == ['2026-09-15', '2026-09-18']


def test_history_replaces_same_day_entry(tmp_path):
    path = tmp_path / 'history.csv'
    projection = pd.DataFrame([
        {'Season': 2026, 'Team': 'Alpha', 'Rank': 1, 'Points': 80.0,
         'TitleProb': 0.7, 'Top4Prob': 1.0, 'RelegationProb': 0.0},
    ])
    main.record_history(projection, 2026, 40, path=str(path))
    main.record_history(projection, 2026, 50, path=str(path))
    stored = pd.read_csv(path)
    assert len(stored) == 1
    assert stored['matches_played'].iloc[0] == 50

def test_history_records_when_odds_move_without_new_matches(tmp_path, monkeypatch):
    path = tmp_path / 'history.csv'

    def make(title_alpha):
        return pd.DataFrame([
            {'Season': 2026, 'Team': 'Alpha', 'Rank': 1, 'Points': 80.0,
             'TitleProb': title_alpha, 'Top4Prob': 1.0, 'RelegationProb': 0.0},
            {'Season': 2026, 'Team': 'Beta', 'Rank': 2, 'Points': 70.0,
             'TitleProb': 1 - title_alpha, 'Top4Prob': 1.0, 'RelegationProb': 0.0},
        ])

    class FixedDate(datetime.date):
        current = datetime.date(2026, 9, 15)

        @classmethod
        def today(cls):
            return cls.current

    monkeypatch.setattr(main.datetime, 'date', FixedDate)

    main.record_history(make(0.6), 2026, 40, path=str(path))
    FixedDate.current = datetime.date(2026, 9, 18)
    main.record_history(make(0.45), 2026, 40, path=str(path))
    stored = pd.read_csv(path)
    assert sorted(stored['recorded_on'].unique()) == ['2026-09-15', '2026-09-18']
    assert sorted(stored['matches_played'].unique()) == [40]


@pytest.fixture(scope='module')
def engineered():
    frame = features.load_data(DATA_PATH)
    enriched, _ = features.engineer_features(frame)
    return enriched


def test_simulation_features_match_training_features(engineered):
    top = engineered[(engineered['division'] == 'Premier League') & (engineered['season'] < engineered['season'].max())]
    season = int(top['season'].max())
    season_top = top[top['season'] == season]
    match_day = season_top['match_date'].sort_values().iloc[len(season_top) // 2]
    fixtures = season_top[season_top['match_date'] == match_day]
    history = engineered[engineered['match_date'] < match_day]
    _, ratings = features.compute_elo(history)
    teams = sorted(set(fixtures['home_team_name']) | set(fixtures['away_team_name']))
    index = {team: i for i, team in enumerate(teams)}
    state = simulate.init_state(history, ratings, teams, 1)
    previous = features.team_prev_tier_map(history, season)
    prev_tier = np.array([previous.get(team, 1.0) for team in teams])
    home = np.array([index[t] for t in fixtures['home_team_name']])
    away = np.array([index[t] for t in fixtures['away_team_name']])
    simulated = simulate.fixture_features(state, np.zeros(len(home), dtype=int), home, away, prev_tier)
    expected = fixtures[features.FEATURE_COLUMNS].reset_index(drop=True)
    assert list(simulated.columns) == features.FEATURE_COLUMNS
    np.testing.assert_allclose(simulated.values, expected.values, rtol=1e-9, atol=1e-9)


def test_promoted_flag_only_marks_top_flight_newcomers():
    tier = pd.Series([1, 1, 1, 2, 2])
    prev = pd.Series([2.0, 1.0, 3.0, 3.0, 2.0])
    assert features.promoted_to_top(tier, prev).tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_warm_up_seasons_are_not_training_rows(engineered):
    train = main.training_rows(engineered)
    assert train['season'].min() == main.FIRST_TRAINING_SEASON
    assert engineered['season'].min() < main.FIRST_TRAINING_SEASON


def test_evaluation_folds_do_not_overlap(engineered):
    top = engineered[engineered['division'] == 'Premier League']
    folds = main.rolling_origin_folds(top)
    seen = set()
    for _, cutoff, test_index in folds:
        assert seen.isdisjoint(test_index)
        seen.update(test_index)
        assert (top.loc[test_index, 'match_date'] >= cutoff).all()
        assert top.loc[test_index, 'season'].min() >= main.FIRST_TRAINING_SEASON + main.MIN_TRAINING_SEASONS


def test_sampled_scorelines_respect_the_drawn_outcome():
    rng = np.random.default_rng(0)
    outcome = np.tile([0, 1, 2], 4000)
    lam_home = np.full(len(outcome), 4.0)
    lam_away = np.full(len(outcome), 0.2)
    hg, ag = simulate.sample_scorelines(rng, outcome, lam_home, lam_away)
    drawn = np.where(hg > ag, 2, np.where(hg < ag, 0, 1))
    assert (drawn == outcome).all()
    assert (hg >= 0).all() and (ag >= 0).all()


class _FixedModel:
    classes_ = np.array(['away team win', 'draw', 'home team win'])

    def predict_proba(self, frame):
        return np.tile([0.3, 0.25, 0.45], (len(frame), 1))


def test_simulated_seasons_are_complete_and_consistent(engineered):
    season = int(engineered['season'].max())
    top = engineered[(engineered['division'] == 'Premier League') & (engineered['season'] == season - 1)]
    teams = sorted(set(top['home_team_name']))
    history = engineered[engineered['season'] < season]
    _, ratings = features.compute_elo(history)
    results = simulate.run_simulations(_FixedModel(), history, season - 1, teams, ratings,
                                       n_future_seasons=1, n_simulations=50, project_current=True)
    for table in results['tables'].values():
        played = table['Wins'] + table['Draws'] + table['Losses']
        assert (played == 38).all()
        assert (table['Points'] == 3 * table['Wins'] + table['Draws']).all()
        assert (table['GoalsFor'].sum(axis=1) == table['GoalsAgainst'].sum(axis=1)).all()
        assert (table['Wins'].sum(axis=1) == table['Losses'].sum(axis=1)).all()
    summary = simulate.summarize_simulations(results, teams)
    for _, group in summary.groupby('Season'):
        assert group['TitleProb'].sum() == pytest.approx(1.0)
        assert group['RelegationProb'].sum() == pytest.approx(3.0)


def _schedule_entries(teams, start='2026-08-01'):
    rounds = simulate.round_robin_schedule(teams, np.random.default_rng(3))
    entries = []
    day = pd.Timestamp(start, tz='UTC')
    for number, pairs in enumerate(rounds, start=1):
        for home, away in pairs:
            entries.append({'RoundNumber': number, 'DateUtc': (day + datetime.timedelta(days=7 * number)).strftime('%Y-%m-%d %H:%M:%SZ'),
                            'HomeTeam': home, 'AwayTeam': away})
    return entries


def test_schedule_aliases_reach_canonical_names():
    assert schedule.schedule_name('Man Utd') == 'Manchester United'
    assert schedule.schedule_name('Spurs') == 'Tottenham Hotspur'
    assert schedule.schedule_name("Nott'm Forest") == 'Nottingham Forest'
    assert schedule.schedule_name('Brighton') == 'Brighton & Hove Albion'


def test_schedule_alias_maps_sheffield_utd():
    assert schedule.schedule_name('Sheffield Utd') == 'Sheffield United'


def test_schedule_validation_accepts_a_full_season_and_rejects_gaps():
    teams = [f'Club{i}' for i in range(20)]
    frame = schedule.parse_schedule(_schedule_entries(teams), 2026)
    assert schedule.schedule_problems(frame) == []
    assert schedule.schedule_problems(frame.iloc[1:]) != []
    duplicated = pd.concat([frame.iloc[1:], frame.iloc[[1]]], ignore_index=True)
    assert 'duplicate fixtures' in schedule.schedule_problems(duplicated)


def test_load_schedule_requires_the_current_clubs(tmp_path):
    teams = [f'Club{i}' for i in range(20)]
    path = tmp_path / 'fixtures.csv'
    schedule.parse_schedule(_schedule_entries(teams), 2026).to_csv(path, index=False)
    assert schedule.load_schedule(2026, teams, str(path)) is not None
    assert schedule.load_schedule(2026, teams[:-1] + ['Other'], str(path)) is None
    assert schedule.load_schedule(2025, teams, str(path)) is None


def test_rounds_in_order_keep_order_and_never_double_book():
    fixtures = [('A', 'B'), ('C', 'D'), ('A', 'C'), ('B', 'D'), ('E', 'F'), ('A', 'D')]
    rounds = simulate.rounds_in_order(fixtures)
    assert [pair for rnd in rounds for pair in rnd] == fixtures
    for rnd in rounds:
        clubs = [club for pair in rnd for club in pair]
        assert len(clubs) == len(set(clubs))


def test_ordered_fixtures_skip_played_games():
    teams = ['A', 'B', 'C']
    played = pd.DataFrame([{'home_team_name': 'A', 'away_team_name': 'B'}])
    sched = pd.DataFrame({'kickoff_utc': pd.to_datetime(['2026-08-01', '2026-08-08', '2026-08-15', '2026-08-22', '2026-08-29', '2026-09-05'], utc=True),
                          'home_team_name': ['A', 'C', 'B', 'C', 'A', 'B'], 'away_team_name': ['B', 'A', 'C', 'B', 'C', 'A']})
    ordered, unscheduled = simulate.ordered_remaining_fixtures(played, teams, sched)
    assert ordered == [('C', 'A'), ('B', 'C'), ('C', 'B'), ('A', 'C'), ('B', 'A')]
    assert unscheduled == []


def test_simulated_outcomes_rebuild_the_tables_and_stakes_are_consistent(engineered, monkeypatch):
    monkeypatch.setattr(insights, 'MIN_STAKE_SAMPLES', 1)
    season = int(engineered['season'].max())
    top = engineered[(engineered['division'] == 'Premier League') & (engineered['season'] == season - 1)]
    teams = sorted(set(top['home_team_name']))
    history = engineered[engineered['season'] < season]
    _, ratings = features.compute_elo(history)
    partial = top.sort_values('match_date').head(100)
    frame = history[~history.index.isin(top.index.difference(partial.index))]
    results = simulate.run_simulations(_FixedModel(), frame, season - 1, teams, ratings,
                                       n_future_seasons=0, n_simulations=400, project_current=True)
    table = results['tables'][season - 1]
    assert results['outcomes'].shape == (400, 280)
    assert (results['outcomes'] >= 0).all()
    index = {team: i for i, team in enumerate(teams)}
    banked = simulate.table_from_results(partial, teams)
    points = np.tile([banked[t]['Points'] for t in teams], (400, 1))
    for j, (home, away) in enumerate(results['fixtures']):
        outcome = results['outcomes'][:, j]
        points[:, index[home]] += np.where(outcome == 2, 3, np.where(outcome == 1, 1, 0))
        points[:, index[away]] += np.where(outcome == 0, 3, np.where(outcome == 1, 1, 0))
    assert (points == table['Points']).all()

    fixture = results['fixtures'][0]
    stake = insights.match_stakes(results, season - 1, [fixture])[0]
    outcome = results['outcomes'][:, 0]
    checked = 0
    for team, entry in stake.items():
        if entry is None or None in (entry['home_win'], entry['draw'], entry['away_win']):
            continue
        checked += 1
        shares = [(outcome == code).mean() for code in (2, 1, 0)]
        parts = [entry['home_win'], entry['draw'], entry['away_win']]
        assert sum(s * p for s, p in zip(shares, parts)) == pytest.approx(entry['overall'], abs=1e-9)
    assert checked == 2


def test_prediction_log_keeps_first_date_until_the_forecast_changes(tmp_path):
    path = str(tmp_path / 'log.csv')
    row = {'predicted_on': '2026-10-01', 'season': 2026, 'kickoff_utc': '2026-10-10 11:30',
           'home_team_name': 'A', 'away_team_name': 'B', 'p_home': 0.5, 'p_draw': 0.3, 'p_away': 0.2}
    insights.update_prediction_log([row], '2026-10-01', path)
    insights.update_prediction_log([dict(row, predicted_on='2026-10-02')], '2026-10-02', path)
    assert pd.read_csv(path)['predicted_on'].tolist() == ['2026-10-01']
    insights.update_prediction_log([dict(row, predicted_on='2026-10-03', p_home=0.55, p_draw=0.25)], '2026-10-03', path)
    stored = pd.read_csv(path)
    assert stored['predicted_on'].tolist() == ['2026-10-03']
    assert stored['p_home'].tolist() == [0.55]
    other = dict(row, home_team_name='C', away_team_name='D', kickoff_utc='2026-10-17 14:00')
    insights.update_prediction_log([other], '2026-10-04', path)
    assert len(pd.read_csv(path)) == 2


def test_prediction_log_scoring_matches_hand_calculation():
    log = pd.DataFrame([{'predicted_on': '2026-10-01', 'season': 2026, 'home_team_name': 'A', 'away_team_name': 'B',
                         'p_home': 0.6, 'p_draw': 0.25, 'p_away': 0.15},
                        {'predicted_on': '2026-10-01', 'season': 2026, 'home_team_name': 'C', 'away_team_name': 'D',
                         'p_home': 0.3, 'p_draw': 0.3, 'p_away': 0.4}])
    played = pd.DataFrame([{'division': 'Premier League', 'season': 2026, 'home_team_name': 'A', 'away_team_name': 'B',
                            'result': 'home team win', 'odds_home': 1.5, 'odds_draw': 4.0, 'odds_away': 6.0},
                           {'division': 'Premier League', 'season': 2026, 'home_team_name': 'C', 'away_team_name': 'D',
                            'result': 'draw', 'odds_home': 2.5, 'odds_draw': 3.2, 'odds_away': 2.9}])
    scored = insights.score_prediction_log(log, played)
    assert scored['matches'] == 2
    assert scored['accuracy'] == pytest.approx(0.5)
    assert scored['log_loss'] == pytest.approx(-(np.log(0.6) + np.log(0.3)) / 2)
    implied = 1 / np.array([1.5, 4.0, 6.0])
    assert insights.bookmaker_probabilities(played)[0] == pytest.approx(implied[::-1] / implied.sum())


def test_upcoming_fixtures_take_the_next_round_only():
    sched = pd.DataFrame({'round': [1, 1, 2, 2, 3], 'home_team_name': ['A', 'C', 'A', 'B', 'A'],
                          'away_team_name': ['B', 'D', 'C', 'D', 'D'],
                          'kickoff_utc': pd.to_datetime(['2026-08-01', '2026-08-02', '2026-08-08', '2026-08-09', '2026-08-15'], utc=True)})
    played = pd.DataFrame([{'home_team_name': 'A', 'away_team_name': 'B'}])
    upcoming = insights.upcoming_fixtures(sched, played, pd.Timestamp('2026-08-03', tz='UTC'))
    assert [(h, a) for _, h, a, _ in upcoming] == [('A', 'C'), ('B', 'D')]
    assert insights.upcoming_fixtures(None, played, pd.Timestamp('2026-08-03', tz='UTC')) == []


def test_site_writes_every_page_and_valid_json(projection, tmp_path):
    import json
    season = int(projection['Season'].min())
    teams = projection[projection['Season'] == season]['Team'].tolist()
    meta = {'status_line': 'test', 'played': 50, 'total': 380, 'simulations': 100, 'accuracy': '53.1%',
            'last_match_date': pd.Timestamp('2026-09-20'), 'standings': None, 'form': None, 'history': None,
            'positions': None, 'upcoming': [], 'clubs': {}, 'metrics': None, 'live_score': None, 'generated': '2026-10-01'}
    site_builder.write_site(projection, season, meta, str(tmp_path))
    for page in ['index.html', 'matches.html', 'accuracy.html', 'methodology.html', 'clubs/index.html']:
        text = (tmp_path / page).read_text()
        assert text.rstrip().endswith('</html>')
        assert '{' not in text.split('<style>')[0]
    for team in teams:
        assert (tmp_path / 'clubs' / f'{dashboard.club_slug(team)}.html').exists()
    for name in ['projection.json', 'fixtures.json', 'accuracy.json']:
        json.loads((tmp_path / 'api' / name).read_text())
    assert (tmp_path / '404.html').read_text().rstrip().endswith('</html>')
    import xml.dom.minidom
    sitemap = xml.dom.minidom.parse(str(tmp_path / 'sitemap.xml'))
    locations = [node.firstChild.data for node in sitemap.getElementsByTagName('loc')]
    assert len(locations) == len(teams) + 5
    for location in locations:
        relative = location[len(dashboard.SITE_URL):] or 'index.html'
        assert (tmp_path / relative).exists()
    assert len(json.loads((tmp_path / 'api' / 'projection.json').read_text())['teams']) == len(teams)
    assert (tmp_path / '404.html').read_text().rstrip().endswith('</html>')
    import xml.dom.minidom
    sitemap = xml.dom.minidom.parse(str(tmp_path / 'sitemap.xml'))
    locations = [node.firstChild.data for node in sitemap.getElementsByTagName('loc')]
    assert len(locations) == 4 + 1 + len(teams)
    assert all(loc.startswith(dashboard.SITE_URL) for loc in locations)


def test_club_slugs_are_unique_and_url_safe(projection):
    teams = projection['Team'].unique()
    slugs = [dashboard.club_slug(t) for t in teams]
    assert len(set(slugs)) == len(slugs)
    assert all(s and all(ch.isalnum() or ch == '-' for ch in s) for s in slugs)


def _season_fixtures(teams, season=2026, start='2026-08-01'):
    return schedule.parse_schedule(_schedule_entries(teams, start), season)


def _top_results(fixtures, played):
    rows = fixtures.iloc[:played].copy()
    rows['division'] = 'Premier League'
    rows['tier'] = 1
    rows['match_date'] = pd.to_datetime(rows['kickoff_utc']).dt.strftime('%Y-%m-%d')
    return rows[['season', 'tier', 'division', 'match_date', 'home_team_name', 'away_team_name']]


def test_health_check_accepts_a_complete_current_fixture_list():
    teams = [f'Club{i}' for i in range(20)]
    fixtures = _season_fixtures(teams)
    assert pipeline_checks.fixture_list_problems(fixtures, _top_results(fixtures, 30), 2026) == []


def test_health_check_fails_on_a_missing_or_empty_fixture_list():
    teams = [f'Club{i}' for i in range(20)]
    results = _top_results(_season_fixtures(teams), 30)
    assert pipeline_checks.fixture_list_problems(None, results, 2026) != []
    assert pipeline_checks.fixture_list_problems(_season_fixtures(teams).iloc[0:0], results, 2026) != []


def test_health_check_fails_on_a_fixture_list_for_another_season():
    teams = [f'Club{i}' for i in range(20)]
    fixtures = _season_fixtures(teams, season=2025)
    problems = pipeline_checks.fixture_list_problems(fixtures, _top_results(fixtures, 0), 2026)
    assert any('stale' in problem for problem in problems)


def test_health_check_fails_when_clubs_or_fixtures_are_not_20_and_380():
    teams = [f'Club{i}' for i in range(20)]
    fixtures = _season_fixtures(teams)
    short = pipeline_checks.fixture_list_problems(fixtures.iloc[1:], _top_results(fixtures, 0), 2026)
    assert any('379 fixtures' in problem for problem in short)
    nineteen = fixtures[(fixtures['home_team_name'] != 'Club19') & (fixtures['away_team_name'] != 'Club19')]
    problems = pipeline_checks.fixture_list_problems(nineteen, _top_results(fixtures, 0), 2026)
    assert any('19 clubs' in problem for problem in problems)


def test_health_check_fails_when_a_club_with_results_is_not_in_the_fixture_list():
    teams = [f'Club{i}' for i in range(20)]
    fixtures = _season_fixtures(teams)
    results = _top_results(fixtures, 10)
    results.iloc[0, results.columns.get_loc('home_team_name')] = 'Stranger'
    problems = pipeline_checks.fixture_list_problems(fixtures, results, 2026)
    assert any('Stranger' in problem for problem in problems)


def test_stale_guard_fails_when_played_fixtures_have_no_results():
    teams = [f'Club{i}' for i in range(20)]
    fixtures = _season_fixtures(teams)
    results = _top_results(fixtures, 50)
    newest = pd.to_datetime(results['match_date']).max()
    now = newest + pd.Timedelta(days=15, hours=17)
    assert pipeline_checks.stale_results_problem(results, fixtures, now) is not None


def test_stale_guard_passes_with_recent_results():
    teams = [f'Club{i}' for i in range(20)]
    fixtures = _season_fixtures(teams)
    results = _top_results(fixtures, 50)
    now = pd.to_datetime(results['match_date']).max() + pd.Timedelta(days=3, hours=17)
    assert pipeline_checks.stale_results_problem(results, fixtures, now) is None


def test_stale_guard_passes_through_a_break_with_no_fixtures():
    teams = [f'Club{i}' for i in range(20)]
    fixtures = _season_fixtures(teams)
    results = _top_results(fixtures, 50)
    newest = pd.to_datetime(results['match_date']).max()
    later = pd.to_datetime(fixtures['kickoff_utc']) > newest + pd.Timedelta(days=1)
    gap = pd.to_datetime(fixtures['kickoff_utc']) < newest + pd.Timedelta(days=16)
    fixtures = fixtures[~(later & gap)]
    now = newest + pd.Timedelta(days=15, hours=17)
    assert (now - newest).days > pipeline_checks.MAX_RESULT_AGE_DAYS
    assert pipeline_checks.stale_results_problem(results, fixtures, now) is None


def test_stale_guard_passes_after_the_season_ends():
    teams = [f'Club{i}' for i in range(20)]
    fixtures = _season_fixtures(teams)
    results = _top_results(fixtures, 380)
    now = pd.to_datetime(results['match_date']).max() + pd.Timedelta(days=40)
    assert pipeline_checks.stale_results_problem(results, fixtures, now) is None


def test_run_checks_reports_problems_from_files(tmp_path):
    teams = [f'Club{i}' for i in range(20)]
    fixtures = _season_fixtures(teams)
    results = _top_results(fixtures, 50)
    data_path = tmp_path / 'matches.csv'
    schedule_path = tmp_path / 'fixtures.csv'
    results.to_csv(data_path, index=False)
    fixtures.to_csv(schedule_path, index=False)
    newest = pd.to_datetime(results['match_date']).max()
    assert pipeline_checks.run_checks(str(data_path), str(schedule_path), newest + pd.Timedelta(days=2)) == []
    assert pipeline_checks.run_checks(str(data_path), str(schedule_path), newest + pd.Timedelta(days=20)) != []
    assert pipeline_checks.run_checks(str(data_path), str(tmp_path / 'none.csv'), newest + pd.Timedelta(days=2)) != []


def _division_rows(season, division, count):
    return pd.DataFrame({'season': [season] * count, 'division': [division] * count})


def test_update_fails_when_the_source_returns_fewer_matches_than_on_file(tmp_path, monkeypatch):
    path = tmp_path / 'matches.csv'
    existing = pd.concat([_division_rows(2026, 'Premier League', 70), _division_rows(2026, 'Championship', 100)])
    existing.to_csv(path, index=False)
    before = path.read_bytes()
    fresh = _division_rows(2026, 'Championship', 110)
    monkeypatch.setattr(update_data, 'fetch_season', lambda season, quiet=False: (fresh, []))
    assert update_data.update_dataset(str(path), 2026) == update_data.STATUS_SOURCE_ERROR
    assert path.read_bytes() == before


def test_update_fails_when_a_division_is_unreachable(tmp_path, monkeypatch):
    path = tmp_path / 'matches.csv'
    _division_rows(2026, 'Premier League', 70).to_csv(path, index=False)
    monkeypatch.setattr(update_data, 'fetch_season', lambda season, quiet=False: (None, ['E0']))
    assert update_data.update_dataset(str(path), 2026) == update_data.STATUS_SOURCE_ERROR


def test_divisions_losing_matches_ignores_growth_and_other_seasons():
    existing = pd.concat([_division_rows(2026, 'Premier League', 70), _division_rows(2025, 'Championship', 552)])
    fresh = _division_rows(2026, 'Premier League', 80)
    assert update_data.divisions_losing_matches(existing, fresh, 2026) == []


def test_fixture_download_failure_keeps_saved_copy_with_a_warning(tmp_path, monkeypatch, capsys):
    teams = [f'Club{i}' for i in range(20)]
    path = tmp_path / 'fixtures.csv'
    _season_fixtures(teams).to_csv(path, index=False)
    before = path.read_bytes()
    monkeypatch.setattr(schedule, 'fetch_schedule', lambda season: None)
    assert schedule.update_schedule(2026, str(path)) is False
    assert '::warning::' in capsys.readouterr().out
    assert path.read_bytes() == before


def _division_season(season, tier, division, clubs, date):
    pairs = list(zip(clubs[0::2], clubs[1::2]))
    return pd.DataFrame({'season': season, 'tier': tier, 'division': division, 'match_date': date,
                         'home_team_name': [h for h, _ in pairs], 'away_team_name': [a for _, a in pairs]})


def _rollover_history(started_below=True):
    top = [f'Club{i}' for i in range(20)]
    second = [f'Champ{i}' for i in range(24)]
    frames = [_division_season(2026, 1, 'Premier League', top, '2027-05-20'),
              _division_season(2026, 2, 'Championship', second, '2027-05-01')]
    if started_below:
        staying_down = top[17:] + second[3:]
        frames.append(_division_season(2027, 2, 'Championship', staying_down, '2027-08-07'))
    return pd.concat(frames, ignore_index=True), top[:17] + second[:3]


def test_health_check_passes_when_the_new_fixture_list_matches_the_inferred_clubs():
    history, clubs = _rollover_history()
    fixtures = _season_fixtures(clubs, season=2027, start='2027-08-07')
    assert pipeline_checks.expected_clubs(history, 2027) == set(clubs)
    assert pipeline_checks.fixture_list_problems(fixtures, history, 2027) == []


def test_health_check_fails_before_kick_off_when_a_promoted_club_is_misspelled():
    history, clubs = _rollover_history()
    fixtures = _season_fixtures(clubs, season=2027, start='2027-08-07').replace({'Champ2': 'Champ Two FC'})
    problems = pipeline_checks.fixture_list_problems(fixtures, history, 2027)
    assert any('Champ2' in problem and 'Champ Two FC' in problem for problem in problems)


def test_health_check_cannot_infer_clubs_before_the_lower_divisions_start():
    history, clubs = _rollover_history(started_below=False)
    fixtures = _season_fixtures(clubs, season=2027, start='2027-08-07').replace({'Champ2': 'Champ Two FC'})
    assert pipeline_checks.expected_clubs(history, 2027) is None
    assert pipeline_checks.fixture_list_problems(fixtures, history, 2027) == []


def test_health_check_fails_when_the_fixture_list_has_last_seasons_clubs():
    history, clubs = _rollover_history()
    old_clubs = [f'Club{i}' for i in range(20)]
    fixtures = _season_fixtures(old_clubs, season=2027, start='2027-08-07')
    problems = pipeline_checks.fixture_list_problems(fixtures, history, 2027)
    assert any('Champ0' in problem and 'Club19' in problem for problem in problems)


def _elo_model(engineered):
    train = main.training_rows(engineered[engineered['season'] < int(engineered['season'].max())])
    model = models.ScaledLogisticModel(C=main.MODEL_C)
    return model.fit(train[features.FEATURE_COLUMNS], train['result'])


def _projection_inputs(engineered):
    season = int(engineered['season'].max())
    top = engineered[(engineered['division'] == 'Premier League') & (engineered['season'] == season - 1)]
    teams = sorted(set(top['home_team_name']))
    history = engineered[engineered['season'] < season]
    _, ratings = features.compute_elo(history)
    return history, season - 1, teams, ratings


def test_zero_elo_sigma_reproduces_the_unperturbed_projection(engineered):
    model = _elo_model(engineered)
    history, season, teams, ratings = _projection_inputs(engineered)
    default = simulate.run_simulations(model, history, season, teams, ratings, n_future_seasons=1,
                                       n_simulations=40, project_current=False, seed=42)
    zero = simulate.run_simulations(model, history, season, teams, ratings, n_future_seasons=1,
                                    n_simulations=40, project_current=False, seed=42, elo_sigma=0.0)
    perturbed = simulate.run_simulations(model, history, season, teams, ratings, n_future_seasons=1,
                                         n_simulations=40, project_current=False, seed=42, elo_sigma=60.0)
    for key in simulate.TABLE_KEYS:
        np.testing.assert_array_equal(default['tables'][season + 1][key], zero['tables'][season + 1][key])
    assert not np.array_equal(default['tables'][season + 1]['Points'], perturbed['tables'][season + 1]['Points'])
    assert main.SIMULATION_ELO_SIGMA == 0.0


def test_strength_perturbation_is_one_draw_per_club_and_simulation():
    state = {'elo': np.tile(np.array([1500.0, 1600.0, 1400.0]), (2000, 1))}
    perturbed = simulate.perturb_strength({'elo': state['elo'].copy()}, 30.0, 42)
    noise = perturbed['elo'] - state['elo']
    assert noise.shape == (2000, 3)
    assert noise.std() == pytest.approx(30.0, rel=0.05)
    assert simulate.perturb_strength(state, 0.0, 42)['elo'] is state['elo']


def test_draw_calibrated_probabilities_sum_to_one(engineered):
    model = _elo_model(engineered)
    rows = engineered[engineered['season'] == int(engineered['season'].max())][features.FEATURE_COLUMNS].head(50)
    for factor in [0.8, 1.0, 1.1, 1.3]:
        proba = models.DrawCalibratedModel(model, factor).predict_proba(rows)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all()
    np.testing.assert_allclose(models.DrawCalibratedModel(model, 1.0).predict_proba(rows), model.predict_proba(rows))
    draw = list(model.classes_).index('draw')
    raised = models.DrawCalibratedModel(model, 1.2).predict_proba(rows)[:, draw]
    assert (raised > model.predict_proba(rows)[:, draw]).all()


def _held_out_season(draw_rate, n=400, seed=0):
    rng = np.random.default_rng(seed)
    proba = np.tile([0.3, 0.25, 0.45], (n, 1))
    codes = rng.choice(3, size=n, p=[(1 - draw_rate) * 0.4, draw_rate, (1 - draw_rate) * 0.6])
    results = pd.DataFrame({'result': [simulate.OUTCOME_ORDER[c] for c in codes]})
    return proba, results


def test_draw_calibration_is_fitted_without_the_evaluated_season():
    held_out = {2022: _held_out_season(0.25, seed=1), 2023: _held_out_season(0.25, seed=2), 2024: _held_out_season(0.45, seed=3)}
    factor = main.draw_factor_before(held_out, 2024)
    changed = dict(held_out)
    changed[2024] = _held_out_season(0.05, seed=4)
    assert main.draw_factor_before(changed, 2024) == factor
    assert main.draw_factor_before(held_out, 2025) > factor
    assert main.draw_factor_before(held_out, 2022) == 1.0


def _live_log_and_results():
    fixtures = [('A', 'B', '2026-10-10 14:00', 6, (0.6, 0.25, 0.15), (2, 0), (1.5, 4.0, 6.0)),
                ('C', 'D', '2026-10-11 14:00', 6, (0.3, 0.3, 0.4), (1, 1), (2.5, 3.2, 2.9)),
                ('A', 'C', '2026-10-17 14:00', 7, (0.5, 0.3, 0.2), (0, 1), (2.0, 3.5, 3.8)),
                ('B', 'D', '2026-10-18 14:00', 7, (0.2, 0.3, 0.5), (0, 2), (4.0, 3.6, 1.9))]
    log = pd.DataFrame([{'predicted_on': '2026-10-06', 'season': 2026, 'kickoff_utc': k, 'home_team_name': h, 'away_team_name': a,
                         'p_home': p[0], 'p_draw': p[1], 'p_away': p[2]} for h, a, k, _, p, _, _ in fixtures])
    results = pd.DataFrame([{'division': 'Premier League', 'season': 2026, 'home_team_name': h, 'away_team_name': a,
                             'home_team_score': s[0], 'away_team_score': s[1],
                             'result': update_data.result_of(s[0], s[1]),
                             'odds_home': o[0], 'odds_draw': o[1], 'odds_away': o[2]} for h, a, _, _, _, s, o in fixtures])
    schedule = pd.DataFrame([{'season': 2026, 'round': r, 'kickoff_utc': k, 'home_team_name': h, 'away_team_name': a}
                             for h, a, k, r, _, _, _ in fixtures])
    return log, results, schedule


def test_live_match_records_match_hand_calculation():
    log, results, schedule = _live_log_and_results()
    records = insights.live_match_records(log, results, schedule)
    assert [r['round'] for r in records] == [6, 6, 7, 7]
    assert [r['model_pick'] for r in records] == [2, 0, 2, 0]
    assert [r['result'] for r in records] == [2, 1, 0, 0]
    assert [r['model_correct'] for r in records] == [True, False, False, True]
    implied = 1 / np.array([2.9, 3.2, 2.5])
    assert records[1]['book_p'] == pytest.approx(list(implied / implied.sum()))
    assert records[1]['book_pick'] == 2 and records[1]['book_correct'] is False
    assert records[0]['score'] == '2-0'


def test_cumulative_track_record_ends_at_the_overall_score():
    log, results, schedule = _live_log_and_results()
    records = insights.live_match_records(log, results, schedule)
    cumulative = insights.cumulative_track_record(records)
    overall = insights.score_prediction_log(log, results)
    assert [row['round'] for row in cumulative] == [6, 7]
    assert [row['matches'] for row in cumulative] == [2, 4]
    assert cumulative[0]['log_loss'] == pytest.approx(-(np.log(0.6) + np.log(0.3)) / 2)
    assert cumulative[-1]['log_loss'] == pytest.approx(overall['log_loss'])
    assert cumulative[-1]['accuracy'] == pytest.approx(overall['accuracy'])
    assert cumulative[-1]['book_log_loss'] == pytest.approx(overall['book_log_loss'])
    assert cumulative[-1]['book_accuracy'] == pytest.approx(overall['book_accuracy'])


def test_track_record_says_plainly_when_bookmakers_lead():
    behind = {'matches': 30, 'accuracy': 0.5, 'log_loss': 1.004, 'first': '2026-10-06', 'book_accuracy': 0.5, 'book_log_loss': 0.951}
    ahead = dict(behind, log_loss=0.90)
    assert insights.live_leader(behind) == 'bookmakers'
    assert insights.live_leader(ahead) == 'model'
    assert 'Bookmakers are ahead' in dashboard.track_record_verdict(behind)
    assert 'model is ahead' in dashboard.track_record_verdict(ahead)
    assert 'Bookmakers are ahead' not in dashboard.track_record_verdict(ahead)


def test_track_record_section_renders_without_javascript():
    log, results, schedule = _live_log_and_results()
    records = insights.live_match_records(log, results, schedule)
    live = insights.score_prediction_log(log, results)
    section = dashboard.track_record_section(live, records, insights.cumulative_track_record(records), recent_rounds=1)
    assert '<script' not in section.lower()
    assert 'accuracy.html#track-record' in section
    assert section.count('<tr>') == 1 + 2
    assert '<svg' in section
    full = dashboard.track_record_section(live, records, insights.cumulative_track_record(records), recent_rounds=0)
    assert full.count('<tr>') == 1 + 4

