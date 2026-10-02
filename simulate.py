import numpy as np
import pandas as pd
from features import (BASE_ELO, ELO_K, HOME_ADVANTAGE, ELO_MARGIN_MULTIPLIER, FORM_WINDOW,
                      FEATURE_COLUMNS, TIER_ADJUSTMENT, TOP_DIVISION, team_prev_tier_map, promoted_to_top)

LEAGUE_AVG_GOALS = 1.35
LEAGUE_AVG_SOT = 4.5
LONG_WINDOW = 10
SCORELINE_TRIES = 12
FORM_KEYS = ['points', 'gf', 'ga', 'sot_for', 'sot_against']
FORM_DEFAULTS = {'points': 1.35, 'gf': LEAGUE_AVG_GOALS, 'ga': LEAGUE_AVG_GOALS,
                 'sot_for': LEAGUE_AVG_SOT, 'sot_against': LEAGUE_AVG_SOT}
TABLE_KEYS = ['Points', 'Wins', 'Draws', 'Losses', 'GoalsFor', 'GoalsAgainst']
OUTCOME_ORDER = ['away team win', 'draw', 'home team win']


def round_robin_schedule(teams, rng):
    teams = list(teams)
    rng.shuffle(teams)
    n = len(teams)
    if n % 2 == 1:
        teams.append(None)
        n += 1
    rounds = []
    arr = teams[:]
    for _ in range(n - 1):
        pairs = []
        for i in range(n // 2):
            t1, t2 = arr[i], arr[n - 1 - i]
            if t1 is not None and t2 is not None:
                pairs.append((t1, t2))
        rounds.append(pairs)
        arr.insert(1, arr.pop())
    second_leg = [[(b, a) for (a, b) in rnd] for rnd in rounds]
    return rounds + second_leg


def remaining_fixtures(season_matches, teams):
    played = set(zip(season_matches['home_team_name'], season_matches['away_team_name']))
    fixtures = []
    for home in teams:
        for away in teams:
            if home != away and (home, away) not in played:
                fixtures.append((home, away))
    return fixtures


def group_into_rounds(fixtures, rng):
    remaining = list(fixtures)
    rng.shuffle(remaining)
    rounds = []
    while remaining:
        used = set()
        current = []
        leftover = []
        for home, away in remaining:
            if home in used or away in used:
                leftover.append((home, away))
                continue
            current.append((home, away))
            used.add(home)
            used.add(away)
        rounds.append(current)
        remaining = leftover
    return rounds


def rounds_in_order(fixtures):
    rounds = []
    current = []
    used = set()
    for home, away in fixtures:
        if home in used or away in used:
            rounds.append(current)
            current = []
            used = set()
        current.append((home, away))
        used.add(home)
        used.add(away)
    if current:
        rounds.append(current)
    return rounds


def ordered_remaining_fixtures(season_matches, teams, schedule):
    remaining = set(remaining_fixtures(season_matches, teams))
    ordered = []
    if schedule is not None:
        for row in schedule.sort_values(['kickoff_utc', 'home_team_name'], kind='mergesort').itertuples():
            pair = (row.home_team_name, row.away_team_name)
            if pair in remaining:
                ordered.append(pair)
                remaining.discard(pair)
    return ordered, sorted(remaining)


def empty_table(teams):
    return {t: {'Points': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0, 'GoalsFor': 0, 'GoalsAgainst': 0, 'Played': 0} for t in teams}


def _apply_result(table, home, away, hg, ag):
    if hg > ag:
        table[home]['Points'] += 3
        table[home]['Wins'] += 1
        table[away]['Losses'] += 1
    elif ag > hg:
        table[away]['Points'] += 3
        table[away]['Wins'] += 1
        table[home]['Losses'] += 1
    else:
        table[home]['Points'] += 1
        table[away]['Points'] += 1
        table[home]['Draws'] += 1
        table[away]['Draws'] += 1
    table[home]['GoalsFor'] += hg
    table[home]['GoalsAgainst'] += ag
    table[away]['GoalsFor'] += ag
    table[away]['GoalsAgainst'] += hg
    table[home]['Played'] += 1
    table[away]['Played'] += 1


def table_from_results(season_matches, teams):
    table = empty_table(teams)
    for row in season_matches.itertuples():
        home, away = row.home_team_name, row.away_team_name
        if home not in table or away not in table:
            continue
        _apply_result(table, home, away, int(row.home_team_score), int(row.away_team_score))
    return table


def last_played_tier(history):
    ordered = history.sort_values('match_date', kind='mergesort')
    tiers = {}
    for row in ordered.itertuples():
        tiers[row.home_team_name] = row.tier
        tiers[row.away_team_name] = row.tier
    return tiers


def _team_history(history, team):
    played = history[(history['home_team_name'] == team) | (history['away_team_name'] == team)]
    played = played.sort_values('match_date', kind='mergesort')
    at_home = (played['home_team_name'] == team).values
    gf = np.where(at_home, played['home_team_score'], played['away_team_score']).astype(float)
    ga = np.where(at_home, played['away_team_score'], played['home_team_score']).astype(float)
    sot_for = np.where(at_home, played['home_shots_on_target'], played['away_shots_on_target']).astype(float)
    sot_against = np.where(at_home, played['away_shots_on_target'], played['home_shots_on_target']).astype(float)
    points = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
    return {'points': points, 'gf': gf, 'ga': ga, 'sot_for': sot_for, 'sot_against': sot_against}


def _long_mean(values, default):
    tail = values[-LONG_WINDOW:]
    return float(tail.mean()) if len(tail) else default


def init_state(history, elo_ratings, teams, n_simulations, target_tier=1, window=FORM_WINDOW):
    tiers = last_played_tier(history)
    n_teams = len(teams)
    base = {key: np.zeros(n_teams) for key in ['elo', 'attack', 'defense', 'sot_attack', 'sot_defense']}
    buffers = {key: np.zeros((n_teams, window)) for key in FORM_KEYS}
    counts = np.zeros(n_teams, dtype=int)
    for i, team in enumerate(teams):
        past = _team_history(history, team)
        base['attack'][i] = _long_mean(past['gf'], LEAGUE_AVG_GOALS)
        base['defense'][i] = _long_mean(past['ga'], LEAGUE_AVG_GOALS)
        base['sot_attack'][i] = _long_mean(past['sot_for'], LEAGUE_AVG_SOT)
        base['sot_defense'][i] = _long_mean(past['sot_against'], LEAGUE_AVG_SOT)
        recent = min(len(past['gf']), window)
        counts[i] = recent
        if recent:
            for key in FORM_KEYS:
                buffers[key][i, :recent] = past[key][-recent:]
        rating = elo_ratings.get(team, BASE_ELO)
        rating -= (tiers.get(team, target_tier) - target_tier) * TIER_ADJUSTMENT
        base['elo'][i] = rating
    state = {key: np.tile(values, (n_simulations, 1)) for key, values in base.items()}
    state['buffers'] = {key: np.tile(values, (n_simulations, 1, 1)) for key, values in buffers.items()}
    state['counts'] = np.tile(counts, (n_simulations, 1))
    return state


def _form_means(state):
    counts = state['counts']
    means = {}
    for key in FORM_KEYS:
        totals = state['buffers'][key].sum(axis=2)
        means[key] = np.where(counts > 0, totals / np.maximum(counts, 1), FORM_DEFAULTS[key])
    return means


def _push_form(state, sims, teams, values):
    window = state['buffers']['points'].shape[2]
    counts = state['counts'][sims, teams]
    full = counts >= window
    for key in FORM_KEYS:
        rows = state['buffers'][key][sims, teams]
        shifted = np.concatenate([rows[:, 1:], values[key][:, None]], axis=1)
        rows = np.where(full[:, None], shifted, rows)
        open_slots = ~full
        rows[open_slots, counts[open_slots]] = values[key][open_slots]
        state['buffers'][key][sims, teams] = rows
    state['counts'][sims, teams] = np.minimum(counts + 1, window)


def fixture_features(state, sims, home, away, prev_tier):
    form = _form_means(state)
    columns = {
        'elo_diff': state['elo'][sims, home] - state['elo'][sims, away],
        'home_form_points': form['points'][sims, home],
        'away_form_points': form['points'][sims, away],
        'home_form_gf': form['gf'][sims, home],
        'home_form_ga': form['ga'][sims, home],
        'away_form_gf': form['gf'][sims, away],
        'away_form_ga': form['ga'][sims, away],
        'home_form_sot_for': form['sot_for'][sims, home],
        'home_form_sot_against': form['sot_against'][sims, home],
        'away_form_sot_for': form['sot_for'][sims, away],
        'away_form_sot_against': form['sot_against'][sims, away],
        'home_prev_tier': prev_tier[home],
        'away_prev_tier': prev_tier[away],
        'home_promoted_top': promoted_to_top(1, prev_tier[home]),
        'away_promoted_top': promoted_to_top(1, prev_tier[away]),
    }
    return pd.DataFrame({name: columns[name] for name in FEATURE_COLUMNS})


def sample_scorelines(rng, outcome, lam_home, lam_away, tries=SCORELINE_TRIES):
    n = len(outcome)
    home_goals = np.zeros(n, dtype=int)
    away_goals = np.zeros(n, dtype=int)
    done = np.zeros(n, dtype=bool)
    for _ in range(tries):
        hg = rng.poisson(lam_home)
        ag = rng.poisson(lam_away)
        drawn = np.where(hg > ag, 2, np.where(hg < ag, 0, 1))
        accept = ~done & (drawn == outcome)
        home_goals[accept] = hg[accept]
        away_goals[accept] = ag[accept]
        done |= accept
        if done.all():
            return home_goals, away_goals
    left = ~done
    target = outcome[left]
    winner_home = np.maximum(1, rng.poisson(lam_home[left]))
    winner_away = np.maximum(1, rng.poisson(lam_away[left]))
    margin = 1 + rng.poisson(0.5, size=left.sum())
    level = rng.poisson((lam_home[left] + lam_away[left]) / 2.0)
    home_goals[left] = np.where(target == 2, winner_home, np.where(target == 0, np.maximum(0, winner_away - margin), level))
    away_goals[left] = np.where(target == 2, np.maximum(0, winner_home - margin), np.where(target == 0, winner_away, level))
    return home_goals, away_goals


def _update_ratings(state, sims, home, away, hg, ag, home_sot, away_sot):
    rh = state['elo'][sims, home]
    ra = state['elo'][sims, away]
    expected = 1.0 / (1.0 + 10.0 ** (-((rh + HOME_ADVANTAGE) - ra) / 400.0))
    actual = np.where(hg > ag, 1.0, np.where(hg < ag, 0.0, 0.5))
    delta = ELO_K * (1.0 + ELO_MARGIN_MULTIPLIER * np.log1p(np.abs(hg - ag))) * (actual - expected)
    state['elo'][sims, home] = rh + delta
    state['elo'][sims, away] = ra - delta
    for key, low, high, home_value, away_value in [
        ('attack', 0.15, 5.0, hg, ag), ('defense', 0.15, 5.0, ag, hg),
        ('sot_attack', 0.5, 15.0, home_sot, away_sot), ('sot_defense', 0.5, 15.0, away_sot, home_sot),
    ]:
        state[key][sims, home] = np.clip(0.85 * state[key][sims, home] + 0.15 * home_value, low, high)
        state[key][sims, away] = np.clip(0.85 * state[key][sims, away] + 0.15 * away_value, low, high)


def _record(table, sims, home, away, hg, ag):
    home_points = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
    away_points = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
    table['Points'][sims, home] += home_points
    table['Points'][sims, away] += away_points
    table['Wins'][sims, home] += hg > ag
    table['Wins'][sims, away] += ag > hg
    table['Draws'][sims, home] += hg == ag
    table['Draws'][sims, away] += hg == ag
    table['Losses'][sims, home] += hg < ag
    table['Losses'][sims, away] += ag < hg
    table['GoalsFor'][sims, home] += hg
    table['GoalsAgainst'][sims, home] += ag
    table['GoalsFor'][sims, away] += ag
    table['GoalsAgainst'][sims, away] += hg
    return home_points, away_points


def play_rounds(model, home_slots, away_slots, state, prev_tier, table, rng, fixture_slots=None, outcomes=None):
    classes = list(model.classes_)
    order = [classes.index(outcome) for outcome in OUTCOME_ORDER]
    for r in range(home_slots.shape[1]):
        valid = home_slots[:, r, :] >= 0
        sims = np.nonzero(valid)[0]
        if len(sims) == 0:
            continue
        home = home_slots[:, r, :][valid]
        away = away_slots[:, r, :][valid]
        if outcomes is not None:
            fixture_ids = fixture_slots[:, r, :][valid]
        proba = model.predict_proba(fixture_features(state, sims, home, away, prev_tier))[:, order]
        cumulative = np.cumsum(proba, axis=1)
        cumulative[:, -1] = 1.0
        outcome = (rng.random(len(sims))[:, None] > cumulative).sum(axis=1)

        lam_home = np.clip(state['attack'][sims, home] * state['defense'][sims, away] / LEAGUE_AVG_GOALS, 0.15, 6.0)
        lam_away = np.clip(state['attack'][sims, away] * state['defense'][sims, home] / LEAGUE_AVG_GOALS * 0.9, 0.1, 6.0)
        hg, ag = sample_scorelines(rng, outcome, lam_home, lam_away)
        if outcomes is not None:
            outcomes[sims, fixture_ids] = outcome
        sot_home = np.clip(state['sot_attack'][sims, home] * state['sot_defense'][sims, away] / LEAGUE_AVG_SOT, 0.5, 16.0)
        sot_away = np.clip(state['sot_attack'][sims, away] * state['sot_defense'][sims, home] / LEAGUE_AVG_SOT, 0.5, 16.0)
        home_sot = np.maximum(hg, rng.poisson(sot_home))
        away_sot = np.maximum(ag, rng.poisson(sot_away))

        home_points, away_points = _record(table, sims, home, away, hg, ag)
        _update_ratings(state, sims, home, away, hg, ag, home_sot, away_sot)
        _push_form(state, sims, home, {'points': home_points.astype(float), 'gf': hg.astype(float), 'ga': ag.astype(float),
                                       'sot_for': home_sot.astype(float), 'sot_against': away_sot.astype(float)})
        _push_form(state, sims, away, {'points': away_points.astype(float), 'gf': ag.astype(float), 'ga': hg.astype(float),
                                       'sot_for': away_sot.astype(float), 'sot_against': home_sot.astype(float)})
    return table


def pack_rounds(schedules, index, fixture_index=None):
    n_rounds = max(len(schedule) for schedule in schedules)
    n_slots = max(len(rnd) for schedule in schedules for rnd in schedule)
    shape = (len(schedules), n_rounds, n_slots)
    home_slots = -np.ones(shape, dtype=int)
    away_slots = -np.ones(shape, dtype=int)
    fixture_slots = -np.ones(shape, dtype=int)
    for s, schedule in enumerate(schedules):
        for r, pairs in enumerate(schedule):
            for p, (home, away) in enumerate(pairs):
                home_slots[s, r, p] = index[home]
                away_slots[s, r, p] = index[away]
                if fixture_index is not None:
                    fixture_slots[s, r, p] = fixture_index[(home, away)]
    return home_slots, away_slots, fixture_slots


def _table_arrays(n_simulations, start):
    return {key: np.tile(np.asarray(start[key], dtype=int), (n_simulations, 1)) for key in TABLE_KEYS}


def run_simulations(model, history, current_season, current_teams, elo_ratings,
                    n_future_seasons, n_simulations, project_current=True, seed=42, schedule=None):
    rng = np.random.default_rng(seed)
    teams = list(current_teams)
    index = {team: i for i, team in enumerate(teams)}
    season_matches = history[(history['division'] == TOP_DIVISION) & (history['season'] == current_season)]
    previous_tiers = team_prev_tier_map(history, current_season)
    current_prev_tier = np.array([previous_tiers.get(team, 1.0) for team in teams])
    future_prev_tier = np.ones(len(teams))
    state = init_state(history, elo_ratings, teams, n_simulations)
    tables = {}
    fixtures = []
    outcomes = np.zeros((n_simulations, 0), dtype=np.int8)

    if project_current:
        played = table_from_results(season_matches, teams)
        start = {key: [played[team][key] for team in teams] for key in TABLE_KEYS}
        table = _table_arrays(n_simulations, start)
        ordered, unscheduled = ordered_remaining_fixtures(season_matches, teams, schedule)
        fixtures = ordered + unscheduled
        if fixtures:
            fixture_index = {pair: i for i, pair in enumerate(fixtures)}
            outcomes = -np.ones((n_simulations, len(fixtures)), dtype=np.int8)
            fixed = rounds_in_order(ordered)
            schedules = [fixed + (group_into_rounds(unscheduled, rng) if unscheduled else []) for _ in range(n_simulations)]
            home_slots, away_slots, fixture_slots = pack_rounds(schedules, index, fixture_index)
            play_rounds(model, home_slots, away_slots, state, current_prev_tier, table, rng, fixture_slots, outcomes)
        tables[current_season] = table

    for offset in range(n_future_seasons):
        table = _table_arrays(n_simulations, {key: [0] * len(teams) for key in TABLE_KEYS})
        schedules = [round_robin_schedule(teams, rng) for _ in range(n_simulations)]
        home_slots, away_slots, _ = pack_rounds(schedules, index)
        play_rounds(model, home_slots, away_slots, state, future_prev_tier, table, rng)
        tables[current_season + 1 + offset] = table
    return {'tables': tables, 'fixtures': fixtures, 'outcomes': outcomes, 'teams': teams}


def predict_fixtures(model, history, elo_ratings, current_season, teams, fixtures):
    if not fixtures:
        return np.zeros((0, 3))
    teams = list(teams)
    index = {team: i for i, team in enumerate(teams)}
    state = init_state(history, elo_ratings, teams, 1)
    previous_tiers = team_prev_tier_map(history, current_season)
    prev_tier = np.array([previous_tiers.get(team, 1.0) for team in teams])
    home = np.array([index[h] for h, _ in fixtures])
    away = np.array([index[a] for _, a in fixtures])
    frame = fixture_features(state, np.zeros(len(fixtures), dtype=int), home, away, prev_tier)
    classes = list(model.classes_)
    return model.predict_proba(frame)[:, [classes.index(o) for o in OUTCOME_ORDER]]


def finishing_positions(table):
    points = table['Points']
    goal_diff = table['GoalsFor'] - table['GoalsAgainst']
    n_sims, n_teams = points.shape
    positions = np.zeros((n_sims, n_teams), dtype=int)
    tiebreak = np.arange(n_teams)
    for s in range(n_sims):
        order = np.lexsort((tiebreak, -table['GoalsFor'][s], -goal_diff[s], -points[s]))
        positions[s, order] = np.arange(1, n_teams + 1)
    return positions


def summarize_simulations(results, current_teams):
    frames = []
    teams = list(current_teams)
    relegation_line = len(teams) - 2
    for season, table in results['tables'].items():
        positions = finishing_positions(table)
        frame = pd.DataFrame({
            'Season': season,
            'Team': teams,
            'Points': table['Points'].mean(axis=0),
            'Wins': table['Wins'].mean(axis=0),
            'Draws': table['Draws'].mean(axis=0),
            'Losses': table['Losses'].mean(axis=0),
            'GoalsFor': table['GoalsFor'].mean(axis=0),
            'GoalsAgainst': table['GoalsAgainst'].mean(axis=0),
            'GoalDiff': (table['GoalsFor'] - table['GoalsAgainst']).mean(axis=0),
            'TitleProb': (positions == 1).mean(axis=0),
            'Top4Prob': (positions <= 4).mean(axis=0),
            'RelegationProb': (positions >= relegation_line).mean(axis=0),
        })
        frame = frame.sort_values(['Points', 'GoalDiff'], ascending=False, kind='mergesort')
        frame['Rank'] = np.arange(1, len(teams) + 1)
        frames.append(frame)
    summary = pd.concat(frames, ignore_index=True)
    round_cols = ['Points', 'Wins', 'Draws', 'Losses', 'GoalsFor', 'GoalsAgainst', 'GoalDiff']
    summary[round_cols] = summary[round_cols].round(2)
    prob_cols = ['TitleProb', 'Top4Prob', 'RelegationProb']
    summary[prob_cols] = summary[prob_cols].round(4)
    cols = ['Season', 'Rank', 'Team', 'Points', 'Wins', 'Draws', 'Losses',
            'GoalsFor', 'GoalsAgainst', 'GoalDiff', 'TitleProb', 'Top4Prob', 'RelegationProb']
    return summary[cols].reset_index(drop=True)
