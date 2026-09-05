import numpy as np
import pandas as pd
from features import BASE_ELO, ELO_K, HOME_ADVANTAGE, ELO_MARGIN_MULTIPLIER, FORM_WINDOW, FEATURE_COLUMNS

LEAGUE_AVG_GOALS = 1.35
LEAGUE_AVG_SOT = 4.5
MATCHES_PER_TEAM_SEASON = 38

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

def empty_table(teams):
    return {t: {'Points': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0, 'GoalsFor': 0, 'GoalsAgainst': 0, 'Played': 0} for t in teams}

def table_from_results(season_matches, teams):
    table = empty_table(teams)
    for row in season_matches.itertuples():
        home, away = row.home_team_name, row.away_team_name
        if home not in table or away not in table:
            continue
        hg, ag = int(row.home_team_score), int(row.away_team_score)
        _apply_result(table, home, away, hg, ag)
    return table

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

def _recent_values(matches, team, home_col, away_col, default):
    values = []
    for row in matches.itertuples():
        if row.home_team_name == team:
            values.append(getattr(row, home_col))
        else:
            values.append(getattr(row, away_col))
    return values or [default]

def init_simulation_state(history, elo_ratings, teams, window=FORM_WINDOW):
    state = {}
    for team in teams:
        team_matches = history[(history['home_team_name'] == team) | (history['away_team_name'] == team)]
        team_matches = team_matches.sort_values('match_date', kind='mergesort')
        recent = team_matches.tail(window)
        long_window = team_matches.tail(max(window, 10))

        gf = _recent_values(recent, team, 'home_team_score', 'away_team_score', LEAGUE_AVG_GOALS)
        ga = _recent_values(recent, team, 'away_team_score', 'home_team_score', LEAGUE_AVG_GOALS)
        sot_for = _recent_values(recent, team, 'home_shots_on_target', 'away_shots_on_target', LEAGUE_AVG_SOT)
        sot_against = _recent_values(recent, team, 'away_shots_on_target', 'home_shots_on_target', LEAGUE_AVG_SOT)
        gf_long = _recent_values(long_window, team, 'home_team_score', 'away_team_score', LEAGUE_AVG_GOALS)
        ga_long = _recent_values(long_window, team, 'away_team_score', 'home_team_score', LEAGUE_AVG_GOALS)
        sot_long = _recent_values(long_window, team, 'home_shots_on_target', 'away_shots_on_target', LEAGUE_AVG_SOT)
        sot_against_long = _recent_values(long_window, team, 'away_shots_on_target', 'home_shots_on_target', LEAGUE_AVG_SOT)

        points = [3.0 if f > a else (1.0 if f == a else 0.0) for f, a in zip(gf, ga)]
        state[team] = {
            'elo': elo_ratings.get(team, BASE_ELO),
            'attack': float(np.mean(gf_long)),
            'defense': float(np.mean(ga_long)),
            'sot_attack': float(np.mean(sot_long)),
            'sot_defense': float(np.mean(sot_against_long)),
            'form_pts': list(points),
            'form_gf': list(gf),
            'form_ga': list(ga),
            'form_sot_for': list(sot_for),
            'form_sot_against': list(sot_against),
        }
    return state

def init_h2h_state(history, teams, window=3):
    h2h = {}
    relevant = history[history['home_team_name'].isin(teams) & history['away_team_name'].isin(teams)]
    relevant = relevant.sort_values('match_date', kind='mergesort')
    for row in relevant.itertuples():
        key = tuple(sorted((row.home_team_name, row.away_team_name)))
        outcome = 1.0 if row.home_team_score > row.away_team_score else (0.0 if row.home_team_score < row.away_team_score else 0.5)
        if row.home_team_name != key[0]:
            outcome = 1.0 - outcome if outcome != 0.5 else 0.5
        h2h.setdefault(key, []).append(outcome)
    return {k: v[-window:] for k, v in h2h.items()}

def _h2h_rate(h2h_state, home, away):
    key = tuple(sorted((home, away)))
    values = h2h_state.get(key)
    if not values:
        return 0.45
    rate = float(np.mean(values))
    return rate if home == key[0] else 1.0 - rate

def _rolling_mean(values, window=FORM_WINDOW, default=1.35):
    tail = values[-window:]
    return float(np.mean(tail)) if tail else default

def _clip(value, low, high):
    return min(max(value, low), high)

def _update_elo(state, home, away, hg, ag):
    rh, ra = state[home]['elo'], state[away]['elo']
    expected_home = 1.0 / (1.0 + 10.0 ** (-((rh + HOME_ADVANTAGE) - ra) / 400.0))
    actual_home = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
    k_eff = ELO_K * (1.0 + ELO_MARGIN_MULTIPLIER * np.log1p(abs(hg - ag)))
    delta = k_eff * (actual_home - expected_home)
    state[home]['elo'] = rh + delta
    state[away]['elo'] = ra - delta

def _update_form(state, home, away, hg, ag, home_sot, away_sot):
    home_pts = 3.0 if hg > ag else (1.0 if hg == ag else 0.0)
    away_pts = 3.0 if ag > hg else (1.0 if hg == ag else 0.0)
    state[home]['form_pts'].append(home_pts)
    state[home]['form_gf'].append(hg)
    state[home]['form_ga'].append(ag)
    state[home]['form_sot_for'].append(home_sot)
    state[home]['form_sot_against'].append(away_sot)
    state[away]['form_pts'].append(away_pts)
    state[away]['form_gf'].append(ag)
    state[away]['form_ga'].append(hg)
    state[away]['form_sot_for'].append(away_sot)
    state[away]['form_sot_against'].append(home_sot)

    state[home]['attack'] = _clip(0.85 * state[home]['attack'] + 0.15 * hg, 0.15, 5.0)
    state[home]['defense'] = _clip(0.85 * state[home]['defense'] + 0.15 * ag, 0.15, 5.0)
    state[away]['attack'] = _clip(0.85 * state[away]['attack'] + 0.15 * ag, 0.15, 5.0)
    state[away]['defense'] = _clip(0.85 * state[away]['defense'] + 0.15 * hg, 0.15, 5.0)
    state[home]['sot_attack'] = _clip(0.85 * state[home]['sot_attack'] + 0.15 * home_sot, 0.5, 15.0)
    state[home]['sot_defense'] = _clip(0.85 * state[home]['sot_defense'] + 0.15 * away_sot, 0.5, 15.0)
    state[away]['sot_attack'] = _clip(0.85 * state[away]['sot_attack'] + 0.15 * away_sot, 0.5, 15.0)
    state[away]['sot_defense'] = _clip(0.85 * state[away]['sot_defense'] + 0.15 * home_sot, 0.5, 15.0)

def _update_h2h(h2h_state, home, away, hg, ag, window=3):
    key = tuple(sorted((home, away)))
    outcome = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
    if home != key[0] and outcome != 0.5:
        outcome = 1.0 - outcome
    h2h_state.setdefault(key, []).append(outcome)
    h2h_state[key] = h2h_state[key][-window:]

def _sample_scoreline(rng, target_result, lam_home, lam_away, max_tries=12):
    for _ in range(max_tries):
        hg = rng.poisson(lam_home)
        ag = rng.poisson(lam_away)
        drawn = 'home team win' if hg > ag else ('away team win' if ag > hg else 'draw')
        if drawn == target_result:
            return int(hg), int(ag)
    if target_result == 'home team win':
        hg = max(1, int(rng.poisson(lam_home)))
        return hg, max(0, hg - 1 - int(rng.poisson(0.5)))
    if target_result == 'away team win':
        ag = max(1, int(rng.poisson(lam_away)))
        return max(0, ag - 1 - int(rng.poisson(0.5))), ag
    g = int(rng.poisson((lam_home + lam_away) / 2.0))
    return g, g

def build_fixture_features(pairs, state, h2h_state, prev_tier_map):
    rows = []
    for home, away in pairs:
        sh, sa = state[home], state[away]
        rows.append({
            'home': home,
            'away': away,
            'elo_diff': sh['elo'] - sa['elo'],
            'home_form_points': _rolling_mean(sh['form_pts']),
            'away_form_points': _rolling_mean(sa['form_pts']),
            'home_form_gf': _rolling_mean(sh['form_gf']),
            'home_form_ga': _rolling_mean(sh['form_ga']),
            'away_form_gf': _rolling_mean(sa['form_gf']),
            'away_form_ga': _rolling_mean(sa['form_ga']),
            'home_form_sot_for': _rolling_mean(sh['form_sot_for'], default=LEAGUE_AVG_SOT),
            'home_form_sot_against': _rolling_mean(sh['form_sot_against'], default=LEAGUE_AVG_SOT),
            'away_form_sot_for': _rolling_mean(sa['form_sot_for'], default=LEAGUE_AVG_SOT),
            'away_form_sot_against': _rolling_mean(sa['form_sot_against'], default=LEAGUE_AVG_SOT),
            'h2h_home_rate': _h2h_rate(h2h_state, home, away),
            'home_prev_tier': prev_tier_map.get(home, 1.0),
            'away_prev_tier': prev_tier_map.get(away, 1.0),
        })
    return pd.DataFrame(rows)

def play_rounds(model, rounds, state, h2h_state, prev_tier_map, table, rng, season_label, match_sink=None):
    for pairs in rounds:
        if not pairs:
            continue
        feats = build_fixture_features(pairs, state, h2h_state, prev_tier_map)
        proba = model.predict_proba(feats[FEATURE_COLUMNS])
        classes = model.classes_
        for i, row in feats.iterrows():
            home, away = row['home'], row['away']
            p = proba[i]
            result = rng.choice(classes, p=p / p.sum())

            lam_home = _clip(state[home]['attack'] * state[away]['defense'] / LEAGUE_AVG_GOALS, 0.15, 6.0)
            lam_away = _clip((state[away]['attack'] * state[home]['defense'] / LEAGUE_AVG_GOALS) * 0.9, 0.1, 6.0)
            hg, ag = _sample_scoreline(rng, result, lam_home, lam_away)

            sot_lam_home = _clip(state[home]['sot_attack'] * state[away]['sot_defense'] / LEAGUE_AVG_SOT, 0.5, 16.0)
            sot_lam_away = _clip(state[away]['sot_attack'] * state[home]['sot_defense'] / LEAGUE_AVG_SOT, 0.5, 16.0)
            home_sot = max(hg, int(rng.poisson(sot_lam_home)))
            away_sot = max(ag, int(rng.poisson(sot_lam_away)))

            _apply_result(table, home, away, hg, ag)
            _update_elo(state, home, away, hg, ag)
            _update_form(state, home, away, hg, ag, home_sot, away_sot)
            _update_h2h(h2h_state, home, away, hg, ag)

            if match_sink is not None:
                match_sink.append({
                    'season': season_label,
                    'home_team': home,
                    'away_team': away,
                    'predicted_result': result,
                    'home_goals': hg,
                    'away_goals': ag,
                })
    return table

def _table_rows(table, teams, season_label, sim_index):
    rows = []
    for team in teams:
        row = dict(table[team])
        row['Team'] = team
        row['Season'] = season_label
        row['GoalDiff'] = row['GoalsFor'] - row['GoalsAgainst']
        row['Simulation'] = sim_index
        rows.append(row)
    return rows

def run_simulations(model, history, current_season, current_teams, elo_ratings,
                    n_future_seasons, n_simulations, seed=42):
    rng = np.random.default_rng(seed)
    top_flight = history[history['division'] == 'Premier League']
    season_matches = top_flight[top_flight['season'] == current_season]
    played_matches = len(season_matches)
    season_complete = played_matches >= len(current_teams) * (len(current_teams) - 1)

    from features import team_prev_tier_map
    current_prev_tier = team_prev_tier_map(history, current_season)
    future_prev_tier = {t: 1.0 for t in current_teams}

    history_before = history[history['season'] < current_season] if season_complete else history
    all_match_rows = []
    all_table_rows = []

    for sim in range(n_simulations):
        state = init_simulation_state(history, elo_ratings, current_teams)
        h2h_state = init_h2h_state(history, current_teams)
        sink = all_match_rows if sim == 0 else None

        if not season_complete:
            table = table_from_results(season_matches, current_teams)
            fixtures = remaining_fixtures(season_matches, current_teams)
            rounds = group_into_rounds(fixtures, rng)
            play_rounds(model, rounds, state, h2h_state, current_prev_tier, table, rng, current_season, sink)
            all_table_rows.extend(_table_rows(table, current_teams, current_season, sim))
            next_season = current_season + 1
        else:
            next_season = current_season + 1

        for offset in range(n_future_seasons):
            season_label = next_season + offset
            table = empty_table(current_teams)
            rounds = round_robin_schedule(current_teams, rng)
            play_rounds(model, rounds, state, h2h_state, future_prev_tier, table, rng, season_label, sink)
            all_table_rows.extend(_table_rows(table, current_teams, season_label, sim))

    return pd.DataFrame(all_match_rows), pd.DataFrame(all_table_rows)

def summarize_simulations(table_df, current_teams, n_simulations):
    summary = table_df.groupby(['Season', 'Team']).agg(
        Points=('Points', 'mean'),
        Wins=('Wins', 'mean'),
        Draws=('Draws', 'mean'),
        Losses=('Losses', 'mean'),
        GoalsFor=('GoalsFor', 'mean'),
        GoalsAgainst=('GoalsAgainst', 'mean'),
        GoalDiff=('GoalDiff', 'mean'),
    ).reset_index()

    rank_frames = []
    for _, group in table_df.groupby(['Season', 'Simulation']):
        ranked = group.sort_values(['Points', 'GoalDiff', 'GoalsFor'], ascending=False).reset_index(drop=True)
        ranked['Rank'] = ranked.index + 1
        rank_frames.append(ranked[['Season', 'Team', 'Rank']])
    ranks = pd.concat(rank_frames, ignore_index=True)

    title_prob = ranks[ranks['Rank'] == 1].groupby(['Season', 'Team']).size().div(n_simulations).rename('TitleProb')
    top4_prob = ranks[ranks['Rank'] <= 4].groupby(['Season', 'Team']).size().div(n_simulations).rename('Top4Prob')
    relegation_prob = ranks[ranks['Rank'] >= len(current_teams) - 2].groupby(['Season', 'Team']).size().div(n_simulations).rename('RelegationProb')

    summary = summary.set_index(['Season', 'Team'])
    summary = summary.join(title_prob).join(top4_prob).join(relegation_prob).fillna(0.0).reset_index()
    summary = summary.sort_values(['Season', 'Points', 'GoalDiff'], ascending=[True, False, False])
    summary['Rank'] = summary.groupby('Season').cumcount() + 1
    round_cols = ['Points', 'Wins', 'Draws', 'Losses', 'GoalsFor', 'GoalsAgainst', 'GoalDiff']
    summary[round_cols] = summary[round_cols].round(2)
    prob_cols = ['TitleProb', 'Top4Prob', 'RelegationProb']
    summary[prob_cols] = summary[prob_cols].round(3)
    cols = ['Season', 'Rank', 'Team', 'Points', 'Wins', 'Draws', 'Losses',
            'GoalsFor', 'GoalsAgainst', 'GoalDiff', 'TitleProb', 'Top4Prob', 'RelegationProb']
    return summary[cols].reset_index(drop=True)
