import numpy as np
import pandas as pd
from features import BASE_ELO, ELO_K, HOME_ADVANTAGE, FEATURE_COLUMNS

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
    first_leg = rounds
    second_leg = [[(b, a) for (a, b) in rnd] for rnd in rounds]
    return first_leg + second_leg

def _recent_team_values(team_matches, team, home_col, away_col, default):
    values = []
    for r in team_matches.itertuples():
        if getattr(r, 'home_team_name') == team:
            values.append(getattr(r, home_col))
        else:
            values.append(getattr(r, away_col))
    return values or [default]

def init_simulation_state(df_pl, elo_ratings, current_teams):
    state = {}
    for team in current_teams:
        team_matches = df_pl[(df_pl['home_team_name'] == team) | (df_pl['away_team_name'] == team)]
        team_matches = team_matches.sort_values(['season', 'round_num'])
        long_window = team_matches.tail(10)
        short_window = team_matches.tail(7)

        gf_long = _recent_team_values(long_window, team, 'home_team_score', 'away_team_score', 1.35)
        ga_long = _recent_team_values(long_window, team, 'away_team_score', 'home_team_score', 1.35)
        sot_long = _recent_team_values(long_window, team, 'home_shots_on_target', 'away_shots_on_target', 4.5)
        sot_against_long = _recent_team_values(long_window, team, 'away_shots_on_target', 'home_shots_on_target', 4.5)

        gf_short = _recent_team_values(short_window, team, 'home_team_score', 'away_team_score', 1.35)
        ga_short = _recent_team_values(short_window, team, 'away_team_score', 'home_team_score', 1.35)
        sot_for_short = _recent_team_values(short_window, team, 'home_shots_on_target', 'away_shots_on_target', 4.5)
        sot_against_short = _recent_team_values(short_window, team, 'away_shots_on_target', 'home_shots_on_target', 4.5)

        points_short = []
        for gf, ga in zip(gf_short, ga_short):
            points_short.append(3.0 if gf > ga else (1.0 if gf == ga else 0.0))

        state[team] = {
            'elo': elo_ratings.get(team, BASE_ELO),
            'attack': float(np.mean(gf_long)),
            'defense': float(np.mean(ga_long)),
            'sot_attack': float(np.mean(sot_long)),
            'sot_defense': float(np.mean(sot_against_long)),
            'form_pts': list(points_short),
            'form_gf': list(gf_short),
            'form_ga': list(ga_short),
            'form_sot_for': list(sot_for_short),
            'form_sot_against': list(sot_against_short),
        }
    return state

def _rolling_mean(values, window=7):
    tail = values[-window:]
    return float(np.mean(tail)) if tail else 1.35

def _update_elo(state, home, away, hg, ag):
    rh, ra = state[home]['elo'], state[away]['elo']
    exp_home = 1.0 / (1.0 + 10.0 ** (-((rh + HOME_ADVANTAGE) - ra) / 400.0))
    s_home = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
    margin = abs(hg - ag)
    k_eff = ELO_K * (1.0 + 0.35 * np.log1p(margin))
    delta = k_eff * (s_home - exp_home)
    state[home]['elo'] = rh + delta
    state[away]['elo'] = ra - delta

def _clip_rate(value, low=0.15, high=5.0):
    return min(max(value, low), high)

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

    state[home]['attack'] = _clip_rate(0.85 * state[home]['attack'] + 0.15 * hg)
    state[home]['defense'] = _clip_rate(0.85 * state[home]['defense'] + 0.15 * ag)
    state[away]['attack'] = _clip_rate(0.85 * state[away]['attack'] + 0.15 * ag)
    state[away]['defense'] = _clip_rate(0.85 * state[away]['defense'] + 0.15 * hg)

    state[home]['sot_attack'] = _clip_rate(0.85 * state[home]['sot_attack'] + 0.15 * home_sot, low=0.5, high=15.0)
    state[home]['sot_defense'] = _clip_rate(0.85 * state[home]['sot_defense'] + 0.15 * away_sot, low=0.5, high=15.0)
    state[away]['sot_attack'] = _clip_rate(0.85 * state[away]['sot_attack'] + 0.15 * away_sot, low=0.5, high=15.0)
    state[away]['sot_defense'] = _clip_rate(0.85 * state[away]['sot_defense'] + 0.15 * home_sot, low=0.5, high=15.0)

def _sample_scoreline(rng, model_result, lam_home, lam_away, max_tries=12):
    for _ in range(max_tries):
        hg = rng.poisson(lam_home)
        ag = rng.poisson(lam_away)
        actual = 'home team win' if hg > ag else ('away team win' if ag > hg else 'draw')
        if actual == model_result:
            return hg, ag
    if model_result == 'home team win':
        hg = max(1, rng.poisson(lam_home))
        ag = max(0, hg - 1 - rng.poisson(0.5))
        return hg, max(0, ag)
    if model_result == 'away team win':
        ag = max(1, rng.poisson(lam_away))
        hg = max(0, ag - 1 - rng.poisson(0.5))
        return max(0, hg), ag
    g = rng.poisson((lam_home + lam_away) / 2.0)
    return g, g

def build_fixture_features(pairs, state, h2h_state, prev_tier=1.0):
    rows = []
    for home, away in pairs:
        sh, sa = state[home], state[away]
        elo_diff = sh['elo'] - sa['elo']
        h2h_rate = h2h_state.get((home, away), h2h_state.get((away, home), 0.45))
        rows.append({
            'home': home, 'away': away,
            'home_elo': sh['elo'], 'away_elo': sa['elo'], 'elo_diff': elo_diff,
            'home_form_points': _rolling_mean(sh['form_pts']),
            'away_form_points': _rolling_mean(sa['form_pts']),
            'home_form_gf': _rolling_mean(sh['form_gf']),
            'home_form_ga': _rolling_mean(sh['form_ga']),
            'away_form_gf': _rolling_mean(sa['form_gf']),
            'away_form_ga': _rolling_mean(sa['form_ga']),
            'home_form_sot_for': _rolling_mean(sh['form_sot_for']),
            'home_form_sot_against': _rolling_mean(sh['form_sot_against']),
            'away_form_sot_for': _rolling_mean(sa['form_sot_for']),
            'away_form_sot_against': _rolling_mean(sa['form_sot_against']),
            'h2h_home_rate': h2h_rate,
            'home_prev_tier': prev_tier,
            'away_prev_tier': prev_tier,
        })
    return pd.DataFrame(rows)

LEAGUE_AVG_GOALS = 1.35
LEAGUE_AVG_SOT = 4.5

def simulate_seasons(model, current_teams, elo_ratings, df_pl, n_seasons, n_simulations, start_season, seed=42):
    rng = np.random.default_rng(seed)
    all_match_rows = []
    all_table_rows = []

    for sim in range(n_simulations):
        state = init_simulation_state(df_pl, elo_ratings, current_teams)
        h2h_state = {}
        season_label = start_season
        for season_idx in range(n_seasons):
            schedule = round_robin_schedule(current_teams, rng)
            table = {t: {'Points': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0, 'GoalsFor': 0, 'GoalsAgainst': 0} for t in current_teams}
            for rnd_pairs in schedule:
                feats = build_fixture_features(rnd_pairs, state, h2h_state)
                X_pred = feats[FEATURE_COLUMNS]
                proba = model.predict_proba(X_pred)
                classes = model.classes_
                for i, row in feats.iterrows():
                    home, away = row['home'], row['away']
                    p = proba[i]
                    result = rng.choice(classes, p=p / p.sum())

                    lam_home = min(6.0, max(0.15, LEAGUE_AVG_GOALS * (state[home]['attack'] / LEAGUE_AVG_GOALS) * (state[away]['defense'] / LEAGUE_AVG_GOALS)))
                    lam_away = min(6.0, max(0.1, (LEAGUE_AVG_GOALS - 0.15) * (state[away]['attack'] / LEAGUE_AVG_GOALS) * (state[home]['defense'] / LEAGUE_AVG_GOALS)))
                    hg, ag = _sample_scoreline(rng, result, lam_home, lam_away)

                    sot_lam_home = min(16.0, max(0.5, LEAGUE_AVG_SOT * (state[home]['sot_attack'] / LEAGUE_AVG_SOT) * (state[away]['sot_defense'] / LEAGUE_AVG_SOT)))
                    sot_lam_away = min(16.0, max(0.5, LEAGUE_AVG_SOT * (state[away]['sot_attack'] / LEAGUE_AVG_SOT) * (state[home]['sot_defense'] / LEAGUE_AVG_SOT)))
                    home_sot = max(hg, rng.poisson(sot_lam_home))
                    away_sot = max(ag, rng.poisson(sot_lam_away))

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

                    _update_elo(state, home, away, hg, ag)
                    _update_form(state, home, away, hg, ag, home_sot, away_sot)
                    h2h_state[(home, away)] = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)

                    if sim == 0:
                        all_match_rows.append({
                            'season': season_label, 'home_team': home, 'away_team': away,
                            'predicted_result': result, 'home_goals': hg, 'away_goals': ag,
                        })

            for t in current_teams:
                row = table[t]
                row['Team'] = t
                row['Season'] = season_label
                row['GoalDiff'] = row['GoalsFor'] - row['GoalsAgainst']
                row['Simulation'] = sim
                all_table_rows.append(row)

            season_label += 1

    matches_df = pd.DataFrame(all_match_rows)
    table_df = pd.DataFrame(all_table_rows)
    return matches_df, table_df

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
    for season, grp in table_df.groupby(['Season', 'Simulation']):
        s, sim = season
        ranked = grp.sort_values(['Points', 'GoalDiff', 'GoalsFor'], ascending=False).reset_index(drop=True)
        ranked['Rank'] = ranked.index + 1
        rank_frames.append(ranked[['Season', 'Team', 'Rank']])
    ranks = pd.concat(rank_frames, ignore_index=True)

    title_prob = ranks[ranks['Rank'] == 1].groupby(['Season', 'Team']).size().div(n_simulations).rename('TitleProb')
    top4_prob = ranks[ranks['Rank'] <= 4].groupby(['Season', 'Team']).size().div(n_simulations).rename('Top4Prob')
    relegation_prob = ranks[ranks['Rank'] >= len(current_teams) - 2].groupby(['Season', 'Team']).size().div(n_simulations).rename('RelegationProb')

    summary = summary.set_index(['Season', 'Team'])
    summary = summary.join(title_prob).join(top4_prob).join(relegation_prob).fillna(0.0)
    summary = summary.reset_index()
    summary = summary.sort_values(['Season', 'Points', 'GoalDiff'], ascending=[True, False, False])
    summary['Rank'] = summary.groupby('Season').cumcount() + 1
    top10 = summary[summary['Rank'] <= 10].reset_index(drop=True)
    round_cols = ['Points', 'Wins', 'Draws', 'Losses', 'GoalsFor', 'GoalsAgainst', 'GoalDiff']
    top10[round_cols] = top10[round_cols].round(2)
    prob_cols = ['TitleProb', 'Top4Prob', 'RelegationProb']
    top10[prob_cols] = top10[prob_cols].round(3)
    cols = ['Season', 'Rank', 'Team', 'Points', 'Wins', 'Draws', 'Losses',
            'GoalsFor', 'GoalsAgainst', 'GoalDiff', 'TitleProb', 'Top4Prob', 'RelegationProb']
    return top10[cols]
