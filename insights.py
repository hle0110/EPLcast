import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from simulate import finishing_positions, OUTCOME_ORDER

PREDICTION_LOG_PATH = "predictions/match_predictions.csv"
LOG_COLUMNS = ['predicted_on', 'season', 'kickoff_utc', 'home_team_name', 'away_team_name', 'p_home', 'p_draw', 'p_away']
MIN_STAKE_SAMPLES = 100
METRICS = [('TitleProb', 'win the title', lambda pos, n: pos == 1),
           ('Top4Prob', 'finish top four', lambda pos, n: pos <= 4),
           ('RelegationProb', 'be relegated', lambda pos, n: pos >= n - 2)]


def position_probabilities(table, teams):
    positions = finishing_positions(table)
    n_teams = len(teams)
    counts = np.stack([(positions == p).mean(axis=0) for p in range(1, n_teams + 1)], axis=1)
    frame = pd.DataFrame(counts, columns=[str(p) for p in range(1, n_teams + 1)])
    frame.insert(0, 'Team', list(teams))
    return frame


def points_ranges(table, teams):
    points = table['Points']
    return pd.DataFrame({
        'Team': list(teams),
        'P10': np.percentile(points, 10, axis=0),
        'P50': np.percentile(points, 50, axis=0),
        'P90': np.percentile(points, 90, axis=0),
    })


def upcoming_fixtures(schedule, season_matches, now):
    if schedule is None:
        return []
    played = set(zip(season_matches['home_team_name'], season_matches['away_team_name']))
    future = schedule[schedule['kickoff_utc'] >= now].copy()
    future = future[[(h, a) not in played for h, a in zip(future['home_team_name'], future['away_team_name'])]]
    if len(future) == 0:
        return []
    future = future.sort_values(['kickoff_utc', 'home_team_name'], kind='mergesort')
    next_round = int(future['round'].iloc[0])
    chosen = future[future['round'] == next_round]
    return [(row.kickoff_utc, row.home_team_name, row.away_team_name, int(row.round)) for row in chosen.itertuples()]


def remaining_for_team(schedule, season_matches, team):
    if schedule is None:
        return []
    played = set(zip(season_matches['home_team_name'], season_matches['away_team_name']))
    rows = schedule[(schedule['home_team_name'] == team) | (schedule['away_team_name'] == team)]
    rows = rows.sort_values('kickoff_utc', kind='mergesort')
    return [(row.kickoff_utc, row.home_team_name, row.away_team_name)
            for row in rows.itertuples() if (row.home_team_name, row.away_team_name) not in played]


def match_stakes(results, season, fixtures):
    table = results['tables'][season]
    teams = results['teams']
    positions = finishing_positions(table)
    n_teams = len(teams)
    lookup = {pair: i for i, pair in enumerate(results['fixtures'])}
    stakes = []
    for home, away in fixtures:
        if (home, away) not in lookup:
            stakes.append(None)
            continue
        outcome = results['outcomes'][:, lookup[(home, away)]]
        entry = {}
        for team in (home, away):
            column = teams.index(team)
            best = None
            for key, label, test in METRICS:
                hit = test(positions[:, column], n_teams)
                by_outcome = []
                for code in (2, 1, 0):
                    mask = outcome == code
                    by_outcome.append(float(hit[mask].mean()) if mask.sum() >= MIN_STAKE_SAMPLES else None)
                known = [v for v in by_outcome if v is not None]
                if len(known) < 2:
                    continue
                swing = max(known) - min(known)
                if best is None or swing > best['swing']:
                    best = {'metric': key, 'label': label, 'overall': float(hit.mean()),
                            'home_win': by_outcome[0], 'draw': by_outcome[1], 'away_win': by_outcome[2], 'swing': swing}
            entry[team] = best
        stakes.append(entry)
    return stakes


SCENARIO_TARGETS = [('stay up', lambda pos, n: pos <= n - 3, lambda others, n: np.sort(others, axis=1)[:, 2]),
                    ('finish top four', lambda pos, n: pos <= 4, lambda others, n: -np.sort(-others, axis=1)[:, 3]),
                    ('win the title', lambda pos, n: pos == 1, lambda others, n: others.max(axis=1))]
SCENARIO_RANGE = (0.005, 0.995)
KEY_FIXTURES = 3
KEY_FIXTURE_HORIZON = 5


def simulated_fixture_probabilities(results):
    probabilities = {}
    outcomes = results['outcomes']
    for index, pair in enumerate(results['fixtures']):
        column = outcomes[:, index]
        played = column >= 0
        if played.any():
            probabilities[pair] = np.array([(column[played] == code).mean() for code in (0, 1, 2)])
    return probabilities


def _team_outcomes(results, team):
    columns, venues = [], []
    for index, (home, away) in enumerate(results['fixtures']):
        if team in (home, away):
            columns.append(index)
            venues.append(home == team)
    if not columns:
        return [], np.zeros((results['outcomes'].shape[0], 0), dtype=int)
    raw = results['outcomes'][:, columns].astype(int)
    at_home = np.array(venues)
    view = np.where(raw == 1, 1, np.where((raw == 2) == at_home, 2, 0))
    return columns, view


def club_scenarios(results, season, team, positions, current_points, min_samples=MIN_STAKE_SAMPLES):
    table = results['tables'][season]
    teams = results['teams']
    n_teams = len(teams)
    col = teams.index(team)
    points = table['Points']
    others = np.delete(points, col, axis=1)
    columns, view = _team_outcomes(results, team)
    if not columns:
        return []
    wins = (view == 2).sum(axis=1)
    draws = (view == 1).sum(axis=1)
    scenarios = []
    for label, test, line_of in SCENARIO_TARGETS:
        hit = test(positions[:, col], n_teams)
        probability = float(hit.mean())
        entry = {'label': label, 'probability': probability, 'games': len(columns), 'status': 'open'}
        if probability < SCENARIO_RANGE[0]:
            entry['status'] = 'out of reach'
        elif probability > SCENARIO_RANGE[1]:
            entry['status'] = 'all but certain'
        if entry['status'] != 'open':
            scenarios.append(entry)
            continue
        line = line_of(others, n_teams)
        entry['line_median'] = float(np.median(line))
        entry['line_p90'] = float(np.percentile(line, 90))
        entry['needed_median'] = max(0.0, entry['line_median'] + 1 - current_points)
        entry['needed_p90'] = max(0.0, entry['line_p90'] + 1 - current_points)
        if hit.sum() >= min_samples:
            won = int(round(np.median(wins[hit])))
            drawn = min(int(round(np.median(draws[hit]))), len(columns) - won)
            entry['record'] = (won, drawn, len(columns) - won - drawn)
        key = []
        for position, index in enumerate(columns[:KEY_FIXTURE_HORIZON]):
            by_result = []
            for code in (2, 1, 0):
                mask = view[:, position] == code
                by_result.append(float(hit[mask].mean()) if mask.sum() >= min_samples else None)
            known = [v for v in by_result if v is not None]
            if len(known) < 2:
                continue
            home, away = results['fixtures'][index]
            key.append({'home': home, 'away': away, 'win': by_result[0], 'draw': by_result[1], 'lose': by_result[2],
                        'swing': max(known) - min(known)})
        entry['key_fixtures'] = sorted(key, key=lambda item: item['swing'], reverse=True)[:KEY_FIXTURES]
        scenarios.append(entry)
    return scenarios


def update_prediction_log(rows, today, path=PREDICTION_LOG_PATH):
    fresh = pd.DataFrame(rows, columns=LOG_COLUMNS)
    if os.path.exists(path):
        existing = pd.read_csv(path)
    else:
        existing = pd.DataFrame(columns=LOG_COLUMNS)
    if len(fresh) == 0:
        return existing
    key = ['season', 'home_team_name', 'away_team_name']
    if len(existing) == 0:
        fresh['predicted_on'] = today
        combined = fresh.sort_values(['season', 'kickoff_utc', 'home_team_name'], kind='mergesort').reset_index(drop=True)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        combined.to_csv(path, index=False)
        return combined
    merged = fresh.merge(existing[key + ['predicted_on', 'p_home', 'p_draw', 'p_away']], on=key, how='left', suffixes=('', '_old'))
    unchanged = ((merged['p_home'].round(4) == merged['p_home_old'].round(4)) &
                 (merged['p_draw'].round(4) == merged['p_draw_old'].round(4)) &
                 (merged['p_away'].round(4) == merged['p_away_old'].round(4)))
    merged['predicted_on'] = np.where(unchanged, merged['predicted_on_old'], today)
    fresh = merged[LOG_COLUMNS]
    fresh_keys = set(map(tuple, fresh[key].astype(str).values))
    keep = existing[[tuple(map(str, k)) not in fresh_keys for k in existing[key].values]]
    combined = pd.concat([keep, fresh], ignore_index=True)
    combined = combined.sort_values(['season', 'kickoff_utc', 'home_team_name'], kind='mergesort').reset_index(drop=True)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def bookmaker_probabilities(frame):
    implied = 1.0 / frame[['odds_away', 'odds_draw', 'odds_home']].astype(float).values
    return implied / implied.sum(axis=1, keepdims=True)


def outcome_codes(results):
    return np.array([OUTCOME_ORDER.index(r) for r in results])


def score_probabilities(probabilities, results):
    codes = outcome_codes(results)
    picked = probabilities.argmax(axis=1)
    accuracy = float((picked == codes).mean())
    loss = float(log_loss(codes, probabilities, labels=[0, 1, 2]))
    return accuracy, loss


def _scored_log(log, matches):
    if log is None or len(log) == 0:
        return None
    top = matches[matches['division'] == 'Premier League']
    joined = log.merge(top, on=['season', 'home_team_name', 'away_team_name'], how='inner')
    if len(joined) == 0:
        return None
    order = [column for column in ['kickoff_utc', 'home_team_name'] if column in joined.columns]
    joined = joined.sort_values(order, kind='mergesort').reset_index(drop=True)
    model_probs = joined[['p_away', 'p_draw', 'p_home']].astype(float).values
    model_probs = model_probs / model_probs.sum(axis=1, keepdims=True)
    return joined, model_probs


def score_prediction_log(log, matches):
    scored = _scored_log(log, matches)
    if scored is None:
        return None
    joined, model_probs = scored
    accuracy, loss = score_probabilities(model_probs, joined['result'])
    scored = {'matches': len(joined), 'accuracy': accuracy, 'log_loss': loss,
              'first': str(joined['predicted_on'].min()), 'book_accuracy': None, 'book_log_loss': None}
    with_odds = joined.dropna(subset=['odds_home', 'odds_draw', 'odds_away'])
    if len(with_odds) == len(joined):
        scored['book_accuracy'], scored['book_log_loss'] = score_probabilities(bookmaker_probabilities(with_odds), with_odds['result'])
    return scored


def live_match_records(log, matches, schedule=None):
    scored = _scored_log(log, matches)
    if scored is None:
        return []
    joined, model_probs = scored
    codes = outcome_codes(joined['result'])
    priced = joined[['odds_home', 'odds_draw', 'odds_away']].notna().all(axis=1).values
    book_probs = np.full(model_probs.shape, np.nan)
    if priced.any():
        book_probs[priced] = bookmaker_probabilities(joined[priced])
    rounds = {}
    if schedule is not None and len(schedule):
        rounds = {(int(r.season), r.home_team_name, r.away_team_name): int(r.round) for r in schedule.itertuples()}
    records = []
    for i, row in enumerate(joined.itertuples()):
        model_pick = int(model_probs[i].argmax())
        book_pick = int(book_probs[i].argmax()) if priced[i] else None
        records.append({
            'round': rounds.get((int(row.season), row.home_team_name, row.away_team_name)),
            'kickoff': str(row.kickoff_utc), 'home': row.home_team_name, 'away': row.away_team_name,
            'score': f'{int(row.home_team_score)}-{int(row.away_team_score)}',
            'p': [float(v) for v in model_probs[i]], 'book_p': [float(v) for v in book_probs[i]] if priced[i] else None,
            'result': int(codes[i]), 'model_pick': model_pick, 'book_pick': book_pick,
            'model_correct': model_pick == int(codes[i]),
            'book_correct': (book_pick == int(codes[i])) if book_pick is not None else None,
        })
    return records


def _record_loss(probabilities, code):
    return float(-np.log(max(probabilities[code], 1e-15)))


def cumulative_track_record(records):
    keys = sorted({r['round'] if r['round'] is not None else 0 for r in records})
    rows = []
    for key in keys:
        upto = [r for r in records if (r['round'] if r['round'] is not None else 0) <= key]
        priced = all(r['book_p'] is not None for r in upto)
        rows.append({
            'round': key if key else None, 'matches': len(upto),
            'accuracy': float(np.mean([r['model_correct'] for r in upto])),
            'log_loss': float(np.mean([_record_loss(r['p'], r['result']) for r in upto])),
            'book_accuracy': float(np.mean([r['book_correct'] for r in upto])) if priced else None,
            'book_log_loss': float(np.mean([_record_loss(r['book_p'], r['result']) for r in upto])) if priced else None,
        })
    return rows


def live_leader(live):
    if not live or live.get('book_log_loss') is None:
        return None
    model, book = live['log_loss'], live['book_log_loss']
    if book < model:
        return 'bookmakers'
    if model < book:
        return 'model'
    return 'level'


def calibration_bins(probabilities, results, bins=(0.0, 0.2, 0.35, 0.5, 0.65, 1.0)):
    codes = outcome_codes(results)
    rows = []
    edges = list(bins)
    flat_p = probabilities.reshape(-1)
    flat_hit = (np.arange(3)[None, :] == codes[:, None]).reshape(-1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (flat_p >= low) & (flat_p < high) if high < 1.0 else (flat_p >= low)
        if mask.sum() == 0:
            continue
        rows.append({'range': f"{low * 100:.0f}-{high * 100:.0f}%", 'predicted': float(flat_p[mask].mean()),
                     'observed': float(flat_hit[mask].mean()), 'count': int(mask.sum())})
    return rows


def elo_history(engineered, team, since_season):
    rows = engineered[(engineered['season'] >= since_season) &
                      ((engineered['home_team_name'] == team) | (engineered['away_team_name'] == team))]
    rows = rows.sort_values('match_date', kind='mergesort')
    rating = np.where(rows['home_team_name'] == team, rows['home_elo'], rows['away_elo'])
    return pd.DataFrame({'match_date': rows['match_date'].values, 'elo': rating, 'tier': rows['tier'].values})


def recent_results(matches, team, count=5):
    rows = matches[(matches['home_team_name'] == team) | (matches['away_team_name'] == team)]
    rows = rows.sort_values('match_date', kind='mergesort').tail(count)
    out = []
    for row in rows.itertuples():
        at_home = row.home_team_name == team
        scored = row.home_team_score if at_home else row.away_team_score
        conceded = row.away_team_score if at_home else row.home_team_score
        mark = 'W' if scored > conceded else ('D' if scored == conceded else 'L')
        out.append({'date': row.match_date, 'opponent': row.away_team_name if at_home else row.home_team_name,
                    'venue': 'H' if at_home else 'A', 'score': f"{int(scored)}-{int(conceded)}",
                    'mark': mark, 'division': row.division})
    return out
