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


def score_prediction_log(log, matches):
    if log is None or len(log) == 0:
        return None
    top = matches[matches['division'] == 'Premier League']
    joined = log.merge(top, on=['season', 'home_team_name', 'away_team_name'], how='inner')
    if len(joined) == 0:
        return None
    model_probs = joined[['p_away', 'p_draw', 'p_home']].astype(float).values
    accuracy, loss = score_probabilities(model_probs, joined['result'])
    scored = {'matches': len(joined), 'accuracy': accuracy, 'log_loss': loss,
              'first': str(joined['predicted_on'].min()), 'book_accuracy': None, 'book_log_loss': None}
    with_odds = joined.dropna(subset=['odds_home', 'odds_draw', 'odds_away'])
    if len(with_odds) == len(joined):
        scored['book_accuracy'], scored['book_log_loss'] = score_probabilities(bookmaker_probabilities(with_odds), with_odds['result'])
    return scored


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
