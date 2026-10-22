import os
import sys
import time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features import load_data, engineer_features, compute_elo, FEATURE_COLUMNS, TOP_DIVISION
from models import ScaledLogisticModel, DrawCalibratedModel
from simulate import run_simulations, table_from_results, finishing_positions, TABLE_KEYS
from main import MODEL_C, training_rows, held_out_probabilities, draw_factor_before

DATA_PATH = os.path.join(ROOT, 'data', 'matches.csv')
SEASONS = [2023, 2024, 2025]
CHECKPOINTS = [0, 100, 200]
SIGMAS = [0.0, 15.0, 30.0, 45.0, 60.0]
N_SIMULATIONS = 2000
SEEDS = [42, 7]


def actual_positions(season_matches, teams):
    table = table_from_results(season_matches, teams)
    arrays = {key: np.array([[table[t][key] for t in teams]]) for key in TABLE_KEYS}
    return finishing_positions(arrays)[0], arrays['Points'][0]


def replay_points(df, calibrated=False):
    top = df[df['division'] == TOP_DIVISION]
    held_out = held_out_probabilities(df, top) if calibrated else None
    points = []
    for season in SEASONS:
        season_matches = top[top['season'] == season].sort_values('match_date', kind='mergesort')
        teams = sorted(set(season_matches['home_team_name']) | set(season_matches['away_team_name']))
        model = ScaledLogisticModel(C=MODEL_C)
        train = training_rows(df[df['match_date'] < season_matches['match_date'].min()])
        model.fit(train[FEATURE_COLUMNS], train['result'])
        if calibrated:
            model = DrawCalibratedModel(model, draw_factor_before(held_out, season))
        final_positions, final_points = actual_positions(season_matches, teams)
        for checkpoint in CHECKPOINTS:
            cutoff = season_matches['match_date'].iloc[checkpoint]
            history = df[df['match_date'] < cutoff]
            _, ratings = compute_elo(history)
            remaining = season_matches[season_matches['match_date'] >= cutoff]
            schedule = pd.DataFrame({'kickoff_utc': remaining['match_date'].values,
                                     'home_team_name': remaining['home_team_name'].values,
                                     'away_team_name': remaining['away_team_name'].values})
            points.append({'season': season, 'checkpoint': checkpoint, 'played': int((season_matches['match_date'] < cutoff).sum()),
                           'model': model, 'history': history, 'ratings': ratings, 'teams': teams, 'schedule': schedule,
                           'final_positions': final_positions, 'final_points': final_points})
    return points


def score(point, sigma, seed):
    results = run_simulations(point['model'], point['history'], point['season'], point['teams'], point['ratings'],
                              n_future_seasons=0, n_simulations=N_SIMULATIONS, project_current=True, seed=seed,
                              schedule=point['schedule'], elo_sigma=sigma)
    table = results['tables'][point['season']]
    positions = finishing_positions(table)
    n_teams = len(point['teams'])
    actual = point['final_positions']
    position_probs = np.stack([(positions == k).mean(axis=0) for k in range(1, n_teams + 1)], axis=1)
    onehot = np.eye(n_teams)[actual - 1]
    out = {'position': float(((position_probs - onehot) ** 2).sum(axis=1).mean())}
    for name, hit in [('title', positions == 1), ('top4', positions <= 4), ('relegation', positions >= n_teams - 2)]:
        prob = hit.mean(axis=0)
        if name == 'title':
            truth = actual == 1
        elif name == 'top4':
            truth = actual <= 4
        else:
            truth = actual >= n_teams - 2
        out[name] = float(((prob - truth) ** 2).mean())
    out['points_rmse'] = float(np.sqrt(((table['Points'].mean(axis=0) - point['final_points']) ** 2).mean()))
    return out


def main():
    calibrated = '--calibrated' in sys.argv
    sigmas = [0.0] if calibrated else SIGMAS
    df, _ = engineer_features(load_data(DATA_PATH))
    points = replay_points(df, calibrated)
    print(f"{'Draw-calibrated model. ' if calibrated else ''}Season replay: {len(points)} points ({SEASONS[0]}-{SEASONS[-1]}, checkpoints {CHECKPOINTS}), "
          f"{N_SIMULATIONS} simulations, seeds {SEEDS}")
    rows = []
    for sigma in sigmas:
        start = time.time()
        for seed in SEEDS:
            for point in points:
                row = score(point, sigma, seed)
                row.update({'sigma': sigma, 'seed': seed, 'season': point['season'], 'checkpoint': point['checkpoint']})
                rows.append(row)
        print(f"  sigma {sigma:4.0f} done in {time.time() - start:5.1f}s")
    frame = pd.DataFrame(rows)
    metrics = ['position', 'title', 'top4', 'relegation', 'points_rmse']
    print("\nMean over replay points and seeds")
    print(frame.groupby('sigma')[metrics].mean().round(4).to_string())
    print("\nBy seed")
    print(frame.groupby(['sigma', 'seed'])[metrics].mean().round(4).to_string())
    print("\nPosition Brier by checkpoint")
    print(frame.groupby(['sigma', 'checkpoint'])['position'].mean().unstack().round(4).to_string())
    print("\nPosition Brier by season")
    print(frame.groupby(['sigma', 'season'])['position'].mean().unstack().round(4).to_string())


if __name__ == "__main__":
    main()
