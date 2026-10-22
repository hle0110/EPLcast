import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features import load_data, engineer_features, FEATURE_COLUMNS, TOP_DIVISION
from models import ScaledLogisticModel
from simulate import OUTCOME_ORDER
from insights import calibration_bins, outcome_codes
from main import MODEL_C, FIRST_TRAINING_SEASON, training_rows, rolling_origin_folds

DATA_PATH = os.path.join(ROOT, 'data', 'matches.csv')
FIRST_FIT_SEASON = FIRST_TRAINING_SEASON + 1
GRID = np.round(np.arange(0.70, 1.3001, 0.01), 2)
DRAW_BINS = (0.0, 0.2, 0.25, 0.3, 1.0)
MAX_SEASON_WORSENING = 0.005


def temperature(p, t):
    scaled = np.exp(np.log(np.clip(p, 1e-12, 1.0)) / t)
    return scaled / scaled.sum(axis=1, keepdims=True)


def draw_adjust(p, d):
    adjusted = p.copy()
    adjusted[:, 1] *= d
    return adjusted / adjusted.sum(axis=1, keepdims=True)


VARIANTS = {
    'temperature': (lambda p, a, b: temperature(p, a), lambda: [(t, 1.0) for t in GRID]),
    'draw factor': (lambda p, a, b: draw_adjust(p, b), lambda: [(1.0, d) for d in GRID]),
    'temperature + draw': (lambda p, a, b: draw_adjust(temperature(p, a), b), lambda: [(t, d) for t in GRID for d in GRID]),
}


def log_loss(p, codes):
    return float(-np.log(np.clip(p[np.arange(len(codes)), codes], 1e-15, 1.0)).mean())


def season_predictions(df, top, season):
    test = top[top['season'] == season]
    train = training_rows(df[df['match_date'] < test['match_date'].min()])
    model = ScaledLogisticModel(C=MODEL_C)
    model.fit(train[FEATURE_COLUMNS], train['result'])
    classes = list(model.classes_)
    return model.predict_proba(test[FEATURE_COLUMNS])[:, [classes.index(o) for o in OUTCOME_ORDER]], test


def fit(apply, candidates, p, codes):
    return min(candidates(), key=lambda params: log_loss(apply(p, *params), codes))


def draw_table(p, results):
    codes = outcome_codes(results)
    draw_p = p[:, 1]
    hit = codes == 1
    rows = []
    for low, high in zip(DRAW_BINS[:-1], DRAW_BINS[1:]):
        mask = (draw_p >= low) & (draw_p < high)
        if mask.any():
            rows.append(f"    {low * 100:3.0f}-{high * 100:3.0f}%  predicted {draw_p[mask].mean():.3f}  "
                        f"observed {hit[mask].mean():.3f}  n={mask.sum()}")
    rows.append(f"    all       predicted {draw_p.mean():.3f}  observed {hit.mean():.3f}  n={len(hit)}")
    return "\n".join(rows)


def print_calibration(label, p, results):
    print(f"\n  {label}: all outcomes")
    for row in calibration_bins(p, results):
        print(f"    {row['range']:>8s}  predicted {row['predicted']:.3f}  observed {row['observed']:.3f}  n={row['count']}")
    print(f"  {label}: draws only")
    print(draw_table(p, results))


def main():
    df, _ = engineer_features(load_data(DATA_PATH))
    top = df[df['division'] == TOP_DIVISION]
    eval_seasons = [int(top.loc[index, 'season'].iloc[0]) for _, _, index in rolling_origin_folds(top)]
    all_seasons = list(range(FIRST_FIT_SEASON, max(eval_seasons) + 1))
    preds = {s: season_predictions(df, top, s) for s in all_seasons}
    codes = {s: outcome_codes(preds[s][1]['result']) for s in all_seasons}

    results = {'baseline': {}}
    params = {}
    for name, (apply, candidates) in VARIANTS.items():
        results[name] = {}
        params[name] = {}
        for season in eval_seasons:
            fit_seasons = [s for s in all_seasons if s < season]
            assert season not in fit_seasons
            fit_p = np.vstack([preds[s][0] for s in fit_seasons])
            fit_codes = np.concatenate([codes[s] for s in fit_seasons])
            best = fit(apply, candidates, fit_p, fit_codes)
            params[name][season] = (best, fit_seasons)
            results[name][season] = apply(preds[season][0], *best)
    for season in eval_seasons:
        results['baseline'][season] = preds[season][0]

    print("Rolling-origin evaluation, calibration fitted only on earlier held-out seasons")
    for name in VARIANTS:
        print(f"  {name}: " + ", ".join(f"{s}: T={p[0][0]:.2f} d={p[0][1]:.2f} fit on {p[1]}" for s, p in params[name].items()))

    n = {s: len(codes[s]) for s in eval_seasons}
    total = sum(n.values())
    header = "  ".join(f"{s}-{str(s + 1)[-2:]}" for s in eval_seasons)
    print(f"\n  {'variant':22s}  pooled   {header}   worst change  decision")
    base = {s: log_loss(results['baseline'][s], codes[s]) for s in eval_seasons}
    for name, by_season in results.items():
        losses = {s: log_loss(by_season[s], codes[s]) for s in eval_seasons}
        pooled = sum(losses[s] * n[s] for s in eval_seasons) / total
        worst = max(losses[s] - base[s] for s in eval_seasons)
        base_pooled = sum(base[s] * n[s] for s in eval_seasons) / total
        if name == 'baseline':
            decision = ''
        else:
            decision = 'adopt' if pooled < base_pooled and worst <= MAX_SEASON_WORSENING else 'reject'
        print(f"  {name:22s}  {pooled:.4f}  " + "  ".join(f"{losses[s]:.4f}" for s in eval_seasons)
              + f"   {worst:+.4f}       {decision}")

    pooled_results = pd.concat([preds[s][1]['result'] for s in eval_seasons])
    for name, by_season in results.items():
        p = np.vstack([by_season[s] for s in eval_seasons])
        accuracy = float((p.argmax(axis=1) == outcome_codes(pooled_results)).mean())
        print_calibration(f"{name} (accuracy {accuracy * 100:.1f}%)", p, pooled_results)


if __name__ == "__main__":
    main()
