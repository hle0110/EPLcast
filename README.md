# EPLcast

Predicts Premier League match outcomes and simulates future season standings using machine learning.

## Overview

EPLcast trains a classifier on recent English football results (2021-22 season through 2025-26) and uses it to simulate future Premier League seasons via Monte Carlo simulation. Older data (pre-2021) was dropped: the game has changed enough since then (transfers, tactics, VAR, squad turnover) that including it hurt more than it helped. Match data (results, shots, shots on target, corners, cards) is sourced directly from football-data.co.uk for all five seasons, across the top four English divisions, so promoted/relegated teams carry a consistent history.

Model: logistic regression trained on engineered features:

- Elo ratings, updated match by match across the top four English divisions, so promoted/relegated teams carry a real strength signal
- Rolling 7-match form (points per game, goals for, goals against)
- Rolling 7-match shots-on-target for/against, a less noisy signal of attacking and defensive quality than goals alone
- Head-to-head record between the two sides
- Whether each side was promoted from a lower division last season

On a held-out test of the two most recent completed seasons, this scores 51.8% match outcome accuracy (a more robust cross-validated estimate across three different train/test splits puts it at 53.4%), versus 45.9% for the original team-id + last-3-games model and a 41.7% always-predict-home-win baseline. For reference, published football prediction models rarely clear the mid-50s on this exact task, since match outcomes carry a large amount of irreducible randomness - a team can dominate on shots and still lose 1-0.

Season simulation runs 30 Monte Carlo trials of a real double round-robin schedule among the current 20 Premier League clubs, with Elo, form, shot quality and goal rates all evolving match by match. Scorelines are drawn from a Poisson goal model calibrated to each team's attack/defense strength. Output standings include title, top-4 and relegation probabilities alongside average points.

The whole pipeline (data load, evaluation, training, 30-season Monte Carlo simulation) runs in under 10 seconds on a laptop.

## Installation

```bash
git clone https://github.com/hle0110/EPLcast.git
cd EPLcast
pip install -r requirements.txt
```

Only four lightweight dependencies: pandas, numpy, scikit-learn, joblib.

## Usage

```bash
python main.py
```

This will:

1. Load `data/matches.csv` and engineer Elo/form/shots-on-target/head-to-head features
2. Report held-out accuracy versus the original model
3. Train the final model on all available Premier League history and save it to `models/epl_predictive_model.pkl`
4. Simulate the next 5 seasons and save results to `predictions/epl_next5years_matches.csv` and `predictions/epl_top10_next5years.csv`

## Files

- `features.py` - data loading and feature engineering (Elo, form, shots on target, head-to-head, promotion status)
- `models.py` - the logistic regression classifier
- `simulate.py` - round-robin scheduling and Monte Carlo season simulation
- `main.py` - orchestrates training, evaluation and simulation
