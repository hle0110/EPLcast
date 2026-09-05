# EPLcast

Predicts Premier League match outcomes, projects how the current season will finish, and simulates seasons beyond it.

## What it does

EPLcast keeps a dataset of every match in the top four English divisions since the 2021-22 season, including the season currently in progress. It trains a classifier on that history and runs Monte Carlo simulations of every remaining fixture to produce a projected final table with title, top-four and relegation probabilities.

When a season is underway, the projection starts from the real table - points already won are banked, and only the fixtures still to be played are simulated. Once a season finishes, it rolls forward and projects the following ones instead.

## Accuracy

Measured with rolling-origin validation: for each held-out window the model is trained only on matches played before that window started, which is how it would actually be used.

| | Accuracy | Log loss |
|---|---|---|
| Always predict a home win | 42.6% | - |
| Original model (team id + last-3 form) | 45.6% | 1.177 |
| Current model | 53.5% | 0.980 |

Evaluated over 1,730 held-out Premier League matches across 7 windows (full seasons and second halves, 2023-24 onward). Windows where fewer than two seasons of training data were available are excluded, since a model trained on a single season is far noisier than the one actually shipped.

For context, published football models rarely clear the mid-50s on three-way match outcome prediction. A team can dominate a match on every underlying metric and still lose to a deflection, and that irreducible randomness is a large share of what is left.

## How it works

Features, all computed only from matches played before the one being predicted:

- Elo rating difference, updated match by match across all four divisions so promoted and relegated teams carry a real strength signal
- Rolling 9-match form: points per game, goals for, goals against
- Rolling 9-match shots on target for and against, a less noisy quality signal than goals alone
- Head-to-head record between the two sides
- Which division each side played in last season

Model: multinomial logistic regression, trained on all four divisions. Gradient boosting, random forests and various blends were tested against the same validation protocol and all scored worse, so the simpler model shipped.

Simulation: each remaining fixture is drawn from the model's predicted probabilities, scorelines come from a Poisson goal model calibrated to each team's attack and defense strength, and Elo, form and shot rates all update match by match as the simulated season unfolds.

## Installation

```bash
git clone https://github.com/hle0110/EPLcast.git
cd EPLcast
pip install -r requirements.txt
```

Four dependencies: pandas, numpy, scikit-learn, joblib.

## Usage

```bash
python update_data.py   # pull in results from matches played since the last run
python main.py          # retrain, re-evaluate and re-project
```

`main.py` prints the accuracy comparison above, saves the trained model to `models/`, and writes:

- `predictions/epl_season_projection.csv` - projected final table for the current season and the seasons after it, with title, top-four and relegation probabilities
- `predictions/epl_simulated_matches.csv` - a sample simulated run, match by match

`update_data.py` works out which season is current from today's date, downloads the latest results for all four divisions from football-data.co.uk, and merges them in. It replaces the current season's rows wholesale each time, so running it twice changes nothing. It exits 0 when new matches were added and 1 when there was nothing new, which makes it easy to script. `python update_data.py --rebuild` rebuilds the whole dataset from source.

## Files

- `features.py` - feature engineering (Elo, form, shots on target, head-to-head, promotion status)
- `models.py` - the classifier
- `simulate.py` - fixture scheduling and Monte Carlo simulation, including rest-of-season projection
- `main.py` - evaluation, training and projection pipeline
- `update_data.py` - fetches and merges new results
- `team_name_map.py` - maps source team abbreviations to full club names

## Limitations

Promotion and relegation are not modelled for future seasons: seasons beyond the current one are simulated with the current 20 clubs. Injuries, transfers, fixture congestion and European commitments are not represented. The dataset starts at 2021-22 deliberately, as older football differs enough that including it measurably hurt results.
