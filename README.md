# EPLcast

Premier League match prediction and season projection.

**[Live table](https://hle0110.github.io/EPLcast/)**, updated automatically twice a week.

## What it does

Points already won are taken from real results, and every remaining fixture is simulated 40 times to project how the season finishes, with title, top four and relegation probabilities. The page also shows the live table, recent form and how the title race has shifted over the season. Once a season ends the projection rolls forward to the next one.

The model is trained on every match in the top four English divisions since the 2021/22 season, using Elo ratings, rolling 9 match form, shots on target, head to head records and last season's division.

## Accuracy

| | Accuracy |
|---|---|
| Always predict a home win | 42.6% |
| Original model | 45.6% |
| Current model | **53.5%** |

Measured over 1,730 held out matches, training only on games played before each test window. Published football models rarely clear the mid 50s on three way outcome prediction, since a team can dominate a match and still lose to a deflection.

## Staying current

A GitHub Actions workflow runs every Tuesday and Friday on GitHub's servers. Tuesday picks up the weekend round, Friday picks up midweek fixtures, which account for about a fifth of all matches across the four divisions. It downloads the latest results from football data co uk, retrains, rebuilds the projection and publishes the updated page. No local machine involved.

## Running locally

```bash
pip install -r requirements.txt
python update_data.py
python main.py
```

Needs pandas, numpy, scikit learn and joblib. A full run takes about 15 seconds and writes `docs/index.html`, `predictions/epl_season_projection.csv` and `predictions/probability_history.csv`, which records the probabilities whenever new matches have been played, so the title race chart can be drawn.

Tests run with `python -m pytest tests`, covering data integrity, projection maths, feature leakage and the fetch exit codes. The scheduled workflow runs them too.

## Limitations

Promotion and relegation are not modelled for future seasons, so anything beyond the current one uses today's 20 clubs. Injuries, transfers and fixture congestion are not represented.

Data from [football-data.co.uk](https://www.football-data.co.uk/). Predictions are statistical estimates, not betting advice.

## License

MIT, see [LICENSE](LICENSE). This covers the code, not the match data.
