# EPLcast

Premier League match prediction and season projection.

**[Live table](https://hle0110.github.io/EPLcast/)**, updated automatically twice a week.

## What it does

Points already won are taken from real results, and every remaining fixture is simulated 5,000 times in its published order to project how the season finishes, with title, top four and relegation probabilities. Once a season ends the projection rolls forward to the next one.

The site has five sections:

- **Table**: live table, projected finish, title, top four and relegation chances, the title, top four and relegation races over the season, and the probability of every club finishing in every position.
- **Matches**: home, draw and away probabilities for the next round, and how each result would change the season for both clubs.
- **Clubs**: a page per club with its finishing position chances, likely points range, Elo history, recent results and every remaining fixture.
- **Accuracy**: season by season results against held-out matches, a comparison with bookmakers' closing odds, a calibration check, and a live record of predictions saved before kick-off.
- **Method**: how the model and simulation work, and what they cannot see.

The same numbers are published as JSON at `api/projection.json`, `api/fixtures.json` and `api/accuracy.json`.

The model is trained on every match in the top four English divisions since the 2021/22 season, using Elo ratings, rolling 9 match form, shots on target, last season's division and whether a club is in its first Premier League season after promotion. Seasons from 2017/18 to 2020/21 are loaded too, but only to warm up the ratings and form, never as training rows, so no club starts the training period on a blank rating.

## Accuracy

Current accuracy and log loss, with bookmakers' closing odds and an always-home-win baseline for comparison, are on the [Accuracy page](https://hle0110.github.io/EPLcast/accuracy.html). They are recalculated on every run over Premier League matches from 2023/24 onward, each season predicted by a model trained only on games played before it started, so they move as the season goes on. Published football models rarely clear the mid 50s on three way outcome prediction, since a team can dominate a match and still lose to a deflection.

Season projections are checked separately by replaying 2023/24 to 2025/26 from three points in each season. Clubs in their first Premier League season had been overrated all season long, by about 0.16 points a game, so the model now learns that effect directly.

## Staying current

A GitHub Actions workflow runs every Tuesday and Friday on GitHub's servers. Tuesday picks up the weekend round, Friday picks up midweek fixtures, which account for about a fifth of all matches across the four divisions. It downloads the latest results and closing odds from football data co uk and the fixture list from fixturedownload.com, retrains, rebuilds the projection and publishes the updated site. No local machine involved.

## Running locally

```bash
pip install -r requirements.txt
python update_data.py
python main.py
```

Needs pandas, numpy and scikit learn. A full run takes under a minute and writes the site to `docs/`, plus these files in `predictions/`:

| File | Contents |
|---|---|
| `epl_season_projection.csv` | Projected points and probabilities for every club |
| `position_probabilities.csv` | Chance of each club finishing in each position |
| `probability_history.csv` | Probabilities recorded whenever they move, for the race charts |
| `match_predictions.csv` | Every match prediction, saved before kick-off |

`data/matches.csv` holds results, shots, cards, corners, closing odds and, from 2026/27, expected goals. `data/fixtures.csv` holds the current fixture list. `python update_data.py --rebuild` downloads everything from scratch.

Tests run with `python -m pytest tests`, covering data integrity, projection maths, feature leakage, the fetch exit codes, and a check that the simulator builds exactly the same features the model was trained on. The scheduled workflow runs them too.

## Limitations

Promotion and relegation are not modelled for future seasons, so anything beyond the current one uses today's 20 clubs. Injuries, transfers and fixture congestion are not represented. If the fixture list cannot be fetched, the last saved copy is used, and without one the remaining fixtures are simulated in a random order and the Matches page stays empty.

Season rollover is not automated. Promotion and relegation for 2027-28 require a manual update of the team list: when the promoted clubs are known, check that the fixture source's spellings map to the names in the results data (`SCHEDULE_ALIASES` in `schedule.py`). If a spelling does not map, the scheduled run fails its data health check once the new season's clubs can be inferred from the results data, usually after the Championship's opening weekend and before the Premier League starts, and keeps failing until the alias is added. Before that point a misspelling cannot be detected.

Data from [football-data.co.uk](https://www.football-data.co.uk/) and [fixturedownload.com](https://fixturedownload.com/). Predictions are statistical estimates, not betting advice.

## License

MIT, see [LICENSE](LICENSE). This covers the code, not the match data.
