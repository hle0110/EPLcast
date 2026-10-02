import datetime
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import load_data, engineer_features, FEATURE_COLUMNS, TOP_DIVISION
from models import ScaledLogisticModel
from simulate import run_simulations, summarize_simulations, predict_fixtures, OUTCOME_ORDER
from schedule import load_schedule
from insights import (position_probabilities, points_ranges, upcoming_fixtures, remaining_for_team, match_stakes,
                      update_prediction_log, score_prediction_log, score_probabilities, bookmaker_probabilities,
                      calibration_bins, elo_history, recent_results, PREDICTION_LOG_PATH)
from site_builder import write_site

DATA_PATH = "data/matches.csv"
PRED_TABLE_PATH = "predictions/epl_season_projection.csv"
HISTORY_PATH = "predictions/probability_history.csv"
DOCS_DIR = "docs"
POSITION_PATH = "predictions/position_probabilities.csv"
N_FUTURE_SEASONS = 1
N_SIMULATIONS = 5000
MODEL_C = 0.8
TEAMS_PER_SEASON = 20
FIRST_TRAINING_SEASON = 2021
MIN_TRAINING_SEASONS = 2

def training_rows(df):
    return df[df['season'] >= FIRST_TRAINING_SEASON]

def rolling_origin_folds(top_flight):
    seasons = sorted(s for s in top_flight['season'].unique() if s >= FIRST_TRAINING_SEASON)
    folds = []
    for season in seasons[MIN_TRAINING_SEASONS:]:
        season_matches = top_flight[top_flight['season'] == season]
        if len(season_matches) < 20:
            continue
        folds.append((f"{season}-{str(season + 1)[-2:]}", season_matches['match_date'].min(), season_matches.index))
    return folds

def evaluate_model(df, top_flight):
    folds = rolling_origin_folds(top_flight)
    if not folds:
        print("Not enough seasons for a held-out evaluation, skipping it")
        return None

    seasons = []
    pooled_probs = []
    pooled_results = []
    print(f"\nRolling-origin evaluation ({len(folds)} held-out seasons, each trained only on matches before it started)")
    for name, cutoff_date, test_index in folds:
        test = top_flight.loc[test_index]
        train = training_rows(df[df['match_date'] < cutoff_date])
        model = ScaledLogisticModel(C=MODEL_C)
        model.fit(train[FEATURE_COLUMNS], train['result'])
        classes = list(model.classes_)
        proba = model.predict_proba(test[FEATURE_COLUMNS])[:, [classes.index(o) for o in OUTCOME_ORDER]]
        accuracy, loss = score_probabilities(proba, test['result'])
        row = {'season': name, 'matches': len(test), 'baseline': float((test['result'] == 'home team win').mean()),
               'accuracy': accuracy, 'log_loss': loss, 'book_accuracy': None, 'book_log_loss': None}
        if test[['odds_home', 'odds_draw', 'odds_away']].notna().all().all():
            row['book_accuracy'], row['book_log_loss'] = score_probabilities(bookmaker_probabilities(test), test['result'])
        seasons.append(row)
        pooled_probs.append(proba)
        pooled_results.append(test['result'])
        book = f"  bookmakers {row['book_accuracy']:.3f}" if row['book_accuracy'] is not None else ""
        print(f"  {name:8s} n={len(test):3d}  home-win baseline {row['baseline']:.3f}  model {accuracy:.3f}  log loss {loss:.3f}{book}")

    n = sum(r['matches'] for r in seasons)
    totals = {key: sum(r[key] * r['matches'] for r in seasons) / n for key in ['baseline', 'accuracy', 'log_loss']}
    print(f"\n  Weighted over {n:,} held-out matches")
    print(f"    Always-predict-home-win baseline : accuracy {totals['baseline']:.4f}")
    print(f"    Model                            : accuracy {totals['accuracy']:.4f}  log loss {totals['log_loss']:.4f}")
    summary = {'accuracy': totals['accuracy'], 'log_loss': totals['log_loss'], 'baseline': totals['baseline'],
               'matches': n, 'seasons': seasons, 'book_accuracy': None, 'book_log_loss': None}
    priced = [r for r in seasons if r['book_accuracy'] is not None]
    if len(priced) == len(seasons):
        summary['book_accuracy'] = sum(r['book_accuracy'] * r['matches'] for r in seasons) / n
        summary['book_log_loss'] = sum(r['book_log_loss'] * r['matches'] for r in seasons) / n
        print(f"    Bookmakers' closing odds         : accuracy {summary['book_accuracy']:.4f}  log loss {summary['book_log_loss']:.4f}")
    print()
    summary['calibration'] = calibration_bins(np.vstack(pooled_probs), pd.concat(pooled_results))
    return summary

def resolve_current_teams(df, top_flight, current_season):
    season_matches = top_flight[top_flight['season'] == current_season]
    appeared = set(season_matches['home_team_name']) | set(season_matches['away_team_name'])
    if len(appeared) == TEAMS_PER_SEASON:
        return sorted(appeared), season_matches, True

    lower_divisions = df[(df['season'] == current_season) & (df['tier'] > 1)]
    playing_below = set(lower_divisions['home_team_name']) | set(lower_divisions['away_team_name'])
    previous = df[df['season'] == current_season - 1]
    stayed_up = set(previous[previous['tier'] == 1]['home_team_name']) | set(previous[previous['tier'] == 1]['away_team_name'])
    came_up = set(previous[previous['tier'] == 2]['home_team_name']) | set(previous[previous['tier'] == 2]['away_team_name'])
    inferred = appeared | ((stayed_up | came_up) - playing_below)
    if len(inferred) == TEAMS_PER_SEASON:
        print(f"  only {len(appeared)} clubs have played so far, "
              f"inferred the full {TEAMS_PER_SEASON} from this season's lower divisions")
        return sorted(inferred), season_matches, True

    print(f"  cannot identify {TEAMS_PER_SEASON} clubs for this season yet "
          f"({len(appeared)} have played, {len(inferred)} inferred)")
    return sorted(stayed_up), season_matches, False


def actual_table(season_matches, teams):
    rows = []
    for team in teams:
        home = season_matches[season_matches['home_team_name'] == team]
        away = season_matches[season_matches['away_team_name'] == team]
        wins = int((home['home_team_score'] > home['away_team_score']).sum() + (away['away_team_score'] > away['home_team_score']).sum())
        draws = int((home['home_team_score'] == home['away_team_score']).sum() + (away['away_team_score'] == away['home_team_score']).sum())
        losses = int((home['home_team_score'] < home['away_team_score']).sum() + (away['away_team_score'] < away['home_team_score']).sum())
        goals_for = int(home['home_team_score'].sum() + away['away_team_score'].sum())
        goals_against = int(home['away_team_score'].sum() + away['home_team_score'].sum())
        rows.append({
            'Team': team,
            'Played': wins + draws + losses,
            'Points': wins * 3 + draws,
            'GoalDiff': goals_for - goals_against,
        })
    table = pd.DataFrame(rows).sort_values(['Points', 'GoalDiff'], ascending=False).reset_index(drop=True)
    table['Position'] = table.index + 1
    return table


def recent_form(season_matches, history, teams, window=5):
    form = {}
    pool = history
    for team in teams:
        played = pool[(pool['home_team_name'] == team) | (pool['away_team_name'] == team)]
        played = played.sort_values('match_date', kind='mergesort').tail(window)
        marks = []
        for row in played.itertuples():
            if row.home_team_name == team:
                scored, conceded = row.home_team_score, row.away_team_score
            else:
                scored, conceded = row.away_team_score, row.home_team_score
            marks.append('W' if scored > conceded else ('D' if scored == conceded else 'L'))
        form[team] = marks
    return form


def record_history(projection, season, matches_played, path=HISTORY_PATH):
    today = datetime.date.today().isoformat()
    snapshot = projection[projection['Season'] == season][['Team', 'Rank', 'Points', 'TitleProb', 'Top4Prob', 'RelegationProb']].copy()
    snapshot.insert(0, 'recorded_on', today)
    snapshot.insert(1, 'season', season)
    snapshot.insert(2, 'matches_played', matches_played)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        prior = existing[existing['season'] == season]
        if len(prior):
            latest = prior[prior['recorded_on'] == prior['recorded_on'].max()]
            same_progress = int(latest['matches_played'].iloc[0]) == matches_played
            previous_odds = latest.set_index('Team')['TitleProb'].round(4)
            current_odds = snapshot.set_index('Team')['TitleProb'].round(4)
            same_odds = previous_odds.reindex(current_odds.index).equals(current_odds)
            if same_progress and same_odds:
                return existing
        existing = existing[~((existing['recorded_on'] == today) & (existing['season'] == season))]
        snapshot = pd.concat([existing, snapshot], ignore_index=True)
    snapshot.to_csv(path, index=False)
    return snapshot


def club_details(df, results, season, teams, schedule, season_matches, model, elo_ratings):
    table = results['tables'][season]
    ranges = points_ranges(table, teams).set_index('Team')
    positions = position_probabilities(table, teams).set_index('Team')
    remaining = {}
    if season_matches is not None and schedule is not None:
        remaining = {team: remaining_for_team(schedule, season_matches, team) for team in teams}
    pairs = sorted({(home, away) for fixtures in remaining.values() for _, home, away in fixtures})
    probabilities = predict_fixtures(model, df, elo_ratings, season, teams, pairs)
    fixture_probs = {pair: p for pair, p in zip(pairs, probabilities)}
    details = {}
    for team in teams:
        details[team] = {
            'range': ranges.loc[team].to_dict(),
            'positions': positions.loc[team].values.tolist(),
            'elo': elo_history(df, team, season - 1),
            'recent': recent_results(df, team, 6),
            'remaining': [(kickoff, home, away, fixture_probs[(home, away)]) for kickoff, home, away in remaining.get(team, [])],
        }
    return details


def main():
    print("Starting Premier League Predictor")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: data file not found at {DATA_PATH}")
        print("Run 'python update_data.py --rebuild' to download the dataset first")
        return

    df = load_data(DATA_PATH)
    print(f"Loaded {len(df):,} matches from {DATA_PATH} ({df['match_date'].min().date()} to {df['match_date'].max().date()})")

    df, elo_ratings = engineer_features(df)
    top_flight = df[df['division'] == TOP_DIVISION]
    print(f"{len(top_flight):,} {TOP_DIVISION} matches, {len(df):,} matches in total across four divisions")

    metrics = evaluate_model(df, top_flight)

    train = training_rows(df)
    print(f"Training final model on {len(train):,} matches from {FIRST_TRAINING_SEASON}-{str(FIRST_TRAINING_SEASON + 1)[-2:]} onward, "
          f"earlier seasons only warm up the ratings")
    final_model = ScaledLogisticModel(C=MODEL_C)
    final_model.fit(train[FEATURE_COLUMNS], train['result'])

    current_season = int(top_flight['season'].max())
    season_label = f"{current_season}-{str(current_season + 1)[-2:]}"
    current_teams, season_matches, teams_known = resolve_current_teams(df, top_flight, current_season)
    played = len(season_matches)
    total_fixtures = TEAMS_PER_SEASON * (TEAMS_PER_SEASON - 1)
    project_current = teams_known and played < total_fixtures
    schedule = load_schedule(current_season, current_teams) if project_current else None

    if project_current:
        print(f"\n{season_label} is in progress: {played} of {total_fixtures} matches played")
        print(f"Simulating the remaining {total_fixtures - played} fixtures {N_SIMULATIONS:,} times, "
              f"then {N_FUTURE_SEASONS} further seasons")
        print("  using the published fixture order" if schedule is not None else "  fixture list unavailable, using a random fixture order")
    elif teams_known:
        print(f"\n{season_label} is complete, simulating {N_FUTURE_SEASONS} future seasons {N_SIMULATIONS:,} times")
    else:
        print(f"\n{season_label} has only just started, projecting full seasons with last season's clubs")

    results = run_simulations(
        final_model, df, current_season, current_teams, elo_ratings,
        n_future_seasons=N_FUTURE_SEASONS, n_simulations=N_SIMULATIONS,
        project_current=project_current, schedule=schedule,
    )
    projection = summarize_simulations(results, current_teams)

    os.makedirs("predictions", exist_ok=True)
    projection.to_csv(PRED_TABLE_PATH, index=False)

    headline_season = current_season if project_current else current_season + 1
    headline = projection[projection['Season'] == headline_season]
    label = f"{headline_season}-{str(headline_season + 1)[-2:]}"
    print(f"\nProjected final {label} table (average of {N_SIMULATIONS:,} simulations)")
    display = headline[['Rank', 'Team', 'Points', 'Wins', 'Draws', 'Losses', 'GoalDiff', 'TitleProb', 'Top4Prob', 'RelegationProb']]
    print(display.to_string(index=False))

    position_table = position_probabilities(results['tables'][headline_season], current_teams)
    position_table.insert(0, 'Season', headline_season)
    position_table.to_csv(POSITION_PATH, index=False)

    upcoming = []
    prediction_log = None
    if project_current:
        status_line = f"{played} of {total_fixtures} matches played. Remaining fixtures simulated {N_SIMULATIONS:,} times."
        standings = actual_table(season_matches, current_teams)
        form = recent_form(season_matches, df, current_teams)
        history_rows = record_history(projection, headline_season, played)
        now = pd.Timestamp.now(tz='UTC')
        upcoming = upcoming_fixtures(schedule, season_matches, now)
        pairs = [(home, away) for _, home, away, _ in upcoming]
        probabilities = predict_fixtures(final_model, df, elo_ratings, current_season, current_teams, pairs)
        stakes = match_stakes(results, current_season, pairs)
        upcoming = [{'kickoff': kickoff, 'home': home, 'away': away, 'round': rnd, 'p': p, 'stakes': stake}
                    for (kickoff, home, away, rnd), p, stake in zip(upcoming, probabilities, stakes)]
        log_rows = [{'predicted_on': datetime.date.today().isoformat(), 'season': current_season,
                     'kickoff_utc': match['kickoff'].strftime('%Y-%m-%d %H:%M'), 'home_team_name': match['home'],
                     'away_team_name': match['away'], 'p_home': round(float(match['p'][2]), 4),
                     'p_draw': round(float(match['p'][1]), 4), 'p_away': round(float(match['p'][0]), 4)} for match in upcoming]
        prediction_log = update_prediction_log(log_rows, datetime.date.today().isoformat(), PREDICTION_LOG_PATH)
    elif teams_known:
        status_line = f"Season complete. Following seasons simulated {N_SIMULATIONS:,} times."
        standings = None
        form = None
        history_rows = None
    else:
        status_line = (f"{season_label} has only just begun and its line-up is not confirmed in the data yet, "
                       f"so this shows the season after.")
        standings = None
        form = None
        history_rows = None
    if prediction_log is None and os.path.exists(PREDICTION_LOG_PATH):
        prediction_log = pd.read_csv(PREDICTION_LOG_PATH)

    clubs = club_details(df, results, headline_season, current_teams, schedule,
                         season_matches if project_current else None, final_model, elo_ratings)
    accuracy_text = f"{metrics['accuracy'] * 100:.1f}%" if metrics else "n/a"
    meta = {
        'status_line': status_line,
        'played': played if project_current else 0,
        'total': total_fixtures,
        'simulations': N_SIMULATIONS,
        'accuracy': accuracy_text,
        'last_match_date': top_flight['match_date'].max(),
        'standings': standings,
        'form': form,
        'history': history_rows,
        'positions': position_table,
        'upcoming': upcoming,
        'clubs': clubs,
        'metrics': metrics,
        'live_score': score_prediction_log(prediction_log, df),
        'generated': datetime.date.today().isoformat(),
    }
    write_site(projection, headline_season, meta, DOCS_DIR)

    print(f"\nSaved projected standings to {PRED_TABLE_PATH}")
    print(f"Saved finishing position probabilities to {POSITION_PATH}")
    if history_rows is not None:
        print(f"Saved probability history to {HISTORY_PATH}")
    if upcoming:
        print(f"Logged {len(upcoming)} upcoming match predictions to {PREDICTION_LOG_PATH}")
    print(f"Saved the site to {DOCS_DIR}/")
    print("All done")

if __name__ == "__main__":
    main()
