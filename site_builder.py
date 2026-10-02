import glob
import html
import json
import math
import os
import numpy as np
import pandas as pd
from dashboard import (build_page, page_shell, club_link, club_slug, _pct, _season_label, _format_date,
                       _form_cell, LINE_COLOURS, SITE_URL)

try:
    from zoneinfo import ZoneInfo
    LOCAL_ZONE = ZoneInfo('Europe/London')
    LOCAL_LABEL = 'UK time'
except Exception:
    LOCAL_ZONE = None
    LOCAL_LABEL = 'UTC'


def local_time(timestamp):
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize('UTC')
    if LOCAL_ZONE is not None:
        stamp = stamp.tz_convert(LOCAL_ZONE)
    return stamp.strftime('%a %-d %b, %H:%M')


def _probability_bar(p_home, p_draw, p_away):
    return (f'<div class="prob3"><b style="width:{p_home * 100:.1f}%;background:var(--win)"></b>'
            f'<b style="width:{p_draw * 100:.1f}%;background:var(--draw)"></b>'
            f'<b style="width:{p_away * 100:.1f}%;background:var(--loss)"></b></div>')


def _stake_line(team, stake, home, away):
    if not stake:
        return ''
    is_home = team == home
    win = stake['home_win'] if is_home else stake['away_win']
    loss = stake['away_win'] if is_home else stake['home_win']
    parts = []
    for label, value in (('win', win), ('draw', stake['draw']), ('lose', loss)):
        parts.append(f"{label} {value * 100:.0f}%" if value is not None else f"{label} n/a")
    return (f'<div class="stake">{html.escape(team)}\u2019s chance to {stake["label"]}: now {stake["overall"] * 100:.0f}%, '
            f'if they {", ".join(parts)}</div>')


def _live_summary(live):
    if not live:
        return '<p class="muted">Live tracking has started. Scores appear here once the first predicted matches are played.</p>'
    text = (f'<p>Since {html.escape(live["first"])}, {live["matches"]} matches were predicted before kick-off. '
            f'Model: {live["accuracy"] * 100:.1f}% correct, log loss {live["log_loss"]:.3f}.')
    if live.get('book_accuracy') is not None:
        text += f' Bookmakers on the same matches: {live["book_accuracy"] * 100:.1f}%, log loss {live["book_log_loss"]:.3f}.'
    return text + '</p>'


def build_matches_page(meta):
    upcoming = meta.get('upcoming') or []
    if upcoming:
        rows = []
        for match in upcoming:
            p_away, p_draw, p_home = (float(v) for v in match['p'])
            stakes = match.get('stakes') or {}
            rows.append(
                f'<tr><td class="team">{local_time(match["kickoff"])}</td>'
                f'<td class="team">{club_link(match["home"])} v {club_link(match["away"])}'
                f'{_probability_bar(p_home, p_draw, p_away)}'
                f'{_stake_line(match["home"], stakes.get(match["home"]), match["home"], match["away"])}'
                f'{_stake_line(match["away"], stakes.get(match["away"]), match["home"], match["away"])}</td>'
                f'<td>{p_home * 100:.0f}%</td><td>{p_draw * 100:.0f}%</td><td>{p_away * 100:.0f}%</td></tr>')
        heading = f'Round {upcoming[0]["round"]}'
        table = (f'<div class="scroll"><table><thead><tr><th class="team">Kick-off ({LOCAL_LABEL})</th><th class="team">Match</th>'
                 f'<th>Home</th><th>Draw</th><th>Away</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
                 f'<p class="legend">Bars show home win, draw and away win. The lines under each match show how the '
                 f'result moves the most affected season outcome for each club, measured across every simulation.</p>')
    else:
        heading = 'Next round'
        table = '<p class="muted">No upcoming fixtures are published yet.</p>'
    body = (f'<h1>{heading}</h1><p class="sub">Match probabilities from the current model, and what each result '
            f'would mean for the season.</p>{table}<h2>Live track record</h2>{_live_summary(meta.get("live_score"))}'
            f'<p class="legend">Every prediction is saved before kick-off in '
            f'<a href="https://github.com/hle0110/EPLcast/blob/main/predictions/match_predictions.csv">match_predictions.csv</a> '
            f'and committed to the public repository, so each one has a dated history.</p>')
    description = 'Premier League match predictions for the next round, with what each result means for the title, top four and relegation.'
    return page_shell('EPLcast - next round predictions', description, body, 'matches.html')


def _number(value, digits=3):
    return f'{value:.{digits}f}' if value is not None else '-'


def _percent(value):
    return f'{value * 100:.1f}%' if value is not None else '-'


def build_accuracy_page(meta):
    metrics = meta.get('metrics')
    if not metrics:
        body = '<h1>Accuracy</h1><p class="muted">Not enough completed seasons to evaluate yet.</p>'
        return page_shell('EPLcast - accuracy', 'How accurate EPLcast is.', body, 'accuracy.html')
    cards = (f'<div class="cards"><div class="card"><div class="label">Model accuracy</div><div class="value">{_percent(metrics["accuracy"])}</div></div>'
             f'<div class="card"><div class="label">Bookmakers</div><div class="value">{_percent(metrics.get("book_accuracy"))}</div></div>'
             f'<div class="card"><div class="label">Always home win</div><div class="value">{_percent(metrics["baseline"])}</div></div>'
             f'<div class="card"><div class="label">Matches tested</div><div class="value">{metrics["matches"]:,}</div></div></div>')
    rows = []
    for row in metrics['seasons']:
        rows.append(f'<tr><td class="team">{html.escape(row["season"])}</td><td>{row["matches"]}</td>'
                    f'<td>{_percent(row["baseline"])}</td><td class="group">{_percent(row["accuracy"])}</td>'
                    f'<td>{_number(row["log_loss"])}</td><td class="group">{_percent(row["book_accuracy"])}</td>'
                    f'<td>{_number(row["book_log_loss"])}</td></tr>')
    rows.append(f'<tr><td class="team"><strong>All</strong></td><td>{metrics["matches"]:,}</td><td>{_percent(metrics["baseline"])}</td>'
                f'<td class="group"><strong>{_percent(metrics["accuracy"])}</strong></td><td><strong>{_number(metrics["log_loss"])}</strong></td>'
                f'<td class="group">{_percent(metrics.get("book_accuracy"))}</td><td>{_number(metrics.get("book_log_loss"))}</td></tr>')
    season_table = ('<div class="scroll"><table><thead><tr><th class="team">Season</th><th>Matches</th><th>Home win</th>'
                    '<th class="group">Model</th><th>Log loss</th><th class="group">Bookmakers</th><th>Log loss</th></tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table></div>')
    calibration_rows = ''.join(
        f'<tr><td class="team">{html.escape(c["range"])}</td><td>{c["predicted"] * 100:.1f}%</td>'
        f'<td>{c["observed"] * 100:.1f}%</td><td>{c["count"]:,}</td></tr>' for c in metrics.get('calibration', []))
    calibration = ('<h2>Calibration</h2><div class="scroll"><table><thead><tr><th class="team">Predicted chance</th><th>Average predicted</th>'
                   f'<th>Actually happened</th><th>Outcomes</th></tr></thead><tbody>{calibration_rows}</tbody></table></div>'
                   '<p class="legend">Every home win, draw and away win probability from the held-out seasons, grouped by size. '
                   'When the two middle columns are close, a 40% forecast really does come true about 40% of the time.</p>')
    body = (f'<h1>Accuracy</h1><p class="sub">Each season is predicted by a model trained only on matches played before it '
            f'started, so none of these numbers use hindsight.</p>{cards}{season_table}'
            f'<p class="legend">Accuracy counts how often the most likely result happened. Log loss rewards honest probabilities '
            f'and punishes confident misses; lower is better, and guessing a third for everything scores 1.099. Bookmaker figures '
            f'use the average closing odds with the margin removed. They are the benchmark, not a target the model is tuned to.</p>'
            f'{calibration}<h2>Live track record</h2>{_live_summary(meta.get("live_score"))}')
    description = f'EPLcast accuracy: {_percent(metrics["accuracy"])} over {metrics["matches"]:,} held-out Premier League matches, compared with bookmakers.'
    return page_shell('EPLcast - accuracy', description, body, 'accuracy.html')


def build_methodology_page(meta):
    metrics = meta.get('metrics') or {}
    accuracy = _percent(metrics.get('accuracy'))
    body = f"""<h1>How it works</h1>
<div class="prose">
<h2>Data</h2>
<p>Results, shots on target and closing odds for the top four English divisions come from football-data.co.uk, and fixture
dates from fixturedownload.com. Seasons from 2017-18 to 2020-21 only warm up the ratings; the model learns from 2021-22 onward.</p>
<h2>The match model</h2>
<p>A multinomial logistic regression gives the probability of a home win, draw and away win from fifteen inputs: the gap in
Elo rating, each club's average points, goals and shots on target over its last nine matches, the division each club played
in last season, and whether a club is in its first Premier League season after promotion. Promoted clubs had been overrated
by about 0.16 points a game before that last input was added.</p>
<h2>The season simulation</h2>
<p>Points already won are banked. Every remaining fixture is played in its published order {meta['simulations']:,} times. Each
simulated result updates the ratings and form that the following matches use, so a bad run carries forward as it does in
reality. Title, top four and relegation chances are the share of simulations in which each happens.</p>
<h2>How good is it</h2>
<p>Over held-out seasons the model picks the right result {accuracy} of the time. Bookmakers manage a little more, because
they also see injuries, line-ups and team news. Three-way football results are hard to predict: published models rarely clear
the mid 50s. See <a href="accuracy.html">Accuracy</a> for every number.</p>
<h2>What it cannot see</h2>
<ul><li>Injuries, suspensions, transfers and managerial changes, until they show up in results.</li>
<li>Fixture congestion and cup or European commitments.</li>
<li>Points deductions, until they appear in the official results.</li>
<li>Expected goals, which the data only includes from 2026-27; it will be added once there is enough history to test it.</li></ul>
<h2>Updates</h2>
<p>The site rebuilds itself every Tuesday and Friday. Code and data are open at
<a href="https://github.com/hle0110/EPLcast">github.com/hle0110/EPLcast</a>.</p>
</div>"""
    return page_shell('EPLcast - how it works', 'How EPLcast predicts Premier League matches and simulates the season.', body, 'methodology.html')


def _elo_chart(frame):
    if frame is None or len(frame) < 2:
        return ''
    width, height = 720, 220
    pad_left, pad_right, pad_top, pad_bottom = 46, 14, 14, 30
    values = frame['elo'].astype(float).values
    low = math.floor((values.min() - 20) / 50) * 50
    high = math.ceil((values.max() + 20) / 50) * 50
    span = max(high - low, 1)
    count = len(values)

    def x_of(i):
        return pad_left + (width - pad_left - pad_right) * i / (count - 1)

    def y_of(v):
        return pad_top + (height - pad_top - pad_bottom) * (1 - (v - low) / span)

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img">']
    step = 50 if span <= 300 else 100
    tick = low
    while tick <= high:
        y = y_of(tick)
        parts.append(f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" stroke="#262b35" stroke-width="1"/>')
        parts.append(f'<text x="{pad_left - 8}" y="{y + 4:.1f}" fill="#9aa3b2" font-size="11" text-anchor="end">{tick}</text>')
        tick += step
    dates = pd.to_datetime(frame['match_date'])
    for i in (0, count - 1):
        anchor = 'start' if i == 0 else 'end'
        parts.append(f'<text x="{x_of(i):.1f}" y="{height - 8}" fill="#9aa3b2" font-size="11" text-anchor="{anchor}">'
                     f'{dates.iloc[i].strftime("%b %Y")}</text>')
    points = ' '.join(f'{x_of(i):.1f},{y_of(v):.1f}' for i, v in enumerate(values))
    parts.append(f'<polyline fill="none" stroke="{LINE_COLOURS[0]}" stroke-width="2.5" stroke-linejoin="round" points="{points}"/>')
    parts.append('</svg>')
    return ('<h2>Elo rating</h2><div class="chart">' + ''.join(parts) + '</div>'
            '<p class="legend">Rating before each match since the start of last season, in all competitions in the data. '
            'Ratings from a lower division are not directly comparable with Premier League ones.</p>')


def _position_bars(positions):
    if not positions:
        return ''
    count = len(positions)
    peak = max(max(positions), 1e-9)
    bars = []
    for i, value in enumerate(positions, start=1):
        css = '' if i <= 4 else ('low' if i > count - 3 else 'mid')
        bars.append(f'<div class="{css}" style="height:{value / peak * 100:.1f}%" title="{i}: {value * 100:.1f}%"></div>')
    labels = ''.join(f'<span>{i}</span>' for i in range(1, count + 1))
    return ('<h2>Finishing position</h2><div class="posbar">' + ''.join(bars) + f'</div><div class="poslabels">{labels}</div>'
            '<p class="legend">How often the club finishes in each position across all simulations.</p>')


def build_club_page(team, record, standing, detail, form, season):
    cards = []
    if standing is not None:
        cards.append(('Position now', f'{int(standing.Position)}'))
        cards.append(('Points now', f'{int(standing.Points)} from {int(standing.Played)}'))
    cards.append(('Projected points', f'{record["Points"]:.1f}'))
    if detail.get('range'):
        cards.append(('Likely range', f'{detail["range"]["P10"]:.0f} to {detail["range"]["P90"]:.0f}'))
    cards.append(('Title', _pct(record['TitleProb'])))
    cards.append(('Top 4', _pct(record['Top4Prob'])))
    cards.append(('Relegated', _pct(record['RelegationProb'])))
    card_html = '<div class="cards">' + ''.join(
        f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>' for label, value in cards) + '</div>'

    recent_rows = ''.join(
        f'<tr><td class="team">{_format_date(pd.Timestamp(r["date"]))}</td><td class="team">{r["venue"]} v '
        f'{html.escape(str(r["opponent"]))}</td><td class="hide">{html.escape(str(r["division"]))}</td>'
        f'<td>{r["score"]}</td><td>{_form_cell([r["mark"]])}</td></tr>' for r in reversed(detail.get('recent', [])))
    recent = ('<h2>Recent results</h2><div class="scroll"><table><thead><tr><th class="team">Date</th><th class="team">Opponent</th>'
              f'<th class="hide">Competition</th><th>Score</th><th></th></tr></thead><tbody>{recent_rows}</tbody></table></div>') if recent_rows else ''

    fixture_rows = []
    for kickoff, home, away, p in detail.get('remaining', []):
        p_away, p_draw, p_home = (float(v) for v in p)
        at_home = home == team
        opponent = away if at_home else home
        win, lose = (p_home, p_away) if at_home else (p_away, p_home)
        fixture_rows.append(f'<tr><td class="team">{local_time(kickoff)}</td><td class="team">{"H" if at_home else "A"} v '
                            f'{club_link(opponent, "../")}</td><td>{win * 100:.0f}%</td><td>{p_draw * 100:.0f}%</td>'
                            f'<td>{lose * 100:.0f}%</td></tr>')
    fixtures = ('<h2>Remaining fixtures</h2><div class="scroll"><table><thead><tr><th class="team">Kick-off</th>'
                '<th class="team">Opponent</th><th>Win</th><th>Draw</th><th>Lose</th></tr></thead>'
                f'<tbody>{"".join(fixture_rows)}</tbody></table></div><p class="legend">Probabilities use the ratings and form '
                'as they stand today. The season simulation lets them change as results come in, so later fixtures are less certain '
                'than these single numbers suggest.</p>') if fixture_rows else ''

    form_html = f'<p class="sub">Form {_form_cell(form)}</p>' if form else ''
    body = (f'<h1>{html.escape(team)}</h1><p class="sub">{_season_label(season)} projection</p>{form_html}{card_html}'
            f'{_position_bars(detail.get("positions"))}{_elo_chart(detail.get("elo"))}{recent}{fixtures}')
    description = (f'{team} {_season_label(season)}: projected {record["Points"]:.0f} points, '
                   f'title {_pct(record["TitleProb"])}, top four {_pct(record["Top4Prob"])}, relegation {_pct(record["RelegationProb"])}.')
    return page_shell(f'EPLcast - {team}', description, body, f'clubs/{club_slug(team)}.html', '../')


def build_clubs_index(headline, season):
    rows = ''.join(
        f'<tr><td class="rank">{int(r["Rank"])}</td><td class="team">{club_link(r["Team"], "../")}</td>'
        f'<td>{r["Points"]:.1f}</td><td>{_pct(r["TitleProb"])}</td><td>{_pct(r["Top4Prob"])}</td>'
        f'<td>{_pct(r["RelegationProb"])}</td></tr>' for r in headline.sort_values('Rank').to_dict('records'))
    body = (f'<h1>Clubs</h1><p class="sub">{_season_label(season)}, ordered by projected finish.</p>'
            '<div class="scroll"><table><thead><tr><th class="rank"></th><th class="team">Club</th><th>Proj</th><th>Title</th><th>Top 4</th>'
            f'<th>Rel</th></tr></thead><tbody>{rows}</tbody></table></div>')
    return page_shell('EPLcast - clubs', f'Every Premier League club\'s {_season_label(season)} projection.', body, 'clubs/index.html', '../')


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else round(number, 4)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def api_payloads(projection, season, meta):
    headline = projection[projection['Season'] == season].sort_values('Rank')
    clubs = meta.get('clubs') or {}
    teams = []
    for record in headline.to_dict('records'):
        detail = clubs.get(record['Team'], {})
        entry = {key.lower() if key != 'GoalDiff' else 'goal_diff': record[key] for key in
                 ['Rank', 'Team', 'Points', 'Wins', 'Draws', 'Losses', 'GoalDiff']}
        entry.update({'title': record['TitleProb'], 'top4': record['Top4Prob'], 'relegation': record['RelegationProb']})
        if detail.get('range'):
            entry.update({'points_p10': detail['range']['P10'], 'points_p50': detail['range']['P50'], 'points_p90': detail['range']['P90']})
        if detail.get('positions'):
            entry['position_probabilities'] = detail['positions']
        teams.append(entry)
    projection_json = {'season': _season_label(season), 'generated': meta.get('generated'),
                       'matches_played': meta.get('played'), 'simulations': meta.get('simulations'), 'teams': teams}
    fixtures_json = {'generated': meta.get('generated'), 'fixtures': [
        {'kickoff_utc': pd.Timestamp(m['kickoff']).strftime('%Y-%m-%dT%H:%MZ'), 'round': m['round'], 'home': m['home'],
         'away': m['away'], 'home_win': m['p'][2], 'draw': m['p'][1], 'away_win': m['p'][0], 'stakes': m.get('stakes')}
        for m in (meta.get('upcoming') or [])]}
    metrics = meta.get('metrics') or {}
    accuracy_json = {'generated': meta.get('generated'),
                     'held_out': {k: metrics.get(k) for k in ['matches', 'accuracy', 'log_loss', 'baseline', 'book_accuracy', 'book_log_loss']},
                     'seasons': metrics.get('seasons', []), 'calibration': metrics.get('calibration', []),
                     'live': meta.get('live_score')}
    return {'projection.json': _clean(projection_json), 'fixtures.json': _clean(fixtures_json), 'accuracy.json': _clean(accuracy_json)}


def build_not_found_page():
    body = ('<h1>Page not found</h1><p class="sub">That page does not exist, or a club has left the Premier League.</p>'
            f'<p><a href="{SITE_URL}">Go to the projected table</a></p>')
    return page_shell('EPLcast - page not found', 'Page not found.', body, '404.html', SITE_URL)


def build_sitemap(pages):
    entries = ''.join(f'<url><loc>{html.escape(SITE_URL + page)}</loc></url>' for page in sorted(pages))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{entries}</urlset>\n')


def _write(path, text):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)


def write_site(projection, season, meta, docs_dir):
    _write(os.path.join(docs_dir, 'index.html'), build_page(projection, season, meta))
    _write(os.path.join(docs_dir, 'matches.html'), build_matches_page(meta))
    _write(os.path.join(docs_dir, 'accuracy.html'), build_accuracy_page(meta))
    _write(os.path.join(docs_dir, 'methodology.html'), build_methodology_page(meta))

    headline = projection[projection['Season'] == season]
    clubs_dir = os.path.join(docs_dir, 'clubs')
    _write(os.path.join(clubs_dir, 'index.html'), build_clubs_index(headline, season))
    standings = meta.get('standings')
    live = {str(r.Team): r for r in standings.itertuples()} if standings is not None else {}
    written = {os.path.join(clubs_dir, 'index.html')}
    for record in headline.to_dict('records'):
        team = record['Team']
        path = os.path.join(clubs_dir, f'{club_slug(team)}.html')
        detail = (meta.get('clubs') or {}).get(team, {})
        _write(path, build_club_page(team, record, live.get(team), detail, (meta.get('form') or {}).get(team), season))
        written.add(path)
    for stale in glob.glob(os.path.join(clubs_dir, '*.html')):
        if stale not in written:
            os.remove(stale)

    _write(os.path.join(docs_dir, '404.html'), build_not_found_page())
    pages = ['', 'matches.html', 'accuracy.html', 'methodology.html'] + [
        os.path.relpath(path, docs_dir).replace(os.sep, '/') for path in written]
    _write(os.path.join(docs_dir, 'sitemap.xml'), build_sitemap(pages))

    for name, payload in api_payloads(projection, season, meta).items():
        _write(os.path.join(docs_dir, 'api', name), json.dumps(payload, indent=1, allow_nan=False) + '\n')
    return docs_dir
