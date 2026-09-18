import html
import os

STYLES = """
:root {
  --bg: #0f1115;
  --card: #171a21;
  --line: #262b35;
  --text: #e8eaee;
  --muted: #9aa3b2;
  --accent: #4ade80;
  --warn: #f87171;
  --bar: #2b3240;
  --win: #22c55e;
  --draw: #6b7280;
  --loss: #ef4444;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 20px 64px;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 17px; margin: 40px 0 12px; }
.sub { color: var(--muted); margin: 0 0 28px; font-size: 14px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; flex: 1 1 170px; }
.card .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
.card .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
th, td { padding: 9px 11px; text-align: right; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; white-space: nowrap; }
th.team, td.team { text-align: left; white-space: nowrap; }
th.group { border-left: 1px solid var(--line); }
td.group { border-left: 1px solid var(--line); }
td.rank { color: var(--muted); width: 38px; }
tbody tr:last-child td { border-bottom: none; }
tr.ucl td.rank { color: var(--accent); font-weight: 600; }
tr.drop td.rank { color: var(--warn); font-weight: 600; }
.form { display: inline-flex; gap: 3px; }
.form i { width: 16px; height: 16px; border-radius: 3px; font-style: normal; font-size: 10px; line-height: 16px; text-align: center; color: #0f1115; font-weight: 700; }
.form i.W { background: var(--win); }
.form i.D { background: var(--draw); color: #e8eaee; }
.form i.L { background: var(--loss); }
.move { font-size: 12px; color: var(--muted); }
.legend { color: var(--muted); font-size: 13px; margin-top: 14px; }
.chart { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 18px 16px 10px; }
.chart svg { width: 100%; height: auto; display: block; }
.keys { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; font-size: 13px; color: var(--muted); }
.keys span { display: inline-flex; align-items: center; gap: 6px; }
.keys b { width: 18px; height: 3px; border-radius: 2px; display: inline-block; }
.footer { color: var(--muted); font-size: 13px; margin-top: 36px; border-top: 1px solid var(--line); padding-top: 16px; }
.footer a { color: var(--text); }
@media (max-width: 760px) {
  body { padding: 20px 12px 48px; }
  th.hide, td.hide { display: none; }
}
"""

LINE_COLOURS = ['#60a5fa', '#f472b6', '#facc15', '#4ade80', '#fb923c', '#a78bfa']


def _pct(value):
    return f"{value * 100:.0f}%" if value >= 0.005 else "-"


def _bar(value):
    return round(min(max(value, 0.0), 1.0) * 100)


def _goal_diff(value):
    return "0.0" if abs(value) < 0.05 else f"{value:+.1f}"


def _season_label(season):
    return f"{season}-{str(season + 1)[-2:]}"


def _format_date(timestamp):
    return timestamp.strftime('%-d %B %Y') if hasattr(timestamp, 'strftime') else str(timestamp)


def _form_cell(marks):
    if not marks:
        return '<span class="form"></span>'
    pills = ''.join(f'<i class="{m}">{m}</i>' for m in marks)
    return f'<span class="form">{pills}</span>'


def _movement(projected_rank, actual_position):
    if actual_position is None:
        return ''
    delta = actual_position - projected_rank
    if delta == 0:
        return '<span class="move">=</span>'
    arrow = '&#9650;' if delta > 0 else '&#9660;'
    colour = 'var(--accent)' if delta > 0 else 'var(--warn)'
    return f'<span class="move" style="color:{colour}">{arrow}{abs(delta)}</span>'


def _title_race_chart(history, season):
    if history is None or len(history) == 0:
        return ''
    frame = history[history['season'] == season]
    dates = sorted(frame['recorded_on'].unique())
    if len(dates) < 2:
        return ''

    latest = frame[frame['recorded_on'] == dates[-1]].sort_values('TitleProb', ascending=False)
    teams = [t for t in latest['Team'].head(len(LINE_COLOURS)) if latest[latest['Team'] == t]['TitleProb'].iloc[0] > 0.01]
    if not teams:
        return ''

    width, height = 720, 250
    pad_left, pad_right, pad_top, pad_bottom = 42, 14, 14, 30
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def x_of(index):
        if len(dates) == 1:
            return pad_left + plot_w / 2
        return pad_left + plot_w * index / (len(dates) - 1)

    def y_of(prob):
        return pad_top + plot_h * (1 - min(max(prob, 0.0), 1.0))

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img">']
    for tick in [0, 25, 50, 75, 100]:
        y = y_of(tick / 100)
        parts.append(f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" stroke="#262b35" stroke-width="1"/>')
        parts.append(f'<text x="{pad_left - 8}" y="{y + 4:.1f}" fill="#9aa3b2" font-size="11" text-anchor="end">{tick}%</text>')

    for label_index in (0, len(dates) - 1):
        label = dates[label_index][5:]
        anchor = 'start' if label_index == 0 else 'end'
        parts.append(f'<text x="{x_of(label_index):.1f}" y="{height - 8}" fill="#9aa3b2" font-size="11" text-anchor="{anchor}">{html.escape(label)}</text>')

    keys = []
    for position, team in enumerate(teams):
        colour = LINE_COLOURS[position % len(LINE_COLOURS)]
        points = []
        for index, date in enumerate(dates):
            row = frame[(frame['recorded_on'] == date) & (frame['Team'] == team)]
            if len(row) == 0:
                continue
            points.append(f'{x_of(index):.1f},{y_of(float(row["TitleProb"].iloc[0])):.1f}')
        if len(points) < 2:
            continue
        parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" points="{" ".join(points)}"/>')
        keys.append(f'<span><b style="background:{colour}"></b>{html.escape(str(team))}</span>')

    parts.append('</svg>')
    if not keys:
        return ''
    return (f'<h2>Title race</h2><div class="chart">{"".join(parts)}'
            f'<div class="keys">{"".join(keys)}</div></div>'
            f'<p class="legend">Probability of winning the league, recorded after each update.</p>')


def _main_table(projection, headline_season, standings, form):
    headline = projection[projection['Season'] == headline_season]
    team_count = len(headline)
    positions = {}
    played_map = {}
    points_map = {}
    if standings is not None:
        for row in standings.itertuples():
            positions[row.Team] = int(row.Position)
            played_map[row.Team] = int(row.Played)
            points_map[row.Team] = int(row.Points)

    live = standings is not None
    header = ['<tr><th class="rank"></th><th class="team">Team</th>']
    if live:
        header.append('<th class="hide">Form</th><th class="group">Pl</th><th>Pts</th>')
    header.append('<th class="group">Proj</th><th class="hide">W</th><th class="hide">D</th><th class="hide">L</th><th class="hide">GD</th>')
    header.append('<th class="group">Title</th><th>Top 4</th><th>Rel</th></tr>')

    rows = []
    for record in headline.to_dict('records'):
        rank = int(record['Rank'])
        team = str(record['Team'])
        if rank <= 4:
            row_class = 'ucl'
        elif rank > team_count - 3:
            row_class = 'drop'
        else:
            row_class = ''
        cells = [f'<tr class="{row_class}"><td class="rank">{rank}</td>',
                 f'<td class="team">{html.escape(team)} {_movement(rank, positions.get(team))}</td>']
        if live:
            cells.append(f'<td class="hide">{_form_cell((form or {}).get(team))}</td>')
            cells.append(f'<td class="group">{played_map.get(team, 0)}</td>')
            cells.append(f'<td>{points_map.get(team, 0)}</td>')
        cells.append(f'<td class="group"><strong>{record["Points"]:.1f}</strong></td>')
        cells.append(f'<td class="hide">{record["Wins"]:.1f}</td>')
        cells.append(f'<td class="hide">{record["Draws"]:.1f}</td>')
        cells.append(f'<td class="hide">{record["Losses"]:.1f}</td>')
        cells.append(f'<td class="hide">{_goal_diff(record["GoalDiff"])}</td>')
        for key in ['TitleProb', 'Top4Prob', 'RelegationProb']:
            value = record[key]
            css = 'class="group"' if key == 'TitleProb' else ''
            cells.append(f'<td {css} style="background:linear-gradient(to left, var(--bar) {_bar(value)}%, transparent {_bar(value)}%)">{_pct(value)}</td>')
        cells.append('</tr>')
        rows.append(''.join(cells))

    return f'<table><thead>{"".join(header)}</thead><tbody>{"".join(rows)}</tbody></table>'


def _future_block(projection, headline_season):
    later = sorted(s for s in projection['Season'].unique() if s > headline_season)
    if not later:
        return ''
    season = later[0]
    block = projection[projection['Season'] == season].head(6)
    rows = []
    for record in block.to_dict('records'):
        rows.append(
            f'<tr><td class="rank">{int(record["Rank"])}</td>'
            f'<td class="team">{html.escape(str(record["Team"]))}</td>'
            f'<td>{record["Points"]:.1f}</td>'
            f'<td>{_pct(record["TitleProb"])}</td>'
            f'<td>{_pct(record["Top4Prob"])}</td></tr>'
        )
    return (f'<h2>{_season_label(season)} outlook</h2><table><thead>'
            f'<tr><th class="rank"></th><th class="team">Team</th><th>Pts</th><th>Title</th><th>Top 4</th></tr>'
            f'</thead><tbody>{"".join(rows)}</tbody></table>')


def build_page(projection, headline_season, meta):
    standings = meta.get('standings')
    form = meta.get('form')
    history = meta.get('history')

    played_card = f"{meta['played']} / {meta['total']}" if meta['played'] else "not started"
    table_html = _main_table(projection, headline_season, standings, form)
    chart_html = _title_race_chart(history, headline_season)
    future_html = _future_block(projection, headline_season)
    note = ('Pts shows points won so far, Proj shows where the model expects each team to finish. '
            'The arrow compares current position with projected finish.') if standings is not None else (
            'Projected points are the average across every simulation.')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EPLcast - {_season_label(headline_season)} projection</title>
<style>{STYLES}</style>
</head>
<body>
<div class="wrap">
<h1>{_season_label(headline_season)} projected table</h1>
<p class="sub">{html.escape(meta['status_line'])}</p>

<div class="cards">
  <div class="card"><div class="label">Matches played</div><div class="value">{played_card}</div></div>
  <div class="card"><div class="label">Simulations</div><div class="value">{meta['simulations']}</div></div>
  <div class="card"><div class="label">Model accuracy</div><div class="value">{meta['accuracy']}</div></div>
  <div class="card"><div class="label">Data through</div><div class="value">{_format_date(meta['last_match_date'])}</div></div>
</div>

{table_html}
<p class="legend">{note}</p>

{chart_html}

{future_html}

<div class="footer">
Built by <a href="https://github.com/hle0110/EPLcast">EPLcast</a>. Match data from football-data.co.uk.
Predictions are statistical estimates, not betting advice.
</div>
</div>
</body>
</html>
"""


def write_dashboard(projection, headline_season, meta, path):
    page = build_page(projection, headline_season, meta)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(page)
    return path
