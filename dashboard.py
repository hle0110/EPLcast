import html
import os
import re

SITE_URL = "https://hle0110.github.io/EPLcast/"
SITE_NAME = "EPLcast"
FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
           "%3Crect width='32' height='32' rx='7' fill='%23171a21'/%3E"
           "%3Crect x='6' y='8' width='5' height='18' rx='1' fill='%234ade80'/%3E"
           "%3Crect x='13.5' y='13' width='5' height='13' rx='1' fill='%2360a5fa'/%3E"
           "%3Crect x='21' y='18' width='5' height='8' rx='1' fill='%23f87171'/%3E%3C/svg%3E")

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
  --draw: #eab308;
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
td.finish { white-space: nowrap; }
tbody tr:last-child td { border-bottom: none; }
tr.ucl td.finish { color: var(--accent); font-weight: 600; }
tr.drop td.finish { color: var(--warn); font-weight: 600; }
.form { display: inline-flex; gap: 3px; }
.form i { width: 16px; height: 16px; border-radius: 3px; font-style: normal; font-size: 10px; line-height: 16px; text-align: center; color: #0f1115; font-weight: 700; }
.form i.W { background: var(--win); }
.form i.D { background: var(--draw); }
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
a { color: inherit; }
td.team a { text-decoration: none; }
td.team a:hover { text-decoration: underline; }
.nav { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 26px; }
.nav a { color: var(--muted); text-decoration: none; padding: 6px 12px; border: 1px solid var(--line); border-radius: 999px; font-size: 13px; }
.nav a.on, .nav a:hover { color: var(--text); background: var(--card); }
.scroll { overflow-x: auto; }
.heat td { padding: 6px 4px; font-size: 11px; text-align: center; min-width: 26px; }
.heat th { padding: 6px 4px; text-align: center; }
.heat td.team { text-align: left; font-size: 13px; padding-left: 11px; }
.prob3 { display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 6px; min-width: 120px; }
.prob3 b { display: block; height: 100%; }
.muted { color: var(--muted); }
.stake { font-size: 13px; color: var(--muted); }
.posbar { display: flex; align-items: flex-end; gap: 3px; height: 120px; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 12px 0; }
.posbar div { flex: 1; background: var(--accent); border-radius: 2px 2px 0 0; min-height: 1px; }
.posbar div.mid { background: #60a5fa; }
.posbar div.low { background: var(--warn); }
.poslabels { display: flex; gap: 3px; padding: 4px 12px 0; font-size: 10px; color: var(--muted); }
.poslabels span { flex: 1; text-align: center; }
.prose { max-width: 760px; }
.prose p, .prose li { color: var(--text); }
@media (max-width: 760px) {
  body { padding: 20px 12px 48px; }
  th.hide, td.hide { display: none; }
}
"""

LINE_COLOURS = ['#60a5fa', '#f472b6', '#facc15', '#4ade80', '#fb923c', '#a78bfa']
NAV_ITEMS = [('index.html', 'Table'), ('matches.html', 'Matches'), ('clubs/index.html', 'Clubs'),
             ('accuracy.html', 'Accuracy'), ('methodology.html', 'Method')]


def club_slug(team):
    return re.sub(r'[^a-z0-9]+', '-', str(team).lower()).strip('-')


def club_link(team, prefix=''):
    return f'<a href="{prefix}clubs/{club_slug(team)}.html">{html.escape(str(team))}</a>'


def nav_bar(current, prefix=''):
    links = []
    for path, label in NAV_ITEMS:
        css = ' class="on"' if path == current else ''
        links.append(f'<a{css} href="{prefix}{path}">{label}</a>')
    return f'<nav class="nav">{"".join(links)}</nav>'


def page_shell(title, description, body, current, prefix=''):
    url = SITE_URL + current
    safe_title = html.escape(title)
    safe_description = html.escape(description)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<meta name="description" content="{safe_description}">
<link rel="canonical" href="{url}">
<meta property="og:site_name" content="{SITE_NAME}">
<meta property="og:type" content="website">
<meta property="og:title" content="{safe_title}">
<meta property="og:description" content="{safe_description}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{SITE_URL}og-image.png">
<meta property="og:image:width" content="1280">
<meta property="og:image:height" content="640">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{SITE_URL}og-image.png">
<link rel="icon" href="{FAVICON}">
<meta name="twitter:title" content="{safe_title}">
<meta name="twitter:description" content="{safe_description}">
<style>{STYLES}</style>
</head>
<body>
<div class="wrap">
{nav_bar(current, prefix)}
{body}
<div class="footer">
Built by <a href="https://github.com/hle0110/EPLcast">EPLcast</a>. Match data from football-data.co.uk, fixture dates from fixturedownload.com.
Data as JSON: <a href="{prefix}api/projection.json">projection</a>, <a href="{prefix}api/fixtures.json">fixtures</a>, <a href="{prefix}api/accuracy.json">accuracy</a>.
Predictions are statistical estimates, not betting advice.
</div>
</div>
</body>
</html>
"""


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


def race_chart(history, season, column='TitleProb', heading='Title race', legend='Probability of winning the league, recorded after each update.', threshold=0.01):
    if history is None or len(history) == 0:
        return ''
    frame = history[history['season'] == season]
    dates = sorted(frame['recorded_on'].unique())
    if len(dates) < 2:
        return ''

    latest = frame[frame['recorded_on'] == dates[-1]].sort_values(column, ascending=False)
    teams = [t for t in latest['Team'].head(len(LINE_COLOURS)) if latest[latest['Team'] == t][column].iloc[0] > threshold]
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
            points.append(f'{x_of(index):.1f},{y_of(float(row[column].iloc[0])):.1f}')
        if len(points) < 2:
            continue
        parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" points="{" ".join(points)}"/>')
        keys.append(f'<span><b style="background:{colour}"></b>{html.escape(str(team))}</span>')

    parts.append('</svg>')
    if not keys:
        return ''
    return (f'<h2>{heading}</h2><div class="chart">{"".join(parts)}'
            f'<div class="keys">{"".join(keys)}</div></div>'
            f'<p class="legend">{legend}</p>')


def _main_table(projection, headline_season, standings, form):
    headline = projection[projection['Season'] == headline_season]
    team_count = len(headline)
    projected = {str(r['Team']): r for r in headline.to_dict('records')}

    if standings is None:
        order = [(int(r['Rank']), str(r['Team'])) for r in headline.to_dict('records')]
        live = {}
    else:
        order = [(int(r.Position), str(r.Team)) for r in standings.itertuples()]
        live = {str(r.Team): r for r in standings.itertuples()}

    header = ['<tr><th class="rank"></th><th class="team">Team</th>']
    if live:
        header.append('<th class="hide">Form</th><th>Pl</th><th>Pts</th><th class="hide">GD</th>')
    header.append('<th class="group">Proj</th><th>Finish</th>')
    header.append('<th class="group">Title</th><th>Top 4</th><th>Rel</th></tr>')

    rows = []
    for position, team in order:
        record = projected.get(team)
        if record is None:
            continue
        finish = int(record['Rank'])
        if finish <= 4:
            row_class = 'ucl'
        elif finish > team_count - 3:
            row_class = 'drop'
        else:
            row_class = ''
        cells = [f'<tr class="{row_class}"><td class="rank">{position}</td>',
                 f'<td class="team">{club_link(team)}</td>']
        if live:
            stat = live[team]
            cells.append(f'<td class="hide">{_form_cell((form or {}).get(team))}</td>')
            cells.append(f'<td>{int(stat.Played)}</td>')
            cells.append(f'<td><strong>{int(stat.Points)}</strong></td>')
            cells.append(f'<td class="hide">{int(stat.GoalDiff):+d}</td>')
        cells.append(f'<td class="group">{record["Points"]:.1f}</td>')
        cells.append(f'<td class="finish">{finish}{_movement(finish, position if live else None)}</td>')
        for key in ['TitleProb', 'Top4Prob', 'RelegationProb']:
            value = record[key]
            css = 'class="group"' if key == 'TitleProb' else ''
            cells.append(f'<td {css} style="background:linear-gradient(to left, var(--bar) {_bar(value)}%, transparent {_bar(value)}%)">{_pct(value)}</td>')
        cells.append('</tr>')
        rows.append(''.join(cells))

    return f'<div class="scroll"><table><thead>{"".join(header)}</thead><tbody>{"".join(rows)}</tbody></table></div>'


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
            f'<td class="team">{club_link(record["Team"])}</td>'
            f'<td>{record["Points"]:.1f}</td>'
            f'<td>{_pct(record["TitleProb"])}</td>'
            f'<td>{_pct(record["Top4Prob"])}</td></tr>'
        )
    return (f'<h2>{_season_label(season)} outlook</h2><div class="scroll"><table><thead>'
            f'<tr><th class="rank"></th><th class="team">Team</th><th>Pts</th><th>Title</th><th>Top 4</th></tr>'
            f'</thead><tbody>{"".join(rows)}</tbody></table></div>')


def position_heatmap(positions, projection, season, prefix=''):
    if positions is None or len(positions) == 0:
        return ''
    order = projection[projection['Season'] == season].sort_values('Rank')['Team'].tolist()
    frame = positions.set_index('Team')
    columns = [c for c in frame.columns if c.isdigit()]
    header = '<tr><th class="team">Team</th>' + ''.join(f'<th>{c}</th>' for c in columns) + '</tr>'
    rows = []
    for team in order:
        if team not in frame.index:
            continue
        cells = [f'<td class="team">{club_link(team, prefix)}</td>']
        for column in columns:
            value = float(frame.loc[team, column])
            alpha = min(1.0, value * 2.5)
            text = f'{value * 100:.0f}' if value >= 0.005 else ''
            cells.append(f'<td style="background:rgba(74,222,128,{alpha:.2f})">{text}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return (f'<h2>Where each club finishes</h2><div class="scroll"><table class="heat"><thead>{header}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            f'<p class="legend">Percentage of simulations in which each club finishes in each position. Blank means under 0.5%.</p>')


def build_page(projection, headline_season, meta):
    standings = meta.get('standings')
    form = meta.get('form')
    history = meta.get('history')

    played_card = f"{meta['played']} / {meta['total']}" if meta['played'] else "not started"
    table_html = _main_table(projection, headline_season, standings, form)
    charts = ''.join([
        race_chart(history, headline_season),
        race_chart(history, headline_season, 'Top4Prob', 'Top four race',
                   'Probability of a top four finish for the clubs nearest the line, recorded after each update.', 0.05),
        race_chart(history, headline_season, 'RelegationProb', 'Relegation fight',
                   'Probability of finishing in the bottom three, recorded after each update.', 0.05),
    ])
    heatmap_html = position_heatmap(meta.get('positions'), projection, headline_season)
    future_html = _future_block(projection, headline_season)
    note = ('Ordered by the live table. Proj is the projected final points total and Finish the projected '
            'position, with the arrow showing the expected move from where each club sits now. '
            'Select a club for its own page.') if standings is not None else (
            'Projected points are the average across every simulation.')
    label = _season_label(headline_season)
    leader = projection[projection['Season'] == headline_season].sort_values('TitleProb', ascending=False).iloc[0]
    description = (f"{label} Premier League projection: {leader['Team']} {leader['TitleProb'] * 100:.0f}% to win the title. "
                   f"Every remaining fixture simulated {meta['simulations']:,} times.")

    body = f"""<h1>{label} projected table</h1>
<p class="sub">{html.escape(meta['status_line'])}</p>

<div class="cards">
  <div class="card"><div class="label">Matches played</div><div class="value">{played_card}</div></div>
  <div class="card"><div class="label">Simulations</div><div class="value">{meta['simulations']:,}</div></div>
  <div class="card"><div class="label">Model accuracy</div><div class="value">{meta['accuracy']}</div></div>
  <div class="card"><div class="label">Data through</div><div class="value">{_format_date(meta['last_match_date'])}</div></div>
</div>

{table_html}
<p class="legend">{note}</p>

{charts}

{heatmap_html}

{future_html}"""
    return page_shell(f"EPLcast - {label} projection", description, body, 'index.html')


def write_dashboard(projection, headline_season, meta, path):
    page = build_page(projection, headline_season, meta)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(page)
    return path
