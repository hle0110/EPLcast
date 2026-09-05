import html
import os

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EPLcast - {season_label} projection</title>
<style>
:root {{
  --bg: #0f1115;
  --card: #171a21;
  --line: #262b35;
  --text: #e8eaee;
  --muted: #9aa3b2;
  --accent: #4ade80;
  --warn: #f87171;
  --bar: #2b3240;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 32px 20px 64px;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
.wrap {{ max-width: 1040px; margin: 0 auto; }}
h1 {{ font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }}
.sub {{ color: var(--muted); margin: 0 0 28px; font-size: 14px; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; }}
.card {{
  background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 16px; flex: 1 1 180px;
}}
.card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
.card .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }}
th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }}
th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; white-space: nowrap; }}
td.team {{ white-space: nowrap; }}
th.team, td.team {{ text-align: left; }}
td.rank {{ color: var(--muted); width: 40px; }}
tbody tr:last-child td {{ border-bottom: none; }}
tr.ucl td.rank {{ color: var(--accent); font-weight: 600; }}
tr.drop td.rank {{ color: var(--warn); font-weight: 600; }}
.legend {{ color: var(--muted); font-size: 13px; margin-top: 16px; }}
.footer {{ color: var(--muted); font-size: 13px; margin-top: 32px; border-top: 1px solid var(--line); padding-top: 16px; }}
.footer a {{ color: var(--text); }}
h2 {{ font-size: 17px; margin: 36px 0 12px; }}
@media (max-width: 720px) {{
  body {{ padding: 20px 12px 48px; }}
  th.hide, td.hide {{ display: none; }}
}}
</style>
</head>
<body>
<div class="wrap">
<h1>{season_label} projected table</h1>
<p class="sub">{status_line}</p>

<div class="cards">
  <div class="card"><div class="label">Matches played</div><div class="value">{played} / {total}</div></div>
  <div class="card"><div class="label">Simulations</div><div class="value">{simulations}</div></div>
  <div class="card"><div class="label">Model accuracy</div><div class="value">{accuracy}</div></div>
  <div class="card"><div class="label">Data through</div><div class="value">{last_match_date}</div></div>
</div>

<table>
<thead>
<tr>
<th class="rank"></th>
<th class="team">Team</th>
<th>Pts</th>
<th class="hide">W</th>
<th class="hide">D</th>
<th class="hide">L</th>
<th class="hide">GD</th>
<th>Title</th>
<th>Top 4</th>
<th>Relegation</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<p class="legend">Projected points are the average across {simulations} simulations of every remaining fixture. Percentages are how often each team finished in that position.</p>

{future}

<div class="footer">
Built by <a href="https://github.com/hle0110/EPLcast">EPLcast</a>. Match data from football-data.co.uk.
Predictions are statistical estimates, not betting advice.
</div>
</div>
</body>
</html>
"""

ROW_TEMPLATE = """<tr class="{row_class}">
<td class="rank">{rank}</td>
<td class="team">{team}</td>
<td>{points}</td>
<td class="hide">{wins}</td>
<td class="hide">{draws}</td>
<td class="hide">{losses}</td>
<td class="hide">{goal_diff}</td>
<td style="background:linear-gradient(to left, var(--bar) {title_bar}%, transparent {title_bar}%)">{title}</td>
<td style="background:linear-gradient(to left, var(--bar) {top4_bar}%, transparent {top4_bar}%)">{top4}</td>
<td style="background:linear-gradient(to left, var(--bar) {releg_bar}%, transparent {releg_bar}%)">{releg}</td>
</tr>"""

FUTURE_TEMPLATE = """<h2>{label} outlook</h2>
<table>
<thead><tr><th class="rank"></th><th class="team">Team</th><th>Pts</th><th>Title</th><th>Top 4</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>"""

FUTURE_ROW = """<tr><td class="rank">{rank}</td><td class="team">{team}</td><td>{points}</td><td>{title}</td><td>{top4}</td></tr>"""

def _goal_diff(value):
    return "0.0" if abs(value) < 0.05 else f"{value:+.1f}"

def _pct(value):
    return f"{value * 100:.0f}%" if value >= 0.005 else "-"

def _bar(value):
    return round(min(max(value, 0.0), 1.0) * 100)

def _season_label(season):
    return f"{season}-{str(season + 1)[-2:]}"

def _format_date(timestamp):
    return timestamp.strftime('%-d %B %Y') if hasattr(timestamp, 'strftime') else str(timestamp)

def build_page(projection, headline_season, meta):
    headline = projection[projection['Season'] == headline_season]
    team_count = len(headline)
    rows = []
    for record in headline.to_dict('records'):
        rank = int(record['Rank'])
        if rank <= 4:
            row_class = 'ucl'
        elif rank > team_count - 3:
            row_class = 'drop'
        else:
            row_class = ''
        rows.append(ROW_TEMPLATE.format(
            row_class=row_class,
            rank=rank,
            team=html.escape(str(record['Team'])),
            points=f"{record['Points']:.1f}",
            wins=f"{record['Wins']:.1f}",
            draws=f"{record['Draws']:.1f}",
            losses=f"{record['Losses']:.1f}",
            goal_diff=_goal_diff(record['GoalDiff']),
            title=_pct(record['TitleProb']),
            top4=_pct(record['Top4Prob']),
            releg=_pct(record['RelegationProb']),
            title_bar=_bar(record['TitleProb']),
            top4_bar=_bar(record['Top4Prob']),
            releg_bar=_bar(record['RelegationProb']),
        ))

    future_seasons = sorted(s for s in projection['Season'].unique() if s > headline_season)
    future_html = ''
    if future_seasons:
        next_season = future_seasons[0]
        block = projection[projection['Season'] == next_season].head(6)
        future_rows = [FUTURE_ROW.format(
            rank=int(r['Rank']),
            team=html.escape(str(r['Team'])),
            points=f"{r['Points']:.1f}",
            title=_pct(r['TitleProb']),
            top4=_pct(r['Top4Prob']),
        ) for r in block.to_dict('records')]
        future_html = FUTURE_TEMPLATE.format(label=_season_label(next_season), rows='\n'.join(future_rows))

    return PAGE_TEMPLATE.format(
        season_label=_season_label(headline_season),
        status_line=html.escape(meta['status_line']),
        played=meta['played'],
        total=meta['total'],
        simulations=meta['simulations'],
        accuracy=meta['accuracy'],
        last_match_date=_format_date(meta['last_match_date']),
        rows='\n'.join(rows),
        future=future_html,
    )

def write_dashboard(projection, headline_season, meta, path):
    page = build_page(projection, headline_season, meta)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(page)
    return path
