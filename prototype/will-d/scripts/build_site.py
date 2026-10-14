"""Build a local results site from control sweep evaluations.

Reads ``results/<sweep-id>/evaluation.json`` files (written by
``latency-ml evaluate``) and renders a single self-contained
``site/index.html``: one section per sweep with a latency-response table and
an inline SVG bar chart. No videos, no dependencies beyond the standard
library.

With no evaluations on disk it writes a placeholder page pointing at the
README instead of failing, so ``./view.sh`` works before anything is trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASK_TITLES = {
    "cartpole": "CartPole",
    "lunarlander": "LunarLander",
}

CSS = """
:root {
  --bg: #0b0e14; --bg-raised: #12161f; --bg-card: #161b26;
  --ink: #e8ecf3; --dim: #8b93a5; --faint: #4a5265; --line: #232a38;
  --good: #2ea44f; --bad: #d64541; --max: 1000px;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: var(--max); margin: 0 auto; padding: 0 28px; }
header.hero { padding: 56px 0 32px; border-bottom: 1px solid var(--line); }
.hero .kicker {
  font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--dim); margin: 0 0 12px;
}
.hero h1 { margin: 0 0 12px; font-size: clamp(28px, 4vw, 40px); letter-spacing: -0.02em; }
.hero p { margin: 0; max-width: 720px; color: var(--dim); }
section { padding: 40px 0 8px; }
section h2 { margin: 0 0 4px; font-size: 26px; }
section .meta { color: var(--faint); font-size: 13px; margin: 0 0 20px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 24px; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); }
th { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--faint); }
td { color: var(--dim); font-variant-numeric: tabular-nums; }
td:first-child { color: var(--ink); font-weight: 600; }
td.num { text-align: right; }
tr:hover td { background: var(--bg-raised); }
.bars { display: flex; align-items: flex-end; gap: 10px; height: 180px;
  padding: 16px; background: var(--bg-raised); border: 1px solid var(--line);
  border-radius: 12px; margin-top: 8px; }
.bar-group { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 6px; }
.bar { width: 60%; border-radius: 4px 4px 0 0; background: var(--good); min-height: 2px; }
.bar-group span { font-size: 11px; color: var(--faint); }
.bar-group strong { font-size: 12px; color: var(--dim); font-weight: 600; }
.empty {
  margin: 40px 0; padding: 32px; border: 1px dashed var(--line); border-radius: 12px;
  color: var(--dim); background: var(--bg-raised);
}
.empty code { color: var(--ink); background: var(--bg); padding: 2px 6px; border-radius: 5px; }
footer { margin-top: 56px; border-top: 1px solid var(--line); padding: 24px 0 48px;
  color: var(--faint); font-size: 13px; }
footer code {
  color: var(--dim); background: var(--bg-raised);
  padding: 2px 6px; border-radius: 5px;
}
"""


def load_evaluations(results_dir: Path) -> list[dict]:
    """All ``evaluation.json`` payloads under ``results_dir``, newest first."""
    found = []
    if not results_dir.is_dir():
        return found
    for path in sorted(results_dir.glob("*/evaluation.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows = payload.get("latency_response", [])
        if isinstance(rows, list) and rows:
            found.append({"sweep_id": path.parent.name, "rows": rows})
    return found


def task_of(sweep_id: str) -> str:
    name = sweep_id.split("-")[-1]
    return TASK_TITLES.get(name, sweep_id)


def bars_svg(rows: list[dict]) -> str:
    """One bar per latency condition, scaled to the best mean return."""
    best = max((r.get("mean_return") or 0) for r in rows) or 1.0
    groups = []
    for r in rows:
        mean = r.get("mean_return") or 0
        frac = max(mean / best, 0.0)
        ms = round(r["latency_ns"] / 1_000_000)
        colour = "var(--good)" if frac > 0.9 else "var(--bad)" if frac < 0.5 else "var(--dim)"
        groups.append(f'<div class="bar-group"><strong>{mean:.0f}</strong><div class="bar" style="height: {frac * 100:.1f}%; background: {colour}"></div><span>{ms} ms</span></div>')
    return '<div class="bars">' + "".join(groups) + "</div>"


def table_html(rows: list[dict]) -> str:
    trs = []
    for r in sorted(rows, key=lambda r: r["latency_ns"]):
        ms = round(r["latency_ns"] / 1_000_000)
        trs.append(
            '<tr><td>{} ms</td><td class="num">{}</td><td class="num">{:.1f} ± {:.1f}</td><td class="num">{:.1f}%</td><td class="num">{:.2f} s</td></tr>'.format(
                ms,
                r.get("action_steps", "—"),
                r.get("mean_return", 0),
                r.get("std_across_seeds", 0),
                100 * (r.get("failure_rate") or 0),
                r.get("mean_survival_s") or 0,
            )
        )
    return "<table><thead><tr><th>latency</th><th>hold</th><th>mean return ± sd</th><th>failure rate</th><th>mean survival</th></tr></thead><tbody>" + "".join(trs) + "</tbody></table>"


def section_html(sweep: dict) -> str:
    rows = sweep["rows"]
    n_seeds = rows[0].get("n_seeds", "?")
    return f'<section><h2>{task_of(sweep["sweep_id"])}</h2><p class="meta">{sweep["sweep_id"]} · {len(rows)} latency conditions · {n_seeds} seeds each</p>{bars_svg(rows)}{table_html(rows)}</section>'


def page_html(sweeps: list[dict]) -> str:
    if not sweeps:
        body = '<div class="empty"><p>No sweep evaluations found under <code>results/</code> yet.</p><p>Train one (see README <em>How to train</em>), then re-run <code>./view.sh</code>.</p></div>'
    else:
        body = "\n".join(section_html(s) for s in sweeps)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>will-d · latency response</title>
<style>{CSS}</style>
</head>
<body>
<header class="hero"><div class="wrap">
<p class="kicker">will-d · Hartley bounded ReLU</p>
<h1>What inference latency does to control</h1>
<p>Mean episodic return across the latency grid, one section per trained sweep.
Higher latency holds each action longer; the bar chart is scaled to the best condition.</p>
</div></header>
<main class="wrap">
{body}
</main>
<footer><div class="wrap">
<p>Built locally by <code>scripts/build_site.py</code> from
<code>results/&lt;sweep-id&gt;/evaluation.json</code>. Rebuild with
<code>./view.sh --build-only</code>.</p>
</div></footer>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the local results site.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args(argv)
    sweeps = load_evaluations(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(page_html(sweeps))
    print(f"site written to {args.out}/ ({len(sweeps)} sweeps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
