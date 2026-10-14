"""Build the local visualisation site from committed results.

A single self-contained page that puts the rendered rollouts next to the numbers
they correspond to: for each task, the five latency conditions of the latency
grid -- which are five *timestep* conditions, since the action is held for
``N = latency / 20 ms`` integration steps -- shown as the time-aligned ladder,
as individual episodes, and as final states, with the sweep's 200-episode means
alongside the single rendered episode for comparison.

Nothing here measures anything. The page is generated from
``results/videos/render_manifest.json`` and ``results/shared-trunk/`` so what it
displays is traceable to the runs that produced it, and it is rebuilt rather
than hand-edited -- the same convention ``latency-ml plot`` follows.

Assets are copied into ``site/assets/``, so the directory is self-contained and
can be served from anywhere:

    uv run python scripts/build_site.py
    uv run python -m http.server 8000 --directory site
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Latency colour ramp, matching the video overlays so a panel, a card and a
#: table row are the same condition everywhere on the page.
LATENCY_COLOURS: dict[int, str] = {
    20: "#2ea44f",
    40: "#c9a227",
    60: "#e07b39",
    80: "#d64541",
    100: "#b02a30",
}

TASK_TITLES: dict[str, str] = {
    "cartpole": "CartPole-v1",
    "lunarlander": "LunarLander-v3",
}

TASK_BLURBS: dict[str, str] = {
    "cartpole": (
        "Balance the pole. The episode ends the moment it falls, so the only "
        "failure mode is visible in the frame."
    ),
    "lunarlander": (
        "Land between the flags. Higher latency spends fuel on corrections that "
        "arrive after they were needed."
    ),
}


#: One latency condition as the page presents it.
@dataclass(frozen=True)
class Condition:
    latency_ms: int
    hold_steps: int
    steps: int
    decisions: int
    return_total: float
    outcome: str
    video: str
    colour: str
    #: The sweep's 200-episode mean for this condition, when available.
    sweep_mean: float | None
    failure_rate: float | None
    survival_s: float | None


@dataclass(frozen=True)
class TaskSection:
    task: str
    title: str
    blurb: str
    ladder: str
    contact_sheet: str
    frames: int
    fps: int
    duration_s: float
    conditions: tuple[Condition, ...]


def load_manifest(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text())
    return payload


def load_sweep(path: Path, train_seed: int) -> dict[str, dict[int, dict[str, Any]]]:
    """``{task: {latency_ns: evaluation summary}}`` for one training seed.

    The rendered videos show a single episode per condition; the sweep's means
    are over 200 paired episodes. Showing both lets a reader tell a single
    unlucky rollout from the condition's actual behaviour.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    table: dict[str, dict[int, dict[str, Any]]] = {}
    for cell in payload.get("cells", []):
        if cell.get("train_seed") != train_seed:
            continue
        for task, evaluation in cell.get("evaluations", {}).items():
            table.setdefault(task, {})[int(cell["latency_ns"])] = evaluation
    return table


def build_sections(
    manifest: dict[str, Any], sweep: dict[str, dict[int, dict[str, Any]]]
) -> list[TaskSection]:
    sections = []
    for render in manifest["renders"]:
        task = render["task"]
        conditions = []
        for summary in render["conditions"]:
            latency_ns = summary["latency_ns"]
            evaluation = sweep.get(task, {}).get(latency_ns, {})
            conditions.append(
                Condition(
                    latency_ms=latency_ns // 1_000_000,
                    hold_steps=max(1, latency_ns // 20_000_000),
                    steps=summary["integration_steps"],
                    decisions=summary["decisions"],
                    return_total=summary["return"],
                    outcome=summary["outcome"],
                    video=f"assets/{task}_{latency_ns // 1_000_000}ms.mp4",
                    colour=LATENCY_COLOURS.get(latency_ns // 1_000_000, "#8b949e"),
                    sweep_mean=evaluation.get("mean_return"),
                    failure_rate=evaluation.get("failure_rate"),
                    survival_s=evaluation.get("mean_survival_s"),
                )
            )
        sections.append(
            TaskSection(
                task=task,
                title=TASK_TITLES.get(task, task),
                blurb=TASK_BLURBS.get(task, ""),
                ladder=f"assets/{task}_latency_ladder.mp4",
                contact_sheet=f"assets/{task}_contact_sheet.png",
                frames=render["frames"],
                fps=render["fps"],
                duration_s=render["frames"] / render["fps"],
                conditions=tuple(conditions),
            )
        )
    return sections


def outcome_badge(condition: Condition) -> str:
    if condition.outcome == "terminated":
        return '<span class="badge bad">terminated</span>'
    if condition.outcome == "truncated":
        return '<span class="badge good">survived</span>'
    return '<span class="badge">running</span>'


def condition_card(condition: Condition) -> str:
    """One condition's video, with the numbers from the rendered episode."""
    hold_text = f"{condition.hold_steps} step"
    if condition.hold_steps > 1:
        hold_text += "s"
    return f"""      <figure class="card" style="--accent: {condition.colour}">
        <div class="card-media">
          <video src="{condition.video}" muted loop playsinline preload="metadata"></video>
        </div>
        <figcaption>
          <div class="card-head">
            <span class="latency">{condition.latency_ms} ms</span>
            {outcome_badge(condition)}
          </div>
          <dl>
            <div><dt>acts every</dt><dd>{hold_text}</dd></div>
            <div><dt>decisions</dt><dd>{condition.decisions}</dd></div>
            <div><dt>return</dt><dd>{condition.return_total:+.1f}</dd></div>
            <div><dt>survived</dt><dd>{condition.steps * 0.02:.2f} s</dd></div>
          </dl>
        </figcaption>
      </figure>"""


def condition_row(condition: Condition) -> str:
    """One condition as a table row, rendered episode next to the sweep mean."""
    sweep = f"{condition.sweep_mean:+.1f}" if condition.sweep_mean is not None else "—"
    failure = f"{condition.failure_rate * 100:.1f}%" if condition.failure_rate is not None else "—"
    survival = f"{condition.survival_s:.2f} s" if condition.survival_s is not None else "—"
    dot = f'<span class="dot" style="background: {condition.colour}"></span>'
    return f"""        <tr>
          <td>{dot}{condition.latency_ms} ms</td>
          <td>{condition.hold_steps}</td>
          <td class="num">{condition.decisions}</td>
          <td class="num">{condition.return_total:+.1f}</td>
          <td>{condition.outcome or "—"}</td>
          <td class="num">{sweep}</td>
          <td class="num">{failure}</td>
          <td class="num">{survival}</td>
        </tr>"""


LADDER_CAPTION = """      <p class="caption">
        Each panel plays the policy <em>trained and evaluated</em> at that latency — the matched
        diagonal of the experiment's design — from the same paired initial state. One frame is
        20&nbsp;ms of simulated time, so the video's clock is the physics' clock. A frozen panel
        with a red frame is an episode that has already ended.
      </p>"""


def task_section_html(section: TaskSection) -> str:
    cards = "\n".join(condition_card(condition) for condition in section.conditions)
    rows = "\n".join(condition_row(condition) for condition in section.conditions)
    return f"""  <section id="{section.task}">
    <div class="section-head">
      <div>
        <h2>{section.title}</h2>
        <p class="blurb">{section.blurb}</p>
      </div>
      <div class="section-meta">
        <span>{section.frames} frames</span>
        <span>{section.fps} fps</span>
        <span>{section.duration_s:.0f} s of simulated time</span>
      </div>
    </div>

    <h3>The latency ladder <small>five conditions, one timeline</small></h3>
    <div class="ladder">
      <video src="{section.ladder}" muted loop playsinline autoplay preload="metadata"></video>
    </div>
{LADDER_CAPTION}

    <h3>Each condition on its own <small>hover to play</small></h3>
    <div class="cards">
{cards}
    </div>

    <h3>The numbers <small>rendered episode vs the sweep's 200-episode means</small></h3>
    <table>
      <thead>
        <tr>
          <th>latency</th><th>hold</th><th>decisions</th><th>episode return</th>
          <th>outcome</th><th>mean return</th><th>failure rate</th><th>mean survival</th>
        </tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>

    <h3>Final states</h3>
    <img class="sheet" src="{section.contact_sheet}"
         alt="{section.title} final states across the latency grid" />
  </section>"""


PAGE_CSS = """
:root {
  --bg: #0b0e14;
  --bg-raised: #12161f;
  --bg-card: #161b26;
  --ink: #e8ecf3;
  --dim: #8b93a5;
  --faint: #4a5265;
  --line: #232a38;
  --good: #2ea44f;
  --bad: #d64541;
  --max: 1240px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
      "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: var(--max); margin: 0 auto; padding: 0 28px; }

header.hero {
  padding: 72px 0 40px;
  background:
    radial-gradient(800px 300px at 20% -50px, rgba(46, 164, 79, 0.12), transparent),
    radial-gradient(800px 300px at 80% -50px, rgba(176, 42, 48, 0.12), transparent),
    var(--bg);
  border-bottom: 1px solid var(--line);
}
.hero .kicker {
  font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--dim); margin: 0 0 14px;
}
.hero h1 {
  margin: 0 0 16px;
  font-size: clamp(30px, 4.4vw, 46px);
  line-height: 1.12;
  letter-spacing: -0.02em;
  font-weight: 700;
}
.hero h1 em { font-style: normal; color: var(--dim); }
.hero p.lede { margin: 0; max-width: 760px; color: var(--dim); font-size: 17px; }
.legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 26px; }
.legend span {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 5px 12px; border: 1px solid var(--line); border-radius: 999px;
  font-size: 13px; color: var(--dim); background: var(--bg-raised);
}
.legend i { width: 9px; height: 9px; border-radius: 50%; }

nav.tabs {
  position: sticky; top: 0; z-index: 10;
  background: rgba(11, 14, 20, 0.85);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--line);
}
nav.tabs .wrap { display: flex; gap: 4px; }
nav.tabs a {
  padding: 13px 16px; color: var(--dim); text-decoration: none;
  font-size: 14px; font-weight: 500; border-bottom: 2px solid transparent;
}
nav.tabs a:hover { color: var(--ink); }
nav.tabs a.active { color: var(--ink); border-bottom-color: var(--good); }

section { padding: 56px 0 8px; }
.section-head {
  display: flex; justify-content: space-between; align-items: flex-end;
  gap: 24px; flex-wrap: wrap;
}
.section-head h2 { margin: 0; font-size: 30px; letter-spacing: -0.01em; }
.section-head .blurb { margin: 6px 0 0; color: var(--dim); max-width: 620px; }
.section-meta { display: flex; gap: 8px; }
.section-meta span {
  font-size: 12px; color: var(--dim); background: var(--bg-raised);
  border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px;
}

h3 {
  margin: 44px 0 14px; font-size: 13px; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--dim);
}
h3 small { letter-spacing: 0.04em; text-transform: none; font-weight: 400; color: var(--faint); }
h3 small::before { content: "· "; }

video, img.sheet { display: block; width: 100%; border-radius: 12px; background: #000; }
.ladder video { border: 1px solid var(--line); box-shadow: 0 24px 60px rgba(0, 0, 0, 0.45); }
p.caption { color: var(--dim); font-size: 14px; max-width: 860px; margin: 14px 0 0; }
p.caption em { color: var(--ink); font-style: italic; }

.cards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }
@media (max-width: 1100px) { .cards { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 700px) { .cards { grid-template-columns: repeat(2, 1fr); } }
.card {
  margin: 0; background: var(--bg-card); border: 1px solid var(--line);
  border-radius: 12px; overflow: hidden;
  border-top: 3px solid var(--accent);
  transition: transform 0.18s ease, border-color 0.18s ease;
}
.card:hover { transform: translateY(-3px); border-color: var(--accent); }
.card-media video { border-radius: 0; aspect-ratio: 600 / 492; object-fit: cover; }
.card figcaption { padding: 12px 14px 14px; }
.card-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 10px;
}
.card-head .latency { font-weight: 700; font-size: 17px; letter-spacing: -0.01em; }
.badge {
  font-size: 11px; padding: 3px 9px; border-radius: 999px;
  border: 1px solid var(--line); color: var(--dim);
}
.badge.good {
  color: var(--good); border-color: rgba(46, 164, 79, 0.4);
  background: rgba(46, 164, 79, 0.08);
}
.badge.bad {
  color: var(--bad); border-color: rgba(214, 69, 65, 0.4);
  background: rgba(214, 69, 65, 0.08);
}
.card dl { margin: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
.card dl div { display: flex; justify-content: space-between; font-size: 12.5px; }
.card dt { color: var(--faint); }
.card dd { margin: 0; color: var(--ink); font-variant-numeric: tabular-nums; }

table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); }
th {
  font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--faint); font-weight: 600;
}
td { color: var(--dim); font-variant-numeric: tabular-nums; }
td:first-child { color: var(--ink); font-weight: 600; }
td.num { text-align: right; }
tr:hover td { background: var(--bg-raised); }
.dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 9px; vertical-align: 1px;
}

img.sheet { border: 1px solid var(--line); }

footer {
  margin-top: 72px; border-top: 1px solid var(--line);
  padding: 32px 0 56px; color: var(--faint); font-size: 13px;
}
footer code {
  color: var(--dim); background: var(--bg-raised);
  padding: 2px 6px; border-radius: 5px;
}
"""

PAGE_JS = """
// Play-on-hover for the per-condition cards; the ladders autoplay.
document.querySelectorAll('.card').forEach(function (card) {
  var video = card.querySelector('video');
  if (!video) return;
  card.addEventListener('mouseenter', function () { video.play().catch(function () {}); });
  card.addEventListener('mouseleave', function () { video.pause(); video.currentTime = 0; });
});

// Highlight the nav entry for the section in view.
var links = Array.prototype.slice.call(document.querySelectorAll('nav.tabs a'));
var sections = links.map(function (a) { return document.querySelector(a.getAttribute('href')); });
var observer = new IntersectionObserver(function (entries) {
  entries.forEach(function (entry) {
    if (!entry.isIntersecting) return;
    links.forEach(function (a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + entry.target.id);
    });
  });
}, { rootMargin: '-40% 0px -55% 0px' });
sections.forEach(function (section) { if (section) observer.observe(section); });
"""


def page_html(sections: list[TaskSection], manifest_path: Path, manifest: dict[str, Any]) -> str:
    nav = "\n".join(f'      <a href="#{section.task}">{section.title}</a>' for section in sections)
    legend = "\n".join(
        f'      <span><i style="background: {LATENCY_COLOURS[latency]}"></i>'
        f"{latency} ms · act every {latency // 20} step"
        f"{'s' if latency > 20 else ''}</span>"
        for latency in sorted(LATENCY_COLOURS)
    )
    bodies = "\n\n".join(task_section_html(section) for section in sections)
    seed = manifest["train_seed"]
    provenance = (
        f"built from <code>{manifest_path}</code> · checkpoints "
        f"<code>{manifest['checkpoint_root']}</code> (training seed {seed}) · "
        f"{manifest['fps']} fps, one frame per 20 ms integration step"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>latency-ml · rollout visualisations</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<header class="hero">
  <div class="wrap">
    <p class="kicker">latency-ml · Phase 4 shared trunk</p>
    <h1>What inference latency <em>looks like</em></h1>
    <p class="lede">
      The experiment's variable is inference latency: the policy decides once, and its
      action is held for <em>N</em> = latency ÷ 20&nbsp;ms environment timesteps. The five
      latency conditions are therefore also five timestep conditions. These are the trained
      policies playing both tasks across that grid — greedy and deterministic, from the
      same paired initial state — next to the sweep's own 200-episode numbers.
    </p>
    <div class="legend">
{legend}
    </div>
  </div>
</header>

<nav class="tabs">
  <div class="wrap">
{nav}
  </div>
</nav>

<main class="wrap">
{bodies}
</main>

<footer>
  <div class="wrap">
    <p>{provenance}</p>
    <p>Rebuild with <code>uv run python scripts/build_site.py</code>;
       serve with <code>uv run python -m http.server 8000 --directory site</code>.</p>
  </div>
</footer>
<script>{PAGE_JS}</script>
</body>
</html>
"""


def copy_assets(sections: list[TaskSection], videos_dir: Path, site_dir: Path) -> None:
    assets = site_dir / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)
    for section in sections:
        shutil.copy2(videos_dir / Path(section.ladder).name, assets / Path(section.ladder).name)
        shutil.copy2(
            videos_dir / Path(section.contact_sheet).name,
            assets / Path(section.contact_sheet).name,
        )
        for condition in section.conditions:
            shutil.copy2(
                videos_dir / Path(condition.video).name, assets / Path(condition.video).name
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--videos",
        type=Path,
        default=Path("results/videos"),
        help="rendered videos directory",
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("results/shared-trunk/shared_trunk.json"),
        help="sweep results, for the 200-episode means",
    )
    parser.add_argument("--out", type=Path, default=Path("site"), help="output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.videos / "render_manifest.json"
    if not manifest_path.exists():
        print(
            f"no render manifest at {manifest_path}; run scripts/render_videos.py first",
            file=sys.stderr,
        )
        return 1

    manifest = load_manifest(manifest_path)
    sweep = load_sweep(args.sweep, int(manifest["train_seed"]))
    sections = build_sections(manifest, sweep)

    args.out.mkdir(parents=True, exist_ok=True)
    copy_assets(sections, args.videos, args.out)
    (args.out / "index.html").write_text(page_html(sections, manifest_path, manifest))

    total = sum(len(section.conditions) for section in sections)
    print(f"site written to {args.out}/ ({len(sections)} tasks, {total} conditions)")
    print(f"serve:  uv run python -m http.server 8000 --directory {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
