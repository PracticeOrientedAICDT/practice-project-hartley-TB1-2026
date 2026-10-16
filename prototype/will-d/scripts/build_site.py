"""Static results site: tabbed viewer with overlaid activation charts and rollout videos."""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import math
import re
import shutil
import sys
from pathlib import Path
from statistics import stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latency_ml.timing import derive_icdi, format_ns

CSS = """
:root { color-scheme: light dark; --ink: #111827; --muted: #6b7280; --line: #e8ebf0; --accent: #2563eb; --orange: #ea580c; --green: #16a34a; }
* { box-sizing: border-box; }
body { font: 15px/1.6 -apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, "Segoe UI", Roboto, sans-serif; color: var(--ink); margin: 0; background: #fafbfc; }
main { max-width: 1280px; margin: 0 auto; padding: 40px 24px 72px; }
h1 { font-size: 24px; font-weight: 650; letter-spacing: -0.01em; margin: 0 0 4px; }
h2 { font-size: 16px; font-weight: 600; margin: 0 0 12px; }
h3 { font-size: 12px; font-weight: 600; margin: 28px 0 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.sub { color: var(--muted); margin: 0 0 24px; }
nav.tabs { display: inline-flex; gap: 2px; margin-bottom: 24px; background: #eef1f5; border-radius: 10px; padding: 3px; }
nav.tabs button { font: inherit; font-size: 14px; padding: 7px 20px; border: none; border-radius: 8px; background: none; cursor: pointer; color: var(--muted); }
nav.tabs button.active { background: #fff; color: var(--ink); box-shadow: 0 1px 3px rgba(16, 24, 40, .12); }
[hidden] { display: none !important; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 20px; }
.card { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 24px; box-shadow: 0 1px 2px rgba(16, 24, 40, .04); }
table { border-collapse: collapse; width: 100%; margin: 0 0 20px; font-size: 13.5px; font-variant-numeric: tabular-nums; }
.twrap { overflow-x: auto; }
tr[data-ms] { cursor: pointer; }
#modal { position: fixed; inset: 0; z-index: 50; }
.modal-backdrop { position: absolute; inset: 0; background: rgba(10, 15, 30, .55); }
.modal-box { position: relative; max-width: 1080px; margin: 6vh auto; background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 20px 22px; }
.modal-vids { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
.modal-vids p { font-size: 12px; color: var(--muted); margin: 0 0 6px; }
.modal-x { position: absolute; top: 10px; right: 14px; font-size: 20px; border: none; background: none; cursor: pointer; color: var(--muted); }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { color: var(--muted); font-weight: 500; font-size: 11.5px; }
tbody tr:hover { background: #f4f7fd; }
th.relu, td.relu { color: var(--accent); }
th.capped, td.capped { color: var(--orange); }
th.hartree, td.hartree { color: var(--green); }
figure { margin: 0; }
figcaption { color: var(--muted); font-size: 12.5px; margin-top: 8px; }
svg { display: block; width: 100%; height: auto; }
svg text { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, "Segoe UI", Roboto, sans-serif; }
.grid { stroke: var(--line); stroke-dasharray: 2 4; }
.axis { stroke: var(--line); }
text { fill: var(--muted); font-size: 10.5px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .92em; }
video { width: 100%; border-radius: 8px; background: #000; }
"""

JS = """
document.querySelectorAll('nav.tabs button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('nav.tabs button').forEach(x => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tabpage').forEach(p => { p.hidden = p.id !== 'tab-' + b.dataset.tab; });
  history.replaceState(null, '', '#' + b.dataset.tab);
}));
(document.querySelector('nav.tabs button[data-tab="' + location.hash.slice(1) + '"]') || document.querySelector('nav.tabs button')).click();
function closeModal() {
  document.getElementById('modal').hidden = true;
  document.querySelectorAll('#modal video').forEach(v => { v.pause(); v.removeAttribute('src'); v.load(); });
}
function openModal(row) {
  const d = row.dataset;
  const key = (s) => s + '|' + d.task + '|' + d.ms + '|' + d.mode;
  const firstUrl = VIDEOS[key(d.relu)] || VIDEOS[key(d.hartree)];
  const trial = d.mode === 'push' && firstUrl ? ' - push trial ' + (firstUrl.match(/push_t(\\d+)/) || [])[1] : '';
  document.getElementById('modal-title').textContent = d.task + ' - ' + d.kind + ' - ' + d.ms + ' ms' + trial;
  const slots = [['relu', 'modal-relu', 'modal-label-relu', d.relu], ['capped', 'modal-capped', 'modal-label-capped', d.capped]];
  slots.push(['hartree', 'modal-hartree', 'modal-label-hartree', d.hartree]);
  slots.forEach(([slot, vid, lab, sweep]) => {
    const name = slot === 'capped' ? 'capped-relu' : slot;
    if (!sweep) { document.getElementById(lab).textContent = name + ' - no sweep yet'; document.getElementById(vid).removeAttribute('src'); return; }
    const v = document.getElementById(vid);
    const url = VIDEOS[sweep + '|' + d.task + '|' + d.ms + '|' + d.mode];
    document.getElementById(lab).textContent = url ? name : name + ' - no recording yet (run latency-ml render)';
    if (url) { v.src = url; v.load(); } else { v.removeAttribute('src'); }
  });
  document.getElementById('modal').hidden = false;
}
document.querySelectorAll('tr[data-ms]').forEach(tr => tr.addEventListener('click', () => openModal(tr)));
document.querySelectorAll('#modal [data-close]').forEach(el => el.addEventListener('click', closeModal));
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
"""

ACTIVATIONS = (("relu", "var(--accent)", "#2563eb"), ("capped-relu", "var(--orange)", "#ea580c"), ("hartree", "var(--green)", "#16a34a"))


def _fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(headers: list[tuple[str, str]], rows: list[list[str]], row_attrs: list[str] | None = None) -> str:
    head = "".join(f'<th class="{cls}">{html.escape(h)}</th>' if cls else f"<th>{html.escape(h)}</th>" for h, cls in headers)
    attrs = row_attrs or [""] * len(rows)
    body = "".join(f"<tr{attr}>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row, attr in zip(rows, attrs, strict=True))
    return f"<div class='twrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _tick(value: float) -> str:
    if abs(value) >= 99.5:
        return f"{value:,.0f}"
    if abs(value) >= 9.95:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round 1-2-5 tick positions covering [lo, hi]."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    norm = raw / mag
    step = (1.0 if norm < 1.5 else 2.0 if norm < 3.0 else 5.0 if norm < 7.0 else 10.0) * mag
    decimals = max(0, -math.floor(math.log10(step)))
    start = math.floor(lo / step) * step
    count = round((math.ceil(hi / step) * step - start) / step)
    return [round(start + i * step, decimals) for i in range(count + 1)]


def _chart_multi(series: list[tuple[str, str, list[tuple[float, float]], list[float]]], y_label: str, x_label: str, ymin: float | None = None, ymax: float | None = None) -> str:
    """Overlaid line chart, one polyline per activation, rendered as inline SVG."""
    w, h = 680, 260
    pad_l, pad_r, pad_t, pad_b = 64, 150, 14, 36
    series = [(label, color, pts, err) for label, color, pts, err in series if pts]
    if not series:
        return ""
    xs = sorted({x for _, _, pts, _ in series for x, _ in pts})
    lo = ymin if ymin is not None else min(y - e for _, _, pts, err in series for (x, y), e in zip(pts, err, strict=True))
    hi = ymax if ymax is not None else max(y + e for _, _, pts, err in series for (x, y), e in zip(pts, err, strict=True))
    ticks = _nice_ticks(lo, hi)
    y_lo, y_hi = ticks[0], ticks[-1]
    x_lo, x_hi = (xs[0], xs[-1]) if len(xs) > 1 else (xs[0] - 1, xs[0] + 1)

    def sx(x: float) -> float:
        return pad_l + (x - x_lo) / (x_hi - x_lo) * (w - pad_l - pad_r)

    def sy(y: float) -> float:
        return h - pad_b - (y - y_lo) / (y_hi - y_lo) * (h - pad_t - pad_b)

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(y_label)} vs {html.escape(x_label)}">',
        f'<text transform="translate(13,{(h - pad_b + pad_t) / 2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>',
        f'<text x="{w - pad_r}" y="{h - 6}" text-anchor="end">{html.escape(x_label)}</text>',
    ]
    for t in ticks:
        parts.append(f'<line class="grid" x1="{pad_l}" y1="{sy(t):.1f}" x2="{w - pad_r}" y2="{sy(t):.1f}"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{sy(t) + 3:.1f}" text-anchor="end">{_tick(t)}</text>')
    parts.append(f'<line class="axis" x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}"/>')
    for _, color, pts, err in series:
        poly = " ".join(f"{sx(x):.1f},{sy(min(max(y, y_lo), y_hi)):.1f}" for x, y in pts)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{poly}"/>')
        for (x, y), e in zip(pts, err, strict=True):
            if e:
                bar_lo, bar_hi = max(y - e, y_lo), min(y + e, y_hi)
                if bar_hi > bar_lo:
                    parts.append(f'<line x1="{sx(x):.1f}" y1="{sy(bar_lo):.1f}" x2="{sx(x):.1f}" y2="{sy(bar_hi):.1f}" stroke="{color}" opacity="0.6"/>')
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(min(max(y, y_lo), y_hi)):.1f}" r="3.5" fill="{color}"/>')
    for i, (label, color, _, _) in enumerate(series):
        ly = pad_t + 8 + i * 18
        parts.append(f'<line x1="{w - pad_r + 8}" y1="{ly}" x2="{w - pad_r + 30}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{w - pad_r + 36}" y="{ly + 4}">{html.escape(label)}</text>')
    for x in xs:
        parts.append(f'<text x="{sx(x):.1f}" y="{h - pad_b + 14:.1f}" text-anchor="middle">{format_ns(int(x))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _activation_of(sweep_id: str) -> str:
    if sweep_id.endswith("-hartree"):
        return "hartree"
    return "capped-relu" if sweep_id.endswith("-capped") else "relu"


def _latency_cell(latency_ns: int) -> str:
    icdi = derive_icdi(int(latency_ns))
    return f"{format_ns(int(latency_ns))} <span class='cap'>({icdi.action_steps}x)</span>"


def _control_card(title: str, groups: dict[str, dict], yrange: tuple[float, float], sweeps: dict[str, str], task: str, kind: str) -> str:
    acts = [a for a, _, _ in ACTIVATIONS if a in groups]
    latencies = sorted({int(r["latency_ns"]) for a in acts for r in groups[a].get("latency_response", [])})
    lookup = {a: {int(r["latency_ns"]): r for r in groups[a].get("latency_response", [])} for a in acts}
    headers = [("latency", "")]
    for a, cls, _ in ACTIVATIONS:
        if a in groups:
            headers += [(f"{a} return", cls), (f"{a} fail", cls)]
    rows = []
    attrs = []
    for lat in latencies:
        row = [_latency_cell(lat)]
        for a in acts:
            r = lookup[a].get(lat, {})
            row += [_fmt(r.get("mean_return")), _fmt(r.get("failure_rate"), 2)]
        rows.append(row)
        ms = lat // 1_000_000
        attrs.append(
            f' data-task="{task}" data-kind="{kind}" data-mode="control" data-ms="{ms}"'
            f' data-relu="{sweeps.get("relu", "")}" data-capped="{sweeps.get("capped-relu", "")}"'
            f' data-hartree="{sweeps.get("hartree", "")}"'
        )
    series = []
    for a, _, color in ACTIVATIONS:
        if a not in groups:
            continue
        pts = [(float(r["latency_ns"]), float(r["mean_return"])) for r in groups[a].get("latency_response", [])]
        err = [float(r.get("std_across_seeds", 0.0)) for r in groups[a].get("latency_response", [])]
        series.append((a, color, pts, err))
    chart = _chart_multi(series, "mean return", "control latency", ymin=yrange[0], ymax=yrange[1])
    caption = "Mean return across seeds (±1 std). Click a row to watch the rollouts."
    card = f"<div class='card'><h2>{html.escape(title)}</h2>{_table(headers, rows, attrs)}"
    return f"{card}<figure>{chart}<figcaption>{caption}</figcaption></figure></div>"


def _push_card(title: str, groups: dict[str, dict], task: str, kind: str) -> str:
    from latency_ml.experiments import push as push_mod

    acts = [a for a, _, _ in ACTIVATIONS if a in groups]
    sweeps = {a: str(groups[a].get("sweep_id", "")) for a in acts}
    latencies = sorted({int(r["latency_ns"]) for a in acts for r in groups[a].get("per_latency", [])})
    lookup = {a: {int(r["latency_ns"]): r for r in groups[a].get("per_latency", [])} for a in acts}
    headers = [("latency", "")]
    for a, cls, _ in ACTIVATIONS:
        if a in groups:
            headers += [(f"{a} tts", cls), (f"{a} success", cls)]
    rows = []
    attrs = []
    for lat in latencies:
        row = [_latency_cell(lat)]
        for a in acts:
            r = lookup[a].get(lat, {})
            row += [_fmt(r.get("mean_tts_ms_penalized")), _fmt(r.get("success_rate"), 2)]
        rows.append(row)
        ms = lat // 1_000_000
        attrs.append(
            f' data-task="{task}" data-kind="{kind}" data-mode="push" data-ms="{ms}"'
            f' data-relu="{sweeps.get("relu", "")}" data-capped="{sweeps.get("capped-relu", "")}"'
            f' data-hartree="{sweeps.get("hartree", "")}"'
        )
    series = []
    for a, _, color in ACTIVATIONS:
        if a not in groups:
            continue
        pts = [(float(r["latency_ns"]), float(r["mean_tts_ms_penalized"])) for r in groups[a].get("per_latency", []) if r.get("mean_tts_ms_penalized") is not None]
        err = []
        for r in groups[a].get("per_latency", []):
            if r.get("mean_tts_ms_penalized") is None:
                continue
            seeds = [v for v in r.get("per_seed_tts_frames_penalized", []) if v is not None]
            # Push runs on cartpole (20 ms env frames); every seed scores, failures included.
            err.append(float(stdev(seeds)) * 20.0 if len(seeds) > 1 else 0.0)
        series.append((a, color, pts, err))
    chart = _chart_multi(series, "mean time to stability, failures at horizon (ms)", "control latency")
    horizon_ms = push_mod.DEFAULT_HORIZON * 20
    caption = (
        "Mean recovery time after a random kick. Failed trials count at the "
        f"{push_mod.DEFAULT_HORIZON}-frame horizon (about {horizon_ms} ms), so failing more can't improve the mean - "
        "read it with the success column. Click a row to watch the rollouts."
    )
    return f"<div class='card'><h2>{html.escape(title)}</h2>{_table(headers, rows, attrs)}<figure>{chart}<figcaption>{caption}</figcaption></figure></div>"


def _collect_controls(results: Path) -> tuple[dict[tuple[str, str], dict[str, dict]], dict[tuple[str, str], dict[str, str]]]:
    """Group evaluation.json files by (task, kind); also map (task, kind, activation) to base sweep id."""
    groups: dict[tuple[str, str], dict[str, dict]] = {}
    sweeps: dict[tuple[str, str], dict[str, str]] = {}
    for fp in sorted(results.glob("*/evaluation.json")) if results.exists() else []:
        name = fp.parent.name
        if "-shared-" in name:
            base, task = name.split("-shared-", 1)
            kind = "shared"
        else:
            base, kind = name, "dedicated"
            task = next((c.name for c in sorted(fp.parent.iterdir()) if c.is_dir() and c.name not in ("push", "videos")), "unknown")
        data = json.loads(fp.read_text())
        groups.setdefault((task, kind), {})[_activation_of(base)] = data
        sweeps.setdefault((task, kind), {})[_activation_of(base)] = base
    return groups, sweeps


def _collect_pushes(results: Path) -> dict[tuple[str, str], dict[str, dict]]:
    groups: dict[tuple[str, str], dict[str, dict]] = {}
    for fp in sorted(results.glob("*/push/*/summary.json")) if results.exists() else []:
        data = json.loads(fp.read_text())
        key = (str(data.get("task", "?")), str(data.get("kind", "?")))
        groups.setdefault(key, {})[str(data.get("activation", "?"))] = data
    return groups


def _collect_videos(results: Path, site_dir: Path) -> list[dict[str, str]]:
    dest_dir = site_dir / "videos"
    dest_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for mp4 in sorted(results.glob("*/videos/*.mp4")) if results.exists() else []:
        sweep = mp4.parent.parent.name
        kind = "shared" if (mp4.parent.parent / "shared").exists() else "dedicated"
        stem = mp4.stem.split("__", 1)[1] if "__" in mp4.stem else mp4.stem
        match = re.match(r"^(.+)_(\d+)ms(_push_t\d+)?_seed\d+$", stem)
        if match is None:
            continue
        dest = dest_dir / f"{sweep}__{mp4.name}"
        if not dest.exists() or dest.stat().st_size != mp4.stat().st_size:
            shutil.copy2(mp4, dest)
        mode = "push" if match.group(3) else "control"
        clips.append({"sweep": sweep, "kind": kind, "task": match.group(1), "ms": match.group(2), "mode": mode, "file": dest.name, "activation": _activation_of(sweep)})
    return clips


def _model_stats() -> list[tuple[str, str, int, str]]:
    """(name, structure, params, note) for every trained architecture, counted live."""
    import torch.nn as nn
    from latency_ml.envs.factory import probe_env_dims
    from latency_ml.models.architectures import CONTROL_HIDDEN, MLPPolicy, bounded_relu
    from latency_ml.models.shared_trunk import SHARED_HIDDEN, SharedTrunkPolicy, probe_task_shapes

    def count(module) -> int:
        return sum(p.numel() for p in module.parameters())

    def chain(*widths: int) -> str:
        return "-".join(str(w) for w in widths)

    clip = bounded_relu().clip
    stats = []
    for task in ("cartpole", "lunarlander"):
        obs_dim, n_actions = probe_env_dims(task)
        policy = MLPPolicy(obs_dim, n_actions, hidden=CONTROL_HIDDEN, activation=nn.ReLU())
        stats.append(
            (
                f"dedicated {task}",
                f"MLP {chain(obs_dim, *CONTROL_HIDDEN, n_actions)} actor + {chain(obs_dim, *CONTROL_HIDDEN, 1)} critic",
                count(policy),
                f"actor {count(policy.actor):,} + critic {count(policy.critic):,} params",
            )
        )
    shapes = probe_task_shapes(("cartpole", "lunarlander"))
    trunk = SharedTrunkPolicy(dict(shapes), activation=nn.ReLU())
    heads = sum(count(trunk.adapters[t]) + count(trunk.actors[t]) + count(trunk.critics[t]) for t in trunk.tasks)
    stats.append(
        (
            "shared trunk",
            f"adapter + {chain(trunk.shared_width, *SHARED_HIDDEN)} trunk + per-task heads",
            count(trunk),
            f"trunk {count(trunk.trunk):,} + heads {heads:,} params",
        )
    )
    stats.append(
        (
            "activations",
            f"relu = ReLU; capped-relu = ReLU clipped at {clip:g}",
            0,
            "same parameter count in both MLP arms; only the nonlinearity differs",
        )
    )
    try:
        from latency_ml.models.hartree import HARTREE_WIDTH, build_hartree_policy, hartree_param_count

        hp = build_hartree_policy("cartpole")
        n = hartree_param_count(hp)
        with contextlib.suppress(Exception):
            hp.close()
        stats.append(
            (
                "dedicated cartpole hartree",
                f"actor: Linear 4-{HARTREE_WIDTH} encoder + frozen hmlp-ti 256-256-256-256 trunk + Linear head; critic: normal MLP",
                n,
                "private hmlp-ti emulator (vendor/, gitignored); actor constrained, critic normal",
            )
        )
    except Exception as exc:
        stats.append(
            (
                "dedicated cartpole hartree",
                "Linear 4-256 encoder + hmlp-ti 256-256-256-256 trunk + Linear heads",
                0,
                f"private hmlp-ti emulator not vendored locally ({exc}); train needs prototype/will-d/vendor/",
            )
        )
    return stats


def _details_tab(results: Path) -> str:
    from latency_ml.config import load_config
    from latency_ml.experiments import push as push_mod

    proto = Path(__file__).resolve().parents[1]
    cfg = load_config(proto / "configs" / "cartpole.yaml")
    t = cfg.training
    latencies = [int(v) for v in cfg.latency.sweep_grid_ns]
    grid = ", ".join(f"{v // 1_000_000} ms ({v // 20_000_000}x)" for v in latencies)
    ded_sweeps = [p.parent.name for p in results.glob("*/evaluation.json")] if results.exists() else []
    ded_sweeps = [s for s in ded_sweeps if "-shared-" not in s]
    shared_sweeps = [p.name for p in results.iterdir() if p.is_dir() and (p / "shared").exists()] if results.exists() else []
    cells = len(latencies) * int(t.control_training_seeds)
    per_cell = {sweep: (2 if sweep in shared_sweeps else 1) * int(t.total_integration_steps) for sweep in ded_sweeps + shared_sweeps}
    total_steps = sum(cells * steps for steps in per_cell.values())

    hyper = [
        ["algorithm", "PPO (duration-aware rollout buffer)"],
        ["learning rate", f"{t.learning_rate:g} (linear anneal)" if t.anneal_lr else f"{t.learning_rate:g}"],
        ["optimizer", "Adam (eps 1e-5)"],
        ["rollout", f"{t.num_envs} envs x {t.steps_per_decision} decisions"],
        ["update", f"{t.update_epochs} epochs x {t.num_minibatches} minibatches"],
        ["clip coef", f"{t.clip_coef:g}"],
        ["discount (cartpole)", f"gamma {t.gamma_phys:g}, lambda {t.gae_lambda_phys:g}, duration-aware"],
        ["discount (lunarlander)", "gamma 0.999, lambda 0.98, plain GAE"],
        ["entropy coef", f"{t.ent_coef:g}"],
        ["value coef", f"{t.vf_coef:g}"],
        ["max grad norm", f"{t.max_grad_norm:g}"],
        ["normalize advantages", "yes" if t.norm_adv else "no"],
    ]
    budget = [
        ["integration steps / cell / task", f"{int(t.total_integration_steps):,}"],
        ["cells / sweep", f"{len(latencies)} latencies x {t.control_training_seeds} seeds = {cells}"],
        ["dedicated sweeps", f"{len(ded_sweeps)} ({', '.join(sorted(ded_sweeps))})"],
        ["shared sweeps", f"{len(shared_sweeps)} ({', '.join(sorted(shared_sweeps))})"],
        ["total env steps (this report)", f"{total_steps:,}"],
        ["project seed", f"{cfg.seed}"],
    ]
    envs = [
        ["cartpole", "CartPole-v1", "4", "2", "20 ms", "500 steps", "pole |theta| > 0.2095 rad or cart |x| > 2.4 m"],
        ["lunarlander", "LunarLander-v3", "8", "4 (discrete)", "20 ms", "1000 steps", "crash / out of bounds (landing between flags scores)"],
    ]
    models = [[name, struct, f"{n:,}" if n else "-", note] for name, struct, n, note in _model_stats()]
    eval_proto = [
        ["policy", "greedy argmax over logits"],
        ["episodes / train seed", f"{cfg.evaluation.control_episodes_per_seed}"],
        ["train seeds", f"{t.control_training_seeds} (0..{int(t.control_training_seeds) - 1})"],
        ["episode seeds", "eval_seed_base + train_seed*100 + episode (paired across latencies)"],
        ["failure rate", "fraction of episodes ending in termination (pole fell / lander crashed)"],
        ["truncation rate", "fraction hitting the gym time limit (survived)"],
        ["mean survival", "simulated env-seconds per episode (hold frames x dt)"],
        ["spread", "std across train-seed means (population std)"],
    ]
    push_proto = [
        ["trials / policy", f"{push_mod.DEFAULT_TRIALS}"],
        ["horizon", f"{push_mod.DEFAULT_HORIZON} env frames"],
        ["kick (external force)", f"± {push_mod.KICK_FORCE_N:g} N on the cart for {push_mod.KICK_STEPS:d} env frames (120 ms); policy keeps acting, pose stays continuous"],
        ["stable band", f"|theta| <= {push_mod.STABLE_THETA_RAD:g} rad and |theta-dot| <= {push_mod.STABLE_THETA_DOT:g} rad/s, held {push_mod.STABLE_HOLD_STEPS} env frames"],
        ["time to stability", "env frames from kick to the start of the first stable window (None = censored); the headline mean scores failures at the horizon"],
        ["success rate", "fraction of trials with a measured TTS"],
        ["trial seeds", "derive_seed(config.seed + 500000, latency, train_seed, trial)"],
    ]
    sections = [
        ("PPO hyperparameters", _table([("parameter", ""), ("value", "")], hyper)),
        ("Training budget", _table([("item", ""), ("value", "")], budget)),
        (
            "Latency model",
            f"<p>Each decision is held for <code>ceil(latency / dt)</code> whole env frames (dt = 20 ms in both tasks), so the grid maps exactly onto 1x..5x holds: {html.escape(grid)}.</p>"
            "<p>No observation or action noise is added; latency is pure action repetition.</p>",
        ),
        ("Environments", _table([("task", ""), ("gym id", ""), ("obs", ""), ("actions", ""), ("dt", ""), ("limit", ""), ("episode ends", "")], envs)),
        ("Models", _table([("model", ""), ("structure", ""), ("params", ""), ("breakdown", "")], models)),
        ("Control evaluation protocol", _table([("item", ""), ("value", "")], eval_proto)),
        ("Push recovery protocol", _table([("item", ""), ("value", "")], push_proto)),
        (
            "Reproduce",
            "<p><code>uv run latency-ml train --phase cartpole --activation relu --sweep-id cp-relu --workers 12 --threads 1</code><br>"
            "<code>uv run latency-ml train --phase cartpole --activation hartree --sweep-id cp-hartree --workers 4 --threads 1</code> (needs private hmlp-ti in <code>vendor/</code>, gitignored)<br>"
            "<code>uv run latency-ml push --sweep-id cp-relu --task cartpole --activation relu</code><br>"
            "<code>uv run latency-ml render --sweep-id cp-relu</code><br>"
            "<code>uv run latency-ml view --build-only</code></p>"
            "<p>Checkpoints live at <code>results/&lt;sweep&gt;/&lt;task&gt;/&lt;latency&gt;ns/seed&lt;n&gt;/policy.pt</code> "
            "(shared: <code>results/&lt;sweep&gt;/shared/...</code>). "
            "Re-runs skip cells whose <code>policy.pt</code> exists.</p>"
            "<p>Hartree arm: obs 4 -&gt; Linear encoder -&gt; sigmoid*255 IN codes (256) -&gt; frozen hmlp-ti "
            "DeviceTorch (f64, draws=False, inference-only) -&gt; y volts (256) -&gt; Linear actor head; "
            "the critic is a normal MLP trained cleanly. PPO trains encoder + actor head + critic; Hartree codes never update.</p>",
        ),
    ]
    body = "".join(f"<h3>{title}</h3>{content}" for title, content in sections)
    return f"<div class='card'>{body}</div>"


def build_site(results: Path, out: Path) -> Path:
    results = Path(results)
    out = Path(out)
    controls, control_sweeps = _collect_controls(results)
    pushes = _collect_pushes(results)
    order = [(t, k) for t in ("cartpole", "lunarlander") for k in ("dedicated", "shared")]
    mlp_activations = ("relu", "capped-relu")
    ranges: dict[str, tuple[float, float]] = {}
    for task in ("cartpole", "lunarlander"):
        vals = []
        for kind in ("dedicated", "shared"):
            for activation in mlp_activations:
                rows = controls.get((task, kind), {}).get(activation, {}).get("latency_response", [])
                vals += [float(r["mean_return"]) for r in rows]
        if vals:
            ranges[task] = (min(vals), max(vals))
    cards = []
    for task, kind in order:
        if (task, kind) in controls:
            mlp_groups = {a: d for a, d in controls[(task, kind)].items() if a in mlp_activations}
            mlp_sweeps = {a: s for a, s in control_sweeps.get((task, kind), {}).items() if a in mlp_activations}
            if mlp_groups and task in ranges:
                cards.append(_control_card(f"{task} - {kind} control", mlp_groups, ranges[task], mlp_sweeps, task, kind))
    for task, kind in (("cartpole", "dedicated"), ("cartpole", "shared")):
        if (task, kind) in pushes:
            cards.append(_push_card(f"{task} - {kind} push recovery", pushes[(task, kind)], task, kind))
    for task, kind in order:
        hartree_data = controls.get((task, kind), {}).get("hartree")
        if hartree_data is None:
            continue
        vals = [float(r["mean_return"]) for r in hartree_data.get("latency_response", [])]
        if not vals:
            continue
        hrange = (min(vals), max(vals))
        hgroups = {"hartree": hartree_data}
        hsweeps = {"hartree": control_sweeps.get((task, kind), {}).get("hartree", "")}
        cards.append(_control_card(f"{task} - {kind} hartree runs", hgroups, hrange, hsweeps, task, kind))
    charts = f"<div class='cards'>{''.join(cards)}</div>" if cards else "<div class='card'><h2>No results yet</h2><p>Run a sweep first.</p></div>"

    clips = _collect_videos(results, out.parent)
    lookup = {f"{c['sweep']}|{c['task']}|{c['ms']}|{c['mode']}": f"videos/{c['file']}" for c in clips}
    details = _details_tab(results)
    modal = (
        "<div id='modal' hidden><div class='modal-backdrop' data-close></div>"
        "<div class='modal-box'><button class='modal-x' data-close aria-label='close'>x</button>"
        "<h2 id='modal-title'></h2><div class='modal-vids'>"
        "<div><p id='modal-label-relu'>relu</p><video id='modal-relu' controls playsinline preload='none'></video></div>"
        "<div><p id='modal-label-capped'>capped-relu</p><video id='modal-capped' controls playsinline preload='none'></video></div>"
        "<div><p id='modal-label-hartree'>hartree</p><video id='modal-hartree' controls playsinline preload='none'></video></div>"
        "</div></div></div>"
    )

    page = (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>will-d latency sweeps</title>"
        f"<style>{CSS}</style></head><body><main>"
        "<h1>will-d: control latency sweeps</h1>"
        f"<p class='sub'>{len(cards)} comparison charts, {len(clips)} rollout videos. Click any table row to watch.</p>"
        "<nav class='tabs'><button data-tab='charts' class='active'>Charts</button><button data-tab='details'>Details</button></nav>"
        f"<div id='tab-charts' class='tabpage'>{charts}</div>"
        f"<div id='tab-details' class='tabpage' hidden>{details}</div>"
        f"{modal}"
        f"<script>const VIDEOS = {json.dumps(lookup, sort_keys=True)};</script>"
        f"<script>{JS}</script></main></body></html>\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static results site.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    written = build_site(Path(args.results), Path(args.out))
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
