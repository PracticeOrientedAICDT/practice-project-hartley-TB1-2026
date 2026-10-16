from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.build_site import _nice_ticks, build_site


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _latencies() -> list[dict]:
    return [
        {"latency_ns": lat, "action_steps": steps, "mean_return": 500 - 40 * steps, "std_across_seeds": 12.5, "failure_rate": 0.0, "mean_survival_s": 10.0, "n_seeds": 10}
        for lat, steps in ((20000000, 1), (40000000, 2), (60000000, 3))
    ]


def _push_rows() -> list[dict]:
    return [
        {
            "latency_ns": 20000000,
            "action_steps": 1,
            "n_seeds": 10,
            "n_trials": 200,
            "success_rate": 0.85,
            "mean_tts_frames": 20.0,
            "mean_tts_decisions": 20.0,
            "mean_tts_ms": 400.0,
            "mean_tts_frames_penalized": 65.0,
            "mean_tts_ms_penalized": 1300.0,
            "per_seed_tts_frames": [18.0, 22.0],
            "per_seed_tts_frames_penalized": [60.0, 70.0],
        },
        {
            "latency_ns": 100000000,
            "action_steps": 5,
            "n_seeds": 10,
            "n_trials": 200,
            "success_rate": 0.4,
            "mean_tts_frames": 120.0,
            "mean_tts_decisions": 24.0,
            "mean_tts_ms": 2400.0,
            "mean_tts_frames_penalized": 310.0,
            "mean_tts_ms_penalized": 6200.0,
            "per_seed_tts_frames": [100.0, 140.0],
            "per_seed_tts_frames_penalized": [300.0, 320.0],
        },
    ]


def _eval_payload(sweep_id: str) -> dict:
    return {"sweep_id": sweep_id, "n_policies": 10, "latency_response": _latencies()}


def _push_payload(sweep_id: str, activation: str) -> dict:
    return {"sweep_id": sweep_id, "task": "cartpole", "kind": "dedicated", "activation": activation, "per_latency": _push_rows()}


def test_build_site_overlays_both_activations(tmp_path):
    results = tmp_path / "results"
    for sweep in ("cp-relu", "cp-capped"):
        _write(results / sweep / "evaluation.json", _eval_payload(sweep))
        (results / sweep / "cartpole").mkdir(parents=True, exist_ok=True)
        activation = "relu" if sweep == "cp-relu" else "capped-relu"
        _write(results / sweep / "push" / "dedicated" / "summary.json", _push_payload(sweep, activation))
    out = build_site(results, tmp_path / "site" / "index.html")
    html = out.read_text()
    charts = html.split("tab-charts")[1].split("tab-details")[0]
    assert "data-tab='charts'" in html and "data-tab='details'" in html
    assert "data-tab='videos'" not in html
    assert charts.count("<table>") == 2  # one control card + one push card
    assert charts.count("<svg") == 2
    assert charts.count("<polyline") == 4  # relu + capped-relu on each chart
    assert "cartpole - dedicated control" in html
    assert "cartpole - dedicated push recovery" in html
    assert "failing more can&#x27;t improve the mean" in html or "failing more can't improve the mean" in html
    assert "1300" in html  # penalized, not success-only, TTS in the table
    assert "20 ms" in html and "100 ms" in html
    assert "PPO hyperparameters" in html
    assert "dedicated cartpole" in html
    assert "34,691" in html  # live parameter count, thousands-separated
    assert 'data-task="cartpole" data-kind="dedicated" data-mode="control" data-ms="20"' in html
    assert 'data-task="cartpole" data-kind="dedicated" data-mode="push" data-ms="20"' in html
    assert 'data-relu="cp-relu" data-capped="cp-capped"' in html
    assert "id='modal'" in html and "const VIDEOS" in html


def test_build_site_copies_videos_into_site(tmp_path):
    results = tmp_path / "results"
    _write(results / "cp-relu" / "evaluation.json", _eval_payload("cp-relu"))
    (results / "cp-relu" / "cartpole").mkdir(parents=True, exist_ok=True)
    videos = results / "cp-relu" / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    (videos / "cartpole_20ms_seed5.mp4").write_bytes(b"fake-mp4")
    (videos / "cartpole_20ms_push_t2_seed5.mp4").write_bytes(b"fake-push-mp4")
    out = build_site(results, tmp_path / "site" / "index.html")
    html = out.read_text()
    assert (tmp_path / "site" / "videos" / "cp-relu__cartpole_20ms_seed5.mp4").exists()
    assert (tmp_path / "site" / "videos" / "cp-relu__cartpole_20ms_push_t2_seed5.mp4").exists()
    assert "<video" in html  # modal player skeleton
    assert "cp-relu__cartpole_20ms_seed5.mp4" in html  # VIDEOS lookup map
    assert "id='modal'" in html


def test_nice_ticks_are_round():
    assert _nice_ticks(174.3, 500.0) == [100, 200, 300, 400, 500]
    assert _nice_ticks(-252.1, 270.3) == [-300, -200, -100, 0, 100, 200, 300]
    assert _nice_ticks(280.9, 1500.1) == [0, 500, 1000, 1500, 2000]


def test_control_charts_share_y_axis_per_task(tmp_path):
    results = tmp_path / "results"
    for sweep in ("cp-relu", "cp-capped"):
        _write(results / sweep / "evaluation.json", _eval_payload(sweep))
        (results / sweep / "cartpole").mkdir(parents=True, exist_ok=True)
    for sweep in ("sh-relu", "sh-capped"):
        (results / sweep / "shared").mkdir(parents=True, exist_ok=True)
        _write(results / f"{sweep}-shared-cartpole" / "evaluation.json", _eval_payload(sweep))
    out = build_site(results, tmp_path / "site" / "index.html")
    svgs = re.findall(r"<svg.*?</svg>", out.read_text())
    assert len(svgs) == 2
    ticks = [re.findall(r'x="56"[^>]*>([^<]+)', svg) for svg in svgs]
    assert len(ticks[0]) > 1
    assert ticks[0] == ticks[1]


def test_build_site_handles_empty_results(tmp_path):
    out = build_site(tmp_path / "missing", tmp_path / "site" / "index.html")
    html = out.read_text()
    assert "No results yet" in html
    charts = html.split("tab-charts")[1].split("tab-details")[0]
    assert "<table>" not in charts
    assert "PPO hyperparameters" in html  # method details render regardless
