"""Browse persistent pilot folders by code. streamlit run dashboard.py"""
import argparse, json, zipfile
from io import BytesIO
from pathlib import Path
import pandas as pd
import streamlit as st
from pilot_runs import list_runs

p = argparse.ArgumentParser()
p.add_argument("--runs-root", default="runs")
a, _ = p.parse_known_args()
st.set_page_config(page_title="Matched-delay pilot", layout="wide")
st.title("Matched-delay pilot")
st.caption(
    "Each policy is deployed at its own training delay. Lower latency is not assumed to win."
)
root = Path(st.sidebar.text_input("Runs folder", a.runs_root))
runs = list_runs(root)
if not runs:
    st.info("No saved runs. Run evaluate_matched.py or save_report.py first.")
    st.stop()
codes = [x.name for x in runs]
requested = st.query_params.get("run", codes[0])
code = st.sidebar.selectbox(
    "Saved run code", codes, index=codes.index(requested) if requested in codes else 0
)
st.query_params["run"] = code
run = root / code
meta = json.loads((run / "run.json").read_text())
st.code(code)
st.caption(
    "Bookmark this URL, or select this code again. Keep/share the run folder: the code is not cloud storage."
)
st.json(meta, expanded=False)
if not (run / "report" / "diagonal.csv").exists():
    st.warning("Run is incomplete; report not yet available.")
    st.stop()
summary = pd.read_csv(run / "report" / "diagonal.csv").sort_values("latency_ns")
per_model = pd.read_csv(run / "report" / "per_model.csv").sort_values(
    ["trained_ns", "model"]
)
metrics = [
    x
    for x in ["rms_error", "tracking", "lost", "p95_amplitude", "settled", "settle_us"]
    if x in summary
]
metric = st.selectbox("Metric", metrics)
st.line_chart(summary.set_index("latency_ns")[[metric]])
st.caption(
    "Numeric latency axis. Error bars and individual training runs are shown in the saved plots."
)
plot = run / "report" / f"{metric}.png"
if plot.exists():
    st.image(str(plot))
st.subheader("Matched-delay summary")
st.dataframe(summary)
st.subheader("Individual policies")
st.dataframe(per_model)
videos = sorted((run / "videos").glob("*.mp4"))
if videos:
    video = st.selectbox("Matched-delay video", videos, format_func=lambda x: x.name)
    st.video(str(video))
else:
    st.caption(
        "No videos saved for this run. Use render_matched.py with --out pointing to this run’s videos folder."
    )
buffer = BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
    for file in run.rglob("*"):
        if file.is_file():
            z.write(file, Path(code) / file.relative_to(run))
st.download_button(
    "Download portable run bundle",
    buffer.getvalue(),
    file_name=code + ".zip",
    mime="application/zip",
)
