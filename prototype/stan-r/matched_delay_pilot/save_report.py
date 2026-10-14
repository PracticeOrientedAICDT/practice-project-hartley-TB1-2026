"""Archive existing cross-evaluation CSV or generated HTML as a diagonal run."""
import argparse, shutil
from io import StringIO
from pathlib import Path
import pandas as pd
from pilot_runs import create_run, finish_run
from report_generator import diagonal, summarize, write_report


def save(source, root="runs", code=None):
    source = Path(source)
    if source.suffix.lower() in (".html", ".htm"):
        tables = pd.read_html(StringIO(source.read_text()))
        table = next(
            (
                t
                for t in tables
                if {"model", "trained_ns", "inference_ns", "rms_error"}
                <= set(t.columns)
            ),
            None,
        )
        if table is None:
            raise ValueError("No compatible summary table found in HTML")
        matched = diagonal(table).rename(columns={"inference_ns": "deployed_ns"})
        if matched.duplicated(["model", "trained_ns"]).any():
            raise ValueError("Multiple profiles in report; import separately")
        per_model = matched
        kind = "imported-summary"
        warning = (
            "Imported summary only: raw episodes and original videos are unavailable."
        )
    else:
        df = pd.read_csv(source)
        per_model = summarize(df)
        kind = "imported-episodes"
        warning = "Imported existing matched rows; no new evaluation performed."
    run = create_run(root, code, kind=kind, source_name=source.name, note=warning)
    shutil.copy2(source, run / ("source" + source.suffix))
    if kind == "imported-episodes":
        diagonal(df).to_csv(run / "episodes.csv", index=False)
    write_report(per_model, run / "report")
    finish_run(run)
    print(f"Saved run code: {run.name}")
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--code")
    a = p.parse_args()
    save(a.source, a.runs_root, a.code)


if __name__ == "__main__":
    main()
