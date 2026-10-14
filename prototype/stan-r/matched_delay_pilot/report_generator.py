"""Diagonal reports, numerically ordered and explicit about uncertainty units."""
import argparse, html
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = [
    "rms_error",
    "mean_error",
    "p95_amplitude",
    "tracking",
    "lost",
    "settled",
    "settle_us",
]


def diagonal(df):
    df = df.copy()
    for c in ["trained_ns", "inference_ns"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isclose(df.trained_ns, df.inference_ns, atol=1e-6, rtol=0)].copy()
    if df.empty:
        raise ValueError("No matched train/deploy rows found")
    return df.sort_values(["trained_ns", "model"])


def summarize(df):
    df = diagonal(df)
    for key in ["environment", "dynamics_profile", "backend", "latency_source"]:
        if key in df and df[key].nunique(dropna=False) > 1:
            raise ValueError(
                f"Mixed {key} values; save separate runs rather than pool them"
            )
    rows = []
    for (budget, model), g in df.groupby(["trained_ns", "model"], sort=True):
        row = dict(trained_ns=budget, model=model, episodes=len(g))
        for metric in METRICS:
            if metric not in g:
                continue
            v = pd.to_numeric(g[metric], errors="coerce").dropna()
            row[metric] = float(v.mean()) if len(v) else np.nan
            row[metric + "_n"] = len(v)
            row[metric + "_se"] = (
                float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["trained_ns", "model"])


def aggregate(per_model):
    rows = []
    for budget, g in per_model.groupby("trained_ns", sort=True):
        row = dict(latency_ns=budget, training_runs=len(g))
        for metric in METRICS:
            if metric not in g:
                continue
            v = g[metric].dropna()
            row[metric] = float(v.mean()) if len(v) else np.nan
            row[metric + "_n_models"] = len(v)
            row[metric + "_se_between_models"] = (
                float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("latency_ns")


def write_report(per_model, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per_model = per_model.sort_values(["trained_ns", "model"])
    summary = aggregate(per_model)
    per_model.to_csv(out / "per_model.csv", index=False)
    summary.to_csv(out / "diagonal.csv", index=False)
    plots = []
    for metric in METRICS:
        if metric not in summary:
            continue
        fig, ax = plt.subplots()
        ax.plot(
            summary.latency_ns,
            summary[metric],
            marker="o",
            label="Mean across trained policies",
        )
        se = summary[metric + "_se_between_models"]
        valid = se.notna() & summary[metric].notna()
        if valid.any():
            ax.errorbar(
                summary.latency_ns[valid],
                summary.loc[valid, metric],
                yerr=se[valid],
                fmt="none",
            )
        ax.scatter(
            per_model.trained_ns, per_model[metric], label="Individual trained policies"
        )
        ax.axvline(500, linestyle=":", label="500 ns reference, not a proven cliff")
        ax.set_xlabel("Matched training and deployment compute delay (ns)")
        ax.set_ylabel(metric)
        ax.legend(fontsize="small")
        fig.tight_layout()
        name = metric + ".png"
        fig.savefig(out / name, dpi=140)
        plt.close(fig)
        plots.append(name)
    note = (
        "Each model is evaluated only at its training delay. Curves are numerically sorted. "
        "Points are means across evaluation episodes; aggregate means weight trained policies equally. "
        "Error bars show one standard error between trained policies when at least two exist. "
        "A single training seed cannot establish training robustness. Settling time is conditional on success; "
        "inspect settled and loss fractions alongside it. 500 ns is a reference, not an imposed failure threshold. "
        "Simulation budgets are separate from measured host runtime. Profiles are not machine calibrated."
    )
    page = (
        '<html><head><meta charset="utf-8"><title>Matched-delay pilot</title></head><body><h1>Matched-delay pilot</h1><p>'
        + html.escape(note)
        + "</p>"
    )
    page += summary.to_html(index=False)
    for name in plots:
        page += f'<h2>{name}</h2><img src="{name}" style="max-width:100%">'
    page += (
        "<h2>Individual policies, ordered by latency</h2>"
        + per_model.to_html(index=False)
        + "</body></html>"
    )
    (out / "report.html").write_text(page)
    (out / "report.md").write_text("# Matched-delay pilot\n\n" + note + "\n")
    return summary


def generate(csv, out):
    return write_report(summarize(pd.read_csv(csv)), out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--out", default="report")
    a = p.parse_args()
    generate(a.csv, a.out)


if __name__ == "__main__":
    main()
