import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    r = Path(a.results)
    out = Path(a.out) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    evals = sorted(r.glob("*/evaluation.json")) if r.exists() else []
    lines = ["<html><body><h1>Control sweeps</h1>"]
    if not evals:
        lines.append("<p>No evaluations yet.</p>")
    for fp in evals:
        data = json.loads(fp.read_text())
        lines.append(
            f"<section><h2>{fp.parent.name}</h2><pre>{json.dumps(data, sort_keys=True)[:2000]}</pre></section>"
        )
    lines.append("</body></html>")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
