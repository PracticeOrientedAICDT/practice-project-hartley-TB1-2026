# Matched-delay control pilot

Does control get worse with latency even when each policy is trained for that latency?
This pilot trains a delay sweep, then evaluates only the diagonal: 100 ns-trained at
100 ns, 500 ns-trained at 500 ns, and so on. It does not confuse delay mismatch with
a physical latency limit. The accelerator and laser plants are simulation benchmarks,
not calibrated hardware models. A flat diagonal is a valid result.

## Install

Use Python 3.11 in your existing environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Run a pilot

```bash
python run_demo.py --environment laser --code laser-pilot-01 \
  --budgets-ns 0 100 250 500 750 1000 --seeds 0 1 2 \
  --steps 250000 --episodes 100
streamlit run dashboard.py
```

For the accelerator, use `--environment accelerator --dynamics-profile mild`.
Profiles are `deterministic`, `mild`, `uncertain`, and `stress`. The laser already
has process disturbance; accelerator profiles do not apply to it. The default actor
is `paper16`; `--arch max` selects three 256-wide layers.

The demo saves models, episode data, plots, a report and matched-delay videos in
`runs/laser-pilot-01/`. Videos show one training seed, ordered by numeric latency,
with at most four policies per grid. Each tile uses its own training delay. Videos
are illustrations of one held-out episode, not statistical evidence of a cliff.

For a quick pipeline check, use `--steps 8192 --episodes 2 --seeds 0
--budgets-ns 100 500 --max-frames 60` with a different code.

## Use models you already trained

```bash
python evaluate_matched.py laser_experiments/models_det \
  --episodes 100 --code laser-existing-01
python render_matched.py laser_experiments/models_det \
  --out runs/laser-existing-01/videos --training-seed 0
```

Pass a model directory, `manifest.json`, or explicit model stems. Do not use a
wildcard that includes the same model's ZIP, JSON and NPZ separately. Models in one
run must have the same plant, profile and architecture. The evaluator uses the
saved configuration unchanged and rejects episode-randomized training budgets.

## Reopen and share results

Each run has a unique code. Select it in Streamlit, or bookmark the URL with its
`?run=CODE` query parameter. Codes cannot overwrite existing folders. The code is a
local identifier, not a cloud backup: keep the run folder or download its portable
ZIP from the dashboard. A collaborator can extract it under their `runs/` folder.

Import an old evaluation CSV or one of our generated HTML reports:

```bash
python save_report.py experiments/results_mild_to_mild/episodes.csv --code mild-old-01
python save_report.py report-5.html --code laser-old-01
streamlit run dashboard.py
```

Only matched rows are presented. HTML imports preserve the available summary;
they cannot recover missing raw episodes, models or videos. The five supplied
reports are already archived as `previous-report-1` through `previous-report-5`.

Use `streamlit run dashboard.py -- --runs-root /path/to/shared/runs` to browse a
shared folder. Store data in your team's chosen shared storage; nothing is uploaded
automatically.

## Read the results

The main plots are RMS error, target occupancy, loss fraction and p95 amplitude.
Latencies are sorted numerically: 0, 100, 250, 500, 750, 1000. Tables retain individual
policies; aggregate curves weight each trained policy equally. Error bars are one
standard error across trained policies, not across pooled evaluation episodes.
With one training seed, there is no between-training uncertainty estimate.

Use the same held-out episode seeds and several training seeds. Check convergence
before attributing worse performance to latency. Settling time is conditional on
success, so inspect loss and settled fractions too. The 500 ns line is a reference,
not a built-in cliff. Compute budgets and measured host runtime are kept separate.

The older full cross-evaluation scripts remain available for secondary mismatch
analysis. They are not used by this demo. See `docs/` for plant assumptions and
uncertainty parameters, and `VALIDATION.md` for what was actually tested.
