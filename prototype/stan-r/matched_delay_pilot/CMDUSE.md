Latency-Aware Control Pilot

This project investigates a simple question:

Does controller performance degrade as latency increases when each policy is trained for that latency?

The default workflow evaluates only the matched-delay diagonal:

100 ns policy -> 100 ns deployment
250 ns policy -> 250 ns deployment
500 ns policy -> 500 ns deployment
...


This avoids confusing delay mismatch with an intrinsic latency limit.

Quick check

Run a very small experiment to verify the pipeline:

python run_demo.py \
  --environment laser \
  --code smoke-laser \
  --budgets-ns 100 500 \
  --seeds 0 \
  --steps 8192 \
  --episodes 2

Full laser experiment
python run_demo.py \
  --environment laser \
  --code laser-pilot-01 \
  --budgets-ns 0 100 250 500 750 1000 2000 \
  --seeds 0 1 2 \
  --steps 500000 \
  --episodes 100 \
  --no-video

Full accelerator experiment
python run_demo.py \
  --environment accelerator \
  --dynamics-profile mild \
  --code accelerator-mild-01 \
  --budgets-ns 0 100 250 500 750 1000 \
  --seeds 0 1 2 \
  --steps 500000 \
  --episodes 100 \
  --no-video

Open results
streamlit run dashboard.py


Select the run code from the sidebar.

Render matched-delay videos
python render_matched.py \
  runs/laser-pilot-01/models \
  --training-seed 0 \
  --out runs/laser-pilot-01/videos


Each tile is evaluated at its own training delay.

Evaluate existing models
python evaluate_matched.py \
  path/to/models \
  --episodes 100 \
  --code existing-models-01


No retraining is performed.

Import old reports
python save_report.py report.html \
  --code previous-report


or

python save_report.py episodes.csv \
  --code previous-results


Imported reports appear in the dashboard like any other run.

Reading the plots

Focus on:

RMS error
Tracking fraction
Loss fraction
Settled fraction

Latencies are always shown in numeric order:

0
100
250
500
750
1000
...


A flat matched-delay curve is a valid result. It means the controller successfully learns to compensate for delay.

Notes
Run codes are local identifiers, not cloud storage.
Keep the saved run folder or export its ZIP from the dashboard.
Use multiple training seeds before drawing conclusions about latency effects.
Videos illustrate individual episodes; reports provide the statistical comparison.
The matched-delay workflow is the primary experiment. Cross-evaluations are available for secondary analysis but are not the main focus of the pilot.