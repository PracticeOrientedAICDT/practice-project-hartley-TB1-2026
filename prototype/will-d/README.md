# latency-ml v2

Latency-sensitive neural inference for control and financial decision-making.

> **EXPERIMENT VERSION:** v2
> **ACTIVATION:** Hartley bounded ReLU, clip 2.0
> **SYNTHETIC SWEEP (Phase G):** complete — 130 training cells, 0 failures.
> **HISTORICAL LOB (V2B + L1R):** latency sensitivity **established** (25.0% out-of-sample
> edge decay); temporal context **established** (200-event lookback is reliably worse
> than 20, Δlift −0.044); economic selectivity **not established** (capture −3.41,
> lift +0.008 unresolved). See `RESULTS.md`, and `PROTOCOL.md` sections 12–16.

## What this is

A study of how the latency of a neural policy's inference affects the
effectiveness of the system it drives, and what that implies for applications
that would benefit from a highly compute-efficient neural accelerator.

The research question is in `PROTOCOL.md`, which also records the frozen
experimental design: latency grid, budgets, discounting, seeds, and the claims
this repository is and is not entitled to make.

## What distinguishes it from the previous version

One architectural variable: the nonlinearity.

```text
previous    standard ReLU
this        bounded ReLU, f(x) = 0 for x<=0, x for 0<x<=2, 2 for x>2
```

Defined once, in `src/latency_ml/models/activations.py`.

The gradient is the substantive difference, not the output range. Above the
clip it is zero, so a unit driven into saturation stops receiving learning
signal entirely. That is a different credit-assignment regime rather than a
cosmetic variant, and bounding the range is also a reason to expect a shared
trunk to be *helped* rather than hindered — one task's large activations cannot
dominate another's input — which is a hypothesis to test rather than assume.

Because the activation is the only intended difference, everything else had to
stay put. Two things enforce that:

- `tests/test_parity_v1.py` retrains a reference cell from the previous
  version through this repository's trainer, with an unbounded ReLU **explicitly
  injected**, and requires every weight to match the committed checkpoint under
  `torch.equal`. No tolerance: a reordered RNG draw or a reassociated sum shows
  up in the last bits long before it shows up visibly. This is what licenses
  attributing a later difference to the activation.
- The same suite asserts that the previous version's recorded training protocol
  and this repository's differ in exactly one field.

Both are marked `slow` where they train a full budget: `make test` skips them,
`make check-all` runs them. **Run `make check-all` before launching a sweep.**

## Setup

```bash
make sync     # resolve and install (uv, Python 3.12)
make check    # lint + types + fast tests
make check-all  # the above plus the slow gate
```

Requires `uv`. LunarLander additionally needs Box2D, installed as a default
extra.

## Layout

```text
configs/              experiment configuration
PROTOCOL.md           the frozen design and the claims it licenses
docs/ENVIRONMENT.md   dependency pins, and the traps behind them
src/latency_ml/
  models/             policies, shared trunk, activations, chip-compatibility
  rl/                 PPO, rollout buffer, the TaskPolicy interface
  envs/               action-hold latency wrapper
  timing.py           latency grid, decision-interval derivation, resolution states
  hardware/           MAC counting and the accelerator contract
  finance/            LOBSTER loading, event-time replay, features
  experiments/        sweep runners, manifests, matched evaluation
  analysis/           protocol fingerprints, aggregation, Phase 4 reporting
tests/                including tests/fixtures/v1/, frozen previous-version artefacts
results/              output of this repository's own runs. Currently empty.
data/lobster/         downloaded LOBSTER sessions (gitignored)
```

## Two conventions worth knowing before reading any code

1. **All latency is integer nanoseconds internally.** Conversion to float
   seconds happens only at the Gymnasium boundary.

2. **Resolution is three-state, not a boolean.** A requested latency may be
   exactly representable, silently quantised upward, or below what the simulator
   can resolve at all. The third must never be plotted as an independent
   measurement, and the second must never be plotted as distinct from the
   condition it quantises onto.

## Running things

```bash
# a control latency sweep, at the budget the config declares
latency-ml sweep --config configs/cartpole.yaml --sweep-id cp

# the shared-trunk arm, same trainer with several tasks
latency-ml shared-trunk --config configs/base.yaml --out results/shared-trunk

# fixed-policy latency sweep on a real LOBSTER session
latency-ml historical --data data/lobster/LOBSTER_SampleFile_SPY_2012-06-21_30 \
                      --out results/historical.json

# figures, regenerated from committed results rather than redrawn
latency-ml plot latency-response --evaluation <file> --out <file>
```

`latency-ml --help` lists everything. Every run writes a manifest **before**
training and marks it completed only on success, recording its realised budget,
its resolved protocol fingerprint and the library versions that produced it — so
a result can be traced to the run that made it.

## Status

The activation comparison is complete and the result is **null**: no detectable
effect on latency sensitivity, with v1 s latency boundary and convergence-reliability
findings both replicating. `RESULTS.md` has the numbers and the power analysis
behind that conclusion. The previous version's
committed results are **not** here: they live under `tests/fixtures/v1/` as
frozen external references, read-only inputs to the parity suite rather than
output this repository could accumulate by accident.