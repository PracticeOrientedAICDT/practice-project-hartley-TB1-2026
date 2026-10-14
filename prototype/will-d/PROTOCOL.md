# PROTOCOL

The frozen experimental protocol. Everything here is held constant across
conditions; changing any of it changes the experiment and should be a deliberate
edit to this file rather than a runtime option.

This document replaces the specification and implementation-plan documents of
the previous repository. They were largely a record of how that experiment was
built, including a design that specified an unbounded ReLU and a LOBSTER layout
that turned out to be wrong. What survives here is the part that is still true.

---

## 1. The question

How does neural-inference latency affect the effectiveness of the resulting
decision system, and what applications would benefit from a highly
compute-efficient neural accelerator?

This repository exists to isolate **one** architectural variable: the
nonlinearity. The comparison is against a previous repository whose only
intended difference is that variable. Section 8 states what has to be true for
that comparison to mean what it claims.

## 2. Activation

Hartley's bounded ReLU, defined once in `src/latency_ml/models/activations.py`:

```text
f(x) = 0    for x <= 0
f(x) = x    for 0 < x <= CLIP
f(x) = CLIP for x > CLIP        CLIP = 2.0
```

`CLIP` is a module constant, not a configuration field. Changing it changes the
experiment.

The gradient is one strictly inside the interval and zero outside it, and zero
at the boundary as well — the function is continuous at both ends but its
derivative is not, matching `torch.clamp` and `hardtanh`. This is the
substantive difference from an unbounded ReLU and the reason to expect different
behaviour rather than a cosmetic variant: a unit pushed above the clip receives
no learning signal at all.

Models accept an explicit `nn.Module` in place of the default. Production code
never passes one; the parity suite does, in order to reproduce a reference
checkpoint trained under an unbounded ReLU. A named-activation registry would
imply a menu of supported options that does not exist.

## 3. Architectures

Chip-constrained, fixed across every latency condition.

| Role | Shape |
|---|---|
| Control, CartPole | `4 -> 128 -> 128 -> 2` |
| Control, LunarLander | `8 -> 128 -> 128 -> 4` |
| Shared trunk | `128 -> 128 -> 128`, per-task `obs -> 128` adapters and heads |
| HFT small / large | `n_features -> 32 -> 32 -> 3` / `-> 128 -> 128 -> 3` |

Control trunk: 17,152 MACs. Within the 4-layer / 256-width accelerator contract.

Each policy carries a critic with a matching trunk. It is a training artefact
and is excluded from accelerator accounting: benchmarking measures the policy
path, because that is the part that selects an action.

## 4. Latency

All latency is **integer nanoseconds** internally. Conversion to float seconds
happens only at the Gymnasium boundary.

### Inference-constrained decision interval

For an environment that can only change action on its own ticks:

```text
N            = max(1, ceil(latency_ns / dt_env_ns))
effective_ns = N * dt_env_ns
```

### Resolution is three-state, not a boolean

| State | Meaning |
|---|---|
| `EXACT` | `effective_ns == requested_ns` |
| `QUANTIZED_UP` | representable request that was rounded up to a coarser condition |
| `UNRESOLVED` | below what the simulator can resolve at all |

A boolean cannot distinguish an exactly-representable condition from one
silently rounded onto a different one. `UNRESOLVED` must never be plotted as an
independent measurement, and `QUANTIZED_UP` must collapse onto its
`effective_ns` rather than appear as a second point.

### Control sweep grid

```text
20 ms  40 ms  60 ms  80 ms  100 ms
```

Integer multiples of the 20 ms native timestep, giving a 1–5 held-step sweep
with no quantisation. Sub-20 ms conditions are unresolvable natively and are
marked `UNRESOLVED` rather than measured.

### HFT sweep grid

```text
100 ns   300 ns   1 us    3 us    10 us   30 us
100 us   300 us   1 ms    3 ms    10 ms   30 ms   100 ms
```

Reaches the sub-millisecond regime because in event-time replay latency is a
timestamp shift rather than a held action, so the market stream resolves it.

## 5. Budgets and fairness

Equal **simulated task time** across latency conditions, measured in
environment integration steps. A slower policy simply gets fewer decisions in
the same simulated time — that is the phenomenon under study, not a confound to
be corrected away.

```text
total_integration_steps  4,000,000 per task per condition
training_seeds          10 (control arms), 3 (shared-trunk arms)
evaluation_episodes     200 per trained policy
```

Iterations follow as `budget // (num_envs * steps_per_decision * action_steps)`,
so they *fall* with latency: 7,812 updates at 20 ms, 1,562 at 100 ms, for the
4M budget.

### Fixed minibatch

```text
steps_per_decision   128      (fixed for every action-hold length)
num_minibatches      4        (fixed)
minibatch_size       128      invariant across latencies -- asserted in tests
```

An earlier draft held transitions-per-iteration constant by shrinking the
minibatch from 128 to 25 as latency rose. That makes gradient noise grow in
lockstep with latency, confounding "latency degrades performance" with "smaller
minibatches degrade optimisation". Rejected.

`gradient_steps` falling with latency (~125,000 to ~25,000 across the grid) is a
real consequence of equal simulated-time exposure, biased *against* the
high-latency conditions. It is logged per run rather than corrected, so it can be
reported instead of inferred.

## 6. Discounting

Action holding makes one transition span a variable amount of physical time, so
discounting has to know the hold length.

CartPole uses the **physical** basis: `gamma_i = gamma_phys ** N_i`, with
duration-aware GAE.

LunarLander uses the **decision** basis — `gamma = 0.999`, `gae_lambda = 0.98`
applied unscaled — giving a credit horizon of roughly 48 decisions.

This is a deliberate per-task departure, and the reason is invariance rather
than tuning. Under the physical basis the per-decision credit horizon shrinks as
latency rises (about 17 decisions at 20 ms, about 4 at 100 ms), so raising
latency would change credit assignment at the same time as control bandwidth.
The decision basis holds the horizon still. The values are taken from the RL
Baselines3 Zoo LunarLander-v3 entry as an external reference; no search was run.

## 7. Optimisation

Frozen PPO hyperparameters, identical across every condition.

```text
algorithm          PPO (CleanRL hyperparameters, env handling rewritten)
learning_rate      2.5e-4, annealed
num_envs           4
steps_per_decision 128
update_epochs      4
clip_coef          0.2
ent_coef           0.01
vf_coef            0.5
max_grad_norm      0.5
norm_adv           true
clip_vloss         true
anneal_lr          true
```

The learning rate is annealed on **fraction of budget in integration steps**,
not on update count, so every condition traverses the same schedule at equal
simulated progress. It is denominated in the *effective* budget rather than the
configured one: using the configured value made any run with an overridden
budget reach `lr = 0` halfway and then sit frozen while still reporting the
larger budget.

Termination and truncation stay distinct throughout. A truncated state is
artificially time-limited and valid, so its value is bootstrapped; a terminated
one is genuinely terminal and is not. Conflating them biases the value target
precisely for the outcome whose frequency changes with latency.

## 8. Seeds

Three roles, derived differently on purpose.

- **training** — the cell's `train_seed` verbatim. Latency is deliberately *not*
  mixed in, so every latency condition at a given replicate draws the same
  initial weights and the same minibatch order. Common random numbers: a
  difference between two latencies is not partly a difference in which
  minibatches happened to be drawn. Conditions still train on entirely different
  data, because the action-hold factor changes the state trajectory.
- **env** — `config.seed + 10_000`, mixed with latency and the replicate index.
- **eval** — `eval_seed_base`, reused across every condition so comparisons are
  paired.

Weight initialisation draws from the global torch RNG, so **seeding must precede
policy construction**. A policy built from ambient state gets starting weights
its seed does not determine, and re-running the cell will not reproduce them.
The caller owns this ordering, since the trainer receives an already-built
policy.

Every run writes a manifest before training starts and marks it completed only on
success, carrying its realised counts, its config hash and the resolved library
versions. A protocol fingerprint is recorded per task, so a shared-versus-dedicated
comparison can be re-verified from the artifacts rather than from trust in the
code.

## 9. What must hold for a comparison to be attributable

Two things, both enforced rather than asserted.

**The trainer is numerically inert.** `tests/test_parity_v1.py` retrains a v1
reference cell through the current trainer with an explicitly injected unbounded
ReLU and requires every tensor to equal the committed v1 checkpoint under
`torch.equal` — no tolerance. A reordered RNG draw or a reassociated sum shows up
in the last bits long before it shows up visibly, and "close enough" would defeat
the purpose. Two of its tests train the full 4M budget and are marked `slow`:
`make test` skips them, `make check-all` runs them.

**The protocol differs in exactly one field.** The same suite asserts that v1's
recorded protocol and this repository's differ only in `activation`, field by
field against v1's own recorded configuration. A discount factor or budget that
had also changed would leave the result with two candidate causes.

The activation is injected explicitly in the gate rather than defaulted,
precisely because the default here is the bounded activation.

### A third thing: the design's resolution is stated before the verdict

**A null result is a statement about the experiment until proven otherwise.** "No
difference was detected" and "no difference exists" are different claims, and the
first is what a finite design can actually deliver. Confusing them is the most
common way a replication study quietly becomes an equivalence study.

So every comparison in this repository reports three things, not one: the
observed paired difference, the **minimum detectable effect** at the design's
own power, and the verdict those two together license. The verdicts are
`detected` or `inconclusive`. There is deliberately no "no effect" verdict,
because `latency_ml.analysis.power` has no way to produce one and the module
would be trivial if it did.

Seed-wise pairing is required for the same reason. The same seed draws the same
initial weights and the same minibatch order on both sides, so the arms are not
independent samples; differencing seed by seed removes between-seed variance,
which is the dominant term here. Neither side may be sorted before differencing —
that preserves the appearance of a paired test while destroying the pairing, and
it shrinks the standard error, so the error is in the direction that flatters a
null.

This was not in the original protocol, which is itself the finding: the first
comparison run here came back with point estimates a fifth to a fiftieth of what
the design could resolve, and nothing in the protocol had required that be
computed before writing the conclusion. The power calculation is now part of the
protocol and is executed by `latency_ml.analysis.power.compare_paired` rather
than by hand, so it cannot drift.

## 10. Hardware model: unfrozen

No accelerator model is in scope. Throughput, precision, fixed overhead and
memory remain `TBD`, and any path that would need them raises rather than
inventing a number.

Consequently **no hardware latency figure is reported**, and the 2x2
capacity-versus-speed experiment is not run. Latency is treated as the
independent variable throughout, which is what the control experiments need and
what the financial replay can support without a frozen model.

## 11. Financial replay

Fixed across every latency condition, so latency is the only intervention:
instrument and session, chronological event sequence, input features, action
space, train/validation split, optimisation procedure, transaction costs,
execution logic.

Execution is the simplest defensible model: an aggressive fill against the book
visible at execution time, marked out over a fixed horizon, with a flat
transaction cost. Queue position, partial fills and exchange matching are out of
scope.

Each run records a content hash of its event stream. Every latency point in a
sweep must replay identical data, and that claim is checkable from the artifact
rather than taken on trust.

Results are historical replay evidence, not evidence of profitability.

### Two claims, not one

The synthetic generator contains a short-lived latent order-flow state that
creates predictive but decaying information. Its oracle mechanism test measures
whether opportunity decays under a *correct* decision delayed — the decay comes
from the market process, and no policy is involved.

Historical data carries no such ground truth. `signal` is zero throughout by
design: fabricating one would leak into a policy input through the
`signal_alive_ratio` feature. So on real data the oracle test **cannot** run, and
the available claim is narrower:

- **synthetic**: latency reduces the value of a *known* opportunity.
- **historical**: latency reduces the realised value of a *trained* policy.

They are different claims and are written up separately. The mechanism's decay is
never parameterised as a function of latency; latency enters only at replay, so
the response is an interaction rather than a restated input.

## 12. Frozen V2B design

Fixed before any training, on evidence from `latency-ml diagnose-target` and
nothing else. Nothing in this section may be adjusted after model performance is
seen; a change here voids the run.

### Target

| | |
|---|---|
| Horizon `H` | **10 s**, as a duration: `t_H = min{t_i : t_i - t_0 >= H}` |
| Band `k` | **6 price granules** (3 cents; the granule is a GCD of observed mid differences, 50 units on these sessions) |
| Labels | three-class BUY / HOLD / SELL, inclusive on both sides |
| Execution | aggressive touch crossing, fixed |
| Latency sweep | 0.1, 1, 10, 30, 100 ms |

The input interface is the existing seven features, and one of them is inert on real
data. `signal_alive_ratio` is derived from the hidden `signal` array, which
`lobster.py` sets to zero throughout by design — fabricating a latent state would
leak into a policy input. So the seven-dimensional interface carries **six
informative features and one constant**, and the normalised column is identically
zero. This is recorded rather than changed: the feature dimension is frozen, and L1
varied context *within* this representation, so a constant column is present in every
arm equally. It does mean "seven features" should not be read as seven sources of
information.

`H` is a duration and not an event count. On AAPL the median event gap is
0.639 ms but the mean is 58.44 ms — a 91x ratio, because the book is bursty — so
the same event count names windows of wildly different duration, and reasoning
from the median spacing is wrong by two orders of magnitude. A 100 ms horizon
buys about 45 events.

`k` is a granule count and not a quantile of absolute moves. A quantile
degenerates on a step-function mid: if the share of unmoved windows exceeds the
quantile it lands in the point mass at zero, and a zero band makes
`build_labels` collapse the three-class problem to `sign(move)`. Measured at
`H` = 10 s on AAPL, 26% of windows are HOLD, so the band is not degenerate and
the split is 36 / 26 / 39.

### Arms

| Arm | Symbol | `H` | `k` | Role |
|---|---|---|---|---|
| Primary | AAPL | 10 s | 6 | development and the main economic claim |
| Secondary | AAPL | 5 s | 6 | deliberately latency-sensitive stress arm (47.7% decay vs 14.6%) |
| Held out | AMZN | 25 s | 6 | cross-symbol validation at its own viable horizon |
| Reference | INTC, MSFT | — | — | diagnostics only; no economic claim |

AMZN is held out, never tuned on. Its horizon is established independently from
its own viability table and is **not** matched to AAPL's; the comparison is
cross-symbol, not like-for-like, and is labelled as such.

INTC and MSFT are excluded from the primary claim on **absolute value**, not on
signal scarcity. In wall-clock terms they are not degenerate — their half-spread
is one granule, so post-cost survival reaches 95–100% and they clear the spread
at almost every horizon. Their net outcome is simply 0.2–1.3 granules against
AAPL's 24.6, because the moves available are smaller than the moves available on
AAPL by more than the cost advantage. Signal viability and economic magnitude are
separate properties and only the second one disqualifies them.

### Identifiability, checked before training

`V(tau)` is a ceiling: the mean post-cost outcome of a **perfect** forecaster
whose signals land at `t_0 + tau`. A trained policy earns some share `q` of it,
so the gap the sweep can hope to see is `q * (V(tau_fast) - V(tau_slow))`.

The floor is the share of oracle value a policy must capture for that gap to
clear the minimum detectable effect at 80% power, alpha = 0.05, on a **single
session**:

| Arm | gap (granules) | effective windows | SE | MDE | `q` floor |
|---|---|---|---|---|---|
| Primary 10 s / k6 | 0.923 | 2,340 | 0.134 | 0.375 | **0.407** |
| Secondary 5 s / k6 | 1.190 | 4,680 | 0.093 | 0.261 | **0.219** |
| Long 25 s / k6 | 0.615 | 936 | 0.214 | 0.600 | 0.976 |

Effective windows are `session span / H`, not the observation count. The
look-ahead windows overlap, so consecutive observations share nearly all their
events and are not independent draws; AAPL offers about 2,340 non-overlapping
windows at a 10 s horizon, not 400,000 observations. Using the raw count would
understate the standard error by roughly thirteen and would license a design that
cannot resolve its own effect.

Consequences, stated plainly:

- On one session a policy below roughly **41%** of oracle produces a null. At
  `q` = 0.2 the expected gap is 0.18 against an MDE of 0.38.
- The floor falls as `1/sqrt(n_sessions)`. The experiment is not run on one
  session; across ten the primary floor falls to about **0.13**, at which even a
  20%-of-oracle policy resolves the effect.
- A 25 s horizon is close to unresolvable at any model quality on this much data
  (floor 0.98), which is a second reason it is not the primary beyond its weak
  latency sensitivity.
- The secondary arm is **better** powered than the primary (0.22 vs 0.41), not
  worse: its gap is larger and it has twice the effective windows. It is
  high-sensitivity, not fragile.

A null result below these floors is a statement about the model, not about
latency having no effect. That distinction has to be reported.

## 13. V2B primary result, and two withdrawn claims

The definitive V2B result is the out-of-sample one. `TradingConfig.holdout_events`
is documented as excluding trailing events "so the evaluation region is untouched",
and `sweep_stage_a` did not apply it -- it replayed the whole stream, so 60% of the
events in the first run were fitted. That defect was found and corrected; the
in-sample numbers are retained as artifacts but are not findings.

**AAPL, H = 10 s, k = 6, 160,157 held-out events:**

| quantity | value |
|---|---|
| out-of-sample edge decay, 0.1 ms to 100 ms | **25.0%** |
| oracle decay over the same grid | 14.6% |
| oracle capture at 0.1 ms | **-3.41** |
| trade rate | 98.1% of evaluated events |
| out-of-sample directional lift | +0.0082 — **not resolved**; 2 SE 0.041 |

Two claims are withdrawn. The secondary arm's 23.7% decay and AMZN's 13.1% decay
were measured in sample and fall to 3.2% and 6.3% out of sample. The cross-arm
ordering was an artefact of the leakage defect, and only the primary carries a
strong latency decay. Neither arm's economics change in sign: capture is -4.02 on
the secondary and -1.94 on the held-out.

### Confidence orders direction, but not magnitude

> **Corrected.** The subsection this replaces concluded that confidence "does not order
> opportunities", on the basis that the most confident 1% of events was *less* accurate
> than the base rate. That was computed with the defective lift endpoint described in
> section 14 — the reference was the policy's own accuracy and the numerator and
> denominator were over different event sets — and the conclusion does not survive its
> repair. The corrected result is below and is the opposite in its first half.

Thresholding the trained policy's own confidence — act only when
`max(p_BUY, p_SELL) > c`, else HOLD — retraining nothing. Reference is the
constant-direction baseline, 0.5363.

| threshold | trades | accuracy given trade | lift | net per trade (granules) | capture |
|---|---|---|---|---|---|
| 0.000 (the frozen policy) | 157,052 | 0.544 | +0.008 | −26.0 | −4.037 |
| 0.400 | 105,375 | 0.550 | +0.014 | −25.0 | −2.597 |
| 0.500 | 19,285 | 0.565 | +0.028 | −20.5 | −0.391 |
| 0.550 | 6,770 | 0.606 | +0.070 | −21.5 | −0.144 |
| 0.600 | 1,631 | 0.634 | +0.098 | −23.1 | −0.037 |
| 0.650 | 305 | 0.658 | +0.122 | −28.9 | −0.009 |

**Directional accuracy rises monotonically with the policy's own confidence**, from
0.544 at 98% traded to 0.658 at 0.2% traded. That is a real ordering, reproduced
across six thresholds and therefore much harder to attribute to noise than any single
point; the individual high-threshold cells are small (305 trades, one standard error
0.027) and are not separately established.

**Net value per trade does not follow.** It sits between −26 and −21 granules across
the whole range, and capture never reaches zero — it only asymptotes towards it as the
trade count collapses to nothing. The ordering therefore carries no economic
information: the moves the policy is most confident about are not systematically
larger than the ones it is least confident about.

That is a sharper boundary than "no selectivity", and it points somewhere specific.
The frozen target is three-class with a fixed 6-granule band, so it never asks the
policy how *far* the price will move — only whether it moves by more than 6 granules
(3 cents). The crossing cost is half the spread, 15 granules. A policy that classified
the frozen target perfectly would still have no estimate of whether a signalled move is
6 granules or 60, and 40% of signalled moves do not cover the spread.

**This is a hypothesis about the mechanism and is not a finding.** Target granularity
and execution cost were *neither* independently manipulated: `k` was frozen at 6 before
any model was trained, and aggressive touch crossing was frozen as an execution
assumption at the same time. Nothing here varies one while holding the other, so the
gap between a 6-granule label and a 15-granule cost cannot be assigned a causal role.
It is the most economical account of an observation — direction is ordered, payoff is
not — and it is consistent with alternatives this experiment did not test, such as the
information about magnitude simply not being present in a 20-event touch history.

What is demonstrated, and no more: confidence orders directional accuracy monotonically
across six thresholds; net value per trade does not follow that ordering; and no
threshold on this policy reaches positive capture. The compound reading — the policy
knows more about direction than the frozen target's economics let it exploit — is a
description of those three facts together, not a fourth result.

It also leaves the frozen result untouched: the pre-registered policy acts whenever
the argmax is a direction, trades 98.1% of events, and captures −3.41. Thresholding is
a post-hoc diagnostic and is not substituted for it.

## 14. Experiment L1: lookback ablation (pre-registered)

Registered before any run. Nothing in this section may be revised after seeing L1
results.

### Question

Does the length of the historical context the policy observes change its ability to
identify economically valuable opportunities, as measured by out-of-sample
directional lift?

The structural motivation is that 20 events is unlikely to be a stable temporal
context against a 10 s target, on a stream whose median event gap is 0.639 ms and
whose mean is 58.44 ms. That is a reason to run the experiment. It is **not** a
prediction that more context helps, and no such prediction is registered: the
outcome may be monotonic improvement, improvement then saturation, no effect, or
deterioration, and all four are results.

### Frozen baseline

The completed V2B experiment is not altered. AAPL, `H` = 10 s, `k` = 6 granules,
the existing seven features, bounded ReLU, the existing MLP architecture, the
existing supervised training protocol, the existing latency grid, aggressive
touch-crossing, the existing train/out-of-sample split, the existing seeds and
budget. Its result stands as recorded in section 13.

Note that the training is supervised, not PPO; the frozen protocol's accuracy
figures come from a cross-entropy fit, and PPO is not involved in this branch.

### Independent variable

Historical lookback `L` in events: **20, 50, 100, 200**.

`L` is a window length and not an input width. The features are seven summaries,
three of which are window-averaged, so feature dimensionality is 7 for every `L` and
the architecture needs no change. That is verified rather than assumed. Windows clamp
at the start of the stream rather than padding, so the first `L - 1` rows use a
shorter window -- 199 rows at `L` = 200 against roughly 240,000 fitted ones.

### Primary endpoint

**Out-of-sample directional lift.**

    lift = accuracy(policy's direction, on events it trades)
         - directional base rate over the same out-of-sample events

Both terms are computed on the same held-out events and both are restricted to
events whose label is BUY or SELL, since accuracy is undefined where no direction
exists. Registered sign: **positive lift is improvement; zero or negative lift does
not establish useful selectivity.** No other quantity may be promoted to the success
criterion after the fact.

### Secondary endpoints

Reported, not decisive: trade rate; accuracy conditional on trading; net value per
trade; net value per event; oracle capture; latency decay; number of out-of-sample
opportunities.

### Controls

The train and evaluation regions are identical across all four values of `L`. This
is verified by test, because a split that moved with `L` would make any difference
between arms a difference in what was trained on.

**No observation from the out-of-sample region may inform feature construction,
normalisation, target fitting, model selection or checkpoint selection.** The
feature normaliser carries the boundary it was fitted to, and any evaluation naming
a region that starts inside it raises. The boundary is currently carried as a value
rather than enforced by construction, so this is a check that can fail rather than a
proof; it is a test because the convention behind it has already been violated once.

### Model selection

All four values of `L` are reported as independent arms. No lookback is selected on
out-of-sample performance, and no winner is declared. With four arms and one
endpoint there is no multiple-comparison correction to make, and adding a validation
region would create a second split whose own boundary could leak.

### Model size

Frozen. Lookback and capacity are not varied together. If a longer lookback changes
lift, capacity is the next question; if it does not, capacity becomes the more
informative experiment, because a negative result would then distinguish "the
information is absent" from "the architecture cannot use it" only once capacity has
also been varied.

### Reporting

Whatever the outcome, including a null, is reported against section 13's baseline.
A null here is evidence about the representation, and specifically it is the
evidence that would make a capacity ablation worth running.

### Result: no effect

All four arms, identical train and evaluation regions, model size and every other
setting frozen. Lift is out-of-sample directional accuracy minus a constant-direction
baseline, with both terms over the same directional events.

| L | train acc | OOS acc\|trade | baseline | lift | 2 SE | significant | trade rate |
|---|---|---|---|---|---|---|---|
| 20 | 0.428 | 0.5444 | 0.5363 | +0.0082 | 0.041 | no | 98.3% |
| 50 | 0.437 | 0.5485 | 0.5363 | +0.0122 | 0.041 | no | 97.5% |
| 100 | 0.445 | 0.5441 | 0.5363 | +0.0079 | 0.042 | no | 96.3% |
| 200 | 0.460 | 0.5166 | 0.5363 | −0.0196 | 0.041 | no | 92.2% |

The pre-registered sign rule says positive lift is improvement. Three of four arms
are nominally positive and **none is distinguishable from zero**: the largest is +0.012
against a two-standard-error band of 0.041. Under the registered rule this is not an
improvement.

The error bar is computed on 824 non-overlapping windows, not on the 118,257
directional events. Consecutive look-ahead windows share nearly all of their events;
using the raw count would shrink the standard error by a factor of about 144 and
would have made all four arms appear significant.

Two things are worth stating because they were not obvious in advance. Training
accuracy rises monotonically with L, 0.428 to 0.460, while out-of-sample accuracy
falls from L = 50 onward — the extra context is being used to fit the training region
rather than to generalise. And the trade rate falls with L, 98.3% to 92.2%, so a
longer window makes the policy marginally less indiscriminate without making it
selective.

Per the registration, no lookback was selected and no winner is declared.

### Two defects in the endpoint, found by running it

The endpoint was implemented wrongly on the first attempt and gave a different
answer, so both are recorded.

Its reference was the policy's own accuracy rather than an independent baseline. Both
terms then shared the numerator `count(correct)`, so lift reduced to
`correct * (1/n_traded - 1/n_directional)` — negative whenever the policy trades more
events than there are directional ones, *including for a perfect forecaster*. The
first run appeared to show lift improving monotonically with lookback; that was the
reference drifting toward zero, not the policy improving.

Its numerator was restricted to directional events while the denominator counted all
traded events, so the two terms were not over the same events. A constant-direction
policy did not score zero lift.

Both are now pinned by tests: a constant-direction policy scores exactly zero, and a
perfect one scores positive. Under the corrected endpoint the base rate is 0.5363 —
always betting the more common direction — and the policy reaches 0.5444 at best.

### The arms are single-seed, and that is the binding limitation

All four L1 arms were trained at `seed = 0`. The four lifts span 0.004 to 0.028
against a two-standard-error band of 0.041, so **variation in the training procedure
alone could account for the entire spread between arms.** L1 therefore compares four
models whose initialisation variance was never estimated, and reports the differences
as properties of the lookback. Its conclusion — that more context did not help — is
not contradicted by anything, but it is weaker than four arms at one seed can support.

`PROTOCOL.md` section 8 requires ten seeds for control arms. That requirement was
never applied to the finance branch, where `train_policy` had no seed plumbing at all.

## 15. Experiment L1R: lookback replication (pre-registered)

Registered before any run, for the reason in section 14: L1's endpoint had two defects
and the feature-lookback plumbing had a third, so the infrastructure is settled before
this one rather than after it.

### Question

Does the lookback effect observed in L1 survive variation in training initialisation?

### Design

`L` in {20, 50, 100, 200} crossed with seed in {0, 1, 2}. Twelve arms. Everything
else frozen as in section 14: AAPL, `H` = 10 s, `k` = 6, the seven features, bounded
ReLU, the small architecture, the supervised protocol, the latency grid, aggressive
touch-crossing, the train and evaluation regions, and the training budget.

The no-information baseline is **held fixed across every arm**. It is a random
position stream, and re-drawing it per seed would add baseline noise into every
between-arm comparison, so a difference between two arms would no longer be
attributable to the policy. Only the policy's seed varies.

### Endpoints

Per `(L, seed)`: out-of-sample directional lift with its standard error; trade rate;
out-of-sample directional accuracy; net value per event at the fastest and slowest
delays; oracle capture; latency decay.

### Two variances, reported separately

**Sampling uncertainty within a run.** The standard error already reported with the
lift: 824 non-overlapping windows, giving a two-standard-error band near 0.041. This
is how precisely one trained policy's measured lift is known on this data.

**Between-seed variance.** The spread of the lift across the three initialisations at
a fixed `L`. This is how much of a lookback difference could be produced by
initialisation alone.

These answer different questions and are **not** pooled. Pooling three seeds into one
observation would produce a standard error that mixes sampling noise with training
noise and attributes the difference to neither. The between-seed spread is reported
as a range and a standard deviation across three points, which is a weak estimator and
is labelled as such.

### Registered reading

- **A.** A lookback trend that is large relative to the between-seed spread survives.
  Temporal context matters and capacity becomes the sensible next question.
- **B.** Everything sits near +0.01 lift with between-seed spread of the same order.
  The target is weakly learnable or unresolved, and increasing capacity is not yet
  justified.
- **C.** One `(L, seed)` cell produces a lift far outside the others. That cell is
  investigated before anything else runs, so that a favourable initialisation is not
  promoted into a capacity result.

No outcome is excluded in advance.

### Result: the seed concern was unfounded, and the lookback effect survives it

Twelve arms. The pre-registered worry was that initialisation variance could account
for the entire L1 spread. It does not.

| L | lift by seed | mean | between-seed sd | range | 2 SE within-run | sd ÷ 2 SE |
|---|---|---|---|---|---|---|
| 20 | +0.0082 +0.0076 +0.0089 | +0.0082 | 0.0007 | 0.0013 | 0.0414 | **0.02** |
| 50 | +0.0122 +0.0077 +0.0115 | +0.0105 | 0.0024 | 0.0046 | 0.0413 | **0.06** |
| 100 | +0.0079 +0.0063 −0.0055 | +0.0029 | 0.0073 | 0.0133 | 0.0423 | **0.17** |
| 200 | −0.0196 −0.0130 −0.0035 | −0.0120 | 0.0081 | 0.0162 | 0.0407 | **0.20** |

Between-seed spread is **5x to 50x smaller** than the sampling band at every lookback.
Registering L1R as a check on initialisation was the right precaution and the answer
is that initialisation is not a material source of variance here. The sampling
uncertainty from 824 non-overlapping windows dominates everything, which is the same
conclusion the power calculation reached from the other direction.

So the outcome is closest to **A**, with the direction reversed from what A anticipated.

Sign agreement across the three initialisations is consistent at both ends:

- `L` = 20 and `L` = 50 are **positive at all three seeds**, mean +0.008 and +0.011.
- `L` = 200 is **negative at all three seeds**, mean −0.012.
- `L` = 100 is mixed, two positive and one negative.

The robust claim is that directional skill **degrades** as lookback extends past
about 50 events, and the ordering is reproduced by every initialisation. The magnitude
is not established: the difference between the best and worst lookback is 0.020
against a per-arm band of 0.041, and no paired interval on the between-arm difference
has been computed. Three points also make the between-seed standard deviations weak
estimators, as registered.

This falsifies the hypothesis that motivated L1. More temporal context was supposed to
help, and it hurts. Training accuracy rises monotonically across the four lookbacks,
0.428 to 0.452, while out-of-sample accuracy falls — the additional window is being
spent fitting the training region. On a stream whose median event gap is 0.639 ms and
mean 58.44 ms, a 200-event window is not a longer look at the same timescale; it spans
a heterogeneous and variable stretch of wall time, so there is no stable period for it
to average over.

The secondary endpoints move together with it:

| L | mean latency decay | mean oracle capture | mean train acc |
|---|---|---|---|
| 20 | **24.6%** | −3.392 | 0.428 |
| 50 | 16.5% | −3.330 | 0.437 |
| 100 | 17.5% | −3.394 | 0.443 |
| 200 | 3.1% | −3.293 | 0.452 |

The latency effect is **strongest at the shortest lookback and essentially disappears
at L = 200**. The frozen V2B baseline is L = 20, so the arm carrying the headline
25.0% decay is also the best of the four on that measure — the freeze was not
unlucky. Capture is flat at −3.3 across all twelve arms: the economics are uniformly
unviable regardless of lookback or initialisation, which is a robustness check on the
V2B conclusion rather than a new finding.

### What this does not authorise

Magnitude is established for the extreme and not for the middle, once the comparison
is **paired**. Every arm is evaluated on the same 824 windows, so the two arms are
wrong together in the same direction, and differencing their independent standard
errors adds two variances where the data supports one:

| comparison | mean paired difference | paired 2 SE | 95% CI | excludes 0 |
|---|---|---|---|---|
| L50 − L20 | −0.0040 | 0.012 | [−0.0155, +0.0076] | no |
| L100 − L20 | −0.0170 | 0.020 | [−0.0366, +0.0025] | no |
| L200 − L20 | **−0.0443** | 0.021 | **[−0.0651, −0.0235]** | **yes** |

Pairing is worth roughly a factor of five: the paired two-standard-error band is
0.021 where the marginal one was 0.041, and it is 0.011 for L50 − L20.

**L = 200 is robustly worse than L = 20.** All three seeds exclude zero
individually — −0.0529, −0.0419, −0.0381 — so this is three independent replications
of the same sign, not one result with a wide interval. **L = 50 is
indistinguishable from L = 20**, and L = 100 sits between them with a point estimate
of −0.017 that does not reach significance.

This corrects the statement above. "The magnitude is not established" was computed
from the marginal 0.041 band and was too pessimistic: pairing was the missing step,
and under it the L = 200 degradation is a resolved effect of about −0.044 lift while
the L = 20 to L = 50 difference is resolved as nothing.

The aggregation across seeds is stated rather than assumed. The three seeds share the
same windows, so the dominant sampling error is **common** to them and does not shrink
with the seed count; dividing by `sqrt(3)` would treat one set of windows as three
independent samples. The aggregate interval is
`se = sqrt(median(within-seed paired se)^2 + between_seed_sd^2 / n_seeds)`, in which
only the second term is reduced by the number of seeds. With three seeds that term is
negligible here — initialisation variance is small — so the aggregate interval is
essentially the within-window one.

Capacity is **not** the next experiment. The registered reasoning for it was that
temporal context is the binding constraint; L1R shows context is not, and that the
best available lift is +0.011 at a resolution of 0.041. Raising capacity on a
representation whose best result is a fraction of its own noise band would produce
another ambiguous null. The constraint is measurement power, and the power calculation
puts the remedy at roughly 3,463 non-overlapping windows against the 824 available
from one session — more independent market time, not a larger network.

## 16. Conclusion: the historical LOB branch

Three layers, at three different levels of support. They are kept separate because
collapsing them is how a latency result turns into a profitability claim that the
evidence does not carry.

### 1. Latency sensitivity — established

The trained policy's out-of-sample edge over a delay-matched random baseline decays
monotonically with execution delay: **25.0%** across 0.1 ms to 100 ms in the frozen
primary arm, against a 14.6% oracle opportunity decay. The decay is present in every
arm and every seed, is a within-window paired comparison, and the mechanism survives
the transition from oracle to learned policy — which is the substantive V2B result.

### 2. Economic selectivity — not established

Absolute capture is **−3.41** and negative at every confidence threshold and every
arm. The policy trades 98.1% of events against a 15-granule half-spread.

Its directional lift over the constant-direction baseline is **+0.0082, and that
number is not resolved**: the two-standard-error band is 0.041, five times the
estimate. It is neither established as positive nor established as zero.

The one thing that *is* established here is a mechanism boundary rather than a
result: the policy's confidence orders directional accuracy monotonically, from 0.544
at 98% traded to 0.658 at 0.2% traded, while net value per trade stays flat at −26 to
−21 granules throughout and capture never reaches zero. Directional skill is present
and measurable; discrimination between large and small signalled moves is not, and the
frozen three-class target does not ask the policy for it.

### 3. Temporal context — established effect

Extending input lookback from 20 to 200 events produces a **resolved −0.0443 lift
difference**, 95% CI [−0.0651, −0.0235], with all three seeds excluding zero
individually. L = 50 is indistinguishable from L = 20 (−0.0040, CI [−0.0155, +0.0076]);
L = 100 is unresolved (−0.0170, CI [−0.0366, +0.0025]). More recent event history does
not help and, past roughly 100 events, reliably hurts.

### The relationship between layers 1 and 3

Latency sensitivity tracks the quality of the underlying signal: mean decay is **24.6%
at L = 20** and **3.1% at L = 200**, and L = 200 is the arm whose directional skill is
demonstrably worse. The natural reading is that a latency effect is only measurable when
the policy contains enough useful selectivity to expose the opportunity's decay at
all.

That is an **association across arms, not an identified mechanism.** Selectivity was
not independently manipulated while latency was held fixed, so this experiment does not
establish that selectivity causes latency sensitivity. It is consistent with that
reading and does not exclude alternatives — a policy that predicts worse may simply be
less sensitive to delay for reasons unrelated to selectivity.

### What the branch does not support

No claim that the policy is profitable. No claim that its directional edge is
statistically established. No capacity experiment: the registered motivation was that
network capacity might be the binding constraint, and layer 3 removes it — the best
available lift is a fraction of its own noise band at every lookback tried.

### Open, and deliberately not pursued

Whether the ~+0.008 lift is real would take roughly 3,463 non-overlapping windows
against the 824 one session provides — about 4 to 5 further AAPL days. More data would
answer only that narrow question. It cannot make the economics work, because capture
is negative for reasons of selectivity that a tighter confidence interval does not
touch.

## 17. Interpretation

The scientific conclusion is not predetermined. What counts as a finding:

- performance saturating below some latency;
- a task-dependent threshold;
- a bounded nonlinearity changing where that threshold falls, or how reliably it
  is reached at all.

A result is reported with its seed-level distribution, not a mean alone. The
basic unit of analysis is the independently trained policy; episodes within a
policy are not independent training replicates.