# Results

Experiment: Hartley bounded ReLU (clip 2.0) against the same protocol under a
standard ReLU. See `PROTOCOL.md` for the design and for what would have had to
be true for the comparison to be attributable.

## 0. The historical LOB branch, in three results

This section is the front page for the V2B branch (real LOBSTER replay, frozen target,
learned policy). **Everything below it is subordinate to these three**, and they are
kept apart because they do not have the same standing.

| # | result | standing | figure |
|---|---|---|---|
| 1 | **Latency sensitivity** | established | 25.0% out-of-sample edge decay across 0.1–100 ms, against 14.6% oracle decay |
| 2 | **Temporal context** | established | L200 − L20 = **−0.0443** lift, 95% CI [−0.0651, −0.0235], sign replicated on all three seeds; L50 indistinguishable from L20 |
| 3 | **Economic selectivity** | **not established** | capture **−3.41**; directional lift **+0.0082 unresolved** against a 0.041 two-standard-error band |

Two of the three are established; the third is explicitly unresolved, and neither
"profitable" nor "no edge" is claimed for it. The relationship between 1 and 2 is that
latency sensitivity tracks signal quality across arms — 24.6% decay at L = 20 falling
to 3.1% at L = 200, where directional skill is demonstrably worse. That is an
**association across arms, not an identified mechanism**: selectivity was never
manipulated independently of latency.

Full derivations in `PROTOCOL.md` sections 12–16.

## 1. Three claims, three confidence levels

This experiment licenses three separate conclusions. They are listed separately
because they do not have the same standing, and collapsing them is the specific
error this document exists to prevent: **"no difference was detected" is a claim
about an experiment, not about the thing measured**, and it turns into "no
difference exists" by repetition rather than by argument.

> **Claim 1 — activation effect: inconclusive at the experiment's current
> statistical resolution.**
> The observed paired differences were substantially smaller than the design's
> minimum detectable effects. These results should be reported as *inconclusive*
> — not as evidence of equivalence, and not as evidence of the absence of an
> activation effect. The experiment was not powered to see an effect of the size
> actually observed, so nothing follows about the underlying quantity in either
> direction.

> **Claim 2 — latency boundary: replicated under bounded ReLU.**
> The degradation boundary remains between 80 and 100 ms, in the same place, with
> the same magnitude. Support for this claim is **not uniform across the two
> tasks** and section 4 says which part is established and which part is
> descriptive only.

> **Claim 3 — shared-task failure: replicated under bounded ReLU, including
> within-version transfer initialisation.**
> Zero of 3 seeds reach the convergence threshold in either shared arm, at any
> latency, in either version. Reproducing a solved representation does not fix
> it.

Claims 2 and 3 are positive findings, not absence of findings. A result that
survives a change of nonlinearity is stronger evidence than one that did not.

## 2. Control latency response: Claim 1

4M integration steps, 5 latencies, 10 seeds, 200 evaluation episodes per policy.
Paired by seed — the same seed draws the same initial weights and minibatch order
in both versions. Verdicts computed by `latency_ml.analysis.power.compare_paired`;
MDE is the minimum detectable effect at 80% power, α = 0.05.

### CartPole

| latency | v1 ReLU | v2 bounded | Δ | MDE | Δ/MDE | t | converged | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 20 ms | 500.0 ± 0.0 | 500.0 ± 0.0 | +0.0 | 0.0 | — | +0.00 | 10/10 → 10/10 | inconclusive |
| 40 ms | 494.4 ± 16.8 | 500.0 ± 0.0 | +5.6 | 17.7 | 0.32 | +1.00 | 9/10 → 10/10 | inconclusive |
| 60 ms | 499.0 ± 2.9 | 500.0 ± 0.0 | +1.0 | 3.1 | 0.32 | +1.00 | 10/10 → 10/10 | inconclusive |
| 80 ms | 498.2 ± 2.0 | 499.8 ± 0.5 | +1.7 | 2.1 | 0.81 | +2.53 | 10/10 → 10/10 | inconclusive |
| 100 ms | 411.2 ± 9.8 | 416.7 ± 7.2 | +5.5 | 7.7 | 0.72 | +2.26 | 0/10 → 0/10 | inconclusive |

### LunarLander

| latency | v1 ReLU | v2 bounded | Δ | MDE | Δ/MDE | t | converged | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 20 ms | 269.2 ± 7.5 | 270.0 ± 10.8 | +0.8 | 15.2 | 0.06 | +0.17 | 10/10 → 10/10 | inconclusive |
| 40 ms | 267.4 ± 3.4 | 264.1 ± 11.6 | −3.3 | 13.2 | 0.25 | −0.78 | 10/10 → 10/10 | inconclusive |
| 60 ms | 260.6 ± 21.6 | 250.4 ± 20.5 | −10.2 | 33.0 | 0.31 | −0.97 | 9/10 → 9/10 | inconclusive |
| 80 ms | 222.3 ± 40.4 | 237.4 ± 35.5 | +15.1 | 58.5 | 0.26 | +0.81 | 6/10 → 7/10 | inconclusive |
| 100 ms | 183.8 ± 49.3 | 196.0 ± 43.9 | +12.1 | 75.7 | 0.16 | +0.50 | 3/10 → 3/10 | inconclusive |

"Converged" is the fraction of seeds reaching the LunarLander threshold of 230
that v1's own results justified, and 475 for CartPole.

## 3. Why Claim 1 is inconclusive rather than null

Every one of the nine comparable cells is inconclusive, with observed differences
between 6% and 81% of the detectable effect. Seeds required to resolve a
20-unit difference, at the dispersion actually observed:

| latency | CartPole | LunarLander |
|---:|---:|---:|
| 20 ms | — | 7 |
| 40 ms | 9 | 6 |
| 60 ms | 3 | 24 |
| 80 ms | 3 | 70 |
| 100 ms | 4 | 116 |

Settling the activation question needs a stated target effect and then the seed
count that implies — at LunarLander's 100 ms dispersion, a 20-unit effect needs
about 116 seeds, not 10. Adding seeds without fixing that number first is how a
design stays underpowered indefinitely.

One subtlety, because it is the kind that gets reported as a result by accident.
CartPole at 80 ms clears the critical value: |t| = 2.53 against 2.26, on an
effect of 1.7 where the detectable effect is 2.1. The verdict is still
`inconclusive`, because significance describes one draw while the detectable
effect describes what this design can be relied on to reproduce, and a first-look
effect that has not been replicated is exactly the result that fails to replicate.

### Two things not claimed

Seed dispersion fell modestly in LunarLander's degraded regime (100 ms, sd
49.3 → 43.9; 80 ms, 40.4 → 35.5) and rose in the stable one (40 ms, 3.4 → 11.6).
Two individual seeds moved: CartPole 40 ms went 9/10 → 10/10 converged,
LunarLander 80 ms went 6/10 → 7/10. Neither is a result. The 20 ms CartPole cell
is reported as inconclusive rather than as agreement: every seed returned exactly
500.0 in both versions, so the paired differences have no spread and the test is
degenerate.

## 4. Latency boundary: Claim 2, and where it is weaker

The boundary claim needs its own test rather than an inspection of the table, so
the 80 → 100 ms step was compared within each version, paired by seed. Here the
effect is large, so the test has real power — unlike Claim 1.

| version | task | 80 ms | 100 ms | Δ | MDE | t | verdict |
|:--|:--|---:|---:|---:|---:|---:|:--|
| v1 ReLU | CartPole | 498.2 ± 2.0 | 411.2 ± 9.8 | −87.0 | 11.1 | −24.61 | **detected** |
| v2 bounded | CartPole | 499.8 ± 0.5 | 416.7 ± 7.2 | −83.1 | 7.7 | −34.10 | **detected** |
| v1 ReLU | LunarLander | 222.3 ± 40.4 | 183.8 ± 49.3 | −38.4 | 53.7 | −2.25 | inconclusive |
| v2 bounded | LunarLander | 237.4 ± 35.5 | 196.0 ± 43.9 | −41.4 | 59.9 | −2.18 | inconclusive |

The same test one step earlier (60 → 80 ms) returns `inconclusive` in all four
cells, so there is no resolvable degradation below 80 ms on either task.

**What this supports.** On CartPole the boundary is established: a large,
precisely measured collapse at 100 ms in both versions, and nothing at 80 ms. The
two versions' step sizes agree to 3.9 units on a step of about 85 — that
agreement is the replication, and it is a strong result.

**What it does not support.** On LunarLander the 80 → 100 ms drop is *below the
detectable effect in both versions*. The decline across the grid is real in the
means but is not statistically resolved at n = 10, because the seed dispersion
there is large. So on LunarLander, "both versions agree on where the boundary
is" is supported, while "there is a boundary" is a descriptive reading of the
means. An earlier draft of this document stated the boundary as established on
both tasks; that was stronger than the LunarLander evidence supports.

**Independent support for the reliability reading.** The 100 ms CartPole cell
reaches 0/10 converged seeds in both versions, and LunarLander's 100 ms converged
count is 3/10 in both. The high-latency deficit shows up as seeds that fail to
reach the stable regime, rather than as converged seeds performing worse.

## 5. Shared representation: Claim 3

3 seeds × 5 latencies, same 4M per-task budget. LunarLander mean return:

| latency | v1 4C | v2 4C | v1 4D | v2 4D |
|---:|---:|---:|---:|---:|
| 20 ms | 106.3 | 136.2 | −521.5 | −30.1 |
| 40 ms | −25.4 | −2.9 | 166.4 | 59.6 |
| 60 ms | 46.2 | −77.9 | 150.2 | 18.3 |
| 80 ms | −50.7 | −7.4 | 52.3 | −6.2 |
| 100 ms | −216.2 | −88.0 | −12.5 | 31.0 |

4C is a shared trunk trained from scratch; 4D initialises it from a solved
dedicated policy. **Zero of 3 seeds reach the 230 threshold in either arm, at any
latency, in either version.**

### The 4D columns are not a paired comparison

**The v1 and v2 4D columns must not be differenced, and this table is not a
before-and-after.** The two arms did not run the same intervention, for a reason
that is a property of the method rather than of the results: a transfer reproduces
a source's post-activation hidden layer, which is only meaningful when the
nonlinearities match. Verified directly, a standard-ReLU source and a bounded
trunk produce different hidden layers, so v1's ReLU checkpoint is not a valid
source for a bounded trunk — it would not be a "same representation" transfer at
all.

Consequently v2's 4D uses this version's own 20 ms LunarLander seed-0 policy,
which is solved (10/10 seeds past the threshold). That makes the arm a
**within-version** transfer experiment. The substantive question is preserved and
answered — *does transferring a solved LunarLander representation into the shared
architecture fix the shared-task problem?* — and the answer is no, in both
versions. Only the numerical comparison across versions is void.

### Secondary observations

4C preserves CartPole while failing LunarLander, in both versions. 4D moves the
failure mode without restoring performance. Cells run far slower under 4D than
4C (1540 s vs 230 s at 20 ms) because a transferred *solved* representation stops
terminating early, so episodes run their full length instead of ending in a crash
— a runtime artifact, not a result.

## 6. Protocol verification

Both assertions in `PROTOCOL.md` §9 hold, and are enforced by
`tests/test_parity_v1.py` rather than asserted here:

- the consolidated trainer reproduces a previous-version checkpoint
  **tensor-for-tensor** under an explicitly injected unbounded ReLU, so the
  activation is the only executable difference between the two experiments;
- the two versions' recorded training protocols differ in exactly one field.

A third assertion was added after the fact and is recorded here for honesty: the
minimum detectable effect must be stated alongside any comparison before its
verdict is written. It was not in the original protocol, which is itself the
finding — nothing in the protocol required computing it, and the first
conclusion written here did not.

Realised runs match the frozen protocol: minibatch size 128 invariant across
every latency, 7,812 → 1,562 PPO updates and 124,992 → 24,992 gradient steps
across the grid, every manifest recording `HartleyBoundedReLU`.

## 7. What remains open

- **The activation question is unresolved.** Not answered, not closed. §3 gives
  the seed count a target effect would imply.
- **LunarLander's boundary is descriptively, not statistically, established** at
  n = 10 (§4). Separating it from a gradual decline needs either a
  lower-variance outcome than episodic return or many more seeds.
- Whether the shared LunarLander deficit is insufficient trunk capacity or
  cross-task interference. Untouched by this experiment.
- `signal_alive_ratio` is **inert on real data**: `lobster.py` sets `signal` to
  zero throughout rather than fabricate one, so that feature is identically zero
  for every real session. The feature interface is deliberately left at seven
  dimensions — changing it would change the protocol fingerprint and invalidate
  the parity comparison — but any real-data model is carrying one constant column.
- `mechanism.py` is **synthetic-only** and cannot run on real sessions at all:
  its trigger selection is `stream.signal != 0`, which is empty on real data by
  the same construction. Historical evidence is limited to the trained-policy
  replay.
- **Historical LOB validation.** The loader now reads real LOBSTER sessions and
  the fixed-policy replay runs end to end, but the trading threshold it derives
  degenerates on real data: the 30th percentile of absolute forward mid moves is
  exactly zero, because a real mid is a step function. See the open work below.
- Hardware cells. No accelerator model is frozen, so no hardware latency figure
  is reported.

## 8. Reproducing

```bash
make check-all
latency-ml sweep --config configs/cartpole.yaml    --sweep-id latency-cartpole
latency-ml sweep --config configs/lunarlander.yaml --sweep-id latency-lunarlander
latency-ml evaluate --config configs/cartpole.yaml    --sweep-id latency-cartpole
latency-ml evaluate --config configs/lunarlander.yaml --sweep-id latency-lunarlander
latency-ml shared-trunk --config configs/base.yaml --out results/shared-trunk --seeds 0 1 2
latency-ml shared-trunk --config configs/base.yaml --out results/shared-trunk-transfer \
    --seeds 0 1 2 --init-source results/latency-lunarlander/lunarlander/20000000ns/seed0/policy.pt
```

130 training cells (100 control + 15 shared from scratch + 15 shared transfer), about
2 h 20 m on 15 cores at 6 workers. Evaluation and Phase
4 reuse the same checkpoints.

## Appendix: the target diagnostic that preceded any training

`latency-ml diagnose-target` characterises the trade-direction target on real
LOBSTER sessions. It trains nothing and selects nothing. Two findings changed the
design and are recorded here so they are not buried in a later commit.

### Event-count horizons were retired, and the reason is the burst

The first parameterisation counted events. It is not a clock. AAPL's **median**
event gap is 0.639 ms and its **mean** is 58.44 ms — a 91x ratio, because the book
arrives in bursts separated by quiet. The same event count therefore names
windows of wildly different duration, and the same millisecond horizon buys very
different numbers of events on different symbols (AAPL 45, AMZN 32, INTC 165 at
100 ms). Any reasoning from median spacing is wrong by two orders of magnitude.

`H` is now a duration, `t_H = min{t_i : t_i - t_0 >= H}`, with execution at
`t_0 + tau` on the same scale.

### INTC and MSFT are low-value, not signal-starved

An earlier reading of the event-count table concluded these two symbols "are not
viable because they don't generate enough directional opportunities". **That was
wrong**, and the correction matters because the two properties are separate:

- *Signal viability* — do the signals exist and cover their cost? INTC and MSFT
  pass this comfortably. Their half-spread is **one granule** against AAPL's
  fifteen, so post-cost survival reaches 95–100% across the grid, and at
  `H` = 1 s, `k` = 2 they are already positive where AAPL is not.
- *Economic magnitude* — is the residual worth anything? This is where they fail.
  Their net outcome is 0.2–1.3 granules at `H` <= 25 s, against AAPL's 24.6.

So the correct statement is not that they lack signals. It is that a cheap
spread buys survival without buying value, because the moves available on them are
smaller than the moves available on AAPL by more than the cost advantage. They
are retained in the diagnostics and excluded from the primary economic claim on
magnitude alone.

### The frozen design

AAPL at `H` = 10 s, `k` = 6 granules, aggressive touch crossing, latency swept
through 100 ms, with AAPL 5 s / `k` = 6 as a high-sensitivity arm and AMZN held
out at its own viable horizon. The full specification, including the
capture-fraction identifiability check performed *before* training, is
[PROTOCOL.md section 12](PROTOCOL.md#12-frozen-v2b-design). Evidence is under
`results/target-diagnostic/`.


## Experiment L1: lookback ablation

Pre-registered in [PROTOCOL.md section 14](PROTOCOL.md#14-experiment-l1-lookback-ablation-pre-registered).
AAPL, `H` = 10 s, `k` = 6, out-of-sample directional lift against a constant-direction
baseline. `L` is the only variable.

| L | train acc | OOS acc given trade | baseline | lift | 2 SE | significant |
|---|---|---|---|---|---|---|
| 20 | 0.428 | 0.5444 | 0.5363 | +0.0082 | 0.041 | no |
| 50 | 0.437 | 0.5485 | 0.5363 | +0.0122 | 0.041 | no |
| 100 | 0.445 | 0.5441 | 0.5363 | +0.0079 | 0.042 | no |
| 200 | 0.460 | 0.5166 | 0.5363 | −0.0196 | 0.041 | no |

**No effect.** Three arms are nominally positive; none clears two standard errors, so
under the pre-registered rule none is an improvement. The error bar is computed on 824
non-overlapping windows rather than the 118,257 directional events, because consecutive
look-ahead windows share nearly all their events.

Training accuracy rises monotonically with L (0.428 → 0.460) while out-of-sample
accuracy falls from L = 50 onward, so the extra context is being spent fitting the
training region rather than generalising.

The registered reading of this outcome is the one it was written for: the structural
hypothesis that 20 events is insufficient temporal context is **not supported**, and
capacity is now the more informative next experiment — a larger network on L = 20
distinguishes "the information is absent" from "the architecture cannot use it".

Two defects in the endpoint itself were found by running it, both recorded in the
protocol. The reference was originally the policy's own accuracy, which made the two
terms share a numerator and forced lift negative even for a perfect forecaster; the
first run's apparent improvement with lookback was the reference drifting rather than
the policy improving. The numerator was also restricted to directional events while
the denominator was not, so the terms were not over the same events. Both are now
pinned by tests.

## Experiment L1R: lookback replication

Pre-registered in [PROTOCOL.md section 15](PROTOCOL.md#15-experiment-l1r-lookback-replication-pre-registered).
`L` ∈ {20, 50, 100, 200} crossed with seed ∈ {0, 1, 2}. Baseline held fixed at one seed
across all twelve arms so a difference is attributable to the policy.

**The seed concern was unfounded.** Between-seed spread is 5× to 50× smaller than the
sampling band at every lookback (sd ÷ 2 SE ranges 0.02 to 0.20). Initialisation is not
a material source of variance; the 824 non-overlapping windows dominate.

**The lookback effect survives, and it is negative.**

| L | lift by seed | mean | sign |
|---|---|---|---|
| 20 | +0.0082 +0.0076 +0.0089 | +0.0082 | 3/3 positive |
| 50 | +0.0122 +0.0077 +0.0115 | +0.0105 | 3/3 positive |
| 100 | +0.0079 +0.0063 −0.0055 | +0.0029 | 2/3 positive |
| 200 | −0.0196 −0.0130 −0.0035 | −0.0120 | 0/3 positive |

Directional skill **degrades** past ~50 events of lookback, reproduced by every
initialisation. Training accuracy rises monotonically across the four (0.428 → 0.452)
while out-of-sample accuracy falls: the extra window is spent fitting the training
region. On a stream with a 0.639 ms median and 58.44 ms mean event gap, a 200-event
window is not a longer view of the same timescale.

### Paired differences on shared windows

Every arm is evaluated on the same 824 non-overlapping windows, so the arms are wrong
together in the same direction. Differencing their independent standard errors adds two
variances where the data supports one; pairing is worth about a factor of five.

| comparison | mean paired Δ | paired 2 SE | 95% CI | excludes 0 |
|---|---|---|---|---|
| L50 − L20 | −0.0040 | 0.012 | [−0.0155, +0.0076] | no |
| L100 − L20 | −0.0170 | 0.020 | [−0.0366, +0.0025] | no |
| **L200 − L20** | **−0.0443** | 0.021 | **[−0.0651, −0.0235]** | **yes** |

**L = 200 is robustly worse than L = 20** — all three seeds exclude zero individually
(−0.0529, −0.0419, −0.0381), so this is three independent replications of the sign.
**L = 50 is indistinguishable from L = 20.** L = 100 sits between them, not resolved.

This corrects an earlier statement in the protocol that the magnitude "is not
established". That was computed from the marginal 0.041 band and was too pessimistic:
under pairing, the L = 200 degradation is a resolved effect of about −0.044 lift, while
the L = 20 to L = 50 difference is resolved as nothing.

The cross-seed aggregation keeps the shared-window sampling term undiminished — the
three seeds share windows, so that error does not shrink with the seed count, and only
the seed term is divided by it.

This **falsifies the hypothesis that motivated L1** and does not authorise a capacity
ablation. The binding constraint is measurement power, not representation.

## V2B primary result

Out-of-sample, AAPL, `H` = 10 s, `k` = 6, 160,157 held-out events. Recorded in
[PROTOCOL.md section 13](PROTOCOL.md#13-v2b-primary-result-and-two-withdrawn-claims).

| quantity | value |
|---|---|
| edge decay, 0.1 ms → 100 ms | **25.0%** |
| oracle decay, same grid | 14.6% |
| oracle capture at 0.1 ms | **−3.41** |
| trade rate | 98.1% of events |
| directional lift | negative at every confidence threshold |

**The latency effect is real; the economics are not.** A compact trained policy's
edge over a delay-matched random baseline decays monotonically with execution delay,
by more than the oracle opportunity's own decay. But the policy captures none of that
opportunity: it acts on 98% of events against a 15-granule half-spread, so it pays
the crossing cost nearly every event.

**Confidence orders direction but not magnitude.** Thresholding the frozen weights at
`max(p_BUY, p_SELL) > c` improves *directional accuracy* monotonically — 0.544 trading
98% of events, 0.658 trading 0.2% — but net value per trade stays flat at −26 to −21
granules throughout and oracle capture never reaches zero, only asymptoting toward it as
trades collapse to nothing. The policy's most confident predictions are not
systematically over larger moves.

> **Corrected.** This section previously read "confidence does not order opportunities",
> reporting that the most confident 1% was *less* accurate than the base rate. That came
> from the defective lift endpoint (reference set to the policy's own accuracy, with
> numerator and denominator over different event sets). Under the repaired endpoint the
> accuracy ordering is real and monotone. The economic conclusion is unchanged and the
> boundary is sharper: directional skill is present and measurable, magnitude
> discrimination is not, and the frozen three-class target never asks for magnitude.

The frozen result itself is untouched: the pre-registered policy acts whenever the argmax
is a direction, trades 98.1% of events, and captures −3.41. Thresholding is a post-hoc
diagnostic and is not substituted for it.

**The 6-granule band against a 15-granule crossing cost is a hypothesis, not a
finding.** It is the most economical account of why direction is ordered and payoff is
not, but neither side was independently manipulated: `k` and the execution model were
both frozen before any model was trained. Nothing here varies one while holding the
other, so the gap cannot be assigned a causal role. Alternatives this experiment did not
test remain open — including that magnitude information may simply be absent from a
20-event touch history. What is established is the three observations above and nothing
about why.

Two earlier claims are withdrawn. The secondary arm's 23.7% decay and AMZN's 13.1%
were measured in sample; out of sample they are 3.2% and 6.3%. The cross-arm ordering
was an artefact of `sweep_stage_a` replaying fitted events. Their economics do not
change sign.

## Conclusion

Three layers, at three different levels of support.

**Established — latency sensitivity.** The trained policy's out-of-sample edge over a
delay-matched random baseline decays monotonically with execution delay, 25.0% across
0.1–100 ms, against a 14.6% oracle decay. Present in every arm and seed, and it
survives the transition from oracle to learned policy.

**Not established — economic selectivity.** Capture is −3.41 and negative everywhere.
Directional lift is **+0.0082 and unresolved**: the 2 SE band is 0.041, five times the
estimate. Neither positive nor zero is established.

**Established — temporal context.** L=200 is reliably worse than L=20, paired Δlift
−0.0443, CI [−0.0651, −0.0235], all three seeds excluding zero. L=50 is
indistinguishable from L=20; L=100 unresolved.

Latency sensitivity tracks signal quality across arms — 24.6% decay at L=20 against
3.1% at L=200, where directional skill is demonstrably worse. That is an **association,
not an identified mechanism**: selectivity was not independently manipulated, so it does
not establish that selectivity causes latency sensitivity.

Not supported: any profitability claim, any claim that the directional edge is
established, and any capacity experiment — layer 3 removes the registered motivation
for it.

Left open deliberately: whether the ~+0.008 lift is real needs ~3,463 non-overlapping
windows against the 824 available, about 4–5 further AAPL days. That would answer only
that question, and it cannot make the economics work.
