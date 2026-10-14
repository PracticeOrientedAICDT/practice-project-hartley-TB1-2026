# Environment constraints

Verified 2026-09-30 against live package metadata. These are load-bearing: each
one is a failure mode that is easy to hit and awkward to diagnose mid-run.

Do not re-derive these from memory. If a pin here becomes wrong, verify it
against upstream and update this file **and** `tests/test_environment.py`,
which asserts the pins that matter.

## Python: 3.12 or 3.13. Not 3.14.

`Box2D==2.3.10` publishes cp312 and cp313 macOS arm64 wheels. It publishes **no
cp314 wheel and no sdist at all** — the last sdist is 2.3.2 — so on 3.14 the
package cannot be built or downloaded. `gymnasium`'s box2d extra pins
`box2d==2.3.10` exactly, so there is no version to fall back to.

Upstream issue: [pybox2d/pybox2d#144](https://github.com/pybox2d/pybox2d/issues/144)

Pinned to **3.12**.

## Pinned versions

| Package | Pin | Why exact |
|---|---|---|
| `gymnasium` | `==1.3.0` | The box2d packaging fix (PR #1474) shipped in 1.2.3 |
| `numpy` | `>=2.4,<3` | 2.5.x is only weeks old; 2.4.x is the more battle-tested line |
| `torch` | `==2.14.0` | See MPS note below |
| `box2d` | `==2.3.10` | Pinned by the gymnasium box2d extra |
| `pygame-ce` | `>=2.1.3` | 1.3.0 switched from `pygame` to `pygame-ce`; same `import pygame` |

## Never install `gymnasium[all]`

The `all` extra in gymnasium 1.3.0 **still references `box2d-py==2.3.5`**, even
though the `box2d` extra was fixed.

Two problems:

1. `box2d-py` has no macOS arm64 wheel. Building it from source requires
   SWIG, and even with SWIG present the build fails on recent setuptools.
2. It collides with `box2d` on the same `Box2D` import name, so whichever
   installs last silently shadows the other.

Declare the Box2D deps directly instead, which is what `pyproject.toml` does:

```toml
[project.optional-dependencies]
box2d = ["box2d==2.3.10", "pygame-ce>=2.1.3"]
```

`tests/test_environment.py::test_no_box2d_py_installed` guards against this
regressing.

## Train on CPU, not MPS

`torch.multinomial` on MPS can return a zero-probability index.
[pytorch#192577](https://github.com/pytorch/pytorch/issues/192577).

- `exponential_` on MPS emits exact `-0.0`; the n=1 Gumbel fast path then
  computes `0 / -0 = NaN`, and MPS `argmax` propagates it.
- This is the `Categorical.sample()` path — **exactly what PPO uses**.
- Labelled `correctness (silent)` upstream.
- The fix merged to `main` 2026-08-10 but sits 7 commits past the `v2.14.0`
  tag. It first ships in 2.15.0.

Reported rate is ~1/220 draws for a 152k-category distribution with
underflowing tails. For a 2-action CartPole policy probabilities are never
exactly zero, so reachability at this scale is unverified — but the failure is
silent action corruption, and the downside is far worse than the upside.

CPU is also the better choice here regardless: these experiments are many
small independent runs, so 12 single-threaded processes beat one MPS process.

## CleanRL is a hyperparameter reference, not a dependency

`vwxyzjn/cleanrl`, file `cleanrl/ppo.py`:

- last tagged release `v1.0.0` (2022-11-14); file untouched since 2023-11-28
- predates the Gymnasium 1.0 autoreset rework — it reads `infos["final_info"]`
  and unpacks a pre-1.0 `SyncVectorEnv` return signature

Vendor the algorithm and **rewrite** the environment/autoreset handling
against gymnasium 1.3.0. Verify that episode-stat logging actually fires; a
silently no-op stats write is the likely failure mode.

Baseline hyperparameters used (from `master:cleanrl/ppo.py`):

| param | value |
|---|---|
| `num_envs` | 4 |
| `num_steps` | 128 |
| `learning_rate` | 2.5e-4 (linear anneal) |
| `clip_coef` | 0.2 |
| `gae_lambda` | 0.95 |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 |
| `gamma` | 0.99 |
| `num_minibatches` | 4 |
| `update_epochs` | 4 |
| `max_grad_norm` | 0.5 |
| `norm_adv` | True |
| `clip_vloss` | True |

## Toolchain

- `uv 0.12.13`, Python managed via `uv python pin`
- `ruff` for lint and format
- `mypy --strict` for types

`ty` was considered and rejected: it is pre-1.0 and its diagnostics shift
between releases, which makes `make check` non-reproducible. `mypy` is stable.
Swapping is a one-line change in `Makefile` and `pyproject.toml`.

## Parallelism

Process-level, `spawn` (not `fork`) — forking a process that has already
initialised torch and the BLAS thread pool is a known macOS deadlock source.

Each worker pins `OMP_NUM_THREADS=1` and friends; the `Makefile` exports these
before Python starts, since setting them after the BLAS runtime has sized its
pool has no effect.

## macOS 26.7 / Apple M5 Pro

15 cores, 24 GB. `torch` publishes `macosx_*_arm64` wheels only; Intel macOS
support is dropped. Not relevant on Apple Silicon, but it means this project
does not run on x86_64 Macs at all.