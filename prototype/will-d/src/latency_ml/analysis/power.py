"""Minimum detectable effect for a paired comparison, and the verdict it licenses.

Added after this repository's first real comparison came back null, for a reason
that was not visible in the point estimates. Ten seeds per cell could resolve a
LunarLander difference of 13.5 to 67.4 return units depending on the cell, and
the differences actually observed were 0.8 to 15.1. Every one of them was well
inside the noise, so "the bounded activation did not change anything" and "this
design could not have seen a change of that size" were indistinguishable from the
result alone. They are not the same claim, and only one of them is usually true.

What this module refuses to produce
-----------------------------------
There is deliberately no function here that returns "no effect". A comparison
either resolves a difference or it does not, and the failure to resolve one is
evidence about the experiment, never about the thing measured. ``verdict``
returns ``detected`` or ``inconclusive`` and collapses the ambiguous case rather
than letting it default to absence:

``detected``
    The observed difference is at or above the smallest difference this design
    could have resolved at the stated power, and therefore clears the critical
    value.
``inconclusive``
    The observed difference is smaller than the smallest difference this design
    could have resolved. Nothing follows about the underlying quantity.

Why there are only two states
-----------------------------
A third state -- "resolvable in magnitude, yet the test did not clear the
critical value" -- was in the first version of this module and is
**algebraically unreachable**, which is worth recording rather than deleting
quietly. With the minimum detectable effect defined at the requested power,

    observable   <=>  |mean| * sqrt(n) / sd >= t_crit + t_power
    significant  <=>  |mean| * sqrt(n) / sd >  t_crit

and ``t_power`` is positive, so observable implies significant with margin. A
verdict that cannot distinguish two situations is not a verdict, so the state was
removed rather than kept as decoration. The property that makes it unreachable
is the reassuring one: an effect big enough for this design to have had a fair
chance at resolving it did resolve it.

The reachable way to say "the effect was resolvable but this design is weak" is
to read ``detectable_effect`` alongside the verdict, which is what ``describe``
does.

Why paired
----------
Seeds are paired by construction: the same seed draws the same initial weights
and the same minibatch order on both sides, so the two arms are not independent
samples of a population. Differencing seed by seed removes the between-seed
variance, which is by far the dominant term here -- it leaves a residual spread
several times smaller than either arm's own spread, and that is the whole reason
a small effect is visible at all against this noise floor. Testing the arms as
independent would inflate the standard error by roughly the correlation penalty
and could turn a resolvable difference into an inconclusive one.

Why the t quantiles rather than the normal ones
-----------------------------------------------
At n = 10 the normal approximation understates the minimum detectable effect by
roughly a tenth. That is in the wrong direction for a module whose purpose is
to stop over-claiming, so this uses the two-sided critical value and the
quantile at the requested power, both from Student's t with n - 1 degrees of
freedom:

    MDE = (t_{1 - alpha/2, n-1} + t_{power, n-1}) * sd(differences) / sqrt(n)

This is the standard approximation. It is not merely "close enough, and
conservative" as was first claimed here -- the conservatism claim was itself
wrong, and was an artefact of comparing the minimum detectable effect in
units of the residual spread against a mean shift in units of the observation
spread, which differ by a factor of sqrt(n) and happened to land near each other.
Checked directly by simulation at n = 10, the noncentrality giving 80% power is
3.150 against the formula's 3.146, an error of 0.15%, and simulating a true
effect equal to the returned detectable effect reproduces power = 0.800. Both
checks are tests.

When the verdict and the significance test disagree
---------------------------------------------------
They can, and in the one direction that matters they will. An effect can clear
the critical value in one particular sample while still sitting below the size
this design was powered to resolve -- CartPole at 80 ms does exactly that, with
|t| = 2.53 against a critical 2.26 on an effect of 1.7 where the detectable
effect is 2.1.

``verdict`` follows the design, not the sample, and reports that case
``inconclusive``. That is deliberate. Significance is a statement about one draw
at one sample size; the detectable effect is a statement about what this design
can be relied on to reproduce. A single un-replicated effect that clears the bar
on the first look is precisely the result that fails to replicate, so reporting
it as ``detected`` would repeat the error this module exists to prevent. Both
numbers are always carried on the result so a reader can disagree.

What this is not for
--------------------
This says nothing about the number of seeds that *would* be required, because
that depends on an effect size nobody has measured yet. If a comparison comes
back inconclusive, the next step is to state the smallest effect worth detecting
and solve for n against it -- not to reinterpret this one as a null.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

DEFAULT_ALPHA: float = 0.05
DEFAULT_POWER: float = 0.80

Verdict = Literal["detected", "inconclusive"]


class ComparisonError(ValueError):
    """Raised when two samples cannot form the comparison they are asked for."""


_BETA_MAXITER = 200
_BETA_EPS = 3.0e-16
_BETA_TINY = 1.0e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method).

    Modified Lentz evaluation, as in Numerical Recipes. Written out rather than
    imported because scipy is not a dependency and this repository has a
    deliberate policy of pinning a small dependency set.
    """
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _BETA_TINY:
        d = _BETA_TINY
    d = 1.0 / d
    h = d
    for m in range(1, _BETA_MAXITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _BETA_TINY:
            d = _BETA_TINY
        c = 1.0 + aa / c
        if abs(c) < _BETA_TINY:
            c = _BETA_TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _BETA_TINY:
            d = _BETA_TINY
        c = 1.0 + aa / c
        if abs(c) < _BETA_TINY:
            c = _BETA_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETA_EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(t: float, df: int) -> float:
    """``P(T <= t)`` for Student's t with ``df`` degrees of freedom."""
    tail = 0.5 * _betai(0.5 * df, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0.0 else tail


def _student_t(quantile: float, df: int) -> float:
    """Student-t quantile by bisection on the CDF.

    scipy is not a dependency, and the normal approximation is precisely the
    thing this module exists to avoid: at df = 9 it understates the critical
    value by 13% and the minimum detectable effect by about a tenth, both in the
    direction that flatters a null. The CDF is therefore integrated here from
    the incomplete beta function rather than shortcut.
    """
    if not 0.0 < quantile < 1.0:
        raise ComparisonError(f"quantile must lie in (0, 1), got {quantile!r}")
    if df < 1:
        raise ComparisonError(f"degrees of freedom must be >= 1, got {df}")
    lo, hi = -1.0e3, 1.0e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _student_t_cdf(mid, df) < quantile:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def student_t_quantile(quantile: float, df: int) -> float:
    """Public wrapper for the Student-t quantile.

    Exposed because the critical value is worth quoting alongside a verdict: "not
    significant" is more legible as "|t| = 1.00 against a critical 2.26" than as
    a bare boolean.
    """
    return _student_t(quantile, df)


def _as_float_array(
    values: Sequence[float] | NDArray[np.floating], name: str
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ComparisonError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ComparisonError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise ComparisonError(
            f"{name} contains a non-finite value; a missing episode is not a datum"
        )
    return array


def paired_differences(
    treatment: Sequence[float] | NDArray[np.float64],
    control: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Seed-wise ``treatment - control``.

    Order is significant and must correspond across the two arms. Callers should
    not sort either side first: sorting destroys the pairing and silently turns
    a paired test into an unpaired one with a smaller standard error, which is
    the most dangerous way to get this wrong.
    """
    a = _as_float_array(treatment, "treatment")
    b = _as_float_array(control, "control")
    if a.shape != b.shape:
        raise ComparisonError(
            f"paired comparison needs equal-length samples, got {a.size} and {b.size}"
        )
    return a - b


def minimum_detectable_effect(
    differences: Sequence[float] | NDArray[np.float64],
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> float:
    """Smallest true mean difference this design could have resolved.

    Uses the sample standard deviation of the seed-wise differences, which is the
    only variance term a paired design has.
    """
    d = _as_float_array(differences, "differences")
    if d.size < 2:
        raise ComparisonError(f"a detectable effect needs at least 2 pairs, got {d.size}")
    df = d.size - 1
    sd = float(np.std(d, ddof=1))
    if sd == 0.0:
        return 0.0
    scale = _student_t(1.0 - alpha / 2.0, df) + _student_t(power, df)
    return scale * sd / math.sqrt(d.size)


def required_seeds(
    differences: Sequence[float] | NDArray[np.float64],
    target_effect: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> int:
    """Seeds needed to resolve ``target_effect`` at this observed dispersion.

    Provided so that an inconclusive result can be answered with a number rather
    than a shrug. ``target_effect`` is supplied by the caller on purpose: this
    module cannot know what size of effect would be worth detecting, and
    inferring that from an already-inconclusive comparison is how an experiment
    quietly drifts toward reporting what it happened to find.
    """
    d = _as_float_array(differences, "differences")
    if target_effect <= 0.0:
        raise ComparisonError(f"target_effect must be positive, got {target_effect!r}")
    sd = float(np.std(d, ddof=1))
    if sd == 0.0:
        return 2

    def resolves(n: int) -> bool:
        scale = _student_t(1.0 - alpha / 2.0, n - 1) + _student_t(power, n - 1)
        return scale * sd / math.sqrt(n) <= target_effect

    n = max(2, math.ceil((2.0 * sd / target_effect) ** 2) + 1)
    for _ in range(256):
        if resolves(n):
            break
        n += max(1, math.ceil(n * 0.25))
    else:  # pragma: no cover - unreachable in practice
        raise ComparisonError(
            f"no seed count found for target_effect={target_effect!r} and sd={sd!r}"
        )
    while n > 2 and resolves(n - 1):
        n -= 1
    return n


@dataclass(frozen=True)
class PairedComparison:
    """A paired comparison, carrying the verdict its design actually licenses."""

    n: int
    mean_treatment: float
    mean_control: float
    mean_difference: float
    sd_difference: float
    standard_error: float
    t_statistic: float
    critical_t: float
    detectable_effect: float
    alpha: float
    power: float

    @property
    def observable(self) -> bool:
        """Whether an effect of the observed size was resolvable at all.

        Zero spread in the seed-wise differences splits two ways. A constant
        nonzero difference means the two runs differ by exactly that amount on
        every seed, which is the most resolved result a paired comparison can
        produce. A constant zero difference means the two runs are
        element-wise identical, which reports that they agreed -- not that the
        underlying quantity was measured -- so that case stays inconclusive.
        """
        if self.detectable_effect <= 0.0:
            return self.mean_difference != 0.0
        return abs(self.mean_difference) >= self.detectable_effect

    @property
    def significant(self) -> bool:
        return abs(self.t_statistic) > self.critical_t

    @property
    def verdict(self) -> Verdict:
        """Three-way outcome. Never collapses to "no effect"."""
        return "detected" if self.observable else "inconclusive"

    def describe(self) -> str:
        """One line suitable for a results table."""
        return (
            f"{self.mean_difference:+.2f} "
            f"(t={self.t_statistic:+.2f} vs crit {self.critical_t:.2f}, "
            f"MDE {self.detectable_effect:.2f}) -> {self.verdict}"
        )


def compare_paired(
    treatment: Sequence[float] | NDArray[np.float64],
    control: Sequence[float] | NDArray[np.float64],
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> PairedComparison:
    """Compare two seed-aligned samples and return the licensed verdict."""
    d = paired_differences(treatment, control)
    a = _as_float_array(treatment, "treatment")
    b = _as_float_array(control, "control")
    n = d.size
    sd = float(np.std(d, ddof=1))
    se = sd / math.sqrt(n) if sd > 0.0 else 0.0
    mean_diff = float(np.mean(d))
    critical = _student_t(1.0 - alpha / 2.0, n - 1)
    if se > 0.0:
        t_stat = mean_diff / se
    else:
        t_stat = 0.0 if mean_diff == 0.0 else math.copysign(math.inf, mean_diff)
    return PairedComparison(
        n=n,
        mean_treatment=float(np.mean(a)),
        mean_control=float(np.mean(b)),
        mean_difference=mean_diff,
        sd_difference=sd,
        standard_error=se,
        t_statistic=t_stat,
        critical_t=critical,
        detectable_effect=minimum_detectable_effect(d, alpha=alpha, power=power),
        alpha=alpha,
        power=power,
    )
