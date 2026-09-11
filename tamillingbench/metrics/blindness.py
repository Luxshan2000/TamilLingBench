# -*- coding: utf-8 -*-
"""Direction, magnitude, sensitivity ratio z, and ToShip calibration.

The section's entire argument depends on keeping three things apart that a single "does the metric
work?" number would merge:

  DIRECTION  does the metric rank the correct form above the wrong one?   (a win rate)
  MAGNITUDE  by how much, in the metric's own units?                      (a delta)
  z          how big is that next to producing nothing at all?            (DEMETR's ratio)

A metric can win the pairwise test on 95% of items and still charge a penalty a human would never
notice. Reporting only direction hides that; reporting only magnitude is scale-dependent and
unfalsifiable. Both, plus a calibration into "probability a human would agree", is what licenses the
claim.

z's DENOMINATOR — VERIFIED against the PDF 2026-08-08, and it IS DEMETR's, not a variant.
Karpinska, Raj, Thai, Song, Gupta & Iyyer, *DEMETR: Diagnosing Evaluation Metrics for
Translation*, EMNLP 2022, `2022.emnlp-main.649`, §4 "Measuring sensitivity", Eq. (1):

    z_i = [ SCORE(r_i, t_i) - SCORE(r_i, t'_i) ] / [ SCORE(r_i, t_i) - SCORE(r_i, empty) ]

aggregated as a plain mean over items. Our `z_floor` is the same quantity. TWO DEVIATIONS, both
of which must appear in the paper rather than be smoothed over:

  1. DEMETR's "empty string" is **a full stop**, not an empty string -- Appendix Table A1,
     perturbation 32: "since most automatic metrics will not allow an empty string we pass a full
     stop instead". We pass a genuinely empty string. Every metric in our panel accepted it, but
     it makes our denominator slightly larger than DEMETR's, so our z is if anything slightly
     SMALLER than the DEMETR-comparable value -- conservative in the direction of our claim. For
     chrF the denominator is exactly 100 by construction.
  2. DEMETR's SCORE is reference-based on both terms. For the reference-free metrics in our panel
     (CometKiwi-22, MetricX-24-QE) the argument structure is SCORE(src, ·), so their z is an
     analogue rather than the same statistic. Flagged wherever a QE z is reported.

DEMETR's own hedge (footnote 20) applies to us too: the ratio depends heavily on the score given
to the floor hypothesis, so the floor scores are reported as `levels.empty` alongside every z.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

#: Higher-is-better orientation. MetricX is 0-25 lower-is-better; everything else is already
#: higher-is-better. Applied ONCE, here, before any statistic touches a score.
ORIENT = {"metricx24xl": -1.0, "metricx24xlqe": -1.0, "metricx23large": -1.0}

#: Our metric name -> (ToShip key or None, scale factor from NATIVE units to the FITTED units).
#: The scale factor is the 100x trap: the ToShip fits are on each metric's conventionally
#: reported scale, which is 0-100 for the COMET family (native 0-1) and 0-25 for MetricX
#: (native 0-25, factor 1). `tests/test_toship_scale.py` is the guard.
TOSHIP = {
    "comet22":        ("comet22", 100.0),
    "cometkiwi22":    ("cometkiwi22", 100.0),
    "xcometxl":       ("xcometxl", 100.0),
    "xcometxxl":      ("xcometxxl", 100.0),
    "chrf":           ("chrf", 1.0),
    "chrfpp":         (None, 1.0),
    "metricx23large": ("metricx-23-large", 1.0),
    "metricx24xl":    (None, 1.0),
    "metricx24xlqe":  (None, 1.0),
}

EPS = 1e-3          # z guard, in each metric's NATIVE units (pre-orientation magnitude)
B_BOOT = 10_000
SEED = 20260809


def orient(metric: str, scores: np.ndarray) -> np.ndarray:
    return ORIENT.get(metric, 1.0) * np.asarray(scores, dtype=float)


# --------------------------------------------------------------------- direction

@dataclass
class Direction:
    n: int
    n_win: int
    n_tie: int
    n_loss: int
    win_rate: float          # ties split 0.5
    p_binomial: float        # two-sided exact binomial on NON-tied pairs

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def direction(good: np.ndarray, bad: np.ndarray) -> Direction:
    """Pairwise win rate with ties split, plus an exact binomial test on the non-tied pairs.

    Ties are counted and reported separately on purpose: chrF produces genuine ties on Tamil
    minimal pairs (identical character n-gram counts), neural metrics essentially never do, and a
    win rate of 0.5 built out of ties means something very different from a coin flip.
    """
    good, bad = np.asarray(good, float), np.asarray(bad, float)
    win = int((good > bad).sum())
    tie = int((good == bad).sum())
    loss = int((good < bad).sum())
    n = len(good)
    wr = (win + 0.5 * tie) / n if n else float("nan")
    nz = win + loss
    p = float(stats.binomtest(win, nz, 0.5, alternative="two-sided").pvalue) if nz else float("nan")
    return Direction(n, win, tie, loss, wr, p)


# --------------------------------------------------------------------- magnitude

@dataclass
class Magnitude:
    n: int
    mean: float
    median: float
    q25: float
    q75: float
    frac_nonpositive: float   # share of items where the metric does NOT prefer the correct form
    native_scale: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def magnitude(good: np.ndarray, bad: np.ndarray, native_scale: str) -> Magnitude:
    d = np.asarray(good, float) - np.asarray(bad, float)
    return Magnitude(len(d), float(d.mean()), float(np.median(d)),
                     float(np.percentile(d, 25)), float(np.percentile(d, 75)),
                     float((d <= 0).mean()), native_scale)


# --------------------------------------------------------------------- z

@dataclass
class ZFloor:
    n_used: int
    n_dropped_eps: int
    mean: float
    ci: tuple[float, float]
    median: float

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["ci"] = list(self.ci)
        return d


def z_floor(good: np.ndarray, bad: np.ndarray, empty: np.ndarray,
            clusters: list[str], eps: float = EPS,
            b: int = B_BOOT, seed: int = SEED) -> ZFloor:
    """z_i = (s_good - s_bad) / (s_good - s_empty), with the degenerate-denominator guard.

    z ~ 1 means the morpheme costs as much as producing nothing at all; z ~ 0 means invisible.
    Items whose denominator is below `eps` are DROPPED and counted -- dividing by a near-zero
    floor manufactures arbitrarily large z and would be the easiest way to fake this result.
    """
    good, bad, empty = (np.asarray(x, float) for x in (good, bad, empty))
    den = good - empty
    keep = np.abs(den) >= eps
    z = (good[keep] - bad[keep]) / den[keep]
    cl = [c for c, k in zip(clusters, keep) if k]
    lo, hi = cluster_bootstrap_ci(z, cl, b=b, seed=seed)
    return ZFloor(int(keep.sum()), int((~keep).sum()),
                  float(z.mean()) if len(z) else float("nan"), (lo, hi),
                  float(np.median(z)) if len(z) else float("nan"))


def cluster_bootstrap_ci(values: np.ndarray, clusters: list[str],
                         b: int = B_BOOT, seed: int = SEED,
                         alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI resampling CLUSTERS (template_id), not items.

    Items generated from one template share a frame, a verb inventory and a source shape; treating
    them as independent understates the interval, sometimes badly (design effects of 2-24 were
    measured on the same data).
    """
    values = np.asarray(values, float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    by: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        by.setdefault(c, []).append(i)
    keys = list(by)
    idx = [np.asarray(by[k]) for k in keys]
    if len(keys) < 2:
        return (float(values.mean()), float(values.mean()))
    draws = rng.integers(0, len(keys), size=(b, len(keys)))
    means = np.empty(b)
    for j in range(b):
        sel = np.concatenate([idx[t] for t in draws[j]])
        means[j] = values[sel].mean()
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


# --------------------------------------------------------------------- calibration

@dataclass
class Calibration:
    toship_key: str | None
    available: bool
    scale_factor: float | None = None
    delta_native: float | None = None
    delta_fitted: float | None = None
    acc_at_observed: float | None = None
    d_75: float | None = None
    d_90: float | None = None
    ratio_75: float | None = None
    ratio_90: float | None = None
    saturated: bool | None = None
    asymptote: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def calibrate(metric: str, mean_delta_native: float) -> Calibration:
    """Turn a native mean delta into "probability a human comparison would agree".

    Returns `available=False` with a note for MetricX-24 and GEMBA-ESA. Those two are reported as
    `—` in every table: the two newest metrics in the panel have no published human-agreement
    calibration at all, which is itself a small finding about the state of the field. They are
    NEVER given another metric's thresholds.
    """
    from mt_thresholds import accuracy, delta

    key, factor = TOSHIP.get(metric, (None, 1.0))
    if key is None:
        return Calibration(None, False, note=(
            "no entry in the Kocmi et al. (ACL 2024) ToShip23 threshold table; reported "
            "uncalibrated. Borrowing another checkpoint's thresholds would be wrong: the fits "
            "are tied to specific checkpoints, not to a metric family."))
    d_fitted = abs(mean_delta_native) * factor
    acc = accuracy(d_fitted, key) / 100.0
    asym = accuracy(1e6, key) / 100.0
    d75, d90 = delta(0.75, key), delta(0.90, key)
    # ⚠ The ToShip fit is a sigmoid with ceiling `a` (93-99% depending on the metric). Once the
    # observed delta is a few multiples of d_90 the accuracy is PINNED AT THE CEILING and carries
    # no further information: "96.2% human agreement" for COMET-22 is just its asymptote. The
    # informative number in that regime is `ratio_75`, not `acc_at_observed`, and the flag exists
    # so the paper cannot quote the saturated value as if it were an estimate.
    saturated = bool(acc >= asym * (1 - 1e-3))
    return Calibration(
        key, True, factor, float(mean_delta_native), float(d_fitted), float(acc),
        float(d75), float(d90),
        float(d_fitted / d75) if d75 else None,
        float(d_fitted / d90) if d90 else None,
        saturated, float(asym),
        note=("delta rescaled from native units by x%g to the ToShip fitted scale" % factor)
             + ("; ACCURACY IS SATURATED at the fit's ceiling -- quote ratio_75, not "
                "acc_at_observed" if saturated else ""))


# --------------------------------------------------------------------- holm

def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni within a family. Returns key -> reject-at-alpha."""
    items = sorted(((k, v) for k, v in pvals.items() if not math.isnan(v)), key=lambda kv: kv[1])
    m = len(items)
    out = {k: False for k in pvals}
    for i, (k, p) in enumerate(items):
        if p <= alpha / (m - i):
            out[k] = True
        else:
            break
    return out
