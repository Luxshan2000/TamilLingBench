# -*- coding: utf-8 -*-
"""Uncertainty on the prior.

Two distinct sources, both reported:

1. **Sampling error within a corpus** — a *cluster* bootstrap over documents. Forms cluster
   hard by document (one article about a saint uses தாங்கள் throughout), so an i.i.d. token
   bootstrap is badly overconfident. `DEFF = Var_cluster / Var_iid` is reported alongside;
   when it is large, that is *why* the interval is wide, and saying so pre-empts the
   obvious "your CI looks too wide" reviewer note.
2. **Corpus-choice variance** — handled in `priors.py`, because it is between-corpus and
   this module only sees one corpus at a time. It is usually the larger term.

No scipy: the normal quantile is a constant and the bootstrap is a numpy resample.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

Z95 = 1.959963984540054

#: Below this many contributing documents the bootstrap uses exact multinomial weights;
#: above it, Poisson(1) weights. The two agree to O(1/m) and the switch is what keeps a
#: 10^5-cluster bootstrap inside seconds instead of minutes.
POISSON_MIN_DOCS = 2_000


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.lo, self.hi)

    @property
    def width(self) -> float:
        return self.hi - self.lo


def wilson(k: int, n: int, z: float = Z95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Reported as the *i.i.d.* reference only. It is the interval you would get if tokens
    were independent, which they are not — its role here is to be the denominator of DEFF.
    """
    if n == 0:
        return Interval(0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(max(0.0, (c - h) / d), min(1.0, (c + h) / d))


@dataclass(frozen=True)
class BootstrapResult:
    p: dict[str, float]
    ci: dict[str, tuple[float, float]]
    n: dict[str, int]
    total: int
    n_docs: int
    n_docs_nonzero: int
    deff: dict[str, float]
    draws: np.ndarray | None = None       # (B, k) resampled proportions, for DFR propagation
    values: tuple[str, ...] = ()

    def to_json(self, with_draws: bool = False) -> dict:
        d = {"p": self.p, "ci": {k: list(v) for k, v in self.ci.items()}, "n": self.n,
             "total": self.total, "n_docs": self.n_docs,
             "n_docs_nonzero": self.n_docs_nonzero, "deff": self.deff,
             "values": list(self.values)}
        if with_draws and self.draws is not None:
            d["draws_summary"] = {
                v: {"mean": float(self.draws[:, i].mean()),
                    "sd": float(self.draws[:, i].std(ddof=1))}
                for i, v in enumerate(self.values)}
        return d


def cluster_bootstrap(doc_counts: Mapping[str, Mapping[int, int]],
                      values: Sequence[str],
                      n_docs: int,
                      n_boot: int = 10_000,
                      seed: int = 20260807) -> BootstrapResult:
    """Resample DOCUMENTS with replacement; recompute the value distribution each time.

    `doc_counts[value][doc_id] = count`. The cluster is the DOCUMENT, not the token: one
    article uses தாங்கள் throughout, so an i.i.d. token bootstrap is badly overconfident.
    `n_docs` is the corpus's full document count and is carried through for reporting; only
    the documents that actually contain a token of this slot enter the resample, because
    the statistic is a ratio and a zero row cannot move it.
    """
    values = tuple(values)
    docs = sorted({d for v in values for d in doc_counts.get(v, {})})
    idx = {d: i for i, d in enumerate(docs)}
    m = np.zeros((len(docs), len(values)), dtype=np.int64)
    for j, v in enumerate(values):
        for d, c in doc_counts.get(v, {}).items():
            m[idx[d], j] = c

    totals = m.sum(axis=0)
    grand = int(totals.sum())
    p = {v: (float(totals[j]) / grand if grand else 0.0) for j, v in enumerate(values)}
    n = {v: int(totals[j]) for j, v in enumerate(values)}

    if not len(docs) or grand == 0:
        return BootstrapResult(p=p, ci={v: (0.0, 1.0) for v in values}, n=n, total=grand,
                               n_docs=n_docs, n_docs_nonzero=len(docs),
                               deff={v: float("nan") for v in values}, values=values)

    # Documents that contain NO token of this slot are irrelevant to the statistic: the
    # ratio depends only on the documents that contribute counts, so resampling the zero
    # documents cannot move it. They are therefore dropped here, not "forgotten".
    #
    # The resample itself is a WEIGHT draw rather than an index draw. Drawing m document
    # indices per replicate and summing rows is O(B·m) in Python-visible work and, measured
    # on CC-100 (m ≈ 10^5 documents, B = 10^4), did not finish in 8 minutes. Drawing a
    # weight vector and taking one matrix product is the same estimator with the loop
    # pushed into BLAS.
    #   * m ≤ POISSON_MIN_DOCS → exact multinomial weights, i.e. the textbook n-out-of-n
    #     nonparametric bootstrap over documents;
    #   * m >  POISSON_MIN_DOCS → Poisson(1) weights (Hanley & MacGibbon's Poisson
    #     bootstrap), which is the standard large-m approximation and is what makes this
    #     tractable at 10^5 clusters.
    rng = np.random.default_rng(seed)
    B = n_boot
    md = len(docs)
    mf = m.astype(np.float32)
    chunk = max(1, min(B, int(4e7 // max(md, 1)) or 1))
    out: list[np.ndarray] = []
    done = 0
    while done < B:
        c = min(chunk, B - done)
        if md <= POISSON_MIN_DOCS:
            w = rng.multinomial(md, np.full(md, 1.0 / md), size=c).astype(np.float32)
        else:
            w = rng.poisson(1.0, size=(c, md)).astype(np.float32)
        s = w @ mf                                   # (c, |values|)
        t = s.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(t > 0, s / t, np.nan))
        done += c
    draws = np.concatenate(out).astype(np.float64)
    ok = ~np.isnan(draws[:, 0])
    d_ok = draws[ok]
    if not len(d_ok):
        return BootstrapResult(p=p, ci={v: (0.0, 1.0) for v in values}, n=n, total=grand,
                               n_docs=n_docs, n_docs_nonzero=md,
                               deff={v: float("nan") for v in values}, values=values)
    ci = {v: (float(np.percentile(d_ok[:, j], 2.5)),
              float(np.percentile(d_ok[:, j], 97.5))) for j, v in enumerate(values)}

    deff: dict[str, float] = {}
    for j, v in enumerate(values):
        var_cluster = float(d_ok[:, j].var(ddof=1))
        pv = p[v]
        var_iid = pv * (1 - pv) / grand if grand else float("nan")
        deff[v] = var_cluster / var_iid if var_iid else float("nan")

    return BootstrapResult(p=p, ci=ci, n=n, total=grand, n_docs=n_docs,
                           n_docs_nonzero=md, deff=deff, draws=d_ok, values=values)


# --------------------------------------------------------------------------- divergences

def tvd(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def jsd(p: Mapping[str, float], q: Mapping[str, float], base: float = 2.0) -> float:
    keys = sorted(set(p) | set(q))
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    mv = 0.5 * (pv + qv)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask]) / np.log(base)))

    return 0.5 * _kl(pv, mv) + 0.5 * _kl(qv, mv)


def log_ratio(p: float, q: float) -> float:
    if p <= 0 or q <= 0:
        return float("-inf") if p <= 0 else float("inf")
    return math.log2(p / q)
