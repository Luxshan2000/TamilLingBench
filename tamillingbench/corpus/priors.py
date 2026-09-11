# -*- coding: utf-8 -*-
"""Assemble the DFR prior tables and the cross-corpus agreement analysis.

The headline object is a `SlotPrior`: one slot, one counting variant, per-corpus rates with
cluster-bootstrap intervals, a token-weighted pooled rate, and — the part that actually
decides whether the DFR claim survives — the **between-corpus range**.

The design predicted that the between-corpus range would exceed the within-corpus CI, i.e.
that *"the dominant uncertainty in the prior is which corpus you choose, not how many
tokens you count."* This module computes that comparison rather than asserting it, and
`SlotPrior.verdict` states the answer in words.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .index import FormIndex
from .slots import (SECOND_PERSON_CUES, SECOND_PERSON_VERB_SUFFIXES, SlotSpec)
from .stats import BootstrapResult, cluster_bootstrap, jsd, tvd, wilson
from .text import canonicalize, tokenize


@dataclass
class CorpusPrior:
    corpus: str
    slot: str
    variant: str
    values: tuple[str, ...]
    n: dict[str, int]
    p: dict[str, float]
    ci: dict[str, tuple[float, float]]
    ci_iid: dict[str, tuple[float, float]]
    deff: dict[str, float]
    total: int
    n_docs: int
    n_docs_with_slot: int
    n_tokens: int
    register: str
    licence: str
    release_ok: bool

    @property
    def rate_per_million(self) -> float:
        return 1e6 * self.total / self.n_tokens if self.n_tokens else 0.0

    def to_json(self) -> dict:
        d = asdict(self)
        d["values"] = list(self.values)
        d["ci"] = {k: list(v) for k, v in self.ci.items()}
        d["ci_iid"] = {k: list(v) for k, v in self.ci_iid.items()}
        d["slot_tokens_per_million"] = round(self.rate_per_million, 2)
        return d


@dataclass
class SlotPrior:
    slot: str
    variant: str
    values: tuple[str, ...]
    per_corpus: list[CorpusPrior]
    pooled_p: dict[str, float]
    pooled_n: dict[str, int]
    pooled_ci: dict[str, tuple[float, float]]
    between_corpus_range: dict[str, tuple[float, float]]
    headline_interval: dict[str, tuple[float, float]]
    max_pairwise_tvd: float
    max_pairwise_jsd: float
    disagreeing_pair: tuple[str, str]
    within_vs_between: dict[str, dict[str, float]]
    caveats: list[str] = field(default_factory=list)

    @property
    def fragile(self) -> bool:
        """True when the prior is not safe to headline as a single number.

        Two triggers, either sufficient: the corpora disagree more than a token-level
        interval can explain (max pairwise TVD > 0.10), or some value's between-corpus
        range is more than twice its widest within-corpus CI.
        """
        if self.max_pairwise_tvd > 0.10:
            return True
        return any(w["between_over_within"] > 2.0
                   for w in self.within_vs_between.values()
                   if w["between_over_within"] == w["between_over_within"])  # not NaN

    @property
    def verdict(self) -> str:
        n = len(self.per_corpus)
        if n < 2:
            return ("SINGLE CORPUS — no cross-corpus check was possible, so this prior "
                    "must not be presented as a distribution estimate for Tamil.")
        worst = max(self.within_vs_between.items(),
                    key=lambda kv: (kv[1]["between_over_within"]
                                    if kv[1]["between_over_within"] == kv[1]["between_over_within"]
                                    else -1))
        ratio = worst[1]["between_over_within"]
        lead = (f"between-corpus range is {ratio:.1f}× the widest within-corpus CI "
                f"(worst value: {worst[0]}); max pairwise TVD "
                f"{self.max_pairwise_tvd:.3f} between "
                f"{self.disagreeing_pair[0]} and {self.disagreeing_pair[1]}")
        if self.fragile:
            return ("FRAGILE — " + lead + ". Report the prior as a RANGE across corpora, "
                    "run DFR at both ends, and do not headline a point estimate.")
        return ("STABLE — " + lead + ". The corpora agree closely enough that a pooled "
                "point estimate with the union interval is defensible.")

    def to_json(self) -> dict:
        return {
            "slot": self.slot, "variant": self.variant, "values": list(self.values),
            "pooled_p": self.pooled_p, "pooled_n": self.pooled_n,
            "pooled_ci": {k: list(v) for k, v in self.pooled_ci.items()},
            "between_corpus_range": {k: list(v) for k, v in self.between_corpus_range.items()},
            "headline_interval": {k: list(v) for k, v in self.headline_interval.items()},
            "max_pairwise_tvd": self.max_pairwise_tvd,
            "max_pairwise_jsd": self.max_pairwise_jsd,
            "disagreeing_pair": list(self.disagreeing_pair),
            "within_vs_between": self.within_vs_between,
            "fragile": self.fragile,
            "verdict": self.verdict,
            "caveats": self.caveats,
            "per_corpus": [c.to_json() for c in self.per_corpus],
        }


# --------------------------------------------------------------------------- per corpus

def corpus_prior(idx: FormIndex, spec: SlotSpec, n_boot: int = 10_000,
                 seed: int = 20260807) -> CorpusPrior:
    all_forms = [f for v in spec.values for f in spec.forms.get(v, [])]
    per_form = idx.doc_counts(all_forms)
    by_value: dict[str, dict[int, int]] = {}
    for v in spec.values:
        acc: dict[int, int] = {}
        for f in spec.forms.get(v, []):
            for d, c in per_form.get(canonicalize(f), {}).items():
                acc[d] = acc.get(d, 0) + c
        by_value[v] = acc

    boot = cluster_bootstrap(by_value, spec.values, n_docs=idx.n_docs,
                             n_boot=n_boot, seed=seed)
    ci_iid = {v: wilson(boot.n[v], boot.total).as_tuple() for v in spec.values}
    return CorpusPrior(
        corpus=idx.corpus, slot=spec.slot, variant=spec.variant, values=spec.values,
        n=boot.n, p=boot.p, ci=boot.ci, ci_iid=ci_iid, deff=boot.deff,
        total=boot.total, n_docs=idx.n_docs, n_docs_with_slot=boot.n_docs_nonzero,
        n_tokens=idx.n_tokens,
        register=str(idx.meta.get("register", "")),
        licence=str(idx.meta.get("licence", "")),
        release_ok=bool(idx.meta.get("release_ok", False)))


# --------------------------------------------------------------------------- pooling

#: A corpus must contribute at least this many slot tokens to enter the between-corpus
#: spread. Set from the data, not from taste: UD_Tamil-MWTT contributes 35 rationality
#: tokens and IruMozhi 41, and at that size a single document swings the rate by tens of
#: points — letting them define a min–max range would manufacture disagreement. They stay
#: in the per-corpus table, where their smallness is visible.
MIN_TOKENS_FOR_SPREAD = 200


def pool(priors: Sequence[CorpusPrior], spec: SlotSpec,
         min_total: int = MIN_TOKENS_FOR_SPREAD) -> SlotPrior:
    """Token-weighted pool + the between-corpus spread."""
    usable = [c for c in priors if c.total >= min_total]
    pooled_n = {v: sum(c.n[v] for c in priors) for v in spec.values}
    grand = sum(pooled_n.values())
    pooled_p = {v: (pooled_n[v] / grand if grand else 0.0) for v in spec.values}

    # Pooled interval = union of the per-corpus cluster-bootstrap intervals. A bootstrap
    # over the merged document set would be tighter, and tighter is the wrong direction to
    # be wrong in when the corpora are not a random sample of "Tamil".
    if usable:
        pooled_ci = {v: (min(c.ci[v][0] for c in usable), max(c.ci[v][1] for c in usable))
                     for v in spec.values}
    else:
        pooled_ci = {v: (0.0, 1.0) for v in spec.values}

    between = {v: ((min(c.p[v] for c in usable), max(c.p[v] for c in usable))
                   if usable else (float("nan"), float("nan")))
               for v in spec.values}

    headline = {}
    within_vs_between = {}
    for v in spec.values:
        lo = min([pooled_ci[v][0]] + ([between[v][0]] if usable else []))
        hi = max([pooled_ci[v][1]] + ([between[v][1]] if usable else []))
        headline[v] = (lo, hi)
        widest_within = max([c.ci[v][1] - c.ci[v][0] for c in usable], default=float("nan"))
        b = (between[v][1] - between[v][0]) if usable else float("nan")
        within_vs_between[v] = {
            "widest_within_corpus_ci_width": widest_within,
            "between_corpus_range_width": b,
            "between_over_within": (b / widest_within) if widest_within else float("nan"),
        }

    worst_tvd, worst_jsd, pair = 0.0, 0.0, ("", "")
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            t = tvd(usable[i].p, usable[j].p)
            if t > worst_tvd:
                worst_tvd, pair = t, (usable[i].corpus, usable[j].corpus)
            worst_jsd = max(worst_jsd, jsd(usable[i].p, usable[j].p))

    return SlotPrior(
        slot=spec.slot, variant=spec.variant, values=spec.values, per_corpus=list(priors),
        pooled_p=pooled_p, pooled_n=pooled_n, pooled_ci=pooled_ci,
        between_corpus_range=between, headline_interval=headline,
        max_pairwise_tvd=worst_tvd, max_pairwise_jsd=worst_jsd, disagreeing_pair=pair,
        within_vs_between=within_vs_between,
        caveats=list(spec.caveats) + [
            "The pooled interval is the union of the per-corpus cluster-bootstrap "
            "intervals, not a bootstrap over the merged document set; unioning is the "
            "conservative direction (it can only widen the interval).",
            f"Corpora with <{min_total} slot tokens are shown per-corpus but excluded "
            "from the between-corpus spread.",
        ])


# --------------------------------------------------------------------------- brackets

@dataclass
class ThangalBracket:
    """தாங்கள் is 2nd-person DEFERENTIAL *or* 3rd-person REFLEXIVE ('they themselves').

    ThamizhiMorph gives only the reflexive reading (`தாங்கள்+pron+3pl+refl+nom`), so the
    analyser cannot bracket this and a corpus heuristic must. Cue: a co-occurring 2nd-person
    pronoun or 2nd-person finite verb suffix in the same sentence.

    This is a LOWER bound on the address reading (a deferential sentence with pro-drop and
    no other 2nd-person marker is scored as reflexive) and it is reported as such.
    """
    corpus: str
    form: str
    n_sentences_scanned: int
    n_with_2p_cue: int
    rate: float
    ci: tuple[float, float]
    examples_2p: list[str]
    examples_refl: list[str]


def thangal_bracket(idx: FormIndex, form: str = "தாங்கள்",
                    limit: int = 4000) -> ThangalBracket:
    f = canonicalize(form)
    rows = idx._db.execute(
        "SELECT text FROM sent WHERE text LIKE ? LIMIT ?", (f"%{f}%", limit)).fetchall()
    cues = {canonicalize(c) for c in SECOND_PERSON_CUES} - {f}
    n = hit = 0
    ex2: list[str] = []
    exr: list[str] = []
    for (text,) in rows:
        toks = tokenize(text)
        if f not in toks:
            continue
        n += 1
        is2 = any(t in cues for t in toks) or any(
            t.endswith(SECOND_PERSON_VERB_SUFFIXES) for t in toks)
        if is2:
            hit += 1
            if len(ex2) < 5:
                ex2.append(text)
        elif len(exr) < 5:
            exr.append(text)
    rate = hit / n if n else float("nan")
    return ThangalBracket(corpus=idx.corpus, form=f, n_sentences_scanned=n,
                          n_with_2p_cue=hit, rate=rate,
                          ci=wilson(hit, n).as_tuple() if n else (0.0, 1.0),
                          examples_2p=ex2, examples_refl=exr)


def second_person_density(idx: FormIndex) -> dict[str, float]:
    """Tokens per million for each 2nd-person pronoun. The register measurement.

    Tamil Wikipedia is third-person encyclopaedic prose; subtitles are dialogue. If the
    honorificity prior is estimated on Wikipedia, it is estimated on the tiny minority of
    sentences that address anyone at all — which is the register confound made numeric.
    """
    out = {}
    for f in ("நீ", "நீங்கள்", "தாங்கள்", "உன்", "உங்கள்", "தங்கள்"):
        out[f] = round(idx.frequency(f).per_million, 3)
    return out
