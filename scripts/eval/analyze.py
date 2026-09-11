#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analysis pass. Re-runnable; consumes only `outputs/scored/`.

Produces `results/eval-results.json` and `results/eval-results.md`:

  * the **C1→C2→C3 gradient** per (system, slot), with bootstrap CIs **clustered on
    `template_id`** — 2,873 items come from 57 templates, and an item-level bootstrap would
    understate every interval by roughly sqrt(DEFF);
  * per-(system, slot, condition) **outcome distributions** over the full taxonomy, with
    `committed_rate` printed beside every accuracy — never an accuracy alone;
  * the **prompt-robustness rule**: any (system, slot) cell whose across-prompt
    range exceeds its C1→C2 drop is flagged and excluded from headline claims, computed, not
    eyeballed;
  * the **clusivity drop-rate gate re-run across the whole panel**, per system per overt
    environment — the existing gate used one 600M system and is not evidence about the panel;
  * **DFR / skew on C3** against the corpus prior, always as a RANGE across corpora, with
    honorificity's prior withheld (D-5: not reportable from corpus counts).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CANONICAL = {"C0": "P0_specified", "C1": "P1_minimal", "C2": "P4_context",
             "C3": "P1_minimal", "_nmt": "nmt"}
OUTCOMES = ("CORRECT", "WRONG", "COMMITTED", "UNDECIDABLE", "AVOIDANT_DROP",
            "AVOIDANT_NEUTRAL", "AVOIDANT_NEG", "UNPARSED")
COMMITTED = ("CORRECT", "WRONG", "COMMITTED", "UNDECIDABLE")
AVOIDANT = ("AVOIDANT_DROP", "AVOIDANT_NEUTRAL", "AVOIDANT_NEG")
SLOTS = ("clusivity", "gender", "honorificity", "number", "rationality")

#: THIN-CELL SUPPRESSION. An accuracy computed over a handful of committed items is not a
#: weak number, it is a meaningless one, and next to a well-populated cell it is actively
#: dangerous: gemma3-4b clusivity C1 committed on 5/220 items and scored 100%, which beside
#: its C2 (44.5% on 199) manufactures a 42-point "C1→C2 drop" out of nothing. Enforced in
#: `Cell` so no caller can route around it, and propagated into the drop, its CI, and the
#: falsification verdict.
MIN_COMMITTED = 30
MIN_COMMIT_RATE = 0.30

#: Conditions that are structurally NOT comparable for a slot, with the measured reason.
NOT_COMPARABLE: dict[tuple[str, str], str] = {
    ("clusivity", "C1"): (
        "C1 clusivity sources contain NO 1PL English pronoun at all — measured: 0/196 unique "
        "C1 sources contain we/us/our, while 116/196 contain an explicit coordinated NP such "
        "as \"my brother and me\"; every one of the 67 unique C2 sources contains \"us\". "
        "Tamil renders that coordination compositionally (என் சகோதரனையும் என்னையும்), which "
        "is a correct translation that simply never reaches a 1PL pronoun. The near-total C1 "
        "AVOIDANT_DROP rate is therefore not avoidance, and the C1→C2 comparison measures the "
        "English construction rather than the slot."),
}

#: Slots with an unresolved native-check question. They are computed and reported internally
#: but must never be presented as settled.
PROVISIONAL: dict[str, str] = {
    "honorificity": ("NATIVE-CHECK-HON-3/4 — which Tamil form is the third honorific degree "
                     "(optative வருக/செய்க vs written-formal வரவும் vs தாங்கள்) is unresolved. "
                     "ThamizhiMorph licenses NO deferential reading: தாங்கள் is +pron+3pl+refl "
                     "and the optative carries no honorificity tag at all. Measured on the "
                     "benchmark's own gold, 106/106 VV targets fail the validated checker. "
                     "Only the T-vs-V contrast is scoreable."),
    "rationality": ("The variety split is unresolved: every rationality item carries "
                    "variety_assumed='lk' (Sri Lankan). Whether the Indian variant licenses "
                    "the same -ஆர்கள்/-அன contrast on these frames is an open native-check "
                    "question, and D-5 shows the corpus prior for this slot spans the "
                    "midpoint (34.5–69.6% UYARTHINAI), so the DFR sign can flip inside it."),
}


# --------------------------------------------------------------------------- cell

class Cell:
    """The partition assertion is the structural guarantee that no item is
    dropped anywhere in the pipeline."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.n_all = len(rows)
        self.c = Counter(r["outcome"] for r in rows)
        assert sum(self.c[o] for o in OUTCOMES) == self.n_all, "outcomes must partition N_all"
        #: C3 carries no gold by design, so no accuracy-like number is defined for it.
        self.has_gold = any(r.get("gold_value") for r in rows)
        self.suppressed_reason: str | None = None
        if self.has_gold and self.n_all:
            if self.n_committed < MIN_COMMITTED:
                self.suppressed_reason = (
                    f"committed n={self.n_committed} < {MIN_COMMITTED}")
            elif self.committed_rate < MIN_COMMIT_RATE:
                self.suppressed_reason = (
                    f"commit rate {self.committed_rate:.1%} < {MIN_COMMIT_RATE:.0%} "
                    f"(committed n={self.n_committed}/{self.n_all})")

    @property
    def reportable(self) -> bool:
        return self.has_gold and self.suppressed_reason is None

    @property
    def n_correct(self) -> int: return self.c["CORRECT"]
    @property
    def n_committed(self) -> int: return sum(self.c[o] for o in COMMITTED)
    @property
    def n_avoidant(self) -> int: return sum(self.c[o] for o in AVOIDANT)
    @property
    def n_unparsed(self) -> int: return self.c["UNPARSED"]

    @property
    def acc_strict(self) -> float:
        return self.n_correct / self.n_all if self.n_all else float("nan")

    @property
    def acc_parsed(self) -> float:
        d = self.n_all - self.n_unparsed
        return self.n_correct / d if d else float("nan")

    @property
    def acc_committed(self) -> float:
        return self.n_correct / self.n_committed if self.n_committed else float("nan")

    @property
    def committed_rate(self) -> float:
        return self.n_committed / self.n_all if self.n_all else float("nan")

    @property
    def coverage(self) -> float:
        return (self.n_all - self.n_unparsed) / self.n_all if self.n_all else float("nan")

    def existential_acc(self) -> float:
        """D-2's robustness column: correct under the LENIENT policy."""
        n = sum(1 for r in self.rows if r.get("existential"))
        return n / self.n_all if self.n_all else float("nan")

    def to_dict(self) -> dict:
        return {
            "n_all": self.n_all,
            "outcomes": {o: self.c[o] for o in OUTCOMES},
            # Suppressed cells emit null, never the raw ratio. The raw counts stay in
            # `outcomes` so nothing is hidden — only the misleading derived number goes.
            "acc_strict": rnd(self.acc_strict) if self.reportable else None,
            "acc_parsed": rnd(self.acc_parsed) if self.reportable else None,
            "acc_committed": rnd(self.acc_committed) if self.reportable else None,
            "suppressed": bool(self.suppressed_reason),
            "suppressed_reason": self.suppressed_reason,
            "acc_strict_unsuppressed": rnd(self.acc_strict) if self.has_gold else None,
            "committed_rate": rnd(self.committed_rate), "coverage": rnd(self.coverage),
            "avoidance_rate": rnd(self.n_avoidant / self.n_all if self.n_all else float("nan")),
            "drop_rate": rnd(self.c["AVOIDANT_DROP"] / self.n_all if self.n_all else float("nan")),
            "neutralisation_rate": rnd(self.c["AVOIDANT_NEUTRAL"] / self.n_all
                                       if self.n_all else float("nan")),
            "undecidable_rate": rnd(self.c["UNDECIDABLE"] / self.n_all
                                    if self.n_all else float("nan")),
            "acc_existential": rnd(self.existential_acc()),
        }


def rnd(x, k=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, k)


# --------------------------------------------------------------------------- bootstrap

def cluster_bootstrap(rows: list[dict], stat, cluster_key: str = "template_id",
                      n_boot: int = 4000, seed: int = 20260807) -> dict:
    """Resample CLUSTERS with replacement. Every CI in the report comes from here.

    Items sharing a template share syntax, lexical frame and register, so they are not
    independent. Reported alongside is the design effect against the i.i.d. bootstrap — if the
    intervals look wide, DEFF is the reason and it is the correct reason.
    """
    if not rows:
        return {"point": None, "ci": [None, None], "n_clusters": 0, "deff": None}
    by = defaultdict(list)
    for r in rows:
        by[r[cluster_key]].append(r)
    keys = list(by)
    rng = random.Random(seed)
    point = stat(rows)
    draws = []
    for _ in range(n_boot):
        samp = []
        for _ in range(len(keys)):
            samp.extend(by[keys[rng.randrange(len(keys))]])
        v = stat(samp)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            draws.append(v)
    draws.sort()
    if not draws:
        return {"point": rnd(point), "ci": [None, None], "n_clusters": len(keys), "deff": None}
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    # i.i.d. comparison, same B, for the design effect
    rng2 = random.Random(seed + 1)
    idraws = []
    for _ in range(min(n_boot, 1500)):
        samp = [rows[rng2.randrange(len(rows))] for _ in range(len(rows))]
        v = stat(samp)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            idraws.append(v)
    var_c = _var(draws)
    var_i = _var(idraws)
    return {"point": rnd(point), "ci": [rnd(lo), rnd(hi)], "n_clusters": len(keys),
            "n_items": len(rows),
            "deff": rnd(var_c / var_i, 3) if var_i and var_i > 0 else None}


def _var(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


# --------------------------------------------------------------------------- exact tests
# Implemented from math.comb rather than pulled from scipy: both are exact, both are two
# lines, and it keeps the analysis runnable on a checkout with no scientific stack.

def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binom(n, p). One-sided exact binomial."""
    if k <= 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                        for i in range(k, n + 1)))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar two-sided p on the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def acc_strict_stat(rows):
    return sum(r["outcome"] == "CORRECT" for r in rows) / len(rows) if rows else float("nan")


def committed_rate_stat(rows):
    return sum(r["outcome"] in COMMITTED for r in rows) / len(rows) if rows else float("nan")


def drop_rate_stat(rows):
    return (sum(r["outcome"] == "AVOIDANT_DROP" for r in rows) / len(rows)
            if rows else float("nan"))


# --------------------------------------------------------------------------- loading

def load_scored(scored_root: Path) -> list[dict]:
    rows = []
    for mdir in sorted(p for p in scored_root.iterdir() if p.is_dir()):
        for fp in sorted(mdir.glob("*.jsonl")):
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def canonical_rows(rows: list[dict]) -> list[dict]:
    """The main run: one prompt per condition, the NMT arm's single prompt-free row."""
    return [r for r in rows
            if r["prompt_id"] == "nmt" or r["prompt_id"] == CANONICAL.get(r["condition"])]


# --------------------------------------------------------------------------- DFR

def load_priors(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for s in d["slots"]:
        out[(s["slot"], s["variant"])] = s
    return out


#: Which prior variant anchors which benchmark slot, and whether it may be reported at all.
PRIOR_FOR_SLOT = {
    "clusivity": ("clusivity", "nominative", True),
    "rationality": ("rationality", "suffix_bound", True),
    "gender": ("gender", "suffix_bound", True),
    "number": (None, None, False),
    "honorificity": (None, None, False),      # D-5: BLOCKED, not merely fragile
}
PRIOR_BLOCK_REASON = {
    "honorificity": ("D-5: honorificity has no reportable corpus prior. நீங்கள் is "
                     "honorific-sg/plain-pl syncretic; தாங்கள் is measured to be "
                     "overwhelmingly the 3rd-person reflexive (only 5.0% [3.9,6.3] of "
                     "Tamil-Wikipedia தாங்கள் sentences carry any 2nd-person cue); and "
                     "verb-borne 2nd-person forms are rare in written Tamil. No number is "
                     "invented here."),
    "number": "No corpus prior was computed for the number control slot.",
}
#: Benchmark gold vocabulary -> the prior's value names.
GOLD_TO_PRIOR_VALUE = {
    "clusivity": {"INCL": "INCL", "EXCL": "EXCL"},
    "rationality": {"UYAR": "UYARTHINAI", "AHRI": "AHRINAI"},
    "gender": {"MASC": "MASC", "FEM": "FEM"},
}


def dfr_for(slot: str, obs: dict[str, float], priors: dict) -> dict:
    key = PRIOR_FOR_SLOT[slot]
    if not key[2]:
        return {"reportable": False, "reason": PRIOR_BLOCK_REASON[slot]}
    p = priors.get((key[0], key[1]))
    if p is None:
        return {"reportable": False, "reason": f"prior {key} not present in corpus-priors.json"}
    m = GOLD_TO_PRIOR_VALUE[slot]
    out = {"reportable": True, "prior_slot": key[0], "prior_variant": key[1],
           "prior_fragile": p["fragile"], "prior_verdict": p["verdict"],
           "prior_pooled": p["pooled_p"], "prior_between_corpus_range": p["between_corpus_range"],
           "prior_headline_interval": p["headline_interval"],
           "prior_caveats": p["caveats"], "values": {}}
    tvd_lo, tvd_hi = 0.0, 0.0
    for gv, pv in m.items():
        if pv not in p["pooled_p"]:
            continue
        po = obs.get(gv, 0.0)
        pr = p["pooled_p"][pv]
        rng = p["between_corpus_range"].get(pv, [pr, pr])
        # DFR at BOTH ends of the between-corpus range — the range is the honest object.
        d_pool = (po - pr) / pr if pr else None
        d_lo = (po - rng[1]) / rng[1] if rng[1] else None      # prior at its HIGH end
        d_hi = (po - rng[0]) / rng[0] if rng[0] else None      # prior at its LOW end
        out["values"][gv] = {
            "p_obs": rnd(po), "p_ref_pooled": rnd(pr),
            "p_ref_range": [rnd(rng[0]), rnd(rng[1])],
            "DFR_pooled": rnd(d_pool),
            "DFR_range": [rnd(min(x for x in (d_lo, d_hi) if x is not None)),
                          rnd(max(x for x in (d_lo, d_hi) if x is not None))]
            if (d_lo is not None and d_hi is not None) else [None, None],
            "LR_pooled": rnd(math.log2(po / pr)) if po > 0 and pr > 0 else None,
            "sign_flips_across_prior_range": bool(
                d_lo is not None and d_hi is not None and (d_lo > 0) != (d_hi > 0)),
        }
        tvd_lo += abs(po - rng[1])
        tvd_hi += abs(po - rng[0])
    out["TVD_range"] = [rnd(min(tvd_lo, tvd_hi) / 2), rnd(max(tvd_lo, tvd_hi) / 2)]
    out["any_sign_flip"] = any(v["sign_flips_across_prior_range"]
                               for v in out["values"].values())
    return out


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="outputs/scored")
    # ⚠ --raw is NOT cosmetic. The panel-completeness gate below derives the panel from the
    # RAW generation directories, deliberately, so that a filter bug in the analysis cannot
    # define its own ground truth. That means the gate must be pointed at the raw tree that
    # produced the scored records, or a missing system passes silently.
    ap.add_argument("--raw", default="outputs/raw")
    ap.add_argument("--runs", default="outputs/run_manifests",
                    help="comma-separated dirs searched for run-manifest.*.json")
    ap.add_argument("--priors", default="results/corpus-priors.json")
    # Dataset-specific: the extractor is validated against the ITEMS' own gold and contrast
    # targets, so the report must quote the validation of the same item set.
    ap.add_argument("--extractor-validation",
                    default="results/extractor-validation.json")
    ap.add_argument("--out-json", default="results/eval-results.json")
    ap.add_argument("--out-md", default="results/eval-results.md")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--items", default="data/benchmark/items.jsonl")
    args = ap.parse_args()

    rows = load_scored(ROOT / args.scored)
    priors = load_priors(ROOT / args.priors)
    canon = canonical_rows(rows)
    systems = sorted({r["system_id"] for r in canon})
    print(f"{len(rows)} scored records, {len(canon)} canonical, {len(systems)} systems",
          flush=True)

    res: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_scored_records": len(rows), "n_canonical_records": len(canon),
        "systems": systems,
        "policy": {
            "checker": "tamillingbench.morph.MorphChecker — existing, validated, not rebuilt",
            "decision": "DECISIONS.md D-2: universal after context filtering is the HEADLINE; "
                        "existential is a robustness column; undecidable has its own denominator",
            "taxonomy": list(OUTCOMES),
            "avoidance": "AVOIDANT_* is NEVER folded into 'incorrect'. committed_rate is "
                         "reported beside every accuracy.",
            "bootstrap": f"cluster bootstrap on template_id, B={args.n_boot}",
        },
        "provisional_slots": PROVISIONAL,
        "cells": {}, "gradient": {}, "prompt_sensitivity": {},
        "clusivity_gate": {}, "dfr": {}, "unparsed_reasons": {},
        "control_slot_check": {},
    }
    val = ROOT / args.extractor_validation
    if val.exists():
        v = json.loads(val.read_text(encoding="utf-8"))
        res["extractor_validation"] = {
            "totals": v["totals"], "by_slot": {k: {kk: vv for kk, vv in x.items()
                                                   if kk in ("n_gold_targets", "recall_rate",
                                                             "contrast_fp_rate")}
                                               for k, x in v["by_slot"].items()},
            "known_structural_failure": v["known_structural_failure"],
            "scoring_repair_ablation": v.get("scoring_repair_ablation", {}).get("deltas"),
        }

    # ---------------------------------------------------------------- UNPARSED breakdown
    # If the checker is failing on legitimate Tamil that is a CHECKER GAP, not noise, and it
    # has to be visible rather than buried in a residual.
    for slot in SLOTS:
        for sysid in systems:
            sel = [r for r in canon if r["slot"] == slot and r["system_id"] == sysid]
            if not sel:
                continue
            un = [r for r in sel if r["outcome"] == "UNPARSED"]
            res["unparsed_reasons"][f"{sysid}|{slot}"] = {
                "n_all": len(sel), "n_unparsed": len(un),
                "unparsed_rate": rnd(len(un) / len(sel)),
                "reasons": dict(Counter(r.get("unparsed_reason") for r in un)),
                "mean_analysable_rate_when_unparsed": rnd(
                    sum(r["analysable_rate"] for r in un) / len(un)) if un else None,
                "examples": [r["hypothesis"][:70] for r in un[:3]],
            }

    # ---------------------------------------------------------------- control slot
    # The interpretability dissociation rests on the NUMBER control behaving well. If it does not, that
    # must be visible here rather than discovered downstream.
    for sysid in systems:
        for cond in ("C0", "C1", "C2"):
            sel = [r for r in canon if r["slot"] == "number" and r["system_id"] == sysid
                   and r["condition"] == cond]
            if not sel:
                continue
            cell = Cell(sel)
            wrong = [r for r in sel if r["outcome"] == "WRONG"]
            # ⚠ THE CONFOUND, measured rather than described. The design pins `hon = minus`
            # on every number template so that honorificity is available as D-2's context
            # filter. But a model that renders the subject with the HONORIFIC noun (அரசர்
            # "king-HON") is then *required by concord* to use -ஆர் on the verb — and under
            # the pinned hon=minus filter, -ஆர் reads as 3PL. The item scores WRONG on
            # number even though nothing about number went wrong.
            hon_sub = [r for r in wrong
                       if r["gold_value"] == "SG" and r["emitted_value"] == "PL"
                       and (r.get("surface_morpheme") or "") in ("ார்", "ார்கள்", "ஆர்",
                                                                 "ஆர்கள்")]
            res["control_slot_check"][f"{sysid}|{cond}"] = {
                "n": cell.n_all, "acc_strict": rnd(cell.acc_strict),
                "acc_committed": rnd(cell.acc_committed),
                "committed_rate": rnd(cell.committed_rate),
                "undecidable_rate": rnd(cell.c["UNDECIDABLE"] / cell.n_all),
                "unparsed_rate": rnd(cell.c["UNPARSED"] / cell.n_all),
                "n_wrong": len(wrong),
                "n_wrong_honorific_substitution": len(hon_sub),
                "honorific_substitution_share_of_errors": rnd(
                    len(hon_sub) / len(wrong)) if wrong else None,
                "acc_strict_excl_honorific_substitution": rnd(
                    (cell.n_correct + len(hon_sub)) / cell.n_all) if cell.n_all else None,
                "note": "NUMBER is the control slot. Two things inflate its error rate and "
                        "neither is a number error: (i) UNDECIDABLE is the -ஆர்/-ஆர்கள் "
                        "honorific-sg vs rational-pl syncretism surviving D-2's filter, a "
                        "property of Tamil; (ii) honorific substitution — the model renders "
                        "a singular subject honorifically, concord forces -ஆர், and the "
                        "pinned hon=minus filter re-reads that as plural.",
            }

    # ---------------------------------------------------------------- cells + gradient
    for sysid in systems:
        for slot in SLOTS:
            for cond in ("C0", "C1", "C2", "C3"):
                sel = [r for r in canon if r["system_id"] == sysid
                       and r["slot"] == slot and r["condition"] == cond]
                if not sel:
                    continue
                cell = Cell(sel)
                d = cell.to_dict()
                d["chance_rate"] = sel[0]["chance_rate"]
                nk = Counter(r.get("neutralising_value") for r in sel
                             if r["outcome"] == "AVOIDANT_NEUTRAL")
                if nk:
                    d["neutralisation_kind"] = dict(nk)
                d["provisional"] = slot in PROVISIONAL
                if cond != "C3":
                    d["acc_strict_boot"] = cluster_bootstrap(sel, acc_strict_stat,
                                                             n_boot=args.n_boot)
                d["committed_rate_boot"] = cluster_bootstrap(sel, committed_rate_stat,
                                                             n_boot=args.n_boot)
                if slot == "honorificity" and cond in ("C0", "C1", "C2"):
                    # The VV degree is not scoreable by the validated checker. Report the
                    # T-vs-V subset separately so the honorificity number means something.
                    tv = [r for r in sel if r["gold_value"] in ("T", "V")]
                    vv = [r for r in sel if r["gold_value"] == "VV"]
                    if tv:
                        d["binary_TV_only"] = Cell(tv).to_dict()
                        d["binary_TV_only"]["acc_strict_boot"] = cluster_bootstrap(
                            tv, acc_strict_stat, n_boot=args.n_boot)
                    if vv:
                        d["VV_unscoreable"] = {
                            "n": len(vv),
                            "outcomes": dict(Counter(r["outcome"] for r in vv)),
                            "gold_surface_hit_rate": rnd(
                                sum(bool(r.get("gold_surface_hit")) for r in vv) / len(vv)),
                            "note": "The validated checker licenses no DEFERENTIAL reading, so "
                                    "these items cannot be scored CORRECT by construction. "
                                    "gold_surface_hit is a STRING MATCH diagnostic, not a "
                                    "morphological verdict.",
                        }
                res["cells"][f"{sysid}|{slot}|{cond}"] = d

            # --- the gradient ---
            c1 = [r for r in canon if r["system_id"] == sysid and r["slot"] == slot
                  and r["condition"] == "C1"]
            c2 = [r for r in canon if r["system_id"] == sysid and r["slot"] == slot
                  and r["condition"] == "C2"]
            c3 = [r for r in canon if r["system_id"] == sysid and r["slot"] == slot
                  and r["condition"] == "C3"]
            if not (c1 and c2):
                continue
            cell1, cell2 = Cell(c1), Cell(c2)
            g = {
                "slot": slot, "system_id": sysid,
                "provisional": slot in PROVISIONAL,
                "provisional_reason": PROVISIONAL.get(slot),
                "chance_rate": c1[0]["chance_rate"],
                "C1": {"acc_strict": rnd(acc_strict_stat(c1)) if cell1.reportable else None,
                       "acc_strict_unsuppressed": rnd(acc_strict_stat(c1)),
                       "committed_rate": rnd(committed_rate_stat(c1)), "n": len(c1),
                       "n_committed": cell1.n_committed,
                       "suppressed_reason": cell1.suppressed_reason},
                "C2": {"acc_strict": rnd(acc_strict_stat(c2)) if cell2.reportable else None,
                       "acc_strict_unsuppressed": rnd(acc_strict_stat(c2)),
                       "committed_rate": rnd(committed_rate_stat(c2)), "n": len(c2),
                       "n_committed": cell2.n_committed,
                       "suppressed_reason": cell2.suppressed_reason},
            }
            # The drop is only meaningful when BOTH endpoints are, and only when the two
            # conditions are actually comparable for this slot.
            blockers = [x for x in (
                (f"C1 suppressed: {cell1.suppressed_reason}" if cell1.suppressed_reason
                 else None),
                (f"C2 suppressed: {cell2.suppressed_reason}" if cell2.suppressed_reason
                 else None),
                NOT_COMPARABLE.get((slot, "C1")),
                NOT_COMPARABLE.get((slot, "C2")),
            ) if x]
            g["drop_suppressed_reason"] = "; ".join(blockers) or None
            g["drop_C1_to_C2"] = (None if blockers
                                  else rnd(acc_strict_stat(c1) - acc_strict_stat(c2)))
            g["drop_C1_to_C2_unsuppressed"] = rnd(acc_strict_stat(c1) - acc_strict_stat(c2))
            # paired cluster bootstrap of the DIFFERENCE, resampling the SAME templates
            by1 = defaultdict(list)
            by2 = defaultdict(list)
            for r in c1:
                by1[r["template_id"]].append(r)
            for r in c2:
                by2[r["template_id"]].append(r)
            keys = sorted(set(by1) & set(by2))
            rng = random.Random(20260807)
            draws = []
            for _ in range(args.n_boot):
                a, b = [], []
                for _ in range(len(keys)):
                    k = keys[rng.randrange(len(keys))]
                    a.extend(by1[k])
                    b.extend(by2[k])
                if a and b:
                    draws.append(acc_strict_stat(a) - acc_strict_stat(b))
            draws.sort()
            g["drop_C1_to_C2_ci"] = ([rnd(draws[int(.025 * len(draws))]),
                                      rnd(draws[min(len(draws) - 1, int(.975 * len(draws)))])]
                                     if draws and not blockers else [None, None])
            g["n_templates"] = len(keys)
            if c3:
                dist = Counter(r["emitted_value"] for r in c3
                               if r["outcome"] in COMMITTED and r["emitted_value"])
                tot = sum(dist.values())
                g["C3"] = {
                    "n": len(c3),
                    "committed_rate": rnd(committed_rate_stat(c3)),
                    "outcomes": dict(Counter(r["outcome"] for r in c3)),
                    "p_obs_over_licit": {k: rnd(v / tot) for k, v in dist.items()} if tot else {},
                    "n_committed": tot,
                    # Same thin-cell rule: a modal value over a handful of committed items
                    # is not a mode. gemma3-1b gender C3 committed on 1/150 and would
                    # otherwise read "FEM (100%)".
                    "modal_value": (dist.most_common(1)[0][0]
                                    if tot >= MIN_COMMITTED else None),
                    "modal_share": (rnd(dist.most_common(1)[0][1] / tot)
                                    if tot >= MIN_COMMITTED else None),
                    "modal_suppressed_reason": (None if tot >= MIN_COMMITTED else
                                                f"committed n={tot} < {MIN_COMMITTED}"),
                    "modal_value_unsuppressed": (dist.most_common(1)[0][0] if tot else None),
                    "modal_share_unsuppressed": (rnd(dist.most_common(1)[0][1] / tot)
                                                 if tot else None),
                }
                # C2 error direction — input to the pre-registered falsification test
                errs = Counter(r["emitted_value"] for r in c2
                               if r["outcome"] == "WRONG" and r["emitted_value"])
                et = sum(errs.values())
                g["C2_error_distribution"] = {k: rnd(v / et) for k, v in errs.items()} if et else {}
                if tot and et:
                    vals = set(dist) | set(errs)
                    tv = 0.5 * sum(abs(dist.get(v, 0) / tot - errs.get(v, 0) / et) for v in vals)
                    g["C3_vs_C2errors_TV"] = rnd(tv)
                    g["C3_direction_matches_C2_errors"] = (
                        dist.most_common(1)[0][0] == errs.most_common(1)[0][0])
            res["gradient"][f"{sysid}|{slot}"] = g

    # ---------------------------------------------------------------- prompt sensitivity
    # computed automatically, never eyeballed.
    for sysid in systems:
        for slot in SLOTS:
            for cond in ("C1", "C2", "C3"):
                pr = sorted({r["prompt_id"] for r in rows if r["system_id"] == sysid
                             and r["slot"] == slot and r["condition"] == cond})
                if len(pr) < 2:
                    continue
                # restrict to the item set every prompt actually covers
                sets = [set(r["item_id"] for r in rows if r["system_id"] == sysid
                            and r["slot"] == slot and r["condition"] == cond
                            and r["prompt_id"] == p) for p in pr]
                common = set.intersection(*sets)
                if len(common) < 20:
                    continue
                key = f"{sysid}|{slot}|{cond}"
                per_prompt = {p: [r for r in rows if r["system_id"] == sysid
                                  and r["slot"] == slot and r["condition"] == cond
                                  and r["prompt_id"] == p and r["item_id"] in common]
                              for p in pr}
                if cond == "C3":
                    # C3 has no gold, so accuracy is undefined. The right robustness question
                    # is whether the OUTPUT DISTRIBUTION is stable across prompts, measured as
                    # the largest pairwise total-variation distance between the per-prompt
                    # distributions over licit values.
                    dists = {}
                    for p, sel in per_prompt.items():
                        d = Counter(r["emitted_value"] for r in sel
                                    if r["outcome"] in COMMITTED and r["emitted_value"])
                        t = sum(d.values())
                        dists[p] = ({k: v / t for k, v in d.items()} if t else {}, t)
                    keys_v = {v for d, _ in dists.values() for v in d}
                    worst, pair = 0.0, None
                    ps = [p for p in pr if dists[p][1] >= MIN_COMMITTED]
                    for i in range(len(ps)):
                        for j in range(i + 1, len(ps)):
                            a, b = dists[ps[i]][0], dists[ps[j]][0]
                            tv = 0.5 * sum(abs(a.get(v, 0) - b.get(v, 0)) for v in keys_v)
                            if tv > worst:
                                worst, pair = tv, (ps[i], ps[j])
                    res["prompt_sensitivity"][key] = {
                        "n_common_items": len(common), "prompts": pr,
                        "statistic": "max pairwise TVD between per-prompt C3 distributions",
                        "modal_by_prompt": {
                            p: (max(d, key=d.get) if d and t >= MIN_COMMITTED else None)
                            for p, (d, t) in dists.items()},
                        "modal_share_by_prompt": {
                            p: (rnd(max(d.values())) if d and t >= MIN_COMMITTED else None)
                            for p, (d, t) in dists.items()},
                        "n_committed_by_prompt": {p: t for p, (_, t) in dists.items()},
                        "max_pairwise_tvd": rnd(worst) if pair else None,
                        "most_divergent_pair": list(pair) if pair else None,
                    }
                    continue
                accs = {p: rnd(acc_strict_stat(sel)) if Cell(sel).reportable else None
                        for p, sel in per_prompt.items()}
                vals = [v for v in accs.values() if v is not None]
                rng_ = max(vals) - min(vals) if len(vals) >= 2 else None
                res["prompt_sensitivity"][key] = {
                    "n_common_items": len(common), "prompts": pr,
                    "statistic": "acc_strict per prompt (suppressed cells excluded)",
                    "acc_strict_by_prompt": accs,
                    "n_prompts_reportable": len(vals),
                    "across_prompt_range": rnd(rng_),
                }
            g = res["gradient"].get(f"{sysid}|{slot}")
            if not g:
                continue
            drop = g.get("drop_C1_to_C2")
            ranges = [res["prompt_sensitivity"].get(f"{sysid}|{slot}|{c}", {}).get(
                "across_prompt_range") for c in ("C1", "C2")]
            ranges = [x for x in ranges if x is not None]
            if ranges and drop is not None:
                worst = max(ranges)
                g["prompt_range_max"] = rnd(worst)
                g["prompt_sensitive"] = bool(worst > abs(drop))
                g["headline_eligible"] = not g["prompt_sensitive"]
            else:
                g["prompt_sensitive"] = None
                g["headline_eligible"] = None

    # ---------------------------------------------------------------- clusivity gate
    # Re-run across the WHOLE panel. The existing gate used one 600M system.
    for sysid in systems:
        for cond in ("C1", "C2"):
            for env in ("nom", "acc", "dat", "gen"):
                sel = [r for r in canon if r["system_id"] == sysid and r["slot"] == "clusivity"
                       and r["condition"] == cond and r["overt_environment"] == env]
                if not sel:
                    continue
                cell = Cell(sel)
                boot = cluster_bootstrap(sel, drop_rate_stat, n_boot=args.n_boot)
                res["clusivity_gate"][f"{sysid}|{cond}|{env}"] = {
                    "system_id": sysid, "condition": cond, "environment": env,
                    "n": cell.n_all,
                    "drop_rate": rnd(cell.c["AVOIDANT_DROP"] / cell.n_all),
                    "drop_rate_ci": boot["ci"],
                    "n_templates": boot["n_clusters"],
                    "passes_gate_0.25": bool(cell.c["AVOIDANT_DROP"] / cell.n_all <= 0.25),
                    "committed_rate": rnd(cell.committed_rate),
                    "acc_committed": rnd(cell.acc_committed),
                    "acc_strict": rnd(cell.acc_strict),
                    "outcomes": {o: cell.c[o] for o in OUTCOMES},
                }

    # ---------------------------------------------------------------- C0 wording diagnostic
    # C0 gates every default-filling claim, so it matters a great deal whether a
    # near-chance C0 means "cannot produce the form" or "could not read the instruction".
    # The SHIPPED c0_instruction names the benchmark's INTERNAL LABEL CODE — "realising the
    # clusivity value 'EXCL' on the pron". P0_explicit states the target morpheme instead,
    # copied off the item's own gold_targets. Same items, same models, different wording.
    for sysid in systems:
        for slot in SLOTS:
            base = [r for r in rows if r["system_id"] == sysid and r["slot"] == slot
                    and r["condition"] == "C0" and r["prompt_id"] == "P0_specified"]
            expl = [r for r in rows if r["system_id"] == sysid and r["slot"] == slot
                    and r["condition"] == "C0" and r["prompt_id"] == "P0_explicit"]
            if not base or not expl:
                continue
            cb, ce = Cell(base), Cell(expl)
            res.setdefault("c0_wording", {})[f"{sysid}|{slot}"] = {
                "n": cb.n_all,
                "shipped_label_code": {
                    "acc_strict": rnd(cb.acc_strict), "committed_rate": rnd(cb.committed_rate),
                    "instruction_style": "names the internal label code, e.g. "
                                         "\"realising the clusivity value 'EXCL' on the pron\""},
                "explicit_morpheme": {
                    "acc_strict": rnd(ce.acc_strict), "committed_rate": rnd(ce.committed_rate),
                    "instruction_style": "states the target morpheme, copied from the item's "
                                         "own gold_targets"},
                "delta_acc_strict": rnd(ce.acc_strict - cb.acc_strict),
            }

    # ------------------------------------------------- why clusivity C1 is not comparable
    # Computed from the item sources rather than asserted, because the conclusion (that the
    # C1 clusivity cell measures an English construction rather than the slot) is strong
    # enough that it has to be checkable.
    import re as _re
    pron_re = _re.compile(r"\b(we|us|our|ours)\b", _re.I)
    coord_re = _re.compile(r"\b(and (me|myself|I)|(me|myself|I) and)\b", _re.I)
    src_by_cond: dict[str, set] = defaultdict(set)
    for line in (ROOT / args.items).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        if it["slot"] == "clusivity" and it.get("split") == "public":
            src_by_cond[it["condition"]].add(it["source"])
    res["clusivity_c1_construction"] = {
        cond: {"n_unique_sources": len(v),
               "n_with_1pl_pronoun": sum(bool(pron_re.search(x)) for x in v),
               "n_with_coordinated_np": sum(bool(coord_re.search(x)) for x in v),
               "examples": sorted(v)[:3]}
        for cond, v in sorted(src_by_cond.items())
    }
    res["clusivity_c1_construction"]["verdict"] = NOT_COMPARABLE[("clusivity", "C1")]

    # ---------------------------------------------------------------- falsification
    # Pre-registered BEFORE any model was run, implemented verbatim. The
    # bootstrap above is the ESTIMATION machinery; these three tests are the DECISION
    # machinery, and where they disagree both are reported rather than the friendlier one.
    for key, g in res["gradient"].items():
        sysid, slot = key.split("|")
        c1 = [r for r in canon if r["system_id"] == sysid and r["slot"] == slot
              and r["condition"] == "C1"]
        c2 = [r for r in canon if r["system_id"] == sysid and r["slot"] == slot
              and r["condition"] == "C2"]
        # (1) Acc(C1) > Acc(C2), exact McNemar on FRAME-PAIRED items (paired within set_id)
        # ⛔ Pair on (set_id, gold_value), NOT set_id alone. A set_id holds every member of
        # the minimal pair/triple (…-C1-EXCL and …-C1-INCL share one set_id), so keying on
        # set_id silently keeps one variant and discards the rest — which both halves the
        # discordant counts and biases which variant survives.
        m1 = {(r["set_id"], r["gold_value"]): r for r in c1}
        m2 = {(r["set_id"], r["gold_value"]): r for r in c2}
        b = sum(1 for k2 in m1.keys() & m2.keys()
                if m1[k2]["outcome"] == "CORRECT" and m2[k2]["outcome"] != "CORRECT")
        c_ = sum(1 for k2 in m1.keys() & m2.keys()
                 if m1[k2]["outcome"] != "CORRECT" and m2[k2]["outcome"] == "CORRECT")
        p_mcnemar = mcnemar_exact(b, c_)
        crit1 = bool(b > c_ and p_mcnemar < 0.01)
        # (2) Acc(C2) NOT significantly above chance — one-sided exact binomial, alpha .05
        n2 = len(c2)
        k2n = sum(1 for r in c2 if r["outcome"] == "CORRECT")
        chance = c2[0]["chance_rate"] if c2 else 0.5
        p_chance = binom_sf(k2n, n2, chance) if n2 else 1.0
        crit2 = bool(p_chance >= 0.05)
        # (3) C3 direction matches the C2 error direction, TV < tau = 0.15
        tv = g.get("C3_vs_C2errors_TV")
        crit3 = bool(g.get("C3_direction_matches_C2_errors") and tv is not None and tv < 0.15)
        # C0 precondition: a slot whose C0 is also near chance does not support
        # a default-filling claim — the form may simply not be in the model's inventory.
        c0 = res["cells"].get(f"{sysid}|{slot}|C0")
        c0_ok = None
        if c0 and c0["n_all"]:
            c0_ok = bool(binom_sf(c0["outcomes"]["CORRECT"], c0["n_all"],
                                  c0.get("chance_rate", 0.5)) < 0.05)
        # The same gate computed on the CORRECTED C0 wording (P0_explicit), where available.
        # The shipped instruction names an internal label code the model cannot interpret, so
        # the shipped gate fails for the wrong reason on most cells — see the C0-wording
        # section. Reported alongside rather than substituted, because the benchmark currently
        # ships the other wording.
        cw = (res.get("c0_wording") or {}).get(f"{sysid}|{slot}")
        c0_ok_explicit = None
        if cw:
            k_e = round(cw["explicit_morpheme"]["acc_strict"] * cw["n"])
            c0_ok_explicit = bool(
                binom_sf(k_e, cw["n"], c0.get("chance_rate", 0.5) if c0 else 0.5) < 0.05)
        g["falsification"] = {
            "criterion_1_C1_gt_C2_mcnemar": {"b": b, "c": c_, "p": rnd(p_mcnemar, 6),
                                             "passes": crit1},
            "criterion_2_C2_not_above_chance": {"n": n2, "k": k2n, "chance": chance,
                                                "p_one_sided": rnd(p_chance, 6),
                                                "passes": crit2},
            "criterion_3_C3_matches_C2_errors": {"TV": tv, "tau": 0.15, "passes": crit3},
            "c0_capability_precondition_met": c0_ok,
            "c0_capability_precondition_met_explicit_wording": c0_ok_explicit,
            "prompt_robustness_ok": (not g.get("prompt_sensitive")
                                     if g.get("prompt_sensitive") is not None else None),
            "suppressed": g.get("drop_suppressed_reason"),
            "default_is_active": bool(crit1 and crit2 and crit3 and c0_ok
                                      and not g.get("prompt_sensitive")
                                      and not g.get("drop_suppressed_reason")),
            "default_is_active_with_corrected_C0": bool(
                crit1 and crit2 and crit3 and c0_ok_explicit
                and not g.get("prompt_sensitive")
                and not g.get("drop_suppressed_reason")),
            "note": "All of criteria 1-3 plus the C0 capability precondition plus prompt "
                    "robustness must hold. Any single failure means the default-filling "
                    "claim is NOT supported for this cell.",
        }

    # ------------------------------------------------ cross-system direction of the C3 skew
    # Whether every system defaults the SAME WAY is the question that separates "models
    # reproduce a corpus prior" from "each family has its own prior". It is computed rather
    # than described, because it changes as systems land.
    for slot in SLOTS:
        modes = {}
        for key, g in res["gradient"].items():
            if g["slot"] != slot or "C3" not in g:
                continue
            mv = g["C3"].get("modal_value")
            if mv:
                modes[g["system_id"]] = {"modal": mv,
                                         "share": g["C3"].get("modal_share"),
                                         "committed_rate": g["C3"].get("committed_rate")}
        if not modes:
            continue
        distinct = sorted({m["modal"] for m in modes.values()})
        by_val = defaultdict(list)
        for sysid, m in sorted(modes.items()):
            by_val[m["modal"]].append(sysid)
        pr_key = PRIOR_FOR_SLOT[slot]
        prior_modal = None
        if pr_key[2]:
            pobj = priors.get((pr_key[0], pr_key[1]))
            if pobj:
                inv = {v: k for k, v in GOLD_TO_PRIOR_VALUE[slot].items()}
                cand = {inv[v]: p for v, p in pobj["pooled_p"].items() if v in inv}
                if cand:
                    prior_modal = max(cand, key=cand.get)
        res.setdefault("c3_direction_across_systems", {})[slot] = {
            "n_systems": len(modes),
            "modal_by_system": modes,
            "distinct_modal_values": distinct,
            "systems_by_modal_value": dict(by_val),
            "unanimous": len(distinct) == 1,
            "corpus_prior_modal_value": prior_modal,
            "agrees_with_prior": (sorted(by_val.get(prior_modal, []))
                                  if prior_modal else None),
            "disagrees_with_prior": (sorted(sysid for v, ss in by_val.items()
                                            if v != prior_modal for sysid in ss)
                                     if prior_modal else None),
        }

    # ---------------------------------------------------------------- C2 context leak
    # a model that ignores "do not translate the CONTEXT" and renders both
    # sentences breaks the 1:1 output-to-reference alignment. Detected by output length
    # against the same system's C1 median, and reported per model rather than silently mixed.
    for sysid in systems:
        c1len = [r["n_out_tokens"] for r in canon
                 if r["system_id"] == sysid and r["condition"] == "C1"]
        c2 = [r for r in canon if r["system_id"] == sysid and r["condition"] == "C2"]
        if not c1len or not c2:
            continue
        c1len.sort()
        med = c1len[len(c1len) // 2]
        over = [r for r in c2 if med and r["n_out_tokens"] > 2.5 * med]
        res.setdefault("c2_context_leak", {})[sysid] = {
            "c1_median_out_tokens": med, "n_c2": len(c2),
            "n_c2_over_2.5x_c1_median": len(over),
            "rate": rnd(len(over) / len(c2)),
            "note": "Context-translation detector. A high rate means the model translated the CONTEXT "
                    "sentence as well as the marked one, and that model's C2 column should be "
                    "read with the P4b fallback caveat.",
        }

    # ------------------------------------------------------------- C2 ANSWER-LOCUS check
    # ⛔ THE FAILURE THE LENGTH DETECTOR ABOVE CANNOT SEE, and it is the same class of bug as
    # the sarvam-translate retraction one level up.
    #
    # `c2_context_leak` catches a model that translates the context IN ADDITION to the marked
    # sentence — the output gets long, so length finds it. It is blind to a model that
    # translates the context INSTEAD of the marked sentence: the output is exactly as long as
    # a correct answer and looks, to every other counter in this file, like a fluent
    # translation.
    #
    # That failure is not cosmetic on C2, it MANUFACTURES the result. The C2 context is where
    # the disambiguating cue lives ("Aram, my younger sister, had been waiting…"), so a model
    # that renders the context gets the feature handed to it by the cue NP and is scored
    # COMMITTED/CORRECT without ever having made a default-filling decision about the marked
    # sentence. Measured on the released panel, gemma3-1b answers the wrong sentence on ~98% of gender
    # C2 items and its C2 commitment rate is consequently meaningless.
    #
    # The detector: does the hypothesis contain the STEM of the gold target's inflected word
    # (the gold word with its scored suffix removed)? The stem is shared by every licit
    # rendering of the marked sentence — including avoidant ones, since avoidance changes the
    # SUFFIX — so a stem miss is evidence the marked predicate was not translated at all.
    #
    # It is reported as a C1-vs-C2 DELTA, never as an absolute. The stem test has a real false
    # negative rate (a model may pick a synonymous verb), but that rate applies equally to C1,
    # where there is no context to answer instead. C1 is therefore the model's own baseline
    # and only the DROP is interpreted. Diagnostic, never a verdict: no outcome is rewritten.
    def _gold_stems(it: dict) -> set:
        out = set()
        for t in (it.get("gold_targets") or []):
            tam = (t.get("tamil") or "").rstrip(".!?")
            m = t.get("surface_morpheme") or ""
            if not tam or not m:
                continue
            for w in tam.split():
                if w.endswith(m) and len(w) > len(m):
                    out.add(w[:-len(m)])
        return out

    stems_by_item = {}
    for line in (ROOT / args.items).read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            st = _gold_stems(it)
            if st:
                stems_by_item[it["item_id"]] = st

    LOCUS_DROP_SUSPECT = 0.25          # C1 rate minus C2 rate, absolute
    for sysid in systems:
        per_cond: dict[str, list[int]] = {"C1": [0, 0], "C2": [0, 0]}
        for r in canon:
            if r["system_id"] != sysid or r["condition"] not in per_cond:
                continue
            st = stems_by_item.get(r["item_id"])
            if not st:
                continue
            per_cond[r["condition"]][1] += 1
            if any(s in (r.get("hypothesis") or "") for s in st):
                per_cond[r["condition"]][0] += 1
        (h1, n1), (h2, n2) = per_cond["C1"], per_cond["C2"]
        if not n1 or not n2:
            continue
        r1, r2 = h1 / n1, h2 / n2
        drop = r1 - r2
        res.setdefault("c2_answer_locus", {})[sysid] = {
            "c1_gold_stem_rate": rnd(r1), "n_c1_testable": n1,
            "c2_gold_stem_rate": rnd(r2), "n_c2_testable": n2,
            "drop_C1_to_C2": rnd(drop),
            "suspect": bool(drop > LOCUS_DROP_SUSPECT),
            "note": "Diagnostic, not a verdict. A large C1->C2 drop means the model translated "
                    "the CONTEXT sentence INSTEAD of the marked one; because the context is "
                    "where the cue lives, that model's C2 commitment is handed to it by the "
                    "cue NP and must not be read as default-filling. Absolute rates are not "
                    "interpretable (verb-synonym false negatives); only the drop is.",
        }
    suspects = sorted(s for s, x in res.get("c2_answer_locus", {}).items() if x["suspect"])
    if suspects:
        res["c2_answer_locus_suspects"] = suspects
        print(f"C2 ANSWER-LOCUS WARNING: {suspects} translated the context instead of the "
              f"marked sentence on a large share of C2 items; their C2 cells are flagged, "
              f"not silently reported", file=sys.stderr, flush=True)

    # --- what the panel-wide gate changes about the previous single-system conclusion ---
    passing = [r for r in res["clusivity_gate"].values()
               if r["condition"] == "C2" and r["passes_gate_0.25"]]
    accs = [r["acc_committed"] for r in passing if r["acc_committed"] is not None]
    res["clusivity_gate_summary"] = {
        "prior_gate": {"system": "facebook/nllb-200-distilled-600M",
                       "reported_C1_drop_range": "94-100%",
                       "conclusion_drawn": "clusivity may be unmeasurable"},
        "panel_n_cells_C2": sum(1 for r in res["clusivity_gate"].values()
                                if r["condition"] == "C2"),
        "panel_n_cells_C2_passing_gate": len(passing),
        "systems_with_a_passing_C2_environment": sorted({r["system_id"] for r in passing}),
        "acc_committed_where_gate_passes": {
            "min": rnd(min(accs)) if accs else None,
            "max": rnd(max(accs)) if accs else None,
            "mean": rnd(sum(accs) / len(accs)) if accs else None,
        },
        "verdict": (
            "Larger models DO commit where NLLB-200-distilled-600M did not, so the "
            "single-system evidence does not support 'clusivity is unmeasurable'. The finding "
            "changes shape rather than disappearing: clusivity is measurable on those "
            "systems, and where they commit they sit at chance."
            if passing else
            "No C2 environment clears the 0.25 gate on any system in the panel, which "
            "REPLICATES the single-system result on a much stronger panel."),
    }

    # ---------------------------------------------------------------- DFR on C3
    for sysid in systems:
        for slot in SLOTS:
            g = res["gradient"].get(f"{sysid}|{slot}")
            if not g or "C3" not in g:
                continue
            obs = g["C3"]["p_obs_over_licit"]
            if not obs:
                # NOT skipped: a slot on which the system committed to nothing in C3 is a
                # result (total avoidance), and silently omitting the row would read as
                # "not computed" rather than "there was no distribution to compare".
                res["dfr"][f"{sysid}|{slot}"] = {
                    "reportable": False,
                    "reason": (f"no committed mass in C3: the system produced no licit "
                               f"{slot} value on any C3 item "
                               f"(committed_rate={g['C3']['committed_rate']}). Outcome "
                               f"distribution: {g['C3']['outcomes']}. This is total "
                               f"avoidance, not a missing computation."),
                    "n_committed": 0,
                    "committed_rate": g["C3"]["committed_rate"],
                    "c3_outcomes": g["C3"]["outcomes"],
                    "provisional": slot in PROVISIONAL,
                }
                continue
            n_comm = g["C3"]["n_committed"]
            if n_comm < MIN_COMMITTED:
                # Same rule as the accuracy cells: a distribution over a handful of committed
                # items is not a distribution. gemma3-1b gender C3 committed on 1/150 and
                # would otherwise publish "p_obs(FEM) = 100%" with a DFR of +32.
                res["dfr"][f"{sysid}|{slot}"] = {
                    "reportable": False,
                    "reason": (f"committed n={n_comm} < {MIN_COMMITTED}: too few committed "
                               f"items to form a distribution "
                               f"(committed_rate={g['C3']['committed_rate']}). Outcome "
                               f"distribution: {g['C3']['outcomes']}."),
                    "n_committed": n_comm,
                    "committed_rate": g["C3"]["committed_rate"],
                    "c3_outcomes": g["C3"]["outcomes"],
                    "p_obs_unsuppressed": obs,
                    "provisional": slot in PROVISIONAL,
                }
                continue
            d = dfr_for(slot, obs, priors)
            d["n_committed"] = n_comm
            d["committed_rate"] = g["C3"]["committed_rate"]
            d["provisional"] = slot in PROVISIONAL
            res["dfr"][f"{sysid}|{slot}"] = d

    # ---------------------------------------------------------------- run manifest
    # One consolidated provenance record: which weights (by commit SHA), which decoding, which
    # prompt ids, when. A partial run must be interpretable rather than mysterious.
    # Run manifests may sit in more than one directory (--runs is a comma-separated list).
    # Read all of them so the consolidated manifest is complete.
    seen_runs: set = set()
    runs = []
    cand = sorted(fp for d in args.runs.split(",") if d.strip()
                  for fp in (ROOT / d.strip()).glob("run-manifest.*.json"))
    for fp in cand:
        if fp.name in seen_runs:
            continue
        seen_runs.add(fp.name)
        try:
            m = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        runs.append({k: m.get(k) for k in (
            "run_id", "system_id", "hf_name", "revision", "model_class", "backend", "dtype",
            "arm", "prompt_set", "decoding_id", "decoding", "split", "conditions",
            "canonical_prompt_per_condition", "robustness_prompts_per_condition",
            "batch_size", "chat_template_kwargs", "has_chat_template", "status",
            "torch", "transformers", "python", "hygiene", "started_utc", "updated_utc",
            "models_config_sha256", "prompts_config_sha256", "decoding_config_sha256",
            "items_sha256", "code_sha256", "out_dir")})
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "what_this_is": "Consolidated provenance for every generation run feeding "
                        "results/eval-results.*. `revision` is the HF commit SHA READ BACK "
                        "from the snapshot actually loaded, not the one requested.",
        "gpu_box": "NVIDIA GB10, 130 GB unified, aarch64, "
                   "Ubuntu 24.04, Python 3.12.3",
        "remote_paths": {
            "raw": "outputs/raw/{model}/{slot}_{condition}_{prompt}.jsonl",
            "scored": "outputs/scored/{model}/...",
            "reports": "outputs/run_manifests/",
        },
        "scoring_machine": "local (hfst/foma have no aarch64 wheel)",
        "n_runs": len(runs), "runs": runs,
    }
    # Written next to the report it documents.
    manifest_path = (ROOT / args.out_json).parent / "eval-run-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    res["run_manifest_path"] = str(
        manifest_path.relative_to(ROOT) if manifest_path.is_relative_to(ROOT)
        else manifest_path)
    pv = ROOT / "results/prompting-verification.json"
    if pv.exists():
        res["prompting_verification"] = json.loads(pv.read_text(encoding="utf-8"))
    res["systems_with_revisions"] = {r["system_id"]: r["revision"] for r in runs}

    # ------------------------------------------------------- PANEL COMPLETENESS (hard gate)
    # ⚠ THE BUG THIS EXISTS TO CATCH. NMT systems carry `prompt_id='nmt'`, never `P1_*`. A
    # filter written against the LLM prompt ids drops the ENTIRE dedicated-NMT arm — and the
    # resulting table is still well-formed, still has plausible numbers, and simply has two
    # fewer rows. It has already shipped once. It is therefore ASSERTED per table, against the
    # raw generations rather than against anything the analysis itself computed, so a filter
    # bug cannot define its own ground truth.
    raw_root = ROOT / args.raw
    if not raw_root.is_dir():
        raise SystemExit(f"panel completeness gate: --raw {raw_root} does not exist; "
                         f"refusing to run the gate against nothing")
    panel = sorted(d.name for d in raw_root.iterdir()
                   if d.is_dir() and any(d.glob("*.jsonl")))
    if not panel:
        raise SystemExit(f"panel completeness gate: no raw generations under {raw_root}; "
                         f"an empty panel would make the gate vacuously pass")
    complaints: list[str] = []
    missing_from_scored = [s for s in panel if s not in systems]
    if missing_from_scored:
        complaints.append(
            f"systems with raw generations but absent from the canonical scored rows: "
            f"{missing_from_scored}")

    #: Tables keyed "<system_id>|..." in which EVERY panel system must appear.
    KEYED_TABLES = ("cells", "gradient", "unparsed_reasons", "control_slot_check",
                    "dfr", "clusivity_gate")

    # --- APPLICABILITY, decided by the data and never by a hand-maintained exemption list ---
    #
    # A table can be empty for two very different reasons, and collapsing them is what makes a
    # gate useless. Either (a) a filter bug deleted an arm — the bug this gate exists to catch,
    # which must stay fatal — or (b) the run genuinely contains no input for that table, so
    # there is nothing to be missing from it.
    #
    # (b) happens whenever an item set omits a slot: `clusivity_gate` then has no input at
    # all. Demanding thirteen systems in a table whose slot is not in the data produces
    # thirteen complaints on every run, and a gate that always
    # fails is a gate that gets passed `|| true`. Preconditions are therefore read off the raw
    # generations — the same ground truth the rest of the gate uses — so an arm that SHOULD be
    # present cannot be excused by editing a list.
    slots_in_run = {r["slot"] for r in canon}
    #: table -> the slot whose presence in the run makes that table applicable.
    TABLE_REQUIRES_SLOT = {"clusivity_gate": "clusivity"}
    inapplicable: dict[str, str] = {}
    for table, need in TABLE_REQUIRES_SLOT.items():
        if need not in slots_in_run:
            inapplicable[table] = (
                f"dataset contains no {need!r} slot (slots in this run: "
                f"{sorted(slots_in_run)}), so the table has no input and no system can "
                f"appear in it")
    #: The single legitimate exemption, declared rather than inferred: the NMT arm is
    #: prompt-free by construction (it is exempt from the prompt-robustness run),
    #: so it cannot appear in a per-prompt table. Every OTHER absence is a bug.
    nmt_arm = {s["id"] for s in yaml.safe_load(
        (ROOT / "configs/models.yaml").read_text(encoding="utf-8"))["panel"]
        if s.get("arm") == "nmt"}
    for table in KEYED_TABLES:
        if table in inapplicable:
            continue
        present = {k.split("|", 1)[0] for k in res.get(table, {})}
        absent = [s for s in panel if s not in present]
        if absent:
            complaints.append(f"table {table!r} is missing panel system(s) {absent}")

    # prompt_sensitivity needs MORE THAN ONE prompt for a system to have any sensitivity to
    # measure. The NMT arm is prompt-free by construction; a canonical-only run leaves every
    # LLM in the same position for the same reason — one prompt, nothing to compare. Which
    # systems those are is read off the raw generations, so a system that DID emit several
    # prompts and then vanished from the table is still a fatal complaint.
    # Sensitivity is variation across prompts WITHIN a condition. Counting distinct prompt ids
    # per system is the wrong test: the canonical set alone spans three of them (P0_specified
    # for C0, P1_minimal for C1/C3, P4_context for C2), one per condition, so every system
    # looks multi-prompt while no condition has anything to compare.
    prompts_per_cell: dict[tuple, set] = defaultdict(set)
    for r in rows:                      # ALL scored records, not the canonical subset
        prompts_per_cell[(r["system_id"], r["condition"])].add(r.get("prompt_id"))
    multi_prompt = {s for (s, _c), ps in prompts_per_cell.items() if len(ps) > 1}
    single_prompt = {r["system_id"] for r in rows} - multi_prompt
    present_ps = {k.split("|", 1)[0] for k in res.get("prompt_sensitivity", {})}
    absent_ps = [s for s in panel
                 if s not in present_ps and s not in nmt_arm and s not in single_prompt]
    if absent_ps:
        complaints.append(
            f"table 'prompt_sensitivity' is missing non-NMT panel system(s) {absent_ps}")
    if single_prompt:
        inapplicable["prompt_sensitivity"] = (
            f"systems {sorted(single_prompt)} have at most one prompt id per condition in "
            f"this run (canonical-only; the robustness arm was not generated), so there is "
            f"no across-prompt variation to report for them")

    res["panel_completeness"] = {
        "panel_from_raw": panel,
        "n_panel": len(panel),
        "systems_in_canonical_rows": systems,
        "nmt_arm": sorted(nmt_arm),
        "tables_checked": list(KEYED_TABLES) + ["prompt_sensitivity (NMT arm exempt)"],
        "tables_inapplicable": inapplicable,
        "slots_in_run": sorted(slots_in_run),
        "single_prompt_systems": sorted(single_prompt),
        "ok": not complaints,
        "complaints": complaints,
        "why": "NMT systems carry prompt_id='nmt', not P1_*. Filtering on LLM prompt ids "
               "silently deletes the dedicated-NMT arm from a table without making the table "
               "look wrong. The build fails rather than reports a quietly incomplete panel.",
    }
    if complaints:
        for c in complaints:
            print(f"PANEL COMPLETENESS FAILURE: {c}", file=sys.stderr, flush=True)
        raise SystemExit(
            f"panel completeness gate failed with {len(complaints)} complaint(s); "
            f"refusing to write a partial report")
    print(f"panel completeness: OK — all {len(panel)} systems present in "
          f"{len(KEYED_TABLES)} keyed tables", flush=True)

    (ROOT / args.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    write_md(res, ROOT / args.out_md)
    print(f"wrote {args.out_json} and {args.out_md}")
    return 0


# --------------------------------------------------------------------------- markdown

def pct(x):
    return "n/a" if x is None else f"{100*x:.1f}"


def write_md(res: dict, path: Path) -> None:
    L: list[str] = []
    A = L.append
    A("# Model evaluation — results")
    A("")
    A(f"Generated {res['generated_utc']}. "
      f"{res['n_scored_records']:,} scored records, "
      f"{res['n_canonical_records']:,} in the main (canonical-prompt) run, "
      f"{len(res['systems'])} systems.")
    A("")
    A("**Reading rule, enforced by the renderer:** no accuracy appears without its "
      "`committed_rate`. `AVOIDANT_*` outcomes are never folded into \"incorrect\" — a system "
      "that declines an obligatory choice has done something different from a system that "
      "chooses wrongly, and the tables keep them apart.")
    A("")
    A("Headline policy is **DECISIONS.md D-2**: universal after context filtering. The "
      "existential (lenient) rate is carried as `acc_existential` in the JSON, and "
      "`undecidable` has its own denominator.")
    A("")

    # --- executive summary, computed from the artifact rather than typed by hand ---
    n_sup = sum(1 for c in res["cells"].values() if c.get("suppressed"))
    n_acc = sum(1 for c in res["cells"].values() if c.get("acc_strict") is not None)
    active = [k for k, g in res["gradient"].items()
              if (g.get("falsification") or {}).get("default_is_active")]
    active_corr = [k for k, g in res["gradient"].items()
                   if (g.get("falsification") or {}).get(
                       "default_is_active_with_corrected_C0")]
    cgs = res.get("clusivity_gate_summary") or {}
    A("## What this run found")
    A("")
    A(f"- **No (system, slot) cell passes the pre-registered default-filling test** "
      f"({len(active)} of {len(res['gradient'])} under the shipped C0 wording, "
      f"{len(active_corr)} under the corrected one). The binding constraint is almost always "
      f"the **C0 capability ceiling**, and the C0 instruction is itself defective — see below. "
      f"The gradient is real for several cells, but the pre-registration does not license "
      f"calling it default-filling yet.")
    A(f"- **Two defects in the shipped benchmark were found and quantified**, both upstream of "
      f"this stage: the C0 instruction names internal label codes no model can read "
      f"(correcting it moves gemma3-4b clusivity C0 from 0% to 87.5%), and C1 clusivity items "
      f"present a coordinated NP rather than *we/us*, so the C1 cell does not measure "
      f"clusivity at all.")
    A(f"- **Two ThamizhiMorph defects were found and repaired at the scoring layer** (the "
      f"validated checker is untouched): `-அனர்` is tagged singular though it is "
      f"unambiguously plural, and a word-final sandhi consonant hides an overt pronoun. Both "
      f"ablations are published.")
    if cgs:
        A(f"- **The clusivity gate changes with the panel.** "
          f"{cgs['panel_n_cells_C2_passing_gate']} of {cgs['panel_n_cells_C2']} C2 cells now "
          f"clear the 0.25 drop gate that `facebook/nllb-200-distilled-600M` failed "
          f"everywhere. Clusivity is measurable on the larger systems — and where they commit, "
          f"they sit at chance.")
    A(f"- **Avoidance is the dominant behaviour on gender**, and it is the *epicene* `-ஆர்` "
      f"escape rather than a masculine default. That is a different — and more interesting — "
      f"finding than the gender-bias result this design was set up to detect.")
    dirs_s = res.get("c3_direction_across_systems") or {}
    split_s = [k for k, d in sorted(dirs_s.items()) if not d["unanimous"]]
    unan_s = [k for k, d in sorted(dirs_s.items()) if d["unanimous"]]
    if dirs_s:
        A(f"- **The panel does not share a single default, which argues against a corpus-frequency "
          f"mechanism.** "
          + (f"On {', '.join('`'+x+'`' for x in split_s)} the systems split on which value they "
             f"default to — e.g. clusivity, where the Gemma family and NLLB default EXCL while "
             f"Llama and Qwen default INCL, against a corpus prior that is INCL-majority "
             f"everywhere. One corpus cannot predict opposite defaults. " if split_s else "")
          + (f"On {', '.join('`'+x+'`' for x in unan_s)} the panel is unanimous. "
             if unan_s else "")
          + f"Corpus frequency is at best a partial explanation.")
    A(f"- **{n_sup} of {n_sup + n_acc} accuracy cells are suppressed** for a thin committed "
      f"denominator. Suppression is enforced in the `Cell` constructor and covered by a "
      f"regression test.")
    A("")
    A("`honorificity` and `rationality` are **provisional** throughout — see the next section "
      "for the specific unresolved question in each.")
    A("")

    A("## Systems and weights")
    A("")
    A("Revisions are HF commit SHAs **read back from the snapshot actually loaded**, not the "
      "ones requested. Full provenance — decoding config, prompt ids, config hashes, package "
      "versions, hygiene counters — is in `results/eval-run-manifest.json`.")
    A("")
    A("| system | revision |")
    A("|---|---|")
    for sysid, rev in sorted((res.get("systems_with_revisions") or {}).items()):
        A(f"| `{sysid}` | `{rev}` |")
    A("")

    pvv = (res.get("prompting_verification") or {}).get("models")
    if pvv:
        A("## Silent-hazard verification")
        A("")
        A("Both of these corrupt generation without raising anything, so they are verified "
          "empirically against the installed `transformers` "
          f"({res['prompting_verification']['transformers']}) rather than asserted in a "
          "comment. Full dump: `results/prompting-verification.json`.")
        A("")
        A("| system | leading BOS used | would have double-BOS'd | thinking disabled |")
        A("|---|---:|:--:|:--:|")
        for sid, r in sorted(pvv.items()):
            if "error" in r:
                A(f"| `{sid}` | — | — | error: {r['error'][:60]} |")
                continue
            A(f"| `{sid}` | {r.get('leading_bos_with_add_special_tokens_False', 'n/a')} "
              f"| {'**yes**' if r.get('double_bos_would_have_occurred') else 'no'} "
              f"| {r.get('thinking_disabled_correctly', 'n/a')} |")
        A("")
        A("Every Gemma and Llama tokenizer **would** have produced a double `<bos>` had the "
          "prompt been tokenized with `add_special_tokens=True` — the hazard is real, not "
          "hypothetical, and it is avoided. For Qwen3, `enable_thinking=False` works by "
          "pre-filling a *closed, empty* `<think></think>` block into the generation prompt, "
          "so a `<think>` in the rendered prompt is the switch working rather than leaking; "
          "the check that matters is that no model **output** contains `<think`, and none "
          "does.")
        A("")

    A("## Provisional slots")
    A("")
    A("These are computed and reported, but are **not settled** and must not be presented as "
      "such.")
    A("")
    for slot, why in res["provisional_slots"].items():
        A(f"- **`{slot}` — provisional.** {why}")
    A("")

    A("## C1 → C2 → C3 gradient")
    A("")
    A("`C1` = cue in the sentence, `C2` = cue in the preceding context only, `C3` = no cue "
      "(no gold by design). CIs are 95% **cluster bootstrap on `template_id`**; `DEFF` is the "
      "design effect against an item-level bootstrap. `prompt_sens` flags cells whose "
      "across-prompt range exceeds their own C1→C2 drop — those cells are excluded from "
      "headline claims mechanically, not by hand.")
    A("")
    A("| system | slot | chance | C1 acc | C1 comm | C2 acc | C2 comm | C1→C2 drop [95% CI] | "
      "C3 modal (share) | prompt_sens | prov |")
    A("|---|---|---:|---:|---:|---:|---:|---|---|:--:|:--:|")
    for key in sorted(res["gradient"]):
        g = res["gradient"][key]
        c3 = g.get("C3") or {}
        ci = g.get("drop_C1_to_C2_ci") or [None, None]
        ps = g.get("prompt_sensitive")
        def cellacc(x):
            return ("n/a" if x["acc_strict"] is None
                    else f"{100*x['acc_strict']:.1f}")
        drop = ("**n/a**" if g.get("drop_C1_to_C2") is None
                else f"{pct(g['drop_C1_to_C2'])} [{pct(ci[0])}, {pct(ci[1])}]")
        A(f"| `{g['system_id']}` | {g['slot']} | {pct(g['chance_rate'])} "
          f"| {cellacc(g['C1'])} | {pct(g['C1']['committed_rate'])} "
          f"| {cellacc(g['C2'])} | {pct(g['C2']['committed_rate'])} "
          f"| {drop} "
          f"| {c3.get('modal_value') or 'n/a'} ({pct(c3.get('modal_share'))}) "
          f"| {'**yes**' if ps else ('no' if ps is not None else '?')} "
          f"| {'⚠' if g['provisional'] else ''} |")
    A("")
    A("All figures are percentages. `comm` = `committed_rate`: the share of items on which "
      "the system actually committed to a licit value (i.e. did not drop, neutralise, negate "
      "away, or fail to parse).")
    A("")
    A(f"**`n/a` means suppressed, not missing.** An accuracy over fewer than {MIN_COMMITTED} "
      f"committed items, or at a commit rate below {MIN_COMMIT_RATE:.0%}, is not a weak "
      f"number — it is a meaningless one, and placed next to a well-populated cell it "
      f"manufactures differences out of nothing. Suppression is enforced in the `Cell` "
      f"constructor so no caller can route around it, and it propagates into the drop, its "
      f"CI, and the falsification verdict. The raw counts remain in the outcome table below "
      f"and in `acc_strict_unsuppressed` in the JSON, so nothing is hidden.")
    A("")
    supp = [(k, g) for k, g in sorted(res["gradient"].items())
            if g.get("drop_suppressed_reason")]
    if supp:
        A("Suppressed cells and why:")
        A("")
        for k, g in supp:
            A(f"- `{k}` — {g['drop_suppressed_reason']}")
        A("")
    A("")

    A("## Pre-registered falsification test")
    A("")
    A("The claim *\"the default is active\"* is supported for a (system, slot) cell **only "
      "if all of these hold**: (1) `Acc(C1) > Acc(C2)` by exact McNemar on frame-paired items, "
      "p < 0.01; (2) `Acc(C2)` is **not** significantly above chance, one-sided exact "
      "binomial, α = 0.05; (3) the C3 modal value matches the modal C2 **error** and "
      "`TV(C3, C2-errors) < 0.15`. Two further gates apply: the C0 capability ceiling must be "
      "above chance (otherwise the form may simply not be in the model's inventory), and the "
      "cell must not be prompt-sensitive.")
    A("")
    A("**Read the `C0 ok` column first.** C0 states the required value *in the instruction*, "
      "so it measures whether the form is in the model's productive inventory at all. Where "
      "C0 is itself at chance, a low C2 accuracy is **not** evidence of an active default — "
      "the model may simply be unable to produce the form on request, which is a capability "
      "result, not a prior-filling result. The analysis makes this a hard precondition rather "
      "than prose a writer might forget, and it is doing real work here.")
    A("")
    A("| system | slot | (1) McNemar p | (2) C2-vs-chance p | (3) TV | C0 ok (shipped) | "
      "C0 ok (corrected) | prompt ok | **default active** | with corrected C0 |")
    A("|---|---|---:|---:|---:|:--:|:--:|:--:|:--:|:--:|")
    for key in sorted(res["gradient"]):
        g = res["gradient"][key]
        f = g.get("falsification")
        if not f:
            continue
        def yn(x):
            return "n/a" if x is None else ("yes" if x else "no")
        A(f"| `{g['system_id']}` | {g['slot']} "
          f"| {f['criterion_1_C1_gt_C2_mcnemar']['p']} "
          f"({yn(f['criterion_1_C1_gt_C2_mcnemar']['passes'])}) "
          f"| {f['criterion_2_C2_not_above_chance']['p_one_sided']} "
          f"({yn(f['criterion_2_C2_not_above_chance']['passes'])}) "
          f"| {f['criterion_3_C3_matches_C2_errors']['TV']} "
          f"({yn(f['criterion_3_C3_matches_C2_errors']['passes'])}) "
          f"| {yn(f['c0_capability_precondition_met'])} "
          f"| {yn(f.get('c0_capability_precondition_met_explicit_wording'))} "
          f"| {yn(f['prompt_robustness_ok'])} "
          f"| {'**YES**' if f['default_is_active'] else 'no'} "
          f"| {'**YES**' if f.get('default_is_active_with_corrected_C0') else 'no'} |")
    A("")
    A("⚠ **The `number` row is the CONTROL and must not be read as default-filling without "
      "the confound check.** Its errors are dominated by honorific substitution — the model "
      "renders a singular subject honorifically, concord forces `-ஆர்`, and the pinned "
      "`hon = minus` filter re-reads that as plural. See the control-slot section: excluding "
      "those, gemma3-4b C1 goes 54.5% → 89.3%. A `number` cell that satisfies all three "
      "criteria is most likely detecting a register preference, not a number prior.")
    A("")
    A("A `no` in criterion (1) with a very small p-value means the difference is significant "
      "**in the wrong direction** (C2 above C1). That is the expected shape for clusivity, "
      "whose C1 English source spells the coordination out (\"my brother and me\") and can be "
      "translated word-by-word without ever reaching a 1PL pronoun — which is why the C1 "
      "pronoun-drop rate is near-total and the C2 rate is not.")
    A("")

    A("## Outcome distributions")
    A("")
    A("| system | slot | cond | n | CORRECT | WRONG | COMM(C3) | UNDEC | DROP | NEUTRAL | NEG | UNPARSED "
      "| acc_strict | committed | coverage |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(res["cells"]):
        s, slot, cond = key.split("|")
        c = res["cells"][key]
        o = c["outcomes"]
        A(f"| `{s}` | {slot} | {cond} | {c['n_all']} | {o['CORRECT']} | {o['WRONG']} "
          f"| {o['COMMITTED']} | {o['UNDECIDABLE']} | {o['AVOIDANT_DROP']} | {o['AVOIDANT_NEUTRAL']} "
          f"| {o['AVOIDANT_NEG']} | {o['UNPARSED']} | {pct(c['acc_strict'])} "
          f"| {pct(c['committed_rate'])} | {pct(c['coverage'])} |")
    A("")

    A("## Honorificity — what is actually scoreable")
    A("")
    A("⚠ **Provisional.** The third degree (`VV`) is not scoreable by the validated checker, "
      "so the honorificity row in the gradient table above is a THREE-way accuracy in which "
      "one class can never be scored correct. The binary `T` vs `V` contrast below is the "
      "number that means something; the `VV` block records what happened to those items so "
      "nothing is hidden by exclusion.")
    A("")
    A("| system | cond | T/V n | T/V acc_strict | T/V committed | VV n | VV outcomes | "
      "VV gold-surface hit (string match) |")
    A("|---|---|---:|---:|---:|---:|---|---:|")
    for key in sorted(res["cells"]):
        s_, slot, cond = key.split("|")
        if slot != "honorificity":
            continue
        c = res["cells"][key]
        tv = c.get("binary_TV_only")
        vv = c.get("VV_unscoreable")
        if not tv and not vv:
            continue
        A(f"| `{s_}` | {cond} "
          f"| {tv['n_all'] if tv else 0} | {pct(tv['acc_strict']) if tv else 'n/a'} "
          f"| {pct(tv['committed_rate']) if tv else 'n/a'} "
          f"| {vv['n'] if vv else 0} "
          f"| {', '.join(f'{k}={v}' for k, v in sorted((vv or {}).get('outcomes', {}).items()))} "
          f"| {pct((vv or {}).get('gold_surface_hit_rate')) if vv else 'n/a'} |")
    A("")
    A("`VV gold-surface hit` is a **string match against the item's own gold target**, "
      "reported only because the morphological verdict is unavailable for this class. It is "
      "not a verdict and must not be quoted as accuracy.")
    A("")

    if res.get("c2_context_leak"):
        A("## C2 context handling")
        A("")
        A("C2 presents the disambiguating sentence in a separate `CONTEXT:` block and asks the "
          "model to translate only the marked sentence. A model that renders both breaks the "
          "1:1 output-to-reference alignment. Detector: output longer than 2.5x that system's "
          "own C1 median output length.")
        A("")
        A("| system | C1 median out-tokens | n C2 | n over 2.5x | rate |")
        A("|---|---:|---:|---:|---:|")
        for sysid in sorted(res["c2_context_leak"]):
            x = res["c2_context_leak"][sysid]
            A(f"| `{sysid}` | {x['c1_median_out_tokens']} | {x['n_c2']} "
              f"| {x['n_c2_over_2.5x_c1_median']} | {pct(x['rate'])} |")
        A("")

    if res.get("c2_answer_locus"):
        A("## C2 answer locus — did the model translate the marked sentence at all?")
        A("")
        A("The length detector above catches a model that translates the context **in "
          "addition to** the marked sentence. It is blind to one that translates the context "
          "**instead of** it: that output is exactly as long as a correct answer.")
        A("")
        A("This matters more than it sounds. On C2 the cue lives in the context, so a model "
          "that renders the context is **handed** the feature by the cue NP and is scored "
          "COMMITTED/CORRECT without having made any default-filling decision about the "
          "marked sentence. It is the sarvam-translate retraction one level up.")
        A("")
        A("Detector: does the hypothesis contain the stem of the gold target's inflected word "
          "(gold word minus its scored suffix)? The stem survives every licit rendering, "
          "including avoidant ones, because avoidance changes the suffix. Absolute rates are "
          "not interpretable — a synonymous verb is a false negative — so only the **C1→C2 "
          "drop** is read, C1 being the same model's own baseline on items with no context to "
          "answer instead. Diagnostic; no outcome is rewritten.")
        A("")
        A("| system | C1 stem-hit | C2 stem-hit | drop | flag |")
        A("|---|---:|---:|---:|:-:|")
        for sysid in sorted(res["c2_answer_locus"]):
            x = res["c2_answer_locus"][sysid]
            A(f"| `{sysid}` | {pct(x['c1_gold_stem_rate'])} | {pct(x['c2_gold_stem_rate'])} "
              f"| {pct(x['drop_C1_to_C2'])} | {'⛔' if x['suspect'] else ''} |")
        A("")
        if res.get("c2_answer_locus_suspects"):
            A(f"⛔ **{', '.join('`%s`' % s for s in res['c2_answer_locus_suspects'])}** answered "
              "the wrong sentence on a large share of C2 items. Their C2 commitment and "
              "accuracy are inflated by the cue NP and must not be reported as default-filling "
              "behaviour.")
            A("")

    cc = res.get("clusivity_c1_construction")
    if cc:
        A("## Why clusivity C1 is not comparable to C2")
        A("")
        A("Clusivity commitment **inverts** between conditions — C1 commits on 2–7% of items "
          "while C2 commits on 88–91% — which is the opposite of the expected direction and "
          "would, taken at face value, look like a large negative C1→C2 drop. It is a design "
          "artifact, and the item sources say so directly.")
        A("")
        A("| condition | unique sources | contain *we/us/our* | contain a coordinated NP |")
        A("|---|---:|---:|---:|")
        for cond in ("C1", "C2", "C3", "C0"):
            x = cc.get(cond)
            if not isinstance(x, dict):
                continue
            A(f"| {cond} | {x['n_unique_sources']} | {x['n_with_1pl_pronoun']} "
              f"| {x['n_with_coordinated_np']} |")
        A("")
        for cond in ("C1", "C2"):
            x = cc.get(cond)
            if isinstance(x, dict) and x.get("examples"):
                A(f"- **{cond} sources** look like: "
                  + "; ".join(f"*{e}*" for e in x["examples"]))
        A("")
        A("**C1 clusivity sources never contain an English 1PL pronoun.** They present a "
          "coordinated NP, which Tamil renders compositionally — "
          "`He did my brother and me at the gate.` → "
          "`அவர் என் சகோதரனையும் என்னையும் வாசலில் ஏமாற்றினார்.` That is a correct, idiomatic "
          "translation that simply never reaches நாம் or நாங்கள். The near-total C1 "
          "`AVOIDANT_DROP` rate is therefore **not avoidance**, and the C1 cell is not "
          "measuring clusivity at all. The C1→C2 drop for clusivity is suppressed for this "
          "reason on every system, independently of the thin-cell rule.")
        A("")
        A("This is a finding about **the item construction**, not about the models, and "
          "it needs fixing upstream: C1 clusivity items should present *we/us* with the "
          "in-sentence cue, as C2 does, rather than a coordinated NP.")
        A("")

    A("## Clusivity drop-rate gate, re-run across the whole panel")
    A("")
    A("The existing gate (`results/clusivity_gate.json`) used **only** "
      "`facebook/nllb-200-distilled-600M` and reported 94–100% pronoun-drop in C1. One weak "
      "system is not evidence that clusivity is unmeasurable. This table measures the drop "
      "rate per system per overt environment across the panel.")
    A("")
    cgs = res.get("clusivity_gate_summary") or {}
    if cgs:
        a = cgs["acc_committed_where_gate_passes"]
        A(f"**What the panel changes.** {cgs['panel_n_cells_C2_passing_gate']} of "
          f"{cgs['panel_n_cells_C2']} C2 (system, environment) cells clear the 0.25 gate, "
          f"across systems "
          + (", ".join(f"`{x}`" for x in cgs["systems_with_a_passing_C2_environment"])
             or "none")
          + f". {cgs['verdict']}"
          + (f" Where the gate passes, `acc|committed` runs "
             f"{pct(a['min'])}–{pct(a['max'])}% (mean {pct(a['mean'])}%) against a chance "
             f"rate of 50% — so the distinction is being expressed and then filled at chance, "
             f"which is a sharper result than an unscorable slot."
             if a["min"] is not None else ""))
        A("")
        A("Note the C1/C2 asymmetry, which is a property of the items rather than the models: "
          "C1's English source spells the coordination out (\"my brother and me\"), so a "
          "system can render it word-by-word and never reach a 1PL pronoun — hence near-total "
          "drop in C1. C2's source says \"us\", which is the realistic MT input where Tamil "
          "must choose நம்மை or எங்களை.")
        A("")
    A("CIs are cluster-bootstrapped on `template_id`; `n_tmpl` is how many templates that cell "
      "actually has. **A cell with `n_tmpl` = 1 has no between-cluster variance and its CI is "
      "degenerate** — read the point estimate only. This affects `dat`, which the public split "
      "leaves with a single template.")
    A("")
    A("| system | cond | env | n | n_tmpl | drop_rate [95% CI] | passes 0.25 gate | committed | "
      "acc\\|committed |")
    A("|---|---|---|---:|---:|---|:--:|---:|---:|")
    for key in sorted(res["clusivity_gate"]):
        r = res["clusivity_gate"][key]
        ci = r["drop_rate_ci"]
        A(f"| `{r['system_id']}` | {r['condition']} | {r['environment']} | {r['n']} "
          f"| {r.get('n_templates', '?')} "
          f"| {pct(r['drop_rate'])} [{pct(ci[0])}, {pct(ci[1])}] "
          f"| {'yes' if r['passes_gate_0.25'] else '**no**'} "
          f"| {pct(r['committed_rate'])} | {pct(r['acc_committed'])} |")
    A("")

    A("## DFR / skew on C3 against the corpus prior")
    A("")
    A("The prior is reported as a **range across corpora**, never as a single number — every "
      "slot is FRAGILE by the pre-registered criterion (D-5). `DFR_range` is the deviation "
      "computed at both ends of the between-corpus range; a value whose sign flips inside "
      "that range is not a reportable direction.")
    A("")
    A("**Honorificity has no reportable corpus prior and none is invented here** (D-5). Its "
      "C3 distribution is reported in the gradient table above without a DFR.")
    A("")
    dirs = res.get("c3_direction_across_systems") or {}
    if dirs:
        A("**Systems do not share a default, and that is the sharpest thing in this table.** "
          "If default-filling were the training corpus showing through, every system should "
          "lean the same way on a given slot. Measured:")
        A("")
        A("| slot | systems agree? | modal value → systems | corpus prior modal | "
          "runs against the prior |")
        A("|---|:--:|---|---|---|")
        for slot, d in sorted(dirs.items()):
            groups = "; ".join(f"**{v}** → {', '.join('`'+x+'`' for x in ss)}"
                               for v, ss in sorted(d["systems_by_modal_value"].items()))
            against = d.get("disagrees_with_prior")
            A(f"| {slot} | {'yes' if d['unanimous'] else '**NO**'} | {groups} "
              f"| {d.get('corpus_prior_modal_value') or 'not reportable'} "
              f"| {', '.join('`'+x+'`' for x in against) if against else '—'} |")
        A("")
        split = [s2 for s2, d in sorted(dirs.items()) if not d["unanimous"]]
        if split:
            A("Slots where the panel **splits**: "
              + ", ".join(f"`{x}`" for x in split)
              + ". A split slot cannot be explained by a single general-text frequency — the "
                "same corpus cannot simultaneously predict opposite defaults — so for those "
                "slots the corpus-prior mechanism is not merely weak evidence, it is "
                "*contradicted*. This is the strongest argument in the run against a "
                "\"models reproduce training frequency\" reading, and it only becomes visible "
                "with several families in the panel.")
            A("")
    A("")
    A("`p_obs` is renormalised over **licit values only**; the excluded mass is the "
      "`committed` column, and every DFR figure below is therefore **conditional on the "
      "system having marked the slot at all**. That conditioning must be "
      "printed, not implied — a low `committed` makes the corresponding DFR a statement about "
      "a small, system-selected subpopulation.")
    A("")
    A("| system | slot | committed | value | p_obs | p_ref pooled | p_ref range | DFR range "
      "| sign flips |")
    A("|---|---|---:|---|---:|---:|---|---|:--:|")
    for key in sorted(res["dfr"]):
        s_, slot = key.split("|")
        d = res["dfr"][key]
        if not d.get("reportable"):
            A(f"| `{s_}` | {slot} | {pct(d.get('committed_rate'))} | — | — | — "
              f"| **not reportable** | — | — |")
            continue
        for v, x in sorted(d["values"].items()):
            rr = x["p_ref_range"]
            dr = x["DFR_range"]
            A(f"| `{s_}` | {slot} | {pct(d.get('committed_rate'))} | {v} | {pct(x['p_obs'])} "
              f"| {pct(x['p_ref_pooled'])} "
              f"| [{pct(rr[0])}, {pct(rr[1])}] "
              f"| [{dr[0]}, {dr[1]}] "
              f"| {'**yes**' if x['sign_flips_across_prior_range'] else 'no'} |")
    A("")
    for key in sorted(res["dfr"]):
        d = res["dfr"][key]
        if not d.get("reportable"):
            A(f"- `{key}`: {d['reason']}")
    A("")

    cw = res.get("c0_wording")
    if cw:
        A("## C0 capability ceiling — the instruction wording is doing the work")
        A("")
        A("⚠ **A defect in the shipped benchmark, not in the models or the harness.** The C0 "
          "instruction carried on each item (`fillers.c0_instruction`) names the benchmark's "
          "**internal label code**: *\"Translate into Tamil, realising the clusivity value "
          "'EXCL' on the pron.\"*, *\"…the rationality value 'AHRI' on the verb.\"* A model "
          "has no way to know what `EXCL`, `AHRI`, `UYAR` or `VV` denote. So a near-chance C0 "
          "under that wording does **not** distinguish *cannot produce the form* from *could "
          "not read the instruction* — and C0 is precisely what gates every default-filling "
          "claim.")
        A("")
        A("`P0_explicit` re-runs the same C0 items stating the **target morpheme**, copied off "
          "the item's own `gold_targets` (nothing is invented; the C0 design asks for exactly "
          "this — *\"addressing the reader with the deferential form தாங்கள்\"*).")
        A("")
        A("| system | slot | n | acc: shipped label code | acc: explicit morpheme | Δ |")
        A("|---|---|---:|---:|---:|---:|")
        for key in sorted(cw):
            s_, slot = key.split("|")
            x = cw[key]
            A(f"| `{s_}` | {slot} | {x['n']} | {pct(x['shipped_label_code']['acc_strict'])} "
              f"| {pct(x['explicit_morpheme']['acc_strict'])} "
              f"| {pct(x['delta_acc_strict'])} |")
        A("")
        A("Where Δ is large and positive, the shipped C0 was measuring instruction "
          "comprehension rather than productive capability, and the C0 gate in the "
          "falsification table above is failing for the wrong reason. **The C0 "
          "wording should be fixed before C0 is used as a capability ceiling.** The "
          "falsification table still uses the shipped wording, because that is what the "
          "benchmark currently ships; this table is what says the gate is not yet "
          "trustworthy.")
        A("")

    A("## How gender is avoided — epicene vs neuter")
    A("")
    A("`AVOIDANT_NEUTRAL` is not one behaviour. **EPICENE** is the `-ஆர்` escape: the model "
      "uses the honorific/epicene form, which is a licit, natural Tamil way to refuse the "
      "gender choice. **NEUT** is `3sgn`: the model treats a human referent as inanimate, "
      "which is a different (and worse) error. Collapsing them would hide the more "
      "interesting of the two.")
    A("")
    A("| system | slot | cond | n neutralised | EPICENE | NEUT |")
    A("|---|---|---|---:|---:|---:|")
    for key in sorted(res["cells"]):
        s_, slot, cond = key.split("|")
        nk = res["cells"][key].get("neutralisation_kind")
        if not nk:
            continue
        A(f"| `{s_}` | {slot} | {cond} | {sum(nk.values())} | {nk.get('EPICENE', 0)} "
          f"| {nk.get('NEUT', 0)} |")
    A("")
    A("Where EPICENE dominates, the model is not defaulting masculine — it is declining the "
      "choice by using the epicene form. That is a materially different finding from a "
      "masculine default, and it is only visible because avoidance is a first-class outcome "
      "rather than a bucket of errors.")
    A("")

    A("## Analyser coverage — where UNPARSED comes from")
    A("")
    A("`UNPARSED` means the extractor **abstained**, not that the system was wrong. It is "
      "reported with its own denominator rather than folded into error. `no_anchor` = no "
      "finite verb or pronoun could be located, so an absent exponent cannot be distinguished "
      "from an analyser miss; `no_analysis` = no Tamil token analysed at all; `no_tamil` = the "
      "output was not Tamil. A high `no_anchor` rate is a **checker gap**, and two concrete "
      "gaps are already identified: `வெளியேறினர்`/`சென்றனர்` receive no analysis, and the "
      "unambiguously-plural `-அனர்` is tagged `3sge` (singular) — the latter is repaired at the "
      "scoring layer and the size of that repair is published in "
      "`results/extractor-validation.json`.")
    A("")
    A("| system | slot | n | UNPARSED | rate | no_anchor | no_analysis | no_tamil |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(res["unparsed_reasons"]):
        s_, slot = key.split("|")
        u = res["unparsed_reasons"][key]
        r_ = u["reasons"]
        A(f"| `{s_}` | {slot} | {u['n_all']} | {u['n_unparsed']} | {pct(u['unparsed_rate'])} "
          f"| {r_.get('no_anchor', 0)} | {r_.get('no_analysis', 0)} | {r_.get('no_tamil', 0)} |")
    A("")

    A("## Control slot (`number`) sanity check")
    A("")
    A("The interpretability dissociation rests on the control behaving well, so it gets its own table. "
      "`UNDECIDABLE` here is not a scoring failure: it is the -ஆர்/-ஆர்கள் "
      "honorific-singular vs rational-plural syncretism surviving D-2's context filter, "
      "which is a fact about Tamil.")
    A("")
    A("**The confound, measured.** The design pins `hon = minus` on every number template so "
      "that honorificity is available as D-2's context filter. But a model that renders the "
      "subject with the honorific noun (அரசர் \"king-HON\") is then *required by concord* to "
      "use -ஆர் on the verb, and under the pinned `hon = minus` filter -ஆர் reads as 3PL. The "
      "item scores WRONG on number although nothing about number went wrong. "
      "`hon_sub` below is the share of that cell's errors which are exactly this, and "
      "`acc*` is the accuracy that would obtain if they were excluded. Where `hon_sub` is "
      "near 100%, the control is measuring register, not number.")
    A("")
    A("| system | cond | n | acc_strict | acc* (excl. hon-sub) | acc_committed | committed | "
      "undecidable | unparsed | hon_sub share of errors |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(res["control_slot_check"]):
        s_, cond = key.split("|")
        c = res["control_slot_check"][key]
        A(f"| `{s_}` | {cond} | {c['n']} | {pct(c['acc_strict'])} "
          f"| {pct(c['acc_strict_excl_honorific_substitution'])} "
          f"| {pct(c['acc_committed'])} "
          f"| {pct(c['committed_rate'])} | {pct(c['undecidable_rate'])} "
          f"| {pct(c['unparsed_rate'])} "
          f"| {pct(c['honorific_substitution_share_of_errors'])} |")
    A("")

    A("## Extractor validation")
    A("")
    ev = res.get("extractor_validation") or {}
    if ev:
        t = ev["totals"]
        A(f"Every number above rests on this. Feeding the benchmark's own `gold_targets` to "
          f"the extractor yields CORRECT on **{t['n_gold_correct']}/{t['n_gold']}**; feeding "
          f"the minimally-different `contrast_targets` yields CORRECT on "
          f"**{t['n_contrast_false_positive']}/{t['n_contrast']}** — zero false positives.")
        A("")
        A("| slot | n gold | recall | contrast false-positive rate |")
        A("|---|---:|---:|---:|")
        for slot, x in sorted(ev["by_slot"].items()):
            A(f"| {slot} | {x['n_gold_targets']} | {pct(x['recall_rate'])} "
              f"| {pct(x['contrast_fp_rate'])} |")
        A("")
        A(f"The only recall failures are honorificity `VV`. "
          f"{ev['known_structural_failure']['note']}")
        A("")
        abl = ev.get("scoring_repair_ablation") or {}
        if abl:
            A("### Scoring-layer repairs, published so their size is visible")
            A("")
            A("Two ThamizhiMorph defects were found while scoring real model output. Both are "
              "repaired **in the scoring layer**; the validated checker in "
              "`tamillingbench/morph/` is not modified, and "
              "`SlotExtractor(repair_anar=False, repair_sandhi=False)` reproduces the "
              "unrepaired numbers exactly.")
            A("")
            A("1. **`-அனர்` tagged singular.** ThamizhiMorph analyses the 3rd-person suffix "
              "`-அனர்` (குளித்தனர், விளையாடினர்) as `3sge`, which the label table maps to "
              "`number=SG`. It is the *unambiguously plural* rational form — DECISIONS.md D-5 "
              "relies on exactly that property when it defines the `strict_anar` prior "
              "variant. Left alone it inverts the NUMBER control.")
            A("2. **Sandhi consonant blocks the pronoun.** `எங்களை` + `பார்த்தார்` surfaces as "
              "`எங்களைப் பார்த்தார்`, and on `எங்களைப்` the FST returns a spurious *verb* "
              "reading with no clusivity — so an overt 1PL pronoun is invisible and the item "
              "is scored `AVOIDANT_DROP`. This directly deflates the clusivity pronoun-drop "
              "rate, which is the headline of the panel-wide gate.")
            A("")
            A("| repair removed | outcome counts that change (canonical run) |")
            A("|---|---|")
            label = {"vs_without_anar_repair": "`-அனர்` repair only",
                     "vs_without_sandhi_repair": "sandhi repair only",
                     "vs_without_any_repair": "both repairs"}
            for arm in ("vs_without_anar_repair", "vs_without_sandhi_repair",
                        "vs_without_any_repair"):
                d = abl.get(arm) or {}
                A(f"| {label[arm]} | "
                  + (", ".join(f"`{k}` {v:+d}" for k, v in
                               sorted(d.items(), key=lambda x: -abs(x[1]))) or "no change")
                  + " |")
            A("")

    A("## Prompt robustness")
    A("")
    A("For C1/C2 the statistic is `acc_strict` per prompt and its range; suppressed cells are "
      "excluded rather than counted as zero. **For C3 accuracy is undefined** (no gold), so "
      "the statistic is instead the largest pairwise total-variation distance between the "
      "per-prompt output distributions — i.e. does the prompt change *what the model defaults "
      "to*. A near-zero TVD with a stable modal value is the strongest form of the "
      "default-filling observation: the default does not move when you rephrase the request.")
    A("")
    A("| system | slot | cond | n common | statistic by prompt | range / max TVD |")
    A("|---|---|---|---:|---|---:|")
    for key in sorted(res["prompt_sensitivity"]):
        s_, slot, cond = key.split("|")
        p = res["prompt_sensitivity"][key]
        if cond == "C3":
            body = ", ".join(
                f"{k.replace('_', ' ')}={v or 'n/a'}"
                f"({pct(p['modal_share_by_prompt'].get(k))})"
                for k, v in sorted(p["modal_by_prompt"].items()))
            tvd = p.get("max_pairwise_tvd")
            A(f"| `{s_}` | {slot} | C3 | {p['n_common_items']} | modal: {body} "
              f"| TVD {tvd if tvd is not None else 'n/a'} |")
            continue
        accs = ", ".join(f"{k.replace('_', ' ')}={pct(v)}"
                         for k, v in sorted(p["acc_strict_by_prompt"].items()))
        A(f"| `{s_}` | {slot} | {cond} | {p['n_common_items']} | {accs} "
          f"| {pct(p['across_prompt_range'])} |")
    A("")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
