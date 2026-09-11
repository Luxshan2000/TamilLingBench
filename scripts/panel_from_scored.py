#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the canonical single-prompt panel from outputs/scored/ -> results/panel.json.

WHY THIS EXISTS, AND THE BUG IT PREVENTS
----------------------------------------
results/eval-results.json pools every prompt a system happens to have. That pooling is
not comparable across systems, because the panel is prompt-ragged: Gemma-3 1B/4B were run
on four context prompts at full n, Gemma-3 12B and Qwen3-8B on P4 at full n plus P1/P2/P3
at n=40 per slot, and Llama-3.1-8B on P4 only. A headline table built on pooled cells is
built on different prompt mixtures per row.

Two selection mistakes are available here and both are silent:

  1. Filtering on ``prompt_id in {'P1_minimal','P1_context'}`` DROPS THE ENTIRE DEDICATED-NMT
     ARM. NLLB-200 and IndicTrans2 take no prompt; every one of their records carries
     ``prompt_id == 'nmt'``. Losing that arm loses the LLM-vs-dedicated-NMT contrast, which
     is the comparison that makes the result legible to an MT audience.
  2. Choosing P1_context as the canonical C2 prompt silently thins three of the five LLMs.
     Measured coverage of C2 items per slot:
         P1_context : G3-1B 220 · G3-4B 220 · G3-12B 40 · Qwen3-8B 40 · Llama3.1-8B 0
         P4_context : G3-1B 220 · G3-4B 220 · G3-12B 220 · Qwen3-8B 220 · Llama3.1-8B 220
     P4_context is the ONLY context prompt on which every LLM has the complete item set,
     so it is the canonical C2 prompt. P1_context is retained as a robustness column
     wherever it reaches the reporting floor.

Both mistakes fail the build here rather than passing quietly: ``--strict`` asserts that
every system in the panel is present in every headline cell.

THE REPORTING FLOOR
-------------------
Accuracy is undefined at zero commitment, and unstable just above it. A cell reports an
accuracy only if it has >= 30 committed items AND >= 30% commitment; otherwise the
accuracy is None with a stated reason and the paper prints n/a. Commitment and avoidance
are ALWAYS reported, because they are defined everywhere and they are the quantity the
headline is about.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# --scored / --dest point the same code at another scored tree (for example
# outputs/scored_before_repair) without overwriting the released panel.
SCORED = ROOT / "outputs" / "scored"
DEST = ROOT / "results" / "panel.json"

# Fixed panel order.
# Gemma-3n is its own architecture family (MatFormer + per-layer embeddings) and is placed
# after the Gemma-3 scale curve rather than inside it: E2B/E4B are NOT extra points on the
# 1b->27b curve and must not be plotted as such.
SYSTEM_ORDER = ["gemma3-1b", "gemma3-4b", "gemma3-12b", "gemma3-27b",
                "gemma3n-e2b", "gemma3n-e4b",
                "qwen3-8b", "qwen3-14b", "qwen3-32b", "llama31-8b", "sarvam-translate",
                "nllb-3.3b", "indictrans2-1b"]
SLOT_ORDER = ["honorificity", "clusivity", "rationality", "gender", "number"]
CONDITIONS = ["C0", "C1", "C2", "C3"]

# Systems that take no prompt. Every record they emit carries prompt_id == 'nmt'.
# IndicTrans2 will land with the same id and must be picked up without a code change,
# so membership is decided by the data, not by this list.
NMT_PROMPT_ID = "nmt"

# Canonical prompt per condition for prompted systems.
CANONICAL_PROMPT = {
    "C0": "P0_specified",   # the SHIPPED (defective) C0 wording; P0_explicit is the fix
    "C1": "P1_minimal",
    "C2": "P4_context",     # the only context prompt complete for every LLM
    "C3": "P1_minimal",
}
ROBUSTNESS_PROMPT = {"C2": "P1_context"}

MIN_COMMITTED = 30
MIN_COMMIT_RATE = 0.30

#: Cardinality of the outcome space, and hence the chance rate (1/k).
#:
#: ⚠ Keyed by FAMILY, not by slot, and declared here rather than copied off a record.
#:
#: 1. **Copying off a record was a latent bug.** The cell used to report
#:    `records[0]["chance_rate"]`, i.e. whichever item happened to sort first.
#: 2. **Honorificity holds two paradigms of different cardinality.** DECISIONS.md D-6.1
#:    ruled the INDICATIVE binary ({நீ,நீர்} vs {நீங்கள்,தாங்கள்}), and D-7.3 ruled the
#:    IMPERATIVE genuinely ternary (வா < வாருங்கள் < வருக). So HON-V and HON-P are k=2 at
#:    50% while HON-I is k=3 at 33.3%, and **a single slot-level honorificity chance rate
#:    is wrong whichever value you pick**. That is why cells carry `chance_rate_by_family`
#:    and `chance_rate` is None for honorificity rather than a plausible-looking average.
#: 3. Already-scored records on disk carry the pre-D-6 values; they are evidence and are
#:    not rewritten, so `chance_rate_recorded` reports what they say without trusting it.
FAMILY_K = {"HON-V": 2, "HON-P": 2, "HON-I": 3}
SLOT_K = {"clusivity": 2, "rationality": 2, "gender": 2, "number": 2}


def load_records():
    if not SCORED.is_dir():
        raise SystemExit(f"no scored outputs at {SCORED}")
    for sysdir in sorted(SCORED.iterdir()):
        if not sysdir.is_dir():
            continue
        for f in sorted(sysdir.glob("*.jsonl")):
            for line in f.open(encoding="utf-8"):
                line = line.strip()
                if line:
                    yield json.loads(line)


def summarise(records, slot=None):
    """Outcome counts, commitment, avoidance and accuracy for one cell."""
    outcomes = Counter(r["outcome"] for r in records)
    n = len(records)
    committed = [r for r in records if r.get("committed")]
    n_comm = len(committed)
    n_correct = sum(1 for r in committed if r.get("correct"))
    n_avoid = sum(1 for r in records if r.get("avoidant"))
    n_unparsed = outcomes.get("UNPARSED", 0)
    n_undec = sum(1 for r in records if r.get("undecidable"))

    commit_rate = n_comm / n if n else None
    cell = {
        "n": n,
        "n_templates": len({r.get("template_id") for r in records}),
        "outcomes": {k: outcomes.get(k, 0) for k in sorted(outcomes)},
        "n_committed": n_comm,
        "committed_rate": round(commit_rate, 4) if commit_rate is not None else None,
        "avoidance_rate": round(n_avoid / n, 4) if n else None,
        "unparsed_rate": round(n_unparsed / n, 4) if n else None,
        "undecidable_rate": round(n_undec / n, 4) if n else None,
        # Declared, never copied off a record -- see the FAMILY_K / SLOT_K comment.
        # None for honorificity: its families differ in cardinality and any single number
        # would be wrong for at least one of them.
        "chance_rate": (round(1.0 / SLOT_K[slot], 10) if slot in SLOT_K else None),
        "chance_rate_by_family": {
            f: round(1.0 / FAMILY_K[f], 10)
            for f in sorted({r.get("slot_family") for r in records
                             if r.get("slot_family") in FAMILY_K})} or None,
        "chance_rate_recorded": sorted({r.get("chance_rate") for r in records
                                        if r.get("chance_rate") is not None}) or None,
        "provisional": bool(any(r.get("provisional") for r in records)),
    }

    # Accuracy, suppressed below the reporting floor. Two denominators, both stated:
    # acc_committed conditions on having made the choice; acc_strict counts avoidance
    # as not-correct. Neither is folded into the other.
    thin = []
    if n_comm < MIN_COMMITTED:
        thin.append(f"committed n={n_comm} < {MIN_COMMITTED}")
    if commit_rate is not None and commit_rate < MIN_COMMIT_RATE:
        thin.append(f"commitment {commit_rate:.1%} < {MIN_COMMIT_RATE:.0%}")
    if thin:
        cell["acc_committed"] = None
        cell["acc_strict"] = None
        cell["suppressed_reason"] = "; ".join(thin)
    else:
        cell["acc_committed"] = round(n_correct / n_comm, 4)
        cell["acc_strict"] = round(n_correct / n, 4)
        cell["suppressed_reason"] = None
    # Kept for the appendix only: what the number would have been. Never plotted.
    cell["acc_committed_unsuppressed"] = round(n_correct / n_comm, 4) if n_comm else None
    cell["acc_strict_unsuppressed"] = round(n_correct / n, 4) if n else None

    # C3 has no gold: report the distribution over emitted licit values instead.
    emitted = Counter(r["emitted_value"] for r in committed if r.get("emitted_value"))
    total = sum(emitted.values())
    if total:
        cell["emitted_distribution"] = {
            k: round(v / total, 4) for k, v in sorted(emitted.items(),
                                                      key=lambda kv: -kv[1])
        }
        cell["emitted_counts"] = dict(sorted(emitted.items(), key=lambda kv: -kv[1]))
        top = max(emitted.items(), key=lambda kv: kv[1])
        cell["modal_value"] = top[0] if not thin else None
        cell["modal_share"] = round(top[1] / total, 4) if not thin else None
        cell["modal_suppressed_reason"] = "; ".join(thin) if thin else None
    else:
        cell["emitted_distribution"] = {}
        cell["emitted_counts"] = {}
        cell["modal_value"] = None
        cell["modal_share"] = None
        cell["modal_suppressed_reason"] = "no committed mass"
    return cell


def main() -> int:
    global SCORED, DEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="fail if any panel system is missing from any headline cell")
    ap.add_argument("--scored", default=str(SCORED.relative_to(ROOT)))
    ap.add_argument("--dest", default=str(DEST.relative_to(ROOT)))
    args = ap.parse_args()
    SCORED = ROOT / args.scored
    DEST = ROOT / args.dest
    DEST.parent.mkdir(parents=True, exist_ok=True)

    records = list(load_records())
    if not records:
        raise SystemExit("no scored records found")

    # Which systems are prompt-free? Decided by the data so IndicTrans2 needs no edit.
    prompts_by_system = defaultdict(set)
    for r in records:
        prompts_by_system[r["system_id"]].add(r.get("prompt_id"))
    nmt_systems = sorted(s for s, ps in prompts_by_system.items() if ps == {NMT_PROMPT_ID})
    prompted_systems = sorted(set(prompts_by_system) - set(nmt_systems))

    def canonical_prompt(system, condition):
        return NMT_PROMPT_ID if system in nmt_systems else CANONICAL_PROMPT[condition]

    buckets = defaultdict(list)
    for r in records:
        s, sl, cond, p = r["system_id"], r["slot"], r["condition"], r.get("prompt_id")
        if p == canonical_prompt(s, cond):
            buckets[(s, sl, cond)].append(r)
        if cond in ROBUSTNESS_PROMPT and p == ROBUSTNESS_PROMPT[cond]:
            buckets[(s, sl, cond + "_robust")].append(r)

    systems_present = sorted(prompts_by_system)
    cells = {f"{s}|{sl}|{c}": summarise(v, sl)
             for (s, sl, c), v in sorted(buckets.items()) if v}

    # --- completeness: a system silently missing must fail the build, not pass quietly --
    #
    # Completeness is checked over the slots the DATASET actually contains, not over
    # SLOT_ORDER wholesale. SLOT_ORDER is the full five-slot inventory; an item set with
    # fewer slots would otherwise report phantom missing cells on every run
    # and make --strict fire constantly — which trains people to pass --strict a miss.
    # The check that matters is preserved and in fact sharpened: within the slots present,
    # EVERY system must have EVERY condition, and a slot that some systems have and others
    # lack is exactly the silent-drop this guard exists to catch.
    slots_present = sorted({r["slot"] for r in records},
                           key=lambda s: SLOT_ORDER.index(s) if s in SLOT_ORDER else 99)
    slots_absent = [s for s in SLOT_ORDER if s not in slots_present]
    missing = []
    for s in systems_present:
        for sl in slots_present:
            for c in CONDITIONS:
                if f"{s}|{sl}|{c}" not in cells:
                    missing.append(f"{s}|{sl}|{c}")
    panel_gap = [s for s in SYSTEM_ORDER if s not in systems_present]

    out = {
        "_source": "scripts/panel_from_scored.py",
        "_what_this_is": (
            "One canonical prompt per system per condition, so every row of every headline "
            "table is the same prompt. Pooled-across-prompts numbers live in "
            "results/eval-results.json and are NOT comparable across systems, because the "
            "panel is prompt-ragged."
        ),
        "canonical_prompt_policy": {
            "prompted_systems": CANONICAL_PROMPT,
            "dedicated_nmt_systems": {c: NMT_PROMPT_ID for c in CONDITIONS},
            "robustness_prompt": ROBUSTNESS_PROMPT,
            "why_P4_for_C2": (
                "Measured coverage: P4_context is the only context prompt on which every "
                "LLM in the panel has the complete C2 item set. P1_context reaches full n "
                "for Gemma-3 1B/4B only (12B and Qwen3-8B have n=40 per slot; "
                "Llama-3.1-8B has none), so a P1-based headline would be computed on "
                "different item counts per row."
            ),
            "why_nmt_is_not_a_prompt": (
                "Dedicated NMT systems take no prompt and carry prompt_id='nmt'. Selecting "
                "on P1/P4 alone drops the entire dedicated-NMT arm, which is the control "
                "that separates 'MT systems avoid this' from 'LLMs avoid this'."
            ),
        },
        "reporting_floor": {
            "min_committed": MIN_COMMITTED,
            "min_commitment_rate": MIN_COMMIT_RATE,
            "rule": (
                "Accuracy is undefined at zero commitment. A cell below the floor reports "
                "no accuracy; commitment and avoidance are reported everywhere."
            ),
        },
        "systems_present": systems_present,
        "slots_present": slots_present,
        "slots_in_SLOT_ORDER_absent_from_dataset": slots_absent,
        "nmt_systems": nmt_systems,
        "prompted_systems": prompted_systems,
        "prompt_inventory": {s: sorted(x for x in ps if x)
                             for s, ps in sorted(prompts_by_system.items())},
        "panel_systems_expected": SYSTEM_ORDER,
        "panel_systems_missing": panel_gap,
        "cells_missing": missing,
        "cells": cells,
    }
    DEST.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DEST.relative_to(ROOT)}  ({len(cells)} cells, "
          f"{len(systems_present)} systems)")
    if nmt_systems:
        print(f"  dedicated-NMT arm present: {', '.join(nmt_systems)}")
    if panel_gap:
        print(f"  NOTE: expected panel systems not yet run: {', '.join(panel_gap)}")
    if missing:
        print(f"  WARNING: {len(missing)} missing cells: {missing[:8]}"
              f"{' ...' if len(missing) > 8 else ''}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
