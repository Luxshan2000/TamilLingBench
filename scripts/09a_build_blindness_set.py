#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the metric-blindness scoring set and the per-metric input files.

WHAT CHANGED FROM PLAN 09 AS WRITTEN, AND WHY (recorded here because it is load-bearing):

The pre-registered design assumed `data/benchmark/items.jsonl` carries a full-sentence `reference`.
IT DOES NOT. The benchmark's `gold_targets[*].tamil` is the **slot-bearing core only** -- the
templates' `target_frame` is `'{NAME} {VERB}.'` while the source is `'{CUE} {V_PAST} at noon.'`.
Measured: C1 gold targets are 1-3 words (median 2) against 5-9 word English sources. Feeding those
to COMET as references would make every metric penalise the *missing adjunct*, which is a hundred
times larger than the morpheme effect and would render the whole section uninterpretable.

The fix keeps the DEMETR construction and drops the fragment:

    reference := a REAL system output, repaired to the gold slot value and re-verified by the FST
    good      := that same string          (the ceiling anchor; mt == ref, as in DEMETR/ACES)
    bad       := the same string with ONLY the obligatory morpheme wrong
    source    := the real, untrimmed English source

Every string a metric sees is therefore a fluent, complete Tamil translation of the actual source,
and `good` vs `bad` differ in exactly one grapheme-cluster span. That assertion is enforced, not
assumed, and the rejects are counted into `rejected_repairs.jsonl`.

Three arms, all built by the same machinery:

  controlled   from an outcome==CORRECT record. good = the output as emitted;
               bad = morpheme flipped to a wrong value;
               avoided = morpheme flipped to the neutralising/epicene form where one exists.
               Needs no system error, so it is defined wherever any system got the item right.
  natural      from an outcome==WRONG record. bad = the output as emitted; good = repaired.
  avoidance    from an outcome==AVOIDANT_NEUTRAL record. bad = the epicene output as emitted;
               good = repaired to the gold value. THIS IS THE ARM THE SECTION TURNS ON:
               it asks whether any metric charges anything at all for declining the distinction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tamillingbench.morph.checker import MorphChecker              # noqa: E402
from tamillingbench.morph.labels import Slot                        # noqa: E402
from tamillingbench.metrics.surgery import (                        # noqa: E402
    letters, n_diff_spans, normalize_sentence, splice)

OUT = ROOT / "outputs/metric_blindness"
CANONICAL_PROMPT = {"llm": "P1_minimal", "nmt": "nmt"}
SEED = 20260809

#: Slot -> the checker Slot enum used to verify a constructed form.
SLOT_ENUM = {
    "gender": Slot.GENDER,
    "number": Slot.NUMBER,
    "rationality": Slot.RATIONALITY,
    "honorificity": Slot.HONORIFICITY,
    "clusivity": Slot.CLUSIVITY,
}
#: Benchmark label -> checker feature value.
VALUE_MAP = {
    "gender": {"MASC": "MASC", "FEM": "FEM", "EPICENE": "EPICENE", "NEUT": "NEUT"},
    "number": {"SG": "SG", "PL": "PL"},
    "rationality": {"UYAR": "UYARTHINAI", "AHRI": "AHRINAI",
                    "UYARTHINAI": "UYARTHINAI", "AHRINAI": "AHRINAI"},
    # D-6.1: binary; தாங்கள் (VV) realises the same POLITE value as நீங்கள் (V).
    "honorificity": {"T": "FAMILIAR", "V": "POLITE", "VV": "POLITE"},
    "clusivity": {"INCL": "INCL", "EXCL": "EXCL"},
}
#: The neutralising ("avoidance") value per slot, where the grammar licenses one.
#: gender only: the epicene -ஆர் is the escape the panel actually takes.
AVOID_VALUE = {"gender": "EPICENE"}

TAMIL_NUMERALS = ["ஒரு", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "பத்து", "இரு"]
_NUM_RE = re.compile("|".join(TAMIL_NUMERALS) + r"|[0-9௦-௯]+")
_PUNCT = set(".,;:!?()[]{}\"'“”‘’…—–-")


# ------------------------------------------------------------------ helpers

def word_span_at(text: str, span: tuple[int, int]) -> tuple[int, int]:
    """Whitespace-delimited word containing `span`. Used to swap a whole form."""
    a, b = span
    i = text.rfind(" ", 0, a) + 1
    j = text.find(" ", max(b, a))
    return (i, len(text) if j < 0 else j)


def strip_punct(tok: str) -> str:
    return "".join(c for c in tok if c not in _PUNCT)


def content_tokens(text: str, protect: tuple[int, int]) -> list[tuple[int, int]]:
    """Spans of whitespace tokens that carry >=2 grapheme clusters and avoid `protect`."""
    out, i = [], 0
    for tok in text.split(" "):
        a, b = i, i + len(tok)
        i = b + 1
        if b <= protect[0] or a >= protect[1]:
            if len(letters(strip_punct(tok))) >= 2:
                out.append((a, b))
    return out


# ------------------------------------------------------------------ construction

class Builder:
    def __init__(self, checker: MorphChecker):
        self.ck = checker
        self.rejects: list[dict] = []
        self._cache: dict[tuple[str, str, str], bool] = {}
        self._known: dict[str, bool] = {}

    def in_lexicon(self, word: str) -> bool:
        """True if the FST can analyse `word` at all. An out-of-lexicon Tamil token is our
        proxy for a proper name -- which is exactly what the ACES NE control needs."""
        if word not in self._known:
            try:
                self._known[word] = bool(self.ck.analyse(word).analyses)
            except Exception:
                self._known[word] = True      # non-Tamil / artifact: not an NE candidate
        return self._known[word]

    def verify(self, word: str, slot: str, value: str, referent_number: str | None,
               referent_honorificity: str | None = None) -> bool:
        """Does `word` analyse to `slot=value` under D-2 (universal after context filter)?

        ⛔ THE NUMBER SLOT MUST NOT BE FILTERED ON NUMBER. Filtering a number item on the item's
        declared `referent_number` begs the question: verifying that a PL-gold sentence flipped to
        SG really is SG, while filtering out every non-PL reading, can only ever fail. Measured
        before the fix: 252 of 620 number flips were rejected and every survivor was PL->SG, so
        the controlled number cell covered one direction only.

        `MorphChecker.check` already provides the right filter for exactly this case
        (`referent_honorificity`, added 2026-08-08): the design pins `hon = minus` on every
        number template precisely so the OTHER pinned cell is available. It is needed here as well
        as merely allowed, because `-ஆர்கள்` is 3sg-honorific OR 3pl (D-2) and nothing else in the
        string decides.
        """
        key = (word, slot, value, referent_number, referent_honorificity)
        if key in self._cache:
            return self._cache[key]
        want = VALUE_MAP[slot].get(value, value)
        kw = {}
        if slot == "number":
            if referent_honorificity:
                kw["referent_honorificity"] = referent_honorificity
        elif referent_number:
            kw["referent_number"] = referent_number
        try:
            ok = bool(self.ck.check(word, SLOT_ENUM[slot], want, **kw).universal)
        except Exception:
            ok = False
        self._cache[key] = ok
        return ok

    def flip(self, sent: str, span: tuple[int, int], morph: str,
             slot: str, value: str, referent_number: str | None,
             tag: str, meta: dict, target_word: str | None = None,
             referent_honorificity: str | None = None) -> tuple[str, str] | None:
        """Set the obligatory slot to `value`, then assert BOTH gates before returning.

        Gate 1: the result differs from `sent` in exactly one grapheme-cluster span.
        Gate 2: the FST analyses the rebuilt word to `slot=value` under D-2.

        Candidates, in order:
          A. splice the recorded morpheme span -- preserves the model's own lexeme, so it is
             the minimal edit and always preferred;
          B. substitute the benchmark's whole attested target word -- rescues zero-exponent
             targets (the honorificity T imperative is a bare stem, where inserting a suffix
             at an empty span doubles the final vowel sign) and sandhi joins. Rejected by
             gate 1 whenever the model used a different lexeme, which is the common case.

        Failure is a counted data condition, not an exception: it lands in
        `rejected_repairs.jsonl` and in the reported rejection rate.
        """
        cands: list[tuple[str, object]] = [("morpheme-span", splice(sent, span, morph))]
        if target_word:
            wa, wb = word_span_at(sent, span)
            keep = "".join(c for c in sent[wa:wb] if c in _PUNCT)
            cands.append(("attested-word", splice(sent, (wa, wb), target_word + keep)))
        for how, sp in cands:
            if not sp.ok or n_diff_spans(sent, sp.text) != 1:
                continue
            wa, wb = word_span_at(sp.text, (span[0], span[0]))
            if not self.verify(strip_punct(sp.text[wa:wb]), slot, value, referent_number,
                               referent_honorificity):
                continue
            return sp.text, how
        self.rejects.append({**meta, "variant": tag, "reason":
                             "no candidate gave exactly one differing grapheme-cluster span "
                             "that the FST also analyses to the target value",
                             "sentence": sent, "span": list(span), "morpheme": morph,
                             "target_value": value})
        return None

    # ---- DEMETR / ACES control perturbations, all applied to `good` -----------

    def perturb(self, good: str, protect: tuple[int, int], seed_key: str,
                has_name: bool = False) -> dict[str, str]:
        rng = random.Random(int(hashlib.sha256(seed_key.encode()).hexdigest()[:12], 16) ^ SEED)
        toks = content_tokens(good, protect)
        out: dict[str, str] = {}
        if toks:
            a, b = rng.choice(toks)
            out["ctrl_rep"] = normalize_sentence(f"{good[:b]} {good[a:b]}{good[b:]}")   # DEMETR minor
            a, b = rng.choice(toks)
            out["ctrl_del"] = normalize_sentence((good[:a] + good[b:]).replace("  ", " ").strip())
        # ACES hallucination-named-entity. Gated on the item ACTUALLY having a name filler:
        # "out of lexicon" alone also fires on ordinary adverbs the FST simply does not cover
        # (measured: it picked 'பிறகு' as a name), which would make the positive control a
        # generic lexical-substitution control instead of an NE control.
        if has_name:
            for a, b in toks:
                w = strip_punct(good[a:b])
                if w and not self.in_lexicon(w):
                    sub = "இராமன்" if w != "இராமன்" else "கமலா"
                    cand = normalize_sentence(good[:a] + good[a:b].replace(w, sub) + good[b:])
                    if n_diff_spans(good, cand) == 1:
                        out["ctrl_ne"] = cand
                    break
        m = _NUM_RE.search(good)
        if m and not (m.start() < protect[1] and m.end() > protect[0]):
            alt = next(x for x in TAMIL_NUMERALS if x != m.group(0))
            cand = normalize_sentence(good[:m.start()] + alt + good[m.end():])
            if n_diff_spans(good, cand) == 1:
                out["ctrl_num"] = cand
        return out


# ------------------------------------------------------------------ main

def build_swap_table(items: dict[str, dict]) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Attested value->word groups mined from the benchmark's OWN gold/contrast targets.

    The scorer emits `morpheme_char_span` only where the slot is realised by a bound suffix. It is
    null for pronoun-borne slots -- every clusivity CORRECT record and 1,237 of the honorificity
    ones -- because the exponent is a whole word, not a suffix. Without a recovery path those
    slots vanish from the scoring set entirely, and clusivity + honorificity are exactly the two
    the project's provisionality ordering wants in the headline.

    The recovery needs no hand-written Tamil: the benchmark already contains both members of
    every pair, FST-generated and FST-verified, e.g.

        {'INCL': 'நம்மை',  'EXCL': 'எங்களை'}      {'INCL': 'நமது',  'EXCL': 'எங்களது'}
        {'T': 'எழுது',     'V': 'எழுதுங்கள்', 'VV': 'எழுதுக'}

    Returns slot -> word -> [group, ...]; a word can sit in several groups (case syncretism),
    so every group is tried and the FST decides.
    """
    out: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for it in items.values():
        group: dict[str, str] = {}
        for t in it["gold_targets"]:
            w = slot_word(t)
            if w:
                group.setdefault(it["gold_value"], w)
        for t in it.get("contrast_targets") or []:
            w = slot_word(t)
            if w and t.get("value"):
                group.setdefault(t["value"], w)
        if len(group) < 2:
            continue
        for w in group.values():
            if group not in out[it["slot"]][w]:
                out[it["slot"]][w].append(group)
    return {k: dict(v) for k, v in out.items()}


def recover_by_word(sent: str, slot: str, from_value: str, to_value: str,
                    table: dict[str, dict[str, list[dict[str, str]]]]) -> tuple[str, tuple[int, int]] | None:
    """Locate a whitespace token that is the attested `from_value` form and swap it.

    Returns (rebuilt sentence, span of the swapped word in the ORIGINAL sentence), or None.
    The one-span gate and the FST gate are applied by the caller's `flip`, not here.
    """
    groups = table.get(slot, {})
    i = 0
    for tok in sent.split(" "):
        a, b = i, i + len(tok)
        i = b + 1
        w = strip_punct(tok)
        if not w:
            continue
        for g in groups.get(w, ()):
            if g.get(from_value) == w and g.get(to_value):
                wa = a + tok.index(w)
                return g[to_value], (wa, wa + len(w))
    return None


def slot_word(target: dict) -> str | None:
    """The whole attested word carrying the slot morpheme inside a gold/contrast target.

    `target['tamil']` is the benchmark's slot-bearing core ('அகநகை வந்தாள்.'); the word we want
    is the one containing `morpheme_char_span`. Zero-exponent targets have an empty span, which
    still resolves to the right word.
    """
    tam, sp = target.get("tamil"), target.get("morpheme_char_span")
    if not tam or sp is None:
        return None
    tam = normalize_sentence(tam)
    a, b = word_span_at(tam, (int(sp[0]), int(sp[1])))
    return strip_punct(tam[a:b]) or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-metric-inputs", action="store_true")
    args = ap.parse_args()

    items = {}
    for line in open(ROOT / "data/benchmark/items.jsonl"):
        r = json.loads(line)
        if r["condition"] == "C1":
            items[r["item_id"]] = r
    print(f"C1 items: {len(items)}")

    swap = build_swap_table(items)
    print("swap table: " + ", ".join(f"{k}={len(v)} forms" for k, v in sorted(swap.items())))

    ck = MorphChecker()
    b = Builder(ck)
    records: list[dict] = []
    stats = Counter()
    seen_controlled: set[tuple[str, str]] = set()

    files = sorted((ROOT / "outputs/scored").glob("*/*_C1_*.jsonl"))
    for f in files:
        for line in open(f):
            r = json.loads(line)
            it = items.get(r["item_id"])
            if it is None:
                continue
            slot, outcome = r["slot"], r["outcome"]
            arm = {"CORRECT": "controlled", "WRONG": "natural",
                   "AVOIDANT_NEUTRAL": "avoidance"}.get(outcome)
            if arm is None:
                continue
            span, hyp = r.get("morpheme_char_span"), r.get("hypothesis")
            if not hyp:
                stats[f"skip:{arm}:no_hypothesis"] += 1
                continue
            # One controlled record per (item, system): the canonical prompt, so a system's
            # three prompt variants do not silently triple its weight in the controlled arm.
            if arm == "controlled":
                key = (r["item_id"], r["system_id"])
                if key in seen_controlled:
                    continue
                seen_controlled.add(key)

            sent = normalize_sentence(hyp)
            gold_val = it["gold_value"]
            gold_t = it["gold_targets"][0]
            contrasts = it.get("contrast_targets") or []
            ctr = contrasts[0] if contrasts else None
            meta = {"item_id": r["item_id"], "system_id": r["system_id"],
                    "prompt_id": r["prompt_id"], "slot": slot, "arm": arm,
                    "template_id": it["template_id"]}
            refnum = it.get("referent_number")
            # 'minus'/'plus' in the template's holds_constant -> the checker's feature values.
            refhon = {"minus": "FAMILIAR", "plus": "POLITE"}.get(
                (it.get("holds_constant") or {}).get("hon"))
            # The value the emitted string carries, and the value we need to write into it.
            from_v = (r.get("emitted_value") or r.get("neutralising_value")
                      or (gold_val if arm == "controlled" else None))
            to_v = (ctr or {}).get("value") if arm == "controlled" else gold_val
            if not from_v or not to_v:
                stats[f"skip:{arm}:no_target_value"] += 1
                continue

            word_repl = None
            if span:
                span = [int(span[0]), int(span[1])]
                if sent != hyp:            # offsets are only valid on the string they came from
                    frag = normalize_sentence(hyp[span[0]:span[1]])
                    off = sent.find(frag, max(0, span[0] - 4)) if frag else span[0]
                    if off < 0 or (frag and frag not in sent):
                        stats[f"skip:{arm}:normalisation_moved_span"] += 1
                        continue
                    span = [off, off + len(frag)]
            else:
                # Pronoun-borne slot: the exponent is a whole word, so the scorer recorded no
                # suffix span. Recover it from the benchmark's own attested word pairs.
                rec = recover_by_word(sent, slot, from_v, to_v, swap)
                if rec is None:
                    stats[f"skip:{arm}:no_span_and_no_word_recovery"] += 1
                    continue
                word_repl, wspan = rec
                span = list(wspan)
                stats[f"recovered_span:{arm}:{slot}"] += 1
            span = (int(span[0]), int(span[1]))

            how = {}
            if arm == "controlled":
                good = sent
                # bad = flip to the modal wrong value (open question 1: modal system error).
                if word_repl is None and (ctr is None or ctr.get("surface_morpheme") is None):
                    stats["skip:controlled:no_contrast_morpheme"] += 1
                    continue
                got = b.flip(sent, span, word_repl if word_repl is not None
                             else ctr["surface_morpheme"], slot, to_v, refnum, "bad", meta,
                             target_word=None if word_repl else slot_word(ctr),
                             referent_honorificity=refhon)
                if got is None:
                    stats["reject:controlled:bad"] += 1
                    continue
                bad, how["bad"] = got
                variants = {"good": good, "bad": bad}
                bad_value = to_v
                av = AVOID_VALUE.get(slot)
                if av and it.get("neutralization_possible"):
                    epi = b.flip(sent, span, "ார்", slot, av, refnum, "avoided", meta,
                                 referent_honorificity=refhon)
                    if epi is not None:
                        variants["avoided"], how["avoided"] = epi
                variants.update(b.perturb(good, span, r["item_id"] + r["system_id"],
                                          has_name=bool((it.get("fillers") or {}).get("name_id"))))
            else:
                bad = sent
                bad_value = r.get("emitted_value") or r.get("neutralising_value")
                got = b.flip(sent, span, word_repl if word_repl is not None
                             else gold_t["surface_morpheme"], slot, gold_val, refnum,
                             "good", meta,
                             target_word=None if word_repl else slot_word(gold_t),
                             referent_honorificity=refhon)
                if got is None:
                    stats[f"reject:{arm}:good"] += 1
                    continue
                good, how["good"] = got
                variants = {"good": good, "bad": bad}

            variants["empty"] = ""
            records.append({**meta, "source": it["source"],
                            "gold_value": gold_val, "bad_value": bad_value,
                            "outcome": outcome, "morpheme_char_span": list(span),
                            "construction": how,
                            "provisional": bool(it.get("slot") in ("honorificity", "rationality")),
                            "variants": variants})
            stats[f"built:{arm}:{slot}"] += 1

    ck.close()

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "scoring_set.jsonl", "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(OUT / "rejected_repairs.jsonl", "w") as fh:
        for rej in b.rejects:
            fh.write(json.dumps(rej, ensure_ascii=False) + "\n")

    by_arm_slot = Counter((r["arm"], r["slot"]) for r in records)
    by_arm_sys = Counter((r["arm"], r["system_id"]) for r in records)
    attempted = Counter()
    for k, v in stats.items():
        if k.startswith(("built:", "reject:")):
            attempted[k.split(":")[1]] += v
    rej_rate = {a: round(sum(v for k, v in stats.items()
                             if k.startswith(f"reject:{a}")) / max(1, attempted[a]), 4)
                for a in ("controlled", "natural", "avoidance")}
    statsd = {
        "seed": SEED,
        "n_records": len(records),
        "by_arm_slot": {f"{a}|{s}": n for (a, s), n in sorted(by_arm_slot.items())},
        "by_arm_system": {f"{a}|{s}": n for (a, s), n in sorted(by_arm_sys.items())},
        "variant_counts": dict(Counter(k for r in records for k in r["variants"])),
        "repair_rejection_rate_by_arm": rej_rate,
        "counters": dict(sorted(stats.items())),
        "note_reference_construction": (
            "The benchmark has no full-sentence reference; gold_targets are the slot-bearing "
            "core only (median 2 words vs 5-9 word sources). The reference used here is the "
            "'good' variant: a real system output verified by the FST to carry the gold slot "
            "value. This is the DEMETR/ACES construction (perturb a reference to make the "
            "hypothesis) and keeps source, reference and hypothesis length-aligned."),
    }
    (OUT / "scoring_set.stats.json").write_text(json.dumps(statsd, ensure_ascii=False, indent=2))
    print(json.dumps({k: statsd[k] for k in
                      ("n_records", "by_arm_slot", "variant_counts",
                       "repair_rejection_rate_by_arm")}, ensure_ascii=False, indent=2))

    if args.emit_metric_inputs:
        emit_inputs(records)
    return 0


def emit_inputs(records: list[dict]) -> None:
    """Line-aligned metric inputs, deduplicated on (src, mt, ref).

    NEVER join on string equality downstream: segments repeat across systems and the empty
    variant is empty. `index.tsv` maps line number -> (item_id, system_id, prompt_id, variant),
    and `unique.tsv` maps line number -> the deduplicated triple id.
    """
    d = OUT / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    uniq: dict[tuple[str, str, str], int] = {}
    rows: list[tuple[int, dict, str]] = []
    for rec in records:
        ref = rec["variants"]["good"]
        for var, mt in rec["variants"].items():
            key = (rec["source"], mt, ref)
            if key not in uniq:
                uniq[key] = len(uniq)
            rows.append((uniq[key], rec, var))

    triples = [k for k, _ in sorted(uniq.items(), key=lambda kv: kv[1])]
    with open(d / "segments.src.txt", "w") as fs, \
         open(d / "segments.mt.txt", "w") as fm, \
         open(d / "segments.ref.txt", "w") as fr, \
         open(d / "segments.metricx.jsonl", "w") as fx, \
         open(d / "segments.metricx_qe.jsonl", "w") as fq:
        for src, mt, ref in triples:
            fs.write(src.replace("\n", " ") + "\n")
            fm.write(mt.replace("\n", " ") + "\n")
            fr.write(ref.replace("\n", " ") + "\n")
            fx.write(json.dumps({"source": src, "hypothesis": mt, "reference": ref},
                                ensure_ascii=False) + "\n")
            fq.write(json.dumps({"source": src, "hypothesis": mt, "reference": ""},
                                ensure_ascii=False) + "\n")
    with open(d / "index.tsv", "w") as fi:
        fi.write("line\titem_id\tsystem_id\tprompt_id\tarm\tslot\tvariant\n")
        for line, rec, var in rows:
            fi.write(f"{line}\t{rec['item_id']}\t{rec['system_id']}\t{rec['prompt_id']}\t"
                     f"{rec['arm']}\t{rec['slot']}\t{var}\n")
    digest = hashlib.sha256(
        (d / "segments.mt.txt").read_bytes() + (d / "segments.ref.txt").read_bytes()
    ).hexdigest()
    (d / "inputs.sha256").write_text(digest + "\n")
    print(f"emitted {len(triples)} unique (src, mt, ref) triples for {len(rows)} "
          f"(record, variant) pairs -> {d}\nsha256 {digest[:16]}")


if __name__ == "__main__":
    sys.exit(main())
