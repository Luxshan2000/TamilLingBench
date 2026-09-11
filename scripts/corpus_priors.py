#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute the DFR prior tables and write `results/corpus-priors.{json,md}`.

    python scripts/corpus_priors.py                 # all built indexes
    python scripts/corpus_priors.py --boot 2000     # faster, for iteration

Stage 1 builds the **form inventory** — which surface types realise which slot value —
from the UNION of every corpus's type list, run through the FST once. Building it per
corpus would make the corpora incomparable, which is the one thing the cross-corpus
agreement analysis must not allow. It is cached at `data/corpus/slot_forms.json`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tamillingbench.corpus import priors as P                      # noqa: E402
from tamillingbench.corpus.index import FormIndex                  # noqa: E402
from tamillingbench.corpus.slots import (build_inventory, FormInventory,  # noqa: E402
                                         slot_specs, candidate)
from tamillingbench.corpus.sources import SOURCES                  # noqa: E402
from tamillingbench.morph.checker import MorphChecker              # noqa: E402

INDEX = ROOT / "data" / "corpus" / "index"
OUT_JSON = ROOT / "results" / "corpus-priors.json"
OUT_MD = ROOT / "results" / "corpus-priors.md"
#: Gzipped: the plain JSON is ~21 MB (372k `surface_only` diagnostic entries) and
#: compresses to ~1.9 MB, which is committable. It is a releasable artefact — it is
#: the counting rule made explicit, and a reviewer should be able to read it.
INV_PATH = ROOT / "data" / "corpus" / "slot_forms.json.gz"

#: Order matters only for presentation.
CORPUS_ORDER = ["tawiki", "sangraha", "cc100", "sangraha_speech", "opensubs",
                "ud_mwtt", "irumozhi"]


def load_indexes() -> list[FormIndex]:
    out = []
    for key in CORPUS_ORDER:
        p = INDEX / f"{key}.sqlite"
        if p.exists():
            out.append(FormIndex(p))
    return out


def build_or_load_inventory(indexes: list[FormIndex], rebuild: bool) -> FormInventory:
    if INV_PATH.exists() and not rebuild:
        with gzip.open(INV_PATH, "rt", encoding="utf-8") as fh:
            return FormInventory.from_json(json.load(fh))
    types: set[str] = set()
    for idx in indexes:
        for form, n, _df in idx.iter_types(min_n=1):
            if candidate(form):
                types.add(form)
    print(f"  candidate types across {len(indexes)} corpora: {len(types):,}", flush=True)
    t0 = time.time()
    inv = build_inventory(sorted(types), MorphChecker())
    print(f"  FST pass: {inv.n_types_analysed:,} analysed, {inv.n_types_finite:,} finite "
          f"verbs, {time.time()-t0:.0f}s", flush=True)
    INV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(INV_PATH, "wt", encoding="utf-8") as fh:
        json.dump(inv.to_json(), fh, ensure_ascii=False, indent=1)
    return inv


def top_forms(indexes, forms, k=12):
    tot: dict[str, int] = {}
    for idx in indexes:
        for f in forms:
            n = idx.frequency(f).n
            if n:
                tot[f] = tot.get(f, 0) + n
    return [[f, n] for f, n in sorted(tot.items(), key=lambda kv: -kv[1])[:k]]


def correct_honorificity(slot_docs, brackets):
    """Re-estimate the ternary honorificity prior with the தாங்கள் reflexive share removed.

    The raw DEFERENTIAL count is dominated by the 3rd-person reflexive reading. Scaling it
    by the measured address rate is a first-order correction, not a fix: it assumes the
    address rate estimated over sentences transfers to tokens, and the address rate is
    itself a LOWER bound because pro-drop hides the cue. Both directions are stated.
    """
    rate = {b["corpus"]: b["rate"] for b in brackets
            if b["n_sentences_scanned"] >= 200 and b["rate"] == b["rate"]}
    src = next((s for s in slot_docs
                if s["slot"] == "honorificity_pronoun" and s["variant"] == "nominative"),
               None)
    if src is None:
        return None
    rows = []
    for c in src["per_corpus"]:
        r = rate.get(c["corpus"])
        if r is None or c["total"] < 200:
            continue
        n = dict(c["n"])
        n["DEFERENTIAL"] = n["DEFERENTIAL"] * r
        tot = sum(n.values())
        rows.append({
            "corpus": c["corpus"], "address_rate_applied": r,
            "raw": {k: c["p"][k] for k in c["p"]},
            "corrected": {k: (v / tot if tot else 0.0) for k, v in n.items()},
            "deferential_raw_n": c["n"]["DEFERENTIAL"],
            "deferential_corrected_n": round(n["DEFERENTIAL"], 1),
        })
    return {"rows": rows, "narrative": HON_CORRECTED_NARRATIVE}


def write_hon_sample(indexes, n=300, seed=20260807):
    """300 random Tamil-Wikipedia sentences containing நீங்கள், for human annotation.

    The design calls for a human-annotated estimate of the singular/plural split on a
    300-sentence random sample of நீங்கள் tokens, used to bracket the prior. No single point
    estimate for the honorificity prior should be published without this bracket.
    This file is that sample. Wikipedia is CC BY-SA 4.0, so the sentences are releasable.
    """
    import random

    wiki = next((i for i in indexes if i.corpus == "tawiki"), None)
    if wiki is None:
        return None
    f = "நீங்கள்"
    rows = wiki._db.execute("SELECT s.id, s.text, d.ref FROM sent s JOIN doc d ON d.id=s.doc "
                            "WHERE s.text LIKE ?", (f"%{f}%",)).fetchall()
    rows = [r for r in rows if f in r[1].split()]
    random.Random(seed).shuffle(rows)
    out = ROOT / "data" / "corpus" / "hon-bracket-sample.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for sid, textv, ref in rows[:n]:
            fh.write(json.dumps({
                "id": f"NIINGAL-{sid}", "corpus": "tawiki", "doc_ref": ref,
                "sentence": textv,
                "question": ("Is நீங்கள் here addressing ONE person politely (honorific "
                             "singular) or MORE THAN ONE person (plain plural)?"),
                "label": None, "options": ["honorific_singular", "plain_plural",
                                            "indeterminate"],
            }, ensure_ascii=False) + "\n")
    return {"path": str(out.relative_to(ROOT)), "n_written": min(n, len(rows)),
            "n_candidates": len(rows), "seed": seed}


# --------------------------------------------------------------------------- markdown

def _pct(x: float) -> str:
    return "—" if x != x else f"{100*x:.2f}"


def _fmt_ci(t) -> str:
    return "—" if t[0] != t[0] else f"[{100*t[0]:.2f}, {100*t[1]:.2f}]"


def render_md(doc: dict) -> str:
    L: list[str] = []
    a = L.append
    a("# Tamil corpus priors — the DFR anchor\n")
    a(f"*Generated {doc['generated_utc']} by `scripts/corpus_priors.py`. "
      f"All rates are percentages within a slot; intervals are 95%.*\n")

    a("## Recommendation — resolving open question O-2\n")
    a(doc["recommendation"]["statement"] + "\n")
    for b in doc["recommendation"]["bullets"]:
        a(f"- {b}")
    a("")

    a("## 1. Corpora\n")
    a("| Corpus | Register | Documents | Tokens | Types | Licence | Ships? |")
    a("|---|---|---:|---:|---:|---|---|")
    for c in doc["corpora"]:
        a(f"| {c['name']} | {c['register']} | {c['n_docs']:,} | {c['n_tokens']:,} | "
          f"{c['n_types']:,} | {c['licence'][:46]} | "
          f"{'text+counts' if c['release_ok'] else ('counts only' if c['derived_ok'] else 'NOTHING')} |")
    a("")
    a("### Investigated and not obtained\n")
    a("| Corpus | Status | Why |")
    a("|---|---|---|")
    for c in doc["not_obtained"]:
        a(f"| {c['name']} | not obtained | {c['notes']} |")
    a("")

    a("## 2. The four slot priors\n")
    for sp in doc["slots"]:
        a(f"### {sp['slot']} — variant `{sp['variant']}`\n")
        vals = sp["values"]
        a("| Corpus | " + " | ".join(f"{v} %" for v in vals) + " | " +
          " | ".join(f"{v} 95% CI" for v in vals) + " | slot tokens | docs w/ slot | max DEFF |")
        a("|---|" + "---|" * (2 * len(vals) + 3))
        for c in sp["per_corpus"]:
            deffs = [c["deff"][v] for v in vals if c["deff"][v] == c["deff"][v]]
            a(f"| {c['corpus']} | "
              + " | ".join(_pct(c["p"][v]) for v in vals) + " | "
              + " | ".join(_fmt_ci(c["ci"][v]) for v in vals)
              + f" | {c['total']:,} | {c['n_docs_with_slot']:,} | "
              + (f"{max(deffs):.1f}" if deffs else "—") + " |")
        a(f"| **pooled** | " + " | ".join(_pct(sp["pooled_p"][v]) for v in vals) + " | "
          + " | ".join(_fmt_ci(sp["headline_interval"][v]) for v in vals)
          + f" | {sum(sp['pooled_n'].values()):,} | — | — |")
        a("")
        a(f"**Raw counts:** " + ", ".join(f"{v} = {sp['pooled_n'][v]:,}" for v in vals) + "\n")
        a(f"**Between-corpus range:** "
          + ", ".join(f"{v} {_fmt_ci(sp['between_corpus_range'][v])}" for v in vals) + "\n")
        a(f"**Verdict:** {sp['verdict']}\n")
        if sp["caveats"]:
            a("**Caveats:**\n")
            for c in sp["caveats"]:
                a(f"- {c}")
            a("")

    a("## 3. Cross-corpus agreement\n")
    a("| Slot | variant | max pairwise TVD | worst pair | between/within | fragile? |")
    a("|---|---|---:|---|---:|---|")
    for sp in doc["slots"]:
        worst = max(sp["within_vs_between"].values(),
                    key=lambda w: (w["between_over_within"]
                                   if w["between_over_within"] == w["between_over_within"]
                                   else -1))
        a(f"| {sp['slot']} | {sp['variant']} | {sp['max_pairwise_tvd']:.3f} | "
          f"{'–'.join(sp['disagreeing_pair'])} | "
          f"{worst['between_over_within']:.1f} | "
          f"{'**YES**' if sp['fragile'] else 'no'} |")
    a("")

    a("## 4. The register confound\n")
    a(doc["register"]["narrative"] + "\n")
    a("| Corpus | " + " | ".join(doc["register"]["forms"]) + " |")
    a("|---|" + "---:|" * len(doc["register"]["forms"]))
    for row in doc["register"]["density_per_million"]:
        a(f"| {row['corpus']} | "
          + " | ".join(f"{row['d'].get(f, 0):.1f}" for f in doc["register"]["forms"]) + " |")
    a("")

    a("## 5. தாங்கள் — the deferential/reflexive bracket\n")
    a(doc["thangal"]["narrative"] + "\n")
    a("| Corpus | sentences scanned | with a 2nd-person cue | rate | 95% CI |")
    a("|---|---:|---:|---:|---|")
    for t in doc["thangal"]["brackets"]:
        a(f"| {t['corpus']} | {t['n_sentences_scanned']:,} | {t['n_with_2p_cue']:,} | "
          f"{_pct(t['rate'])}% | {_fmt_ci(t['ci'])} |")
    a("")

    hc = doc.get("honorificity_corrected")
    if hc and hc["rows"]:
        a("### 5b. The honorificity prior after the தாங்கள் correction\n")
        a(hc["narrative"] + "\n")
        a("| Corpus | DEFERENTIAL raw % | DEFERENTIAL corrected % | FAMILIAR corr. % | "
          "POLITE corr. % | address rate used |")
        a("|---|---:|---:|---:|---:|---:|")
        for r in hc["rows"]:
            a(f"| {r['corpus']} | {100*r['raw']['DEFERENTIAL']:.2f} | "
              f"{100*r['corrected']['DEFERENTIAL']:.2f} | "
              f"{100*r['corrected']['FAMILIAR']:.2f} | "
              f"{100*r['corrected']['POLITE']:.2f} | {100*r['address_rate_applied']:.1f}% |")
        a("")

    a("## 6. What is actually being counted (audit)\n")
    a("Top forms per slot value, pooled over every indexed corpus. A gender prior whose "
      "MASC column is full of place names is a suffix-frequency table, not a gender "
      "prior; this table is how that is checked.\n")
    for sp in doc["slots"]:
        if sp["variant"] in ("all_forms", "surface_only", "strict_anar"):
            continue
        a(f"**{sp['slot']} / {sp['variant']}**\n")
        for v in sp["values"]:
            forms = sp.get("top_forms", {}).get(v, [])
            a(f"- `{v}` — " + ", ".join(f"{f} ({n:,})" for f, n in forms[:10]))
        a("")

    a("## 7. Honest limits\n")
    for lim in doc["limits"]:
        a(f"- {lim}")
    a("")
    return "\n".join(L)


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--rebuild-inventory", action="store_true")
    ap.add_argument("--md-only", action="store_true",
                    help="re-render the markdown from the existing JSON, refreshing the "
                         "narrative blocks; does not recompute any number")
    args = ap.parse_args()

    if args.md_only:
        doc = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        doc["recommendation"] = RECOMMENDATION
        doc["limits"] = LIMITS
        doc["register"]["narrative"] = REGISTER_NARRATIVE
        doc["thangal"]["narrative"] = THANGAL_NARRATIVE
        if doc.get("honorificity_corrected"):
            doc["honorificity_corrected"]["narrative"] = HON_CORRECTED_NARRATIVE
        OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        OUT_MD.write_text(render_md(doc), encoding="utf-8")
        print(f"re-rendered {OUT_MD}")
        return

    indexes = load_indexes()
    if not indexes:
        raise SystemExit("no indexes built; run scripts/build_corpus.py first")
    print(f"== indexes: {[i.corpus for i in indexes]}", flush=True)

    inv = build_or_load_inventory(indexes, args.rebuild_inventory)
    specs = slot_specs(inv)

    slot_docs = []
    for spec in specs:
        cps = [P.corpus_prior(idx, spec, n_boot=args.boot) for idx in indexes]
        sp = P.pool(cps, spec)
        d = sp.to_json()
        # Top counted forms per value. This is the single most useful audit artefact in the
        # report: it is how a reviewer checks that the "masculine verb" count is verbs and
        # not `பாக்கித்தான்`.
        d["top_forms"] = {v: top_forms(indexes, spec.forms.get(v, []), 12)
                          for v in spec.values}
        slot_docs.append(d)
        print(f"  {spec.slot:22s} {spec.variant:14s} "
              f"pooled={ {v: round(sp.pooled_p[v],4) for v in spec.values} } "
              f"tvd={sp.max_pairwise_tvd:.3f} fragile={sp.fragile}", flush=True)

    reg_forms = ["நீ", "நீங்கள்", "தாங்கள்", "உன்", "உங்கள்", "தங்கள்"]
    density = [{"corpus": i.corpus, "d": P.second_person_density(i)} for i in indexes]
    brackets = [P.thangal_bracket(i).__dict__ for i in indexes
                if i.meta.get("store_sentences")]

    corpora = [{
        "key": i.corpus,
        "name": SOURCES[i.corpus].name if i.corpus in SOURCES else i.corpus,
        "register": i.meta.get("register", ""),
        "licence": i.meta.get("licence", ""),
        "release_ok": bool(i.meta.get("release_ok", False)),
        "derived_ok": SOURCES[i.corpus].derived_ok if i.corpus in SOURCES else True,
        "n_docs": i.n_docs, "n_tokens": i.n_tokens,
        "n_types": int(i.meta.get("n_types", 0)),
        "n_sents": int(i.meta.get("n_sents", 0)),
        "doc_sample_every": i.meta.get("doc_sample_every", 1),
        "url": i.meta.get("source_url", ""),
    } for i in indexes]

    hon_corrected = correct_honorificity(slot_docs, brackets)

    doc = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_bootstrap": args.boot,
        "corpora": corpora,
        "not_obtained": [{"name": s.name, "notes": s.notes, "url": s.url}
                         for s in SOURCES.values() if not s.obtained],
        "inventory": {"n_types_analysed": inv.n_types_analysed,
                      "n_types_finite": inv.n_types_finite,
                      "n_types_noun_vetoed": inv.n_types_noun_vetoed,
                      "noun_veto": inv.noun_veto,
                      "path": str(INV_PATH.relative_to(ROOT))},
        "hon_bracket_sample": write_hon_sample(indexes),
        "slots": slot_docs,
        "register": {"forms": reg_forms, "density_per_million": density,
                     "narrative": REGISTER_NARRATIVE},
        "thangal": {"brackets": brackets, "narrative": THANGAL_NARRATIVE},
        "honorificity_corrected": hon_corrected,
        "recommendation": RECOMMENDATION,
        "limits": LIMITS,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MD.write_text(render_md(doc), encoding="utf-8")
    print(f"wrote {OUT_JSON} and {OUT_MD}")


REGISTER_NARRATIVE = (
    "Prasanna & Arora (IruMozhi, Findings of NAACL 2024) find Spoken Tamil substantially "
    "under-represented in NLP corpora. Tamil Wikipedia is overwhelmingly Literary Tamil, "
    "and a Literary corpus is also a corpus in which almost nobody is *addressed*. The "
    "table below is that confound made numeric: second-person pronoun density per million "
    "tokens. A formal or deferential skew in an honorificity prior estimated on Wikipedia "
    "may be a property of encyclopaedic register, not of Tamil.")

HON_CORRECTED_NARRATIVE = (
    "Scaling the raw DEFERENTIAL count by the measured 2nd-person address rate. This is a "
    "FIRST-ORDER correction and it is stated as such: it assumes the sentence-level "
    "address rate transfers to tokens, and the address rate is itself a lower bound "
    "because pro-drop hides the cue. It is shown because the uncorrected number is not "
    "merely imprecise but wrong by roughly an order of magnitude, and a reader who sees "
    "only the raw table would conclude that a quarter of Tamil Wikipedia's second-person "
    "address is deferential.")

THANGAL_NARRATIVE = (
    "தாங்கள் is 2nd-person DEFERENTIAL *or* 3rd-person REFLEXIVE ('they themselves'), and "
    "ThamizhiMorph analyses it only as `+pron+3pl+refl+nom` — the analyser has no "
    "deferential reading at all, so the FST cannot bracket this and a corpus heuristic "
    "must. A தாங்கள் token is scored as address if the sentence also contains a 2nd-person "
    "pronoun or a 2nd-person finite verb suffix. Pro-drop makes this a LOWER bound on the "
    "address reading.")

RECOMMENDATION: dict = {
    "statement": (
        "**Anchor the DFR prior on a three-corpus pool — Tamil Wikipedia + AI4Bharat "
        "Sangraha (verified Tamil) + CC-100 Tamil — and report it as a RANGE across those "
        "corpora, never as a single point estimate. Licence the claim per slot: clusivity "
        "and the scarcity of feminine agreement are robust enough to headline; "
        "rationality and the masculine/epicene split must be run at both ends of the "
        "between-corpus range; the honorificity prior is NOT reportable from corpus counts "
        "alone and is blocked on the 300-sentence human bracket.**"),
    "bullets": [
        "**No single corpus.** Every slot trips the fragility criterion. Measured, not "
        "asserted: the between-corpus range is 1.2× to 22.5× the widest within-corpus "
        "cluster-bootstrap interval, and max pairwise TVD runs 0.14 (clusivity) to 0.60 "
        "(honorificity). This matched the prediction made before the counts were run. The sentence that belongs in the "
        "paper is: *the dominant uncertainty in the prior is which corpus you choose, not "
        "how many tokens you count.*",
        "**Wikipedia's role is attestation, not arbitration.** It is the one large corpus "
        "whose licence (CC BY-SA 4.0) lets us quote sentences, so it backs the KWIC "
        "concordance for curation and for naturalness grounding. As a frequency "
        "anchor it is the most register-skewed of the three, and on the honorificity and "
        "gender slots it is the outlier.",
        "**Slot-by-slot licensing.** *Clusivity*: நாம் (inclusive) is the majority form in "
        "all three corpora; the direction is corpus-invariant even though the magnitude "
        "moves ~14 points (63.7–77.9%). Report with the range. *Gender*: the feminine rate "
        "is low in every corpus (3.0–12.2%), a robust and paper-worthy finding, but "
        "EPICENE moves 90.0%→65.5% and MASC 7.0%→22.3% between Wikipedia and CC-100 — run "
        "DFR at both ends and say so if the sign flips. *Rationality*: uyartiṇai spans "
        "34.5–69.6%, so the range straddles the midpoint and a DFR sign CAN flip inside "
        "it; report `strict_anar` (40.6% pooled) beside `suffix_bound` (59.0%) — the "
        "18-point gap is the -ஆர்கள் honorific-singular reading (D-2) on its own. "
        "*Honorificity*: see below.",
        "**The honorificity prior is blocked, and that is the honest answer.** Three "
        "independent problems stack: (i) நீங்கள் is honorific-singular OR plain-plural and "
        "nothing in the token decides, so a point estimate "
        "would need a human split; (ii) தாங்கள் is syncretic with the 3rd-person reflexive and "
        "ThamizhiMorph has no deferential reading for it at all — MEASURED, only 5.0% "
        "[3.9, 6.3] of Tamil-Wikipedia தாங்கள் sentences carry ANY 2nd-person cue, so the "
        "apparent 26.5% deferential rate falls to ~1.8% once corrected, a 15× error; "
        "(iii) verb-borne 2nd-person forms are rare in written Tamil — 2,090 tokens in "
        "30M for Wikipedia. `data/corpus/hon-bracket-sample.jsonl` "
        "is a 300-sentence annotation sample for that split; the prior should be "
        "published only after it is labelled.",
        "**Tier-1 priors are still the stronger claim and are NOT done here.** For the "
        "IndicTrans2 arm the training data (BPCC) is public and the prior should be "
        "computed on it — that is the version of the claim that says 'this model's output "
        "rate matches this model's training rate'. Samanantar was rejected on purpose: it "
        "is CC BY-NC (release-blocked like TTB) and its Tamil side is translated from "
        "English, so its slot distribution reflects translator defaults — the very "
        "phenomenon under test. Using it as the anchor would be circular.",
        "**Say the proxy gap out loud.** Gemma, Qwen and Llama do not disclose their "
        "pretraining mixture. Wikipedia, Sangraha and CC-100 are proxies for it. For the "
        "LLM arm the honest wording of the DFR result is *'the output rate deviates from "
        "general Tamil text frequency'*, which is weaker than *'deviates from its own "
        "training frequency'*. Do not let the two sentences be interchanged in the paper.",
    ],
}

LIMITS: list[str] = [
    "**Sampling.** Sangraha is 2 of 53 verified-Tamil shards (~4%) with every 3rd document "
    "kept; CC-100 is every 8th document of the full 1.38 GB xz. Sampling is systematic over "
    "documents, never a prefix, because both files are ordered by crawl/page id and that "
    "correlates with topic. Tamil Wikipedia is complete.",
    "**OPUS OpenSubtitles Tamil is encoding-damaged and was excluded.** Measured "
    "vowel-sign/consonant ratio 0.014 against 0.432 for Wikipedia: a legacy TSCII→Unicode "
    "conversion has dropped the combining vowel signs. 11,260 of 12,190 pseudo-documents "
    "(92.4%) fail the quality gate. This matters more than ordinary noise, because the "
    "deleted signs ARE the slot contrasts — -ஆன்/-ஆள்/-ஆர் collapse toward -அன்/-அள்/-அர். "
    "The 113 surviving documents are reported but are far too small to carry a prior.",
    "**IruMozhi cannot supply the Literary/Spoken prior split the study needed.** Its "
    "Literary side is Tamil script but both Spoken columns are ROMANISED, so no "
    "Tamil-script token count is possible from it. Sangraha's `speech` slice (transcribed "
    "audio, 1,228 documents / 419k tokens) is the substitute register control.",
    "**Not obtained: OSCAR 23.01** (HF `gated: manual`, needs human approval), **CulturaX** "
    "(HF `gated: auto`, needs an authenticated token; none was available on the analysis machine), "
    "**mC4 Tamil** (ungated but ~100 GB and shares Common Crawl provenance with CC-100, so "
    "it buys volume not independence), **IndicCorp v2** (ungated but superseded by Sangraha "
    "from the same lab, with a clearer licence).",
    "**The counting rule still admits lexical homographs.** The FST decides verb-hood and "
    "finiteness and a PNG label must be BOUND to the matching surface suffix, which removes "
    "the guesser's hallucinated nouns (பாக்கித்தான், சுல்தான்) and, via the noun-lexicon "
    "veto, கால்வாய் and ஆண்டாள். What survives is genuine ambiguity no morphology can "
    "resolve — வருவாய் is both 'you will come' and the noun 'revenue'. The `surface_only` "
    "variant bounds how much the gate is doing; the top-forms table shows exactly what is "
    "being counted.",
    "**The verb-borne honorificity slot is under-powered in written Tamil.** After the "
    "noun veto, Tamil Wikipedia has ~1.5k 2nd-person finite verb tokens in 30M — "
    "encyclopaedic prose addresses nobody. Any DFR on this slot rests on Sangraha and "
    "CC-100, not on Wikipedia.",
    "**The pooled interval is the UNION of per-corpus intervals**, not a bootstrap over the "
    "merged document set. Unioning can only widen, which is the safe direction when the "
    "corpora are not a random sample of 'Tamil'.",
    "**Design effects are large** (DEFF up to ~23 on the small corpora, ~5–20 on the large "
    "ones). That is why the intervals are wide, and it is the correct behaviour: forms "
    "clump by document.",
    "**UD_Tamil-TTB was deliberately not indexed.** CC BY-NC-SA 3.0; DECISIONS D-4 excludes "
    "it from the release, and an index file on disk is one `git add` from a licence "
    "violation. UD_Tamil-MWTT (CC BY-SA 4.0) serves the sanity-check role — and at 1,984 "
    "tokens it is a sanity check, not a frequency source.",
]

if __name__ == "__main__":
    main()
