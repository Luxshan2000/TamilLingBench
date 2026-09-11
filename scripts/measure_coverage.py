#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the checker coverage table from the UD files.

ACCEPTANCE GATE: exactly 264 unique TTB-test VERB/AUX forms,
38.6% lexicon-only, 64.0% with the guesser. Any drift is a bug in normalization or union
construction, not a new result.

⚠ LICENCE. UD_Tamil-TTB is CC BY-NC-SA 3.0. This script reads it to compute a PERCENTAGE.
No TTB text is copied into any released artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tamillingbench.morph import MorphChecker            # noqa: E402
from tamillingbench.morph.normalize import normalize     # noqa: E402

CORPORA = {
    "ttb-test": "data/ud/UD_Tamil-TTB/ta_ttb-ud-test.conllu",
    "ttb-train": "data/ud/UD_Tamil-TTB/ta_ttb-ud-train.conllu",
    "ttb-dev": "data/ud/UD_Tamil-TTB/ta_ttb-ud-dev.conllu",
    "mwtt-test": "data/ud/UD_Tamil-MWTT/ta_mwtt-ud-test.conllu",
}


def read_conllu(path: Path):
    """Yield (form, upos, feats) for real tokens only.

    Multiword-range lines (`1-2`) and empty nodes (`1.1`) are skipped: counting them
    would double-count surface material and inflate every denominator.
    """
    for line in path.open(encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) == 10 and "-" not in f[0] and "." not in f[0]:
            yield f[1], f[3], f[5]


def unique_verb_forms(path: Path) -> list[str]:
    seen, out = set(), []
    for form, upos, _ in read_conllu(path):
        if upos in ("VERB", "AUX"):
            n = normalize(form, reject_latin=False)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
    return out


def measure(checker: MorphChecker, forms: list[str]) -> dict:
    # lexicon-only pass
    lex_only = MorphChecker(use_guesser=False, use_fallback=False, use_overrides=False)
    lex_res = lex_only.analyse_batch(forms)
    lex_hit = {f for f, r in zip(forms, lex_res) if r.ok}
    lex_only.close()

    # + guesser (still no fallback/override: this is the raw FST number the proposal cites)
    fst = MorphChecker(use_guesser=True, use_fallback=False, use_overrides=False)
    fst_res = fst.analyse_batch(forms)
    fst_hit = {f for f, r in zip(forms, fst_res) if r.ok}
    fst.close()

    # + fallback + override (what the benchmark actually runs)
    full_res = checker.analyse_batch(forms)
    full_hit = {f for f, r in zip(forms, full_res) if r.ok}

    misses = [f for f in forms if f not in fst_hit]
    artifacts = [f for f in misses if checker.is_artifact_token(f)]

    n = len(forms)
    n_corr = n - len(artifacts)
    corrected = len([f for f in fst_hit if f not in artifacts])
    return {
        "unique_forms": n,
        "lexicon_only": len(lex_hit),
        "lexicon_only_pct": round(100 * len(lex_hit) / n, 1),
        "with_guesser": len(fst_hit),
        "with_guesser_pct": round(100 * len(fst_hit) / n, 1),
        "with_fallback": len(full_hit),
        "with_fallback_pct": round(100 * len(full_hit) / n, 1),
        "misses": len(misses),
        "artifact_tokens": sorted(artifacts),
        "n_artifact_tokens": len(artifacts),
        "artifact_corrected_denominator": n_corr,
        "artifact_corrected_pct": round(100 * corrected / n_corr, 1) if n_corr else 0.0,
        "miss_list": sorted(misses),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/coverage.json")
    # TTB is CC BY-NC-SA 3.0. Anything that QUOTES TTB tokens goes under results/ttb-derived/,
    # which is excluded from the release by scripts/check_release_licence.py. The boundary is
    # structural, not a convention someone has to remember.
    ap.add_argument("--artifacts", default="results/ttb-derived/ttb-artifact-tokens.txt")
    args = ap.parse_args()

    checker = MorphChecker()
    out = {}
    for name, rel in CORPORA.items():
        path = ROOT / rel
        if not path.exists():
            print(f"SKIP {name}: {path} missing")
            continue
        forms = unique_verb_forms(path)
        out[name] = measure(checker, forms)
        r = out[name]
        print(f"{name:11s} unique={r['unique_forms']:4d}  "
              f"lexicon={r['lexicon_only']:4d} ({r['lexicon_only_pct']:4.1f}%)  "
              f"+guesser={r['with_guesser']:4d} ({r['with_guesser_pct']:4.1f}%)  "
              f"+fallback={r['with_fallback']:4d} ({r['with_fallback_pct']:4.1f}%)  "
              f"artifacts={r['n_artifact_tokens']:2d} -> corrected {r['artifact_corrected_pct']:4.1f}%")
    checker.close()

    (ROOT / args.json).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "results/ttb-derived").mkdir(parents=True, exist_ok=True)

    # Split: aggregate counts are facts about coverage and are releasable; the form LISTS
    # quote treebank text and are not (for TTB).
    QUOTES = ("miss_list", "artifact_tokens")
    public = {k: {kk: vv for kk, vv in v.items() if kk not in QUOTES} for k, v in out.items()}
    with open(ROOT / args.json, "w", encoding="utf-8") as fh:
        json.dump(public, fh, ensure_ascii=False, indent=2, sort_keys=True)
    with open(ROOT / "results/ttb-derived/coverage-forms.json", "w", encoding="utf-8") as fh:
        json.dump({k: {kk: v[kk] for kk in QUOTES} for k, v in out.items()},
                  fh, ensure_ascii=False, indent=2, sort_keys=True)

    ttb = out.get("ttb-test")
    if ttb:
        (ROOT / args.artifacts).parent.mkdir(parents=True, exist_ok=True)
        with open(ROOT / args.artifacts, "w", encoding="utf-8") as fh:
            fh.write("# TTB-test tokenization artifacts excluded from the coverage denominator\n")
            fh.write("# TTB's clitic splitting strands a sandhi consonant on the\n")
            fh.write("# left token, producing a non-word. These are NOT analyser failures.\n")
            fh.write(f"# n = {ttb['n_artifact_tokens']}\n")
            fh.write("# column 2: does dropping the stranded consonant recover a form the\n")
            fh.write("#           FST can analyse? (auditability only — NOT part of the test)\n")
            chk = MorphChecker()
            for f in ttb["artifact_tokens"]:
                fh.write(f"{f}\t{'restored-ok' if chk.restored_form_analyses(f) else 'restored-also-unanalysed'}\n")
            chk.close()

    # --- acceptance gate
    ok = True
    if ttb:
        for field, want in (("unique_forms", 264), ("lexicon_only_pct", 38.6),
                            ("with_guesser_pct", 64.0)):
            got = ttb[field]
            flag = "PASS" if got == want else "FAIL"
            if got != want:
                ok = False
            print(f"  {flag}  ttb-test {field}: got {got}, expect {want}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
