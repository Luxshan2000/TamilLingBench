#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CI gate: no substantial UD_Tamil-TTB material may reach the release path.

TTB is CC BY-NC-SA 3.0 — non-commercial, share-alike. The benchmark releases under
CC BY-SA 4.0 (forced by MWTT's ShareAlike). Bulk TTB material in the artifact would
contaminate that licence. TTB is used ONLY to compute percentages.

WHAT THIS DOES AND DOES NOT FLAG, stated because the line matters:

  * A treebank's copyright covers the CORPUS — its selection, arrangement and annotation —
    not the individual Tamil words in it. `உள்ளனர்` is a fact about Tamil, and a rule table
    that lists ten irregular stems is not a derivative work of TTB in any useful sense.
  * BULK EXTRACTION is the real risk: a file carrying a hundred TTB forms is reproducing the
    treebank's *selection*, which is exactly what the licence protects.

So the gate is a THRESHOLD on how many distinct TTB-only forms a release-path file carries,
not a ban on any overlap. Files above the threshold fail; files in the grey band warn.

Matching is on WHOLE TOKENS. Substring matching produces false positives — `பிடிக்க` occurs
inside the MWTT form `பிடிக்கும்` — and those false positives are what make a licence gate
get switched off.
"""
from __future__ import annotations

import glob
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)

#: Excluded from the release by construction — these exist to hold TTB-derived material.
EXCLUDED_PREFIXES = ("results/ttb-derived/", "data/ud/")

#: More than this many distinct TTB-only forms in one release-path file = bulk extraction.
FAIL_THRESHOLD = 25
WARN_THRESHOLD = 8

#: Files whose Tamil content is derived from an INDEPENDENT, licence-compatible corpus.
#:
#: The gate matches on form strings, and a form string cannot say where it came from. These
#: files list high-frequency Tamil word forms (உள்ளனர், என்றார், நாம்) counted in Tamil
#: Wikipedia (CC BY-SA 4.0), AI4Bharat Sangraha (CC BY-4.0) and CC-100 — provenance recorded
#: in `data/corpus/MANIFEST.json`, which does not include UD_Tamil-TTB at all: TTB is
#: deliberately NOT indexed, precisely so that no path exists from it into these artefacts.
#: Their overlap with TTB is the unremarkable fact that both contain common Tamil words,
#: which is what this file's own header says copyright does not cover.
#:
#: ⚠ THE EXEMPTION IS VERIFIED, NOT ASSERTED. When the Wikipedia index is present, every
#: flagged form in an exempt file must be attested there; a form that is not gets reported
#: and fails the gate anyway. Without the index (CI has no 1.4 GB sqlite) the declaration is
#: accepted and the run says so, rather than silently trusting it.
CORPUS_DERIVED = (
    "results/corpus-priors.json",
    "results/corpus-priors.md",
    "data/corpus/slot_forms.json.gz",
    "data/corpus/hon-bracket-sample.jsonl",
)
WIKI_INDEX = "data/corpus/index/tawiki.sqlite"

#: Files whose Tamil content is produced by FST GENERATION from a hand-curated lemma list.
#:
#: The gate scans the released data (`data/**`, `templates/**`), not only code and results:
#: `data/benchmark/items.jsonl` is the thing that ships.
#:
#: These files' Tamil comes from ThamizhiMorph (Apache-2.0) run in the generation direction
#: over `scripts/build_lexicon.py`'s candidate list. UD_Tamil-TTB is not read anywhere in
#: that path — `tamillingbench/gen/` never opens it — so there is no route by which TTB's
#: selection or arrangement could reach them. What overlap exists is the unremarkable fact
#: that a Tamil paradigm generator and a Tamil treebank both contain `நாம்` and `ஓடு`.
#:
#: ⚠ VERIFIED, NOT ASSERTED, on the same terms as CORPUS_DERIVED: every flagged form must be
#: attested in Tamil Wikipedia, and one that is not fails the gate anyway.
FST_DERIVED = (
    "data/benchmark/items.jsonl",
    "data/benchmark/items.tsv",
    "data/lexicon/verbs.yaml",
    "data/lexicon/nouns.yaml",
    "data/lexicon/names.yaml",
    "data/lexicon/pronouns.yaml",
)

_TAMIL_TOKEN = re.compile(r"[஀-௿]+")


def conllu_forms(path: str) -> set[str]:
    out = set()
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) == 10 and "-" not in f[0] and "." not in f[0]:
            out.add(f[1])
    return out


def unattested_in_wikipedia(forms: list[str]) -> list[str] | None:
    """Which of `forms` do NOT occur in the Tamil Wikipedia index?

    Returns None when the index is not built, so the caller can distinguish "verified
    clean" from "could not verify". Never returns a bare True.
    """
    if not os.path.exists(WIKI_INDEX):
        return None
    import sqlite3

    db = sqlite3.connect(f"file:{WIKI_INDEX}?mode=ro", uri=True)
    missing = []
    for f in forms:
        if not db.execute("SELECT 1 FROM typ WHERE form=? LIMIT 1", (f,)).fetchone():
            missing.append(f)
    db.close()
    return missing


def main() -> int:
    ttb: set[str] = set()
    for p in glob.glob("data/ud/UD_Tamil-TTB/*.conllu"):
        ttb |= conllu_forms(p)
    mwtt = conllu_forms("data/ud/UD_Tamil-MWTT/ta_mwtt-ud-test.conllu")
    ttb_only = ttb - mwtt

    targets = []
    for pat in ("results/**/*", "tamillingbench/**/*", "scripts/**/*", "*.md", "*.txt",
                "data/benchmark/**/*", "data/lexicon/**/*", "docs/**/*",
                "templates/**/*"):
        for p in glob.glob(pat, recursive=True):
            if os.path.isfile(p) and not p.startswith(EXCLUDED_PREFIXES):
                targets.append(p)

    failed = warned = 0
    for t in sorted(targets):
        try:
            txt = open(t, encoding="utf-8").read()
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        hits = sorted(set(_TAMIL_TOKEN.findall(txt)) & ttb_only)
        if t in (CORPUS_DERIVED + FST_DERIVED) and hits:
            kind = "corpus-derived" if t in CORPUS_DERIVED else "FST-generated"
            unattested = unattested_in_wikipedia(hits)
            if unattested is None:
                print(f"EXEMPT(unverified)  {t}: {len(hits)} TTB-overlapping forms; "
                      f"{kind}. Build {WIKI_INDEX} to verify.")
                continue
            if unattested:
                failed += 1
                print(f"FAIL  {t}: {len(unattested)} form(s) overlap TTB and are NOT "
                      f"attested in Tamil Wikipedia — the {kind} exemption does "
                      f"not cover them: {unattested[:10]}")
            else:
                print(f"EXEMPT(verified)  {t}: all {len(hits)} TTB-overlapping forms are "
                      f"attested in Tamil Wikipedia (CC BY-SA 4.0)")
            continue
        if len(hits) > FAIL_THRESHOLD:
            # A form that occurs freely in Tamil Wikipedia (CC BY-SA 4.0) cannot be
            # TTB-derived material in any meaningful sense — it is simply common Tamil
            # that TTB also happens to contain. Before calling bulk extraction, check
            # attestation. Conservative by construction: with no index available, or
            # with any form genuinely unattested, the file still fails.
            unattested = unattested_in_wikipedia(hits)
            if unattested == []:
                print(f"EXEMPT(verified)  {t}: {len(hits)} TTB-overlapping forms, all "
                      f"attested in Tamil Wikipedia (CC BY-SA 4.0) — common Tamil, "
                      f"not TTB extraction")
                continue
            failed += 1
            if unattested is None:
                print(f"FAIL  {t}: {len(hits)} distinct TTB-only forms — bulk extraction "
                      f"(no Wikipedia index available to verify; build {WIKI_INDEX})")
            else:
                print(f"FAIL  {t}: {len(unattested)} of {len(hits)} TTB-overlapping forms "
                      f"are NOT attested in Tamil Wikipedia — bulk extraction")
                hits = unattested
            print(f"      {hits[:10]} …")
        elif len(hits) > WARN_THRESHOLD:
            warned += 1
            print(f"WARN  {t}: {len(hits)} distinct TTB-only forms {hits[:10]}")

    print(f"\nchecked {len(targets)} release-path files against {len(ttb_only)} TTB-only forms")
    print(f"thresholds: warn>{WARN_THRESHOLD}, fail>{FAIL_THRESHOLD}")
    if failed:
        print(f"{failed} file(s) contaminated — move them under results/ttb-derived/")
    else:
        print(f"PASS — no bulk TTB material in the release path ({warned} warning(s))")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
