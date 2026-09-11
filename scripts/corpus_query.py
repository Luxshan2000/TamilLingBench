#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Attestation CLI — the curation/verification front door (Purpose 1).

    python scripts/corpus_query.py freq வந்தாள் நீங்கள் தாங்கள்
    python scripts/corpus_query.py kwic வந்தீர்கள் --limit 5
    python scripts/corpus_query.py check forms.txt          # one form per line, TSV out

`freq` answers "does this form occur in real Tamil, and how often"; `kwic` answers "in what
contexts". Both normalise the query through `tamillingbench.corpus.text.canonicalize`, so a
decomposed ெ+ா in a hand-typed template finds the composed corpus form.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tamillingbench.corpus.index import FormIndex          # noqa: E402
from tamillingbench.corpus.text import canonicalize        # noqa: E402

INDEX = ROOT / "data" / "corpus" / "index"
#: Only corpora whose *text* may be quoted. OpenSubtitles and CC-100 are counts-only.
QUOTABLE = ("tawiki", "sangraha", "sangraha_speech", "ud_mwtt", "irumozhi")


def open_all(only_quotable: bool = False) -> list[FormIndex]:
    keys = QUOTABLE if only_quotable else None
    out = []
    for p in sorted(INDEX.glob("*.sqlite")):
        if keys and p.stem not in keys:
            continue
        out.append(FormIndex(p))
    return out


def cmd_freq(args) -> None:
    idxs = open_all()
    rows = []
    for form in args.forms:
        f = canonicalize(form)
        row = {"form": f, "total": 0, "corpora": {}}
        for idx in idxs:
            fr = idx.frequency(f)
            row["corpora"][idx.corpus] = {
                "n": fr.n, "df": fr.df, "per_million": round(fr.per_million, 3)}
            row["total"] += fr.n
        row["attested"] = row["total"] > 0
        rows.append(row)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return
    names = [i.corpus for i in idxs]
    print("form\tattested\ttotal\t" + "\t".join(f"{n}(n/pm)" for n in names))
    for r in rows:
        cells = [f"{r['corpora'][n]['n']}/{r['corpora'][n]['per_million']:.2f}" for n in names]
        print(f"{r['form']}\t{r['attested']}\t{r['total']}\t" + "\t".join(cells))


def cmd_kwic(args) -> None:
    hits = []
    for idx in open_all(only_quotable=True):
        hits.extend(idx.kwic(args.form, limit=args.limit, window=args.window))
    if args.json:
        print(json.dumps([h.__dict__ for h in hits[:args.limit]], ensure_ascii=False,
                         indent=1))
        return
    for h in hits[:args.limit]:
        print(f"[{h.corpus}:{h.doc_ref}]\n   … {h.left}  «{h.match}»  {h.right} …")
    if not hits:
        print("no attested sentences found in the quotable corpora "
              f"({', '.join(QUOTABLE)})")


def cmd_check(args) -> None:
    idxs = open_all()
    forms = [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines()
             if l.strip()]
    print("form\ttotal_n\tn_corpora_attesting\tverdict")
    for form in forms:
        f = canonicalize(form)
        ns = {i.corpus: i.frequency(f).n for i in idxs}
        total = sum(ns.values())
        k = sum(v > 0 for v in ns.values())
        verdict = ("UNATTESTED" if total == 0 else
                   "SINGLE-CORPUS" if k == 1 else
                   "RARE" if total < 5 else "ATTESTED")
        print(f"{f}\t{total}\t{k}\t{verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freq"); f.add_argument("forms", nargs="+")
    f.add_argument("--json", action="store_true"); f.set_defaults(fn=cmd_freq)

    k = sub.add_parser("kwic"); k.add_argument("form")
    k.add_argument("--limit", type=int, default=10)
    k.add_argument("--window", type=int, default=8)
    k.add_argument("--json", action="store_true"); k.set_defaults(fn=cmd_kwic)

    c = sub.add_parser("check"); c.add_argument("file"); c.set_defaults(fn=cmd_check)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
