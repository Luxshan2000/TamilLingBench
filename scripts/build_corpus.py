#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the Tamil corpus indexes.

    python scripts/build_corpus.py tawiki opensubs sangraha cc100 ud_mwtt irumozhi

Each corpus becomes one SQLite index under `data/corpus/index/<key>.sqlite`. Raw dumps
under `data/corpus/raw/` and the indexes are BOTH gitignored — the release ships the
derived `results/corpus-priors.json`, never the corpora.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tamillingbench.corpus import readers, slots            # noqa: E402
from tamillingbench.corpus.index import IndexBuilder        # noqa: E402
from tamillingbench.corpus.sources import SOURCES           # noqa: E402
from tamillingbench.corpus.wiki import iter_pages           # noqa: E402

RAW = ROOT / "data" / "corpus" / "raw"
INDEX = ROOT / "data" / "corpus" / "index"

#: `every` = systematic document sampling rate. Chosen per corpus so that each contributes
#: a comparable token volume without any single download stalling the build. Sampling is
#: SYSTEMATIC over documents, never a prefix: dumps are ordered by page id / crawl order,
#: and both correlate with topic and age.
PLAN = {
    "tawiki":   dict(every=1,  store_sentences=True,  full_types=True),
    "opensubs": dict(every=1,  store_sentences=True,  full_types=True),
    "sangraha": dict(every=3,  store_sentences=True,  full_types=True),
    # The `speech` slice of Sangraha — transcribed audio. This is the register control
    # (OPUS OpenSubtitles, the obvious candidate, is encoding-damaged; see sources.py).
    "sangraha_speech": dict(every=1, store_sentences=True, full_types=True),
    # CC-100: text may not ship (licence unspecified) and 12 GB of it is not needed for a
    # rate estimate. Candidate types only, no sentence store.
    "cc100":    dict(every=8,  store_sentences=False, full_types=False),
    "ud_mwtt":  dict(every=1,  store_sentences=True,  full_types=True),
    "irumozhi": dict(every=1,  store_sentences=True,  full_types=True),
}


def documents(key: str, every: int):
    if key == "tawiki":
        return iter_pages(RAW / "tawiki-latest-pages-articles.xml.bz2", every=every)
    if key == "opensubs":
        return readers.iter_opus_mono(RAW / "opensubtitles-v2024-ta.txt.gz", every=every)
    if key in ("sangraha", "sangraha_speech"):
        paths = sorted(RAW.glob("sangraha-verified-tam-*.parquet"))
        if not paths:
            raise SystemExit("no sangraha shards in data/corpus/raw/")
        types = ("speech",) if key == "sangraha_speech" else None
        return readers.iter_sangraha(paths, every=every, types=types)
    if key == "cc100":
        return readers.iter_cc100(RAW / "cc100-ta.txt.xz", every=every)
    if key == "ud_mwtt":
        return readers.iter_conllu_text(ROOT / "data" / "ud" / "UD_Tamil-MWTT")
    if key == "irumozhi":
        return readers.iter_irumozhi_literary(RAW / "irumozhi-regdataset.csv")
    raise SystemExit(f"unknown corpus {key!r}")


def build(key: str, limit: int | None = None) -> dict:
    cfg = PLAN[key]
    src = SOURCES[key]
    out = INDEX / f"{key}.sqlite"
    b = IndexBuilder(out, key,
                     store_sentences=cfg["store_sentences"],
                     full_types=cfg["full_types"],
                     candidate=slots.candidate,
                     min_doc_tokens=5 if key in ("ud_mwtt", "irumozhi") else 20).open()
    t0 = time.time()
    n = 0
    for doc in documents(key, cfg["every"]):
        b.add(doc)
        n += 1
        if n % 20_000 == 0:
            print(f"  [{key}] {n:>9,} docs  {b.n_tokens:>12,} tok  "
                  f"{len(b._forms):>9,} types  {time.time()-t0:6.0f}s", flush=True)
        if limit and n >= limit:
            break
    b.close(meta={
        "source_key": key,
        "source_name": src.name,
        "source_url": src.url,
        "licence": src.licence,
        "licence_checked": src.licence_checked,
        "register": src.register,
        "release_ok": src.release_ok,
        "derived_ok": src.derived_ok,
        "doc_sample_every": cfg["every"],
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - t0, 1),
    })
    stats = {"corpus": key, "docs_seen": n, "n_docs": b.n_docs, "n_tokens": b.n_tokens,
             "n_types": len(b._forms), "n_sents": b.n_sents,
             "dropped_short": b.n_docs_dropped_short, "dropped_lang": b.n_docs_dropped_lang,
             "dropped_quality": b.n_docs_dropped_quality,
             "lines_deduped": b.n_lines_dedup, "seconds": round(time.time() - t0, 1),
             "path": str(out.relative_to(ROOT)),
             "size_mb": round(out.stat().st_size / 1e6, 1)}
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+", choices=sorted(PLAN))
    ap.add_argument("--limit", type=int, default=None, help="max documents (smoke tests)")
    a = ap.parse_args()
    INDEX.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for key in a.corpora:
        print(f"== {key}", flush=True)
        all_stats.append(build(key, a.limit))
    log = INDEX / "build-log.json"
    prev = json.loads(log.read_text()) if log.exists() else []
    prev = [p for p in prev if p["corpus"] not in {s["corpus"] for s in all_stats}]
    log.write_text(json.dumps(prev + all_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
