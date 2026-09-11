# -*- coding: utf-8 -*-
"""Document iterators, one per acquired corpus.

Every reader yields `Document(ref, text)`. `ref` must identify the source unit precisely
enough to be quoted in a KWIC line's provenance, and the *document* must be the real
clustering unit — a subtitle file, a web page, a Wikipedia article — because the cluster
bootstrap resamples documents, not tokens.

⚠ The OPUS `mono` files and CC-100 are **line/paragraph** oriented, not document oriented.
For CC-100 the document boundary is a blank line (this is CCNet's own document separator);
for OPUS mono there is no document structure at all, so we synthesise pseudo-documents of
`OPUS_DOC_LINES` consecutive lines. That is a real weakening of the clustering assumption
and it is recorded in the report, not hidden.
"""
from __future__ import annotations

import gzip
import lzma
from pathlib import Path
from typing import Iterator

from .index import Document

#: Consecutive subtitle lines grouped into one pseudo-document. OPUS mono files preserve
#: source order, so a run of lines is usually from the same film.
OPUS_DOC_LINES = 200


def iter_opus_mono(path: str | Path, every: int = 1,
                   doc_lines: int = OPUS_DOC_LINES) -> Iterator[Document]:
    n = 0
    buf: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            buf.append(line.rstrip("\n"))
            if len(buf) >= doc_lines:
                if n % every == 0:
                    yield Document(ref=f"opus:{n}", text="\n".join(buf))
                n += 1
                buf = []
    if buf and n % every == 0:
        yield Document(ref=f"opus:{n}", text="\n".join(buf))


def iter_cc100(path: str | Path, every: int = 1,
               max_docs: int | None = None) -> Iterator[Document]:
    """CC-100 plain text: paragraphs separated by blank lines, documents likewise.

    statmt.org's `ta.txt.xz` uses a single blank line between paragraphs of the same
    document and the file is CCNet-ordered. There is no explicit document marker, so we
    treat a blank line as a document break — the conservative choice, since it makes
    clusters *smaller* and the bootstrap *wider*, never narrower.
    """
    n = 0
    out = 0
    buf: list[str] = []
    with lzma.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip():
                buf.append(line)
                continue
            if buf:
                if n % every == 0:
                    yield Document(ref=f"cc100:{n}", text="\n".join(buf))
                    out += 1
                    if max_docs and out >= max_docs:
                        return
                n += 1
                buf = []
    if buf and n % every == 0:
        yield Document(ref=f"cc100:{n}", text="\n".join(buf))


def iter_sangraha(paths: list[str | Path], every: int = 1,
                  types: tuple[str, ...] | None = None) -> Iterator[Document]:
    """Sangraha `verified/tam/data-*.parquet`. Columns: `doc_id`, `type`, `text`.

    `type` is one of `web` | `pdf` | `speech`. The `speech` slice is transcribed audio and
    is the only clean Tamil-script corpus in this project that is not written-register
    prose — it is what carries the register control after OPUS OpenSubtitles turned out to
    be encoding-damaged (see `sources.py`). Counted on shard 0: web 54,772 / pdf 4,967 /
    speech 261 documents per 60,000.
    """
    import pyarrow.parquet as pq

    n = 0
    for p in paths:
        pf = pq.ParquetFile(p)
        cols = set(pf.schema_arrow.names)
        want = [c for c in ("doc_id", "type", "text") if c in cols]
        for batch in pf.iter_batches(batch_size=2048, columns=want):
            d = batch.to_pydict()
            texts = d["text"]
            ids = d.get("doc_id")
            kinds = d.get("type")
            for i, t in enumerate(texts):
                kind = kinds[i] if kinds else "?"
                if types is not None and kind not in types:
                    continue
                if t and n % every == 0:
                    ref = str(ids[i]) if ids else f"{Path(p).stem}:{n}"
                    yield Document(ref=f"sangraha:{kind}:{ref}", text=t)
                n += 1


def iter_conllu_text(conllu_dir: str | Path) -> Iterator[Document]:
    """UD treebank → one Document per `# newdoc`/file, text from `# text =` lines."""
    conllu_dir = Path(conllu_dir)
    for f in sorted(conllu_dir.glob("*.conllu")):
        sents: list[str] = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("# text ="):
                sents.append(line.split("=", 1)[1].strip())
        if sents:
            yield Document(ref=f"ud:{f.name}", text="\n".join(sents))


def iter_irumozhi_literary(csv_path: str | Path) -> Iterator[Document]:
    """IruMozhi's Literary column only — the Spoken columns are romanised (see sources.py)."""
    import csv as _csv

    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(_csv.DictReader(fh))
    text = "\n".join(r["tamil"] for r in rows if r.get("tamil"))
    if text:
        yield Document(ref="irumozhi:literary", text=text)
