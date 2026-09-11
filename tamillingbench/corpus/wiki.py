# -*- coding: utf-8 -*-
"""Tamil Wikipedia dump → plain text.

`mwparserfromhell.strip_code()` alone is not enough: it leaves table pipe-syntax, reference
bodies, infobox residue and `[[படிமம்:…]]` file captions in the output, all of which are
non-running-text and would pollute a frequency count. The pre-strip below removes those
constructs *before* the parser runs, which is both more correct and much faster (tables and
refs are the expensive things to parse).

Verified behaviours, not assumed:
  * `<ns>0</ns>` filters out Category/Template/Help/Portal namespaces.
  * `<redirect .../>` pages carry a stub body and must be skipped or the same lead sentence
    is counted many times.
"""
from __future__ import annotations

import bz2
import re
from pathlib import Path
from typing import Iterator

import mwparserfromhell

from .index import Document

_NS = "{http://www.mediawiki.org/xml/export-0.11/}"

_RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RE_REF_PAIR = re.compile(r"<ref[^>/]*>.*?</ref>", re.S | re.I)
_RE_REF_SELF = re.compile(r"<ref[^>]*/>", re.I)
_RE_TAGGED = re.compile(r"<(math|code|syntaxhighlight|source|gallery|timeline|score|"
                        r"imagemap|nowiki|pre)[^>]*>.*?</\1>", re.S | re.I)
_RE_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
#: Wikilink namespaces whose *caption* is not running text. Matched against the link
#: title by `_drop_media_links`, which is bracket-aware — a regex is not, and Tamil file
#: captions routinely nest `[[…]]` links inside themselves, which is exactly how the naive
#: `\[\[File:.*?\]\]` pattern leaves a stray `]]` behind (observed on கட்டடக்கலை).
_MEDIA_NS = ("படிமம்", "கோப்பு", "file", "image", "media", "படிமம",
             "பகுப்பு", "category", "வார்ப்புரு", "template")
_RE_HEADING = re.compile(r"^=+.*?=+\s*$", re.M)
_RE_LISTMARK = re.compile(r"^[*#:;]+", re.M)
_RE_HTMLTAG = re.compile(r"<[^>]{1,200}>")
_RE_WS = re.compile(r"[ \t]+")
_RE_NL = re.compile(r"\n{2,}")

#: Sections after which the running text stops being prose.
_STOP_SECTIONS = ("மேற்கோள்", "வெளி இணைப்ப", "இவற்றையும்", "ஆதார", "உசாத்துணை",
                  "மேலும் படிக்க", "References", "External links", "See also")


def _drop_media_links(code) -> None:
    """Remove File/Image/Category wikilinks, captions and all, using the parser's own
    bracket matching. Innermost-first so a nested link is not orphaned."""
    links = code.filter_wikilinks(recursive=True)
    for link in reversed(links):
        title = str(link.title).strip().lstrip(":").lower()
        ns = title.split(":", 1)[0].strip() if ":" in title else ""
        if ns and ns in _MEDIA_NS:
            try:
                code.remove(link)
            except ValueError:
                pass


def strip_wikitext(raw: str) -> str:
    """Wikitext → plain running text."""
    s = _RE_COMMENT.sub(" ", raw)
    s = _RE_REF_PAIR.sub(" ", s)
    s = _RE_REF_SELF.sub(" ", s)
    s = _RE_TAGGED.sub(" ", s)
    # Tables nest; three passes clears essentially all real cases.
    for _ in range(3):
        s2 = _RE_TABLE.sub(" ", s)
        if s2 == s:
            break
        s = s2
    # Cut the reference/external-link tail before parsing.
    for marker in _STOP_SECTIONS:
        m = re.search(r"^=+\s*" + re.escape(marker) + r".*?=+\s*$", s, re.M)
        if m:
            s = s[:m.start()]
    s = _RE_HEADING.sub("\n", s)
    try:
        code = mwparserfromhell.parse(s)
        _drop_media_links(code)
        s = code.strip_code(normalize=True, collapse=True)
    except Exception:
        pass
    s = _RE_HTMLTAG.sub(" ", s)
    s = _RE_LISTMARK.sub("", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
    s = _RE_WS.sub(" ", s)
    s = _RE_NL.sub("\n", s)
    return s.strip()


def iter_pages(dump_path: str | Path, limit: int | None = None,
               every: int = 1) -> Iterator[Document]:
    """Stream article-namespace, non-redirect pages out of a `pages-articles` dump.

    `every` takes a systematic sample (every k-th surviving article). Systematic sampling
    keeps the *document* as the sampling unit, which is what the cluster bootstrap
    requires; a prefix would not, because dumps are ordered by page id and
    page id correlates with article age and topic.
    """
    import xml.etree.ElementTree as ET

    kept = 0
    seen = 0
    with bz2.open(dump_path, "rb") as fh:
        ctx = ET.iterparse(fh, events=("end",))
        for _, elem in ctx:
            if not elem.tag.endswith("}page"):
                continue
            ns = elem.findtext(f"{_NS}ns")
            redirect = elem.find(f"{_NS}redirect")
            title = elem.findtext(f"{_NS}title") or ""
            rev = elem.find(f"{_NS}revision")
            text = rev.findtext(f"{_NS}text") if rev is not None else None
            elem.clear()
            if ns != "0" or redirect is not None or not text:
                continue
            seen += 1
            if (seen - 1) % every:
                continue
            body = strip_wikitext(text)
            if not body:
                continue
            kept += 1
            yield Document(ref=title, text=body)
            if limit and kept >= limit:
                return
