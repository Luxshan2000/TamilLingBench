# -*- coding: utf-8 -*-
"""Corpus registry — provenance, licence and release status for every text we index.

⚠ `scripts/check_release_licence.py` enforces a strict release-licence gate and
`DECISIONS.md` D-4 excludes **UD_Tamil-TTB (CC BY-NC-SA 3.0)** from the release entirely.
Every entry therefore carries `release_ok`, and `scripts/check_release_licence.py` reads
this table. A corpus may be *indexed for analysis* and still be forbidden from the release;
the two flags are separate on purpose.

Nothing in this file is inferred. Each `licence` is what the distributor states, and
`licence_checked` records the date it was read.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    url: str
    licence: str
    licence_checked: str
    register: str            # what kind of Tamil this is — the register confound (§ IruMozhi)
    role: str                # why it is in the study
    release_ok: bool         # may derived text ship in the public release?
    #: `derived_ok` = may *counts/statistics* derived from it ship, even if text may not.
    derived_ok: bool = True
    obtained: bool = True
    notes: str = ""
    size_note: str = ""
    citation: str = ""


SOURCES: dict[str, Source] = {}


def _add(s: Source) -> Source:
    SOURCES[s.key] = s
    return s


# --------------------------------------------------------------------- obtained & indexed

_add(Source(
    key="tawiki",
    name="Tamil Wikipedia (tawiki-20260804 pages-articles)",
    url="https://dumps.wikimedia.org/tawiki/latest/tawiki-latest-pages-articles.xml.bz2",
    licence="CC BY-SA 4.0 + GFDL (Wikimedia dual licence)",
    licence_checked="2026-08-07",
    register="Literary / encyclopaedic (செந்தமிழ்)",
    role=("Primary attestation source and the most defensible general frequency anchor: "
          "Wikipedia is plausibly in every open-weight model's pretraining mix."),
    release_ok=True,
    size_note="274.9 MB bz2, dump dated 2026-08-04",
    citation="Wikimedia Foundation, Tamil Wikipedia database dump, 2026-08-04",
    notes=("SHARE-ALIKE. Derived text is CC BY-SA 4.0, which is compatible with D-4's "
           "CC BY-SA 4.0 data licence (already forced by UD_Tamil-MWTT)."),
))

_add(Source(
    key="opensubs",
    name="OPUS OpenSubtitles v2024, Tamil monolingual",
    url="https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/ta.txt.gz",
    licence="OPUS redistribution of OpenSubtitles.org data — no explicit open licence; "
            "OPUS states the data is for research use",
    licence_checked="2026-08-07",
    register="Dialogue / closest available proxy for Spoken Tamil in Tamil script",
    role=("The register control. IruMozhi's Spoken half is romanised (see `irumozhi`), so "
          "subtitles are the only Tamil-script counterweight to Wikipedia's Literary skew."),
    release_ok=False,
    derived_ok=True,
    size_note="29.6 MB gz",
    citation="Lison & Tiedemann, LREC 2016; OPUS OpenSubtitles v2024",
    notes=("⚠ TEXT MUST NOT SHIP. Aggregate counts only. Subtitle text is third-party "
           "copyrighted; OPUS redistributes it without granting a sublicence.\n"
           "⛔ AND, MEASURED 2026-08-08: THE TAMIL SIDE IS ENCODING-DAMAGED. Median "
           "vowel-sign/consonant ratio is 0.014 against 0.432 for Tamil Wikipedia and "
           "0.45 for Sangraha — a legacy TSCII/Bamini→Unicode conversion has dropped the "
           "combining vowel signs (`அனைத்து பறவைகள் இறக்க` → `அனத்தபறவகள்இறக`). This is "
           "disqualifying rather than merely noisy: the deleted signs ARE our slot "
           "contrasts, so -ஆன்/-ஆள்/-ஆர் all collapse toward -அன்/-அள்/-அர். Excluded "
           "from every prior. Sangraha's `speech` slice replaces it as the register "
           "control."),
))

_add(Source(
    key="sangraha_speech",
    name="AI4Bharat Sangraha, verified Tamil — `speech` slice (transcribed audio)",
    url="https://huggingface.co/datasets/ai4bharat/sangraha/tree/main/verified/tam",
    licence="CC BY-4.0 (dataset card)",
    licence_checked="2026-08-07",
    register="Transcribed speech — the closest clean Tamil-script approach to Spoken Tamil",
    role=("THE REGISTER CONTROL. IruMozhi was the first choice for this; IruMozhi's Spoken "
          "side is romanised and OPUS OpenSubtitles is encoding-damaged, so this is the "
          "only usable non-written-register Tamil-script text obtained."),
    release_ok=True,
    size_note="~0.44% of Sangraha verified/tam documents",
    citation="Khan et al., IndicLLMSuite / Sangraha, ACL 2024",
    notes=("Small. Its role is a register CONTRAST, not volume; where n is too small the "
           "report says so rather than quoting a rate."),
))

_add(Source(
    key="sangraha",
    name="AI4Bharat Sangraha, verified Tamil (2 of 53 shards)",
    url="https://huggingface.co/datasets/ai4bharat/sangraha/tree/main/verified/tam",
    licence="CC BY-4.0 (dataset card)",
    licence_checked="2026-08-07",
    register="Mixed web / curated Indic — news, blogs, government, transcribed media",
    role=("Independent curated Indic corpus. AI4Bharat corpora are the stated training data "
          "of the IndicTrans2 arm's ancestry, so this is the nearest thing to a Tier-1 prior "
          "we can compute locally."),
    release_ok=True,
    size_note="53 shards × ~357 MB parquet ≈ 19 GB; 2 shards sampled",
    citation="Khan et al., IndicLLMSuite / Sangraha, ACL 2024",
    notes="Not gated on the Hub; anonymous download works.",
))

_add(Source(
    key="cc100",
    name="CC-100 Tamil (statmt.org)",
    url="https://data.statmt.org/cc-100/ta.txt.xz",
    licence="Unspecified — derived from Common Crawl; statmt.org states 'the data is "
            "released as-is'. HF mirror `statmt/cc100` reports licence 'unknown'.",
    licence_checked="2026-08-07",
    register="Raw web crawl",
    role=("Second independent web crawl, and the one most likely to overlap actual LLM "
          "pretraining mixes: CC-100 is the XLM-R corpus and a documented CCNet product."),
    release_ok=False,
    derived_ok=True,
    size_note="1.38 GB xz (~12 GB plain text); document-sampled, not fully indexed",
    citation="Conneau et al., ACL 2020; Wenzek et al., LREC 2020",
    notes="⚠ TEXT MUST NOT SHIP — licence unspecified. Aggregate counts only.",
))

_add(Source(
    key="ud_mwtt",
    name="UD_Tamil-MWTT r2.18",
    url="https://github.com/UniversalDependencies/UD_Tamil-MWTT",
    licence="CC BY-SA 4.0",
    licence_checked="2026-08-07",
    register="Constructed grammar examples (Lehmann 1993)",
    role="Gold-annotated, tiny. Sanity check on the counting rule, not a frequency source.",
    release_ok=True,
    size_note="~500 sentences",
    citation="Krishnamurthy & Sarveswaran, TLT 2021; Lehmann 1993",
))

_add(Source(
    key="ud_ttb",
    name="UD_Tamil-TTB r2.18",
    url="https://github.com/UniversalDependencies/UD_Tamil-TTB",
    licence="CC BY-NC-SA 3.0",
    licence_checked="2026-08-07",
    register="Formal news",
    role="Gold-annotated, tiny. NOT INDEXED — see notes.",
    release_ok=False,
    derived_ok=False,
    size_note="~600 sentences",
    citation="Ramasamy & Žabokrtský, CICLing 2012",
    obtained=True,
    notes=("⛔ D-4: NON-COMMERCIAL. TTB is excluded from the release entirely and "
           "a CI leakage check enforces it. Nothing TTB-derived — not even counts — may "
           "enter the released artefacts, so it is DELIBERATELY NOT INDEXED here: an "
           "index file on disk is one `git add` away from becoming a licence violation. "
           "UD_Tamil-MWTT (CC BY-SA 4.0) serves the same sanity-check role."),
))

_add(Source(
    key="irumozhi",
    name="IruMozhi (aryaman/irumozhi)",
    url="https://huggingface.co/datasets/aryaman/irumozhi",
    licence="MIT",
    licence_checked="2026-08-07",
    register="Parallel Literary ↔ Spoken Tamil",
    role=("The planned register confound control. See `notes` — it cannot "
          "serve as a Spoken frequency source, and this is a negative result."),
    release_ok=True,
    size_note="500 rows",
    citation="Prasanna & Arora, 'IruMozhi', Findings of NAACL 2024",
    notes=("⛔ MEASURED, NOT ASSUMED: the Literary side is Tamil script but BOTH Spoken "
           "columns ('colloquial: annotator 1/2') are ROMANISED. IruMozhi therefore cannot "
           "supply Spoken-Tamil-script token counts, and the Literary/Spoken prior split "
           "the study needed is not computable from it. Used for the register argument "
           "and the citation only."),
))

# --------------------------------------------------------------------- investigated, not obtained

_add(Source(
    key="oscar",
    name="OSCAR 23.01, Tamil",
    url="https://huggingface.co/datasets/oscar-corpus/OSCAR-2301",
    licence="CC0-1.0 on the corpus metadata; underlying text is Common Crawl",
    licence_checked="2026-08-07",
    register="Raw web crawl",
    role="Would have been a third independent web crawl.",
    release_ok=False,
    obtained=False,
    notes=("NOT OBTAINED. HF API reports `gated: 'manual'` — access requires a human "
           "approval step on the dataset page, which cannot complete today. CC-100 fills "
           "the same role (independent crawl) without a gate."),
))

_add(Source(
    key="culturax",
    name="CulturaX, Tamil",
    url="https://huggingface.co/datasets/uonlp/CulturaX",
    licence="mC4 (ODC-By) + OSCAR terms, inherited",
    licence_checked="2026-08-07",
    register="Cleaned multilingual web (mC4 + OSCAR)",
    role="Would have been the cleaned-web comparator.",
    release_ok=False,
    obtained=False,
    notes=("NOT OBTAINED. HF API reports `gated: 'auto'` — requires an authenticated "
           "token with the terms accepted. No authenticated token was available on the "
           "analysis machine. Deferrable, not blocking: it "
           "is mC4+OSCAR, i.e. not independent of CC-100's provenance."),
))

_add(Source(
    key="mc4",
    name="mC4 Tamil (allenai/c4, multilingual config)",
    url="https://huggingface.co/datasets/allenai/c4",
    licence="ODC-By 1.0",
    licence_checked="2026-08-07",
    register="Cleaned web crawl",
    role="Ungated alternative to CulturaX.",
    release_ok=False,
    obtained=False,
    notes=("NOT OBTAINED — deliberately deprioritised. Ungated and downloadable, but the "
           "Tamil config is ~100 GB and shares Common Crawl provenance with CC-100, so it "
           "buys volume, not independence. Named here so the omission is a choice on "
           "record rather than an oversight."),
))

_add(Source(
    key="indiccorp2",
    name="AI4Bharat IndicCorp v2, Tamil",
    url="https://huggingface.co/datasets/ai4bharat/IndicCorpV2",
    licence="Not stated on the dataset card (`license` field absent)",
    licence_checked="2026-08-07",
    register="Indic news + web",
    role="Established AI4Bharat corpus; superseded for our purposes by Sangraha.",
    release_ok=False,
    obtained=False,
    notes=("NOT OBTAINED. Ungated and downloadable, but Sangraha is the same lab's newer "
           "and larger corpus, carries an explicit CC-BY-4.0, and was obtained. Indexing "
           "both would double the cost for a near-duplicate provenance."),
))

_add(Source(
    key="samanantar",
    name="Samanantar, en–ta parallel",
    url="https://huggingface.co/datasets/ai4bharat/samanantar",
    licence="CC BY-NC-4.0",
    licence_checked="2026-08-07",
    register="Parallel / translationese",
    role="Would be a Tier-1-adjacent prior: BPCC (IndicTrans2's training data) subsumes it.",
    release_ok=False,
    derived_ok=False,
    obtained=False,
    notes=("NOT OBTAINED, and there is a substantive reason beyond cost. (1) ⛔ "
           "NON-COMMERCIAL, so it is release-gated like TTB. (2) The Tamil side is "
           "*translated from English*, so its slot distribution reflects translator "
           "defaults — which is the very phenomenon under test. Using it as the DFR "
           "anchor would make the prior and the measurement circular. Recommended as a "
           "Tier-1 prior for the IndicTrans2 arm ONLY, where circularity is the point."),
))

_add(Source(
    key="tatoeba",
    name="OPUS Tatoeba v2023-04-12, Tamil monolingual",
    url="https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/ta.txt.gz",
    licence="CC BY 2.0 FR",
    licence_checked="2026-08-07",
    register="Constructed conversational sentences",
    role="Downloaded and inspected; too small to contribute a prior (8 KB gz).",
    release_ok=True,
    obtained=True,
    notes="Obtained but NOT indexed as a prior corpus — n is ~2 orders of magnitude too small.",
))


def release_blocked() -> list[Source]:
    """Sources whose *text* must never enter the release (licence gate)."""
    return [s for s in SOURCES.values() if not s.release_ok]


def derived_blocked() -> list[Source]:
    """Sources from which not even counts may be released."""
    return [s for s in SOURCES.values() if not s.derived_ok]
