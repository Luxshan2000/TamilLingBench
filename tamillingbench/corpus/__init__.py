# -*- coding: utf-8 -*-
"""Tamil corpus attestation + frequency resource (form attestation and the DFR prior).

Two consumers, one index:

  * **Attestation** — `FormIndex.frequency()` / `FormIndex.kwic()` answer "does this form
    actually occur in real Tamil, how often, and in what sentences?" for curation and for
    naturalness grounding (items *inspired by attested corpus sentences*).
  * **The DFR prior** — `priors.build_prior()` produces the training-frequency anchor that
    model output skew is measured against.

Everything that touches a Tamil string goes through `text.canonicalize`, which is a thin,
asserted wrapper over `tamillingbench.morph.normalize.normalize`. There must be one
shared canonicalization entry point between the hypothesis path and the corpus-counting
path; `text.assert_shared_canonicalization()` is that assertion.
"""
from .text import canonicalize, tokenize, sentences, is_tamil_line, assert_shared_canonicalization
from .index import FormIndex, KwicHit
from .sources import SOURCES, Source

__all__ = [
    "canonicalize", "tokenize", "sentences", "is_tamil_line",
    "assert_shared_canonicalization", "FormIndex", "KwicHit", "SOURCES", "Source",
]
