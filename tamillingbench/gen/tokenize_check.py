# -*- coding: utf-8 -*-
"""Tokenization pre-check, run before anything else is built.

A contrast pair the tokenizer cannot separate cleanly is a pair no probability-based metric
and no activation patch can use, so this runs *before* the interpretability subset is chosen
and its verdict decides membership.

Definitions, fixed here so `07`, `08` and `10` all use the same ones:

* **divergence index** `d` — the smallest index at which the *k* variant token sequences are
  not all equal. If a shorter sequence ends first and is a prefix of the others, that is
  **`prefix_containment`** and `d` is the shorter length.
* **decision position** — `d − 1`, the last *shared* token: the model has committed to
  everything before the contrast and has not yet emitted it. This is where `10` reads the
  residual stream. `d == 0` means the decision position is the final prompt token.
* **fertility** — tokens per Tamil orthographic (whitespace-delimited) word.

`prefix_containment` is a hard exclusion, not a warning. When one variant's tokens are a
strict prefix of another's, a sum-of-logprobs comparison is structurally biased toward the
shorter form independent of any representational claim, and the decision position is not
well defined. This was predicted for நீ ⊂ நீங்கள் and வா ⊂ வாருங்கள், and the pre-check
measures it rather than assuming it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

#: Tokenizer roster. `google/*` and `meta-llama/*` are gated and this machine has no HF
#: token, so the mirrors below are used; they are byte-identical tokenizers republished by
#: unsloth / NousResearch and the vocab sizes are asserted by the pre-check.
TOKENIZERS: dict[str, dict[str, Any]] = {
    "gemma-3-4b":   {"repo": "unsloth/gemma-3-4b-it",       "mirror_of": "google/gemma-3-4b-it",
                     "vocab": 262144},
    "gemma-3-12b":  {"repo": "unsloth/gemma-3-12b-it",      "mirror_of": "google/gemma-3-12b-it",
                     "vocab": 262144},
    "qwen3-8b":     {"repo": "Qwen/Qwen3-8B",               "mirror_of": None, "vocab": 151643},
    "llama-3.1-8b": {"repo": "NousResearch/Meta-Llama-3.1-8B",
                     "mirror_of": "meta-llama/Llama-3.1-8B", "vocab": 128000},
    "nllb-200":     {"repo": "facebook/nllb-200-3.3B",      "mirror_of": None, "vocab": 256204},
    # ai4bharat/indictrans2-* is gated and has no ungated mirror whose custom tokenizer class
    # loads under transformers 5.x. Recorded as NOT MEASURED rather than silently omitted.
}


@dataclass(frozen=True)
class PairTok:
    tokenizer_id: str
    variant_ids: dict[str, list[int]]
    divergence_index: int | None
    decision_position: int
    prefix_containment: bool
    token_counts: dict[str, int]
    fertility: dict[str, float]
    suffix_lengths: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return dict(gold_ids=None, variant_ids=self.variant_ids,
                    divergence_index=self.divergence_index,
                    decision_position=self.decision_position,
                    prefix_containment=self.prefix_containment,
                    token_counts=self.token_counts, fertility=self.fertility,
                    suffix_lengths=self.suffix_lengths)


def divergence(seqs: Sequence[Sequence[int]]) -> tuple[int | None, bool]:
    """First index where the sequences disagree, and whether one is a strict prefix."""
    if len(seqs) < 2:
        return None, False
    shortest = min(len(s) for s in seqs)
    for i in range(shortest):
        if len({s[i] for s in seqs}) > 1:
            return i, False
    if len({len(s) for s in seqs}) > 1:
        return shortest, True          # one ran out while agreeing => prefix containment
    return None, False


def tokenize_variants(tok: Any, tokenizer_id: str, variants: dict[str, str]) -> PairTok:
    ids = {v: tok(s, add_special_tokens=False)["input_ids"] for v, s in variants.items()}
    d, contained = divergence(list(ids.values()))
    counts = {v: len(x) for v, x in ids.items()}
    fert = {v: (len(ids[v]) / max(1, len(variants[v].split()))) for v in variants}
    suf = {v: (len(x) - d if d is not None else 0) for v, x in ids.items()}
    return PairTok(tokenizer_id=tokenizer_id, variant_ids=ids, divergence_index=d,
                   decision_position=(-1 if d in (None, 0) else d - 1),
                   prefix_containment=contained, token_counts=counts, fertility=fert,
                   suffix_lengths=suf)


def screen(pt: PairTok) -> tuple[bool, str | None]:
    """(usable_for_interp, exclusion_reason). `prefix_containment` is a hard exclude."""
    if pt.prefix_containment:
        return False, "prefix_containment"
    if pt.divergence_index is None:
        return False, "variants_tokenize_identically"
    return True, None


def load_tokenizers(ids: Sequence[str] | None = None) -> dict[str, Any]:
    from transformers import AutoTokenizer
    out = {}
    for name, cfg in TOKENIZERS.items():
        if ids and name not in ids:
            continue
        out[name] = AutoTokenizer.from_pretrained(cfg["repo"], trust_remote_code=True)
    return out
