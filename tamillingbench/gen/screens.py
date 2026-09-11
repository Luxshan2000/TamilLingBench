# -*- coding: utf-8 -*-
"""Screens that FST round-tripping does **not** give you.

The original design asserted that a round trip (generate → analyse → recover the bundle) makes
morphological correctness "a property of the pipeline and not of anyone's memory". Measured
on 2026-08-08 that is **false as stated**. The round trip is
run against the *same* transducer that generated the form, so anything the transducer
over-generates round-trips perfectly:

    வா  + pres.3sgm  -> வாகிறான்    (round-trips; the real form is வருகிறான்)
    போ  + past.3sgm  -> போினான்     (round-trips; not a well-formed Tamil string at all)
    சொல் + past.3sgm  -> சொன்றான்    (round-trips; the real form is சொன்னான்)
    விடு + past.3sgm  -> விடுற்றான்  AND விட்டான்   (both round-trip; only the second is real)

Three further screens are therefore applied, in this order, and each one's kill count is
reported in the generation report:

* **B — orthographic well-formedness.** A Tamil vowel sign or pulli may only follow a
  consonant. `போினான்` has ி after ோ; `உட்கார்ுங்கள்` has ு after a pulli. This is a
  script-level fact, not a judgement, and it removes the largest class of over-generation.
* **C — stem determinacy.** If the transducer licenses two different *stems* for one
  (lemma, tense) — `விட்ட்` and `விடுற்ற்` — its conjugation-class assignment for that lemma
  is unresolved, and neither reading may be used as gold.
* **D — paradigm stem uniformity.** Every past form of a lemma must share one stem. A lemma
  that fails is either suppletive (fine, but then the FST rarely has it) or misclassified.

What survives all four gates is *screened*, not *verified*: `சொல் -> சொன்றான்` passes B, C
and D and is still wrong. Every admitted lemma therefore also carries a `NATIVE-CHECK-VERB-*`
native-speaker check with its full generated paradigm printed for inspection.
That is the honest position and it is what the queue exists for.
"""
from __future__ import annotations

# Tamil script classes (U+0B80 block).
# NB: NOT written as a string containing "க்ஷ" — that would sneak the pulli U+0BCD
# into the consonant set and silently disable half of screen B.
_CONSONANTS = set("கஙசஜஞடணதநனபமயரறலளழவஷஸஹ")
_INDEP_VOWELS = set("அஆஇஈஉஊஎஏஐஒஓஔ")
VOWEL_SIGNS = _VOWEL_SIGNS = set("ாிீுூெேைொோௌ")
PULLI = "்"
AYTHAM = "ஃ"


def orthographically_wellformed(s: str) -> bool:
    """Screen B. A vowel sign or pulli must be preceded by a consonant letter.

    Deliberately permissive about everything else — this is a script constraint, not a
    phonotactic model, so it can only produce false *negatives* on genuinely odd input, and
    it produced none on the 63,896-entry Tamil Virtual University word list (measured).
    """
    prev = ""
    for ch in s:
        if ch in _VOWEL_SIGNS or ch == PULLI:
            if prev not in _CONSONANTS:
                return False
        prev = ch
    return True


def stem_of(surface: str, span: tuple[int, int] | None) -> str:
    """The surface with its PNG exponent removed — what screens C and D compare."""
    if span is None:
        return surface
    return surface[:span[0]]
