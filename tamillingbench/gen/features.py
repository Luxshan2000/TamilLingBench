# -*- coding: utf-8 -*-
"""The Tamil-free bundle vocabulary that templates are written in.

Requirement 1 of the build brief: **no Tamil surface strings in template files.** Templates
name a `bundle` such as `past.3sg.masc`; this module is the only place that knows which FST
label and which surface suffix that corresponds to, and `fstgen` is the only thing that turns
it into characters.

Two of these bundles exist *because* of the `-ஆர்` syncretism:

* `past.3sg.hon`  demands the label set {3sghe, 3sgh} **bound to -ஆர்**   (வந்தார்)
* `past.3pl.rat`  demands the label set {3ple, 3sghe} **bound to -ஆர்கள்** (வந்தார்கள்)

ThamizhiMorph does not agree with itself about which label rides which suffix — `செய்தார்`
is `+3sghe=ஆர்`, `பிறந்தார்` is `+3ple=ஆர்`, `வந்தார்` is nothing at all. Keying the bundle
on **(label ∈ set, suffix)** rather than on the label alone is what stops an item's reading
from depending on which verb happens to be in the lexicon. The checker's `SUFFIX_SYNCRETISM`
repair does the mirror-image job on the scoring side; `scripts/verify_syncretism.py` asserts
the two agree on every honorificity and gender form we ship.
"""
from __future__ import annotations

from ..morph.labels import (AHRINAI, EPICENE, FAMILIAR, FEM, MASC, NEUT, PL, POLITE, SG,
                            Slot, UYARTHINAI)

#: bundle name -> (allowed FST PNG labels, required suffix label string, human gloss)
#: The suffix strings here use the FST's INDEPENDENT-vowel spelling, exactly as it appears in
#: an analysis string (`+3sgm=ஆன்`), never the surface spelling.
VERB_BUNDLES: dict[str, tuple[tuple[str, ...], str | None, str]] = {
    # --- gender (3sg, non-honorific): the calibration slot -------------------------------
    "past.3sg.masc":   (("3sgm",),            "ஆன்",     "வந்தான்-shape"),
    "past.3sg.fem":    (("3sgf",),            "ஆள்",     "வந்தாள்-shape"),
    # the -ஆர் escape hatch: honorific singular, gender NOT expressible
    "past.3sg.hon":    (("3sghe", "3sgh"),    "ஆர்",     "வந்தார்-shape"),
    # --- rationality / number (3pl) ------------------------------------------------------
    "past.3pl.rat":    (("3ple", "3sghe"),    "ஆர்கள்",  "வந்தார்கள்-shape"),
    "past.3pl.irrat":  (("3pln",),            "அன",      "வந்தன-shape"),
    "past.3sg.neut":   (("3sgn",),            "அது",     "வந்தது-shape"),
    # --- 2nd person: HON-V ---------------------------------------------------------------
    "past.2sg.fam":    (("2sg", "2sgn"),      "ஆய்",     "வந்தாய்-shape"),
    "past.2pl.pol":    (("2pl", "2plh"),      "ஈர்கள்",  "வந்தீர்கள்-shape"),
    # the நீர் degree's verb agreement — generated so NATIVE-CHECK-HON-5 can be actioned
    "past.2sgh":       (("2sgh",),            "ஈர்",     "வந்தீர்-shape"),
    # --- 1st person ----------------------------------------------------------------------
    "past.1pl":        (("1pl",),             "ஓம்",     "வந்தோம்-shape"),
    "past.1sg":        (("1sg",),             "ஏன்",     "வந்தேன்-shape"),
    # --- imperative / optative: HON-I -----------------------------------------------------
    "imp.2sg.fam":     (("2sg",),             None,      "வா-shape (zero exponent)"),
    "imp.2pl.pol":     (("2pl",),             "உங்கள்",  "வாருங்கள்-shape"),
    "opt.def":         ((),                   None,      "வருக-shape (optative)"),
}

#: bundle name -> the feature bundle the checker must return for the form to count as gold.
#: `None` means "the form does not commit to this slot", which is itself checkable.
VERB_BUNDLE_FEATURES: dict[str, dict[Slot, str]] = {
    "past.3sg.masc":  {Slot.NUMBER: SG, Slot.GENDER: MASC, Slot.RATIONALITY: UYARTHINAI,
                       Slot.HONORIFICITY: FAMILIAR},
    "past.3sg.fem":   {Slot.NUMBER: SG, Slot.GENDER: FEM, Slot.RATIONALITY: UYARTHINAI,
                       Slot.HONORIFICITY: FAMILIAR},
    "past.3sg.hon":   {Slot.NUMBER: SG, Slot.GENDER: EPICENE, Slot.RATIONALITY: UYARTHINAI,
                       Slot.HONORIFICITY: POLITE},
    "past.3pl.rat":   {Slot.NUMBER: PL, Slot.GENDER: EPICENE, Slot.RATIONALITY: UYARTHINAI,
                       Slot.HONORIFICITY: FAMILIAR},
    "past.3pl.irrat": {Slot.NUMBER: PL, Slot.GENDER: NEUT, Slot.RATIONALITY: AHRINAI},
    "past.3sg.neut":  {Slot.NUMBER: SG, Slot.GENDER: NEUT, Slot.RATIONALITY: AHRINAI},
    "past.2sg.fam":   {Slot.NUMBER: SG, Slot.RATIONALITY: UYARTHINAI,
                       Slot.HONORIFICITY: FAMILIAR},
    "past.2pl.pol":   {Slot.NUMBER: SG, Slot.RATIONALITY: UYARTHINAI,
                       Slot.HONORIFICITY: POLITE},
    "past.2sgh":      {Slot.NUMBER: SG, Slot.RATIONALITY: UYARTHINAI,
                       Slot.HONORIFICITY: POLITE},
    "past.1pl":       {Slot.NUMBER: PL, Slot.RATIONALITY: UYARTHINAI},
    "past.1sg":       {Slot.NUMBER: SG, Slot.RATIONALITY: UYARTHINAI},
    "imp.2sg.fam":    {Slot.HONORIFICITY: FAMILIAR},
    "imp.2pl.pol":    {Slot.HONORIFICITY: POLITE},
    "opt.def":        {},
}

#: pronoun lemma -> (case-bundle -> nothing; the lemma+case pair IS the bundle).
#: Written as (fst_lemma, fst_tag_requirement, feature bundle, gloss). The FST keys oblique
#: pronouns under the OBLIQUE stem (`உங்கள்+pron+pssd+2plh+acc`), not under the nominative
#: citation form, which is why the nominative and the oblique rows have different lemmas.
PRON_ENTRIES: dict[str, dict] = {
    # --- 2nd person, HON-V / HON-P -------------------------------------------------------
    "2sg.T":   dict(lemma="நீ",       require=("2sg",),          gloss="நீ (T, nominative)",
                    features={Slot.NUMBER: SG, Slot.HONORIFICITY: FAMILIAR}),
    "2obl.T":  dict(lemma="உன்",      require=("2sg", "pssd"),   gloss="உன்- (T, oblique)",
                    features={Slot.NUMBER: SG, Slot.HONORIFICITY: FAMILIAR}),
    "2pl.V":   dict(lemma="நீங்கள்",  require=("2plh",),         gloss="நீங்கள் (V, nominative)",
                    features={Slot.HONORIFICITY: POLITE}),
    "2obl.V":  dict(lemma="உங்கள்",   require=("2plh", "pssd"),  gloss="உங்கள்- (V, oblique)",
                    features={Slot.HONORIFICITY: POLITE}),
    # ⚠ NATIVE-CHECK-HON-1. ThamizhiMorph analyses தாங்கள் as `+pron+3pl+refl`, i.e. as the
    # 3rd-person reflexive, NOT as the 2nd-person deferential. The SURFACE FORM is
    # FST-generated and round-trips; the 2nd-person deferential READING is a grammatical
    # claim the transducer does not make and that we must not invent. Every VV item is
    # flagged and every VV target is `verified_by: fst-generate+native-pending`.
    "2obl.VV": dict(lemma="தாங்கள்",  require=("3pl", "refl"),   gloss="தாங்கள்- (VV, oblique)",
                    features={Slot.HONORIFICITY: "DEFERENTIAL"}, flags=("NATIVE-CHECK-HON-1",)),
    "2pl.VV":  dict(lemma="தாங்கள்",  require=("3pl", "refl"),   gloss="தாங்கள் (VV, nominative)",
                    features={Slot.HONORIFICITY: "DEFERENTIAL"}, flags=("NATIVE-CHECK-HON-1",)),
    # D-6.1 CLOSED NATIVE-CHECK-HON-2 and HON-5: நீர் is a VARIANT OF நீ, not an
    # intermediate degree, so it belongs to the FAMILIAR group. This is the grouping the
    # grammars (Lehmann, Schiffman, Arden, Perumalsamy) support, and the FST licenses the
    # form as a genuine 2nd person (`+pron+2sgh+nom`, verb `+2sgh=ஈர்`) — unlike தாங்கள்.
    "2sgh.NEER": dict(lemma="நீர்", require=("2sgh",), gloss="நீர் (T group, nominative)",
                      features={Slot.NUMBER: SG, Slot.HONORIFICITY: FAMILIAR}),
    # --- 1st person plural, clusivity ------------------------------------------------------
    "1pl.INCL":     dict(lemma="நாம்",    require=("1pl", "incl"),         gloss="நாம் (INCL, nom)",
                         features={Slot.CLUSIVITY: "INCL", Slot.NUMBER: PL}),
    "1pl.EXCL":     dict(lemma="நாங்கள்", require=("1pl", "excl"),         gloss="நாங்கள் (EXCL, nom)",
                         features={Slot.CLUSIVITY: "EXCL", Slot.NUMBER: PL}),
    "1plobl.INCL":  dict(lemma="நம்",     require=("1pl", "incl", "pssd"), gloss="நம்- (INCL, oblique)",
                         features={Slot.CLUSIVITY: "INCL", Slot.NUMBER: PL}),
    "1plobl.EXCL":  dict(lemma="எங்கள்",  require=("1pl", "excl", "pssd"), gloss="எங்கள்- (EXCL, oblique)",
                         features={Slot.CLUSIVITY: "EXCL", Slot.NUMBER: PL}),
}

#: The four overt environments for clusivity, as FST case bundles.
OVERT_ENVIRONMENTS = {"nom_focus": "nom.foc", "acc": "acc", "dat": "dat", "gen": "gen"}

#: Slot values.
SLOT_VALUES: dict[str, tuple[str, ...]] = {
    # D-6.1 (native-speaker ruling): second-person address is BINARY. {நீ, நீர்} = T,
    # {நீங்கள், தாங்கள்} = V. The former third value `VV` is RETRACTED.
    "honorificity": ("T", "V"),
    "clusivity": ("INCL", "EXCL"),
    "rationality": ("UYAR", "AHRI"),
    "gender": ("MASC", "FEM"),
    "number": ("SG", "PL"),
}


#: Lemmas ThamizhiMorph gets wrong in a way no automatic screen catches, because the wrong
#: form is orthographically well-formed and stem-uniform. Each entry is grounded in a source
#: outside this project, never in an unaided judgement.
IRREGULAR_EXCLUDE: dict[str, str] = {
    # சொன்னான் is an irregular stem that requires a
    # hand-written fallback table. The FST instead produces சொன்றான் (the -ன்ற் class),
    # which round-trips and passes every screen.
    "சொல்": "irregular stem; FST gives சொன்றான் for சொன்னான்",
    # Same source, same class: the FST has no past paradigm for போ at all after screening,
    # and போகமாட்டான் is among the forms needing the fallback layer.
    "போ": "irregular stem; no screened past paradigm",
}
