# -*- coding: utf-8 -*-
"""Checker acceptance criteria, as pytest. Run from the repo root."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tamillingbench.morph import MorphChecker, Slot                      # noqa: E402
from tamillingbench.morph.labels import (LABEL_TO_SURFACE,               # noqa: E402
                                         label_suffix_to_surface)
from tamillingbench.morph.normalize import NonTamilToken, normalize      # noqa: E402


@pytest.fixture(scope="module")
def c():
    ch = MorphChecker()
    yield ch
    ch.close()


def labels_of(res):
    return {b.label for a in res.analyses for b in a.bindings} | {
        t for a in res.analyses for t in a.tags}


# ------------------------------------------------------------------ criterion 1
# §8b reproduction: all 13 forms return their stated analyses.
SECTION_8B = {
    "வந்தான்": "வா+verb+fin+sim+strong+past=த்+3sgm=ஆன்",
    "வந்தாள்": "வா+verb+fin+sim+strong+past=த்+3sgf=ஆள்",
    "செய்தார்": "செய்+verb+fin+sim+strong+past=த்+3sghe=ஆர்",
    "வந்தன": "வா+verb+fin+sim+strong+past=த்+3pln=அன",
    "வரவில்லை": "வா+verb+nonfin+sim+inf=அ+negpart=இல்லை",
    "வராமல்": "வா+verb+nonfin+sim+neg=ஆ+vpart=மல்",
    "வராதே": "வா+verb+fin+sim+imp=∅+neg=ஆத்+euph=ஏ",
    "நாம்": "நாம்+pron+1pl+incl+nom",
    "நாங்கள்": "நாங்கள்+pron+1pl+excl+nom",
    "நீங்கள்": "நீங்கள்+pron+2plh+nom",
    "தாங்கள்": "தாங்கள்+pron+3pl+refl+nom",
}


@pytest.mark.parametrize("surface,expected", SECTION_8B.items())
def test_section_8b_analyses_reproduce(c, surface, expected):
    assert expected in {a.raw for a in c.analyse(surface).analyses}


def test_section_8b_vandhaarkal_is_ambiguous(c):
    raws = {a.raw for a in c.analyse("வந்தார்கள்").analyses}
    assert "வா+verb+fin+sim+strong+past=த்+3sghe=ஆர்கள்" in raws
    assert "வா+verb+fin+sim+strong+past=த்+3ple=ஆர்கள்" in raws


# ------------------------------------------------------------------ criterion 2
def test_coverage_regression():
    """264 unique TTB-test VERB/AUX forms, 38.6% lexicon-only, 64.0% with guesser.

    Any drift here is a bug in normalization or union construction, not a new result.
    """
    with open("results/coverage.json", encoding="utf-8") as fh:
        cov = json.load(fh)["ttb-test"]
    assert cov["unique_forms"] == 264
    assert cov["lexicon_only_pct"] == 38.6
    assert cov["with_guesser_pct"] == 64.0
    assert cov["n_artifact_tokens"] == 16
    assert cov["artifact_corrected_pct"] == 68.1


# ------------------------------------------------------------------ criterion 3
def test_ambiguity_preserved(c):
    res = c.analyse("வந்தார்கள்")
    assert len(res.analyses) >= 2
    assert {"3sghe", "3ple"} <= labels_of(res)


# ------------------------------------------------------------------ criterion 4
def test_pronoun_not_shadowed(c):
    """நீ must return BOTH the 2sg pronoun and the verb imperative.

    This fails under naive `flookup -a` over separately-loaded nets (the verb net matches
    first and pronoun.fst is never consulted) and is the whole reason for `union net`.
    """
    raws = {a.raw for a in c.analyse("நீ").analyses}
    assert "நீ+pron+2sg+nom" in raws
    assert any("+verb+" in r for r in raws)


def test_union_binary_agrees_with_python_priority_union(c):
    """The published 2-net union and the checker's Python-side priority union must agree.

    Asserted rather than assumed — the checker drives the lexicon and guesser nets
    separately so it can attribute Analysis.source exactly (see Trap A).
    """
    words = ["நீ", "வந்தான்", "வந்தார்கள்", "நாம்", "உட்காருகின்றான்", "செய்தார்"]
    out = subprocess.run(["flookup", "-a", "build/thamizhi-union.bin"],
                         input="\n".join(words) + "\n", capture_output=True, text=True).stdout
    from_bin: dict[str, set[str]] = {w: set() for w in words}
    for line in out.split("\n"):
        if "\t" in line:
            s, a = line.split("\t", 1)
            if a != "+?":
                from_bin.setdefault(s, set()).add(a)
    for w in words:
        got = {a.raw for a in c.analyse(w).analyses if a.source.startswith("fst-")}
        assert got == from_bin[w], w


# ------------------------------------------------------------------ criterion 5
def test_clusivity(c):
    assert c.features("நாம்")[0][Slot.CLUSIVITY] == "INCL"
    assert c.features("நாங்கள்")[0][Slot.CLUSIVITY] == "EXCL"


# ------------------------------------------------------------------ criterion 6
GAPS = {
    "போகமாட்டான்": (Slot.POLARITY, "NEG"), "மாட்டான்": (Slot.POLARITY, "NEG"),
    "வரமாட்டாள்": (Slot.POLARITY, "NEG"), "இல்லை": (Slot.POLARITY, "NEG"),
    "இல்லாமல்": (Slot.POLARITY, "NEG"), "வேண்டாம்": (Slot.POLARITY, "NEG"),
    "சொன்னான்": (Slot.GENDER, "MASC"), "கொடுத்தார்": (Slot.HONORIFICITY, "POLITE"),
    "போனான்": (Slot.GENDER, "MASC"), "தந்தார்": (Slot.HONORIFICITY, "POLITE"),
}


@pytest.mark.parametrize("surface,slot_expected", GAPS.items())
def test_named_gaps_closed_by_fallback(c, surface, slot_expected):
    slot, expected = slot_expected
    res = c.analyse(surface)
    assert res.ok, f"{surface} still unanalysed"
    refnum = "SG" if slot is Slot.HONORIFICITY else None
    assert c.check(surface, slot, expected, referent_number=refnum).universal


def test_negfut_binds_the_png_suffix(c):
    """R1 must produce a real suffix binding, not just a polarity flag."""
    res = c.analyse("போகமாட்டான்")
    assert any(b.label == "3sgm" and b.suffix == "ஆன்"
               for a in res.analyses for b in a.bindings)
    assert res.suffix_bound


# ------------------------------------------------------------------ criterion 7
@pytest.mark.parametrize("surface", ["அல்ல", "அன்று"])
def test_named_false_positives_are_overridden(c, surface):
    """These get WRONG analyses from the FST, not misses.

    A fallback layer that only fills `+?` never sees them — hence the override layer.
    """
    res = c.analyse(surface)
    assert all(a.source == "override" for a in res.analyses)
    assert "negcop" in labels_of(res)
    assert c.check(surface, Slot.POLARITY, "NEG").universal
    # the spurious readings must be GONE, not merely outvoted
    assert not any("அல்லு" in a.raw or "imp=∅" in a.raw for a in res.analyses)


# ------------------------------------------------------------------ criterion 8  (TRAP B)
def test_raw_suffix_never_matches_for_independent_vowel_initial_labels(c):
    """`surface.endswith(binding.suffix)` is ALWAYS False for that class — the trap."""
    independent = set(range(0x0B85, 0x0B95))
    checked = 0
    for w in ["வந்தான்", "வந்தாள்", "வந்தன", "செய்தார்", "வரவில்லை", "வந்தார்கள்"]:
        for a in c.analyse(w).analyses:
            for b in a.bindings:
                if b.suffix and ord(b.suffix[0]) in independent:
                    checked += 1
                    assert not w.endswith(b.suffix), (w, b)
    assert checked >= 6


def test_label_to_surface_fixes_it(c):
    """The sanctioned fix: `Binding.surface_suffix` applies LABEL_TO_SURFACE."""
    cases = {"வந்தான்": "ான்", "வந்தாள்": "ாள்", "வந்தன": "ன",
             "செய்தார்": "ார்", "வரவில்லை": "ில்லை"}
    for w, tail in cases.items():
        last = None
        for a in c.analyse(w).analyses:
            for b in a.bindings:
                if not b.is_zero:
                    last = b
        assert last is not None, w
        assert last.surface_suffix == tail, (w, last)
        assert w.endswith(last.surface_suffix), (w, last)


def test_label_to_surface_table_is_complete():
    """அ is the INHERENT vowel and must map to the empty string, not to a sign."""
    assert LABEL_TO_SURFACE["அ"] == ""
    assert label_suffix_to_surface("ஆன்") == "ான்"
    assert label_suffix_to_surface("அன") == "ன"
    assert label_suffix_to_surface("இல்லை") == "ில்லை"
    assert label_suffix_to_surface("த்") == "த்"      # consonant-initial: unchanged
    assert label_suffix_to_surface("∅") == ""


def test_endswith_holds_across_the_whole_ud_vocabulary(c):
    """Measured claim, not a spot check: >= 99% of last bound morphemes match.

    Not 100%. Three documented exceptions on the UD vocabulary, both real:
      * கூறுகிறார் -> ...+3sghe=ஆர்கள்  — ThamizhiMorph binds the PLURAL allomorph string to
        a form that surfaces with -ஆர். An upstream data error.
      * வைப்போம்   -> வைப்போ+...+fut=உம் — a spurious secondary analysis whose stem is
        vowel-final, so the suffix's உ is absorbed by sandhi. The CORRECT analysis
        (வை+...+1pl=ஓம்) is also returned and does match.
    """
    forms = []
    seen = set()
    for p in ["data/ud/UD_Tamil-TTB/ta_ttb-ud-test.conllu",
              "data/ud/UD_Tamil-MWTT/ta_mwtt-ud-test.conllu"]:
        for line in open(p, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            if len(f) == 10 and "-" not in f[0] and "." not in f[0]:
                try:
                    n = normalize(f[1])       # rejects Latin/digits
                except NonTamilToken:
                    continue
                if n and n not in seen:
                    seen.add(n)
                    forms.append(n)
    ok = bad = 0
    for w, res in zip(forms, c.analyse_batch(forms)):
        for a in res.analyses:
            if a.source != "fst-lexicon":
                continue
            last = None
            for b in a.bindings:
                if not b.is_zero:
                    last = b
            if last is None:
                continue
            if w.endswith(last.surface_suffix):
                ok += 1
            else:
                bad += 1
    assert ok + bad > 300
    assert ok / (ok + bad) >= 0.99, f"{ok} pass / {bad} fail"


# ------------------------------------------------------------------ criterion 9
def test_normalization_is_mandatory_and_load_bearing():
    dec = "கொடு"    # கொடு decomposed
    com = "கொடு"          # கொடு composed
    assert normalize(dec) == com
    assert normalize(com) == com              # idempotent
    with MorphChecker() as on, MorphChecker(normalize=False) as off:
        assert on.analyse(dec).surface == on.analyse(com).surface
        # proving the step is load-bearing, not decorative:
        assert off.analyse(dec).surface != off.analyse(com).surface


def test_normalization_rejects_latin_and_digits():
    with pytest.raises(NonTamilToken):
        normalize("வந்தான்123")
    with pytest.raises(NonTamilToken):
        normalize("vandhaan")


def test_zero_width_and_punctuation_stripped():
    assert normalize("‍வந்தான்‌") == "வந்தான்"
    assert normalize("“வந்தான்”,") == "வந்தான்"


# ------------------------------------------------------------------ criterion 10
def test_mwtt_validation_targets():
    with open("results/checker-validation.json", encoding="utf-8") as fh:
        v = json.load(fh)["after_fallback"]["gender_masc"]
    # against the audited gold (MWTT annotation errors enumerated and excluded)
    assert v["adjusted_precision"] >= 98.0
    assert v["recall_universal"] >= 90.0
    # baseline before the fallback layer, as measured by scripts/validate_checker.py
    with open("results/checker-validation.json", encoding="utf-8") as fh:
        b = json.load(fh)["before_fallback"]["gender_masc"]
    assert b["unique_forms"] == 78
    assert b["gold_tokens"] == 220
    assert b["suffix_bound_pct"] == 61.5
    assert b["feature_only_pct"] == 10.3
    assert b["recall_universal"] == 71.8
    assert b["precision_universal"] == 96.6


def test_mwtt_polite_form_bug_characterisation():
    with open("results/checker-validation.json", encoding="utf-8") as fh:
        n = json.load(fh)["negative_results"]
    assert n["mwtt_aan_verb_aux_tokens"] == 215
    assert n["mwtt_aan_gender_masc"] == 212
    assert n["mwtt_aan_polite_form"] == 20
    assert n["mwtt_aan_verbform_fin"] == 20
    # the bug: the two sets are IDENTICAL, so requiring VerbForm=Fin shrinks 212 -> 20
    assert n["mwtt_polite_eq_verbform_fin"] is True


def test_four_ud_negative_results():
    with open("results/checker-validation.json", encoding="utf-8") as fh:
        n = json.load(fh)["negative_results"]
    assert n["ttb_gender_fem_tokens"] == 0            # #1
    assert n["clusivity_tokens_both_treebanks"] == 0  # #2
    assert n["naam_naangal_byte_identical"] is True   # #2
    assert n["illai_tokens_both_treebanks"] == 41     # #3
    assert n["illai_all_polarity_pos"] is True        # #3
    assert n["mwtt_polarity_neg_tokens"] == 38


# ------------------------------------------------------------------ criterion 11
def test_throughput(c):
    """>= 10,000 forms/second through analyse_batch with a persistent flookup process."""
    forms = ["வந்தான்", "வந்தாள்", "வந்தார்கள்", "நாம்", "நாங்கள்", "செய்தார்",
             "வரவில்லை", "வந்தன", "நீ", "நீங்கள்"] * 1000
    c.analyse_batch(forms[:10])          # warm the subprocesses
    t0 = time.perf_counter()
    res = c.analyse_batch(forms)
    dt = time.perf_counter() - t0
    assert len(res) == len(forms)
    rate = len(forms) / dt
    print(f"\nthroughput: {rate:,.0f} forms/s ({dt*1000:.1f} ms for {len(forms)})")
    assert rate >= 10_000, f"only {rate:,.0f} forms/s"


# ------------------------------------------------------------------ criterion 12
def test_report_regenerates_byte_identically():
    def run():
        subprocess.run([sys.executable, "scripts/validate_checker.py"],
                       capture_output=True, check=False)
        return open("results/checker-validation.md", "rb").read()
    assert run() == run()


# ------------------------------------------------------------------ D-2
def test_check_is_universal_after_context_filtering(c):
    """DECISIONS.md D-2. All three verdicts are outputs, never alternatives."""
    # no context: the analyser's ambiguity is unresolved -> undecidable, NOT a pass
    none = c.check("வந்தார்கள்", Slot.HONORIFICITY, "POLITE")
    assert none.undecidable and not none.universal and none.existential

    # context filter resolves it
    sg = c.check("வந்தார்கள்", Slot.HONORIFICITY, "POLITE", referent_number="SG")
    assert sg.universal and sg.existential and not sg.undecidable
    assert sg.n_surviving == 1 and sg.n_analyses == 2

    pl = c.check("வந்தார்கள்", Slot.HONORIFICITY, "POLITE", referent_number="PL")
    assert not pl.universal and not pl.existential

    # truthiness must be the HEADLINE metric, so `if check(...)` cannot be silently lenient
    assert bool(sg) is True
    assert bool(none) is False


def test_neengal_syncretism_is_resolved_by_context(c):
    """நீங்கள் is honorific-sg / plain-pl syncretic — one analysis, two readings."""
    assert len(c.analyse("நீங்கள்").analyses) == 1
    assert len(c.features("நீங்கள்")) == 2
    assert c.check("நீங்கள்", Slot.HONORIFICITY, "POLITE", referent_number="SG").universal
    assert c.check("நீங்கள்", Slot.HONORIFICITY, "FAMILIAR", referent_number="PL").universal
    assert c.check("நீங்கள்", Slot.HONORIFICITY, "POLITE").undecidable


def test_aar_syncretism_repair_is_uniform(c):
    """-ஆர் licenses honorific-sg AND rational-pl regardless of which the FST emitted.

    ThamizhiMorph is inconsistent here (செய்தார்->3sghe, பிறந்தார்->3ple); left alone that
    would decide honorificity items by which verb is in the lexicon.
    """
    for w in ["செய்தார்", "பிறந்தார்", "வந்தார்கள்"]:
        assert c.check(w, Slot.HONORIFICITY, "POLITE", referent_number="SG").universal, w
        assert c.check(w, Slot.NUMBER, "PL", referent_number="PL").universal, w


def test_thaangal_is_a_documented_override(c):
    """ThamizhiMorph says +pron+3pl+refl; the address-honorific reading is OUR override."""
    res = c.analyse("தாங்கள்")
    assert all(a.source == "override" for a in res.analyses)
    assert c.features("தாங்கள்")[0][Slot.HONORIFICITY] == "DEFERENTIAL"
    assert "R9" in res.analyses[0].why


# ------------------------------------------------------------------ Trap A
def test_guesser_emits_no_suffix_bindings(c):
    """Coverage != suffix-level checking. The guesser gives bare +3sgm, no `=`."""
    res = c.analyse("உட்காருகின்றான்")
    assert res.ok
    assert all(a.source == "fst-guesser" for a in res.analyses)
    assert all(not a.has_suffix_bindings for a in res.analyses)
    assert not res.suffix_bound
    # …while the lexicon does bind
    lex = c.analyse("வந்தான்")
    assert lex.suffix_bound and lex.analyses[0].source == "fst-lexicon"


def test_pronouns_are_lexicon_not_guesser(c):
    """pronoun.fst is a LEXICON net even though its analyses contain no `=`.

    Attributing source by `'=' in raw` misclassifies every pronoun as a guess.
    """
    for w in ["நாம்", "நாங்கள்", "நீங்கள்"]:
        assert {a.source for a in c.analyse(w).analyses} == {"fst-lexicon"}, w


# ------------------------------------------------------------------ R10 artifacts
def test_ttb_artifact_guard(c):
    assert c.is_artifact_token("கூறிய்")
    assert c.is_artifact_token("செல்லவ்")
    # NOT artifacts: real -அனர் rational plurals and verbal nouns
    assert not c.is_artifact_token("என்றனர்")
    assert not c.is_artifact_token("செல்லல்")
    # real words ending in ய் are analysed by the FST and never reach the guard
    assert not c.is_artifact_token("வந்தான்")
