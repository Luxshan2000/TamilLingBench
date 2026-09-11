# -*- coding: utf-8 -*-
"""Tests for the corpus resource.

The load-bearing one is `test_canonicalization_is_shared`: a normalization mismatch between
the corpus-counting path and the hypothesis path is the silent corruption most likely to go
unnoticed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tamillingbench.corpus import slots, text                      # noqa: E402
from tamillingbench.corpus.index import (Document, FormIndex, IndexBuilder,  # noqa: E402
                                         _decode, _encode)
from tamillingbench.corpus.priors import corpus_prior, pool        # noqa: E402
from tamillingbench.corpus.sources import SOURCES                  # noqa: E402
from tamillingbench.corpus.stats import cluster_bootstrap, jsd, tvd, wilson  # noqa: E402
from tamillingbench.corpus.wiki import strip_wikitext              # noqa: E402
from tamillingbench.morph.normalize import normalize as morph_normalize  # noqa: E402

KO_DECOMPOSED = "கொ"     # க + ெ + ா
KO_COMPOSED = "கொ"             # க + ொ


# --------------------------------------------------------------------------- normalization

def test_canonicalization_is_shared():
    text.assert_shared_canonicalization()


def test_two_part_vowel_sign_is_composed():
    assert KO_DECOMPOSED != KO_COMPOSED                    # they really are different bytes
    assert text.canonicalize(KO_DECOMPOSED) == text.canonicalize(KO_COMPOSED)
    assert text.canonicalize(KO_DECOMPOSED) == KO_COMPOSED


def test_corpus_and_checker_paths_agree_on_1000_forms():
    """Round-trip guard between the corpus and checker paths."""
    base = ["வந்தான்", "வந்தாள்", "வந்தார்", "வந்தார்கள்", "வந்தன", "நீங்கள்", "நாம்",
            "நாங்கள்", "தாங்கள்", KO_DECOMPOSED, KO_COMPOSED, "கொடுத்தார்", "செய்தீர்கள்"]
    forms = [f + "‌" * (i % 2) for i in range(80) for f in base][:1000]
    for f in forms:
        assert text.canonicalize(f) == morph_normalize(f, reject_latin=False)


def test_tokenize_drops_latin_and_digits_and_keeps_tamil():
    toks = text.tokenize("அவர் 2019-ல் வந்தார். Hello நீங்கள்!")
    assert toks == ["அவர்", "ல்", "வந்தார்", "நீங்கள்"]


def test_is_tamil_line():
    assert text.is_tamil_line("அவர் வந்தார்")
    assert not text.is_tamil_line("Hello world அவர்")


# --------------------------------------------------------------------------- varint codec

@pytest.mark.parametrize("ids", [[], [0], [0, 1, 2], [5, 300, 70_000, 70_001],
                                 list(range(0, 5000, 7))])
def test_postings_codec_round_trips(ids):
    assert _decode(_encode(ids)) == ids


# --------------------------------------------------------------------------- wikitext

def test_strip_wikitext_removes_markup():
    raw = ("{{infobox|a=b}}\n"
           "'''கட்டடக்கலை''' என்பது [[கட்டடம்|கட்டடங்கள்]] ஆகும்.<ref>noise</ref>\n"
           "[[படிமம்:x.jpg|thumb|இது [[தாஜ் மஹால்]] ஆகும்.]]\n"
           "{| class=\"wikitable\"\n|-\n| junk || junk\n|}\n"
           "[[பகுப்பு:கட்டடக்கலை]]")
    out = strip_wikitext(raw)
    assert "கட்டடக்கலை என்பது கட்டடங்கள் ஆகும்." in out
    assert "noise" not in out
    assert "junk" not in out
    assert "]]" not in out and "[[" not in out
    assert "தாஜ் மஹால்" not in out         # file caption is not running text


def test_strip_wikitext_cuts_reference_section():
    raw = "உரை ஒன்று.\n== மேற்கோள்கள் ==\n* junk reference"
    assert "junk" not in strip_wikitext(raw)


# --------------------------------------------------------------------------- index

@pytest.fixture(scope="module")
def tiny_index(tmp_path_factory) -> FormIndex:
    p = tmp_path_factory.mktemp("corpus") / "tiny.sqlite"
    b = IndexBuilder(p, "tiny", candidate=slots.candidate, min_doc_tokens=3).open()
    b.add(Document("doc-a", "நீங்கள் நேற்று வந்தீர்கள். அவன் வந்தான் என்று அவர் கூறினார்."))
    b.add(Document("doc-b", "நாம் இன்று போவோம். நாங்கள் நேற்று வந்தோம். அவள் வந்தாள்."))
    b.add(Document("doc-c", "மாணவர்கள் வந்தார்கள். புத்தகங்கள் மேசையில் இருந்தன. நீ வா."))
    b.close()
    return FormIndex(p)


def test_frequency_and_attestation(tiny_index):
    f = tiny_index.frequency("நீங்கள்")
    assert f.n == 1 and f.df == 1 and f.attested
    assert not tiny_index.frequency("ஜஜஜஜ").attested
    assert tiny_index.frequency("நீங்கள்").per_million > 0


def test_frequency_is_normalization_insensitive(tiny_index):
    """A decomposed query must find the composed corpus form and vice versa."""
    b = IndexBuilder(Path(tiny_index.path).parent / "norm.sqlite", "norm",
                     min_doc_tokens=1).open()
    b.add(Document("d", " ".join([KO_DECOMPOSED + "டுத்தார்"] * 5)))
    b.close()
    idx = FormIndex(Path(tiny_index.path).parent / "norm.sqlite")
    assert idx.frequency(KO_COMPOSED + "டுத்தார்").n == 5
    assert idx.frequency(KO_DECOMPOSED + "டுத்தார்").n == 5


def test_kwic_returns_attested_sentences_with_provenance(tiny_index):
    hits = tiny_index.kwic("வந்தார்கள்")
    assert len(hits) == 1
    h = hits[0]
    assert h.match == "வந்தார்கள்"
    assert h.doc_ref == "doc-c"
    assert h.corpus == "tiny"
    assert "மாணவர்கள்" in h.left
    assert h.match in h.sentence


def test_kwic_on_unattested_form_is_empty(tiny_index):
    assert tiny_index.kwic("ஜஜஜஜ") == []


def test_doc_counts_cluster_by_document(tiny_index):
    dc = tiny_index.doc_counts(["நாம்", "நாங்கள்"])
    assert set(dc["நாம்"].values()) == {1}
    assert len(dc["நாம்"]) == 1        # both 1PL pronouns live in doc-b only
    assert len(dc["நாங்கள்"]) == 1


# --------------------------------------------------------------------------- slots

def test_candidate_is_a_superset_of_the_pronouns():
    for f in ("நீ", "நீங்கள்", "தாங்கள்", "நாம்", "நாங்கள்", "உங்களுக்கு"):
        assert slots.candidate(f)


def test_candidate_rejects_obvious_non_candidates():
    assert not slots.candidate("புத்தகம்")
    assert not slots.candidate("மேசையில்")


def test_inventory_assigns_the_canonical_paradigm():
    inv = slots.build_inventory(
        ["வந்தான்", "வந்தாள்", "வந்தார்", "வந்தார்கள்", "வந்தன", "வந்தனர்",
         "வந்தாய்", "வந்தீர்கள்", "நாள்", "மகான்", "புத்தகம்"])
    assert "வந்தான்" in inv.forms("gender", slots.MASC)
    assert "வந்தாள்" in inv.forms("gender", slots.FEM)
    assert "வந்தார்" in inv.forms("gender", slots.EPICENE)
    assert "வந்தார்கள்" in inv.forms("rationality", slots.UYARTHINAI)
    assert "வந்தனர்" in inv.forms("rationality", slots.UYARTHINAI)
    assert "வந்தன" in inv.forms("rationality", slots.AHRINAI)
    assert "வந்தாய்" in inv.forms("honorificity_verb", slots.HONV_FAMILIAR)
    assert "வந்தீர்கள்" in inv.forms("honorificity_verb", slots.HONV_POLITE)


def test_inventory_rejects_nouns_that_end_in_verb_suffixes():
    """`நாள்` ('day') ends in -ஆள் and `மகான்` ends in -ஆன்; neither is a finite verb.

    This is what the FST gate buys over a bare `endswith` rule, and it is the difference
    between a gender prior and a suffix-frequency table.
    """
    inv = slots.build_inventory(["நாள்", "மகான்", "வான்"])
    assert inv.forms("gender", slots.FEM) == []
    assert inv.forms("gender", slots.MASC) == []


def test_slot_specs_cover_the_four_slots():
    inv = slots.build_inventory(["வந்தான்", "வந்தாள்", "வந்தார்", "வந்தார்கள்", "வந்தன",
                                 "வந்தாய்", "வந்தீர்கள்"])
    got = {s.slot for s in slots.slot_specs(inv)}
    assert got == {"clusivity", "honorificity_pronoun", "honorificity_verb",
                   "rationality", "gender"}


def test_ambiguous_values_carry_a_caveat():
    inv = slots.build_inventory(["வந்தார்கள்", "வந்தன"])
    specs = [s for s in slots.slot_specs(inv)
             if s.slot == "rationality" and s.variant == "suffix_bound"]
    assert specs and any("UPPER" in c for c in specs[0].caveats)


# --------------------------------------------------------------------------- stats

def test_wilson_interval_brackets_the_point_estimate():
    ci = wilson(30, 100)
    assert ci.lo < 0.30 < ci.hi
    assert wilson(0, 0).as_tuple() == (0.0, 1.0)


def test_cluster_bootstrap_is_wider_than_iid_when_forms_clump():
    """The reason the prior uses a cluster bootstrap.

    Same marginal counts, two layouts: perfectly clumped by document versus evenly spread.
    The clumped layout must produce the wider interval and a DEFF above 1.
    """
    clumped = {"A": {i: 10 for i in range(50)},
               "B": {i: 10 for i in range(50, 100)}}
    spread = {"A": {i: 5 for i in range(100)},
              "B": {i: 5 for i in range(100)}}
    rc = cluster_bootstrap(clumped, ["A", "B"], n_docs=100, n_boot=800)
    rs = cluster_bootstrap(spread, ["A", "B"], n_docs=100, n_boot=800)
    assert abs(rc.p["A"] - 0.5) < 1e-9 and abs(rs.p["A"] - 0.5) < 1e-9
    w_c = rc.ci["A"][1] - rc.ci["A"][0]
    w_s = rs.ci["A"][1] - rs.ci["A"][0]
    assert w_c > w_s * 5
    assert rc.deff["A"] > 5 > rs.deff["A"]


def test_tvd_and_jsd_are_zero_for_identical_distributions():
    p = {"A": 0.7, "B": 0.3}
    assert tvd(p, p) == pytest.approx(0.0)
    assert jsd(p, p) == pytest.approx(0.0, abs=1e-12)
    assert tvd({"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}) == pytest.approx(1.0)


# --------------------------------------------------------------------------- priors

def test_pool_flags_a_fragile_prior(tmp_path):
    """Two corpora that disagree sharply must come out FRAGILE, not averaged."""
    from tamillingbench.corpus.slots import SlotSpec

    idxs = []
    for name, a_reps, b_reps in (("wiki", 1, 20), ("subs", 20, 1)):
        p = tmp_path / f"{name}.sqlite"
        b = IndexBuilder(p, name, candidate=lambda f: True, min_doc_tokens=1).open()
        for i in range(40):
            # Distinct filler per document, and the filler must itself be healthy Tamil:
            # the builder exact-dedups identical lines (40 byte-identical documents would
            # collapse to one) AND drops documents whose vowel-sign ratio falls below
            # MIN_SIGN_RATIO, which a sign-free filler word would trigger.
            b.add(Document(f"{name}-{i}",
                           " ".join(["நாம்"] * a_reps + ["நாங்கள்"] * b_reps
                                    + ["பாடல்"] * (i + 1))))
        b.close()
        idxs.append(FormIndex(p))
    spec = SlotSpec(slot="clusivity", values=("INCL", "EXCL"),
                    forms={"INCL": ["நாம்"], "EXCL": ["நாங்கள்"]},
                    kind="pronoun", variant="nominative")
    sp = pool([corpus_prior(i, spec, n_boot=400) for i in idxs], spec)
    assert sp.pooled_n["INCL"] > 100 and sp.pooled_n["EXCL"] > 100
    assert sp.max_pairwise_tvd > 0.5
    assert sp.fragile
    assert sp.verdict.startswith("FRAGILE")


def test_pool_reports_stable_when_corpora_agree(tmp_path):
    from tamillingbench.corpus.slots import SlotSpec

    idxs = []
    for name in ("wiki", "subs"):
        p = tmp_path / f"agree-{name}.sqlite"
        b = IndexBuilder(p, name, candidate=lambda f: True, min_doc_tokens=1).open()
        for i in range(60):
            b.add(Document(f"{name}-{i}",
                           " ".join(["நாம்"] * 3 + ["நாங்கள்"] * 7 + ["பாடல்"] * (i + 1))))
        b.close()
        idxs.append(FormIndex(p))
    spec = SlotSpec(slot="clusivity", values=("INCL", "EXCL"),
                    forms={"INCL": ["நாம்"], "EXCL": ["நாங்கள்"]},
                    kind="pronoun", variant="nominative")
    sp = pool([corpus_prior(i, spec, n_boot=400) for i in idxs], spec)
    assert sp.pooled_n["INCL"] > 100 and sp.pooled_n["EXCL"] > 100
    assert sp.max_pairwise_tvd < 0.02
    assert not sp.fragile


# --------------------------------------------------------------------------- licences

def test_ttb_is_release_blocked():
    """DECISIONS D-4. If this test ever passes trivially, the gate is gone."""
    ttb = SOURCES["ud_ttb"]
    assert not ttb.release_ok and not ttb.derived_ok


def test_every_source_states_a_licence_and_a_check_date():
    for s in SOURCES.values():
        assert s.licence and s.licence_checked, s.key
        assert s.register and s.role, s.key


def test_non_redistributable_sources_are_marked():
    for key in ("opensubs", "cc100"):
        assert not SOURCES[key].release_ok, key


# --------------------------------------------------------------------------- text quality

def test_vowel_sign_ratio_separates_healthy_from_damaged_tamil():
    """The OPUS OpenSubtitles failure, as a test.

    Healthy Tamil carries a combining vowel sign on roughly half its consonants; the
    TSCII→Unicode-damaged text carries almost none, and the signs it loses are exactly the
    ones that distinguish -ஆன்/-ஆள்/-ஆர். Measured on the real corpora: Wikipedia 0.432,
    Sangraha 0.446–0.455, OpenSubtitles 0.014.
    """
    from tamillingbench.corpus.index import MIN_SIGN_RATIO, vowel_sign_ratio

    healthy = "அனைத்து பறவைகளும் இறக்க வேண்டும் என்று அவர்கள் நினைக்கிறார்கள்"
    damaged = "அனத்தபறவகள்இறக வண்டம்என்றஅவர்கள்நனக்கறர்கள்"
    assert vowel_sign_ratio(healthy) > 0.35
    assert vowel_sign_ratio(damaged) < 0.05
    assert vowel_sign_ratio(damaged) < MIN_SIGN_RATIO < vowel_sign_ratio(healthy)


def test_builder_drops_encoding_damaged_documents(tmp_path):
    p = tmp_path / "damaged.sqlite"
    b = IndexBuilder(p, "damaged", min_doc_tokens=3).open()
    b.add(Document("good", "அனைத்து பறவைகளும் இறக்க வேண்டும் என்று அவர்கள் நினைக்கிறார்கள்"))
    b.add(Document("bad", "அனத்தபறவகள்இறக வண்டம்என்றஅவர்கள்நனக்கறர்கள் மற்றம்அவர்கள்"))
    b.close()
    assert b.n_docs == 1
    assert b.n_docs_dropped_quality == 1


def test_noun_veto_removes_the_measured_false_positives():
    """கால்வாய் ('canal') gets a `+fut+2sg=ஆய்` verb reading; ஆண்டாள் is a proper name.

    Both were the top-frequency entries of their slot value before the veto. The veto must
    remove them and must remove NONE of the canonical paradigm.
    """
    import os

    if not os.path.exists(slots.DEFAULT_NOUN_NET):
        pytest.skip("noun net not built; run scripts/build_noun_net.sh")
    inv = slots.build_inventory(["கால்வாய்", "ஆண்டாள்", "நாள்", "வாய்",
                                 "வந்தான்", "வந்தாள்", "வந்தார்", "வந்தாய்", "வந்தன"])
    assert inv.noun_veto
    assert "கால்வாய்" not in inv.forms("honorificity_verb", slots.HONV_FAMILIAR)
    assert "ஆண்டாள்" not in inv.forms("gender", slots.FEM)
    assert "வந்தாள்" in inv.forms("gender", slots.FEM)
    assert "வந்தான்" in inv.forms("gender", slots.MASC)
    assert "வந்தாய்" in inv.forms("honorificity_verb", slots.HONV_FAMILIAR)


def test_guesser_hallucinated_nouns_are_not_counted():
    """`பாக்கித்தான்` (Pakistan) was the top 'masculine verb' under the loose rule."""
    inv = slots.build_inventory(["பாக்கித்தான்", "சுல்தான்", "செய்தித்தாள்", "வந்தான்"])
    assert inv.forms("gender", slots.MASC) == ["வந்தான்"]
    assert inv.forms("gender", slots.FEM) == []
