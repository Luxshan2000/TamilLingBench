# -*- coding: utf-8 -*-
"""Tests for the benchmark generator.

Each test asserts a property the build brief makes load-bearing, and each one is written so
that it fails on a CONSTRUCTED NEGATIVE rather than merely passing on the current data. A
guard that has never been seen to fire is not a guard.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tamillingbench.gen.fstgen import FstGenerator, GenerationError
from tamillingbench.gen.screens import orthographically_wellformed
from tamillingbench.gen.templates import (TamilInTemplate, TemplateError, load_all,
                                          load_template)
from tamillingbench.gen.tokenize_check import divergence
from tamillingbench.morph.checker import MorphChecker
from tamillingbench.morph.labels import SUFFIX_SYNCRETISM, Slot

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark"


@pytest.fixture(scope="module")
def items():
    """The released items (public split)."""
    p = BENCH / "items.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def gen():
    g = FstGenerator()
    g.load()
    return g


# --------------------------------------------------------------------- no Tamil in templates

def test_no_tamil_in_any_shipped_template():
    """Requirement 1: templates carry feature bundles, never surface strings."""
    manifest = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))
    assert len(load_all()) == manifest["n_templates"]


def test_loader_raises_on_a_constructed_tamil_template(tmp_path):
    """The guard must FIRE, not merely exist."""
    good = yaml.safe_load((ROOT / "templates/gender/GEN-001.yaml").read_text(encoding="utf-8"))
    bad = dict(good)
    bad["target"] = {**good["target"]}
    bad["target"]["MASC"] = {**good["target"]["MASC"], "literal": "வந்தான்"}
    p = tmp_path / "BAD.yaml"
    p.write_text(yaml.safe_dump(bad, allow_unicode=True), encoding="utf-8")
    with pytest.raises(TamilInTemplate):
        load_template(p)


def test_provenance_is_exempt_from_the_tamil_scan(tmp_path):
    good = yaml.safe_load((ROOT / "templates/gender/GEN-001.yaml").read_text(encoding="utf-8"))
    good["provenance"] = {**good["provenance"], "note": "cf. வந்தான் in Lehmann §1.20"}
    p = tmp_path / "OK.yaml"
    p.write_text(yaml.safe_dump(good, allow_unicode=True), encoding="utf-8")
    load_template(p)          # must not raise


@pytest.mark.parametrize("key,value,exc", [
    ("holds_constant.hon", "plus", TemplateError),        # gender must pin -HON
    ("holds_constant.polarity", "negative", TemplateError),  # every slot pins +polarity
])
def test_loader_rejects_feasible_cell_violations(tmp_path, key, value, exc):
    good = yaml.safe_load((ROOT / "templates/gender/GEN-001.yaml").read_text(encoding="utf-8"))
    a, b = key.split(".")
    good[a] = {**good[a], b: value}
    p = tmp_path / "BAD.yaml"
    p.write_text(yaml.safe_dump(good, allow_unicode=True), encoding="utf-8")
    with pytest.raises(exc):
        load_template(p)


def test_loader_rejects_c2_c3_source_mismatch(tmp_path):
    """The byte identity of source.C2 and source.C3 IS the chance-rate argument."""
    good = yaml.safe_load((ROOT / "templates/number/NUM-002.yaml").read_text(encoding="utf-8"))
    good["source"] = {**good["source"], "C3": good["source"]["C3"] + " Really."}
    p = tmp_path / "BAD.yaml"
    p.write_text(yaml.safe_dump(good, allow_unicode=True), encoding="utf-8")
    with pytest.raises(TemplateError):
        load_template(p)


# --------------------------------------------------------------------- FST generation

def test_generation_round_trips(gen):
    f = gen.generate("செய்", "past.3sg.masc", "verb")
    assert f.surface == "செய்தான்"
    assert f.analysis.endswith("+3sgm=ஆன்")
    assert f.surface[f.morpheme_span[0]:f.morpheme_span[1]] == f.morpheme == "ான்"


def test_generation_raises_rather_than_guessing(gen):
    """`வா` has no `+3sghe=ஆர்` in the lexicon, so `வந்தார்` must not be invented."""
    with pytest.raises(GenerationError):
        gen.generate("வா", "past.3sg.hon", "verb")


def test_screen_b_rejects_a_vowel_sign_after_a_pulli():
    assert orthographically_wellformed("வந்தான்")
    assert not orthographically_wellformed("உட்கார்ுங்கள்")   # ு after ்
    assert not orthographically_wellformed("போினான்")        # ி after ோ


def test_screen_d_rejects_a_class_ambiguous_lemma(gen):
    """`விடு` licenses விட்ட-, விடுத்த- and விடுற்ற-; none may ship."""
    par = gen.harvest_verb("விடு")
    assert not par.admitted
    assert "gate-D" in par.rejected["*"]


# --------------------------------------------------------------------- the -ஆர் syncretism

def test_suffix_syncretism_covers_both_persons():
    assert set(SUFFIX_SYNCRETISM) == {"ஆர்", "ஆர்கள்", "ஈர்கள்", "உங்கள்"}


def test_aar_kal_resolves_by_referent_number():
    """The -ஆர்கள் syncretism hazard, end to end.

    `வந்தார்கள்` licenses honorific-singular AND rational-plural. Which one the FST emits
    varies by lexical entry, so the repair fans BOTH out and D-2's context filter decides.
    """
    ck = MorphChecker()
    pl = ck.check("வந்தார்கள்", Slot.RATIONALITY, "UYARTHINAI", referent_number="PL")
    assert pl.universal and pl.n_surviving == 1
    sg = ck.check("வந்தார்கள்", Slot.HONORIFICITY, "POLITE", referent_number="SG")
    assert sg.universal and sg.n_surviving == 1
    # and without the filter it is genuinely undecidable, not silently resolved
    amb = ck.check("வந்தார்கள்", Slot.NUMBER, "PL")
    assert amb.undecidable and not amb.universal


def test_ii_r_kal_resolves_for_the_honorific_singular():
    """The gap this build found: -ஈர்கள் was tagged bare `+2pl` and failed on singular items."""
    ck = MorphChecker()
    r = ck.check("வந்தீர்கள்", Slot.HONORIFICITY, "POLITE", referent_number="SG")
    assert r.universal


def test_number_control_is_decided_by_the_honorificity_pin():
    """Filtering a number item on number would be circular; the design's `hon: minus` pin
    is what makes `வந்தார்கள்` decidable as plural."""
    ck = MorphChecker()
    assert ck.check("வந்தார்கள்", Slot.NUMBER, "PL",
                    referent_honorificity="FAMILIAR").universal


# --------------------------------------------------------------------- shipped items

def test_every_item_declares_referent_number_or_is_the_number_slot(items):
    """DECISIONS.md D-2: ambiguous forms must be resolvable by declared context."""
    bad = [i["item_id"] for i in items
           if i["referent_number"] is None and i["slot"] != "number"]
    assert not bad, bad[:5]


def test_every_item_has_a_non_null_template_id(items):
    """The interpretability GroupKFold and the cluster bootstrap depend on it."""
    assert all(i["template_id"] for i in items)


def test_every_target_morpheme_span_selects_its_morpheme(items):
    """The xCOMET localisation test asserts exactly this."""
    for i in items:
        for t in i["gold_targets"] + i["contrast_targets"]:
            s, e = t["morpheme_char_span"]
            assert t["tamil"][s:e] == t["surface_morpheme"], (i["item_id"], t["tamil"])


def test_c2_sets_are_byte_identical_on_the_source(items):
    """The whole chance-rate argument: any function of the current sentence alone scores 1/k."""
    by_set: dict[str, set[bytes]] = {}
    for i in items:
        if i["condition"] == "C2":
            by_set.setdefault(i["set_id"], set()).add(i["source"].encode("utf-8"))
    assert all(len(v) == 1 for v in by_set.values())


def test_c3_items_carry_no_gold(items):
    assert all(i["gold_value"] is None and not i["gold_targets"]
               for i in items if i["condition"] == "C3")


def test_polarity_is_pinned_positive_on_every_item(items):
    """Negation with -வில்லை deletes every scored slot, so an unpinned item is escapable."""
    assert all(i["holds_constant"].get("polarity") == "positive" for i in items)


def test_rationality_ships_both_variety_realisations(items):
    """Scoring the Indian -அது as avoidance would penalise Indian-aligned output."""
    ahri = [i for i in items if i["slot"] == "rationality" and i["gold_value"] == "AHRI"]
    assert ahri
    for i in ahri:
        assert {t["variety"] for t in i["gold_targets"]} == {"lk", "in"}


def test_ternary_third_degree_is_declared_not_hard_coded(items):
    """தாங்கள் is attested as a deferential address form in ~1.8% of uses; the analysis must be
    able to split the ternary out rather than pool it."""
    tern = [i for i in items if i["k"] == 3]
    assert tern
    for i in tern:
        td = i["third_degree"]
        assert td and td["form"] in ("thaangaL", "optative")
        assert td["attestation"] in ("low", "unmeasured")
        assert td["fst_licenses_reading"] is False
        assert td["alternatives"]


def test_public_release_contains_no_blind_split():
    assert not list(BENCH.rglob("*.blind.jsonl"))
    assert {json.loads(l)["split"] for l in (BENCH / "items.jsonl").read_text(
        encoding="utf-8").splitlines()} == {"public"}


# --------------------------------------------------------------------- tokenization

def test_divergence_detects_prefix_containment():
    assert divergence([[1, 2], [1, 2, 3]]) == (2, True)
    assert divergence([[1, 2, 4], [1, 2, 3]]) == (2, False)
    assert divergence([[1, 2], [1, 2]]) == (None, False)
