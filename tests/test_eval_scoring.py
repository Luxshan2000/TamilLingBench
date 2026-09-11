# -*- coding: utf-8 -*-
"""Regression tests for the scoring pass.

`test_no_gold_never_correct` exists because of a real bug: C3 records were being written with
`outcome=CORRECT` / `correct=True` despite having no gold by design. That inverts the
C1→C2→C3 gradient — C3 outscored C1 — which is the exact opposite of the claim the benchmark
is built to test, and it is nearly invisible downstream because every individual number looks
plausible. The assertion below is cheap and it makes that class of error impossible to reship.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: WHICH ARTIFACTS THESE TESTS VALIDATE.
#:
#: Defaults to the released scored outputs and results. Exporting these points the SAME
#: assertions at another run, so a new run is checked against its own files and never
#: passes because an older tree happens to be green.
SCORED_DIR = os.environ.get("TLB_SCORED", "outputs/scored")
RESULTS_JSON = ROOT / os.environ.get("TLB_EVAL_RESULTS", "results/eval-results.json")
EXTRACTOR_JSON = ROOT / os.environ.get(
    "TLB_EXTRACTOR_VALIDATION", "results/extractor-validation.json")

SCORED = sorted(glob.glob(str(ROOT / SCORED_DIR / "*/*.jsonl")))


def _records():
    for fp in SCORED:
        for line in Path(fp).read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield fp, json.loads(line)


@pytest.mark.skipif(not SCORED, reason="no scored outputs yet")
def test_no_gold_never_correct():
    """No record without a gold may ever be marked correct, or carry a CORRECT/WRONG outcome."""
    bad = []
    for fp, r in _records():
        if r.get("gold_value") is None:
            if r.get("correct") is True:
                bad.append((fp, r["item_id"], "correct is True"))
            if r.get("outcome") in ("CORRECT", "WRONG"):
                bad.append((fp, r["item_id"], f"outcome={r['outcome']}"))
            for f in ("universal", "existential", "undecidable"):
                if r.get(f) is not None:
                    bad.append((fp, r["item_id"], f"{f}={r[f]!r} should be None"))
    assert not bad, f"{len(bad)} no-gold records carry accuracy: {bad[:5]}"


@pytest.mark.skipif(not SCORED, reason="no scored outputs yet")
def test_c3_has_no_gold_and_c0_does():
    """C3 is gold-free by design; C0 must genuinely carry a gold (it is the capability
    ceiling, and a C0 defaulting to correct by the same path would fake the ceiling)."""
    for fp, r in _records():
        if r["condition"] == "C3":
            assert r.get("gold_value") is None, f"{r['item_id']}: C3 must have no gold"
        if r["condition"] == "C0":
            assert r.get("gold_value") is not None, (
                f"{r['item_id']}: C0 must carry a gold — it is the capability ceiling")


@pytest.mark.skipif(not SCORED, reason="no scored outputs yet")
def test_outcomes_partition():
    from tamillingbench.eval.extract import OUTCOMES
    for fp, r in _records():
        assert r["outcome"] in OUTCOMES, f"{r['item_id']}: unknown outcome {r['outcome']}"


@pytest.mark.skipif(not SCORED, reason="no scored outputs yet")
def test_morpheme_span_slices_hypothesis():
    """`hypothesis[span[0]:span[1]] == surface_morpheme` on every record that has
    a span. The xCOMET span-localisation test cannot run without this."""
    for fp, r in _records():
        span = r.get("morpheme_char_span")
        if span:
            assert r["hypothesis"][span[0]:span[1]] == r["surface_morpheme"], (
                f"{r['item_id']}: span {span} does not slice to "
                f"{r['surface_morpheme']!r}")


def test_gold_targets_score_correct_and_contrasts_do_not():
    """The extractor validated against the benchmark's own gold. Recall must be perfect on
    every slot except honorificity, where the VV degree is structurally unscoreable (the
    checker licenses no DEFERENTIAL reading); false positives must be zero everywhere."""
    rep = EXTRACTOR_JSON
    if not rep.exists():
        pytest.skip("run scripts/eval/validate_extractor.py first")
    v = json.loads(rep.read_text(encoding="utf-8"))
    assert v["totals"]["n_contrast_false_positive"] == 0
    # Slot inventory is dataset-dependent. Assert on the slots the report actually contains
    # rather than KeyError-ing on an item set that legitimately lacks one. At least one slot must be
    # present, so an empty report cannot pass this test by vacuity.
    # A slot key can exist with zero gold targets (recall_rate None) on a dataset that does
    # not ship it — that is absence, not a recall of None. Presence means it HAS targets.
    present = {s for s, d in v["by_slot"].items() if d.get("n_gold_targets")}
    assert present, "extractor validation reported no slot with any gold target"
    for slot in ("clusivity", "gender", "number", "rationality"):
        if slot in present:
            assert v["by_slot"][slot]["recall_rate"] == 1.0, (
                f"{slot}: checker scores only "
                f"{v['by_slot'][slot]['recall_rate']:.4f} of the benchmark's OWN gold strings "
                f"as CORRECT ({v['by_slot'][slot]['recall_correct']}/"
                f"{v['by_slot'][slot]['n_gold_targets']}). No system can exceed this ceiling, "
                f"so every accuracy on this slot is understated. Breakdown: "
                f"{v['by_slot'][slot].get('recall_by_gold_value')}")
    if "honorificity" in present:
        hon = v["by_slot"]["honorificity"]
        n_vv = sum(n for k, n in hon["recall_by_gold_value"].items() if k.startswith("VV->"))
        assert hon["recall_correct"] + n_vv == hon["n_gold_targets"], (
            "honorificity recall failures must be exactly the VV items")


def test_no_accuracy_without_its_committed_rate():
    """Table contract: an accuracy may not exist without the denominator
    information that makes it interpretable. Enforced structurally so a future edit to the
    renderer cannot quietly drop the column."""
    rep = RESULTS_JSON
    if not rep.exists():
        pytest.skip("run scripts/eval/analyze.py first")
    res = json.loads(rep.read_text(encoding="utf-8"))
    for key, c in res["cells"].items():
        if c.get("acc_strict") is not None:
            assert c.get("committed_rate") is not None, f"{key}: accuracy without committed_rate"
            assert c.get("coverage") is not None, f"{key}: accuracy without coverage"
            assert c.get("n_all"), f"{key}: accuracy without n_all"
    for key, g in res["gradient"].items():
        for cond in ("C1", "C2"):
            if g.get(cond, {}).get("acc_strict") is not None:
                assert g[cond].get("committed_rate") is not None, key


def test_c3_cells_carry_no_accuracy():
    """C3 has no gold, so no accuracy-like field may be populated for it anywhere."""
    rep = RESULTS_JSON
    if not rep.exists():
        pytest.skip("run scripts/eval/analyze.py first")
    res = json.loads(rep.read_text(encoding="utf-8"))
    for key, c in res["cells"].items():
        if key.endswith("|C3"):
            for f in ("acc_strict", "acc_parsed", "acc_committed"):
                assert c.get(f) is None, f"{key}: {f} must be None for C3"


def test_provisional_slots_are_flagged():
    """Slots with an open native-check question must be marked, with the question named."""
    rep = RESULTS_JSON
    if not rep.exists():
        pytest.skip("run scripts/eval/analyze.py first")
    res = json.loads(rep.read_text(encoding="utf-8"))
    # Only slots this dataset actually contains. The flagging
    # requirement itself is NOT relaxed: any provisional slot that IS present must still be
    # marked and must still name its open question.
    slots_in_run = {g["slot"] for g in res["gradient"].values()}
    for slot in ("honorificity", "rationality"):
        if slot not in slots_in_run:
            continue
        assert slot in res["provisional_slots"], f"{slot} must be marked provisional"
        assert len(res["provisional_slots"][slot]) > 80, (
            f"{slot}: the specific open question must be named, not just flagged")
    for key, g in res["gradient"].items():
        if g["slot"] in ("honorificity", "rationality"):
            assert g["provisional"] is True, key
            assert g.get("provisional_reason"), key


def test_no_accuracy_from_a_thin_committed_denominator():
    """A reported accuracy may never derive from a tiny committed denominator.

    Concrete failure this prevents: gemma3-4b clusivity C1 committed on 5/220 items and scored
    100%, which beside its C2 (44.5% on 199) manufactures a 42-point C1→C2 drop out of
    nothing. Suppression is enforced in `Cell`, and this test checks the published artifact
    rather than the code path, so a renderer change cannot reintroduce it.
    """
    rep = RESULTS_JSON
    if not rep.exists():
        pytest.skip("run scripts/eval/analyze.py first")
    res = json.loads(rep.read_text(encoding="utf-8"))
    for key, c in res["cells"].items():
        if c.get("acc_strict") is None:
            continue
        n_comm = c["n_all"] - sum(c["outcomes"][o] for o in
                                  ("AVOIDANT_DROP", "AVOIDANT_NEUTRAL", "AVOIDANT_NEG",
                                   "UNPARSED"))
        assert n_comm >= 30, f"{key}: accuracy reported over committed n={n_comm} < 30"
        assert n_comm / c["n_all"] >= 0.30, (
            f"{key}: accuracy reported at commit rate {n_comm / c['n_all']:.1%} < 30%")
        assert not c.get("suppressed"), f"{key}: suppressed cell still reports an accuracy"
    for key, g in res["gradient"].items():
        if g.get("drop_C1_to_C2") is not None:
            assert not g.get("drop_suppressed_reason"), key
            for cond in ("C1", "C2"):
                assert g[cond]["acc_strict"] is not None, (
                    f"{key}: drop reported from a suppressed {cond} endpoint")


def test_no_dfr_from_a_thin_committed_denominator():
    """A DFR over a handful of committed items is not a distribution comparison."""
    rep = RESULTS_JSON
    if not rep.exists():
        pytest.skip("run scripts/eval/analyze.py first")
    res = json.loads(rep.read_text(encoding="utf-8"))
    for key, d in res["dfr"].items():
        if d.get("reportable"):
            assert d.get("n_committed", 0) >= 30, (
                f"{key}: DFR reported over committed n={d.get('n_committed')} < 30")


def test_labelled_translation_line_wins_over_context_line():
    """A model that answers with BOTH a labelled context line and a labelled translation line
    must be scored on the TRANSLATION, never on the context.

    This is a real bug with a measured cost. On C2 items the prompt supplies a context
    sentence whose proper noun fixes the referent's gender, and sarvam-translate answers:

        சூழல்: மீனா ... காத்திருந்தாள்.        <- context, re-translated; -ாள- is free here
        மொழிபெயர்ப்பு: நண்பகலில் வந்தார்கள்.   <- the actual answer; epicene -ஆர்கள-

    The old `take_first_tamil_line` rule picked the context line, so the model was credited
    with committing to FEM when its real answer had avoided. That inflated sarvam-translate's
    gender commitment to 71.8% and would have reversed the paper's controlled-pair conclusion.
    27.9% of its outputs take this shape.
    """
    import sys
    sys.path.insert(0, str(ROOT / "scripts/eval"))
    from postprocess import clean

    raw = ("சூழல்: மீனா நண்பகலிலிருந்து ரயில் நிலையத்தில் காத்திருந்தாள்.\n"
           "மொழிபெயர்ப்பு: நண்பகலில் வந்தார்கள்.")
    out, ops = clean(raw)
    assert out == "நண்பகலில் வந்தார்கள்.", out
    assert "take_labelled_translation_line" in ops, ops
    # the context line's feminine suffix must not survive anywhere in the scored string
    assert "காத்திருந்தாள்" not in out

    # A context-labelled line with no explicit translation label is still never the answer.
    out2, ops2 = clean("சூழல்: மீனா காத்திருந்தாள்.\nஅவர்கள் வந்தார்கள்.")
    assert out2 == "அவர்கள் வந்தார்கள்.", out2
    assert "drop_context_labelled_line" in ops2, ops2

    # Single-line answers and the Gemma romanisation rule must be untouched by all of this.
    assert clean("அவர்கள் வந்தார்கள்.")[0] == "அவர்கள் வந்தார்கள்."
    assert clean("அவர்கள் மதியம் வந்தார்கள். (Avargal vandhaarkaal.)")[0] == \
        "அவர்கள் மதியம் வந்தார்கள்."


def test_c2_answer_locus_flag_is_present_and_honoured():
    """A system that translated the CONTEXT instead of the marked sentence must be flagged.

    The failure this pins: on C2 the disambiguating cue lives in the context, so a model that
    renders the context is HANDED the feature by the cue NP and is scored COMMITTED/CORRECT
    without making any default-filling decision about the marked sentence. Its output is
    exactly as long as a correct answer, so the pre-existing length-based `c2_context_leak`
    detector cannot see it — measured on the released panel, gemma3-1b misses the gold verb stem on
    ~98% of gender C2 items while `c2_context_leak` reports 0.3%.

    This test does not assert WHICH systems are flagged (that is data, and it changes with the
    panel). It asserts the detector ran on every system that has both conditions, that its
    fields are well formed, and that the drop is what the flag is computed from — so a future
    edit cannot quietly turn the detector into a no-op that flags nobody.
    """
    if not RESULTS_JSON.exists():
        pytest.skip("run scripts/eval/analyze.py first")
    res = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    locus = res.get("c2_answer_locus")
    if locus is None:
        # A report generated before the detector existed (2026-08-08). Skipping is correct
        # ONLY for such a legacy artifact: re-running analyze.py on the same scored data
        # regenerates it with the key present, so this never silently excuses a live report.
        pytest.skip(f"{RESULTS_JSON.name} predates the C2 answer-locus detector; "
                    f"re-run scripts/eval/analyze.py to populate it")
    assert locus, "c2_answer_locus is present but empty — the detector ran on no system"

    c2_systems = {k.split("|", 1)[0] for k in res["cells"] if k.endswith("|C2")}
    c1_systems = {k.split("|", 1)[0] for k in res["cells"] if k.endswith("|C1")}
    for sysid in sorted(c1_systems & c2_systems):
        assert sysid in locus, (
            f"{sysid} has C1 and C2 cells but no answer-locus check — the detector must not "
            f"silently skip a system")

    for sysid, x in locus.items():
        for f in ("c1_gold_stem_rate", "c2_gold_stem_rate", "drop_C1_to_C2", "suspect"):
            assert x.get(f) is not None, f"{sysid}: answer-locus field {f} missing"
        # Tolerance is 2e-4, not 0: all three numbers are independently rounded to 4 dp on
        # the way into the report, so the difference of the rounded values and the rounded
        # difference can legitimately disagree in the last place. Anything larger means the
        # drop is not the quantity it is labelled as.
        assert abs((x["c1_gold_stem_rate"] - x["c2_gold_stem_rate"])
                   - x["drop_C1_to_C2"]) < 2e-4, (
            f"{sysid}: drop_C1_to_C2 is not C1 minus C2 — the flag is computed from a "
            f"quantity that is not the one reported")
        # The flag must track the drop, not be set independently of it.
        assert x["suspect"] == (x["drop_C1_to_C2"] > 0.25), (
            f"{sysid}: suspect flag disagrees with its own drop of {x['drop_C1_to_C2']}")

    flagged = sorted(s for s, x in locus.items() if x["suspect"])
    assert res.get("c2_answer_locus_suspects", []) == flagged, (
        "the reported suspect list must equal the systems whose flag is set")
