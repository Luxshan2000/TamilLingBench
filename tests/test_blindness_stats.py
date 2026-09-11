# -*- coding: utf-8 -*-
"""Unit tests for the metric-blindness statistics and the Tamil minimal-pair surgery.

The statistics are simple enough that a bug would not look like a crash -- it would look like a
finding. These pin the behaviours the section's argument depends on: ties, the z guard, the
cluster bootstrap resampling clusters rather than items, and the one-grapheme-span assertion that
makes a metric delta attributable to the morpheme and nothing else.
"""
from __future__ import annotations

import numpy as np
import pytest

from tamillingbench.metrics.blindness import (
    ORIENT, calibrate, cluster_bootstrap_ci, direction, holm_bonferroni, magnitude, orient,
    z_floor)
from tamillingbench.metrics.surgery import (
    build_flipped, letters, n_diff_spans, normalize_sentence, splice)


# ------------------------------------------------------------------ direction

def test_direction_counts_ties_separately_and_splits_them():
    d = direction([1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 2.0, 0.0])
    assert (d.n_win, d.n_tie, d.n_loss) == (1, 2, 1)
    assert d.win_rate == pytest.approx(0.5)
    # the binomial test uses only the two non-tied pairs, not all four
    assert d.p_binomial == pytest.approx(1.0)


def test_a_tie_heavy_half_is_not_a_coin_flip():
    """chrF produces genuine ties; a 0.5 win rate made of ties means something different."""
    ties = direction([1.0] * 10, [1.0] * 10)
    coin = direction([1.0] * 5 + [0.0] * 5, [0.0] * 5 + [1.0] * 5)
    assert ties.win_rate == coin.win_rate == 0.5
    assert ties.n_tie == 10 and coin.n_tie == 0


# ------------------------------------------------------------------ orientation

def test_metricx_is_flipped_and_nothing_else_is():
    assert orient("metricx24xl", np.array([3.0]))[0] == -3.0
    assert orient("metricx23large", np.array([3.0]))[0] == -3.0
    assert orient("comet22", np.array([0.8]))[0] == 0.8
    assert set(ORIENT) == {"metricx24xl", "metricx24xlqe", "metricx23large"}


# ------------------------------------------------------------------ magnitude

def test_magnitude_reports_the_share_where_the_metric_prefers_the_wrong_form():
    m = magnitude([1.0, 1.0, 1.0, 1.0], [0.5, 1.5, 1.0, 0.0], "0-1")
    assert m.mean == pytest.approx((0.5 - 0.5 + 0.0 + 1.0) / 4)
    assert m.frac_nonpositive == pytest.approx(0.5)      # one negative, one exactly zero


# ------------------------------------------------------------------ z

def test_z_drops_degenerate_denominators_and_counts_them():
    good = np.array([1.0, 1.0, 1.0])
    bad = np.array([0.9, 0.9, 0.9])
    empty = np.array([0.0, 1.0, 0.9999])          # 2nd and 3rd denominators are ~0
    z = z_floor(good, bad, empty, ["t"] * 3, eps=1e-3, b=50)
    assert z.n_used == 1 and z.n_dropped_eps == 2
    assert z.mean == pytest.approx(0.1)


def test_z_is_one_when_the_error_costs_as_much_as_saying_nothing():
    z = z_floor(np.array([1.0]), np.array([0.0]), np.array([0.0]), ["t"], b=50)
    assert z.mean == pytest.approx(1.0)


# ------------------------------------------------------------------ bootstrap

def test_cluster_bootstrap_resamples_clusters_not_items():
    """Two clusters with wildly different means: an item-level bootstrap would give a tight
    interval, a cluster bootstrap must give a wide one because it can draw both from either."""
    vals = np.array([0.0] * 50 + [1.0] * 50)
    clusters = ["A"] * 50 + ["B"] * 50
    lo, hi = cluster_bootstrap_ci(vals, clusters, b=2000, seed=1)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_cluster_bootstrap_is_deterministic_under_the_fixed_seed():
    vals = np.random.default_rng(0).normal(size=200)
    cl = [f"t{i % 7}" for i in range(200)]
    assert cluster_bootstrap_ci(vals, cl, b=500, seed=3) == \
        cluster_bootstrap_ci(vals, cl, b=500, seed=3)


# ------------------------------------------------------------------ calibration

def test_calibration_rescales_comet_by_a_hundred():
    c = calibrate("comet22", 0.004)
    assert c.available and c.scale_factor == 100.0
    assert c.delta_fitted == pytest.approx(0.4)


def test_metricx24_and_anything_unkeyed_report_a_gap_not_a_borrowed_threshold():
    for m in ("metricx24xl", "metricx24xlqe", "gembaesa", "chrfpp"):
        c = calibrate(m, 0.5)
        assert c.available is False
        assert c.toship_key is None
        assert c.ratio_75 is None and c.acc_at_observed is None


def test_calibration_is_sign_insensitive():
    """A metric that PREFERS the wrong form has a negative delta; the calibrated magnitude is
    about size, and the sign is carried by `direction`, not silently folded in here."""
    assert calibrate("comet22", -0.004).delta_fitted == pytest.approx(0.4)


# ------------------------------------------------------------------ holm

def test_holm_stops_at_the_first_failure():
    r = holm_bonferroni({"a": 0.001, "b": 0.03, "c": 0.9}, alpha=0.05)
    assert r["a"] is True and r["c"] is False


# ------------------------------------------------------------------ surgery

def test_one_grapheme_cluster_span_is_one_span_not_three_codepoints():
    """வந்தாள் -> வந்தான் edits 'ாள்' to 'ான்': three codepoints, ONE letter to a reader."""
    assert n_diff_spans("அகநகை வந்தாள்.", "அகநகை வந்தான்.") == 1


def test_a_changed_name_as_well_as_a_changed_suffix_is_two_spans():
    """The benchmark's own contrast_targets are NOT minimal pairs; this is what rejects them."""
    assert n_diff_spans("அகநகை வந்தாள்.", "அகரன் வந்தான்.") == 2


def test_insertion_and_deletion_each_count_as_one_span():
    assert n_diff_spans("எழுது.", "எழுதுங்கள்.") == 1
    assert n_diff_spans("எழுதுங்கள்.", "எழுது.") == 1


def test_identical_strings_differ_in_no_span():
    assert n_diff_spans("வந்தான்", "வந்தான்") == 0


def test_letters_segments_consonant_plus_vowel_sign_as_one_unit():
    assert letters("வந்தாள்") == ["வ", "ந்", "தா", "ள்"]


def test_normalisation_composes_two_part_vowel_signs():
    decomposed = "க" + "ெ" + "ா"        # க + ெ + ா
    assert normalize_sentence(decomposed) == "கொ"
    assert len(normalize_sentence(decomposed)) < len(decomposed)


def test_splice_rejects_an_out_of_range_span_instead_of_raising():
    s = splice("வந்தான்", (0, 99), "x")
    assert s.ok is False and "out of range" in s.reason


def test_build_flipped_prefers_the_attested_contrast_when_it_is_already_minimal():
    r = build_flipped(normalize_sentence("எழுது."), (5, 5), "ுங்கள்",
                      normalize_sentence("எழுதுங்கள்."))
    assert r.ok and r.text == "எழுதுங்கள்." and "contrast_target" in r.reason


def test_build_flipped_falls_back_to_splicing_when_the_contrast_is_not_minimal():
    r = build_flipped(normalize_sentence("அகநகை வந்தாள்."), (10, 13), "ான்",
                      normalize_sentence("அகரன் வந்தான்."))
    assert r.ok and r.text == "அகநகை வந்தான்." and "spliced" in r.reason
