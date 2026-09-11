# -*- coding: utf-8 -*-
"""The two `mt-thresholds` traps, resolved empirically.

TRAP 1, SCALE. The ToShip23 fits are on each metric's *conventionally reported* scale, which is
0-100 for BLEU / chrF / the whole COMET family and 0-25 for MetricX-23. A native COMET delta of
0.004 must therefore be passed as 0.4. Getting this backwards changes the headline number by 100x
and would make a genuinely invisible penalty look like a large one.

TRAP 2, RETURN UNITS. The upstream README states `accuracy(1.0, "bleu")` returns 0.63989. The
source returns 63.989 and the CLI merely formats it. Every call site must divide by 100, and the
round trip `delta(accuracy(d, k) / 100, k) == d` is asserted here for a grid of d and every key we
use.

Neither is checked by the package itself, so it is checked here and no calibrated number is quoted
until this file passes.
"""
from __future__ import annotations

import importlib.metadata as md

import pytest
from mt_thresholds import METRICS, accuracy, delta

#: Our metric -> (ToShip key, conventional scale max, native scale max, higher_is_better).
#: A key of None means the metric has NO published calibration and must be reported as a gap.
PANEL = {
    "comet22":        ("comet22", 100.0, 1.0, True),
    "cometkiwi22":    ("cometkiwi22", 100.0, 1.0, True),
    "xcometxl":       ("xcometxl", 100.0, 1.0, True),
    "xcometxxl":      ("xcometxxl", 100.0, 1.0, True),
    "cometkiwi23xl":  ("cometkiwi23-xl-src", 100.0, 1.0, True),
    "chrf":           ("chrf", 100.0, 100.0, True),
    "metricx23large": ("metricx-23-large", 25.0, 25.0, False),
    "metricx24xl":    (None, 25.0, 25.0, False),
    "gembaesa":       (None, 100.0, 100.0, True),
}

GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


def keys():
    return [(m, spec[0]) for m, spec in PANEL.items() if spec[0]]


def test_version_is_pinned():
    assert md.version("mt-thresholds") == "1.0.4", (
        "mt-thresholds version changed; the fitted constants are version-specific. "
        "Re-run the scale probe and update requirements before quoting any calibrated number.")


@pytest.mark.parametrize("metric,key", keys())
def test_key_exists(metric, key):
    assert key in METRICS


def test_the_two_metrics_without_a_key_stay_without_one():
    """MetricX-24 and GEMBA-ESA must never silently borrow another metric's thresholds."""
    for cand in ("metricx-24", "metricx24", "metricx-24-hybrid", "gemba", "gemba-esa", "gembaesa"):
        assert cand not in METRICS


def test_return_units_are_percent_not_fraction():
    """The README says 0.63989; the source returns 63.989. Pin the source's behaviour."""
    assert accuracy(1.0, "bleu") == pytest.approx(63.989, abs=1e-2)
    assert 50.0 < accuracy(1.0, "bleu") <= 100.0


@pytest.mark.parametrize("metric,key", keys())
def test_round_trip(metric, key):
    """delta(accuracy(d)/100) == d, which is only true if the /100 is applied.

    Tolerance is RELATIVE, not absolute: `metricx-23-large` has b = 26.3, so the sigmoid is
    already saturated at d = 1 and the inverse loses absolute precision there (observed error
    1.0e-6 at d = 1.0, i.e. 1 part in 1e6). That is float behaviour in a saturated regime, not a
    scale error -- the thing this file exists to catch would be off by 100x, not by 1e-6.
    """
    asym = accuracy(1e6, key) / 100.0            # the fitted 'a', the sigmoid's ceiling
    tested = 0
    for d in GRID:
        acc = accuracy(d, key) / 100.0
        # Above the asymptote the inverse is undefined and the package returns nan. For steep
        # fits (metricx-23-large, b = 26.3) that happens by d = 2, well inside the grid.
        if acc <= 0.5 or acc >= asym * (1 - 1e-9):
            continue
        assert delta(acc, key) == pytest.approx(d, rel=1e-5), (metric, key, d)
        tested += 1
    assert tested >= 2, f"{metric}: round trip exercised on too few points"


@pytest.mark.parametrize("metric,key", keys())
def test_accuracy_saturates_and_delta_is_nan_above_the_asymptote(metric, key):
    """Documented, not discovered later: `delta` returns nan for any accuracy at or above the
    fitted ceiling `a`. Any aggregation must therefore treat nan as 'beyond the fit', never as
    zero -- reading a nan as 0 would report a metric with a huge penalty as having none."""
    import math
    asym = accuracy(1e6, key) / 100.0
    assert asym < 1.0
    assert math.isnan(delta(min(asym + 1e-6, 0.999999), key)) or asym + 1e-6 > 1.0


@pytest.mark.parametrize("metric,key", keys())
def test_d75_is_a_sane_fraction_of_the_metric_scale(metric, key):
    """THE 100x GUARD.

    `delta(0.75, key)` is the threshold at which humans agree three times in four. Expressed as a
    fraction of the metric's conventional range it must be small but not vanishing: measured
    across the whole panel it lies in 0.2%-1.8%. A COMET fit read on the 0-1 scale would put
    d_75 at ~45% of the range, which this bound rejects.
    """
    scale_max = PANEL[metric][1]
    d75 = delta(0.75, key)
    frac = d75 / scale_max
    assert 0.0005 < frac < 0.05, (
        f"{metric}: d_75={d75:.4f} is {frac:.1%} of the conventional 0-{scale_max:g} range. "
        f"Either the scale is misread or the fit changed.")


def test_comet_family_needs_a_hundred_times_rescale():
    """The concrete instantiation of trap 1, asserted rather than remembered."""
    d75_native = delta(0.75, "comet22") / 100.0
    assert 0.001 < d75_native < 0.02, d75_native
    # a native COMET delta of 0.004 is 0.4 on the fitted scale, NOT 0.004
    assert accuracy(0.004 * 100, "comet22") / 100.0 == pytest.approx(
        accuracy(0.4, "comet22") / 100.0)
    # and reading it unscaled would understate the agreement probability badly
    assert accuracy(0.004, "comet22") / 100.0 < 0.51


def test_scale_probe_table(capsys):
    """Emits the appendix scale-probe table. Not an assertion; a printed artefact."""
    rows = []
    for metric, (key, conv, native, hib) in PANEL.items():
        if key is None:
            rows.append((metric, "—", "—", "—", "—", "no published calibration"))
            continue
        d75, d90 = delta(0.75, key), delta(0.90, key)
        rows.append((metric, key, f"0-{conv:g}", f"{d75:.4f}", f"{d90:.4f}",
                     f"{d75 / conv:.2%} of range"))
    print("\n| metric | ToShip key | fitted scale | d_75 | d_90 | d_75 as % of range |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    assert len(rows) == len(PANEL)
