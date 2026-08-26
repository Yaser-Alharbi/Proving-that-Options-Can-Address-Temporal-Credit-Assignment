"""Tests for the aggregation core of `plot.py`, on hand-built inputs."""

import argparse
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from plot import (
    MIN_SEEDS_FOR_IQM,
    Inputs,
    apply_x_scale,
    bootstrap_indices,
    collinear,
    crossing_step,
    estimate,
    iqm,
    series_label,
    settled_steps,
    steps_to_threshold,
    varying_fields,
)

RESAMPLES = 200
GRID_STEP = 10
LAST_STEP = 1000
CROSSING_STEP = 400
GRID_POINTS = len(np.arange(0, LAST_STEP + 1, GRID_STEP))
SETTLE_POINTS = 100
CLIMB_POINTS = 10


def one_stratum(n_seeds: int) -> np.ndarray:
    """Stratum labels putting every seed in the same stratum."""
    return np.zeros(n_seeds, dtype=int)

def default_rng() -> np.random.Generator:
    """A generator fixed so a bootstrap interval is reproducible across runs."""
    return np.random.default_rng(0)

def episodes(crossing_seeds: List[int], n_seeds: int) -> pd.DataFrame:
    """Episodes where only `crossing_seeds` ever return 1, on steps that land on the grid."""
    steps = np.arange(0, LAST_STEP + 1, GRID_STEP)
    return pd.concat([
        pd.DataFrame({
            "cell": "group/cell", "group": "group", "env_id": "env",
            "condition": "option", "family": "grammar",
            "n_options": 64, "option_seed": 0, "tag": "test", "seed": seed,
            "primitive_step": steps,
            "episodic_return": np.where(
                (seed in crossing_seeds) & (steps >= CROSSING_STEP), 1.0, 0.0
            ),
        })
        for seed in range(n_seeds)
    ], ignore_index=True)

def crossings(crossing_seeds: List[int], n_seeds: int) -> pd.Series:
    """The `steps_to_threshold` row for one cell, with smoothing switched off."""
    frame = episodes(crossing_seeds, n_seeds)
    arguments = argparse.Namespace(
        bins=GRID_POINTS, window=1, threshold=0.5, resamples=RESAMPLES
    )
    data = Inputs(frame, pd.DataFrame(), arguments)
    return steps_to_threshold(data, frame, default_rng()).iloc[0]

def settle_steps() -> np.ndarray:
    """The x positions the curves in `curve_table` are sampled at."""
    return np.linspace(0.0, float(LAST_STEP), SETTLE_POINTS)

def curve_table(curves: Dict[str, np.ndarray]) -> pd.DataFrame:
    """An aggregate curve table holding one named point estimate series per entry."""
    steps = settle_steps()
    return pd.concat([
        pd.DataFrame({"cell": name, "condition": "option", "primitive_step": steps,
                      "point": values, "limit": float(LAST_STEP)})
        for name, values in curves.items()
    ], ignore_index=True)

def settling_curve() -> np.ndarray:
    """A curve that climbs to one over the first tenth of the axis and then holds."""
    return np.concatenate([np.linspace(0.0, 1.0, CLIMB_POINTS),
                           np.ones(SETTLE_POINTS - CLIMB_POINTS)])

def climbing_curve() -> np.ndarray:
    """A curve still rising at the last point of the axis."""
    return np.linspace(0.0, 0.5, SETTLE_POINTS)


def test_iqm_discards_the_tails() -> None:
    """The interquartile mean ignores the extreme quarter at each end."""
    values = np.array([-1000.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1000.0])
    assert iqm(values) == pytest.approx(np.mean(values[2:8]))
    assert iqm(values) != pytest.approx(np.mean(values))

def test_estimate_bootstraps_at_or_above_the_cutoff() -> None:
    """At the cutoff the point is the IQM and the interval brackets it."""
    values = np.arange(float(MIN_SEEDS_FOR_IQM))
    result = estimate(values, one_stratum(values.size), RESAMPLES, default_rng())
    assert result.method == "iqm-bootstrap"
    assert result.n_seeds == MIN_SEEDS_FOR_IQM
    assert float(result.point) == pytest.approx(iqm(values))
    assert float(result.low) <= float(result.point) <= float(result.high)

def test_estimate_falls_back_below_the_cutoff() -> None:
    """With too few seeds the estimate is the median and the observed range."""
    values = np.array([1.0, 5.0, 9.0])
    result = estimate(values, one_stratum(values.size), RESAMPLES, default_rng())
    assert result.method == "median-range"
    assert (float(result.point), float(result.low), float(result.high)) == (5.0, 1.0, 9.0)

def test_estimate_carries_the_trailing_axis() -> None:
    """A whole curve is estimated in one call, the seed axis leading."""
    curves = np.tile(np.arange(4.0), (MIN_SEEDS_FOR_IQM + 2, 1))
    result = estimate(curves, one_stratum(curves.shape[0]), RESAMPLES, default_rng())
    assert result.point.shape == (4,)
    # every seed holds the same curve, so no resample can move the estimate
    assert result.point == pytest.approx(np.arange(4.0))
    assert result.low == pytest.approx(result.high)

def test_bootstrap_indices_stay_inside_their_stratum() -> None:
    """A resampled seed is always drawn from the stratum it replaces."""
    strata = np.array([0, 0, 0, 1, 1, 1])
    indices = bootstrap_indices(strata, RESAMPLES, default_rng())
    assert indices.shape == (RESAMPLES, strata.size)
    assert (strata[indices] == strata).all()

def test_steps_to_threshold_excludes_seeds_that_never_cross() -> None:
    """A seed that never reaches the threshold is dropped, not pinned to the limit."""
    row = crossings(crossing_seeds=[0, 1, 2], n_seeds=4)
    assert (row.n_crossed, row.n_seeds) == (3, 4)
    assert row.point == pytest.approx(CROSSING_STEP)
    assert row.method == "median-range"

def test_steps_to_threshold_reports_nothing_when_most_seeds_fail() -> None:
    """Below half the seeds crossing, only the fraction is reported."""
    row = crossings(crossing_seeds=[0], n_seeds=4)
    assert (row.n_crossed, row.n_seeds) == (1, 4)
    assert np.isnan(row.point)
    assert row.method == "censored"

def test_crossing_step_reads_the_episode_not_a_grid_point() -> None:
    """The crossing is the step of the episode that crossed, at full resolution."""
    run = pd.DataFrame({"primitive_step": [100, 200, 4_321, 900_000],
                        "episodic_return": [0.0, 0.0, 1.0, 1.0]})
    assert crossing_step(run, "episodic_return", 0.5, 1) == 4_321.0

def test_crossing_step_needs_a_full_window() -> None:
    """A lucky first episode crosses only once a whole window of them averages past it."""
    run = pd.DataFrame({"primitive_step": [10, 20, 30],
                        "episodic_return": [1.0, 0.0, 0.0]})
    assert crossing_step(run, "episodic_return", 0.5, 1) == 10.0
    assert np.isnan(crossing_step(run, "episodic_return", 0.5, 3))

def test_collinear_detects_a_determined_column() -> None:
    """A return taking one value per termination outcome is determined by it."""
    frame = pd.DataFrame({"terminated": [0, 0, 1, 1],
                          "episodic_return": [0.0, 0.0, 1.0, 1.0]})
    assert collinear(frame, "terminated", "episodic_return")

def test_collinear_passes_a_dense_reward() -> None:
    """A return that varies among terminated episodes carries its own information."""
    frame = pd.DataFrame({"terminated": [0, 0, 1, 1],
                          "episodic_return": [0.0, 0.1, 0.9, 1.0]})
    assert not collinear(frame, "terminated", "episodic_return")

def test_series_label_omits_a_field_that_cannot_apply() -> None:
    """An action cell has no catalogue, so a varying `n_options` never names it."""
    table = pd.DataFrame({
        "cell": ["a", "b", "c"], "env_id": ["env"] * 3,
        "condition": ["action", "option", "option"],
        "family": ["-", "grammar", "grammar"], "option_seed": [0, 0, 0],
        "n_options": [0, 8, 64], "tag": ["exp2"] * 3,
    })
    varying = varying_fields(table)
    assert varying == ["condition", "n_options"]
    assert [series_label(row, varying) for _, row in table.iterrows()] == [
        "action space", "option space, n=8", "option space, n=64",
    ]

def test_settled_steps_ignores_a_curve_that_never_moved() -> None:
    """A cell flat at its floor never learned, so it is not a curve that settled."""
    table = curve_table({"settles": settling_curve(),
                         "flat": np.zeros(SETTLE_POINTS)})
    assert settled_steps(table) == [pytest.approx(settle_steps()[CLIMB_POINTS - 1])]

def test_x_scale_is_logarithmic_when_the_fastest_curve_settles_early() -> None:
    """One cell settling inside a fraction of the axis logs it, however slow the rest."""
    figure, axis = plt.subplots()
    apply_x_scale([axis], curve_table({"settles": settling_curve(),
                                       "climbs": climbing_curve()}))
    assert axis.get_xscale() == "log"
    assert axis.get_xlim() == (pytest.approx(settle_steps()[1]),
                              pytest.approx(float(LAST_STEP)))
    plt.close(figure)

def test_x_scale_is_linear_from_zero_while_a_curve_still_changes() -> None:
    """With nothing settled the axis stays linear and starts before the first episode."""
    figure, axis = plt.subplots()
    apply_x_scale([axis], curve_table({"climbs": climbing_curve()}))
    assert axis.get_xscale() == "linear"
    assert axis.get_xlim() == (0.0, pytest.approx(float(LAST_STEP)))
    plt.close(figure)
