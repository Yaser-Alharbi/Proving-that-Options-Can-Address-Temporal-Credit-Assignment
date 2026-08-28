"""Tests for the aggregation core of `plot.py`, on hand-built inputs."""

import argparse
from typing import Dict, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from matplotlib.colors import to_rgb
from plot import (
    BASE_DELAY,
    CENSOR_FRACTION,
    CONDITION_COLOR,
    CONDITION_LABEL,
    DEFAULT_THRESHOLD,
    GRAMMAR_COLOR,
    MIN_SEEDS_FOR_IQM,
    NO_INTERACTION,
    Inputs,
    MissingData,
    apply_x_scale,
    bootstrap_indices,
    collinear,
    condition_threshold,
    count_overlay,
    crossing_step,
    delay_advantage,
    delay_crossing_fraction,
    delay_slack,
    delay_sweep,
    duration_vs_cap,
    duration_vs_delay,
    estimate,
    iqm,
    never_crossed,
    return_curve,
    series_label,
    settled_steps,
    shared_across_modes,
    steps_to_threshold,
    threshold_for,
    varying_fields,
)

RESAMPLES = 200
GRID_STEP = 10
LAST_STEP = 1000
CROSSING_STEP = 400
GRID_POINTS = len(np.arange(0, LAST_STEP + 1, GRID_STEP))
SETTLE_POINTS = 100
CLIMB_POINTS = 10

DELAY_SEEDS = MIN_SEEDS_FOR_IQM
"""Enough seeds for the bootstrap branch, so a figure's method matches the real runs."""

DELAY_DELAYS: Tuple[int, ...] = (0, 32)
DELAY_MAX_STEPS = 400
DELAY_DISCOUNTS: Tuple[str, ...] = ("decision", "primitive")
DELAY_TAIL = 0.1

CROSSING_AT: Dict[Tuple[str, int], int] = {
    ("action", 0): 100, ("action", 32): 200,
    ("option", 0): 100, ("option", 32): 100,
}
"""Steps to the threshold: action degrades with delay, option does not."""

SOLVE_LENGTH: Dict[str, int] = {"action": 380, "option": 100}
"""Primitive steps to the goal, chosen so action at delay 32 pins at `DELAY_MAX_STEPS`."""

OPTION_DURATION: Dict[str, float] = {"action": 1.0, "option": 4.0}


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

def overlay_episodes() -> pd.DataFrame:
    """An action baseline and two option conditions, each at two catalogue sizes."""
    # from GRID_STEP, not zero: a geometric grid cannot start at a step of zero
    steps = np.arange(GRID_STEP, LAST_STEP + 1, GRID_STEP)
    cells = (("action", 0), ("option", 8), ("option", 64), ("both", 8), ("both", 64))
    return pd.concat([
        pd.DataFrame({
            "cell": f"group/{condition}-n{n_options}", "group": "group", "env_id": "env",
            "condition": condition, "family": "-" if condition == "action" else "grammar",
            "n_options": n_options, "option_seed": 0, "tag": "exp2", "seed": seed,
            "primitive_step": steps,
            "episodic_return": np.full(steps.size, n_options / 256.0),
        })
        for condition, n_options in cells for seed in range(2)
    ], ignore_index=True)

def overlay_args() -> argparse.Namespace:
    """Options an overlay figure reads, with smoothing switched off."""
    return argparse.Namespace(bins=GRID_POINTS, window=1, threshold=None,
                              resamples=RESAMPLES)


def draw_episodes() -> pd.DataFrame:
    """Grammar os=0 and two random draws, all option condition."""
    steps = np.arange(GRID_STEP, LAST_STEP + 1, GRID_STEP)
    cells = (("grammar", 0), ("random", 0), ("random", 1))
    return pd.concat([
        pd.DataFrame({
            "cell": f"group/{family}-os{option_seed}", "group": "group", "env_id": "env",
            "condition": "option", "family": family,
            "n_options": 64, "option_seed": option_seed, "tag": "exp3", "seed": seed,
            "primitive_step": steps,
            "episodic_return": np.full(steps.size, (option_seed + 1) / 4.0),
        })
        for family, option_seed in cells for seed in range(2)
    ], ignore_index=True)

def delay_episodes() -> pd.DataFrame:
    """Both conditions at two delays under both discount modes, every seed identical.

    Identical seeds make each point estimate exact and collapse its interval onto it, so a
    figure can be asserted against arithmetic rather than against a bootstrap draw.
    """
    steps = np.arange(0, LAST_STEP + 1, GRID_STEP)
    frames = []
    for discount in DELAY_DISCOUNTS:
        for condition in ("action", "option"):
            for delay in DELAY_DELAYS:
                solved = np.where(steps >= CROSSING_AT[(condition, delay)], 1.0, 0.0)
                length = min(SOLVE_LENGTH[condition] + delay, DELAY_MAX_STEPS)
                frames += [
                    pd.DataFrame({
                        "cell": f"group/{discount}-{condition}-d{delay}", "group": "group",
                        "env_id": "env", "condition": condition,
                        "family": "-" if condition == "action" else "grammar",
                        "n_options": 0 if condition == "action" else 64,
                        "option_seed": 0, "tag": "exp4", "seed": seed,
                        "primitive_step": steps, "episodic_return": solved,
                        "episodic_length": length, "terminated": solved.astype(int),
                        "max_steps": DELAY_MAX_STEPS, "reward_delay": delay,
                        "discount": discount,
                        "mean_option_duration": OPTION_DURATION[condition],
                    })
                    for seed in range(DELAY_SEEDS)
                ]
    return pd.concat(frames, ignore_index=True)

def delay_args(threshold: Union[float, Dict[str, float]] = 0.5) -> argparse.Namespace:
    """Options a delay figure reads, with smoothing switched off."""
    return argparse.Namespace(bins=GRID_POINTS, window=1, threshold=threshold,
                              resamples=RESAMPLES, tail=DELAY_TAIL)

def without_crossings(frame: pd.DataFrame, condition: str, delay: int,
                      seeds: Sequence[int]) -> pd.DataFrame:
    """`frame` with the named seeds of one condition and delay never reaching the threshold."""
    frame = frame.copy()
    target = ((frame.condition == condition) & (frame.reward_delay == delay)
              & frame.seed.isin(list(seeds)))
    frame.loc[target, ["episodic_return", "terminated"]] = 0
    return frame


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

def test_series_label_names_a_catalogue_draw() -> None:
    """A shared condition is omitted, so a draw is `family, os=` as on the threshold table."""
    table = pd.DataFrame({
        "cell": ["g", "r0", "r1"], "env_id": ["env"] * 3,
        "condition": ["option"] * 3, "family": ["grammar", "random", "random"],
        "option_seed": [0, 0, 1], "n_options": [64] * 3, "tag": ["exp3"] * 3,
    })
    varying = varying_fields(table)
    assert varying == ["family", "option_seed"]
    assert [series_label(row, varying) for _, row in table.iterrows()] == [
        "grammar, os=0", "random, os=0", "random, os=1",
    ]

def duration_meta() -> pd.DataFrame:
    """Four option cells: one grammar, three random draws, two distinct caps."""
    # random rows are not insertion-ordered by measured mean, so a sort is observable
    rows = (
        ("random", 2, 6.33, 9.76, 35.03, 49.0, 51.0),
        ("grammar", 0, 8.25, 6.44, 10.47, 11.0, 11.0),
        ("random", 0, 5.91, 6.63, 29.11, 43.0, 45.0),
        ("random", 1, 6.48, 7.96, 32.69, 49.0, 51.0),
    )
    return pd.DataFrame([
        {
            "cell": f"group/{family}-os{option_seed}", "group": "group", "env_id": "env",
            "condition": "option", "family": family, "n_options": 64,
            "option_seed": option_seed, "tag": "exp3",
            "nominal_option_len": nominal, "mean_option_len": measured,
            "duration_max_lane_mean": lane_mean, "duration_max_lane_max": lane_max,
            "max_option_len": cap,
        }
        for family, option_seed, nominal, measured, lane_mean, lane_max, cap in rows
    ])

def test_duration_vs_cap_separates_means_from_the_cap() -> None:
    """Left x-range is the means; grammar is pinned above random sorted by measured mean."""
    data = Inputs(pd.DataFrame(), duration_meta(), argparse.Namespace())
    figure, table = duration_vs_cap(data)
    means, lanes = figure.get_axes()
    assert [tick.get_text() for tick in means.get_yticklabels()] == [
        "grammar, os=0", "random, os=0", "random, os=1", "random, os=2",
    ]
    assert list(table.mean_option_len) == pytest.approx([6.44, 6.63, 7.96, 9.76])
    left = table[["nominal_option_len", "mean_option_len"]].to_numpy()
    left_pad = max(0.08 * float(left.max() - left.min()), 0.25)
    assert means.get_xlim() == (
        pytest.approx(float(left.min()) - left_pad),
        pytest.approx(float(left.max()) + left_pad),
    )
    assert means.get_xlim()[0] > 0
    right_end = float(table[["duration_max_lane_mean", "duration_max_lane_max",
                             "max_option_len"]].to_numpy().max())
    assert lanes.get_xlim() == (pytest.approx(0.0), pytest.approx(right_end * 1.08))
    plt.close(figure)


def test_count_overlay_facets_by_condition_and_keys_colour_to_count() -> None:
    """A panel per option condition, the baseline dashed in each, colour naming only n."""
    data = Inputs(overlay_episodes(), pd.DataFrame(), overlay_args())
    figure, table = count_overlay(data)
    axes = figure.get_axes()
    assert [axis.get_title() for axis in axes] == [CONDITION_LABEL["option"],
                                                   CONDITION_LABEL["both"]]
    per_count: Dict[str, set] = {}
    for axis in axes:
        assert [line.get_label() for line in axis.lines] == ["action space", "n=8", "n=64"]
        baseline, *counted = axis.lines
        assert to_rgb(baseline.get_color()) == pytest.approx(
            to_rgb(CONDITION_COLOR["action"])
        )
        assert baseline.get_linestyle() != "-"
        assert len({line.get_color() for line in counted}) == len(counted)
        for line in counted:
            per_count.setdefault(line.get_label(), set()).add(line.get_color())
    # one colour per catalogue size across the whole figure, or the legend cannot be shared
    assert all(len(colors) == 1 for colors in per_count.values())
    assert set(table.condition) == {"action", "option", "both"}
    plt.close(figure)

def test_count_overlay_needs_two_catalogue_sizes() -> None:
    """A single catalogue size has no axis to colour, so the figure is skipped."""
    frame = overlay_episodes()
    data = Inputs(frame[frame.n_options != 64], pd.DataFrame(), overlay_args())
    with pytest.raises(MissingData, match="two catalogue sizes"):
        count_overlay(data)


def test_return_curve_colours_lines_by_draw() -> None:
    """Grammar is the reference colour; random draws take the viridis ramp; each band matches its line."""
    data = Inputs(draw_episodes(), pd.DataFrame(), overlay_args())
    figure, _ = return_curve(data)
    axis = figure.get_axes()[0]
    grammar, random_0, random_1 = axis.lines
    assert [line.get_label() for line in axis.lines] == [
        "grammar, os=0", "random, os=0", "random, os=1",
    ]
    assert to_rgb(grammar.get_color()) == pytest.approx(to_rgb(GRAMMAR_COLOR))
    assert to_rgb(random_0.get_color()) != pytest.approx(to_rgb(GRAMMAR_COLOR))
    assert to_rgb(random_0.get_color()) != pytest.approx(to_rgb(random_1.get_color()))
    assert grammar.get_linestyle() == "-"
    assert random_0.get_linestyle() != "-"
    for line, collection in zip(axis.lines, axis.collections):
        assert to_rgb(collection.get_facecolor()[0]) == pytest.approx(to_rgb(line.get_color()))
    plt.close(figure)

def test_delay_crossing_fraction_marks_the_censoring_rule() -> None:
    """The fraction is crossing seeds over all seeds, drawn against the rule the table censors by."""
    frame = without_crossings(delay_episodes(), "action", 32, range(4, DELAY_SEEDS))
    figure, table = delay_crossing_fraction(Inputs(frame, pd.DataFrame(), delay_args()))
    assert (table.crossed_fraction == table.n_crossed / table.n_seeds).all()
    thinned = table[(table.condition == "action") & (table.reward_delay == 32)]
    assert set(thinned.crossed_fraction) == {CENSOR_FRACTION}
    # exactly on the rule, so the cell is reported rather than censored
    assert thinned.point.notna().all()
    for axis in figure.get_axes():
        rule = [line for line in axis.lines
                if "censoring rule" in str(line.get_label())]
        assert [line.get_ydata()[0] for line in rule] == [CENSOR_FRACTION]
    plt.close(figure)

def test_delay_slack_separates_compression_from_the_solve_length() -> None:
    """Row one is `episodic_length - delay` against a `max_steps - delay` line, row two the pinned solves."""
    figure, table = delay_slack(Inputs(delay_episodes(), pd.DataFrame(), delay_args()))
    keyed = table.set_index(["condition", "discount", "reward_delay"])
    assert keyed.loc[("action", "decision", 0), "point"] == pytest.approx(
        SOLVE_LENGTH["action"]
    )
    # 380 + 32 overruns max_steps, so the delay is compressed to 20 and the implied solve
    # length reads 368 rather than the 380 the episode actually took
    assert keyed.loc[("action", "decision", 32), "point"] == pytest.approx(
        DELAY_MAX_STEPS - 32
    )
    assert keyed.loc[("action", "decision", 0), "pinned_point"] == pytest.approx(0.0)
    assert keyed.loc[("action", "decision", 32), "pinned_point"] == pytest.approx(1.0)
    option = table[table.condition == "option"]
    assert option.point.to_numpy() == pytest.approx(SOLVE_LENGTH["option"])
    assert option.pinned_point.to_numpy() == pytest.approx(0.0)
    reference = next(line for line in figure.get_axes()[0].lines
                     if str(line.get_label()) == "max_steps − delay")
    assert list(reference.get_ydata()) == pytest.approx(
        [DELAY_MAX_STEPS - delay for delay in DELAY_DELAYS]
    )
    plt.close(figure)

def test_delay_slack_repeats_a_discount_free_action_arm() -> None:
    """`expand` runs the action arm in one mode, and both slack panels still need it."""
    episodes = delay_episodes()
    one_mode = episodes[(episodes.condition == "option") | (episodes.discount == "decision")]
    figure, table = delay_slack(Inputs(one_mode, pd.DataFrame(), delay_args()))
    for mode in DELAY_DISCOUNTS:
        assert set(table[table.discount == mode].condition) == {"action", "option"}
    plt.close(figure)

def test_delay_sweep_leaves_a_gap_where_a_cell_is_censored() -> None:
    """A censored cell keeps its row but is not drawn, so the line stops rather than dipping."""
    frame = without_crossings(delay_episodes(), "action", 32, range(1, DELAY_SEEDS))
    figure, table = delay_sweep(Inputs(frame, pd.DataFrame(), delay_args()))
    censored = table[(table.condition == "action") & (table.reward_delay == 32)]
    assert (censored.method == "censored").all()
    assert censored.point.isna().all()
    for axis in figure.get_axes():
        # an errorbar labels its container, not the Line2D, which stays `_nolegend_`
        drawn = {str(series.get_label()): list(series.lines[0].get_xdata())
                 for series in axis.containers}
        assert drawn[CONDITION_LABEL["action"]] == [BASE_DELAY]
        assert drawn[CONDITION_LABEL["option"]] == list(DELAY_DELAYS)
    plt.close(figure)

def test_threshold_for_resolves_a_bar_per_condition() -> None:
    """A mapping gives each arm its own bar; a bare override still replaces both."""
    frame = delay_episodes()
    action = frame[frame.condition == "action"]
    option = frame[frame.condition == "option"]
    mapped = Inputs(frame, pd.DataFrame(),
                    delay_args(threshold={"action": 0.15, "option": 0.95}))
    assert threshold_for(mapped, action) == pytest.approx(0.15)
    assert threshold_for(mapped, option) == pytest.approx(0.95)
    # an override naming one condition leaves the other to fall through to the default
    partial = Inputs(frame, pd.DataFrame(), delay_args(threshold={"option": 0.99}))
    assert threshold_for(partial, option) == pytest.approx(0.99)
    assert threshold_for(partial, action) == pytest.approx(DEFAULT_THRESHOLD)
    bare = Inputs(frame, pd.DataFrame(), delay_args(threshold=0.3))
    assert threshold_for(bare, action) == threshold_for(bare, option) == pytest.approx(0.3)

def test_condition_threshold_parses_both_forms() -> None:
    """`--threshold 0.5` is every condition; `--threshold option=0.99` is one of them."""
    assert condition_threshold("0.5") == pytest.approx(0.5)
    assert condition_threshold("option=0.99") == {"option": pytest.approx(0.99)}
    with pytest.raises(argparse.ArgumentTypeError):
        condition_threshold("options=0.99")

def test_shared_across_modes_repeats_a_discount_free_action_arm() -> None:
    """The action arm is run in one mode only, so one cell has to serve both panels."""
    table = pd.DataFrame({
        "cell": ["a", "o-dec", "o-prim"], "condition": ["action", "option", "option"],
        "discount": ["decision", "decision", "primitive"], "reward_delay": [0, 0, 0],
        "point": [100.0, 50.0, 50.0],
    })
    shared = shared_across_modes(table)
    assert len(shared) == 4
    for mode in ("decision", "primitive"):
        panel = shared[shared.discount == mode]
        assert set(panel.condition) == {"action", "option"}
        assert panel[panel.condition == "action"].point.to_numpy() == pytest.approx(100.0)
    # a sweep that did run the action arm per mode is left exactly as it was
    assert shared_across_modes(shared).equals(shared)

def test_never_crossed_names_only_a_condition_censored_at_every_delay() -> None:
    """A condition censored everywhere is a capability floor; one censored at some delay is not."""
    table = pd.DataFrame({
        "condition": ["action", "action", "option", "option"],
        "reward_delay": [0, 32, 0, 32],
        "point": [np.nan, np.nan, float(CROSSING_STEP), np.nan],
    })
    note = never_crossed(table)
    assert CONDITION_LABEL["action"] in note
    assert CONDITION_LABEL["option"] not in note
    assert never_crossed(table[table.condition == "option"]) == ""

def test_delay_advantage_is_exactly_one_at_the_base_delay() -> None:
    """Normalising inside the replicate pins the base delay to one with no interval around it."""
    figure, table = delay_advantage(Inputs(delay_episodes(), pd.DataFrame(), delay_args()))
    base = table[table.reward_delay == BASE_DELAY]
    for column in ("point", "low", "high"):
        assert base[column].to_numpy() == pytest.approx(NO_INTERACTION)
    assert set(table.discount) == set(DELAY_DISCOUNTS)
    plt.close(figure)

def test_delay_advantage_normalises_each_condition_to_its_own_base() -> None:
    """Each arm is in units of its own delay-0 crossing, so arms on different bars compare."""
    figure, table = delay_advantage(Inputs(delay_episodes(), pd.DataFrame(), delay_args()))
    delayed = table[table.reward_delay == 32].set_index("condition")
    for condition in ("action", "option"):
        slowdown = CROSSING_AT[(condition, 32)] / CROSSING_AT[(condition, 0)]
        assert delayed.loc[condition, "point"].to_numpy() == pytest.approx(slowdown)
        assert delayed.loc[condition, "steps"].to_numpy() == pytest.approx(
            CROSSING_AT[(condition, 32)]
        )
    # the signature the figure exists to show: delay doubles action and leaves option alone
    assert delayed.loc["action", "point"].to_numpy() == pytest.approx(2.0)
    assert delayed.loc["option", "point"].to_numpy() == pytest.approx(NO_INTERACTION)
    plt.close(figure)

def test_duration_vs_delay_excludes_the_action_condition() -> None:
    """An action cell takes one primitive step per decision, so only option cells are drawn."""
    figure, table = duration_vs_delay(Inputs(delay_episodes(), pd.DataFrame(), delay_args()))
    assert set(table.condition) == {"option"}
    assert table.point.to_numpy() == pytest.approx(OPTION_DURATION["option"])
    for axis in figure.get_axes():
        assert [str(series.get_label()) for series in axis.containers] == [
            CONDITION_LABEL["option"]
        ]
    plt.close(figure)


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
