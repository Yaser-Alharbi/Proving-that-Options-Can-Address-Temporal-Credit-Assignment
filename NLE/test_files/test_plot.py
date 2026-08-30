"""Aggregation-core tests for `plot.py`."""

import argparse
import pathlib
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb

import plot
from plot import (
    BASE_DELAY,
    CENSOR_FRACTION,
    CONDITION_COLOR,
    CONDITION_LABEL,
    DEFAULT_SOLVED_RATE,
    FAMILY_DASH,
    MIN_SEEDS_FOR_IQM,
    NO_INTERACTION,
    STRUCTURED_COLOR,
    Inputs,
    MissingData,
    apply_x_scale,
    bootstrap_indices,
    caption_for,
    collinear,
    condition_solved_rate,
    count_overlay,
    crossing_step,
    delay_advantage,
    delay_crossing_fraction,
    delay_slack,
    delay_sweep,
    duration_vs_delay,
    estimate,
    family_overlay,
    iqm,
    load_episodes,
    never_crossed,
    parse_arguments,
    return_curve,
    series_label,
    settled_steps,
    solved_rate_for,
    steps_to_threshold,
    threshold_table,
    varying_fields,
)

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

RESAMPLES = 200
GRID_STEP = 10
LAST_STEP = 1000
CROSSING_STEP = 400
GRID_POINTS = len(np.arange(0, LAST_STEP + 1, GRID_STEP))
SETTLE_POINTS = 100
CLIMB_POINTS = 10

UNPAID_RETURN = 0.2
"""A return below every solved rate used here."""

DELAY_SEEDS = MIN_SEEDS_FOR_IQM
"""Enough seeds for the bootstrap branch."""

DELAY_DELAYS: Tuple[int, ...] = (0, 32)
DELAY_MAX_STEPS = 400
DELAY_DISCOUNTS: Tuple[str, ...] = ("decision", "primitive")
DELAY_TAIL = 0.1

CROSSING_AT: Dict[Tuple[str, int], int] = {
    ("action", 0): 100, ("action", 32): 200,
    ("option", 0): 100, ("option", 32): 100,
}
"""Steps to the solved rate. Action degrades with delay; option does not."""

SOLVE_LENGTH: Dict[str, int] = {"action": 380, "option": 100}
"""Primitive steps to the goal. Action at delay 32 pins at `DELAY_MAX_STEPS`."""

OPTION_DURATION: Dict[str, float] = {"action": 1.0, "option": 4.0}


def one_stratum(n_seeds: int) -> np.ndarray:
    """Every seed in one stratum."""
    return np.zeros(n_seeds, dtype=int)

def default_rng() -> np.random.Generator:
    """Fixed generator for reproducible bootstrap intervals."""
    return np.random.default_rng(0)

def episodes(crossing_seeds: List[int], n_seeds: int) -> pd.DataFrame:
    """Episodes where only `crossing_seeds` are paid."""
    steps = np.arange(0, LAST_STEP + 1, GRID_STEP)
    return pd.concat([
        pd.DataFrame({
            "cell": "group/cell", "group": "group", "env_id": "env",
            "condition": "option", "family": "grammar",
            "n_options": 64, "option_seed": 0, "tag": "test", "seed": seed,
            "primitive_step": steps,
            "paid": np.where(
                (seed in crossing_seeds) & (steps >= CROSSING_STEP), 1, 0
            ),
            "episodic_return": UNPAID_RETURN,
        })
        for seed in range(n_seeds)
    ], ignore_index=True)

def crossings(crossing_seeds: List[int], n_seeds: int) -> pd.Series:
    """`steps_to_threshold` row for one cell, unsmoothed."""
    frame = episodes(crossing_seeds, n_seeds)
    return steps_to_threshold(
        Inputs(frame, overlay_args()), frame, default_rng()
    ).iloc[0]

def overlay_episodes() -> pd.DataFrame:
    """Action baseline and two option conditions at two catalogue sizes."""
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
    """Overlay-figure options with smoothing off."""
    return argparse.Namespace(bins=GRID_POINTS, window=1, solved_threshold=None,
                              resamples=RESAMPLES)

def draw_episodes() -> pd.DataFrame:
    """Both structured catalogues and two random draws."""
    steps = np.arange(GRID_STEP, LAST_STEP + 1, GRID_STEP)
    cells = (("grammar", 0), ("grammar_depth", 0), ("random", 0), ("random", 1))
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
    """Both conditions at two delays under both discount modes."""
    steps = np.arange(0, LAST_STEP + 1, GRID_STEP)
    frames = []
    for discount in DELAY_DISCOUNTS:
        for condition in ("action", "option"):
            for delay in DELAY_DELAYS:
                paid = np.where(steps >= CROSSING_AT[(condition, delay)], 1, 0)
                length = min(SOLVE_LENGTH[condition] + delay, DELAY_MAX_STEPS)
                frames += [
                    pd.DataFrame({
                        "cell": f"group/{discount}-{condition}-d{delay}", "group": "group",
                        "env_id": "env", "condition": condition,
                        "family": "-" if condition == "action" else "grammar",
                        "n_options": 0 if condition == "action" else 64,
                        "option_seed": 0, "tag": "exp4", "seed": seed,
                        "primitive_step": steps, "episodic_return": UNPAID_RETURN,
                        "paid": paid, "episodic_length": length,
                        "max_steps": DELAY_MAX_STEPS, "reward_delay": delay,
                        "discount": discount,
                        "mean_option_duration": OPTION_DURATION[condition],
                    })
                    for seed in range(DELAY_SEEDS)
                ]
    return pd.concat(frames, ignore_index=True)

def delay_args(rate: Union[float, Dict[str, float]] = 0.5) -> argparse.Namespace:
    """Delay-figure options with smoothing off."""
    return argparse.Namespace(bins=GRID_POINTS, window=1, solved_threshold=rate,
                              resamples=RESAMPLES, tail=DELAY_TAIL)

def without_crossings(frame: pd.DataFrame, condition: str, delay: int,
                      seeds: Sequence[int]) -> pd.DataFrame:
    """`frame` with the named seeds never paid."""
    frame = frame.copy()
    target = ((frame.condition == condition) & (frame.reward_delay == delay)
              & frame.seed.isin(list(seeds)))
    frame.loc[target, "paid"] = 0
    return frame

def settle_steps() -> np.ndarray:
    """X positions `curve_table` samples at."""
    return np.linspace(0.0, float(LAST_STEP), SETTLE_POINTS)

def curve_table(curves: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Aggregate table of named point-estimate series."""
    steps = settle_steps()
    return pd.concat([
        pd.DataFrame({"cell": name, "condition": "option", "primitive_step": steps,
                      "point": values, "limit": float(LAST_STEP)})
        for name, values in curves.items()
    ], ignore_index=True)

def settling_curve() -> np.ndarray:
    """Climbs to one over the first tenth, then holds."""
    return np.concatenate([np.linspace(0.0, 1.0, CLIMB_POINTS),
                           np.ones(SETTLE_POINTS - CLIMB_POINTS)])

def climbing_curve() -> np.ndarray:
    """Still rising at the last point."""
    return np.linspace(0.0, 0.5, SETTLE_POINTS)


def test_iqm_discards_the_tails() -> None:
    """IQM ignores the extreme quarter at each end."""
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
    """Below the cutoff: median and observed range."""
    values = np.array([1.0, 5.0, 9.0])
    result = estimate(values, one_stratum(values.size), RESAMPLES, default_rng())
    assert result.method == "median-range"
    assert (float(result.point), float(result.low), float(result.high)) == (5.0, 1.0, 9.0)

def test_estimate_carries_the_trailing_axis() -> None:
    """A whole curve is estimated in one call."""
    curves = np.tile(np.arange(4.0), (MIN_SEEDS_FOR_IQM + 2, 1))
    result = estimate(curves, one_stratum(curves.shape[0]), RESAMPLES, default_rng())
    assert result.point.shape == (4,)
    assert result.point == pytest.approx(np.arange(4.0))
    assert result.low == pytest.approx(result.high)

def test_bootstrap_indices_stay_inside_their_stratum() -> None:
    """A resampled seed stays inside its stratum."""
    strata = np.array([0, 0, 0, 1, 1, 1])
    indices = bootstrap_indices(strata, RESAMPLES, default_rng())
    assert indices.shape == (RESAMPLES, strata.size)
    assert (strata[indices] == strata).all()


def test_load_episodes_unions_the_per_seed_files(monkeypatch: "MonkeyPatch",
                                                 tmp_path: pathlib.Path) -> None:
    """A cell's seeds live in one file each; the union is the table."""
    directory = tmp_path / "0101-000000__env__exp1" / "env__action__exp1"
    directory.mkdir(parents=True)
    for seed in (0, 1):
        pd.DataFrame({"seed": [seed], "primitive_step": [10], "paid": [1],
                      "tag": ["exp1"]}).to_csv(
            directory / f"episodes_seed{seed}.csv", index=False
        )
    monkeypatch.setattr(plot, "RUNS", tmp_path)

    frame = load_episodes(["*"])
    assert sorted(frame.seed) == [0, 1]
    assert set(frame.cell) == {"0101-000000__env__exp1/env__action__exp1"}
    assert set(frame.group) == {"0101-000000__env__exp1"}


def test_steps_to_threshold_reads_the_payout_not_the_return() -> None:
    """The crossing is on `paid`, not `episodic_return`."""
    row = crossings(crossing_seeds=[0, 1, 2, 3], n_seeds=4)
    assert row.point == pytest.approx(CROSSING_STEP)
    assert row.n_crossed == 4
    assert UNPAID_RETURN < row.solved_rate

def test_steps_to_threshold_excludes_seeds_that_never_cross() -> None:
    """A seed that never crosses is dropped, not pinned."""
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
    """The crossing is the episode's step, not a grid point."""
    run = pd.DataFrame({"primitive_step": [100, 200, 4_321, 900_000],
                        "paid": [0, 0, 1, 1]})
    assert crossing_step(run, "paid", 0.5, 1) == 4_321.0

def test_crossing_step_needs_a_full_window() -> None:
    """Crossing requires a full window above the bar."""
    run = pd.DataFrame({"primitive_step": [10, 20, 30], "paid": [1, 0, 0]})
    assert crossing_step(run, "paid", 0.5, 1) == 10.0
    assert np.isnan(crossing_step(run, "paid", 0.5, 3))

def test_threshold_table_skips_where_no_cell_crossed() -> None:
    """Nobody paid raises `MissingData`."""
    frame = episodes(crossing_seeds=[], n_seeds=4)
    with pytest.raises(MissingData, match="censored"):
        threshold_table(Inputs(frame, overlay_args()))

def test_threshold_table_draws_where_a_cell_crossed() -> None:
    """One uncensored cell is enough for the table."""
    frame = episodes(crossing_seeds=[0, 1, 2, 3], n_seeds=4)
    figure, table = threshold_table(Inputs(frame, overlay_args()))
    assert table.point.to_numpy() == pytest.approx(CROSSING_STEP)
    plt.close(figure)

def test_collinear_detects_a_determined_column() -> None:
    """A return constant given payout is determined by it."""
    frame = pd.DataFrame({"paid": [0, 0, 1, 1],
                          "episodic_return": [0.0, 0.0, 1.0, 1.0]})
    assert collinear(frame, "paid", "episodic_return")

def test_collinear_passes_a_dense_reward() -> None:
    """A return that varies among paid episodes is not collinear."""
    frame = pd.DataFrame({"paid": [0, 0, 1, 1],
                          "episodic_return": [0.0, 0.1, 0.9, 1.0]})
    assert not collinear(frame, "paid", "episodic_return")

def test_series_label_omits_a_field_that_cannot_apply() -> None:
    """An action cell is not named by `n_options`."""
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
    """A shared condition is omitted from the label."""
    table = pd.DataFrame({
        "cell": ["g", "d", "r0"], "env_id": ["env"] * 3,
        "condition": ["option"] * 3,
        "family": ["grammar", "grammar_depth", "random"],
        "option_seed": [0, 0, 1], "n_options": [64] * 3, "tag": ["exp3"] * 3,
    })
    varying = varying_fields(table)
    assert varying == ["family", "option_seed"]
    assert [series_label(row, varying) for _, row in table.iterrows()] == [
        "grammar, os=0", "grammar_depth, os=0", "random, os=1",
    ]

def identity_table() -> pd.DataFrame:
    """One option cell with every `IDENTITY` column off its default."""
    return pd.DataFrame([{
        "env_id": "NetHackScore-v0", "condition": "option", "family": "grammar_depth",
        "option_seed": 3, "budget": 5_000_000, "max_steps": 2_000, "reward_delay": 32,
        "gamma": 0.99, "discount": "primitive", "tag": "exp4", "n_options": 8,
    }])

def test_caption_for_resolves_every_identity_column() -> None:
    """Every `IDENTITY` name resolves to a `CellArgs` field."""
    table = identity_table()
    assert set(plot.IDENTITY) <= set(table.columns), "fixture stopped covering IDENTITY"
    caption = caption_for(table, [])
    for written in ("grammar_depth catalogue", "n=8", "os=3", "budget=5000000",
                    "max_steps=2000", "reward_delay=32", "gamma=0.99",
                    "discount=primitive"):
        assert written in caption


def test_return_curve_keeps_both_structured_families_off_the_ramp() -> None:
    """Both grammar families stay off the random ramp."""
    data = Inputs(draw_episodes(), overlay_args())
    figure, _ = return_curve(data)
    axis = figure.get_axes()[0]
    grammar, depth, random_0, random_1 = axis.lines
    assert [line.get_label() for line in axis.lines] == [
        "grammar, os=0", "grammar_depth, os=0", "random, os=0", "random, os=1",
    ]
    assert to_rgb(grammar.get_color()) == pytest.approx(to_rgb(STRUCTURED_COLOR["grammar"]))
    assert to_rgb(depth.get_color()) == pytest.approx(
        to_rgb(STRUCTURED_COLOR["grammar_depth"])
    )
    structured = {to_rgb(grammar.get_color()), to_rgb(depth.get_color())}
    for drawn in (random_0, random_1):
        assert to_rgb(drawn.get_color()) not in structured
    assert to_rgb(random_0.get_color()) != pytest.approx(to_rgb(random_1.get_color()))
    assert [line.get_linestyle() for line in axis.lines] == [
        FAMILY_DASH["grammar"], FAMILY_DASH["grammar_depth"],
        FAMILY_DASH["random"], FAMILY_DASH["random"],
    ]
    for line, collection in zip(axis.lines, axis.collections):
        assert to_rgb(collection.get_facecolor()[0]) == pytest.approx(to_rgb(line.get_color()))
    plt.close(figure)

def disjoint_window_episodes() -> pd.DataFrame:
    """Option covered; action seeds never overlap."""
    covered = np.arange(GRID_STEP, LAST_STEP + 1, GRID_STEP)
    spans = {0: np.arange(GRID_STEP, LAST_STEP // 2, GRID_STEP),
             1: np.arange(LAST_STEP, 2 * LAST_STEP + 1, GRID_STEP)}
    frames = []
    for condition in ("action", "option"):
        for seed in (0, 1):
            steps = spans[seed] if condition == "action" else covered
            frames.append(pd.DataFrame({
                "cell": f"group/{condition}", "group": "group", "env_id": "env",
                "condition": condition,
                "family": "-" if condition == "action" else "grammar",
                "n_options": 0 if condition == "action" else 64, "option_seed": 0,
                "tag": "exp1", "seed": seed, "primitive_step": steps,
                "episodic_return": np.full(steps.size, 0.5),
                "solved": np.full(steps.size, int(condition == "action")),
            }))
    return pd.concat(frames, ignore_index=True)

def test_omitted_note_names_the_condition_the_window_dropped() -> None:
    """A window-dropped condition is named on the figure."""
    frame = disjoint_window_episodes()
    figure, table = return_curve(Inputs(frame, overlay_args()))
    assert set(table.condition) == {"option"}
    printed = "\n".join(text.get_text() for text in figure.texts)
    assert "action space omitted: solved 100.00%" in printed
    plt.close(figure)

def test_family_overlay_takes_any_second_family() -> None:
    """The overlay takes any second family."""
    figure, table = family_overlay(Inputs(draw_episodes(), overlay_args()))
    assert set(table.family) == {"grammar", "grammar_depth", "random"}
    assert [axis.get_title() for axis in figure.get_axes()] == ["n = 64"]
    plt.close(figure)

def test_family_overlay_needs_a_second_family() -> None:
    """One family skips the figure."""
    frame = draw_episodes()
    with pytest.raises(MissingData, match="more than one family"):
        family_overlay(Inputs(frame[frame.family == "grammar"], overlay_args()))

def test_count_overlay_facets_by_condition_and_keys_colour_to_count() -> None:
    """One panel per option condition; colour keys to n."""
    data = Inputs(overlay_episodes(), overlay_args())
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
    assert all(len(colors) == 1 for colors in per_count.values())
    assert set(table.condition) == {"action", "option", "both"}
    plt.close(figure)

def test_count_overlay_needs_two_catalogue_sizes() -> None:
    """One catalogue size skips the figure."""
    frame = overlay_episodes()
    data = Inputs(frame[frame.n_options != 64], overlay_args())
    with pytest.raises(MissingData, match="two catalogue sizes"):
        count_overlay(data)


def test_delay_crossing_fraction_marks_the_censoring_rule() -> None:
    """Crossing fraction is drawn against the censoring rule."""
    frame = without_crossings(delay_episodes(), "action", 32, range(4, DELAY_SEEDS))
    figure, table = delay_crossing_fraction(Inputs(frame, delay_args()))
    assert (table.crossed_fraction == table.n_crossed / table.n_seeds).all()
    thinned = table[(table.condition == "action") & (table.reward_delay == 32)]
    assert set(thinned.crossed_fraction) == {CENSOR_FRACTION}
    assert thinned.point.notna().all()
    for axis in figure.get_axes():
        rule = [line for line in axis.lines
                if "censoring rule" in str(line.get_label())]
        assert [line.get_ydata()[0] for line in rule] == [CENSOR_FRACTION]
    plt.close(figure)

def test_delay_slack_separates_compression_from_the_solve_length() -> None:
    """Slack is `episodic_length - delay` against `max_steps - delay`."""
    figure, table = delay_slack(Inputs(delay_episodes(), delay_args()))
    keyed = table.set_index(["condition", "discount", "reward_delay"])
    assert keyed.loc[("action", "decision", 0), "point"] == pytest.approx(
        SOLVE_LENGTH["action"]
    )
    # horizon-compressed: max_steps - delay, not SOLVE_LENGTH
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
    """The action arm appears in both discount panels."""
    frame = delay_episodes()
    one_mode = frame[(frame.condition == "option") | (frame.discount == "decision")]
    figure, table = delay_slack(Inputs(one_mode, delay_args()))
    for mode in DELAY_DISCOUNTS:
        assert set(table[table.discount == mode].condition) == {"action", "option"}
    plt.close(figure)

def test_delay_sweep_leaves_a_gap_where_a_cell_is_censored() -> None:
    """A censored cell is not drawn."""
    frame = without_crossings(delay_episodes(), "action", 32, range(1, DELAY_SEEDS))
    figure, table = delay_sweep(Inputs(frame, delay_args()))
    censored = table[(table.condition == "action") & (table.reward_delay == 32)]
    assert (censored.method == "censored").all()
    assert censored.point.isna().all()
    for axis in figure.get_axes():
        # errorbar labels its container, not the Line2D
        drawn = {str(series.get_label()): list(series.lines[0].get_xdata())
                 for series in axis.containers}
        assert drawn[CONDITION_LABEL["action"]] == [BASE_DELAY]
        assert drawn[CONDITION_LABEL["option"]] == list(DELAY_DELAYS)
    plt.close(figure)

def test_solved_rate_for_resolves_a_bar_per_condition() -> None:
    """A mapping sets a bar per condition; a float replaces both."""
    frame = delay_episodes()
    action = frame[frame.condition == "action"]
    option = frame[frame.condition == "option"]
    mapped = Inputs(frame, delay_args({"action": 0.15, "option": 0.95}))
    assert solved_rate_for(mapped, action) == pytest.approx(0.15)
    assert solved_rate_for(mapped, option) == pytest.approx(0.95)
    partial = Inputs(frame, delay_args({"option": 0.99}))
    assert solved_rate_for(partial, option) == pytest.approx(0.99)
    assert solved_rate_for(partial, action) == pytest.approx(DEFAULT_SOLVED_RATE)
    bare = Inputs(frame, delay_args(0.3))
    assert solved_rate_for(bare, action) == solved_rate_for(bare, option) == pytest.approx(0.3)

def test_condition_solved_rate_parses_both_forms() -> None:
    """Parses a bare float and `condition=value`."""
    assert condition_solved_rate("0.5") == pytest.approx(0.5)
    assert condition_solved_rate("option=0.9") == {"option": pytest.approx(0.9)}
    with pytest.raises(argparse.ArgumentTypeError):
        condition_solved_rate("options=0.9")

def test_the_bar_is_named_for_the_payout_rate_it_is() -> None:
    """`--threshold` is not an alias of `--solved-threshold`."""
    assert parse_arguments(["--solved-threshold", "0.9"]).solved_threshold == 0.9
    assert parse_arguments([]).solved_threshold is None
    with pytest.raises(SystemExit):
        parse_arguments(["--threshold", "0.9"])

def test_never_crossed_names_only_a_condition_censored_at_every_delay() -> None:
    """Only a condition censored at every delay is named."""
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
    """The base delay normalises to one with no interval."""
    figure, table = delay_advantage(Inputs(delay_episodes(), delay_args()))
    base = table[table.reward_delay == BASE_DELAY]
    for column in ("point", "low", "high"):
        assert base[column].to_numpy() == pytest.approx(NO_INTERACTION)
    assert set(table.discount) == set(DELAY_DISCOUNTS)
    plt.close(figure)

def test_delay_advantage_normalises_each_condition_to_its_own_base() -> None:
    """Each arm normalises to its own delay-0 crossing."""
    figure, table = delay_advantage(Inputs(delay_episodes(), delay_args()))
    delayed = table[table.reward_delay == 32].set_index("condition")
    for condition in ("action", "option"):
        slowdown = CROSSING_AT[(condition, 32)] / CROSSING_AT[(condition, 0)]
        assert delayed.loc[condition, "point"].to_numpy() == pytest.approx(slowdown)
        assert delayed.loc[condition, "steps"].to_numpy() == pytest.approx(
            CROSSING_AT[(condition, 32)]
        )
    assert delayed.loc["action", "point"].to_numpy() == pytest.approx(2.0)
    assert delayed.loc["option", "point"].to_numpy() == pytest.approx(NO_INTERACTION)
    plt.close(figure)

def test_duration_vs_delay_excludes_the_action_condition() -> None:
    """Only option cells are drawn."""
    figure, table = duration_vs_delay(Inputs(delay_episodes(), delay_args()))
    assert set(table.condition) == {"option"}
    assert table.point.to_numpy() == pytest.approx(OPTION_DURATION["option"])
    for axis in figure.get_axes():
        assert [str(series.get_label()) for series in axis.containers] == [
            CONDITION_LABEL["option"]
        ]
    plt.close(figure)


def test_settled_steps_ignores_a_curve_that_never_moved() -> None:
    """A flat curve is not treated as settled."""
    table = curve_table({"settles": settling_curve(),
                         "flat": np.zeros(SETTLE_POINTS)})
    assert settled_steps(table) == [pytest.approx(settle_steps()[CLIMB_POINTS - 1])]

def test_x_scale_is_logarithmic_when_the_fastest_curve_settles_early() -> None:
    """An early settle logs the x-axis."""
    figure, axis = plt.subplots()
    apply_x_scale([axis], curve_table({"settles": settling_curve(),
                                       "climbs": climbing_curve()}))
    assert axis.get_xscale() == "log"
    assert axis.get_xlim() == (pytest.approx(settle_steps()[1]),
                               pytest.approx(float(LAST_STEP)))
    plt.close(figure)

def test_x_scale_is_linear_from_zero_while_a_curve_still_changes() -> None:
    """With nothing settled the axis stays linear from zero."""
    figure, axis = plt.subplots()
    apply_x_scale([axis], curve_table({"climbs": climbing_curve()}))
    assert axis.get_xscale() == "linear"
    assert axis.get_xlim() == (0.0, pytest.approx(float(LAST_STEP)))
    plt.close(figure)
