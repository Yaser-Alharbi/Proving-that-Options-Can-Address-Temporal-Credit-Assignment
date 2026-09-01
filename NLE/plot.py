"""Figures for the NLE experiment matrix, one registry entry each.

Every figure returns the frame it drew, and the runner writes that frame beside the image,
so a figure can be regenerated without reading `runs/` again.
"""

import argparse
import fnmatch
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from scipy.stats import trim_mean

from main import RUNS, CellArgs

matplotlib.use("Agg")

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"

MIN_SEEDS_FOR_IQM = 8
"""Below this a 25% trim keeps too few values to be a trimmed mean."""

IQM_TRIM = 0.25
CI_ALPHA = 0.05
BOOTSTRAP_SEED = 0

FIGURE_SIZE = (9.0, 5.5)
PANEL_WIDTH = 5.0
PANEL_HEIGHT = 3.2
ROW_HEIGHT = 0.4
LINE_WIDTH = 2.0
BAND_ALPHA = 0.2
SEED_LINE_ALPHA = 0.6
"""One line per seed of one colour per condition, so they have to be told apart where
they overlap."""
GRID_ALPHA = 0.3
SMALL_FONT = 7
LEGEND_FONT = 8
ERROR_CAP_SIZE = 4
X_LABEL = "primitive steps"

SCI_LIMITS = (-3, 4)
"""Tick magnitudes printed plainly; a step axis exceeds this, a duration does not."""

LEGEND_TOP = 0.0
"""Figure fraction the legend hangs from, so it clears the figure entirely."""

CROSSING_COLUMN = "paid"
"""Whether the episode was paid the goal reward. `solved` is `TASK_SUCCESSFUL` alone, which
a hold flushed by the horizon or by a death does not report even though the bank paid."""

DEFAULT_SOLVED_RATE = 0.5
"""Fraction of a window's episodes that must be paid for the cell to count as crossing.

A rate, not a return: an NLE return is the goal reward plus a per-step time penalty, so a
bar on it conflates solving with solving quickly and moves with the realised episode length.
"""

DENSE_RETURN_THRESHOLD = 60.0
"""Score a trailing mean must reach where the crossing is read off the return.

A rate cannot serve on a dense host: the return there is a score in the tens, so
`DEFAULT_SOLVED_RATE` is crossed by the first full window of every seed. Single
untrained episodes clear this bar; `crossing_step` reads the window mean, which
is what the bar is set against.
"""

TAIL_EPISODES = 1_000
"""Episodes from the end of each seed that `return_ecdf` draws."""

ECDF_ZOOM_LIMIT = 100.0
"""Right edge of `return_ecdf`'s detail panel.

A weak condition's whole distribution can fall inside a fraction of the range a strong
one's tail reaches, and on a single axis it draws as a vertical line at the origin.
"""

ZOOM_MARGIN = 0.05
"""Fraction of the detail panel's span left below the lowest score, so the episode that
scored it draws inside the panel rather than on its edge."""

CENSOR_FRACTION = 0.5
"""Seeds that must cross before a cell's steps-to-threshold is reported at all."""

BASE_DELAY = 0
"""The delay `delay_advantage` normalises to, where the treatment is off."""

NO_INTERACTION = 1.0
"""Normalised advantage where delay changed nothing; the line the figure is read against."""

DISCOUNT_ORDER: Tuple[str, ...] = ("decision", "primitive")
"""Panel order, decision first: it is the `CellArgs` default and every other experiment's arm."""

DELAY_X_LABEL = "reward delay (primitive steps)"

PANEL_KEYS: Tuple[str, ...] = ("condition", "discount", "reward_delay")
"""Read off the colour, the panel title and the x axis, so none of them names a series."""

FRACTION_YLIM = (-0.05, 1.05)
"""A fraction's axis, padded so a marker at 0 or 1 is not half outside the panel."""

REFERENCE_COLOR = "0.4"
REFERENCE_WIDTH = 0.8

SETTLE_TOL = 0.02
"""Fraction of a curve's own range within which it counts as no longer changing."""

LOG_X_RATIO = 0.25
"""Log the x axis once the fastest curve settles inside this fraction of it."""

CONDITION_COLOR: Dict[str, str] = {"action": "#4c72b0", "option": "#dd8452", "both": "#55a868"}
CONDITION_LABEL: Dict[str, str] = {
    "action": "action space", "option": "option space",
    "both": "both (option + action space)",
}

STRUCTURED_COLOR: Dict[str, str] = {"grammar": "black", "grammar_depth": "#8172b3"}
"""The deterministic catalogues, off the viridis ramp that colours random draws. Both of
them: a draw seed selects nothing in either, so neither is a point on that ramp."""

FAMILY_DASH: Dict[str, str] = {"grammar": "-", "grammar_depth": ":", "random": "--"}

COUNT_COLORMAP = "viridis"
COUNT_COLOR_RANGE = (0.15, 0.9)
"""Ends trimmed off the ramp: its pale tail is invisible on white."""

DRAW_BAND_ALPHA = 0.08
"""Lighter than `BAND_ALPHA`: six overlapping draws would otherwise stack opaque."""

BASELINE_DASH = "--"
"""How the action baseline is drawn where colour is spent on catalogue size."""

SERIES_KEYS: Tuple[str, ...] = (
    "env_id", "condition", "family", "option_seed", "budget", "max_steps",
    "reward_delay", "gamma", "discount", "tag",
)
"""Everything identifying a cell but its catalogue size, so a sweep over `n_options` never
pools two cells differing in anything else. `max_forward` and `executor` are in the episode
columns but not here: this track writes a sentinel for both, so neither can separate cells."""

IDENTITY: Tuple[str, ...] = SERIES_KEYS + ("n_options",)

OPTION_ONLY: Tuple[str, ...] = ("family", "n_options", "option_seed")
"""Fields `ppo.cell_identity` fills with a sentinel on an `action` cell, so a difference in
one of them between an action cell and an option cell means only "not applicable"."""

ARGS_FIELD: Dict[str, str] = {"family": "option_family", "max_steps": "max_episode_steps"}
"""Identity columns whose `CellArgs` field carries another name."""

FIELD_PREFIX: Dict[str, str] = {"family": "", "n_options": "n=", "option_seed": "os="}
"""How a varying field is written in a legend entry; anything else is written `key=`."""

DEFAULTS = CellArgs()
"""`Cell.name` elides a field left at its default, so a caption elides it too."""

UNTAGGED = "untagged"
"""Group name for cells run without a tag, which `Cell.name` simply omits."""

CENSORED: Dict[str, object] = {
    "point": float("nan"), "low": float("nan"), "high": float("nan"), "method": "censored",
}

METHOD_PHRASE: Dict[str, str] = {
    "iqm-bootstrap": f"IQM with {1 - CI_ALPHA:.0%} bootstrap CI",
    "median-range": "median with observed range",
    "median-bootstrap": f"median with {1 - CI_ALPHA:.0%} bootstrap CI",
    "per-seed": "one curve per seed, not aggregated across seeds",
}
"""How a method reads on a figure; the sidecar CSVs keep the token itself."""


class MissingData(Exception):
    """A figure cannot be drawn from the cells present."""

def warn(message: str) -> None:
    """Print a non-fatal problem with the store."""
    print(f"  warning: {message}")

def short(name: str) -> str:
    """A name without the environment boilerplate, for ticks and legends."""
    return name.replace("NetHack", "").replace("Delayed", "").replace("-v0", "")

def cell_name(directory: pathlib.Path) -> str:
    """A cell's store-relative name, `{group}/{cell}`."""
    return directory.relative_to(RUNS).as_posix()

def cell_directories(patterns: Sequence[str]) -> List[pathlib.Path]:
    """Every cell directory under a group whose store-relative name matches one of `patterns`."""
    if not RUNS.exists():
        raise SystemExit(f"no results yet: {RUNS} does not exist")
    found = [d for d in sorted(RUNS.glob("*/*"))
             if d.is_dir() and any(fnmatch.fnmatch(cell_name(d), p) for p in patterns)]
    if not found:
        raise SystemExit(f"no cells in {RUNS} matching {list(patterns)}")
    return found

def load_episodes(patterns: Sequence[str]) -> pd.DataFrame:
    """Every matching cell's episodes, concatenated, with `cell` and `group` columns.

    One file per seed, not one per cell: `main.py` runs a cell's seeds as concurrent
    processes into one directory, and they would race on a shared file's header.
    """
    frames = []
    for directory in cell_directories(patterns):
        paths = sorted(directory.glob("episodes_seed*.csv"))
        if not paths:
            warn(f"{cell_name(directory)} has no episodes_seed*.csv")
            continue
        frames += [pd.read_csv(path).assign(cell=cell_name(directory),
                                            group=directory.parent.name)
                   for path in paths]
    if not frames:
        raise MissingData("no matching cell has an episodes_seed*.csv")
    frame = pd.concat(frames, ignore_index=True)
    frame["tag"] = frame.tag.fillna(UNTAGGED).replace("", UNTAGGED)
    return frame

def require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> pd.DataFrame:
    """`frame` less any cell missing one of `columns`."""
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise MissingData(f"{what} needs {', '.join(absent)}, absent from every cell")
    usable = frame.dropna(subset=list(columns))
    for name in sorted(set(frame.cell) - set(usable.cell)):
        warn(f"{what} drops {name}: it predates {', '.join(columns)}")
    if usable.empty:
        raise MissingData(f"{what} found no cell carrying {', '.join(columns)}")
    return usable


@dataclass(frozen=True)
class Estimate:
    """A point estimate with an interval, and the method that produced it."""

    point: np.ndarray
    low: np.ndarray
    high: np.ndarray
    method: str
    n_seeds: int

    def row(self) -> Dict[str, object]:
        """The scalar case, as columns of a table."""
        return {**{k: float(getattr(self, k)) for k in ("point", "low", "high")},
                "method": self.method}

def iqm(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Interquartile mean along `axis`, as rliable defines it."""
    return trim_mean(values, IQM_TRIM, axis=axis)

def bootstrap_indices(strata: np.ndarray, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """`(resamples, n_seeds)` seed positions, resampled with replacement within each stratum."""
    indices = np.empty((resamples, strata.size), dtype=np.intp)
    for stratum in np.unique(strata):
        at = np.flatnonzero(strata == stratum)
        indices[:, at] = rng.choice(at, size=(resamples, at.size), replace=True)
    return indices

def estimate(per_seed: np.ndarray, strata: np.ndarray, resamples: int,
             rng: np.random.Generator) -> Estimate:
    """The IQM of `(n_seeds, ...)` with a stratified bootstrap interval, or median and range."""
    n_seeds = per_seed.shape[0]
    if n_seeds < MIN_SEEDS_FOR_IQM:
        point, low, high = np.median(per_seed, 0), per_seed.min(0), per_seed.max(0)
        method = "median-range"
    else:
        # one draw for the whole array: a curve's band is an envelope of resampled
        # curves, not a stack of independent per-bin intervals
        draws = iqm(per_seed[bootstrap_indices(strata, resamples, rng)], axis=1)
        low, high = np.quantile(draws, [CI_ALPHA / 2, 1 - CI_ALPHA / 2], axis=0)
        point, method = iqm(per_seed), "iqm-bootstrap"
    return Estimate(*(np.asarray(v) for v in (point, low, high)), method, n_seeds)

def first_of(frame: pd.DataFrame) -> Dict[str, object]:
    """The identifying columns of a group, from its first row."""
    first = frame.iloc[0]
    return {key: first[key] for key in IDENTITY if key in frame.columns}

def stratum_of(run: pd.DataFrame) -> str:
    """The bootstrap stratum a seed belongs to: its environment and option seed."""
    first = run.iloc[0]
    return f"{first.env_id}|{first.option_seed}"

def curve_window(frame: pd.DataFrame) -> Tuple[int, int]:
    """The primitive steps from the last seed's first episode to the first seed's last."""
    per_seed = frame.groupby(["cell", "seed"]).primitive_step.agg(["min", "max"])
    return int(per_seed["min"].max()), int(per_seed["max"].min())

def per_seed_curves(frame: pd.DataFrame, column: str, grid: np.ndarray,
                    window: int) -> Tuple[np.ndarray, np.ndarray]:
    """`(n_seeds, len(grid))` smoothed values on `grid`, and each row's stratum."""
    curves, strata = [], []
    for _, run in frame.groupby(["cell", "seed"]):
        run = run.sort_values("primitive_step")
        # min_periods=1, so the curve starts at the seed's first episode rather than its
        # `window`th, which for a fast cell is already past the whole climb. The reason to
        # demand a full window is threshold crossing, and `crossing_step` demands it there.
        smoothed = run[column].rolling(window, min_periods=1).mean().to_numpy()
        # `grid` starts at the last seed's first episode, so no seed is interpolated
        # outside its own range, where `np.interp` would clamp rather than say nothing
        curves.append(np.interp(grid, run.primitive_step.to_numpy(), smoothed))
        strata.append(stratum_of(run))
    return np.stack(curves), np.asarray(strata)

def crossing_step(run: pd.DataFrame, column: str, rate: float, window: int) -> float:
    """One seed's primitive step at its first trailing mean of `column` past `rate`.

    Read off the seed's own episodes, not off an interpolated grid whose bin width scales
    with the run length and quantises every cell to the same few steps.
    """
    run = run.sort_values("primitive_step")
    # min_periods=window: a seed whose first episode happened to succeed would otherwise
    # cross at its first step, and one too short to fill a window never crossed
    smoothed = run[column].rolling(window, min_periods=window).mean().to_numpy()
    at = np.flatnonzero(smoothed >= rate)
    return float(run.primitive_step.to_numpy()[at[0]]) if at.size else float("nan")


@dataclass(frozen=True)
class Inputs:
    """What every figure is handed: the loaded episodes and the options."""

    episodes: pd.DataFrame
    args: argparse.Namespace

    def grid(self, start: int, end: int) -> np.ndarray:
        """Geometrically over `[start, end]`, the span every seed of a cell covers.

        Geometric, not uniform: a uniform grid over a million steps puts its first point
        past the entire climb of a cell that solves in a few thousand.
        """
        return np.geomspace(float(start), float(end), self.args.bins)

def aggregate_curve(data: Inputs, frame: pd.DataFrame, column: str,
                    rng: np.random.Generator) -> pd.DataFrame:
    """One row per cell and grid point, over the span every seed of that cell covers.

    A grid per cell, not one at the shortest cell's limit: the budget buys a cell as many
    decisions as its options are long, so a slow baseline outlives every option cell and
    truncating it to their limit hides it still climbing.
    """
    tables = []
    for name, cell in frame.groupby("cell"):
        start, end = curve_window(cell)
        # an estimator sees every seed at every grid point or none: `trim_mean` and
        # `np.median` both return NaN from a single missing seed, and a NaN curve draws
        # as a legend entry with no line
        if start >= end:
            warn(f"{name} has no span every seed covers: the last seed starts at "
                 f"{start:,}, the first ends at {end:,}")
            continue
        grid = data.grid(start, end)
        curves, strata = per_seed_curves(cell, column, grid, data.args.window)
        result = estimate(curves, strata, data.args.resamples, rng)
        tables.append(pd.DataFrame({
            "cell": name, "group": cell.group.iloc[0], **first_of(cell),
            "primitive_step": grid, "point": result.point, "low": result.low,
            "high": result.high, "method": result.method, "n_seeds": result.n_seeds,
            "limit": end,
        }))
    if not tables:
        raise MissingData("no cell has a span every one of its seeds covers")
    return pd.concat(tables, ignore_index=True)

def tail_of(data: Inputs, run: pd.DataFrame) -> pd.DataFrame:
    """One seed's episodes over the last `--tail` fraction of the steps it reached."""
    reached = run.primitive_step.max()
    return run[run.primitive_step > (1 - data.args.tail) * reached]

def last_episodes(run: pd.DataFrame, count: int) -> pd.DataFrame:
    """One seed's last `count` episodes, or all of them where it finished fewer."""
    return run.sort_values("primitive_step").tail(count)

def crossing_column(frame: pd.DataFrame) -> str:
    """`CROSSING_COLUMN` where it varies, else `episodic_return`.

    A host with no goal to pay writes one value there for every episode, and a bar on a
    constant is crossed at the first full window or never.
    """
    if CROSSING_COLUMN in frame.columns and frame[CROSSING_COLUMN].nunique(dropna=False) > 1:
        return CROSSING_COLUMN
    return "episodic_return"

def aggregate_final(data: Inputs, frame: pd.DataFrame, rng: np.random.Generator) -> Estimate:
    """The estimate over each seed's last `--tail` of primitive steps."""
    values, strata = [], []
    for _, run in frame.groupby(["cell", "seed"]):
        values.append(tail_of(data, run).episodic_return.mean())
        strata.append(stratum_of(run))
    return estimate(np.asarray(values), np.asarray(strata), data.args.resamples, rng)

def solved_rate_for(data: Inputs, cell: pd.DataFrame,
                    column: str = CROSSING_COLUMN) -> float:
    """This cell's bar on `column`: `--solved-threshold`, or the column's own default.

    `--solved-threshold` is taken in the units of whichever column is being crossed, so
    the same flag names a payout rate on a sparse host and a score on a dense one.
    """
    declared = data.args.solved_threshold
    if isinstance(declared, dict):
        condition = str(cell.iloc[0].condition)
        if condition in declared:
            return float(declared[condition])
    elif declared is not None:
        return float(declared)
    return DEFAULT_SOLVED_RATE if column == CROSSING_COLUMN else DENSE_RETURN_THRESHOLD

def cell_crossings(data: Inputs, cell: pd.DataFrame,
                   column: str = CROSSING_COLUMN) -> Tuple[np.ndarray, np.ndarray]:
    """One cell's per-seed crossing steps, NaN where a seed never crossed, and each stratum."""
    rate = solved_rate_for(data, cell, column)
    runs = [run for _, run in cell.groupby("seed")]
    return (
        np.asarray([crossing_step(run, column, rate, data.args.window)
                    for run in runs]),
        np.asarray([stratum_of(run) for run in runs]),
    )

def resampled_points(values: np.ndarray, strata: np.ndarray, resamples: int,
                     rng: np.random.Generator) -> Tuple[float, np.ndarray]:
    """The point `estimate` reports for `values`, and `resamples` bootstrap replicates of it.

    The statistic switches at `MIN_SEEDS_FOR_IQM` exactly as `estimate` does. Unlike
    `estimate`, the sub-cutoff case is still bootstrapped: a ratio of two observed ranges
    is not an interval on the ratio.
    """
    statistic = iqm if values.size >= MIN_SEEDS_FOR_IQM else np.median
    draws = values[bootstrap_indices(strata, resamples, rng)]
    return float(statistic(values)), np.asarray(statistic(draws, axis=1))

def steps_to_threshold(data: Inputs, frame: pd.DataFrame, rng: np.random.Generator,
                       column: str = CROSSING_COLUMN) -> pd.DataFrame:
    """Primitive steps to reach each cell's bar on `column`, excluding seeds that never do."""
    rows = []
    for name, cell in frame.groupby("cell"):
        crossings, strata = cell_crossings(data, cell, column)
        crossed = np.isfinite(crossings)
        n_seeds, n_crossed = crossings.size, int(crossed.sum())
        # pinning a non-crossing seed to the limit would bias the estimate down by
        # an amount the budget sets, so it is dropped and the fraction reported
        measured = estimate(crossings[crossed], strata[crossed],
                            data.args.resamples, rng).row() \
            if n_crossed >= CENSOR_FRACTION * n_seeds else CENSORED
        rows.append({
            "cell": name, "group": cell.group.iloc[0], **first_of(cell),
            "crossing_column": column, "solved_rate": solved_rate_for(data, cell, column),
            **measured, "n_crossed": n_crossed,
            "n_seeds": n_seeds, "limit": curve_window(cell)[1],
        })
    return pd.DataFrame(rows)

def varying_fields(table: pd.DataFrame) -> List[str]:
    """The identity columns whose value differs across `table`, in `IDENTITY` order."""
    option_rows = table[table.condition != "action"]
    return [key for key in IDENTITY if key in table.columns
            and (option_rows if key in OPTION_ONLY else table)[key].nunique(dropna=False) > 1]

def series_label(row: pd.Series, varying: Sequence[str]) -> str:
    """A legend entry: the condition when it varies, then only what else separates this cell."""
    named = [key for key in varying if key != "condition"
             and (row.condition != "action" or key not in OPTION_ONLY)]
    parts = ([CONDITION_LABEL[row.condition]] if "condition" in varying or not named else [])
    parts += [f"{FIELD_PREFIX.get(key, key + '=')}{short(str(row[key]))}"
              for key in named]
    return ", ".join(parts)

def shared_estimate(table: pd.DataFrame) -> str:
    """`n seeds, method` where every estimated cell of `table` agrees on both, else empty.

    Censored cells are left out of the agreement: censoring is the absence of an estimate,
    not a second estimator, and one censored cell would otherwise strip the caption of the
    method every drawn cell shares.
    """
    if not {"n_seeds", "method"} <= set(table.columns):
        return ""
    estimated = table[table.method != CENSORED["method"]]
    if estimated.empty or estimated.n_seeds.nunique() != 1 or estimated.method.nunique() != 1:
        return ""
    method = str(estimated.method.iloc[0])
    return f"{int(estimated.n_seeds.iloc[0])} seeds, {METHOD_PHRASE.get(method, method)}"

def solved_rate_phrase(table: pd.DataFrame, column: str = CROSSING_COLUMN) -> str:
    """The bar and its units, or one bar per condition where they were given their own."""
    noun = "solved rate" if column == CROSSING_COLUMN else "episodic return"
    bars = {str(row.condition): float(row.solved_rate) for _, row in table.iterrows()}
    if len(set(bars.values())) == 1:
        return f"{noun} {next(iter(bars.values())):g}"
    return f"{noun} " + ", ".join(
        f"{value:g} for {CONDITION_LABEL[condition]}"
        for condition, value in sorted(bars.items(), key=lambda pair: -pair[1]))

def limit_span(table: pd.DataFrame) -> str:
    """The range of per-cell truncation limits, as one number where they agree."""
    low, high = int(table.limit.min()), int(table.limit.max())
    return f"{low:,}" if low == high else f"{low:,} to {high:,}"

def caption_for(table: pd.DataFrame, varying: Sequence[str],
                extra: Sequence[str] = ()) -> str:
    """Everything every series of `table` shares, for the legend title.

    A field at its `CellArgs` default is left out, as `Cell.name` leaves it out, so the
    caption and a legend entry together name the cell. `n_options` is the exception, named
    on every option cell there and so named here too. The group is not named: it is the
    directory the figure is written into, and `tag` only repeats it.
    """
    option_rows = table[table.condition != "action"]
    fixed = []
    # the catalogue size beside the family it sizes, where IDENTITY puts it last
    catalogue = ("family", "n_options")
    for key in catalogue + tuple(k for k in IDENTITY if k not in catalogue):
        rows = option_rows if key in OPTION_ONLY else table
        if key in varying or key in ("env_id", "condition", "tag") or key not in table.columns:
            continue
        if rows.empty or rows[key].nunique(dropna=False) != 1:
            continue
        value = rows[key].iloc[0]
        if key != "n_options" and value == getattr(DEFAULTS, ARGS_FIELD.get(key, key)):
            continue
        written = short(str(value))
        fixed.append(f"{written} catalogue" if key == "family"
                     else f"{FIELD_PREFIX.get(key, key + '=')}{written}")

    return "\n".join(filter(None, [
        ", ".join(fixed), ", ".join(filter(None, [shared_estimate(table), *extra])),
    ]))

def draw_bands(axis: Axes, table: pd.DataFrame, varying: Sequence[str],
               colors: Optional[Dict[int, Tuple[float, float, float, float]]] = None) -> None:
    """One line and interval band per cell, coloured by condition, or by `colors` if given.

    Under `colors`, an option cell takes the colour of its catalogue size and is named by
    it alone; the action cell keeps its condition colour and is dashed, as a reference.
    With no `colors`, a varying `option_seed` ramps random draws over viridis; the
    structured families keep their `STRUCTURED_COLOR`.
    """
    # the seed count and method go in the caption where every cell agrees, which is the
    # usual case, and stay on the entry where a short cell fell back to median-and-range
    suffix = "" if shared_estimate(table) else " ({} seeds, {})"
    seed_colors: Optional[Dict[int, Tuple[float, float, float, float]]] = None
    if colors is None and "option_seed" in varying:
        drawn = table[(table.condition != "action")
                      & (~table.family.isin(list(STRUCTURED_COLOR)))]
        seeds = sorted({int(seed) for seed in drawn.option_seed.unique()})
        if seeds:
            low, high = COUNT_COLOR_RANGE
            colormap = matplotlib.colormaps[COUNT_COLORMAP]
            seed_colors = {
                seed: colormap(low + (high - low) * position / max(len(seeds) - 1, 1))
                for position, seed in enumerate(seeds)
            }
    # sorted by what the legend prints, so `n=8` precedes `n=16`; grouping by the cell name
    # alone orders the legend lexically, and sort=False then keeps this order
    order = [*dict.fromkeys(["_condition_at", *varying]), "cell"]
    ordered = table.assign(_condition_at=table.condition.map(list(CONDITION_LABEL).index))
    for _, group in ordered.sort_values(order).groupby("cell", sort=False):
        first = group.iloc[0]
        counted = colors is not None and first.condition != "action"
        color = colors[int(first.n_options)] if counted else CONDITION_COLOR[first.condition]
        if seed_colors is not None and first.condition != "action":
            color = (STRUCTURED_COLOR[first.family] if first.family in STRUCTURED_COLOR
                     else seed_colors[int(first.option_seed)])
        axis.plot(group.primitive_step, group.point, color=color, linewidth=LINE_WIDTH,
                  linestyle=FAMILY_DASH.get(first.family, "-") if colors is None
                  else ("-" if counted else BASELINE_DASH),
                  label=f"n={first.n_options:g}" if counted else
                  series_label(first, varying)
                  + suffix.format(int(first.n_seeds),
                                  METHOD_PHRASE.get(first.method, first.method)))
        axis.fill_between(group.primitive_step, group.low, group.high, color=color,
                          alpha=DRAW_BAND_ALPHA if seed_colors is not None else BAND_ALPHA,
                          linewidth=0, edgecolor="none")

def shared_across_modes(table: pd.DataFrame) -> pd.DataFrame:
    """The action arm repeated into every discount mode present, where it was run in one.

    `expand` does not cross the action condition with `discount`, because a primitive takes
    one step per decision and `gamma ** 1` is `gamma`, so the two modes would be the same
    run twice. The one cell is the correct reference for both panels, which is what this
    puts there; a sweep that did run the action arm per mode is left alone.
    """
    modes = set(table.discount)
    action = table[table.condition == "action"]
    if len(modes) < 2 or action.empty or action.discount.nunique() > 1:
        return table
    missing = [action.assign(discount=mode) for mode in modes - set(action.discount)]
    return pd.concat([table, *missing], ignore_index=True)

def delay_varying(table: pd.DataFrame) -> List[str]:
    """What names a series on a delay figure: the condition, then anything else that varies.

    `condition` is forced in because it is the colour on every one of these figures, and
    without it `series_label` drops it as soon as a second field varies.
    """
    return ["condition"] + [key for key in varying_fields(table) if key not in PANEL_KEYS]

def discount_axes(table: pd.DataFrame, nrows: int = 1) -> Tuple[Figure, np.ndarray, List[str]]:
    """A grid one column per discount mode present, x ticked at the delays present."""
    delays = sorted({int(delay) for delay in table.reward_delay.unique()})
    if len(delays) < 2:
        raise MissingData(f"every loaded cell is reward_delay={delays[0]}, "
                          "so there is no delay axis to sweep")
    modes = [mode for mode in DISCOUNT_ORDER if mode in set(table.discount)]
    # never narrower than a single-axis figure, which is what the caption is sized for
    figure, axes = plt.subplots(
        nrows, len(modes), sharex=True, sharey="row", squeeze=False,
        figsize=(max(FIGURE_SIZE[0], PANEL_WIDTH * len(modes)),
                 max(FIGURE_SIZE[1], PANEL_HEIGHT * nrows)),
    )
    for column, mode in enumerate(modes):
        axes[0][column].set_title(f"discount = {mode}")
        axes[-1][column].set_xlabel(DELAY_X_LABEL)
    for axis in axes.flat:
        axis.set_xticks(delays)
        # `finish` formats x only, and a steps-to-threshold axis runs into the millions
        axis.ticklabel_format(axis="y", style="sci", scilimits=SCI_LIMITS, useMathText=True)
    return figure, axes, modes

def delay_series(axis: Axes, panel: pd.DataFrame, column: str,
                 bounds: Optional[Tuple[str, str]] = None,
                 varying: Sequence[str] = ()) -> None:
    """One series per condition against reward delay, in legend order.

    A row whose `column` is NaN is dropped rather than drawn, so a censored cell leaves a
    gap in the line instead of a point the estimator never produced.
    """
    ordered = panel.assign(_condition_at=panel.condition.map(list(CONDITION_LABEL).index))
    for _, group in ordered.sort_values("_condition_at").groupby("condition", sort=False):
        drawn = group.dropna(subset=[column]).sort_values("reward_delay")
        if drawn.empty:
            continue
        first = drawn.iloc[0]
        error = None
        if bounds is not None:
            low, high = bounds
            # a percentile bootstrap interval need not contain the point estimate
            error = np.clip([drawn[column] - drawn[low], drawn[high] - drawn[column]], 0, None)
        axis.errorbar(drawn.reward_delay, drawn[column], yerr=error,
                      color=CONDITION_COLOR[first.condition], marker="o",
                      capsize=ERROR_CAP_SIZE, linewidth=LINE_WIDTH,
                      label=series_label(first, varying))

def settled_steps(table: pd.DataFrame) -> List[float]:
    """The step past which each cell's curve stops moving, over the cells that moved."""
    curves = {name: group.sort_values("primitive_step").dropna(subset=["point"])
              for name, group in table.groupby("cell")}
    spans = {name: float(group.point.max() - group.point.min())
             for name, group in curves.items() if not group.empty}
    # relative to the cell that moved most: a curve flat at its floor never learned, so it
    # is not evidence that the run settled early
    reference = max(spans.values(), default=0.0)
    settled = []
    for name, span in spans.items():
        if span <= SETTLE_TOL * reference:
            continue
        point = curves[name].point.to_numpy()
        moving = np.flatnonzero(np.abs(point - point[-1]) > SETTLE_TOL * span)
        at = min(int(moving[-1]) + 1, point.size - 1) if moving.size else 0
        settled.append(float(curves[name].primitive_step.to_numpy()[at]))
    return settled

def apply_x_scale(axes: Sequence[Axes], table: pd.DataFrame) -> None:
    """Log x where the fastest curve settles early, linear from zero otherwise."""
    right = float(table.limit.max())
    settled = settled_steps(table)
    logarithmic = bool(settled) and min(settled) < LOG_X_RATIO * right
    # a log axis cannot render zero, so it starts at the first x carrying an estimate, which
    # is later than the first grid point: a cell has none until all of its seeds have
    # finished an episode. A linear axis starts at zero, before any episode has finished.
    drawn = table.primitive_step[table.point.notna() & (table.primitive_step > 0)]
    for axis in axes:
        if logarithmic:
            axis.set_xscale("log")
        axis.set_xlim(float(drawn.min()) if logarithmic else 0.0, right)

def finish(figure: Figure, axes: Sequence[Axes], title: str, caption: str = "") -> None:
    """Grid, scientific x labels, a title, a caption under it, and a legend along the foot."""
    for axis in axes:
        axis.grid(alpha=GRID_ALPHA)
        if axis.get_xscale() == "linear":
            axis.ticklabel_format(axis="x", style="sci", scilimits=SCI_LIMITS,
                                  useMathText=True)
    # caption a line below the title: they share y=1 and collide once the title is long
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.82 if caption else 0.94])
    figure.suptitle(title, x=0.0, y=0.99, ha="left")
    if caption:
        figure.text(0.0, 0.91, caption, ha="left", va="top", fontsize=SMALL_FONT)
    # one legend for the whole figure, in figure coordinates and only once
    # tight_layout has settled the axes: an axes-fraction offset lands on the x
    # label, and a legend per panel is wider than the panel it belongs to
    entries: Dict[str, object] = {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            # by label: panels of one figure draw the same series, and identical labels
            # carry identical styling, so a second entry would be a duplicate
            entries.setdefault(label, handle)
    if entries:
        figure.legend(list(entries.values()), list(entries), loc="upper left",
                      frameon=False, fontsize=LEGEND_FONT, ncols=len(entries),
                      bbox_to_anchor=(axes[0].get_position().x0, LEGEND_TOP))

def environments(frame: pd.DataFrame) -> str:
    """The environments a frame covers, for a figure title."""
    return " / ".join(sorted(frame.env_id.unique()))

def collinear(frame: pd.DataFrame, key: str, column: str) -> bool:
    """True where `column` takes a single value within every group of `key`."""
    return bool((frame.groupby(key)[column].nunique() <= 1).all())

def labels_for(conditions: Iterable[str]) -> str:
    """The given conditions named in legend order, as a phrase, or an empty string."""
    present = set(conditions)
    return ", ".join(label for condition, label in CONDITION_LABEL.items()
                     if condition in present)

def omitted_note(data: Inputs, frame: pd.DataFrame, table: pd.DataFrame) -> str:
    """Conditions this figure's own frame carried that it drew no line for.

    On the figure, not only in `aggregate_curve`'s warning: a condition dropped for want of
    a window reads exactly like a condition that was never run. Against `frame`, not
    `data.episodes`, so a cell `require_columns` dropped is not blamed on the window.

    The rate is `solved`, not `terminated`: NLE sets `terminated` on death and on its own
    step-limit abort too, so it sits near 1 in every cell and says nothing about the
    condition that was dropped.
    """
    absent = set(frame.condition) - set(table.condition)
    if not absent or "solved" not in data.episodes.columns:
        return ""
    rows = data.episodes[data.episodes.condition.isin(absent)]
    return (f"{labels_for(absent)} omitted: solved {rows.solved.mean():.2%} "
            "of episodes, over no step range every seed covers")

def never_crossed(table: pd.DataFrame) -> str:
    """Conditions censored in every cell of a `steps_to_threshold` table."""
    floored = [condition for condition, rows in table.groupby("condition")
               if rows.point.isna().all()]
    if not floored:
        return ""
    return (f"{labels_for(floored)} censored at every delay: never reached the "
            "solved rate, so this is a capability floor and not a slower crossing")

def curve_figure(data: Inputs, frame: pd.DataFrame, column: str,
                 ylabel: str) -> Tuple[Figure, pd.DataFrame]:
    """A column's estimate against primitive steps, one band per cell."""
    table = aggregate_curve(data, frame, column, np.random.default_rng(BOOTSTRAP_SEED))
    varying = varying_fields(table)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    draw_bands(axis, table, varying)
    apply_x_scale([axis], table)
    axis.set_xlabel(X_LABEL)
    axis.set_ylabel(ylabel)
    caption = caption_for(table, varying, (f"{data.args.window}-episode moving average",))
    finish(figure, [axis], environments(frame),
           "\n".join(filter(None, [caption, omitted_note(data, frame, table)])))
    return figure, table

def return_curve(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Episodic return against primitive steps."""
    frame = require_columns(data.episodes, ("primitive_step", "episodic_return"), "return")
    return curve_figure(data, frame, "episodic_return", "episodic return")

def return_ecdf(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Episodic return over each seed's last episodes, one empirical CDF per seed.

    Per seed rather than pooled: pooling reads a shift between seeds as a wider
    distribution within one, which is the question this figure is here to answer.
    """
    frame = require_columns(data.episodes, ("primitive_step", "episodic_return"), "ecdf")
    tables = []
    for name, cell in frame.groupby("cell"):
        for seed, run in cell.groupby("seed"):
            returns = np.sort(last_episodes(run, TAIL_EPISODES).episodic_return.to_numpy())
            tables.append(pd.DataFrame({
                "cell": name, "group": cell.group.iloc[0], **first_of(cell), "seed": seed,
                "episodic_return": returns,
                "ecdf": np.arange(1, returns.size + 1) / returns.size,
                "n_episodes": returns.size, "n_seeds": cell.seed.nunique(),
                "method": "per-seed",
            }))
    table = pd.concat(tables, ignore_index=True)
    varying = varying_fields(table)

    # never narrower than a single-axis figure, which is what the caption is sized for
    figure, axes = plt.subplots(1, 2, sharey=True, squeeze=False,
                                figsize=(max(FIGURE_SIZE[0], PANEL_WIDTH * 2),
                                         FIGURE_SIZE[1]))
    ordered = table.assign(_condition_at=table.condition.map(list(CONDITION_LABEL).index))
    for _, run in ordered.sort_values(["_condition_at", *varying, "cell", "seed"]).groupby(
            ["cell", "seed"], sort=False):
        first = run.iloc[0]
        for axis in axes[0]:
            # post: the curve steps up at the episode's own score rather than before it
            axis.step(run.episodic_return, run.ecdf, where="post", linewidth=LINE_WIDTH,
                      alpha=SEED_LINE_ALPHA, color=CONDITION_COLOR[first.condition],
                      label=series_label(first, varying))
    for axis in axes[0]:
        axis.set_xlabel("episodic return (unclipped NetHack score)")
    axes[0][0].set_title("every episode")
    axes[0][0].set_ylabel("fraction of episodes at or below")
    axes[0][0].set_ylim(*FRACTION_YLIM)
    axes[0][1].set_title(f"detail below {ECDF_ZOOM_LIMIT:g}")
    axes[0][1].axvline(0.0, color=REFERENCE_COLOR, linestyle=BASELINE_DASH,
                       linewidth=REFERENCE_WIDTH, label="zero score")
    # padded past the lowest score rather than cut at zero: a NetHack score goes negative,
    # and where it does the left tail is the thing this panel exists to show
    lowest = float(table.episodic_return.min())
    axes[0][1].set_xlim(lowest - ZOOM_MARGIN * (ECDF_ZOOM_LIMIT - lowest), ECDF_ZOOM_LIMIT)
    short = table[table.n_episodes < TAIL_EPISODES].groupby(["cell", "seed"]).ngroups
    finish(figure, list(axes[0]), environments(frame), caption_for(table, varying, (
        f"last {TAIL_EPISODES:,} episodes of each seed",
        f"{short} seeds finished fewer and contribute every episode they have"
        if short else "",
    )))
    return figure, table

def success_curve(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Fraction of episodes paid the goal reward, against primitive steps."""
    frame = require_columns(
        data.episodes, ("primitive_step", CROSSING_COLUMN, "episodic_return"), "success"
    )
    if crossing_column(frame) != CROSSING_COLUMN:
        raise MissingData(
            f"{CROSSING_COLUMN} is {frame[CROSSING_COLUMN].iloc[0]:g} in every episode of "
            "every loaded cell, so a payout rate is a horizontal line"
        )
    if collinear(frame, CROSSING_COLUMN, "episodic_return"):
        raise MissingData(
            "episodic_return takes one value per payout outcome in every loaded "
            "cell, so a success curve is the return curve rescaled"
        )
    return curve_figure(data, frame, CROSSING_COLUMN, "payout rate")

def length_curve(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Episode length among paid episodes, against primitive steps."""
    frame = require_columns(
        data.episodes, ("primitive_step", "episodic_length", CROSSING_COLUMN), "length"
    )
    # `episodic_length` is counted by the innermost wrapper, primitive steps inside the
    # episode, so it compares across conditions unconverted; among paid episodes only,
    # since an unpaid one reports the horizon rather than a cost of solving
    solved = frame[frame[CROSSING_COLUMN] == 1]
    if solved.empty:
        raise MissingData("no episode of any matching cell was paid the goal reward")
    return curve_figure(data, solved, "episodic_length",
                        "primitive steps per paid episode")

def family_overlay(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Return curves of the catalogue families, one panel per catalogue size."""
    frame = require_columns(data.episodes, ("primitive_step", "episodic_return", "family"),
                            "overlay")
    frame = frame[frame.condition != "action"]
    counts = [n for n, panel in frame.groupby("n_options") if panel.family.nunique() > 1]
    for count in sorted(set(frame.n_options.unique()) - set(counts)):
        warn(f"family_overlay drops n={count}: only one family was run at it")
    if not counts:
        raise MissingData("no catalogue size has more than one family")

    frame = frame[frame.n_options.isin(counts)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    tables = [aggregate_curve(data, frame[frame.n_options == count], "episodic_return", rng)
              for count in counts]
    combined = pd.concat(tables, ignore_index=True)
    # n_options titles the panel, so it never names a series inside one
    varying = [key for key in varying_fields(combined) if key != "n_options"]

    # never narrower than a single-axis figure, which is what the caption is sized for
    figure, axes = plt.subplots(
        1, len(counts), sharey=True, squeeze=False,
        figsize=(max(FIGURE_SIZE[0], PANEL_WIDTH * len(counts)), FIGURE_SIZE[1]),
    )
    for axis, count, table in zip(axes[0], counts, tables):
        draw_bands(axis, table, varying)
        axis.set_title(f"n = {count}")
        axis.set_xlabel(X_LABEL)
    apply_x_scale(list(axes[0]), combined)
    axes[0][0].set_ylabel("episodic return")
    finish(figure, list(axes[0]), environments(frame),
           caption_for(combined, varying, (f"{data.args.window}-episode moving average",)))
    return figure, combined

def count_overlay(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Return curves, one panel per condition, coloured by catalogue size."""
    frame = require_columns(data.episodes,
                            ("primitive_step", "episodic_return", "n_options"), "overlay")
    # one aggregation for the whole frame, then split: a call per panel would bootstrap the
    # action cell again off a mutated rng and give one cell a different band in each panel
    combined = aggregate_curve(data, frame, "episodic_return",
                               np.random.default_rng(BOOTSTRAP_SEED))
    conditions = [condition for condition in CONDITION_LABEL
                  if condition != "action" and condition in set(combined.condition)]
    if not conditions:
        raise MissingData("no option cell has a span every one of its seeds covers")
    counts = sorted({int(n) for n in combined.loc[combined.condition != "action", "n_options"]})
    if len(counts) < 2:
        raise MissingData("count_overlay needs at least two catalogue sizes; every "
                          f"drawable option cell is n={counts[0]}")

    low, high = COUNT_COLOR_RANGE
    colormap = matplotlib.colormaps[COUNT_COLORMAP]
    colors = {count: colormap(low + (high - low) * position / max(len(counts) - 1, 1))
              for position, count in enumerate(counts)}
    reference = combined[combined.condition == "action"]
    varying = varying_fields(combined)

    # never narrower than a single-axis figure, which is what the caption is sized for
    figure, axes = plt.subplots(
        1, len(conditions), sharey=True, squeeze=False,
        figsize=(max(FIGURE_SIZE[0], PANEL_WIDTH * len(conditions)), FIGURE_SIZE[1]),
    )
    for axis, condition in zip(axes[0], conditions):
        panel = pd.concat([reference, combined[combined.condition == condition]],
                          ignore_index=True)
        draw_bands(axis, panel, varying, colors)
        axis.set_title(CONDITION_LABEL[condition])
        axis.set_xlabel(X_LABEL)
    apply_x_scale(list(axes[0]), combined)
    axes[0][0].set_ylabel("episodic return")
    finish(figure, list(axes[0]), environments(frame),
           caption_for(combined, varying, (f"{data.args.window}-episode moving average",)))
    return figure, combined

def option_count_sweep(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Final return against catalogue size, one series per otherwise-equal cell."""
    frame = require_columns(data.episodes, ("episodic_return", "n_options"), "sweep")
    frame = frame[frame.condition != "action"]
    if frame.empty:
        raise MissingData("option_count_sweep found no option cells")
    counts = frame.n_options.unique()
    if counts.size < 2:
        raise MissingData("option_count_sweep needs at least two catalogue sizes; "
                          f"every loaded cell is n={counts[0]:g}")

    keys = [key for key in SERIES_KEYS if key in frame.columns]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # dropna=False: a key absent from cells written before it was logged would otherwise
    # silently drop those cells
    table = pd.DataFrame([
        {**dict(zip(keys + ["n_options"], values)),
         "group": group.group.iloc[0], "cells": " ".join(sorted(group.cell.unique())),
         "n_seeds": group.groupby(["cell", "seed"]).ngroups,
         **aggregate_final(data, group, rng).row()}
        for values, group in frame.groupby(keys + ["n_options"], dropna=False)
    ]).sort_values(keys + ["n_options"])
    # n_options is the x axis, so it never names a series
    varying = [key for key in varying_fields(table) if key != "n_options"]
    suffix = "" if shared_estimate(table) else " ({} seeds, {})"

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for _, group in table.groupby(keys, dropna=False):
        first = group.iloc[0]
        axis.errorbar(
            group.n_options, group.point,
            # a percentile bootstrap interval need not contain the point estimate
            yerr=np.clip([group.point - group.low, group.high - group.point], 0, None),
            color=CONDITION_COLOR[first.condition], marker="o", capsize=ERROR_CAP_SIZE,
            linestyle=FAMILY_DASH.get(first.family, "-"),
            label=series_label(first, varying)
            + suffix.format(int(first.n_seeds),
                            METHOD_PHRASE.get(first.method, first.method)),
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted(table.n_options.unique()))
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.set_xlabel("options in the catalogue")
    axis.set_ylabel(f"episodic return, final {data.args.tail:.0%} of each seed")
    finish(figure, [axis], environments(frame), caption_for(table, varying))
    return figure, table

def option_usage(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Blocked: nothing in the store records which option index was selected."""
    raise MissingData(
        "no per-option selection counts exist: ppo.py would need a bincount over the "
        "sampled actions, and only finished episodes reach episodes_seed*.csv"
    )

def threshold_table(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Primitive steps to reach the crossing bar, as a rendered table."""
    column = crossing_column(data.episodes)
    frame = require_columns(data.episodes, ("primitive_step", column), "threshold")
    table = steps_to_threshold(data, frame, np.random.default_rng(BOOTSTRAP_SEED), column)
    # a table of every cell censored is an empty table, and its `n_crossed/n_seeds`
    # column would be the only thing on it
    if table.empty or table.point.isna().all():
        raise MissingData(f"every cell is censored: fewer than {CENSOR_FRACTION:.0%} of "
                          f"the seeds of any cell crossed on {column}")
    varying = varying_fields(table)
    # what separates this cell from the others, as in a legend entry; everything the rows
    # share is in the caption, so the cell name would repeat it on every row
    body = [
        [series_label(row, varying),
         "-" if np.isnan(row.point) else f"{row.point:,.0f}",
         "-" if np.isnan(row.low) else f"[{row.low:,.0f}, {row.high:,.0f}]",
         f"{row.n_crossed}/{row.n_seeds}", row.method]
        for _, row in table.iterrows()
    ]
    height = 1.4 + ROW_HEIGHT * (len(body) + 1)
    figure, axis = plt.subplots(figsize=(FIGURE_SIZE[0], height))
    axis.axis("off")
    # bbox, not loc: `loc` centres a table sized to its text, leaving the figure
    # mostly blank, and the cell column is then clipped rather than widened
    rendered = axis.table(cellText=body, cellLoc="left", bbox=[0.0, 0.0, 1.0, 1.0],
                          colLabels=["cell", "steps", "interval", "crossed", "method"])
    rendered.auto_set_font_size(False)
    rendered.auto_set_column_width(range(len(body[0])))
    rendered.set_fontsize(SMALL_FONT)
    caption = caption_for(table, varying)
    # tight_layout ignores suptitle; a caption at y=1 sits in the title's band
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.84 if caption else 0.92])
    figure.suptitle(f"primitive steps to {solved_rate_phrase(table, column)} "
                    f"({data.args.window}-episode moving average)",
                    x=0.0, y=0.99, ha="left")
    if caption:
        figure.text(1.0, 0.90, caption, ha="right", va="top", fontsize=SMALL_FONT)
    return figure, table


def delay_crossing_fraction(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Fraction of seeds reaching the solved rate, against reward delay."""
    frame = require_columns(
        data.episodes,
        ("primitive_step", CROSSING_COLUMN, "reward_delay", "discount"), "crossing fraction"
    )
    table = shared_across_modes(
        steps_to_threshold(data, frame, np.random.default_rng(BOOTSTRAP_SEED)))
    table["crossed_fraction"] = table.n_crossed / table.n_seeds
    varying = delay_varying(table)
    figure, axes, modes = discount_axes(table)
    for axis, mode in zip(axes[0], modes):
        delay_series(axis, table[table.discount == mode], "crossed_fraction", varying=varying)
        axis.axhline(CENSOR_FRACTION, color=REFERENCE_COLOR, linestyle=BASELINE_DASH,
                     linewidth=REFERENCE_WIDTH,
                     label=f"censoring rule ({CENSOR_FRACTION:.0%} of seeds)")
    axes[0][0].set_ylabel("seeds reaching the solved rate")
    axes[0][0].set_ylim(*FRACTION_YLIM)
    finish(figure, list(axes.flat), environments(frame), "\n".join(filter(None, [
        caption_for(table, varying, (f"{solved_rate_phrase(table)}, "
                                     f"{data.args.window}-episode moving average",)),
        never_crossed(table),
    ])))
    return figure, table

def delay_slack(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Implied solve length and the fraction of solves pinned at truncation, against delay."""
    frame = require_columns(
        data.episodes,
        ("primitive_step", "episodic_length", CROSSING_COLUMN, "max_steps", "reward_delay",
         "discount"), "slack"
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for name, cell in frame.groupby("cell"):
        lengths, pinned, strata = [], [], []
        for _, run in cell.groupby("seed"):
            tail = tail_of(data, run)
            # on `paid`, not `solved`: a hold flushed by the horizon is exactly the pinned
            # episode this figure measures, and it reports ABORTED rather than success
            solved = tail[tail[CROSSING_COLUMN] == 1]
            if solved.empty:
                continue
            lengths.append(float((solved.episodic_length - solved.reward_delay).median()))
            pinned.append(float((solved.episodic_length >= solved.max_steps).mean()))
            strata.append(stratum_of(run))
        if not lengths:
            warn(f"delay_slack drops {name}: no seed was paid in its last {data.args.tail:.0%}")
            continue
        seeds = np.asarray(strata)
        length = estimate(np.asarray(lengths), seeds, data.args.resamples, rng)
        ceiling = estimate(np.asarray(pinned), seeds, data.args.resamples, rng)
        rows.append({
            "cell": name, "group": cell.group.iloc[0], **first_of(cell), **length.row(),
            **{f"pinned_{key}": value for key, value in ceiling.row().items()},
            "n_seeds": length.n_seeds,
        })
    if not rows:
        raise MissingData("no cell was paid an episode in the tail of any seed")

    table = shared_across_modes(pd.DataFrame(rows))
    varying = delay_varying(table)
    figure, axes, modes = discount_axes(table, nrows=2)
    for column, mode in enumerate(modes):
        panel = table[table.discount == mode]
        delay_series(axes[0][column], panel, "point", ("low", "high"), varying)
        delay_series(axes[1][column], panel, "pinned_point",
                     ("pinned_low", "pinned_high"), varying)
        delays = np.asarray(sorted(panel.reward_delay.unique()), dtype=float)
        axes[0][column].plot(delays, float(panel.max_steps.max()) - delays,
                             color=REFERENCE_COLOR, linestyle=BASELINE_DASH,
                             linewidth=REFERENCE_WIDTH, label="max_steps − delay")
    axes[0][0].set_ylabel("implied primitive steps to solve")
    axes[1][0].set_ylabel("payouts pinned at max_steps")
    axes[1][0].set_ylim(*FRACTION_YLIM)
    finish(figure, list(axes.flat), environments(frame),
           caption_for(table, varying,
                       (f"paid episodes, final {data.args.tail:.0%} of each seed",
                        "a pinned payout kept its reward but received a shorter delay")))
    return figure, table

def delay_sweep(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Steps to the solved rate against reward delay, one panel per discount mode."""
    frame = require_columns(
        data.episodes,
        ("primitive_step", CROSSING_COLUMN, "reward_delay", "discount"), "delay sweep"
    )
    table = shared_across_modes(
        steps_to_threshold(data, frame, np.random.default_rng(BOOTSTRAP_SEED)))
    censored = int(table.point.isna().sum())
    if censored == len(table):
        raise MissingData(f"every cell is censored at solved rate "
                          f"{table.solved_rate.iloc[0]:g}, so there is no curve to draw")
    varying = delay_varying(table)
    figure, axes, modes = discount_axes(table)
    for axis, mode in zip(axes[0], modes):
        delay_series(axis, table[table.discount == mode], "point", ("low", "high"), varying)
    axes[0][0].set_ylabel("primitive steps to the solved rate")
    finish(figure, list(axes.flat), environments(frame), "\n".join(filter(None, [
        caption_for(table, varying, (f"{solved_rate_phrase(table)}, "
                                     f"{data.args.window}-episode moving average",)),
        f"{censored} of {len(table)} cells censored, leaving a gap: fewer than "
        f"{CENSOR_FRACTION:.0%} of their seeds crossed" if censored else "",
        never_crossed(table),
    ])))
    return figure, table

def advantage_records(data: Inputs, frame: pd.DataFrame) -> Dict[Tuple[str, int, str],
                                                                 Dict[str, object]]:
    """Each cell's crossing seeds, keyed by the discount, delay and condition it holds."""
    records: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    for name, cell in frame.groupby("cell"):
        identity = first_of(cell)
        key = (str(identity["discount"]), int(identity["reward_delay"]),
               str(identity["condition"]))
        if key in records:
            raise MissingData(
                f"{records[key]['cell']} and {name} share discount={key[0]}, "
                f"delay={key[1]}, condition={key[2]}, so the ratio is ambiguous"
            )
        crossings, strata = cell_crossings(data, cell)
        crossed = np.isfinite(crossings)
        records[key] = {"cell": name, "group": cell.group.iloc[0], **identity,
                        "solved_rate": solved_rate_for(data, cell),
                        "n_crossed": int(crossed.sum()), "n_seeds": crossings.size,
                        "crossings": crossings[crossed], "strata": strata[crossed]}
    return records

def delay_advantage(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Steps to the solved rate against delay, each condition over its own crossing at delay 0.

    Normalising inside a condition rather than across the two is what lets the arms be
    compared where they hold different rates: each line is in units of its own delay-0
    crossing, so the y axis is how many times delay slowed that arm down.
    """
    frame = require_columns(
        data.episodes,
        ("primitive_step", CROSSING_COLUMN, "reward_delay", "discount"), "advantage"
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records = advantage_records(data, frame)
    # every cell estimated once, so dividing the base delay by itself is exactly one with
    # no interval, and every other delay carries the base's own error through the quotient
    estimates = {key: resampled_points(record["crossings"], record["strata"],
                                       data.args.resamples, rng)
                 for key, record in records.items()
                 if record["n_crossed"] >= CENSOR_FRACTION * record["n_seeds"]}
    rows = []
    for (mode, delay, condition), (point, draws) in estimates.items():
        base = estimates.get((mode, BASE_DELAY, condition))
        if base is None:
            if delay != BASE_DELAY:
                warn(f"delay_advantage drops {condition} at discount={mode}: its delay "
                     f"{BASE_DELAY} cell is censored, so there is nothing to normalise to")
            continue
        base_point, base_draws = base
        low, high = np.quantile(draws / base_draws, [CI_ALPHA / 2, 1 - CI_ALPHA / 2])
        record, base_record = records[(mode, delay, condition)], \
            records[(mode, BASE_DELAY, condition)]
        crossed = min(int(record["n_crossed"]), int(base_record["n_crossed"]))
        rows.append({
            "cell": record["cell"], "group": record["group"], "env_id": record["env_id"],
            "condition": condition, "discount": mode, "reward_delay": delay,
            "tag": record["tag"], "solved_rate": record["solved_rate"], "steps": point,
            "point": point / base_point, "low": float(low), "high": float(high),
            "method": ("iqm-bootstrap" if crossed >= MIN_SEEDS_FOR_IQM
                       else "median-bootstrap"),
            "n_crossed": int(record["n_crossed"]), "n_seeds": int(record["n_seeds"]),
        })
    if not rows:
        raise MissingData(f"no condition is uncensored at delay {BASE_DELAY}, so no series "
                          "has a base to normalise to")

    table = shared_across_modes(pd.DataFrame(rows))
    varying = delay_varying(table)
    figure, axes, modes = discount_axes(table)
    for axis, mode in zip(axes[0], modes):
        delay_series(axis, table[table.discount == mode], "point", ("low", "high"), varying)
        axis.axhline(NO_INTERACTION, color=REFERENCE_COLOR, linestyle=BASELINE_DASH,
                     linewidth=REFERENCE_WIDTH, label="no slowdown")
    axes[0][0].set_ylabel(f"steps to the solved rate, relative to delay {BASE_DELAY}")
    # only where the arms disagree: `caption_for` names a method the whole table shares
    methods = "" if shared_estimate(table) else " / ".join(
        sorted({METHOD_PHRASE.get(method, method) for method in table.method}))
    finish(figure, list(axes.flat), environments(frame),
           caption_for(table, varying, (solved_rate_phrase(table), methods)))
    return figure, table

def duration_vs_delay(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Realised option duration against reward delay, option cells only."""
    frame = require_columns(
        data.episodes,
        ("primitive_step", "mean_option_duration", "reward_delay", "discount"), "duration"
    )
    frame = frame[frame.condition != "action"]
    if frame.empty:
        raise MissingData("duration_vs_delay found no option cells; an action cell takes "
                          "one primitive step per decision by construction")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for name, cell in frame.groupby("cell"):
        values, strata = [], []
        for _, run in cell.groupby("seed"):
            values.append(float(tail_of(data, run).mean_option_duration.mean()))
            strata.append(stratum_of(run))
        result = estimate(np.asarray(values), np.asarray(strata), data.args.resamples, rng)
        rows.append({"cell": name, "group": cell.group.iloc[0], **first_of(cell),
                     **result.row(), "n_seeds": result.n_seeds})

    table = pd.DataFrame(rows)
    varying = delay_varying(table)
    figure, axes, modes = discount_axes(table)
    for axis, mode in zip(axes[0], modes):
        delay_series(axis, table[table.discount == mode], "point", ("low", "high"), varying)
    axes[0][0].set_ylabel("primitive steps per option")
    finish(figure, list(axes.flat), environments(frame),
           caption_for(table, varying,
                       (f"realised duration over the final {data.args.tail:.0%} of each seed",)))
    return figure, table

def delay_return(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Final episodic return against reward delay."""
    frame = require_columns(
        data.episodes,
        ("primitive_step", "episodic_return", "reward_delay", "discount"), "delay return"
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for name, cell in frame.groupby("cell"):
        result = aggregate_final(data, cell, rng)
        rows.append({"cell": name, "group": cell.group.iloc[0], **first_of(cell),
                     **result.row(), "n_seeds": result.n_seeds})

    table = shared_across_modes(pd.DataFrame(rows))
    varying = delay_varying(table)
    figure, axes, modes = discount_axes(table)
    for axis, mode in zip(axes[0], modes):
        delay_series(axis, table[table.discount == mode], "point", ("low", "high"), varying)
    axes[0][0].set_ylabel(f"episodic return, final {data.args.tail:.0%} of each seed")
    finish(figure, list(axes.flat), environments(frame), caption_for(table, varying))
    return figure, table


FIGURES: Dict[str, Callable[[Inputs], Tuple[Figure, pd.DataFrame]]] = {
    "return_curve": return_curve,
    "return_ecdf": return_ecdf,
    "success_curve": success_curve,
    "length_curve": length_curve,
    "option_count_sweep": option_count_sweep,
    "family_overlay": family_overlay,
    "count_overlay": count_overlay,
    "option_usage": option_usage,
    "threshold_table": threshold_table,
    "delay_return": delay_return,
    "delay_crossing_fraction": delay_crossing_fraction,
    "delay_slack": delay_slack,
    "delay_sweep": delay_sweep,
    "delay_advantage": delay_advantage,
    "duration_vs_delay": duration_vs_delay,
}

DISABLED = frozenset({"option_usage"})
"""Registered but not drawn unless named; each raises `MissingData` saying why."""

EXPERIMENT_FIGURES: Dict[str, Tuple[str, ...]] = {
    "exp1": ("return_curve", "return_ecdf", "threshold_table", "length_curve"),
    "exp2": ("option_count_sweep", "threshold_table", "count_overlay","option_usage"),
    "exp3": ("return_curve", "threshold_table","option_usage"),
    "exp4": ("delay_return", "delay_crossing_fraction", "delay_slack", "delay_sweep",
             "delay_advantage", "duration_vs_delay"),
}
"""What each experiment's tag is there to show; an untagged or unknown group draws
everything not `DISABLED`."""

def tags_of(frame: pd.DataFrame) -> List[str]:
    """The tags present in one group's episodes."""
    return [] if frame.empty else [str(tag) for tag in frame["tag"].unique()]

def tag_figures(tags: Sequence[str]) -> List[str]:
    """The registry entries this experiment asks for, in registry order."""
    # startswith, not equality: a variant sweep tags its cells `exp1_hard`, and it
    # wants the figures of the experiment it varies
    wanted = {name for key, names in EXPERIMENT_FIGURES.items()
              if any(tag.startswith(key) for tag in tags) for name in names}
    return [name for name in FIGURES if name in (wanted or set(FIGURES) - DISABLED)]

def selected(args: argparse.Namespace, resolved: Sequence[str]) -> List[str]:
    """The entries of `resolved` to attempt under `--only` and `--skip`."""
    return [name for name in FIGURES
            if (name in args.only if args.only else name in resolved)
            and name not in (args.skip or [])]

def condition_solved_rate(text: str) -> Union[float, Dict[str, float]]:
    """A bare solved rate, or `condition=value` for one condition alone."""
    if "=" not in text:
        return float(text)
    condition, _, value = text.partition("=")
    if condition not in CONDITION_LABEL:
        raise argparse.ArgumentTypeError(
            f"{condition!r} is not a condition: {', '.join(CONDITION_LABEL)}")
    return {condition: float(value)}

def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="*", default=["*"], help="globs, e.g. '*__exp1'")
    parser.add_argument("--only", nargs="*", help=f"figures from: {', '.join(FIGURES)}")
    parser.add_argument("--skip", nargs="*", help="figures to leave out")
    parser.add_argument("--format", default="png", choices=("svg", "pdf", "png"))
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--tail", type=float, default=0.1)
    parser.add_argument("--solved-threshold", type=condition_solved_rate,
                        help="bar a window's trailing mean must reach, in the units of "
                             "whatever is crossed: a fraction of episodes paid the goal "
                             f"reward, default {DEFAULT_SOLVED_RATE:g}, or an episodic "
                             f"return where no cell is paid, default "
                             f"{DENSE_RETURN_THRESHOLD:g}; `option=0.9` sets one condition "
                             "and leaves the rest")
    parser.add_argument("--resamples", type=int, default=2000)
    return parser.parse_args(argv)

def group_keys(frame: pd.DataFrame) -> List[str]:
    """The run groups present in `frame`."""
    return [] if frame.empty else [str(name) for name in frame["group"].unique()]

def slice_of(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """The rows of `frame` belonging to one run group."""
    return frame if frame.empty else frame[frame["group"] == group]

def draw_group(names: Sequence[str], resolved: Sequence[str], data: Inputs,
               directory: pathlib.Path) -> int:
    """Draw `names` for one group, returning how many were written.

    `resolved` is what the group's experiment asks for, which is what an output is an orphan
    against; `names` is the subset this invocation draws.
    """
    csv_directory = directory / "csv"
    csv_directory.mkdir(parents=True, exist_ok=True)
    # cleared before drawing and rewritten only by a figure that succeeds, so a figure
    # dropped from the experiment's set, and one that skipped on MissingData, leave nothing
    # behind to be read as current. A figure the experiment wants but `--only` is not
    # drawing keeps its file: asking for one figure is not a change of configuration.
    for name in FIGURES:
        if name in resolved and name not in names:
            continue
        for stale in (directory / f"{name}.{data.args.format}",
                      csv_directory / f"{name}.csv"):
            stale.unlink(missing_ok=True)
    written = 0
    for name in names:
        try:
            figure, table = FIGURES[name](data)
        except MissingData as error:
            print(f"  skipping {name}: {error}")
            continue
        # bbox_inches: long cell names on a categorical axis overrun a fixed canvas
        figure.savefig(directory / f"{name}.{data.args.format}", bbox_inches="tight")
        plt.close(figure)
        table.to_csv(csv_directory / f"{name}.csv", index=False)
        written += 1
        truncated = (f", truncated at {limit_span(table)} primitive steps"
                     if "limit" in table.columns else "")
        print(f"  wrote {name}.{data.args.format} and csv/{name}.csv{truncated}")
    return written

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Draw every selected figure for each run group of the store."""
    args = parse_arguments(argv)
    # ahead of the loaders, so a typo fails before anything reads `runs/`
    unknown = (set(args.only or []) | set(args.skip or [])) - set(FIGURES)
    if unknown:
        raise SystemExit(f"unknown figures: {', '.join(sorted(unknown))}")

    try:
        episodes = load_episodes(args.cells)
    except MissingData as error:
        raise SystemExit(str(error))

    written = attempted = 0
    for group in sorted(set(group_keys(episodes))):
        directory = PLOTS / short(group)
        print(f"{directory.name}:")
        data = Inputs(slice_of(episodes, group), args)
        resolved = tag_figures(tags_of(data.episodes))
        names = selected(args, resolved)
        if not names:
            warn("no figures selected for this group")
            continue
        written += draw_group(names, resolved, data, directory)
        attempted += len(names)

    print(f"\n{written}/{attempted} figures written under {PLOTS}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
