"""Figures for the navix experiment matrix, one registry entry each.

Every figure returns the frame it drew, and the runner writes that frame beside the image,
so a figure can be regenerated without reading `runs/` again.
"""

import argparse
import fnmatch
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import ScalarFormatter
from scipy.stats import trim_mean

from config import Args
from main import RUNS, THRESHOLDS, cell_name

matplotlib.use("Agg")

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"

MIN_SEEDS_FOR_IQM = 8
"""Below this a 25% trim keeps too few values to be a trimmed mean."""

IQM_TRIM = 0.25
CI_ALPHA = 0.05
BOOTSTRAP_SEED = 0

FIGURE_SIZE = (9.0, 5.5)
PANEL_WIDTH = 5.0
ROW_HEIGHT = 0.4
LINE_WIDTH = 2.0
BAND_ALPHA = 0.2
GRID_ALPHA = 0.3
SMALL_FONT = 7
LEGEND_FONT = 8
CAP_MARKER_SIZE = 400
ERROR_CAP_SIZE = 4
X_LABEL = "primitive steps"

SCI_LIMITS = (-3, 4)
"""Tick magnitudes printed plainly; a step axis exceeds this, a duration does not."""

LEGEND_TOP = 0.0
"""Figure fraction the legend hangs from, so it clears the figure entirely."""

DEFAULT_THRESHOLD = 0.5
"""Crossing threshold for a cell whose sweep declares none."""

SETTLE_TOL = 0.02
"""Fraction of a curve's own range within which it counts as no longer changing."""

LOG_X_RATIO = 0.25
"""Log the x axis once the fastest curve settles inside this fraction of it."""

CONDITION_COLOR: Dict[str, str] = {"action": "#4c72b0", "option": "#dd8452", "both": "#55a868"}
CONDITION_LABEL: Dict[str, str] = {
    "action": "action space", "option": "option space",
    "both": "both (option + action space)",
}
FAMILY_DASH: Dict[str, str] = {"grammar": "-", "random": "--"}

COUNT_COLORMAP = "viridis"
COUNT_COLOR_RANGE = (0.15, 0.9)
"""Ends trimmed off the ramp: its pale tail is invisible on white."""

BASELINE_DASH = "--"
"""How the action baseline is drawn where colour is spent on catalogue size."""

SERIES_KEYS: Tuple[str, ...] = (
    "env_id", "condition", "family", "option_seed", "budget", "max_forward", "max_steps",
    "reward_delay", "gamma", "discount", "executor", "tag",
)
"""Everything identifying a cell but its catalogue size, so a sweep over `n_options` never
pools two cells differing in anything else."""

IDENTITY: Tuple[str, ...] = SERIES_KEYS + ("n_options",)

OPTION_ONLY: Tuple[str, ...] = ("family", "n_options", "option_seed")
"""Fields `Cell.identity` fills with a sentinel on an `action` cell, so a difference in one
of them between an action cell and an option cell means only "not applicable"."""

ARGS_FIELD: Dict[str, str] = {"condition": "action_space", "family": "option_family"}
"""Identity columns whose `Args` field carries another name."""

FIELD_PREFIX: Dict[str, str] = {"family": "", "n_options": "n=", "option_seed": "os="}
"""How a varying field is written in a legend entry; anything else is written `key=`."""

DEFAULTS = Args()
"""`Cell.name` elides a field left at its default, so a caption elides it too."""

UNTAGGED = "untagged"
"""Group name for cells run without a tag, which `Cell.name` simply omits."""

DURATION_MARKERS: Tuple[Tuple[str, str, str], ...] = (
    ("nominal_option_len", "nominal", "o"),
    ("mean_option_len", "measured mean", "s"),
    ("duration_max_lane_mean", "worst lane, mean", "^"),
    ("duration_max_lane_max", "worst lane, max", "D"),
)

CENSORED: Dict[str, object] = {
    "point": float("nan"), "low": float("nan"), "high": float("nan"), "method": "censored",
}

METHOD_PHRASE: Dict[str, str] = {
    "iqm-bootstrap": f"IQM with {1 - CI_ALPHA:.0%} bootstrap CI",
    "median-range": "median with observed range",
}
"""How a method reads on a figure; the sidecar CSVs keep the token itself."""


class MissingData(Exception):
    """A figure cannot be drawn from the cells present."""

def warn(message: str) -> None:
    """Print a non-fatal problem with the store."""
    print(f"  warning: {message}")

def short(name: str) -> str:
    """A name without the environment boilerplate, for ticks and legends."""
    return name.replace("Navix-", "").replace("-v0", "")

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
    """Every matching cell's episodes, concatenated, with `cell` and `group` columns."""
    frames = []
    for directory in cell_directories(patterns):
        path = directory / "episodes.csv"
        if path.exists():
            frames.append(pd.read_csv(path).assign(cell=cell_name(directory),
                                                   group=directory.parent.name))
        else:
            warn(f"{cell_name(directory)} has no episodes.csv")
    if not frames:
        raise MissingData("no matching cell has an episodes.csv")
    frame = pd.concat(frames, ignore_index=True)
    frame["tag"] = frame.tag.fillna(UNTAGGED).replace("", UNTAGGED)
    return frame

def load_meta(patterns: Sequence[str]) -> pd.DataFrame:
    """One row per matching cell, from `meta.json`, with `duration_stats` flattened."""
    rows: List[Dict[str, object]] = []
    for directory in cell_directories(patterns):
        path = directory / "meta.json"
        if not path.exists():
            warn(f"{cell_name(directory)} has no meta.json, so it never finished")
            continue
        meta = json.loads(path.read_text())
        arguments = meta["args"]
        rows.append({
            "cell": cell_name(directory),
            "group": directory.parent.name,
            "env_id": arguments["env_id"],
            "condition": arguments["action_space"],
            "family": arguments["option_family"],
            "n_options": arguments["n_options"],
            "option_seed": arguments["option_seed"],
            "tag": arguments["tag"] or UNTAGGED,
            **{k: meta[k] for k in ("max_option_len", "mean_option_len", "nominal_option_len")},
            **{f"duration_{k}": v for k, v in meta["duration_stats"].items()},
        })
    if not rows:
        raise MissingData("no matching cell has a meta.json")
    return pd.DataFrame(rows)

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

def crossing_step(run: pd.DataFrame, column: str, threshold: float,
                  window: int) -> float:
    """One seed's primitive step at its first trailing mean of `column` past `threshold`.

    Read off the seed's own episodes, not off an interpolated grid whose bin width scales
    with the run length and quantises every cell to the same few steps.
    """
    run = run.sort_values("primitive_step")
    # min_periods=window: a seed whose first episode happened to succeed would otherwise
    # cross at its first step, and one too short to fill a window never crossed
    smoothed = run[column].rolling(window, min_periods=window).mean().to_numpy()
    at = np.flatnonzero(smoothed >= threshold)
    return float(run.primitive_step.to_numpy()[at[0]]) if at.size else float("nan")


@dataclass(frozen=True)
class Inputs:
    """What every figure is handed: the two loaded frames and the options."""

    episodes: pd.DataFrame
    meta: pd.DataFrame
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

def aggregate_final(data: Inputs, frame: pd.DataFrame, rng: np.random.Generator) -> Estimate:
    """The estimate over each seed's last `--tail` of primitive steps."""
    values, strata = [], []
    for _, run in frame.groupby(["cell", "seed"]):
        reached = run.primitive_step.max()
        tail = run[run.primitive_step > (1 - data.args.tail) * reached]
        values.append(tail.episodic_return.mean())
        strata.append(stratum_of(run))
    return estimate(np.asarray(values), np.asarray(strata), data.args.resamples, rng)

def threshold_for(data: Inputs, cell: pd.DataFrame) -> float:
    """`--threshold`, else what this cell's sweep declared, else `DEFAULT_THRESHOLD`."""
    if data.args.threshold is not None:
        return float(data.args.threshold)
    first = cell.iloc[0]
    return THRESHOLDS.get((first.env_id, first.tag), DEFAULT_THRESHOLD)

def steps_to_threshold(data: Inputs, frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Primitive steps to reach each cell's threshold, excluding seeds that never do."""
    rows = []
    for name, cell in frame.groupby("cell"):
        threshold = threshold_for(data, cell)
        crossings = [crossing_step(run, "episodic_return", threshold, data.args.window)
                     for _, run in cell.groupby("seed")]
        strata = np.asarray([stratum_of(run) for _, run in cell.groupby("seed")])
        crossed = np.isfinite(crossings)
        n_seeds, n_crossed = len(crossings), int(crossed.sum())
        # pinning a non-crossing seed to the limit would bias the estimate down by
        # an amount the budget sets, so it is dropped and the fraction reported
        measured = estimate(np.asarray(crossings)[crossed], strata[crossed],
                            data.args.resamples, rng).row() \
            if 2 * n_crossed >= n_seeds else CENSORED
        rows.append({
            "cell": name, "group": cell.group.iloc[0], **first_of(cell),
            "threshold": threshold, **measured, "n_crossed": n_crossed,
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
    """`n seeds, method` where every cell of `table` agrees on both, else empty."""
    if not {"n_seeds", "method"} <= set(table.columns):
        return ""
    if table.n_seeds.nunique() != 1 or table.method.nunique() != 1:
        return ""
    method = str(table.method.iloc[0])
    return f"{int(table.n_seeds.iloc[0])} seeds, {METHOD_PHRASE.get(method, method)}"

def limit_span(table: pd.DataFrame) -> str:
    """The range of per-cell truncation limits, as one number where they agree."""
    low, high = int(table.limit.min()), int(table.limit.max())
    return f"{low:,}" if low == high else f"{low:,} to {high:,}"

def caption_for(table: pd.DataFrame, varying: Sequence[str],
                extra: Sequence[str] = ()) -> str:
    """Everything every series of `table` shares, for the legend title.

    A field at its `Args` default is left out, as `Cell.name` leaves it out, so the caption
    and a legend entry together name the cell. `n_options` is the exception, named on every
    option cell there and so named here too. The group is not named: it is the directory
    the figure is written into, and `tag` only repeats it.
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
    """
    # the seed count and method go in the caption where every cell agrees, which is the
    # usual case, and stay on the entry where a short cell fell back to median-and-range
    suffix = "" if shared_estimate(table) else " ({} seeds, {})"
    # sorted by what the legend prints, so `n=8` precedes `n=16`; grouping by the cell name
    # alone orders the legend lexically, and sort=False then keeps this order
    order = [*dict.fromkeys(["_condition_at", *varying]), "cell"]
    ordered = table.assign(_condition_at=table.condition.map(list(CONDITION_LABEL).index))
    for _, group in ordered.sort_values(order).groupby("cell", sort=False):
        first = group.iloc[0]
        counted = colors is not None and first.condition != "action"
        color = colors[int(first.n_options)] if counted else CONDITION_COLOR[first.condition]
        axis.plot(group.primitive_step, group.point, color=color, linewidth=LINE_WIDTH,
                  linestyle=FAMILY_DASH.get(first.family, "-") if colors is None
                  else ("-" if counted else BASELINE_DASH),
                  label=f"n={first.n_options:g}" if counted else
                  series_label(first, varying)
                  + suffix.format(int(first.n_seeds),
                                  METHOD_PHRASE.get(first.method, first.method)))
        axis.fill_between(group.primitive_step, group.low, group.high, color=color,
                          alpha=BAND_ALPHA, linewidth=0)

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
    """
    absent = set(frame.condition) - set(table.condition)
    if not absent or "terminated" not in data.episodes.columns:
        return ""
    rows = data.episodes[data.episodes.condition.isin(absent)]
    return (f"{labels_for(absent)} omitted: terminated {rows.terminated.mean():.2%} "
            "of episodes, over no step range every seed covers")

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

def success_curve(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Fraction of episodes that terminated, against primitive steps."""
    frame = require_columns(
        data.episodes, ("primitive_step", "terminated", "episodic_return"), "success"
    )
    if collinear(frame, "terminated", "episodic_return"):
        raise MissingData(
            "episodic_return takes one value per termination outcome in every loaded "
            "cell, so a success curve is the return curve rescaled"
        )
    return curve_figure(data, frame, "terminated", "success rate")

def length_curve(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Episode length among terminated episodes, against primitive steps."""
    frame = require_columns(
        data.episodes, ("primitive_step", "episodic_length", "terminated"), "length"
    )
    # `episodic_length` is `episode_t`, primitive steps inside the episode, so it compares
    # across conditions unconverted; among terminated episodes only, since a truncated one
    # reports the truncation limit rather than a cost of solving
    solved = frame[frame.terminated == 1]
    if solved.empty:
        raise MissingData("no episode of any matching cell terminated")
    return curve_figure(data, solved, "episodic_length",
                        "primitive steps per terminated episode")

def family_overlay(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Grammar against random return curves, one panel per catalogue size."""
    frame = require_columns(data.episodes, ("primitive_step", "episodic_return", "family"),
                            "overlay")
    frame = frame[frame.condition != "action"]
    counts = [n for n, panel in frame.groupby("n_options") if panel.family.nunique() > 1]
    for count in sorted(set(frame.n_options.unique()) - set(counts)):
        warn(f"family_overlay drops n={count}: only one family was run at it")
    if not counts:
        raise MissingData("no catalogue size has both grammar and random")

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
    # dropna=False: reward_delay and gamma are absent from cells written before they
    # were logged, and a dropped key would silently drop those cells
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

def duration_vs_cap(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Measured mean duration by catalogue draw, and worst-lane duration against the cap."""
    columns = tuple(column for column, _, _ in DURATION_MARKERS)
    table = require_columns(data.meta, columns + ("max_option_len",), "duration")
    table = table[table.condition != "action"].copy()
    if table.empty:
        raise MissingData("duration_vs_cap found no option cells")
    caps = table.max_option_len.unique()
    if caps.size < 2:
        raise MissingData(f"every loaded cell caps options at {caps[0]:g} primitive steps, "
                          "so there is nothing to compare")
    table["saturation"] = table.duration_max_lane_max / table.max_option_len
    table = pd.concat([
        table[table.family.eq("grammar")].sort_values("mean_option_len"),
        table[table.family.ne("grammar")].sort_values("mean_option_len"),
    ], ignore_index=True)
    varying = varying_fields(table)
    at = np.arange(len(table))
    labels = [series_label(row, varying) for _, row in table.iterrows()]
    mean_series = DURATION_MARKERS[:2]
    lane_series = DURATION_MARKERS[2:]
    mean_columns = [column for column, _, _ in mean_series]
    lane_columns = [column for column, _, _ in lane_series] + ["max_option_len"]
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_grammar = int(table.family.eq("grammar").sum())

    figure, (means, lanes) = plt.subplots(
        1, 2, sharey=True,
        figsize=(FIGURE_SIZE[0], 2.0 + ROW_HEIGHT * len(table)),
    )
    for index, (column, label, marker) in enumerate(mean_series):
        means.scatter(table[column], at, marker=marker, label=label, color=cycle[index])
    left = table[mean_columns].to_numpy()
    left_pad = max(0.08 * float(left.max() - left.min()), 0.25)
    means.set_xlim(float(left.min()) - left_pad, float(left.max()) + left_pad)
    means.set_xlabel("primitive steps per option")
    means.set_yticks(at)
    means.set_yticklabels(labels, fontsize=SMALL_FONT)
    means.invert_yaxis()

    lanes.hlines(at, table.duration_max_lane_mean, table.duration_max_lane_max,
                 color="0.85", zorder=0)
    lanes.scatter(table.max_option_len, at, marker="|", s=CAP_MARKER_SIZE, color="black",
                  label="cap")
    for index, (column, label, marker) in enumerate(lane_series):
        lanes.scatter(table[column], at, marker=marker, label=label,
                      color=cycle[len(mean_series) + index])
    lanes.set_xlim(0.0, float(table[lane_columns].to_numpy().max()) * 1.08)
    lanes.set_xlabel("primitive steps per option")
    lanes.tick_params(axis="y", left=False, labelleft=False)
    finish(figure, [means, lanes], "realised option duration against the cap",
           caption_for(table, varying))
    for axis in (means, lanes):
        axis.yaxis.grid(False)
    if 0 < n_grammar < len(table):
        y = n_grammar - 0.5
        for axis in (means, lanes):
            axis.axhline(y, color="0.4", linewidth=0.8)
        figure.add_artist(ConnectionPatch(
            (means.get_xlim()[1], y), (lanes.get_xlim()[0], y),
            coordsA=means.transData, coordsB=lanes.transData,
            color="0.4", linewidth=0.8, clip_on=False,
        ))
    return figure, table

def option_usage(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Blocked: nothing in the store records which option index was selected."""
    raise MissingData(
        "no per-option selection counts exist: ppo.py would need a bincount over "
        "experience.action, and Results.append keeps only ndim==2 logs"
    )

def threshold_table(data: Inputs) -> Tuple[Figure, pd.DataFrame]:
    """Primitive steps to reach the return threshold, as a rendered table."""
    frame = require_columns(data.episodes, ("primitive_step", "episodic_return"), "threshold")
    table = steps_to_threshold(data, frame, np.random.default_rng(BOOTSTRAP_SEED))
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
    figure.suptitle(f"primitive steps to episodic return {table.threshold.iloc[0]:g} "
                    f"({data.args.window}-episode moving average)",
                    x=0.0, y=0.99, ha="left")
    if caption:
        figure.text(1.0, 0.90, caption, ha="right", va="top", fontsize=SMALL_FONT)
    return figure, table


FIGURES: Dict[str, Callable[[Inputs], Tuple[Figure, pd.DataFrame]]] = {
    "return_curve": return_curve,
    "success_curve": success_curve,
    "length_curve": length_curve,
    "option_count_sweep": option_count_sweep,
    "family_overlay": family_overlay,
    "count_overlay": count_overlay,
    "duration_vs_cap": duration_vs_cap,
    "option_usage": option_usage,
    "threshold_table": threshold_table,
}

DISABLED = frozenset({"option_usage"})
"""Registered but not drawn unless named; each raises `MissingData` saying why."""

EXPERIMENT_FIGURES: Dict[str, Tuple[str, ...]] = {
    "exp1": ("return_curve", "threshold_table", "length_curve"),
    "exp2": ("option_count_sweep", "threshold_table", "count_overlay"),
    "exp3": ("return_curve", "threshold_table","duration_vs_cap"),
}
"""What each experiment's tag is there to show; an untagged or unknown group draws
everything not `DISABLED`."""

def tags_of(frame: pd.DataFrame) -> List[str]:
    """The tags present in one group's episodes."""
    return [] if frame.empty else [str(tag) for tag in frame["tag"].unique()]

def tag_figures(tags: Sequence[str]) -> List[str]:
    """The registry entries this experiment asks for, in registry order."""
    # startswith, not equality: a variant sweep tags its cells `exp1_16x16_Random`, and it
    # wants the figures of the experiment it varies
    wanted = {name for key, names in EXPERIMENT_FIGURES.items()
              if any(tag.startswith(key) for tag in tags) for name in names}
    return [name for name in FIGURES if name in (wanted or set(FIGURES) - DISABLED)]

def selected(args: argparse.Namespace, resolved: Sequence[str]) -> List[str]:
    """The entries of `resolved` to attempt under `--only` and `--skip`."""
    return [name for name in FIGURES
            if (name in args.only if args.only else name in resolved)
            and name not in (args.skip or [])]

def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="*", default=["*"], help="globs, e.g. '*__hard'")
    parser.add_argument("--only", nargs="*", help=f"figures from: {', '.join(FIGURES)}")
    parser.add_argument("--skip", nargs="*", help="figures to leave out")
    parser.add_argument("--format", default="png", choices=("svg", "pdf","png"))
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--tail", type=float, default=0.1)
    parser.add_argument("--threshold", type=float,
                        help=f"overrides the sweep's, default {DEFAULT_THRESHOLD:g}")
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
    # ahead of the figures, and unconditionally: the measured durations convert the budget
    # into decisions whether or not `duration_vs_cap` had two caps to draw
    if not data.meta.empty:
        data.meta.to_csv(csv_directory / "duration.csv", index=False)
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

    loaded = []
    for loader in (load_episodes, load_meta):
        try:
            loaded.append(loader(args.cells))
        except MissingData as error:
            warn(str(error))
            loaded.append(pd.DataFrame())
    episodes, meta = loaded

    keys = sorted(set(group_keys(episodes)) | set(group_keys(meta)))
    written = attempted = 0
    for group in keys:
        directory = PLOTS / short(group)
        print(f"{directory.name}:")
        data = Inputs(slice_of(episodes, group), slice_of(meta, group), args)
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
