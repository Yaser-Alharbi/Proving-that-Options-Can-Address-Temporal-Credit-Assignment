"""Plot episodic return against primitive steps and decisions, side by side."""

import argparse
import pathlib
import textwrap
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from main import RUNS

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"

FIGURE_SIZE = (12, 6)
FIGURE_DPI = 150
MEAN_LINE_WIDTH = 2.5
GRID_ALPHA = 0.3
CAPTION_WRAP_WIDTH = 56
CAPTION_BOTTOM_MARGIN = 0.22
TITLE_TOP_MARGIN = 0.88

DECISIONS_CAPTION = (
    "The decisions axis is not budget-matched: one option decision consumes "
    "several primitive steps. The action condition is identical in both panels "
    "because one decision is one step there."
)


def load(names: Optional[Sequence[str]]) -> pd.DataFrame:
    """Every selected cell's episodes, concatenated into one frame."""
    if not RUNS.exists():
        raise SystemExit(f"no results yet: {RUNS} does not exist")
    directories = sorted(
        directory
        for directory in RUNS.iterdir()
        if (directory / "episodes.csv").exists()
        and (not names or directory.name in names)
    )
    if not directories:
        raise SystemExit(f"no cells in {RUNS} matching {list(names or ['*'])}")
    return pd.concat(
        [
            pd.read_csv(directory / "episodes.csv").assign(cell=directory.name)
            for directory in directories
        ],
        ignore_index=True,
    )


def label_for(cell: pd.DataFrame) -> str:
    """A legend label built from the columns rather than the directory name."""
    first = cell.iloc[0]
    if first.condition == "action":
        parts = ["action", first.tag]
    else:
        parts = [first.condition, first.family, f"n{first.n_options}", first.tag]
    return " ".join(str(part) for part in parts if part)


def default_name(frame: pd.DataFrame) -> str:
    """An output filename summarising the environments and conditions plotted."""
    environments = "-".join(sorted(frame.env_id.unique()))
    conditions = "-".join(sorted(frame.condition.unique()))
    return f"{environments}__{conditions}.png"


def plot_mean_curve(
    axis: Axes,
    seeds: List[Tuple[np.ndarray, np.ndarray]],
    x_max: float,
    bins: int,
    label: Optional[str],
    color: str,
) -> None:
    """Interpolate each seed's curve onto a common grid and plot only the mean."""
    grid = np.linspace(0, x_max, bins)
    stacked = np.stack([np.interp(grid, x, y) for x, y in seeds])  # (S, bins)
    axis.plot(grid, stacked.mean(axis=0), label=label, color=color,
              linewidth=MEAN_LINE_WIDTH)


def format_axis_in_standard_form(axis: Axes, which: str) -> None:
    """Show an axis's tick labels in standard form (scientific notation)."""
    axis.ticklabel_format(axis=which, style="sci", scilimits=(0, 0), useMathText=True)


def add_decisions_caption(figure: Figure, ax_decisions: Axes) -> None:
    """Note under the right panel that the decisions axis is not budget-matched."""
    figure.text(
        ax_decisions.get_position().x0,
        0.02,
        textwrap.fill(DECISIONS_CAPTION, width=CAPTION_WRAP_WIDTH),
        ha="left",
        va="bottom",
        fontsize=8,
    )


def main() -> None:
    """Plot the selected cells, averaging each cell's seeds into one curve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="*", help="cell names (default: every cell)")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--out", default=None,
                        help="bare filenames land in navix/plots/; a path is used as given")
    args = parser.parse_args()

    frame = load(args.cells)
    # every curve is truncated to the shortest seed of any cell, so the
    # comparison is over a budget all of them actually reached
    limit = int(frame.groupby(["cell", "seed"]).primitive_step.max().min())
    frame = frame[frame.primitive_step <= limit]
    print(f"truncating all cells at primitive_step={limit}")

    out = pathlib.Path(args.out or default_name(frame))
    if out.parent == pathlib.Path("."):
        out = PLOTS / out
    out.parent.mkdir(parents=True, exist_ok=True)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    figure, (ax_primitive, ax_decisions) = plt.subplots(
        1, 2, sharey=True, figsize=FIGURE_SIZE
    )
    for index, (_, cell) in enumerate(frame.groupby("cell")):
        color = color_cycle[index % len(color_cycle)]
        seeds = [
            (
                run.primitive_step.to_numpy(),
                run.decision_step.to_numpy(),
                run.episodic_return.rolling(args.window, min_periods=1)
                .mean()
                .to_numpy(),
            )
            for _, run in cell.groupby("seed")
        ]
        plot_mean_curve(
            ax_primitive,
            [(primitive, y) for primitive, _, y in seeds],
            float(limit),
            args.bins,
            f"{label_for(cell)} (n={len(seeds)})",
            color,
        )
        plot_mean_curve(
            ax_decisions,
            [(decision, y) for _, decision, y in seeds],
            min(float(decision.max()) for _, decision, _ in seeds),
            args.bins,
            None,  # the shared legend is drawn once, on the left panel
            color,
        )

    ax_primitive.set_xlabel("primitive steps")
    ax_decisions.set_xlabel("decisions")
    ax_primitive.set_ylabel(f"episodic return ({args.window}-episode moving average)")
    figure.suptitle("-".join(sorted(frame.env_id.unique())))
    ax_primitive.legend()
    ax_primitive.grid(alpha=GRID_ALPHA)
    ax_decisions.grid(alpha=GRID_ALPHA)
    ax_decisions.tick_params(labelleft=False)
    format_axis_in_standard_form(ax_primitive, "x")
    format_axis_in_standard_form(ax_decisions, "x")
    format_axis_in_standard_form(ax_primitive, "y")
    figure.tight_layout()
    figure.subplots_adjust(bottom=CAPTION_BOTTOM_MARGIN, top=TITLE_TOP_MARGIN)
    add_decisions_caption(figure, ax_decisions)
    figure.savefig(out, dpi=FIGURE_DPI)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
