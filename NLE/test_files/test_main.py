"""Tests for the matrix `main.py` expands, not for the runs it launches.

`cd src && python -m pytest test_files/test_main.py -q`.

`SWEEPS` is a table of tuples, so a wrong cell count or a family in the wrong
sweep is an edit that changes what a figure means without failing anything. These
assert the properties the figures rest on: how many cells each experiment has,
that a cell name determines a cell, and that `grammar` denotes one catalogue.
"""

from typing import Dict, List

import pytest

from main import (
    DEPTH_FAMILY,
    DEPTH_FAMILY_TAG,
    SWEEPS,
    Cell,
    CellArgs,
    Sweep,
    expand,
)

EXPECTED_CELL_COUNTS = {"exp1": 3, "exp2": 13, "exp3": 7, "exp4": 15}
"""What each experiment's figures are drawn from. exp2's action arm is one cell
because the action condition has no catalogue to size; exp3's is 1 breadth cell,
1 depth cell and 5 random draws; exp4's is 5 delays for action and 5 delays by 2
discount modes for option."""

EXPERIMENTS = tuple(EXPECTED_CELL_COUNTS)


@pytest.fixture(scope="module")
def matrix() -> Dict[str, List[Cell]]:
    """Every experiment's cells, expanded once."""
    return {name: list(expand(SWEEPS[name])) for name in EXPERIMENTS}


def test_only_exp3_names_the_depth_prior(matrix: Dict[str, List[Cell]]) -> None:
    """`grammar_depth` appears in exp3 and nowhere else.

    If another sweep names it, `grammar` denotes one catalogue in some figures
    and another in the rest, and a reader cannot tell which from the label. exp3
    is where the two priors are compared on purpose, at fixed n.
    """
    for name, cells in matrix.items():
        families = {cell.args.option_family for cell in cells}
        if name == DEPTH_FAMILY_TAG:
            assert DEPTH_FAMILY in families, (
                f"{name} is the sweep that measures the prior and must run it"
            )
        else:
            assert DEPTH_FAMILY not in families, (
                f"{name} names {DEPTH_FAMILY}, so `grammar` no longer means one "
                "catalogue across the figures"
            )


def test_each_experiment_expands_to_its_planned_cell_count(
    matrix: Dict[str, List[Cell]],
) -> None:
    """The matrix is the size the schedule was costed against."""
    assert {name: len(cells) for name, cells in matrix.items()} == EXPECTED_CELL_COUNTS


def test_cell_names_are_unique_within_an_experiment(
    matrix: Dict[str, List[Cell]],
) -> None:
    """Two cells sharing a name would run into one directory.

    The name is the only thing separating them on disk, and every axis a sweep
    varies has to appear in it or two cells interleave their episode rows.
    """
    for name, cells in matrix.items():
        names = [cell.name for cell in cells]
        assert len(set(names)) == len(names), f"{name} has colliding cell names"


def test_the_action_condition_takes_one_discount_mode(
    matrix: Dict[str, List[Cell]],
) -> None:
    """exp4 sweeps both modes but its action arm is not doubled.

    A primitive spends one primitive step, so `gamma ** primitive_steps` is
    `gamma` and the two modes compute the same discount. Running both would be
    two names for one cell.
    """
    action_cells = [cell for cell in matrix["exp4"] if cell.args.condition == "action"]
    assert {cell.args.discount for cell in action_cells} == {"decision"}
    assert len(action_cells) == len(SWEEPS["exp4"].reward_delays)


def test_the_grammar_priors_are_drawn_once_per_count(
    matrix: Dict[str, List[Cell]],
) -> None:
    """No sweep runs a grammar family at more than one option seed.

    Neither grammar key reads `option_seed`, so a second seed is the same
    catalogue under a different directory name and buys nothing.
    """
    for name, cells in matrix.items():
        for cell in cells:
            if cell.args.condition != "action" and cell.args.option_family != "random":
                assert cell.args.option_seed == 0, (
                    f"{name} draws {cell.args.option_family} at option_seed "
                    f"{cell.args.option_seed}, which is the seed-0 catalogue again"
                )


def test_a_bare_string_axis_is_rejected() -> None:
    """`conditions="option"` is a typo that would expand to six letters.

    A string is iterable, so `expand` would treat it as an axis of characters
    and produce cells for conditions that do not exist.
    """
    with pytest.raises(AssertionError, match="trailing comma"):
        Sweep(CellArgs(), conditions="option")  # type: ignore[arg-type]
