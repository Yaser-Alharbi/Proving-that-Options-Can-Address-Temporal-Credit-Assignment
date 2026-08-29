"""Matrix shape and pool eta. `cd NLE && python -m pytest test_files/test_main.py -q`."""

from pathlib import Path
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
    log_progress,
    pool_eta,
    remaining_seconds,
)

EXPECTED_CELL_COUNTS = {"exp1": 3, "exp2": 13, "exp3": 7, "exp4": 15}
EXPERIMENTS = tuple(EXPECTED_CELL_COUNTS)


@pytest.fixture(scope="module")
def matrix() -> Dict[str, List[Cell]]:
    return {name: list(expand(SWEEPS[name])) for name in EXPERIMENTS}


def test_only_exp3_names_the_depth_prior(matrix: Dict[str, List[Cell]]) -> None:
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
    assert {name: len(cells) for name, cells in matrix.items()} == EXPECTED_CELL_COUNTS


def test_cell_names_are_unique_within_an_experiment(
    matrix: Dict[str, List[Cell]],
) -> None:
    for name, cells in matrix.items():
        names = [cell.name for cell in cells]
        assert len(set(names)) == len(names), f"{name} has colliding cell names"


def test_the_action_condition_takes_one_discount_mode(
    matrix: Dict[str, List[Cell]],
) -> None:
    action_cells = [cell for cell in matrix["exp4"] if cell.args.condition == "action"]
    assert {cell.args.discount for cell in action_cells} == {"decision"}
    assert len(action_cells) == len(SWEEPS["exp4"].reward_delays)


def test_the_grammar_priors_are_drawn_once_per_count(
    matrix: Dict[str, List[Cell]],
) -> None:
    for name, cells in matrix.items():
        for cell in cells:
            if cell.args.condition != "action" and cell.args.option_family != "random":
                assert cell.args.option_seed == 0, (
                    f"{name} draws {cell.args.option_family} at option_seed "
                    f"{cell.args.option_seed}, which is the seed-0 catalogue again"
                )


def test_a_bare_string_axis_is_rejected() -> None:
    with pytest.raises(AssertionError, match="trailing comma"):
        Sweep(CellArgs(), conditions="option")  # type: ignore[arg-type]


def test_log_progress_reads_the_last_frames_and_sps(tmp_path: Path) -> None:
    path = tmp_path / "seed0.log"
    path.write_text(
        "cuda\n"
        "n actions: 119\n"
        "SPS: 100\n"
        "frames=1000, episodic_return=0.0\n"
        "SPS: 968\n"
        "frames=8620149, episodic_return=0.0\n"
        "SPS: 970\n"
    )
    assert log_progress(path) == (8_620_149, 970)


def test_log_progress_is_unknown_before_either_line(tmp_path: Path) -> None:
    path = tmp_path / "seed0.log"
    path.write_text("cuda\nn actions: 119\n")
    assert log_progress(path) == (None, None)


def test_remaining_seconds_is_budget_minus_frames_over_sps() -> None:
    assert remaining_seconds(10_000_000, 8_000_000, 1000) == 2000.0


def test_remaining_seconds_at_the_budget_is_zero() -> None:
    assert remaining_seconds(10_000_000, 10_000_000, 1000) == 0.0


def test_remaining_seconds_is_unknown_without_sps() -> None:
    assert remaining_seconds(10_000_000, 100, None) is None


def test_pool_eta_is_the_last_slot_not_the_mean() -> None:
    assert pool_eta((3600.0, 3600.0), (3600.0,), 2) == 7200.0


def test_pool_eta_is_unknown_when_any_duration_is_missing() -> None:
    assert pool_eta((3600.0,), (None,), 2) is None
    assert pool_eta((None, 3600.0), (3600.0,), 2) is None
