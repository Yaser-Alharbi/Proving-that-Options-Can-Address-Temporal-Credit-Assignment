"""Matrix shape and pool eta. `cd NLE && python -m pytest test_files/test_main.py -q`."""

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from delayed import DELAYED_ENVS
from main import (
    ANALYSIS_SEEDS,
    DEPTH_FAMILY,
    FULL_CATALOGUE,
    SWEEPS,
    Cell,
    CellArgs,
    Job,
    Sweep,
    expand,
    log_progress,
    order_jobs,
    pool_eta,
    remaining_seconds,
)

EXPECTED_CELL_COUNTS = {"exp1": 3, "exp2": 13, "exp3": 6, "exp4": 15}
EXPERIMENTS = tuple(EXPECTED_CELL_COUNTS)

EXP2_SEED_MATRIX: Tuple[Tuple[str, Optional[int]], ...] = (
    ("option", 8),
    ("action", None),
    ("both", 8),
    ("option", 16),
    ("both", 16),
    ("option", 32),
    ("both", 32),
    ("option", 64),
    ("both", 64),
    ("option", 128),
    ("both", 128),
    ("option", FULL_CATALOGUE),
    ("both", FULL_CATALOGUE),
)
"""exp2's thirteen cells in launch order within one seed."""


@pytest.fixture(scope="module")
def matrix() -> Dict[str, List[Cell]]:
    return {name: list(expand(SWEEPS[name])) for name in EXPERIMENTS}


def cell_major_jobs(name: str) -> List[Job]:
    """Jobs before `order_jobs`: every seed of a cell, cell by cell."""
    sweep = dataclasses.replace(SWEEPS[name], seeds=ANALYSIS_SEEDS)
    return [Job(cell, seed) for cell in expand(sweep) for seed in cell.seeds]


@pytest.fixture(scope="module")
def queues() -> Dict[str, List[Job]]:
    """Launch queue per experiment at the analysis seed count."""
    return {name: order_jobs(cell_major_jobs(name)) for name in EXPERIMENTS}


def queue_labels(jobs: Sequence[Job]) -> List[Tuple[str, Optional[int], int]]:
    """`(condition, n_options, seed)` per job."""
    return [
        (
            job.cell.args.condition,
            None if job.cell.args.condition == "action" else job.cell.args.n_options,
            job.seed,
        )
        for job in jobs
    ]


def test_ordering_the_queue_runs_the_same_jobs(queues: Dict[str, List[Job]]) -> None:
    """Ordering permutes; it does not drop or duplicate."""
    for name, queue in queues.items():
        jobs = cell_major_jobs(name)
        assert len(queue) == len(jobs), name
        assert set(queue) == set(jobs), name


def test_the_queue_is_seed_major(queues: Dict[str, List[Job]]) -> None:
    """No job of a seed is queued before any job of a lower one."""
    for name, queue in queues.items():
        seeds = [job.seed for job in queue]
        assert seeds == sorted(seeds), name


def test_the_first_wave_covers_every_condition(queues: Dict[str, List[Job]]) -> None:
    """First three slots are the three conditions of seed 0."""
    assert queue_labels(queues["exp1"][:3]) == [
        ("option", 64, 0),
        ("action", None, 0),
        ("both", 64, 0),
    ]


def test_a_secondary_axis_does_not_delay_the_other_conditions(
    queues: Dict[str, List[Job]],
) -> None:
    """exp2 interleaves counts inside a seed; `action` is not delayed."""
    queue = queues["exp2"]
    width = len(EXP2_SEED_MATRIX)
    assert queue_labels(queue[:width]) == [
        (condition, n_options, 0) for condition, n_options in EXP2_SEED_MATRIX
    ]
    assert queue_labels(queue[width : 2 * width]) == [
        (condition, n_options, 1) for condition, n_options in EXP2_SEED_MATRIX
    ]


def test_no_sweep_names_the_depth_prior(matrix: Dict[str, List[Cell]]) -> None:
    """No sweep names `grammar_depth`."""
    for name, cells in matrix.items():
        families = {cell.args.option_family for cell in cells}
        assert DEPTH_FAMILY not in families, (
            f"{name} names {DEPTH_FAMILY}, so `grammar` no longer means one "
            "catalogue across the figures"
        )


def test_each_experiment_expands_to_its_planned_cell_count(
    matrix: Dict[str, List[Cell]],
) -> None:
    assert {name: len(cells) for name, cells in matrix.items()} == EXPECTED_CELL_COUNTS


def test_only_the_delay_sweeps_use_a_delayed_host(
    matrix: Dict[str, List[Cell]],
) -> None:
    """Only exp4 uses a delayed host."""
    for cell in matrix["exp4"]:
        assert cell.args.env_id in DELAYED_ENVS, cell.name
    for name in ("exp1", "exp2"):
        for cell in matrix[name]:
            assert cell.args.env_id not in DELAYED_ENVS, cell.name
        assert {cell.args.max_episode_steps for cell in matrix[name]} == {5_000}


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
