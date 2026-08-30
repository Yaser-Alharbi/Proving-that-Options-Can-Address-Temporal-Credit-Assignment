"""Episode record and decision-trace tests."""

import csv
import json
import pathlib
import re
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from nle import nethack

from envs import OBSERVATION_KEYS, make_env
from options import make_options
from ppo import (
    BLSTATS_INDEX,
    ENCODER_OBS_DTYPE,
    EPISODE_COLUMNS,
    LOG_SCHEMA_VERSION,
    TRACE_BLSTATS_FIELDS,
    TRACE_DEVICE_FIELDS,
    TRACE_SCHEMA,
    Args,
    load_checkpoint,
    next_trace_part,
    provenance_record,
    save_checkpoint,
    trace_device_columns,
)

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

HERE = pathlib.Path(__file__).resolve().parent.parent

ENV_ID = "NetHackChallenge-v0"
GAMMA = 0.999
FULL_CATALOGUE = 227

RUN_SEED = 1
RUN_ENVS = 4
RUN_STEPS = 32
RUN_BUDGET = 6_000
RUN_MAX_EPISODE_STEPS = 400
RUN_OPTIONS = 32
RUN_TRACE_FLUSH = 2
"""Shared `option` run; flush=2 so row-group cadence fires."""

SHORT_EPISODE = 100


def ppo_argv(directory: pathlib.Path, *extra: str, budget: int = RUN_BUDGET) -> List[str]:
    """Argv for one short run into `directory`."""
    return [
        sys.executable,
        str(HERE / "ppo.py"),
        "--directory", str(directory),
        "--budget", str(budget),
        "--num-envs", str(RUN_ENVS),
        "--num-steps", str(RUN_STEPS),
        "--max-episode-steps", str(RUN_MAX_EPISODE_STEPS),
        "--condition", "option",
        "--n-options", str(RUN_OPTIONS),
        "--seed", str(RUN_SEED),
        "--trace-flush-iterations", str(RUN_TRACE_FLUSH),
        "--no-cuda",
        *extra,
    ]


@pytest.fixture(scope="module")
def training_run(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Directory of one completed short run."""
    directory = tmp_path_factory.mktemp("training_run")
    subprocess.run(ppo_argv(directory), cwd=str(HERE), check=True, capture_output=True)
    return directory


@pytest.fixture(scope="module")
def kept_run(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Directory of one completed run that kept its checkpoint."""
    directory = tmp_path_factory.mktemp("kept_run")
    subprocess.run(
        ppo_argv(directory, "--keep-checkpoint"),
        cwd=str(HERE),
        check=True,
        capture_output=True,
    )
    return directory


def tiny_trainer() -> Tuple[nn.Module, optim.Adam]:
    """Stand-in `Agent` for the checkpoint functions."""
    agent = nn.Linear(2, 2)
    return agent, optim.Adam(agent.parameters(), lr=1e-4, eps=1e-5)


@pytest.fixture(scope="module")
def episodes(training_run: pathlib.Path) -> List[Dict[str, str]]:
    """Every row of the episode CSV."""
    with open(training_run / f"episodes_seed{RUN_SEED}.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "the run finished no episodes, so nothing below is exercised"
    return rows


def decision_parts(run: pathlib.Path) -> List[pathlib.Path]:
    """Part files in part-index order."""
    parts = sorted(run.glob(f"decisions_seed{RUN_SEED}_part*.parquet"))
    assert parts, "the run wrote no trace parts"
    return parts


@pytest.fixture(scope="module")
def trace(training_run: pathlib.Path) -> Dict[str, np.ndarray]:
    """Concatenated trace parts in part then row-group order."""
    table = pa.concat_tables([pq.read_table(path) for path in decision_parts(training_run)])
    return {field.name: table.column(field.name).to_numpy() for field in TRACE_SCHEMA}


def completed_episodes(trace: Dict[str, np.ndarray], env_index: int) -> List[np.ndarray]:
    """Row indices of each finished episode in one lane, oldest first."""
    lane = np.flatnonzero(trace["env_id"] == env_index)
    blocks: List[np.ndarray] = []
    current: List[int] = []
    for position in lane:
        if trace["done"][position]:
            if current:
                blocks.append(np.array(current))
            current = []
        else:
            current.append(int(position))
    return blocks


def test_message_is_observed_but_never_reaches_the_encoder() -> None:
    """`message` is observed for the record; the encoder does not read it."""
    assert "message" in OBSERVATION_KEYS
    assert set(ENCODER_OBS_DTYPE) == {"glyphs", "blstats"}, (
        "the encoder reads exactly these two; a third key here would change the "
        "network's input and the rollout buffer's size"
    )


def test_the_vector_gives_every_env_its_own_xlogfile() -> None:
    """Each env has its own vardir."""
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=ENV_ID,
                seed=0,
                idx=index,
                condition="action",
                gamma=GAMMA,
                max_episode_steps=SHORT_EPISODE,
                clip_reward=True,
                n_options=FULL_CATALOGUE,
                option_family="grammar",
                option_seed=0,
                reward_delay=0,
            )
            for index in range(2)
        ],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    vardirs = [env.unwrapped.nethack._vardir for env in envs.envs]
    envs.close()
    assert len(set(vardirs)) == 2, (
        f"both envs write to {vardirs}; last-line reads would interleave and "
        "attribute one env's character to the other"
    )


def test_the_ingame_frame_is_the_last_primitive_of_the_decision() -> None:
    """On a live step the snapshot is the returned observation."""
    env = make_env(
        env_id=ENV_ID,
        seed=0,
        idx=0,
        condition="option",
        gamma=GAMMA,
        max_episode_steps=RUN_MAX_EPISODE_STEPS,
        clip_reward=True,
        n_options=RUN_OPTIONS,
        option_family="grammar",
        option_seed=0,
        reward_delay=0,
    )()
    observation, _ = env.reset(seed=0)
    generator = np.random.default_rng(0)
    n_actions = int(env.action_space.n)

    advanced_within_a_decision = 0
    for _ in range(200):
        before = int(observation["blstats"][nethack.NLE_BL_TIME])
        observation, _, terminated, truncated, info = env.step(
            int(generator.integers(n_actions))
        )
        if terminated or truncated:
            observation, _ = env.reset()
            continue
        assert np.array_equal(info["ingame_blstats"], observation["blstats"]), (
            "on a live step the last in-game frame is the observation the "
            "decision returned; a mismatch means the snapshot was taken at some "
            "earlier primitive of the option"
        )
        if info["primitive_steps"] > 1 and int(
            info["ingame_blstats"][nethack.NLE_BL_TIME]
        ) > before:
            advanced_within_a_decision += 1
    env.close()

    assert advanced_within_a_decision > 0, (
        "no multi-step decision advanced the turn counter, so the test never "
        "distinguished the first primitive of an option from its last"
    )


def test_the_ingame_frame_survives_the_game_ending() -> None:
    """The snapshot is the last live frame; NetHack zeroes the terminal one."""
    env = make_env(
        env_id=ENV_ID,
        seed=0,
        idx=0,
        condition="action",
        gamma=GAMMA,
        max_episode_steps=SHORT_EPISODE,
        clip_reward=True,
        n_options=FULL_CATALOGUE,
        option_family="grammar",
        option_seed=0,
        reward_delay=0,
    )()
    _, names, _ = make_options(
        env.unwrapped.actions, "action", FULL_CATALOGUE, "grammar", 0
    )
    env.reset(seed=0)
    env.step(names.index("up"))
    observation, _, terminated, _, info = env.step(names.index("nw"))
    env.close()

    assert terminated, "up then northwest should end the game, not truncate it"
    assert not observation["blstats"].any(), (
        "NetHack is expected to zero blstats once the game is over; if this "
        "fails the trap this column guards against is gone"
    )
    assert int(info["ingame_blstats"][nethack.NLE_BL_TIME]) > 0, (
        "the recorded frame is the zeroed game-over one, so every terminal "
        "column of every finished episode is a zero"
    )
    assert bytes(info["ingame_message"]).split(b"\0")[0], (
        "the terminal message is empty, so truncated episodes lose the only "
        "record of why they ended"
    )


def test_provenance_is_written_without_a_completed_cell(
    training_run: pathlib.Path,
) -> None:
    """A lone `ppo.py` writes provenance and does not write `meta.json`."""
    provenance = json.loads(
        (training_run / f"provenance_seed{RUN_SEED}.json").read_text()
    )
    assert provenance["log_schema_version"] == LOG_SCHEMA_VERSION
    assert provenance["nle_version"]
    assert provenance["host"]
    assert "git_sha" in provenance
    assert not (training_run / "meta.json").exists(), (
        "ppo.py must not write meta.json; that name belongs to main.py's "
        "finish_cell and one would clobber the other"
    )


def test_the_episode_header_is_the_declared_schema(
    training_run: pathlib.Path,
) -> None:
    """The CSV header is `EPISODE_COLUMNS`."""
    with open(training_run / f"episodes_seed{RUN_SEED}.csv", newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == EPISODE_COLUMNS


def test_the_trace_is_one_row_per_decision_at_the_declared_dtypes(
    trace: Dict[str, np.ndarray], training_run: pathlib.Path
) -> None:
    """One row per decision; columns are the declared dtypes."""
    for path in decision_parts(training_run):
        assert re.fullmatch(
            rf"decisions_seed{RUN_SEED}_part\d{{2}}\.parquet", path.name
        ), path.name
    assert not (training_run / f"decisions_seed{RUN_SEED}").exists()

    assert list(trace) == list(TRACE_SCHEMA.names)
    for field in TRACE_SCHEMA:
        expected = np.int32 if pa.types.is_int32(field.type) else np.float32
        assert trace[field.name].dtype == expected, (field.name, trace[field.name].dtype)

    lengths = {name: trace[name].shape for name in trace}
    assert len(set(lengths.values())) == 1, f"ragged part columns: {lengths}"
    assert np.isin(trace["env_id"], np.arange(RUN_ENVS)).all()


def test_the_written_schema_matches_the_declared_schema(training_run: pathlib.Path) -> None:
    """The file schema is `TRACE_SCHEMA`."""
    written = pq.read_schema(decision_parts(training_run)[0])
    assert list(written.names) == list(TRACE_SCHEMA.names)
    for written_field, declared in zip(written, TRACE_SCHEMA):
        assert written_field.type == declared.type, (written_field, declared)


def test_the_trace_row_count_matches_the_buffered_decisions(
    trace: Dict[str, np.ndarray], training_run: pathlib.Path
) -> None:
    """Every iteration writes `num_steps * num_envs` rows."""
    n_rows = trace["env_id"].shape[0]
    per_iteration = RUN_STEPS * RUN_ENVS
    assert n_rows % per_iteration == 0, (
        f"{n_rows} rows is not a whole number of iterations of {per_iteration}, "
        "so an iteration wrote a short batch"
    )
    assert n_rows >= per_iteration
    n_from_files = sum(pq.ParquetFile(path).metadata.num_rows for path in decision_parts(training_run))
    assert n_from_files == n_rows


def test_the_batched_transfer_matches_seven_separate_ones() -> None:
    """One stacked `.cpu()` matches seven per-tensor transfers."""
    steps, envs = 3, 4
    (width,) = nethack.BLSTATS_SHAPE
    ramp = torch.arange(steps * envs, dtype=torch.float32).reshape(steps, envs)
    columns = {
        "actions": ramp,
        "primitive_steps": ramp + 0.75,
        "dones": (ramp % 2.0),
        "rewards": -ramp / 8.0,
        "values": ramp / 3.0 - 2.0,
        "logprobs": -ramp - 0.5,
    }
    blstats = torch.arange(
        steps * envs * width, dtype=torch.float32
    ).reshape(steps, envs, width) / 2.0 - 1.0
    indices = [BLSTATS_INDEX[name] for name in TRACE_BLSTATS_FIELDS]

    batched = trace_device_columns(*columns.values(), blstats, indices)
    separate = [tensor.cpu().numpy().reshape(-1) for tensor in columns.values()]
    slice_ = blstats[:, :, indices].cpu().numpy()
    separate += [slice_[:, :, column].reshape(-1) for column in range(len(indices))]

    assert batched.shape == (len(TRACE_DEVICE_FIELDS), steps * envs)
    for row, (name, reference) in enumerate(zip(TRACE_DEVICE_FIELDS, separate)):
        for dtype in (np.int32, np.float32):
            assert (
                batched[row].astype(dtype).tobytes()
                == reference.astype(dtype).tobytes()
            ), (name, dtype)


def test_the_trace_schema_names_every_transferred_column() -> None:
    """`TRACE_DEVICE_FIELDS` names every transferred column."""
    host_only = ("global_step", "env_id", "term_cause", "undiscounted_reward")
    assert set(TRACE_DEVICE_FIELDS).isdisjoint(host_only)
    assert set(TRACE_DEVICE_FIELDS) | set(host_only) == set(TRACE_SCHEMA.names)


def test_a_single_column_reads_without_the_rest(training_run: pathlib.Path) -> None:
    """A single column reads without the rest."""
    path = decision_parts(training_run)[0]
    table = pq.read_table(path, columns=["value"])
    assert table.column_names == ["value"]
    assert table.schema.field("value").type == pa.float32()
    assert table.num_rows == pq.read_table(path).num_rows


def test_row_groups_are_capped_by_the_flush_cadence(training_run: pathlib.Path) -> None:
    """Row groups are capped by the flush cadence."""
    parts = decision_parts(training_run)
    assert len(parts) == 1, "a 6k-frame run must not hit the 500k-frame part rotation"
    parquet_file = pq.ParquetFile(parts[0])
    bound = RUN_TRACE_FLUSH * RUN_STEPS * RUN_ENVS
    counts = [
        parquet_file.metadata.row_group(index).num_rows
        for index in range(parquet_file.num_row_groups)
    ]
    assert parquet_file.num_row_groups > 1, counts
    assert all(count <= bound for count in counts), counts
    assert sum(counts) == parquet_file.metadata.num_rows
    assert counts[0] < parquet_file.metadata.num_rows


def test_an_unfinalised_part_leaves_earlier_parts_readable(tmp_path: pathlib.Path) -> None:
    """A kill after part 0 closes leaves that part readable."""
    script = """
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ppo import TRACE_SCHEMA

directory = sys.argv[1]
n_rows = 8
batch = pa.record_batch(
    [
        pa.array(
            np.zeros(n_rows, dtype=np.int32 if pa.types.is_int32(field.type) else np.float32),
            type=field.type,
        )
        for field in TRACE_SCHEMA
    ],
    schema=TRACE_SCHEMA,
)
part0 = os.path.join(directory, "part0.parquet")
writer = pq.ParquetWriter(part0, TRACE_SCHEMA, compression="zstd", compression_level=3)
writer.write_batch(batch)
writer.close()
part1 = os.path.join(directory, "part1.parquet")
writer = pq.ParquetWriter(part1, TRACE_SCHEMA, compression="zstd", compression_level=3)
writer.write_batch(batch)
os._exit(1)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=str(HERE),
        check=False,
    )
    assert pq.read_table(tmp_path / "part0.parquet").num_rows == 8
    with pytest.raises(pa.ArrowInvalid, match="magic bytes"):
        pq.read_table(tmp_path / "part1.parquet")


def test_the_two_discount_conventions_accumulate_different_sums(
    episodes: List[Dict[str, str]], trace: Dict[str, np.ndarray]
) -> None:
    """The two discount conventions recompute from different trace sums."""
    blocks = {
        (env_index, int(trace["global_step"][block[-1]])): block
        for env_index in range(RUN_ENVS)
        for block in completed_episodes(trace, env_index)
    }
    checked = 0
    diverged = 0
    for row in episodes:
        key = (int(row["env_seed"]) - RUN_SEED, int(row["primitive_step"]))
        if key not in blocks:
            continue
        block = blocks[key]
        assert int(row["decision_count"]) == len(block), (
            "the episode's decision count disagrees with the number of traced "
            "decisions, so the two records are not of the same episode"
        )
        steps = trace["primitive_steps"][block].astype(np.int64)
        primitives_before = np.concatenate(([0], np.cumsum(steps)[:-1]))
        expected_primitive = float(
            (GAMMA**primitives_before * trace["reward"][block]).sum()
        )
        expected_decision = float(
            (
                GAMMA ** np.arange(len(block))
                * trace["undiscounted_reward"][block]
            ).sum()
        )
        assert float(row["return_primitive"]) == pytest.approx(
            expected_primitive, abs=1e-4
        )
        assert float(row["return_decision"]) == pytest.approx(
            expected_decision, abs=1e-4
        )
        assert int(row["primitive_count"]) == int(steps.sum())
        checked += 1
        if abs(expected_primitive - expected_decision) > 1e-6:
            diverged += 1

    assert checked, "no episode could be matched to its trace rows"
    assert diverged, (
        "the two conventions agreed on every episode, so this run cannot tell "
        "them apart; it needs one with a reward inside a multi-step option"
    )


def test_steps_to_first_reward_is_an_episode_index_not_an_offset(
    episodes: List[Dict[str, str]], trace: Dict[str, np.ndarray]
) -> None:
    """`steps_to_first_reward` indexes from episode start and freezes at first reward."""
    blocks = {
        (env_index, int(trace["global_step"][block[-1]])): block
        for env_index in range(RUN_ENVS)
        for block in completed_episodes(trace, env_index)
    }
    checked = 0
    for row in episodes:
        key = (int(row["env_seed"]) - RUN_SEED, int(row["primitive_step"]))
        if key not in blocks:
            continue
        block = blocks[key]
        rewarding = np.flatnonzero(trace["undiscounted_reward"][block] != 0.0)
        recorded = int(row["steps_to_first_reward"])
        if rewarding.size == 0:
            assert recorded == -1
            assert int(row["n_nonzero_reward_steps"]) == 0
            continue
        first = int(rewarding[0])
        steps = trace["primitive_steps"][block].astype(np.int64)
        before = int(steps[:first].sum())
        assert before <= recorded < before + int(steps[first]), (
            f"steps_to_first_reward={recorded} is outside the decision that "
            f"earned it, which spent primitives [{before}, {before + int(steps[first])})"
        )
        checked += 1
    assert checked, "no episode could be matched to its trace rows"


def test_a_completed_run_leaves_no_checkpoint(training_run: pathlib.Path) -> None:
    """A finished run leaves no checkpoint."""
    assert not list(training_run.glob("checkpoint_seed*")), (
        "a finished run kept its checkpoint, which is what filled the quota"
    )


def test_a_kept_checkpoint_is_one_file_named_by_its_seed(kept_run: pathlib.Path) -> None:
    """`--keep-checkpoint` keeps one file named by seed."""
    assert [path.name for path in sorted(kept_run.glob("checkpoint_seed*"))] == [
        f"checkpoint_seed{RUN_SEED}.pt"
    ]


def test_a_second_save_lands_on_the_first(tmp_path: pathlib.Path) -> None:
    """A second save overwrites the first in place."""
    agent, optimizer = tiny_trainer()
    args = Args(seed=RUN_SEED, num_envs=RUN_ENVS)
    path = tmp_path / f"checkpoint_seed{RUN_SEED}.pt"

    save_checkpoint(path, agent, optimizer, 500_000, 10, 20, 30, args)
    save_checkpoint(path, agent, optimizer, 1_000_000, 11, 21, 31, args)

    assert [entry.name for entry in sorted(tmp_path.iterdir())] == [path.name], (
        "a second save left a second file, so the run still grows with cadence"
    )
    assert load_checkpoint(path, agent, optimizer, args) == (1_000_000, 11, 21, 31)


def test_a_save_killed_mid_write_leaves_the_previous_checkpoint(
    tmp_path: pathlib.Path, monkeypatch: "MonkeyPatch"
) -> None:
    """A kill mid-write leaves the previous checkpoint intact."""
    agent, optimizer = tiny_trainer()
    args = Args(seed=RUN_SEED, num_envs=RUN_ENVS)
    path = tmp_path / f"checkpoint_seed{RUN_SEED}.pt"
    save_checkpoint(path, agent, optimizer, 500_000, 10, 20, 30, args)
    survivor = path.read_bytes()

    def die_mid_write(state: object, target: pathlib.Path) -> None:
        pathlib.Path(target).write_bytes(b"half a checkpoint")
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr("ppo.torch.save", die_mid_write)
    with pytest.raises(RuntimeError):
        save_checkpoint(path, agent, optimizer, 1_000_000, 11, 21, 31, args)

    assert path.read_bytes() == survivor
    assert load_checkpoint(path, agent, optimizer, args) == (500_000, 10, 20, 30)


def test_the_next_trace_part_follows_the_highest_on_disk(tmp_path: pathlib.Path) -> None:
    """The next part index is one past the highest on disk."""
    assert next_trace_part(tmp_path, RUN_SEED) == 0
    for index in (0, 1, 2):
        (tmp_path / f"decisions_seed{RUN_SEED}_part{index:02d}.parquet").touch()
    (tmp_path / f"decisions_seed{RUN_SEED + 1}_part07.parquet").touch()
    assert next_trace_part(tmp_path, RUN_SEED) == 3, (
        "another seed's parts were counted, so concurrent seeds would skip indices"
    )


def test_provenance_records_no_resume_for_an_uninterrupted_run(
    training_run: pathlib.Path,
) -> None:
    """An uninterrupted run records no resume."""
    provenance = json.loads(
        (training_run / f"provenance_seed{RUN_SEED}.json").read_text()
    )
    assert provenance["resumed_at_frames"] == []


def test_provenance_accumulates_every_resume(tmp_path: pathlib.Path) -> None:
    """Each resume appends; it does not replace the list."""
    path = tmp_path / f"provenance_seed{RUN_SEED}.json"
    path.write_text(json.dumps(provenance_record(path, None)))
    path.write_text(json.dumps(provenance_record(path, 1_000)))
    assert json.loads(path.read_text())["resumed_at_frames"] == [1_000]

    path.write_text(json.dumps(provenance_record(path, 5_000)))
    assert json.loads(path.read_text())["resumed_at_frames"] == [1_000, 5_000]


def test_a_resume_continues_the_run_and_appends_to_its_logs(
    kept_run: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A rerun with budget left continues the frame counter and appends logs."""
    directory = tmp_path / "resumed"
    shutil.copytree(kept_run, directory)
    first_part = directory / f"decisions_seed{RUN_SEED}_part00.parquet"
    rows_in_first_part = pq.ParquetFile(first_part).metadata.num_rows
    with open(directory / f"episodes_seed{RUN_SEED}.csv", newline="") as handle:
        episodes_before = list(csv.DictReader(handle))
    reached = max(int(row["primitive_step"]) for row in episodes_before)

    subprocess.run(
        ppo_argv(directory, budget=RUN_BUDGET * 2), cwd=str(HERE), check=True, capture_output=True
    )

    assert pq.ParquetFile(first_part).metadata.num_rows == rows_in_first_part, (
        "the resume reopened part00, and `ParquetWriter` truncates"
    )
    assert (directory / f"decisions_seed{RUN_SEED}_part01.parquet").exists()

    with open(directory / f"episodes_seed{RUN_SEED}.csv", newline="") as handle:
        episodes_after = list(csv.DictReader(handle))
    assert len(episodes_after) > len(episodes_before), "the resume finished no episodes"
    assert episodes_after[: len(episodes_before)] == episodes_before, (
        "the resume truncated the episodes already logged"
    )
    steps = [int(row["primitive_step"]) for row in episodes_after]
    assert steps == sorted(steps), "the frame counter restarted, so the x-axis doubles back"
    assert steps[-1] > reached

    resumed = json.loads(
        (directory / f"provenance_seed{RUN_SEED}.json").read_text()
    )["resumed_at_frames"]
    assert len(resumed) == 1 and resumed[0] >= RUN_BUDGET, resumed


def test_resuming_a_finished_run_writes_no_trace_part_and_drops_the_checkpoint(
    kept_run: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A rerun with no frames left writes no part and drops the checkpoint."""
    directory = tmp_path / "finished"
    shutil.copytree(kept_run, directory)
    parts_before = {
        path.name: pq.ParquetFile(path).metadata.num_rows
        for path in directory.glob(f"decisions_seed{RUN_SEED}_part*.parquet")
    }
    episodes_before = (directory / f"episodes_seed{RUN_SEED}.csv").read_bytes()

    subprocess.run(ppo_argv(directory), cwd=str(HERE), check=True, capture_output=True)

    parts_after = {
        path.name: pq.ParquetFile(path).metadata.num_rows
        for path in directory.glob(f"decisions_seed{RUN_SEED}_part*.parquet")
    }
    assert parts_after == parts_before, (
        "the rerun opened a trace writer for a run with no frames left, so it "
        "either added an empty part or truncated one"
    )
    assert (directory / f"episodes_seed{RUN_SEED}.csv").read_bytes() == episodes_before
    assert not list(directory.glob("checkpoint_seed*")), (
        "the rerun reached the budget and kept the checkpoint anyway"
    )
    resumed = json.loads(
        (directory / f"provenance_seed{RUN_SEED}.json").read_text()
    )["resumed_at_frames"]
    assert len(resumed) == 1 and resumed[0] >= RUN_BUDGET, resumed
