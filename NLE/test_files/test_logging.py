"""Tests for the episode record and the decision trace.

These columns are written once per episode and read only offline, so a wrong one
stays wrong until the analysis and still plots. The two pinned hardest are the
ones that would look plausible: the terminal game state, which NetHack zeroes
the moment the game ends, and the two discount conventions, which differ only in
which of the wrapper's two sums they accumulate.

The trace is an independent record of the same rollout, so the episode returns
are checked by recomputing them from the parts rather than by restating the
trainer's arithmetic.
"""

import csv
import json
import pathlib
import re
import subprocess
import sys
from typing import Dict, List

import gymnasium as gym
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
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
    trace_device_columns,
)

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
"""One short run, shared by every test that reads its output. `option`, because
it is the only condition whose decisions span more than one primitive step and
so the only one under which the two discount conventions can disagree. Flush every
two iterations so the cadence fires; at the default 64 this run is one row group."""

SHORT_EPISODE = 100


@pytest.fixture(scope="module")
def training_run(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """The run directory of one completed short training run."""
    directory = tmp_path_factory.mktemp("training_run")
    subprocess.run(
        [
            sys.executable,
            str(HERE / "ppo.py"),
            "--directory", str(directory),
            "--budget", str(RUN_BUDGET),
            "--num-envs", str(RUN_ENVS),
            "--num-steps", str(RUN_STEPS),
            "--max-episode-steps", str(RUN_MAX_EPISODE_STEPS),
            "--condition", "option",
            "--n-options", str(RUN_OPTIONS),
            "--seed", str(RUN_SEED),
            "--trace-flush-iterations", str(RUN_TRACE_FLUSH),
            "--no-cuda",
        ],
        cwd=str(HERE),
        check=True,
        capture_output=True,
    )
    return directory


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
    """Every part of the run concatenated in part then row-group order."""
    table = pa.concat_tables([pq.read_table(path) for path in decision_parts(training_run)])
    return {field.name: table.column(field.name).to_numpy() for field in TRACE_SCHEMA}


def completed_episodes(trace: Dict[str, np.ndarray], env_index: int) -> List[np.ndarray]:
    """Row indices of each finished episode in one lane, oldest first.

    `done` marks the phantom transition that follows an episode end, so an
    episode is a maximal run of `done == 0` rows closed by a phantom. A trailing
    run with no phantom after it is the episode still in flight, and is dropped.
    """
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
    """`message` is on the env for the episode record only; the network's whitelist is unchanged."""
    assert "message" in OBSERVATION_KEYS
    assert set(ENCODER_OBS_DTYPE) == {"glyphs", "blstats"}, (
        "the encoder reads exactly these two; a third key here would change the "
        "network's input and the rollout buffer's size"
    )


def test_the_vector_gives_every_env_its_own_xlogfile() -> None:
    """Two envs must not share a vardir: the episode record reads the last line of each."""
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
    """On a live step the snapshot is the returned observation, not the state the option started in."""
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
    """NetHack zeroes blstats and message at game over; the snapshot is the frame before it.

    `up` then northwest escapes the dungeon on turn one, which is a real game
    end rather than a truncation, so this is the state the episode record would
    otherwise write as dlvl=0, score=0, turn=0.
    """
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
    """A lone `ppo.py` writes its own provenance. `finish_cell` never ran here, and no group has ever cleared it."""
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
    """The CSV header is `EPISODE_COLUMNS`, so a reader can trust the schema version."""
    with open(training_run / f"episodes_seed{RUN_SEED}.csv", newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == EPISODE_COLUMNS


def test_the_trace_is_one_row_per_decision_at_the_declared_dtypes(
    trace: Dict[str, np.ndarray], training_run: pathlib.Path
) -> None:
    """Parts are int32/float32 and named `decisions_seed{N}_part{K}`."""
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
    """The file's schema is `TRACE_SCHEMA`, not inferred from the first batch."""
    written = pq.read_schema(decision_parts(training_run)[0])
    assert list(written.names) == list(TRACE_SCHEMA.names)
    for written_field, declared in zip(written, TRACE_SCHEMA):
        assert written_field.type == declared.type, (written_field, declared)


def test_the_trace_row_count_matches_the_buffered_decisions(
    trace: Dict[str, np.ndarray], training_run: pathlib.Path
) -> None:
    """Every iteration writes `num_steps * num_envs` rows, phantoms included."""
    checkpoint = torch.load(
        sorted(training_run.glob(f"checkpoint_seed{RUN_SEED}_f*.pt"))[-1],
        map_location="cpu",
        weights_only=False,
    )
    expected = checkpoint["iteration"] * RUN_STEPS * RUN_ENVS
    assert trace["env_id"].shape[0] == expected
    n_from_files = sum(pq.ParquetFile(path).metadata.num_rows for path in decision_parts(training_run))
    assert n_from_files == expected


def test_the_batched_transfer_matches_seven_separate_ones() -> None:
    """One stacked `.cpu()` writes the bytes seven per-tensor transfers wrote.

    Values are chosen to catch a cast or an ordering change: negative and
    fractional entries, which truncate toward zero into an int column, and a
    distinct value per `(step, env)` cell, which a transposed reshape reorders.
    """
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
    """`TRACE_DEVICE_FIELDS` is a split of one transfer, so a schema column it
    omits is one nothing writes."""
    host_only = ("global_step", "env_id", "term_cause", "undiscounted_reward")
    assert set(TRACE_DEVICE_FIELDS).isdisjoint(host_only)
    assert set(TRACE_DEVICE_FIELDS) | set(host_only) == set(TRACE_SCHEMA.names)


def test_a_single_column_reads_without_the_rest(training_run: pathlib.Path) -> None:
    """Column projection is the reason this is parquet rather than npz."""
    path = decision_parts(training_run)[0]
    table = pq.read_table(path, columns=["value"])
    assert table.column_names == ["value"]
    assert table.schema.field("value").type == pa.float32()
    assert table.num_rows == pq.read_table(path).num_rows


def test_row_groups_are_capped_by_the_flush_cadence(training_run: pathlib.Path) -> None:
    """The bound is the property; `> 1` holds because the fixture forces the cadence to fire."""
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
    """A kill after part 0 is closed still leaves that part readable; part 1 is lost."""
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
    """`return_primitive` discounts the wrapper's sum by elapsed primitives; `return_decision` discounts the undiscounted sum by elapsed decisions.

    Recomputed from the trace, which carries both sums separately, so feeding
    one sum to both accumulators fails here rather than producing two plausible
    curves.
    """
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
    """The column counts primitives from the episode's start, and freezes at the first one.

    An intra-option offset would sit in `[0, step_limit)` and so fall below the
    primitives already spent by the decision that earned the reward.
    """
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


def test_checkpoints_are_named_by_frame_count(training_run: pathlib.Path) -> None:
    """Checkpoints carry their frame count and never overwrite the previous one."""
    checkpoints = sorted(training_run.glob("checkpoint_seed*.pt"))
    assert checkpoints, "the run wrote no checkpoint"
    for path in checkpoints:
        assert re.fullmatch(rf"checkpoint_seed{RUN_SEED}_f\d{{9}}\.pt", path.name), (
            f"{path.name} is not named by its frame count, so a later save "
            "would overwrite it"
        )
    assert not (training_run / f"checkpoint_seed{RUN_SEED}.pt").exists(), (
        "the frameless name is the one that used to be overwritten"
    )
