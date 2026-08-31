"""Tests for the exp4 reward delay."""

import csv
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pytest
from nle.env.base import NLE
from nle.env.tasks import NetHackOracle, NetHackStaircase, NetHackStaircasePet

from delayed import DELAYED_ENVS, HeldGoal
from envs import (
    OBSERVATION_KEYS,
    REWARD_CLIP,
    TASK_SUCCESSFUL,
    StreamFlushClip,
    make_env,
)
from options import _catalogue, catalogue_digest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

HERE = pathlib.Path(__file__).resolve().parent.parent

DELAYED_STAIRCASE = "DelayedStaircase-v0"
HORIZON = 40
"""Covers NLE start-up menus; still reaches the horizon."""

FORCE_SOLVED = """
import runpy
import torch

torch.set_num_threads(1)
from nle.env.tasks import NetHackStaircase

NetHackStaircase._is_episode_end = (
    lambda self, observation: self.StepStatus.TASK_SUCCESSFUL
)
runpy.run_path("ppo.py", run_name="__main__")
"""
"""`ppo.py` under `__main__` with the goal always firing."""

CSV_RUN_SEED = 3
CSV_RUN_ENVS = 2
CSV_RUN_STEPS = 16
CSV_RUN_BUDGET = 128
"""One forced-solve run."""

STAIRS_DOWN = 4
"""Index of the stairs_down flag in `internal`."""

NO_REWARD = 0.0
"""Reward of a step that does not flush the bank."""

GOAL_REWARD = 1.0


def delayed_env(reward_delay: int, horizon: int = HORIZON) -> Any:
    """One reset `DelayedStaircase`, latch clear."""
    env = DELAYED_ENVS[DELAYED_STAIRCASE](
        observation_keys=OBSERVATION_KEYS,
        max_episode_steps=horizon,
        reward_delay=reward_delay,
    )
    env.reset(seed=0)
    return env


def solved_observation(env: Any) -> Tuple[np.ndarray, ...]:
    """Live observation with the stairs_down flag set."""
    observation = tuple(array.copy() for array in env.last_observation)
    observation[env._internal_index][STAIRS_DOWN] = 1
    return observation


def statuses_until_terminal(env: Any, observation: Any, limit: int) -> list:
    """`_is_episode_end` until the first non-RUNNING answer."""
    statuses = []
    for _ in range(limit):
        status = env._is_episode_end(observation)
        statuses.append(status)
        if status != env.StepStatus.RUNNING:
            break
    return statuses


@pytest.mark.parametrize("reward_delay", [0, 1, 4, 16])
def test_the_goal_is_held_for_exactly_reward_delay_steps(reward_delay: int) -> None:
    """The task ends `reward_delay` steps after first solved."""
    env = delayed_env(reward_delay)
    statuses = statuses_until_terminal(env, solved_observation(env), reward_delay + 2)
    env.close()

    assert statuses[-1] == env.StepStatus.TASK_SUCCESSFUL
    assert all(status == env.StepStatus.RUNNING for status in statuses[:-1])
    assert len(statuses) - 1 == reward_delay


def test_an_unsolved_observation_never_latches() -> None:
    """A game that does not reach the goal keeps its own end status."""
    env = delayed_env(reward_delay=8)
    unsolved = tuple(array.copy() for array in env.last_observation)
    assert unsolved[env._internal_index][STAIRS_DOWN] == 0, (
        "the fixture must not start on a staircase or the test is vacuous"
    )

    statuses = [env._is_episode_end(unsolved) for _ in range(20)]
    banked = env._reward_fn(unsolved, 0, unsolved, env.StepStatus.ABORTED)
    env.close()

    assert all(status == env.StepStatus.RUNNING for status in statuses)
    assert banked == pytest.approx(NO_REWARD)


def test_a_delayed_host_charges_no_per_step_penalty() -> None:
    """The goal reward is the only nonzero reward."""
    env = delayed_env(reward_delay=0)
    penalty_step = env.penalty_step
    env.close()

    assert penalty_step == 0.0


def test_nothing_is_paid_during_the_hold() -> None:
    """Held steps earn nothing; the goal reward arrives once."""
    reward_delay = 5
    env = delayed_env(reward_delay)
    observation = solved_observation(env)

    held = []
    for _ in range(reward_delay):
        status = env._is_episode_end(observation)
        held.append(env._reward_fn(observation, 0, observation, status))
    final_status = env._is_episode_end(observation)
    paid = env._reward_fn(observation, 0, observation, final_status)
    env.close()

    assert final_status == env.StepStatus.TASK_SUCCESSFUL
    assert held == pytest.approx([NO_REWARD] * reward_delay)
    assert paid == pytest.approx(GOAL_REWARD)


@pytest.mark.parametrize("end_status", ["ABORTED", "DEATH"])
def test_a_terminal_step_inside_the_hold_still_pays(end_status: str) -> None:
    """Horizon and death inside the hold flush the bank."""
    env = delayed_env(reward_delay=1000)
    observation = solved_observation(env)
    env._is_episode_end(observation)
    paid = env._reward_fn(
        observation, 0, observation, getattr(env.StepStatus, end_status)
    )
    env.close()

    assert paid == pytest.approx(GOAL_REWARD)


def test_reset_clears_the_latch() -> None:
    """Reset clears the hold and the payout flag."""
    env = delayed_env(reward_delay=8)
    env._is_episode_end(solved_observation(env))
    env._reward_fn(
        env.last_observation, 0, env.last_observation, env.StepStatus.ABORTED
    )
    assert env._held is not None
    assert env._paid

    env.reset(seed=1)
    assert env._held is None
    assert not env._paid
    unpaid = env._reward_fn(
        env.last_observation, 0, env.last_observation, env.StepStatus.ABORTED
    )
    env.close()
    assert unpaid == pytest.approx(NO_REWARD)


def test_the_internal_horizon_terminates_rather_than_truncates() -> None:
    """NLE's step limit arrives as `terminated` with ABORTED."""
    env = delayed_env(reward_delay=8, horizon=HORIZON)
    for step in range(HORIZON):
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    env.close()

    assert step + 1 == HORIZON, "the horizon should be what ends this episode"
    assert terminated and not truncated
    assert info["end_status"] == NetHackStaircase.StepStatus.ABORTED


def always_solved(self: Any, observation: Any) -> int:
    """Goal predicate that fires every step."""
    del observation
    return self.StepStatus.TASK_SUCCESSFUL


def test_a_horizon_flush_reports_paid_without_reporting_success(
    monkeypatch: "MonkeyPatch",
) -> None:
    """A hold that outlives the horizon pays and still ends ABORTED."""
    monkeypatch.setattr(NetHackStaircase, "_is_episode_end", always_solved)
    env = delayed_env(reward_delay=HORIZON * 2)

    for _ in range(HORIZON):
        _, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    env.close()

    assert terminated and not truncated
    assert info["end_status"] == NetHackStaircase.StepStatus.ABORTED
    assert info["paid"], "the flush at the horizon has to reach the info dict"
    assert reward > 0, "a flushed bank pays the goal reward on its terminal step"


def test_a_flushed_episode_logs_paid_without_logging_solved(
    monkeypatch: "MonkeyPatch",
) -> None:
    """Trainer writes `paid=1`, `solved=0` for a flush."""
    monkeypatch.setattr(NetHackStaircase, "_is_episode_end", always_solved)
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=DELAYED_STAIRCASE,
                seed=0,
                idx=0,
                condition="action",
                gamma=0.999,
                max_episode_steps=HORIZON,
                clip_reward=True,
                n_options=64,
                option_family="grammar",
                option_seed=0,
                reward_delay=HORIZON * 2,
            )
        ],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    envs.reset(seed=0)

    infos: dict = {}
    for _ in range(HORIZON):
        _, _, _, _, infos = envs.step(np.zeros(1, dtype=int))
        if "episode" in infos:
            break
    envs.close()

    assert "episode" in infos, "the horizon should have ended the episode"
    solved = int(int(infos["end_status"][0]) == TASK_SUCCESSFUL)
    paid_mask = infos.get("_paid")
    paid = int(infos["paid"][0]) if paid_mask is not None and paid_mask[0] else solved
    assert (paid, solved) == (1, 0)


def test_a_flushed_episode_reaches_the_csv_as_paid_and_unsolved(
    tmp_path: pathlib.Path,
) -> None:
    """`paid` and `solved` reach the episode CSV as written."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            FORCE_SOLVED,
            "--directory", str(tmp_path),
            "--env-id", DELAYED_STAIRCASE,
            "--reward-delay", str(HORIZON * 5),
            "--max-episode-steps", str(HORIZON),
            "--budget", str(CSV_RUN_BUDGET),
            "--num-envs", str(CSV_RUN_ENVS),
            "--num-steps", str(CSV_RUN_STEPS),
            "--condition", "action",
            "--seed", str(CSV_RUN_SEED),
            "--no-cuda",
            "--no-log-trace",
        ],
        cwd=str(HERE),
        check=True,
        capture_output=True,
    )

    with open(tmp_path / f"episodes_seed{CSV_RUN_SEED}.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, "the run finished no episode, so no row was written to assert on"
    assert {(row["paid"], row["solved"]) for row in rows} == {("1", "0")}, (
        "a flushed hold is paid the bank and still reports no goal"
    )
    assert all(
        int(row["end_status"]) == NetHackStaircase.StepStatus.ABORTED for row in rows
    ), "the horizon has to be what ended these episodes, not the countdown"


def test_an_undelayed_env_reports_no_payout_flag() -> None:
    """The exp1 env never sets `paid`."""
    env = gym.make(
        "NetHackChallenge-v0", observation_keys=OBSERVATION_KEYS, max_episode_steps=HORIZON
    )
    _, info = env.reset(seed=0)
    env.close()

    assert "paid" not in info


def test_each_delayed_id_mixes_the_hold_over_its_own_predicate() -> None:
    """`HeldGoal` wraps each host's `_is_episode_end`."""
    hosts = {
        "DelayedStaircase-v0": NetHackStaircase,
        "DelayedStaircasePet-v0": NetHackStaircasePet,
        "DelayedOracle-v0": NetHackOracle,
    }
    assert set(DELAYED_ENVS) == set(hosts) | {"DelayedChallenge-v0"}
    for env_id, task in hosts.items():
        mro = DELAYED_ENVS[env_id].__mro__
        assert mro.index(HeldGoal) < mro.index(task)


def test_the_delayed_envs_enumerate_the_exp1_catalogue() -> None:
    """Delayed envs enumerate the same catalogue as exp1."""
    challenge = gym.make(
        "NetHackChallenge-v0", observation_keys=OBSERVATION_KEYS, max_episode_steps=HORIZON
    )
    expected = catalogue_digest(_catalogue(challenge.unwrapped.actions))
    challenge.close()

    for env_id in DELAYED_ENVS:
        env = DELAYED_ENVS[env_id](
            observation_keys=OBSERVATION_KEYS, max_episode_steps=HORIZON
        )
        digest = catalogue_digest(_catalogue(env.unwrapped.actions))
        env.close()
        assert digest == expected, f"{env_id} enumerates a different catalogue"


def test_a_delay_on_an_env_without_the_mechanism_is_rejected() -> None:
    """`--reward-delay` on the exp1 env is rejected."""
    with pytest.raises(AssertionError, match="no delay mechanism"):
        make_env(
            env_id="NetHackChallenge-v0",
            seed=0,
            idx=0,
            condition="action",
            gamma=0.999,
            max_episode_steps=HORIZON,
            clip_reward=True,
            n_options=64,
            option_family="grammar",
            option_seed=0,
            reward_delay=8,
        )


def test_the_delay_reaches_the_env_through_make_env() -> None:
    """`make_env` builds the delayed class and sets the delay."""
    reward_delay = 12
    env = make_env(
        env_id=DELAYED_STAIRCASE,
        seed=0,
        idx=0,
        condition="action",
        gamma=0.999,
        max_episode_steps=HORIZON,
        clip_reward=True,
        n_options=64,
        option_family="grammar",
        option_seed=0,
        reward_delay=reward_delay,
    )()
    inner = env.unwrapped
    env.close()

    assert isinstance(inner, HeldGoal)
    assert inner._reward_delay == reward_delay
    assert inner._max_episode_steps == HORIZON


STREAM_DELAYS = (0, 100, 300, 1000)
"""The exp4-pilot ladder."""

STREAM_EPISODE_STEPS = 1200
"""Longer than the longest delay, so some items dequeue before the flush and some do not."""

STREAM_REWARDS = (
    (11, 6.0),
    (148, -4.0),
    (300, -13.0),
    (500, 0.5),
    (700, -0.25),
    (950, 3.0),
    (1110, 28.0),
    (1150, 7.5),
)
"""`(step, reward)`, both signs and either side of `REWARD_CLIP`.

Placed so every delay's flush window holds more than one clippable item. Where a
window holds one, clipping the sum and clipping each item agree and the bug is
silent: an evenly spaced series only fails at delay 1000.
"""

STREAM_SENTINEL = 0.125
"""An in-clip value that is not the flush sum, so a re-read would be visible."""

TERMINAL_STATUSES = ("DEATH", "ABORTED")


def reward_series() -> List[float]:
    """One episode of scripted per-step rewards, zero except at the `STREAM_REWARDS` steps."""
    series = [0.0] * STREAM_EPISODE_STEPS
    for step, value in STREAM_REWARDS:
        series[step] = value
    return series


class RewardSeriesHost(gym.Env):
    """NLE-shaped host that pays a scripted reward per step and ends on the last one."""

    StepStatus = NLE.StepStatus
    action_space = gym.spaces.Discrete(1)
    observation_space = gym.spaces.Discrete(1)

    def __init__(self, series: List[float], terminal_status: int) -> None:
        self._series = series
        self._terminal_status = terminal_status
        self._index = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Rewind to the first scripted reward."""
        del seed, options
        self._index = 0
        return 0, {}

    def _reward_fn(
        self,
        last_observation: Any,
        action: int,
        observation: Any,
        end_status: int,
    ) -> float:
        """The scripted reward for the step being taken."""
        del last_observation, action, observation, end_status
        return self._series[self._index]

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """One step, calling `_reward_fn` with the step's `end_status` as `NLE.step` does."""
        last = self._index == len(self._series) - 1
        end_status = self._terminal_status if last else self.StepStatus.RUNNING
        reward = float(self._reward_fn(0, action, 0, end_status))
        self._index += 1
        return 0, reward, last, False, {"end_status": int(end_status)}


class StreamSeriesHost(HeldGoal, RewardSeriesHost):
    """`HeldGoal` in stream mode over a scripted series, so no NetHack is booted."""

    _stream = True


def stream_stack(reward_delay: int, terminal_status: int) -> gym.Env:
    """`make_env`'s wrap order for the stream host, over a scripted series."""
    env: gym.Env = StreamSeriesHost(
        reward_series(), terminal_status, reward_delay=reward_delay
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipReward(env, -REWARD_CLIP, REWARD_CLIP)
    return StreamFlushClip(env)


def undelayed_stack(terminal_status: int) -> gym.Env:
    """The same series through a host with no delay mechanism, wrapped as exp1 wraps it."""
    env: gym.Env = RewardSeriesHost(reward_series(), terminal_status)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return gym.wrappers.ClipReward(env, -REWARD_CLIP, REWARD_CLIP)


def episode_totals(env: gym.Env) -> Tuple[float, float]:
    """`(clipped, unclipped)` episode totals, the second off `RecordEpisodeStatistics`."""
    env.reset()
    clipped = 0.0
    info: Dict[str, Any] = {}
    for _ in range(STREAM_EPISODE_STEPS):
        _, reward, terminated, truncated, info = env.step(0)
        clipped += float(reward)
        if terminated or truncated:
            break
    assert "episode" in info, "the scripted series has to end the episode"
    return clipped, float(info["episode"]["r"])


@pytest.mark.parametrize("terminal_status", TERMINAL_STATUSES)
def test_the_clipped_total_is_the_same_at_every_delay(terminal_status: str) -> None:
    """Clipping each banked item makes the learner's total delay-invariant."""
    status = getattr(NLE.StepStatus, terminal_status)
    totals = [episode_totals(stream_stack(delay, status))[0] for delay in STREAM_DELAYS]
    per_item = sum(
        min(max(value, -REWARD_CLIP), REWARD_CLIP) for _, value in STREAM_REWARDS
    )

    assert totals == pytest.approx([per_item] * len(STREAM_DELAYS))


@pytest.mark.parametrize("terminal_status", TERMINAL_STATUSES)
def test_the_unclipped_total_is_the_same_at_every_delay(terminal_status: str) -> None:
    """The delay moves reward mass through time without creating or destroying any."""
    status = getattr(NLE.StepStatus, terminal_status)
    totals = [episode_totals(stream_stack(delay, status))[1] for delay in STREAM_DELAYS]
    unclipped = sum(value for _, value in STREAM_REWARDS)

    assert totals == pytest.approx([unclipped] * len(STREAM_DELAYS))


@pytest.mark.parametrize("terminal_status", TERMINAL_STATUSES)
def test_delay_zero_is_the_undelayed_host_exactly(terminal_status: str) -> None:
    """`reward_delay=0` is a passthrough, clipped and unclipped."""
    status = getattr(NLE.StepStatus, terminal_status)

    assert episode_totals(stream_stack(0, status)) == pytest.approx(
        episode_totals(undelayed_stack(status))
    )


@pytest.mark.parametrize("terminal_status", TERMINAL_STATUSES)
def test_a_flush_cannot_be_consumed_twice(terminal_status: str) -> None:
    """The handoff dies on the read that spends it, so no later read re-pays the flush."""
    status = getattr(NLE.StepStatus, terminal_status)
    env = stream_stack(STREAM_DELAYS[-1], status)
    env.reset()
    for _ in range(STREAM_EPISODE_STEPS):
        _, terminal_reward, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break

    assert terminal_reward > REWARD_CLIP, (
        "the flush has to outgrow a single clip or this test passes vacuously"
    )
    assert env.unwrapped._stream_flush is None
    assert env.reward(STREAM_SENTINEL) == pytest.approx(STREAM_SENTINEL)


def wrapper_stack(env: gym.Env) -> List[type]:
    """The wrapper classes around `env`, outermost first."""
    stack: List[type] = []
    while isinstance(env, gym.Wrapper):
        stack.append(type(env))
        env = env.env
    return stack


def test_only_the_stream_host_gets_the_flush_clip() -> None:
    """`StreamFlushClip` wraps the stream host outside the per-step clip, and no latch host."""
    common = dict(
        seed=0,
        idx=0,
        condition="action",
        gamma=0.999,
        max_episode_steps=HORIZON,
        clip_reward=True,
        n_options=64,
        option_family="grammar",
        option_seed=0,
        reward_delay=8,
    )
    stream = make_env(env_id="DelayedChallenge-v0", **common)()
    stream_wrappers = wrapper_stack(stream)
    stream.close()

    latch = make_env(env_id=DELAYED_STAIRCASE, **common)()
    latch_wrappers = wrapper_stack(latch)
    latch.close()

    assert stream_wrappers.index(StreamFlushClip) < stream_wrappers.index(
        gym.wrappers.ClipReward
    ), "the flush clip has to sit outside the per-step clip it corrects"
    assert StreamFlushClip not in latch_wrappers
