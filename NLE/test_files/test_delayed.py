"""Tests for the exp4 reward delay.

The predicate and the reward function are called against a hand-constructed
observation. One test rolls out, to pin where NLE's internal horizon reports itself.
"""

from typing import TYPE_CHECKING, Any, Tuple

import gymnasium as gym
import numpy as np
import pytest
from nle.env.tasks import NetHackOracle, NetHackStaircase, NetHackStaircasePet

from delayed import DELAYED_ENVS, HeldGoal
from envs import OBSERVATION_KEYS, TASK_SUCCESSFUL, make_env
from options import _catalogue, catalogue_digest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

DELAYED_STAIRCASE = "DelayedStaircase-v0"
HORIZON = 40
"""Long enough for NLE start-up menus, short enough to reach."""

STAIRS_DOWN = 4
"""Index of the stairs_down flag in `internal`. What `NetHackStaircase._is_episode_end` reads."""

FROZEN_STEP_PENALTY = -0.01
"""`NetHackScore.penalty_step`. These tests reuse one observation, so time never advances."""

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
    """The task ends `reward_delay` primitive steps after it was first solved. Delay 0 is NLE's own task."""
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
    assert banked == pytest.approx(FROZEN_STEP_PENALTY)


def test_nothing_is_paid_during_the_hold() -> None:
    """Held steps carry the time penalty only. The goal reward arrives once, or return would scale with delay."""
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
    assert held == pytest.approx([FROZEN_STEP_PENALTY] * reward_delay)
    assert paid == pytest.approx(FROZEN_STEP_PENALTY + GOAL_REWARD)


@pytest.mark.parametrize("end_status", ["ABORTED", "DEATH"])
def test_a_terminal_step_inside_the_hold_still_pays(end_status: str) -> None:
    """Horizon and death inside the hold both flush the bank. Without that, a long delay silently cuts the return."""
    env = delayed_env(reward_delay=1000)
    observation = solved_observation(env)
    env._is_episode_end(observation)
    paid = env._reward_fn(
        observation, 0, observation, getattr(env.StepStatus, end_status)
    )
    env.close()

    assert paid == pytest.approx(FROZEN_STEP_PENALTY + GOAL_REWARD)


def test_reset_clears_the_latch() -> None:
    """The next episode does not inherit the last one's hold or payout flag."""
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
    assert unpaid == pytest.approx(FROZEN_STEP_PENALTY)


def test_the_internal_horizon_terminates_rather_than_truncates() -> None:
    """NLE's own step limit arrives as `terminated` with end_status ABORTED. `terminated` is also death and the goal."""
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
    """Goal predicate that fires every step, so the latch is reachable without walking to a staircase."""
    del observation
    return self.StepStatus.TASK_SUCCESSFUL


def test_a_horizon_flush_reports_paid_without_reporting_success(
    monkeypatch: "MonkeyPatch",
) -> None:
    """A hold outliving the horizon pays the bank and still ends ABORTED. Through `env.step`: `_reward_fn` never fills info."""
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
    """The trainer writes `paid=1`, `solved=0` for a flush. Through the vector env, so `_paid` is `_add_info`'s mask."""
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
    # the assignment ppo.py makes for the row it writes
    solved = int(int(infos["end_status"][0]) == TASK_SUCCESSFUL)
    paid_mask = infos.get("_paid")
    paid = int(infos["paid"][0]) if paid_mask is not None and paid_mask[0] else solved
    assert (paid, solved) == (1, 0)


def test_an_undelayed_env_reports_no_payout_flag() -> None:
    """The exp1 env never sets `paid`, so the trainer's fallback to `solved` is live."""
    env = gym.make(
        "NetHackChallenge-v0", observation_keys=OBSERVATION_KEYS, max_episode_steps=HORIZON
    )
    _, info = env.reset(seed=0)
    env.close()

    assert "paid" not in info


def test_each_delayed_id_mixes_the_hold_over_its_own_predicate() -> None:
    """One mixin, three hosts. `HeldGoal` wraps `_is_episode_end` rather than replacing the task."""
    hosts = {
        "DelayedStaircase-v0": NetHackStaircase,
        "DelayedStaircasePet-v0": NetHackStaircasePet,
        "DelayedOracle-v0": NetHackOracle,
    }
    assert set(DELAYED_ENVS) == set(hosts)
    for env_id, task in hosts.items():
        mro = DELAYED_ENVS[env_id].__mro__
        assert mro.index(HeldGoal) < mro.index(task)


def test_the_delayed_envs_enumerate_the_exp1_catalogue() -> None:
    """exp4's `grammar` is the same catalogue as exp1. Goal tasks default to `TASK_ACTIONS` (49 rows, no DIR)."""
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
    """`--reward-delay` on the exp1 env is a mistake, not a no-op."""
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
    """`make_env` builds the delayed class directly. `gym.make` would add a TimeLimit and leave NLE's horizon at 5000."""
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
