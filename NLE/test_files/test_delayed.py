"""Tests for the exp4 reward delay.

`cd src && python -m pytest test_files/test_delayed.py -q`.

The predicate and the reward function are called directly against a
hand-constructed observation. Reaching a staircase by playing is neither quick
nor deterministic, and the properties under test are functions of the latch
alone, so a rollout would only add a source of failure that is not the mechanism.

One test does roll out, to pin the env fact the rest depends on: where NLE's
internal horizon reports itself.
"""

from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import pytest
from nle.env.tasks import NetHackOracle, NetHackStaircase, NetHackStaircasePet

from delayed import DELAYED_ENVS, HeldGoal
from envs import OBSERVATION_KEYS, make_env
from options import _catalogue, catalogue_digest

DELAYED_STAIRCASE = "DelayedStaircase-v0"
HORIZON = 40
"""Long enough for NLE to finish its start-up menus, short enough to reach."""

STAIRS_DOWN = 4
"""Index of the stairs_down flag in NLE's `internal` observation, which is what
`NetHackStaircase._is_episode_end` reads."""

FROZEN_STEP_PENALTY = -0.01
"""`NetHackScore.penalty_step`, paid whenever game time did not advance. Every
call below reuses one observation, so time never advances and the penalty is this
on every step."""

GOAL_REWARD = 1.0


def delayed_env(reward_delay: int, horizon: int = HORIZON) -> Any:
    """One reset `DelayedStaircase`, its latch clear."""
    env = DELAYED_ENVS[DELAYED_STAIRCASE](
        observation_keys=OBSERVATION_KEYS,
        max_episode_steps=horizon,
        reward_delay=reward_delay,
    )
    env.reset(seed=0)
    return env


def solved_observation(env: Any) -> Tuple[np.ndarray, ...]:
    """The live observation with the stairs_down flag set."""
    observation = tuple(array.copy() for array in env.last_observation)
    observation[env._internal_index][STAIRS_DOWN] = 1
    return observation


def statuses_until_terminal(env: Any, observation: Any, limit: int) -> list:
    """`_is_episode_end` repeatedly, stopping at the first non-RUNNING answer."""
    statuses = []
    for _ in range(limit):
        status = env._is_episode_end(observation)
        statuses.append(status)
        if status != env.StepStatus.RUNNING:
            break
    return statuses


@pytest.mark.parametrize("reward_delay", [0, 1, 4, 16])
def test_the_goal_is_held_for_exactly_reward_delay_steps(reward_delay: int) -> None:
    """The task ends `reward_delay` primitive steps after it was first solved.

    Delay 0 is the unmodified task: the first solved observation is terminal, so
    exp4's baseline arm is the env NLE ships and not a variant of it.
    """
    env = delayed_env(reward_delay)
    statuses = statuses_until_terminal(env, solved_observation(env), reward_delay + 2)
    env.close()

    assert statuses[-1] == env.StepStatus.TASK_SUCCESSFUL
    assert all(status == env.StepStatus.RUNNING for status in statuses[:-1])
    assert len(statuses) - 1 == reward_delay


def test_an_unsolved_observation_never_latches() -> None:
    """A game that does not reach the goal keeps its own end status.

    The hold must not turn a non-solve into a solve, or every exp4 episode ends
    TASK_SUCCESSFUL and the `solved` column is constant.
    """
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
    """Held steps carry the time penalty only, and the goal reward arrives once.

    Navix drops the re-fired terminal reward on each held step for the same
    reason: paying per held step would make the return a function of the delay.
    """
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
    """The horizon and a death inside the hold both flush the bank.

    This is what makes the delay compression rather than deletion: the realised
    delay is `min(reward_delay, horizon - solve_step)` and the payout is fixed at
    1 when the goal fires, so `episodic_return` is delay-invariant. Without the
    flush, a long delay would silently reduce the return it is measured against.
    """
    env = delayed_env(reward_delay=1000)
    observation = solved_observation(env)
    env._is_episode_end(observation)
    paid = env._reward_fn(
        observation, 0, observation, getattr(env.StepStatus, end_status)
    )
    env.close()

    assert paid == pytest.approx(FROZEN_STEP_PENALTY + GOAL_REWARD)


def test_reset_clears_the_latch() -> None:
    """The next episode does not inherit the last one's hold."""
    env = delayed_env(reward_delay=8)
    env._is_episode_end(solved_observation(env))
    assert env._held is not None

    env.reset(seed=1)
    assert env._held is None
    unpaid = env._reward_fn(
        env.last_observation, 0, env.last_observation, env.StepStatus.ABORTED
    )
    env.close()
    assert unpaid == pytest.approx(FROZEN_STEP_PENALTY)


def test_the_internal_horizon_terminates_rather_than_truncates() -> None:
    """NLE's own step limit arrives as `terminated` with end_status ABORTED.

    The reason exp4 needs a `solved` column rather than reading `terminated`: on
    this env the flag is set by the horizon and by death as well as by the goal.
    It also pins the ordering the flush relies on, since `_reward_fn` only runs
    when the horizon is NLE's and not a `TimeLimit` above it.
    """
    env = delayed_env(reward_delay=8, horizon=HORIZON)
    for step in range(HORIZON):
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    env.close()

    assert step + 1 == HORIZON, "the horizon should be what ends this episode"
    assert terminated and not truncated
    assert info["end_status"] == NetHackStaircase.StepStatus.ABORTED


def test_each_delayed_id_mixes_the_hold_over_its_own_predicate() -> None:
    """One mixin, three hosts, so the choice of task env is a registration.

    The three tasks differ only in `_is_episode_end`, which is the method the
    hold wraps, so `HeldGoal` has to sit above each of them and not replace them.
    """
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
    """exp4's `grammar` is the same 64 options as exp1's and exp2's.

    The goal tasks default to `TASK_ACTIONS`, 23 keys, which enumerates to 49
    rows with no directional row at all, so n=64 would be unreachable and any
    smaller n would be a different catalogue under the same label. `HeldGoal`
    passes `nethack.ACTIONS` for that reason, and this is the assertion that
    stops the label meaning two things across the figures.
    """
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
    """`--reward-delay 8` on the exp1 env is a mistake, not a no-op.

    Silently ignoring it would produce an exp4-shaped sweep whose delay axis did
    nothing, which looks like a null result.
    """
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
    """`make_env` builds the delayed class directly, not through `gym.make`.

    `gym.make` would consume `max_episode_steps` for a `TimeLimit` of its own and
    leave NLE's internal horizon at 5000, so the flush would never fire and the
    delay ladder would be measured against the wrong horizon.
    """
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
