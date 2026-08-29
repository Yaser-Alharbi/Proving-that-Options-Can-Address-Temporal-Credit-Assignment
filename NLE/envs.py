from typing import Callable, Literal

import gymnasium as gym
import nle  # noqa: F401  registers NetHack envs with gymnasium
from nle.env.base import NLE
from nle.env.tasks import NetHackStaircase

from delayed import DELAYED_ENVS
from options import OptionWrapper, make_options

OBSERVATION_KEYS = ("glyphs", "blstats", "inv_letters", "misc", "message")
"""Wider than the encoder: map/`inv_letters` for `I`, `misc` for the drain."""

REWARD_CLIP = 1.0

TASK_SUCCESSFUL = int(NetHackStaircase.StepStatus.TASK_SUCCESSFUL)
"""`info["end_status"]` value meaning the goal fired, which is what the `solved`
column records. Read off the task class rather than written as a literal: base
NLE's `StepStatus` has no such member, so the number only exists on the goal
tasks, and the delayed envs inherit this enum."""

STATUS_RUNNING = int(NLE.StepStatus.RUNNING)
"""`end_status` on an episode the game itself did not end. `gymnasium.make`
takes `max_episode_steps` for a `TimeLimit` of its own, so a truncated
`NetHackChallenge` episode arrives with the game still in progress and NetHack
has written no xlogfile record for it. The episode log reads the character off
that record, so it must know the difference."""


def make_nle(env_id: str, max_episode_steps: int, reward_delay: int = 0) -> gym.Env:
    """The bare env for `env_id`, before any wrapper.

    The delayed envs are constructed rather than made, because `gymnasium.make`
    takes `max_episode_steps` for a `TimeLimit` of its own; see
    `delayed.DELAYED_ENVS`. One function so that anything needing an env for an
    `env_id` — the trainer, main.py's catalogue preflight — agrees on which
    class and which horizon that is.
    """
    if env_id in DELAYED_ENVS:
        return DELAYED_ENVS[env_id](
            observation_keys=OBSERVATION_KEYS,
            max_episode_steps=max_episode_steps,
            reward_delay=reward_delay,
        )
    return gym.make(
        env_id,
        observation_keys=OBSERVATION_KEYS,
        max_episode_steps=max_episode_steps,
    )


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    condition: Literal["action", "option", "both"],
    gamma: float,
    max_episode_steps: int,
    clip_reward: bool,
    n_options: int,
    option_family: Literal["grammar", "grammar_depth", "random"],
    option_seed: int,
    reward_delay: int,
) -> Callable[[], gym.Env]:
    """Return a thunk building one NLE env that satisfies the trainer's contract."""
    assert reward_delay == 0 or env_id in DELAYED_ENVS, (
        f"{env_id} has no delay mechanism, so reward_delay={reward_delay} would "
        f"be silently ignored; the delayed envs are {sorted(DELAYED_ENVS)}"
    )

    def thunk() -> gym.Env:
        env = make_nle(env_id, max_episode_steps, reward_delay)
        # innermost, so `l` counts primitive steps and `r` sums raw undiscounted
        # reward. Above the clip or the option wrapper it would instead report
        # decisions and a discounted sum, which are different quantities in the
        # option condition than in the action condition and so not comparable.
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if clip_reward:
            # below OptionWrapper, so the clip applies per primitive step rather
            # than to an option's discounted sum. Clipping the decision reward
            # would be a different transformation for a 16-step option than for
            # a primitive.
            env = gym.wrappers.ClipReward(env, -REWARD_CLIP, REWARD_CLIP)
        rows, _, _ = make_options(
            env.unwrapped.actions, condition, n_options, option_family, option_seed
        )
        env = OptionWrapper(env, rows, gamma=gamma)
        env.reset(seed=seed + idx)
        return env

    return thunk
