from typing import Callable, Literal, SupportsFloat

import gymnasium as gym
import nle  # noqa: F401  registers NetHack envs with gymnasium
import numpy as np
from nle.env.base import NLE
from nle.env.tasks import NetHackStaircase

from delayed import DELAYED_ENVS
from options import OptionWrapper, make_options

OBSERVATION_KEYS = ("glyphs", "blstats", "inv_letters", "misc", "message")
"""Wider than the encoder: map/`inv_letters` for `I`, `misc` for the drain."""

REWARD_CLIP = 1.0

TASK_SUCCESSFUL = int(NetHackStaircase.StepStatus.TASK_SUCCESSFUL)
"""`end_status` for goal fired (`solved`). Off the task class: base NLE has no such member."""

STATUS_RUNNING = int(NLE.StepStatus.RUNNING)
"""`end_status` when the game itself did not end. Gymnasium TimeLimit truncations write no xlogfile."""


class StreamFlushClip(gym.RewardWrapper):
    """Rewrites a stream terminal flush as the sum of its individually clipped items.

    A flush is many primitive steps arriving as one scalar, so `ClipReward` below
    truncates the whole episode's banked reward to `REWARD_CLIP`. Clipping each
    banked item instead makes the clipped total delay-invariant.
    """

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """The wrapped reward, unless the inner env just flushed its pending queue."""
        flushed = self.unwrapped._stream_flush
        if flushed is None:
            return reward
        self.unwrapped._stream_flush = None
        return float(np.clip(flushed, -REWARD_CLIP, REWARD_CLIP).sum())


def make_nle(env_id: str, max_episode_steps: int, reward_delay: int = 0) -> gym.Env:
    """Bare env for `env_id`. Delayed envs are constructed, not `gym.make` (that would add another TimeLimit)."""
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
    """Thunk for one NLE env that satisfies the trainer's contract."""
    assert reward_delay == 0 or env_id in DELAYED_ENVS, (
        f"{env_id} has no delay mechanism, so reward_delay={reward_delay} would "
        f"be silently ignored; the delayed envs are {sorted(DELAYED_ENVS)}"
    )

    def thunk() -> gym.Env:
        env = make_nle(env_id, max_episode_steps, reward_delay)
        # innermost: `l` is primitive steps, `r` is raw undiscounted reward
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if clip_reward:
            # below OptionWrapper so the clip is per primitive, not per decision
            env = gym.wrappers.ClipReward(env, -REWARD_CLIP, REWARD_CLIP)
            if env_id == "DelayedChallenge-v0":
                env = StreamFlushClip(env)
        rows, _, _ = make_options(
            env.unwrapped.actions, condition, n_options, option_family, option_seed
        )
        env = OptionWrapper(env, rows, gamma=gamma)
        env.reset(seed=seed + idx)
        return env

    return thunk
