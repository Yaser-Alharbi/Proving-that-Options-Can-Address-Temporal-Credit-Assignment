from typing import Callable, Literal

import gymnasium as gym
import nle  # noqa: F401  registers NetHack envs with gymnasium

from options import OptionWrapper, make_options

OBSERVATION_KEYS = ("glyphs", "blstats", "inv_letters")
"""Wider than the encoder consumes: `inv_letters` decides the initiation sets."""

REWARD_CLIP = 1.0


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    condition: Literal["action", "option", "both"],
    gamma: float,
    max_episode_steps: int,
    clip_reward: bool,
) -> Callable[[], gym.Env]:
    """Return a thunk building one NLE env that satisfies the trainer's contract."""

    def thunk() -> gym.Env:
        env = gym.make(
            env_id,
            observation_keys=OBSERVATION_KEYS,
            max_episode_steps=max_episode_steps,
        )
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
        sequences, _, required_slots = make_options(env.unwrapped.actions, condition)
        env = OptionWrapper(env, sequences, required_slots, gamma=gamma)
        env.reset(seed=seed + idx)
        return env

    return thunk
