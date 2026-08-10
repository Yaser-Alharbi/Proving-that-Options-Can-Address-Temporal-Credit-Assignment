import gymnasium as gym
import nle  # noqa: F401  registers NetHack envs with gymnasium
import numpy as np
from nle import nethack

from options import EpisodeAccounting, OptionWrapper, make_options

GLYPH_SHAPE = nethack.DUNGEON_SHAPE
GLYPH_SIZE = int(np.prod(GLYPH_SHAPE))
NUM_GLYPHS = nethack.MAX_GLYPH + 1


class NLEObsWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        stat_size = int(np.prod(env.observation_space["blstats"].shape))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (GLYPH_SIZE + stat_size,), np.float32)

    def observation(self, obs):
        glyphs = obs["glyphs"].astype(np.float32).ravel()
        blstats = obs["blstats"].astype(np.float32)
        return np.concatenate([glyphs, blstats])


def make_env(env_id, seed, idx, use_options):
    def thunk():
        env = gym.make(env_id)
        if use_options:
            options, _ = make_options(env.unwrapped.actions)
            env = OptionWrapper(env, options)
        env = NLEObsWrapper(env)
        env = EpisodeAccounting(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        return env

    return thunk