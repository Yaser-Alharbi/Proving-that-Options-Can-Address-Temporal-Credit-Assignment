import gymnasium as gym
import nle  # noqa: F401  registers NetHack envs with gymnasium
import numpy as np
from nle import nethack

from options import OptionWrapper, make_options

MAP_H, MAP_W = 21, 79
CROP = 9
N_STATS = 27

MAP_SIZE = MAP_H * MAP_W
CROP_SIZE = CROP * CROP
OBS_SIZE = MAP_SIZE + CROP_SIZE + N_STATS


class NLEObsWrapper(gym.ObservationWrapper):
    """Flatten NLE's dict observation into full map | agent-centred crop | stats"""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            0.0, float(nethack.MAX_GLYPH), (OBS_SIZE,), np.float32
        )
        self.pad = CROP // 2

    def observation(self, obs):
        glyphs = obs["glyphs"]
        blstats = obs["blstats"]

        x, y = int(blstats[0]), int(blstats[1])
        padded = np.pad(glyphs, self.pad, mode="constant")
        crop = padded[y : y + CROP, x : x + CROP]

        return np.concatenate([
            glyphs.ravel().astype(np.float32),
            crop.ravel().astype(np.float32),
            blstats.astype(np.float32),
        ])


def make_env(env_id, seed, idx, use_options):
    def thunk():
        env = gym.make(env_id)
        if use_options:
            options, _ = make_options(env.unwrapped.actions)
            env = OptionWrapper(env, options)
        env = NLEObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        return env

    return thunk