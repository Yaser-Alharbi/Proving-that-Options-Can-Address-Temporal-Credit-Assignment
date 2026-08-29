"""Tests for the env-level facts the trainer's accounting rests on.

Neither test touches the network. They pin where a NEXT_STEP phantom lands and
how wide the glyph embedding has to be. Both are silent if wrong.
"""

from typing import Callable, List, Tuple

import gymnasium as gym
import numpy as np
import pytest
from nle import nethack

from envs import make_env

NUM_ENVS = 2
TRUNCATE_AT = 20
CONSTANT_ACTION = 0
FULL_CATALOGUE = 187
"""Every row `NetHackChallenge-v0` admits, so no test here is also a subsample."""


def env_thunk(idx: int, condition: str = "action") -> Callable[[], gym.Env]:
    """One env of the vector, truncating at `TRUNCATE_AT`."""
    return make_env(
        env_id="NetHackChallenge-v0",
        seed=0,
        idx=idx,
        condition=condition,
        gamma=0.999,
        max_episode_steps=TRUNCATE_AT,
        clip_reward=True,
        n_options=FULL_CATALOGUE,
        option_family="grammar",
        option_seed=0,
        reward_delay=0,
    )


@pytest.fixture(scope="module")
def truncating_rollout() -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
    """One step past truncation. Truncation rather than death, so the ending step is exact."""
    envs = gym.vector.SyncVectorEnv(
        [env_thunk(index) for index in range(NUM_ENVS)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    envs.reset(seed=0)
    actions = np.full(NUM_ENVS, CONSTANT_ACTION, dtype=np.int64)
    steps = []
    for _ in range(TRUNCATE_AT + 1):
        _, reward, terminated, truncated, infos = envs.step(actions)
        steps.append((reward.copy(), terminated.copy(), truncated.copy(), infos))
    envs.close()
    return steps


def test_next_step_autoreset_emits_one_flagless_transition(
    truncating_rollout: List[Tuple[np.ndarray, np.ndarray, np.ndarray, dict]],
) -> None:
    """The step after an episode ends carries no flags, no reward and no steps. `ppo.py` finds it as `dones[step]`."""
    reward, terminated, truncated, infos = truncating_rollout[TRUNCATE_AT - 1]
    assert truncated.all(), "every env should truncate on its max_episode_steps-th step"
    assert not terminated.any(), "truncation, not termination, is the mechanism under test"
    assert (infos["primitive_steps"] == 1 + infos["drain_steps"]).all(), (
        "the terminating decision consumed a primitive step and must be paid for, "
        "plus any modal state it inherited and drained"
    )

    reward, terminated, truncated, infos = truncating_rollout[TRUNCATE_AT]
    assert not terminated.any() and not truncated.any(), (
        "the autoreset transition belongs to neither episode and must carry no flags"
    )
    assert (reward == 0.0).all(), "the autoreset transition must carry no reward"
    assert (infos["primitive_steps"] == 0).all(), (
        "the autoreset transition consumed no primitive steps, so it must cost the "
        "budget nothing; a 1 here would let the option condition buy extra frames"
    )
    assert infos["_available"].all(), (
        "`available` must survive the autoreset, or the first decision of every "
        "episode is taken under a mask of zeros"
    )


def test_observation_space_matches_encoder_assumptions() -> None:
    """`glyphs` admits `MAX_GLYPH`, so the embedding needs `MAX_GLYPH + 1` rows. Box high is inclusive."""
    env = env_thunk(0)()
    glyphs = env.observation_space["glyphs"]
    env.close()

    assert glyphs.high.max() == nethack.MAX_GLYPH, (
        f"expected the glyph bound to be MAX_GLYPH={nethack.MAX_GLYPH}"
    )
    top = np.full(glyphs.shape, nethack.MAX_GLYPH, dtype=glyphs.dtype)
    assert glyphs.contains(top), (
        "the bound is inclusive, so MAX_GLYPH is a legal index and the embedding "
        "must have MAX_GLYPH + 1 rows"
    )


def test_reset_emits_zero_primitive_steps_and_a_mask() -> None:
    """Reset reports steps=0 and a full mask. The trainer asserts this."""
    env = env_thunk(0)()
    _, info = env.reset(seed=0)
    n_actions = env.action_space.n
    env.close()
    assert info["primitive_steps"] == 0
    assert info["decision_t"] == 0
    assert info["available"].all()
    assert info["available"].shape == (n_actions,)
