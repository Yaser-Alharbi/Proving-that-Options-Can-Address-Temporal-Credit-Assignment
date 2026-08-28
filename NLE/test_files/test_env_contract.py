"""Tests for the env-level facts the trainer's accounting rests on.

`cd src && python -m pytest test_files/test_env_contract.py -q`.

Neither test touches the network or the loss. They pin two properties of the
environment that `ppo.py` cannot check for itself at run time: where a NEXT_STEP
autoreset puts its phantom transition, and how wide the glyph embedding has to
be. Both are silent if wrong.
"""

from typing import Callable, List, Tuple

import gymnasium as gym
import numpy as np
import pytest
from nle import nethack

from envs import OBSERVATION_KEYS, make_env
from options import OptionWrapper, make_options

NUM_ENVS = 2
TRUNCATE_AT = 20
CONSTANT_ACTION = 0
INVENTORY_LENGTH = 55
FULL_CATALOGUE = 188
"""Every row `NetHackChallenge-v0`'s action set admits, so no test here is
subsampling the catalogue while checking something else."""


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
    """One step past truncation, from envs that truncate at a known step count.

    Truncation rather than death, so the step at which the episode ends is exact
    rather than a property of how long a random character survives.
    """
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
    """The step after an episode ends carries no flags, no reward and no steps.

    This is the transition `ppo.py` excludes from every reduction, and it locates
    it as `dones[step]` rather than with a buffer of its own. That identification
    is only valid if the step following a done is flagless, so that no genuine
    transition is ever masked out with it.
    """
    reward, terminated, truncated, infos = truncating_rollout[TRUNCATE_AT - 1]
    assert truncated.all(), "every env should truncate on its max_episode_steps-th step"
    assert not terminated.any(), "truncation, not termination, is the mechanism under test"
    assert (infos["primitive_steps"] == 1).all(), (
        "the terminating decision consumed a primitive step and must be paid for"
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
    """`glyphs` admits `MAX_GLYPH`, so the embedding needs `MAX_GLYPH + 1` rows.

    Upstream's NetHackNet sizes the table `MAX_GLYPH` and indexes out of range on
    the top glyph. This asserts the premise of that fix rather than the fix, so it
    fails if an NLE version ever makes the bound exclusive.
    """
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
    """The contract the trainer asserts on: reset reports steps=0 and a full mask."""
    env = env_thunk(0)()
    _, info = env.reset(seed=0)
    n_actions = env.action_space.n
    env.close()
    assert info["primitive_steps"] == 0
    assert info["decision_t"] == 0
    assert info["available"].all()
    assert info["available"].shape == (n_actions,)


def test_initiation_excludes_rows_naming_an_empty_inventory_slot() -> None:
    """A row that names an inventory letter is offered exactly when that slot is full.

    Hand-constructed inventories rather than a rollout: the property is a function
    of `inv_letters` alone, and reaching an empty pack by playing is neither quick
    nor deterministic. Calls the predicate directly for the same reason.
    """
    env = gym.make(
        "NetHackChallenge-v0",
        observation_keys=OBSERVATION_KEYS,
        max_episode_steps=TRUNCATE_AT,
    )
    sequences, names, required_slots = make_options(
        env.unwrapped.actions, "option", FULL_CATALOGUE, "grammar", 0
    )
    wrapper = OptionWrapper(env, sequences, required_slots, gamma=0.999)

    empty_pack = np.zeros(INVENTORY_LENGTH, dtype=np.uint8)
    only_slot_a = empty_pack.copy()
    only_slot_a[0] = ord("a")

    with_empty_pack = wrapper._initiation({"inv_letters": empty_pack})
    with_slot_a = wrapper._initiation({"inv_letters": only_slot_a})
    env.close()

    gated = np.array([slot is not None for slot in required_slots])
    assert gated.any() and not gated.all(), (
        "the table must contain both gated and ungated rows or the test is vacuous"
    )
    assert not with_empty_pack[gated].any(), (
        "every row naming a letter should be excluded when the pack is empty"
    )
    assert with_empty_pack[~gated].all(), (
        "rows with no precondition this wrapper can decide stay selectable"
    )
    assert with_slot_a[names.index("eat_a")], "slot a is occupied, so eat_a is offered"
    assert not with_slot_a[names.index("eat_b")], "slot b is empty, so eat_b is not"
    assert np.array_equal(with_slot_a[~gated], with_empty_pack[~gated]), (
        "inventory contents must not move the ungated rows"
    )
