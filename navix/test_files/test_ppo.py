"""Tests for the PPO agent's use of the option environment.

`cd navix && JAX_PLATFORMS=cpu python -m pytest test_files/test_ppo.py -q`.

Both tests read one rollout of the real training wiring. Neither covers whether
`initiation` is the right gate at a state, which is `test_options.py`'s subject.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from config import Args
from ppo import PPO, Buffer, TrainingState
from train import build_cell, make_agent


@pytest.fixture(scope="session")
def rollout() -> Tuple[PPO, TrainingState, Buffer]:
    """An agent, the state after one collection, and the experience it collected."""
    args = Args(
        action_space="option",
        n_options=16,
        max_forward=3,
        num_envs=4,
        num_steps=4,
        num_minibatches=1,
    )
    agent = make_agent(args, build_cell(args))
    state, experience = agent.collect_experience(agent.init(jax.random.PRNGKey(0)))
    return agent, state, experience


def test_the_loss_recomputes_logits_under_the_stored_pre_step_mask(
    rollout: Tuple[PPO, TrainingState, Buffer],
) -> None:
    """With parameters held fixed, the importance ratio is 1 on every stored transition.

    `approx_kl` is the mean of `(r - 1) - log r`, which is non-negative and
    vanishes only at `r == 1`, so a zero mean is the per-transition property.
    """
    agent, state, experience = rollout
    post_step = experience.info["available"]
    assert bool(jnp.any(experience.available != post_step)), (
        "the two masks coincide here, so the test would be vacuous"
    )
    chosen_after_the_step = jnp.take_along_axis(
        post_step, experience.action[..., None], axis=-1
    )
    assert not bool(jnp.all(chosen_after_the_step)), (
        "every stored action survives into I(s_t+1), so the post-step mask would "
        "differ only by a normaliser"
    )
    assert bool(jnp.array_equal(experience.available[1:], post_step[:-1])), (
        "the stored mask is I at the state the action was taken in"
    )

    last_val = state.value_fn(state.params, state.env_state.observation)
    values, advantages, targets = agent.evaluate_experience(state, experience, last_val)

    def flatten(leaf: Array) -> Array:
        """(T, N, ...) to (T * N, ...), the batch shape the loss is called on."""
        return leaf.reshape((-1,) + leaf.shape[2:])

    _, logs = agent.ppo_loss(
        state.params,
        jax.tree.map(flatten, experience),
        flatten(advantages),
        flatten(targets),
        flatten(values),
    )
    # not exact equality: collection evaluates the encoder at num_envs and the
    # loss at num_envs * num_steps, so the logits need not agree bitwise
    assert float(logs["loss/approx_kl"]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("discount_mode", ["primitive", "decision"])
def test_gae_discounts_by_the_option_duration_only_under_primitive(
    rollout: Tuple[PPO, TrainingState, Buffer], discount_mode: str
) -> None:
    """The per-step discount is `gamma ** primitive_steps` under `primitive`, `gamma` under `decision`."""
    agent, state, experience = rollout
    primitive_steps = experience.info["primitive_steps"]
    assert int(jnp.max(primitive_steps)) > 1, (
        "no decision spans several primitive steps, so the two modes coincide"
    )

    agent = agent.replace(hparams=agent.hparams.replace(discount_mode=discount_mode))
    last_val = state.value_fn(state.params, state.env_state.observation)
    values, advantages, _ = agent.evaluate_experience(state, experience, last_val)

    gamma = agent.env.gamma
    per_step = (
        gamma**primitive_steps
        if discount_mode == "primitive"
        else jnp.full_like(experience.reward, gamma)
    )
    discount = (1 - experience.done) * per_step
    bootstrapped = jnp.concatenate([values, last_val[None]], axis=0)
    delta = experience.reward + discount * bootstrapped[1:] - values

    horizon = agent.hparams.num_steps
    expected = []
    for start in range(horizon):
        weight = jnp.ones_like(last_val)
        advantage = jnp.zeros_like(last_val)
        for lag in range(start, horizon):
            advantage = advantage + weight * delta[lag]
            weight = weight * agent.hparams.gae_lambda * discount[lag]
        expected.append(advantage)

    assert bool(jnp.allclose(advantages, jnp.stack(expected), atol=1e-6))
