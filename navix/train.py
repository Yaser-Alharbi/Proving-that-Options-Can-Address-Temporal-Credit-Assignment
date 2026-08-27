"""Per-cell training, as a library.

A cell is one point of the experiment matrix: an environment, a condition, an
option family and count, and a set of training seeds. `main.py` owns the
configuration, the results store and the reporting; nothing here writes to disk.
"""

from functools import partial
import time
from typing import Callable, Dict, List, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

import navix as nx
from navix import observations
from navix.agents.models import ActorCritic, ConvEncoder

from config import Args
from options import (
    OptionEnv,
    OptionSpec,
    action_names,
    action_table,
    make_options,
    mean_nominal_duration,
    measure_duration_stats,
    missing_interactions,
)
from ppo import PPO, PPOHparams

UPDATES_PER_CHUNK = 16
"""Updates between host syncs, traded off against the transfer cost."""

OnChunk = Callable[[Sequence[int], Dict[str, Array]], None]
"""Called after each chunk with the seeds its logs' leading axis indexes."""


class Cell(NamedTuple):
    """The environment and action table one cell trains against."""

    env: OptionEnv
    spec: OptionSpec
    table: List[Tuple[int, ...]]
    table_names: List[str]
    primitive_names: List[str]
    n_generated_options: int
    duration_stats: Dict[str, float]
    nominal_option_len: float
    missing_interactions: List[str]

    @property
    def mean_option_len(self) -> float:
        """Measured primitive steps per decision, which the budget divides by."""
        return self.duration_stats["mean"]


def build_cell(args: Args) -> Cell:
    """Construct the environment, the option table and its duration statistics."""
    env = nx.make(
        args.env_id,
        max_steps=args.max_steps,
        gamma=args.gamma,
        observation_fn=observations.symbolic,
    )
    names = action_names(env)

    options: List[Tuple[int, ...]] = []
    labels: List[str] = []
    if args.action_space != "action":
        options, labels = make_options(
            args.option_family,
            args.n_options,
            args.max_forward,
            names,
            option_seed=args.option_seed,
        )
    table, table_names = action_table(args.action_space, names, options, labels)

    spec = OptionSpec.create(table, names)
    wrapped = OptionEnv.create(
        env, spec, executor=args.executor, reward_delay=args.reward_delay
    )
    return Cell(
        env=wrapped,
        spec=spec,
        table=table,
        table_names=table_names,
        primitive_names=names,
        n_generated_options=len(options),
        duration_stats=measure_duration_stats(wrapped, seed=args.option_seed),
        nominal_option_len=mean_nominal_duration(table),
        missing_interactions=missing_interactions(table, names),
    )


def make_agent(args: Args, cell: Cell) -> PPO:
    """Wire the hyperparameters and the network onto the cell's environment."""
    hparams = PPOHparams(
        budget=args.budget,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        num_epochs=args.num_epochs,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        anneal_lr=args.anneal_lr,
        mean_option_len=cell.mean_option_len,
        discount_mode=args.discount,
        stagger_envs=args.stagger_envs,
    )
    encoder = ConvEncoder(hidden_size=args.hidden_size)
    return PPO(
        hparams=hparams,
        network=ActorCritic(
            action_dim=len(cell.table), actor_encoder=encoder, critic_encoder=encoder
        ),
        env=cell.env,
    )


def train_cell(
    args: Args,
    cell: Cell,
    seeds: Sequence[int],
    on_chunk: OnChunk,
    vmap_seeds: bool = True,
    updates_per_chunk: int = UPDATES_PER_CHUNK,
) -> float:
    """Train every seed of a cell, handing each chunk's logs to `on_chunk`.

    Returns wall-clock seconds. `on_chunk` always sees a leading seed axis;
    without `vmap_seeds` the seeds run one after another, an axis of 1 at a
    time, which is what attributes a failure to a seed since vmap fuses them.
    """
    agent = make_agent(args, cell)

    def advance(seed_group: Sequence[int]) -> None:
        """Run one group of seeds to completion, a chunk at a time."""
        keys = [jax.random.PRNGKey(int(seed)) for seed in seed_group]
        if vmap_seeds:
            state = jax.vmap(agent.init)(jnp.stack(keys))
            step = jax.jit(jax.vmap(partial(agent.run, num_updates=updates_per_chunk)))
        else:
            state = agent.init(keys[0])
            step = jax.jit(partial(agent.run, num_updates=updates_per_chunk))
        # the slowest seed of the group, so every one of them reaches the
        # budget; the faster ones overrun and `episode_frame` drops what they
        # record past it
        while int(jnp.min(state.frames)) < args.budget:
            state, logs = step(state)
            if not vmap_seeds:
                logs = jax.tree.map(lambda leaf: leaf[None], logs)
            on_chunk(seed_group, logs)

    start = time.time()
    if vmap_seeds:
        advance(list(seeds))
    else:
        for seed in seeds:
            advance([seed])
    return time.time() - start
