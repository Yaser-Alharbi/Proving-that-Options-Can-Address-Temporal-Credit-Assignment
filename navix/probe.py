import argparse

import jax
import jax.numpy as jnp
import numpy as np

import navix as nx
from navix import observations

from options import (
    OptionEnv,
    OptionSpec,
    action_names,
    action_table,
    make_options,
    mean_nominal_duration,
)


def rollout(env, num_envs, num_decisions, seed):

    def step(carry, _):
        timestep, rng = carry
        rng, key = jax.random.split(rng)
        logits = jnp.where(timestep.info["available"], 0.0, -1e8)
        action = jax.random.categorical(key, logits)
        timestep = jax.vmap(env.step)(timestep, action)
        return (timestep, rng), (
            timestep.info["primitive_steps"],
            timestep.is_done(),
        )

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    timestep = jax.vmap(env.reset)(jax.random.split(key, num_envs))
    _, (steps, done) = jax.lax.scan(step, (timestep, rng), None, num_decisions)
    return np.asarray(steps), np.asarray(done)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Navix-DoorKey-8x8-v0")
    p.add_argument("--action-space", default="option",
                   choices=["action", "option", "both"])
    p.add_argument("--n-options", type=int, default=64)
    p.add_argument("--option-family", default="random", choices=["random", "grammar"])
    p.add_argument("--option-seed", type=int, default=0)
    p.add_argument("--max-forward", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--decisions", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = nx.make(
        args.env_id, max_steps=args.max_steps, observation_fn=observations.symbolic
    )
    names = action_names(env)
    print(f"{len(names)} primitive actions:")
    for i, name in enumerate(names):
        print(f"  {i:3d} {name}")

    options, labels = [], []
    if args.action_space != "action":
        options, labels = make_options(
            args.option_family, args.n_options, args.max_forward, names,
            args.option_seed,
        )
    table, table_names = action_table(args.action_space, names, options, labels)
    spec = OptionSpec.create(table, names)

    print(f"\n{len(table)} actions in the `{args.action_space}` space "
          f"(family={args.option_family}):")
    print("      heading reach interact follow")
    for i, (name, row) in enumerate(zip(table_names, table)):
        heading, reach, interact, follow = row
        print(f"  {i:3d} {heading:7d} {reach:5d} {interact:8d} {follow:6d}  {name}")

    print(f"\ncount:                 {len(table)}")
    print(f"horizon:               {spec.horizon}")
    print(f"nominal duration:      {mean_nominal_duration(table):.2f}")

    wrapped = OptionEnv.create(env, spec)
    steps, done = rollout(wrapped, args.num_envs, args.decisions, args.seed)
    decisions = steps.size
    primitive = int(steps.sum())
    print(f"\nunder uniform-random selection over {decisions} decisions "
          f"in {args.num_envs} envs:")
    print(f"  primitive steps:            {primitive}")
    print(f"  mean primitive per decision: {primitive / decisions:.2f}")
    print(f"  episodes finished:           {int(done.sum())}")


if __name__ == "__main__":
    main()
