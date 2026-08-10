import argparse

import numpy as np

from envs import make_env
from options import make_options

import gymnasium as gym
import nle  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="NetHackChallenge-v0")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--use-options", action="store_true")
    args = p.parse_args()

    raw = gym.make(args.env_id)
    print("n primitive actions:", raw.action_space.n)
    for i, a in enumerate(raw.unwrapped.actions):
        print(i, repr(a))

    options, names = [], []
    if args.use_options:
        options, names = make_options(raw.unwrapped.actions)
        print(f"\n{len(options)} options, mean length "
              f"{np.mean([len(o) for o in options]):.2f}")

    env = make_env(args.env_id, 0, 0, args.use_options)()
    obs, _ = env.reset(seed=0)

    primitive = 0
    decisions = 0
    episodes = 0
    durations = []

    while primitive < args.steps:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        decisions += 1
        step_cost = info.get("primitive_steps", 1)
        primitive += step_cost
        durations.append(step_cost)
        if terminated or truncated:
            episodes += 1
            env.reset()

    print(f"\nprimitive steps: {primitive}")
    print(f"decisions:       {decisions}")
    print(f"mean duration:   {np.mean(durations):.2f}")
    print(f"episodes:        {episodes}")

    for name, seq in zip(names, options):
        print(f"{name}: {seq}")


if __name__ == "__main__":
    main()