"""Report an action table's size and the durations it realises. Writes nothing.

`cd src && python probe.py --condition option`
"""

import argparse
from typing import List

import gymnasium as gym
import nle  # noqa: F401  registers NetHack envs with gymnasium
import numpy as np

from envs import make_env
from options import make_options


def main() -> None:
    """Print the catalogue, then the durations a uniform-over-`I` policy draws."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="NetHackChallenge-v0")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--condition", default="option", choices=["action", "option", "both"])
    parser.add_argument("--max-episode-steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = gym.make(args.env_id)
    sequences, names, required_slots = make_options(raw.unwrapped.actions, args.condition)
    gated = sum(slot is not None for slot in required_slots)
    print(f"n primitive actions: {raw.action_space.n}")
    print(
        f"{len(sequences)} rows, {gated} gated on an inventory slot, "
        f"mean length {np.mean([len(s) for s in sequences]):.2f}"
    )
    raw.close()

    env = make_env(
        env_id=args.env_id,
        seed=args.seed,
        idx=0,
        condition=args.condition,
        gamma=0.999,
        max_episode_steps=args.max_episode_steps,
        clip_reward=True,
    )()
    _, info = env.reset(seed=args.seed)

    rng = np.random.default_rng(args.seed)
    primitive = 0
    decisions = 0
    episodes = 0
    durations: List[int] = []
    available_fracs: List[float] = []

    while primitive < args.steps:
        # uniform over the available rows, not over the whole table: that is the
        # distribution the masked policy starts from, so it is the one whose
        # realised duration predicts the trainer's decisions-per-frame
        offered = np.flatnonzero(info["available"])
        _, _, terminated, truncated, info = env.step(int(rng.choice(offered)))
        decisions += 1
        primitive += info["primitive_steps"]
        durations.append(info["primitive_steps"])
        available_fracs.append(info["available_frac"])
        if terminated or truncated:
            episodes += 1
            _, info = env.reset()
    env.close()

    print(f"\nprimitive steps: {primitive}")
    print(f"decisions:       {decisions}")
    print(f"mean duration:   {np.mean(durations):.2f}")
    print(f"available_frac:  {np.mean(available_fracs):.3f}")
    print(f"episodes:        {episodes}")

    for name, sequence in zip(names, sequences):
        print(f"{name}: {sequence}")


if __name__ == "__main__":
    main()
