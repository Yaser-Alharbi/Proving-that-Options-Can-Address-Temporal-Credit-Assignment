"""Combined env-plus-network step rate against num_envs. Writes nothing.

Forks a fresh process per configuration: creating, playing and closing several
differently-sized vector envs in one process reproduced a SIGABRT during Q5.

`cd src && python sps_probe.py`
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Callable, List

import gymnasium as gym
import numpy as np
import torch

from envs import make_env
from ppo import Agent

NUM_ENVS_GRID = (4, 8, 16, 32)
STEPS_PER_DECISION_GRID = (1, 4)
DECISIONS = 200
WARMUP = 20


class RepeatSteps(gym.Wrapper):
    """Stand-in for OptionWrapper's cost: `k` primitive steps per decision."""

    def __init__(self, env: gym.Env, k: int) -> None:
        super().__init__(env)
        self.k = k

    def step(self, action: int) -> tuple:
        total = 0.0
        terminated = truncated = False
        observation = None
        info: dict = {}
        for _ in range(self.k):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        return observation, total, terminated, truncated, info


def make_vector(num_envs: int, k: int, async_mode: bool) -> gym.vector.VectorEnv:
    """Build the vector env under test. `k==1` is the action condition."""

    def thunk(index: int) -> Callable[[], gym.Env]:
        def inner() -> gym.Env:
            env = make_env(
                env_id="NetHackChallenge-v0",
                seed=0,
                idx=index,
                condition="action",
                gamma=0.999,
                max_episode_steps=100_000,
                clip_reward=True,
                n_options=64,
                option_family="grammar",
                option_seed=0,
                reward_delay=0,
            )()
            return RepeatSteps(env, k) if k > 1 else env

        return inner

    ctor = gym.vector.AsyncVectorEnv if async_mode else gym.vector.SyncVectorEnv
    kwargs = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}
    if async_mode:
        kwargs["context"] = "fork"
        kwargs["shared_memory"] = True
    return ctor([thunk(i) for i in range(num_envs)], **kwargs)


def run_child(num_envs: int, k: int, async_mode: bool) -> dict:
    """One configuration in this process. Prints one JSON object on stdout."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = make_vector(num_envs, k, async_mode)
    agent = Agent(envs).to(device)
    obs_np, infos = envs.reset(seed=0)
    obs = {
        "glyphs": torch.as_tensor(obs_np["glyphs"], dtype=torch.int16, device=device),
        "blstats": torch.as_tensor(obs_np["blstats"], dtype=torch.float32, device=device),
    }
    available = torch.as_tensor(infos["available"], dtype=torch.bool, device=device)
    for _ in range(WARMUP):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs, available)
        obs_np, _, _, _, infos = envs.step(action.cpu().numpy())
        obs = {
            "glyphs": torch.as_tensor(obs_np["glyphs"], dtype=torch.int16, device=device),
            "blstats": torch.as_tensor(obs_np["blstats"], dtype=torch.float32, device=device),
        }
        available = torch.as_tensor(infos["available"], dtype=torch.bool, device=device)

    t0 = time.perf_counter()
    for _ in range(DECISIONS):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs, available)
        obs_np, _, _, _, infos = envs.step(action.cpu().numpy())
        obs = {
            "glyphs": torch.as_tensor(obs_np["glyphs"], dtype=torch.int16, device=device),
            "blstats": torch.as_tensor(obs_np["blstats"], dtype=torch.float32, device=device),
        }
        available = torch.as_tensor(infos["available"], dtype=torch.bool, device=device)
    elapsed = time.perf_counter() - t0
    envs.close()
    primitive = num_envs * DECISIONS * k
    return {
        "num_envs": num_envs,
        "k": k,
        "async": async_mode,
        "seconds": round(elapsed, 3),
        "primitive_sps": round(primitive / elapsed),
        "decisions_per_s": round(num_envs * DECISIONS / elapsed),
        "device": str(device),
    }


def main() -> int:
    """Sweep num_envs × k × {sync, async}, one subprocess each."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--async-env", action="store_true")
    args = parser.parse_args()

    if args.child:
        print(json.dumps(run_child(args.num_envs, args.k, args.async_env)))
        return 0

    print("num_envs  k  mode   primitive_sps  decisions/s  seconds  device")
    for num_envs in NUM_ENVS_GRID:
        for k in STEPS_PER_DECISION_GRID:
            for async_mode in (False, True):
                cmd: List[str] = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--child",
                    "--num-envs",
                    str(num_envs),
                    "--k",
                    str(k),
                ]
                if async_mode:
                    cmd.append("--async-env")
                completed = subprocess.run(
                    cmd, cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True
                )
                if completed.returncode != 0:
                    print(
                        f"{num_envs:8d} {k:2d} {'async' if async_mode else 'sync':5s}  "
                        f"FAILED: {completed.stderr.strip()[-200:]}"
                    )
                    continue
                row = json.loads(completed.stdout.strip().splitlines()[-1])
                print(
                    f"{row['num_envs']:8d} {row['k']:2d} "
                    f"{'async' if row['async'] else 'sync':5s}  "
                    f"{row['primitive_sps']:13d} {row['decisions_per_s']:11d}  "
                    f"{row['seconds']:7.2f}  {row['device']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
