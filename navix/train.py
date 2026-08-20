import csv
import dataclasses
import json
import pathlib
import time
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import navix as nx
from navix import observations
from navix.agents.models import ActorCritic, ConvEncoder
from navix.environments.environment import StepType

from options import (
    OptionEnv,
    OptionSpec,
    action_names,
    action_table,
    make_options,
    mean_nominal_duration,
    measure_duration,
)
from ppo import PPO, PPOHparams

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


@dataclass
class Args:
    env_id: str = "Navix-DoorKey-8x8-v0"
    action_space: Literal["action", "option", "both"] = "action"
    """primitives only, options only, or the union of both"""
    n_options: int = 64
    """how many options to generate; ignored when action_space is `action`"""
    option_family: Literal["random", "grammar"] = "random"
    """uniform draws from the controller catalogue, or the catalogue ordered
    longest-reach first"""
    option_seed: int = 0
    """seed for the random option family, kept separate from the training seed"""
    max_forward: int = 10
    """furthest an option's policy will walk before terminating; sets the
    catalogue size"""
    max_steps: int = 400
    """primitive steps before truncation. Set explicitly: DoorKey-8x8 solves in
    ~25 steps, so the registry default of 100 leaves no horizon for credit
    assignment to matter."""
    budget: int = 1_000_000
    """training budget in PRIMITIVE steps"""
    discount: Literal["decision", "primitive"] = "decision"
    """gamma once per decision, or raised to the option's duration (SMDP)"""
    stagger_envs: bool = True
    """offset each env's episode clock so truncations do not all land together"""
    seed: int = 1
    tag: str = ""
    """optional label appended to the run directory name"""

    num_envs: int = 16
    num_steps: int = 128
    num_minibatches: int = 8
    num_epochs: int = 1
    lr: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 64


def episode_writer(writer, flush, num_envs):
    """Host callback turning one update's (T, N) logs into episode rows."""
    state = {"updates": 0}

    def write(
        done, returns, lengths, primitive_steps, step_type, frames_before,
        decisions_before,
    ):
        done = np.asarray(done)  # (T, N)
        returns = np.asarray(returns)
        lengths = np.asarray(lengths)
        steps = np.asarray(primitive_steps)
        terminated = np.asarray(step_type) == int(StepType.TERMINATION)

        primitive = int(frames_before) + np.cumsum(steps.sum(axis=1))  # (T,)
        decision = int(decisions_before) + (np.arange(len(done)) + 1) * num_envs

        finished = np.argwhere(done)
        for t, n in finished:
            writer.writerow(
                [
                    decision[t],
                    primitive[t],
                    float(returns[t, n]),
                    int(lengths[t, n]),
                    int(terminated[t, n]),
                ]
            )
        flush()

        state["updates"] += 1
        mean_return = float(returns[done].mean()) if len(finished) else float("nan")
        print(
            f"update={state['updates']} decisions={decision[-1]} "
            f"primitive={primitive[-1]} episodes={len(finished)} "
            f"return={mean_return:.3f} terminated={int(terminated[done].sum())}",
            flush=True,
        )

    return write


def main(args: Args):
    env = nx.make(
        args.env_id,
        max_steps=args.max_steps,
        gamma=args.gamma,
        observation_fn=observations.symbolic,
    )
    names = action_names(env)

    options, labels = [], []
    if args.action_space != "action":
        options, labels = make_options(
            args.option_family, args.n_options, args.max_forward, names, args.option_seed
        )
    table, table_names = action_table(args.action_space, names, options, labels)
    spec = OptionSpec.create(table, names)
    wrapped = OptionEnv.create(env, spec)
    nominal_option_len = mean_nominal_duration(table)
    mean_option_len = measure_duration(wrapped, seed=args.option_seed)

    parts = [
        args.env_id,
        args.action_space,
        f"seed{args.seed}",
        time.strftime("%m-%d_%H-%M"),
    ]
    if args.tag:
        parts.append(args.tag.strip().replace(" ", "-"))
    run_name = "__".join(parts)
    run_dir = RESULTS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = dataclasses.asdict(args)
    config.update(
        run_name=run_name,
        n_primitives=len(names),
        n_actions=len(table),
        n_generated_options=len(options),
        max_option_len=spec.horizon,
        mean_option_len=mean_option_len,
        nominal_option_len=nominal_option_len,
        primitives=names,
        options={name: list(row) for name, row in zip(table_names, table)},
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(
        f"{run_name}\n{len(table)} actions, {mean_option_len:.2f} primitive steps "
        f"per option measured ({nominal_option_len:.2f} nominal), at most "
        f"{spec.horizon}"
    )

    csv_file = open(run_dir / "episodes.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "decision_step",
            "primitive_step",
            "episodic_return",
            "episodic_length",
            "terminated",
        ]
    )

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
        mean_option_len=mean_option_len,
        discount_mode=args.discount,
        stagger_envs=args.stagger_envs,
    )
    encoder = ConvEncoder(hidden_size=args.hidden_size)
    agent = PPO(
        hparams=hparams,
        network=ActorCritic(
            action_dim=len(table), actor_encoder=encoder, critic_encoder=encoder
        ),
        env=wrapped,
        episode_callback=episode_writer(csv_writer, csv_file.flush, args.num_envs),
    )

    start = time.time()
    train_state, logs = agent.train(jax.random.PRNGKey(args.seed))
    frames = int(jax.block_until_ready(train_state.frames))
    elapsed = time.time() - start
    csv_file.close()

    print(
        f"done: {frames} primitive steps, {int(train_state.decisions)} decisions, "
        f"{elapsed:.1f}s, {frames / elapsed:.0f} primitive steps/s"
    )
    np.savez(
        run_dir / "logs.npz",
        **{k: np.asarray(v) for k, v in logs.items() if jnp.ndim(v) <= 3},
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
