"""Per-cell arguments, shared by the runner and the trainer.

Kept free of JAX so `main.py` can build and print the run matrix without
importing it, and so the environment can still be configured beforehand.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Args:
    """Everything that defines a cell except its seeds."""

    env_id: str = "Navix-DoorKey-8x8-v0"
    action_space: Literal["action", "option", "both"] = "action"
    """primitives only, options only, or the union of both"""
    n_options: int = 64
    """how many options to generate; ignored when action_space is `action`"""
    option_family: Literal["random", "grammar"] = "random"
    """uniform draws from the controller catalogue, or the catalogue in
    priority order: non-following rows first, then longest reach"""
    option_seed: int = 0
    """seed for the random option family, kept separate from the training seeds"""
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
    executor: Literal["scan", "while_loop"] = "scan"
    """how the option executor iterates: always run the table's horizon and
    mask, or stop each lane at its own beta. Same trajectories, different cost;
    scan measured faster at max_forward=4"""
    stagger_envs: bool = True
    """offset each env's episode clock so truncations do not all land together"""
    tag: str = ""
    """optional label carried into the cell name and the results"""

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
