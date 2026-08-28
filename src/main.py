"""Runner for the NLE experiment matrix.

One process is one seed of one cell: PyTorch cannot vmap seeds. `ppo.py` owns
the cell; this file owns the group directory, the launch, and the preflight.
`batch_size` and `minibatch_size` are resolved inside `ppo.py`.
"""

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Iterator, List, Literal, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"
GROUP_STAMP = time.strftime("%m%d-%H%M%S")

Condition = Literal["action", "option", "both"]
Discount = Literal["decision", "primitive"]


@dataclass(frozen=True)
class CellArgs:
    """Everything that defines a cell except its seeds."""

    env_id: str = "NetHackChallenge-v0"
    condition: Condition = "action"
    discount: Discount = "decision"
    budget: int = 10_000_000
    max_episode_steps: int = 100_000
    clip_reward: bool = True
    gamma: float = 0.999
    num_envs: int = 16
    num_steps: int = 128
    tag: str = ""


_DEFAULT_CELL = CellArgs()
_NAMED_IF_NONDEFAULT = ("budget", "max_episode_steps", "gamma", "discount")


@dataclass(frozen=True)
class Sweep:
    """The arguments a matrix holds fixed, and the axes it varies over them."""

    base: CellArgs = dataclasses.field(default_factory=CellArgs)
    conditions: Tuple[str, ...] = ("action", "option", "both")
    seeds: Tuple[int, ...] = (0, 1, 2)


SWEEPS: Dict[str, Sweep] = {
    "exp1": Sweep(base=CellArgs(tag="exp1")),
}


@dataclass(frozen=True)
class Cell:
    """One point of the matrix, named without its seeds, inside a timestamped group."""

    args: CellArgs
    seeds: Tuple[int, ...]

    @property
    def name(self) -> str:
        parts = [self.args.env_id, self.args.condition]
        for field in _NAMED_IF_NONDEFAULT:
            value = getattr(self.args, field)
            if value != getattr(_DEFAULT_CELL, field):
                parts.append(f"{field}{value}")
        if self.args.tag:
            parts.append(self.args.tag)
        return "__".join(parts)

    @property
    def group(self) -> str:
        parts = [GROUP_STAMP, self.args.env_id]
        if self.args.tag:
            parts.append(self.args.tag)
        return "__".join(parts)

    @property
    def directory(self) -> pathlib.Path:
        return RUNS / self.group / self.name


def expand(sweep: Sweep) -> Iterator[Cell]:
    """Every cell of a sweep."""
    for condition in sweep.conditions:
        yield Cell(
            args=dataclasses.replace(sweep.base, condition=condition),
            seeds=sweep.seeds,
        )


def select(cells: Sequence[Cell], names: Optional[Sequence[str]]) -> List[Cell]:
    """The cells `--cell` named, or all of them."""
    if not names:
        return list(cells)
    return [cell for cell in cells if cell.name in names]


def assert_gpu() -> None:
    """Abort unless torch can see a CUDA device."""
    import torch

    print(f"torch.cuda.is_available={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no CUDA device. NLE training is not configured for CPU.")


def ppo_command(cell: Cell, seed: int) -> List[str]:
    """The argv that runs one seed of `cell` inside `cell.directory`."""
    args = cell.args
    return [
        sys.executable,
        str(HERE / "ppo.py"),
        "--directory",
        str(cell.directory),
        "--env-id",
        args.env_id,
        "--condition",
        args.condition,
        "--discount",
        args.discount,
        "--budget",
        str(args.budget),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--gamma",
        str(args.gamma),
        "--num-envs",
        str(args.num_envs),
        "--num-steps",
        str(args.num_steps),
        "--seed",
        str(seed),
        "--tag",
        args.tag,
        "--clip-reward" if args.clip_reward else "--no-clip-reward",
    ]


def run_seed(cell: Cell, seed: int) -> None:
    """Run one seed; the child writes episodes and checkpoints into `cell.directory`."""
    print(f"  seed {seed}", flush=True)
    completed = subprocess.run(ppo_command(cell, seed), cwd=str(HERE))
    if completed.returncode != 0:
        raise RuntimeError(f"{cell.name} seed {seed} exited {completed.returncode}")


def finish_cell(cell: Cell, elapsed: float) -> None:
    """Write the `meta.json` that marks the cell complete."""
    meta = {
        "cell": cell.name,
        "group": cell.group,
        "seeds": list(cell.seeds),
        "args": dataclasses.asdict(cell.args),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
    }
    (cell.directory / "meta.json").write_text(json.dumps(meta, indent=2))


def run_cell(cell: Cell) -> None:
    """Train every seed of a cell sequentially into one directory."""
    import shutil

    cell.directory.mkdir(parents=True, exist_ok=True)
    print(f"{cell.name}\n  seeds {list(cell.seeds)}", flush=True)
    start = time.time()
    try:
        for seed in cell.seeds:
            run_seed(cell, seed)
    except Exception:
        shutil.rmtree(cell.directory, ignore_errors=True)
        raise
    finish_cell(cell, time.time() - start)
    print(f"  done in {time.time() - start:.1f}s", flush=True)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="exp1", choices=sorted(SWEEPS))
    parser.add_argument("--cell", action="append", help="run only these cells")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the selected cells, returning a process exit status."""
    options = parse_arguments(argv)
    cells = select(list(expand(SWEEPS[options.sweep])), options.cell)
    if not cells:
        print("no cells match those names")
        return 1

    if options.dry_run:
        print(f"{len(cells)} cells in sweep {options.sweep!r}:")
        for cell in cells:
            print(f"  {cell.name} seeds={list(cell.seeds)}")
        return 0

    assert_gpu()

    failures: List[Cell] = []
    for cell in cells:
        try:
            run_cell(cell)
        except Exception:
            traceback.print_exc()
            failures.append(cell)

    print(f"\n{len(cells) - len(failures)}/{len(cells)} cells complete.")
    for cell in failures:
        print(
            f"  failed: {cell.name}\n"
            f"    {sys.executable} {HERE / 'main.py'} --sweep {options.sweep} "
            f"--cell {cell.name}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
