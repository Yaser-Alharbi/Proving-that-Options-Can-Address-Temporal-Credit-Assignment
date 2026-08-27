"""Runner for the navix experiment matrix.

JAX reads its device and cache configuration from the environment at import
time, so the environment is set here and `train`, `options` and `navix` are
imported inside functions. Building and printing the matrix therefore needs
neither a GPU nor a JAX install.
"""

import argparse
import dataclasses
import itertools
import json
import os
import pathlib
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Tuple

from config import Args

if TYPE_CHECKING:
    from train import Cell as BuiltCell

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"

GROUP_STAMP = time.strftime("%m%d-%H%M%S") # one stamp per invocation for `{group}`

_DEFAULT_ARGS = Args() # default arguments for `Cell.name`.

_NAMED_IF_NONDEFAULT = ("budget", "max_forward", "max_steps", "gamma", "discount", "executor", "reward_delay") # fields appended to `Cell.name` if non-default.


def cell_name(directory: pathlib.Path) -> str:
    """A cell's store-relative name, `{group}/{cell}`."""
    return directory.relative_to(RUNS).as_posix()


def configure_environment() -> None:
    """Set JAX's device and compilation-cache variables, where unset."""
    defaults = {
        # cuda with no cpu fallback: the preflight asserts the GPU, and listing
        # cpu would let a broken jaxlib train silently on the host instead
        "JAX_PLATFORMS": "cuda",
        # no fraction of the arena is obtainable on this box, so grow on demand
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_COMPILATION_CACHE_DIR": str(
            pathlib.Path.home() / ".cache" / "jax-compilation"
        ),
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1.0",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    pathlib.Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).mkdir(
        parents=True, exist_ok=True
    )


@dataclass(frozen=True)
class Sweep:
    """The arguments a matrix holds fixed, and the axes it varies over them."""

    base: Args = dataclasses.field(default_factory=Args)
    conditions: Tuple[str, ...] = ("action", "option", "both")
    families: Tuple[str, ...] = ("random", "grammar")
    n_options: Tuple[int, ...] = (8, 16, 32, 64, 128)
    option_seeds: Tuple[int, ...] = (0,)
    """draws of the `random` catalogue per count; `grammar` ignores the seed"""
    seeds: Tuple[int, ...] = (0, 1, 2)
    threshold: Optional[float] = None
    """episodic return plot.py times the crossing of, where this environment's
    asymptote puts its own default out of reach"""
    reward_delays: Tuple[int, ...] = ()
    """empty uses `base.reward_delay`, so a sweep that does not name this stays one cell"""
    discounts: Tuple[str, ...] = ()
    gammas: Tuple[float, ...] = ()


ANALYSIS_SEEDS: Tuple[int, ...] = tuple(range(10))
"""plot.py's MIN_SEEDS_FOR_IQM is 8, if below estimate falls back to median-and-range not bootstrapped IQM interval."""

SWEEPS: Dict[str, Sweep] = {
    "exp1": Sweep(
        Args(env_id="Navix-DoorKey-8x8-v0", tag="exp1", max_forward=4),
        families=("grammar",),
        n_options=(64,),
        seeds=ANALYSIS_SEEDS,
    ),
    "exp1_16x16_Random": Sweep(
        Args(
            env_id="Navix-DoorKey-Random-16x16-v0",
            max_forward=4,
            tag="exp1_16x16_Random",
        ),
        families=("grammar",),
        n_options=(64,),
        seeds=ANALYSIS_SEEDS,
        threshold=0.15,
    ),
    "exp1_16x16": Sweep(
        Args(
            env_id="Navix-DoorKey-16x16-v0",
            max_forward=4,
            tag="exp1_16x16",
        ),
        families=("grammar",),
        n_options=(64,),
        seeds=ANALYSIS_SEEDS,
        threshold=0.15,
    ),
    "exp2": Sweep(
        Args(max_forward=4, tag="exp2"),
        conditions=("action", "option", "both"),
        families=("grammar",),
        n_options=(8, 16, 32, 64, 128),
        seeds=ANALYSIS_SEEDS,
    ),
    "exp2_16x16": Sweep(
        Args(
            env_id="Navix-DoorKey-16x16-v0",
            max_forward=8,
            tag="exp2_16x16",
        ),
        conditions=("action", "option", "both"),
        families=("grammar",),
        n_options=(8, 16, 32, 64, 128, 256),
        seeds=ANALYSIS_SEEDS,
        threshold=0.1,
    ),
    "exp2_16x16_Random": Sweep(
        Args(
            env_id="Navix-DoorKey-Random-16x16-v0",
            max_forward=8,  
            tag="exp2_16x16_Random",  
        ),
        conditions=("action", "option", "both"),
        families=("grammar",),
        n_options=(8, 16, 32, 64, 128, 256),
        seeds=ANALYSIS_SEEDS,
    ),
    # n=64, not 128: at max_forward=4 the catalogue is 128 rows, so at n=128
    # `random.sample` returns the whole catalogue and the two families coincide
    "exp3": Sweep(
        Args(max_forward=4, tag="exp3"),
        conditions=("option",),
        families=("random", "grammar"),
        n_options=(64,),
        seeds=ANALYSIS_SEEDS,
        option_seeds=(0, 1, 2, 3, 4),
    ),
    "exp3_16x16_Random": Sweep(
        Args(
            env_id="Navix-DoorKey-Random-16x16-v0",
            max_forward=8,
            tag="exp3",
            option_seed=0,
            budget=5_000_000,
        ),
        conditions=("option",),
        families=("random", "grammar"),
        n_options=(64,),
        option_seeds=(0, 1, 2, 3, 4),
        seeds=ANALYSIS_SEEDS,
        threshold=0.15,
    ),
    "exp4_probe": Sweep(
    Args(
        env_id="Navix-DoorKey-8x8-v0", #env_id="Navix-KeyCorridorS6R3-v0"
        max_steps=400,
        budget=1_000_000,
        tag="exp4_probe",
    ),
    conditions=("action",),
    reward_delays=(0, 16, 32, 64),
    seeds=ANALYSIS_SEEDS,
),
    "exp4_decision": Sweep(
        Args(
            env_id="Navix-DoorKey-Random-16x16-v0",
            max_forward=8,
            max_steps=400,
            budget=5_000_000,
            option_family="grammar",
            option_seed=0,
            tag="exp4_decision",
        ),
        conditions=("option"),
        families=("grammar",),
        n_options=(64,),
        reward_delays=(0, 8, 16, 32, 64),
        discounts=("decision",),
        seeds=ANALYSIS_SEEDS,
        option_seeds=(0,),
    ),
    "exp4_primitive": Sweep(
        Args(
            env_id="Navix-DoorKey-Random-16x16-v0",
            max_forward=8,
            max_steps=400,
            budget=5_000_000,
            option_family="grammar",
            option_seed=0,
            tag="exp4_primitive",
        ),
        conditions=("action", "option"),
        families=("grammar",),
        n_options=(64,),
        reward_delays=(0, 8, 16, 32, 64),
        discounts=("primitive",),
        seeds=ANALYSIS_SEEDS,
        option_seeds=(0,),
    ),
    "long-baseline": Sweep(
        Args(max_forward=4, budget=3_000_000, tag="long-baseline-3M"),
        conditions=("action",),
    ),
    "smoke": Sweep(
    Args(max_forward=4, budget=200_000, tag="smoke"),
    n_options=(8, 16),
    seeds=tuple(range(10))
    ),
}

THRESHOLDS: Dict[Tuple[str, str], float] = {
    (sweep.base.env_id, sweep.base.tag): sweep.threshold
    for sweep in SWEEPS.values()
    if sweep.threshold is not None
}
"""What plot.py reads a declared threshold by: the pair a run group's name is built from."""


@dataclass(frozen=True)
class Cell:
    """One point of the matrix, named without its seeds, inside a timestamped group."""

    args: Args
    seeds: Tuple[int, ...]

    @property
    def name(self) -> str:
        parts = [self.args.env_id, self.args.action_space]
        if self.args.action_space != "action":
            parts += [
                self.args.option_family,
                f"n{self.args.n_options}",
                f"os{self.args.option_seed}",
            ]
        for field in _NAMED_IF_NONDEFAULT:
            value = getattr(self.args, field)
            if value != getattr(_DEFAULT_ARGS, field):
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

    @property
    def identity(self) -> Dict[str, object]:
        """The columns naming this cell on every one of its result rows."""
        options = self.args.action_space != "action"
        return {
            "env_id": self.args.env_id,
            "condition": self.args.action_space,
            "family": self.args.option_family if options else "-",
            "n_options": self.args.n_options if options else 0,
            "option_seed": self.args.option_seed,
            "budget": self.args.budget,
            "max_forward": self.args.max_forward,
            "max_steps": self.args.max_steps,
            "reward_delay": self.args.reward_delay,
            "gamma": self.args.gamma,
            "discount": self.args.discount,
            "executor": self.args.executor,
            "tag": self.args.tag,
        }


def expand(sweep: Sweep) -> Iterator[Cell]:
    """Every cell of a sweep: `action` once, the rest per family, count and draw.

    An empty `reward_delays`, `discounts` or `gammas` uses the value on `base`.
    """
    delays = sweep.reward_delays or (sweep.base.reward_delay,)
    modes = sweep.discounts or (sweep.base.discount,)
    gammas = sweep.gammas or (sweep.base.gamma,)
    for condition in sweep.conditions:
        if condition == "action":
            variants: List[Dict[str, object]] = [{}]
        else:
            variants = []
            for family in sweep.families:
                # the grammar catalogue is a function of n and max_forward alone,
                # so drawing it once per option seed would repeat the same cell
                seeds = (0,) if family == "grammar" else sweep.option_seeds
                variants += [
                    {
                        "option_family": family,
                        "n_options": count,
                        "option_seed": option_seed,
                    }
                    for count, option_seed in itertools.product(sweep.n_options, seeds)
                ]
        for delay, mode, gamma, variant in itertools.product(
            delays, modes, gammas, variants
        ):
            yield Cell(
                args=dataclasses.replace(
                    sweep.base,
                    action_space=condition,
                    reward_delay=delay,
                    discount=mode,
                    gamma=gamma,
                    **variant,
                ),
                seeds=sweep.seeds,
            )


def select(cells: Sequence[Cell], names: Optional[Sequence[str]]) -> List[Cell]:
    """The cells `--cell` named, or all of them."""
    if not names:
        return list(cells)
    return [cell for cell in cells if cell.name in names]


def assert_gpu() -> None:
    """Abort unless JAX's default device is a GPU."""
    import jax

    devices = jax.devices()
    print(f"JAX devices: {devices} | backend: {jax.default_backend()}", flush=True)
    if devices[0].platform != "gpu":
        raise SystemExit(
            "ABORT: JAX is not on the GPU. Check JAX_PLATFORMS and the jaxlib build."
        )


def assert_catalogues(cells: Sequence[Cell]) -> None:
    """Abort if any cell asks for more options than its family can yield.

    A catalogue reaching no `toggle` row is a warning, not an abort: the run
    still carries information and `meta.json` records the gap.
    """
    import navix as nx

    from options import action_names, make_options, missing_interactions

    names_for: Dict[str, List[str]] = {}
    fatal: List[str] = []
    incomplete: List[str] = []
    for cell in cells:
        if cell.args.action_space == "action":
            continue
        names = names_for.setdefault(
            cell.args.env_id, action_names(nx.make(cell.args.env_id))
        )
        try:
            rows, _ = make_options(
                cell.args.option_family,
                cell.args.n_options,
                cell.args.max_forward,
                names,
            )
        except ValueError as error:
            fatal.append(f"{cell.name}: {error}")
            continue
        missing = missing_interactions(rows, names)
        if missing and cell.args.action_space == "option":
            incomplete.append(f"{cell.name} lacks {','.join(missing)}")

    for line in fatal:
        print(f"  FATAL {line}")
    for line in incomplete:
        print(f"  WARNING option-only {line}")
    if not fatal and not incomplete:
        print("  every catalogue reaches every interaction")
    if fatal:
        raise SystemExit("ABORT: an option count exceeds the catalogue.")


def episode_frame(cell: Cell, seeds: Sequence[int], logs: Dict[str, object]):
    """One row per episode that finished in a chunk, with the cell's columns.

    Logs are indexed (seed, update, step, env). A transition's primitive step
    is the frame count before its update plus what every env spent up to it.
    """
    import numpy as np
    import pandas as pd
    from navix.environments.environment import StepType

    done = np.asarray(logs["done_mask"])
    num_steps, num_envs = done.shape[2], done.shape[3]

    per_decision = np.asarray(logs["primitive_steps"]).sum(axis=3)
    frames_before = np.asarray(logs["iter/frames"]) - per_decision.sum(axis=2)
    primitive = frames_before[..., None] + np.cumsum(per_decision, axis=2)
    decisions_before = np.asarray(logs["iter/decisions"]) - num_steps * num_envs
    decision = decisions_before[..., None] + (np.arange(num_steps) + 1) * num_envs

    seed_at, update_at, step_at, _ = np.nonzero(done)
    terminated = np.asarray(logs["step_type"])[done] == int(StepType.TERMINATION)
    lengths = np.asarray(logs["lengths"])[done]
    decisions = np.asarray(logs["decision_t"])[done]
    frame = pd.DataFrame(
        {
            **cell.identity,
            "seed": np.asarray(seeds)[seed_at],
            "decision_step": decision[seed_at, update_at, step_at],
            "primitive_step": primitive[seed_at, update_at, step_at],
            "episodic_return": np.asarray(logs["returns"])[done],
            "episodic_length": lengths,
            "terminated": terminated.astype(int),
            "mean_option_duration": lengths / decisions,
        }
    )
    # a seed keeps training past its own budget only until the slowest seed of
    # its group reaches one. What it records there is not part of the run: its
    # trajectory up to the budget is the same either way, so dropping these rows
    # is what stopping the seed at the budget would have left behind
    return frame[frame.primitive_step <= cell.args.budget]


class Results:
    """One cell's store: episodes appended per chunk, metadata written at the end."""

    def __init__(self, cell: Cell) -> None:
        self.cell = cell
        self.episodes = 0
        self.diagnostics: Dict[int, List[Dict[str, object]]] = {}
        cell.directory.mkdir(parents=True, exist_ok=True)

    def append(self, seeds: Sequence[int], logs: Dict[str, object]) -> None:
        """Write a chunk's episodes and keep its per-update diagnostics."""
        import numpy as np

        path = self.cell.directory / "episodes.csv"
        frame = episode_frame(self.cell, seeds, logs)
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)
        self.episodes += len(frame)
        for position, seed in enumerate(seeds):
            # Store only data with shape (num_seeds, updates) or (num_seeds, updates, num_options);
            # skip arrays that have more dimensions (per-step or render data).
   
            self.diagnostics.setdefault(seed, []).append(
                {
                    key: np.asarray(value)[position]
                    for key, value in logs.items()
                    if np.ndim(value) in (2, 3)
                }
            )

    def finish(self, built: "BuiltCell", elapsed: float) -> int:
        """Write `logs.npz` and the `meta.json` that marks the cell complete."""
        import numpy as np

        # a key per seed rather than one array stacked over them: under
        # `--no-vmap` each seed loops to its own chunk count and the stack would
        # be ragged
        stacked = {
            f"{key}/seed{seed}": np.concatenate(
                [chunk[key] for chunk in self.diagnostics[seed]]
            )
            for seed in self.cell.seeds
            for key in self.diagnostics[seed][0]
        }
        np.savez(self.cell.directory / "logs.npz", **stacked)
        # clipped, since a seed's last updates are the overrun `episode_frame`
        # drops rather than steps the cell spent
        frames = sum(
            min(int(stacked[f"iter/frames/seed{seed}"][-1]), self.cell.args.budget)
            for seed in self.cell.seeds
        )
        meta = {
            "cell": self.cell.name,
            "group": self.cell.group,
            "seeds": list(self.cell.seeds),
            "args": dataclasses.asdict(self.cell.args),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 1),
            "primitive_steps": frames,
            "episodes": self.episodes,
            "n_generated_options": built.n_generated_options,
            "max_option_len": built.spec.horizon,
            "mean_option_len": built.mean_option_len,
            "nominal_option_len": built.nominal_option_len,
            "duration_stats": built.duration_stats,
            "missing_interactions": built.missing_interactions,
            "primitives": built.primitive_names,
            "options": {
                name: list(row) for name, row in zip(built.table_names, built.table)
            },
        }
        (self.cell.directory / "meta.json").write_text(json.dumps(meta, indent=2))
        return frames


def check_finite(logs: Dict[str, object], seeds: Sequence[int]) -> None:
    """Raise if any seed's loss went non-finite during a chunk."""
    import numpy as np

    bad = ~np.isfinite(np.asarray(logs["loss/total_loss"]))
    if not bad.any():
        return
    seed_at, update_at = np.nonzero(bad)
    raise RuntimeError(
        f"loss went non-finite: seed {seeds[int(seed_at[0])]} at update "
        f"{int(update_at[0])} of this chunk"
    )


def report(chunk: int, chunks: int, logs: Dict[str, object]) -> None:
    """Print one progress line summarising a chunk across its seeds."""
    import numpy as np

    done = np.asarray(logs["done_mask"])
    returns = np.asarray(logs["returns"])
    finished = int(done.sum())
    mean = float(returns[done].mean()) if finished else float("nan")

    def average(key: str) -> float:
        return float(np.asarray(logs[key]).mean())

    print(
        f"  chunk {chunk}/~{chunks} "
        f"frames={int(np.asarray(logs['iter/frames'])[:, -1].mean())} "
        f"episodes={finished} return={mean:.3f} "
        f"steps={average('option/steps_mean'):.2f} "
        f"available={average('option/available_frac'):.3f} "
        f"loss={average('loss/total_loss'):.4f}",
        flush=True,
    )


def run_cell(cell: Cell, vmap_seeds: bool = True) -> None:
    """Build, train and record one cell, removing its directory if it fails."""
    from train import UPDATES_PER_CHUNK, build_cell, make_agent, train_cell

    built = build_cell(cell.args)
    agent = make_agent(cell.args, built)
    # an estimate, since the run ends on the budget in primitive steps and the
    # duration a policy draws drifts away from the one measured here
    updates = agent.estimated_updates
    chunks = max(updates // UPDATES_PER_CHUNK, 1)
    if not vmap_seeds:
        chunks *= len(cell.seeds)

    print(
        f"{cell.name}\n"
        f"  seeds {list(cell.seeds)}, {len(built.table)} actions, "
        f"{built.mean_option_len:.2f} primitive steps per option measured "
        f"({built.nominal_option_len:.2f} nominal), at most {built.spec.horizon}; "
        f"{cell.args.budget} primitive steps, about {updates} updates "
        f"in {chunks} chunks",
        flush=True,
    )
    if built.missing_interactions:
        print(
            f"  WARNING: no action performs "
            f"{', '.join(built.missing_interactions)}; any task needing one is "
            f"unreachable for this cell",
            flush=True,
        )

    results = Results(cell)
    counter = itertools.count(1)

    def on_chunk(seeds: Sequence[int], logs: Dict[str, object]) -> None:
        check_finite(logs, seeds)
        results.append(seeds, logs)
        report(next(counter), chunks, logs)

    try:
        elapsed = train_cell(cell.args, built, cell.seeds, on_chunk, vmap_seeds)
    except Exception:
        shutil.rmtree(cell.directory, ignore_errors=True)
        raise
    frames = results.finish(built, elapsed)
    print(
        f"  done: {frames} primitive steps, {elapsed:.1f}s, "
        f"{frames / elapsed:.0f} primitive steps/s, {results.episodes} episodes",
        flush=True,
    )


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="exp1", choices=sorted(SWEEPS))
    parser.add_argument("--cell", action="append", help="run only these cells")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-vmap",
        action="store_true",
        help="run the seeds one at a time, so a failure names a seed",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the selected cells, returning a process exit status."""
    options = parse_arguments(argv)
    configure_environment()

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
    print("Checking option catalogues...")
    assert_catalogues(cells)

    failures: List[Cell] = []
    for cell in cells:
        try:
            run_cell(cell, vmap_seeds=not options.no_vmap)
        except Exception:
            traceback.print_exc()
            failures.append(cell)

    print(f"\n{len(cells) - len(failures)}/{len(cells)} cells complete.")
    for cell in failures:
        print(
            f"  failed: {cell.name}\n"
            f"    {sys.executable} {HERE / 'main.py'} --sweep {options.sweep} "
            f"--cell {cell.name} --no-vmap"
        )
    if len(failures) < len(cells):
        # every cell of an invocation shares one group, and the trailing `*` is
        # what reaches the cells inside it: a pattern matches `{group}/{cell}`
        print(f"  plot it:\n    python plot.py --cells '{cells[0].group}*'")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
