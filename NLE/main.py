"""Runner for the NLE experiment matrix.

One process is one seed of one cell: PyTorch cannot vmap seeds. `ppo.py` owns
the cell; this file owns the group directory, the launch pool, and the preflight.
`batch_size` and `minibatch_size` are resolved inside `ppo.py`.

Torch, gymnasium and NLE are imported inside the functions that need them, so
`--dry-run` prints the matrix without paying for them.
"""

import argparse
import dataclasses
import itertools
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import IO, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"
GROUP_STAMP = time.strftime("%m%d-%H%M%S")

Condition = Literal["action", "option", "both"]
Discount = Literal["decision", "primitive"]
Family = Literal["grammar", "grammar_depth", "random"]

DEPTH_FAMILY: Family = "grammar_depth"
DEPTH_FAMILY_TAG = "exp3"
"""The only sweep allowed to name `grammar_depth`, so that `grammar` denotes one
catalogue in every figure. Asserted by test_only_exp3_names_the_depth_prior."""

FULL_CATALOGUE = 227
"""Rows `NetHackChallenge-v0`'s action set admits, so exp2's last n is the whole
catalogue and its top point is not a family-dependent draw. `preflight` checks it
against the realised size rather than trusting it."""

DEFAULT_PER_GPU = 3
"""Concurrent trainers per GPU. Three measured about 1,030 primitive steps per
second each against 2,230 for one alone, so the bottleneck is shared and a
fourth is not assumed to buy anything."""

LAUNCH_STAGGER_SECONDS = 20.0
"""Delay between launches. Six simultaneous starts fail inside NLE's dlopen with
"failed to map segment from shared object"; twenty seconds apart mostly do not."""

POLL_SECONDS = 5.0
STATUS_EVERY_SECONDS = 600.0

_SPS_RE = re.compile(r"^SPS: (\d+)\s*$", re.MULTILINE)
_FRAMES_RE = re.compile(r"^frames=(\d+),", re.MULTILINE)

PILOT_SEEDS: Tuple[int, ...] = tuple(range(3))
"""The working default. Enough to see whether a condition separates at all, not
enough to claim a difference: plot.py's MIN_SEEDS_FOR_IQM is 8, so below it an
interval is the observed range of the seeds rather than a bootstrap, and any
result read off one has to say so."""

ANALYSIS_SEEDS: Tuple[int, ...] = tuple(range(10))
"""Final runs, above plot.py's MIN_SEEDS_FOR_IQM."""


@dataclass(frozen=True)
class CellArgs:
    """Everything that defines a cell except its seeds."""

    env_id: str = "NetHackChallenge-v0"
    condition: Condition = "action"
    n_options: int = 64
    option_family: Family = "grammar"
    option_seed: int = 0
    reward_delay: int = 0
    discount: Discount = "decision"
    budget: int = 10_000_000
    max_episode_steps: int = 100_000
    clip_reward: bool = True
    gamma: float = 0.999
    num_envs: int = 16
    num_steps: int = 128
    trace_flush_iterations: int = 64
    """rollout iterations per parquet row group; matches `ppo.Args`"""
    tag: str = ""


CHECKPOINT_KEEP_ALL_TAG = "exp1"
"""The sweep whose cells keep every checkpoint, all three conditions of it, since
the headline comparison is the one whose intermediate policies get inspected.
Every other cell keeps three. Not a `CellArgs` field: it selects a cell rather
than describing one, and adding it there would put it in the cell name and in
every `meta.json`."""

_DEFAULT_CELL = CellArgs()
_NAMED_IF_NONDEFAULT = (
    "budget",
    "max_episode_steps",
    "gamma",
    "discount",
    "reward_delay",
)
_STRING_TUPLE_FIELDS = ("conditions", "families", "discounts")


@dataclass(frozen=True)
class Sweep:
    """The arguments a matrix holds fixed, and the axes it varies over them."""

    base: CellArgs = dataclasses.field(default_factory=CellArgs)
    conditions: Tuple[Condition, ...] = ("action", "option", "both")
    families: Tuple[Family, ...] = ("grammar",)
    n_options: Tuple[int, ...] = (64,)
    option_seeds: Tuple[int, ...] = (0,)
    """draws of the `random` catalogue; both grammar priors ignore the seed"""
    reward_delays: Tuple[int, ...] = ()
    """empty uses `base.reward_delay`, so a sweep that does not name this stays one cell"""
    discounts: Tuple[Discount, ...] = ()
    seeds: Tuple[int, ...] = PILOT_SEEDS

    def __post_init__(self) -> None:
        """Reject a bare string where a tuple of strings is meant."""
        for field_name in _STRING_TUPLE_FIELDS:
            value = getattr(self, field_name)
            # a string is iterable, so `expand` would silently take its
            # characters for the axis
            assert not isinstance(value, str), (
                f"{field_name}={value!r} is a string, not a tuple: a "
                "single-element tuple needs its trailing comma"
            )


SWEEPS: Dict[str, Sweep] = {
        # host gate: does anything solve staircase at all
    "gate0": Sweep(
        CellArgs(
            env_id="DelayedStaircase-v0",
            max_episode_steps=5_000,
            budget=5_000_000,
            tag="gate0",
        ),
    ),
    "exp1": Sweep(CellArgs(tag="exp1")),
    "exp2": Sweep(
        CellArgs(tag="exp2"),
        n_options=(8, 16, 32, 64, 128, FULL_CATALOGUE),
    ),
    # n=64, not the full catalogue: at 227 every family returns the whole thing
    # and the three coincide
    "exp3": Sweep(
        CellArgs(tag="exp3"),
        conditions=("option",),
        families=("grammar", DEPTH_FAMILY, "random"),
        option_seeds=(0, 1, 2, 3, 4),
    ),
    # one sweep, so both discount arms land in one run group: plot.py slices its
    # inputs per group and a figure panelled by `discount` needs to see both
    "exp4": Sweep(
        CellArgs(
            env_id="DelayedStaircase-v0",
            max_episode_steps=5_000,
            tag="exp4",
        ),
        conditions=("action", "option"),
        reward_delays=(0, 8, 16, 32, 64),
        discounts=("decision", "primitive"),
    ),
    "smoke": Sweep(
        CellArgs(budget=50_000, max_episode_steps=200, tag="smoke"),
        n_options=(8,),
        seeds=(0,),
    ),
}


@dataclass(frozen=True)
class Cell:
    """One point of the matrix, named without its seeds, inside a timestamped group."""

    args: CellArgs
    seeds: Tuple[int, ...]

    @property
    def name(self) -> str:
        parts = [self.args.env_id, self.args.condition]
        if self.args.condition != "action":
            parts += [
                self.args.option_family,
                f"n{self.args.n_options}",
                f"os{self.args.option_seed}",
            ]
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
    def group_directory(self) -> pathlib.Path:
        return RUNS / self.group

    @property
    def directory(self) -> pathlib.Path:
        return self.group_directory / self.name


@dataclass(frozen=True)
class Job:
    """One seed of one cell: the unit the pool schedules."""

    cell: Cell
    seed: int


@dataclass
class Running:
    """A launched job and the handles the pool has to reclaim."""

    process: "subprocess.Popen[bytes]"
    job: Job
    gpu: int
    log: IO[str]


def expand(sweep: Sweep) -> Iterator[Cell]:
    """Every cell of a sweep: `action` once, the rest per family, count and draw.

    An empty `reward_delays` or `discounts` uses the value on `base`. The
    `action` condition takes only the first discount mode, being invariant to it.
    """
    delays = sweep.reward_delays or (sweep.base.reward_delay,)
    modes = sweep.discounts or (sweep.base.discount,)
    for condition in sweep.conditions:
        if condition == "action":
            variants: List[Dict[str, object]] = [{}]
        else:
            variants = []
            for family in sweep.families:
                # both grammar priors are a function of n alone, so drawing one
                # per option seed would repeat the same cell
                draws = sweep.option_seeds if family == "random" else (0,)
                variants += [
                    {
                        "option_family": family,
                        "n_options": count,
                        "option_seed": option_seed,
                    }
                    for count, option_seed in itertools.product(sweep.n_options, draws)
                ]
        # ppo.py raises gamma to the decision's primitive steps, which is 1 for a
        # primitive, so an action cell computes the same discount in either mode
        cell_modes = modes[:1] if condition == "action" else modes
        for delay, mode, variant in itertools.product(delays, cell_modes, variants):
            yield Cell(
                args=dataclasses.replace(
                    sweep.base,
                    condition=condition,
                    reward_delay=delay,
                    discount=mode,
                    **variant,
                ),
                seeds=sweep.seeds,
            )


def select(cells: Sequence[Cell], names: Optional[Sequence[str]]) -> List[Cell]:
    """The cells `--cell` named, or all of them."""
    if not names:
        return list(cells)
    return [cell for cell in cells if cell.name in names]


def resolve_gpus(requested: Optional[str]) -> List[int]:
    """The device indices to pin to, defaulting to every device torch sees."""
    import torch

    if requested:
        return [int(part) for part in requested.split(",")]
    return list(range(torch.cuda.device_count()))


def assert_gpu(gpus: Sequence[int]) -> None:
    """Abort unless every requested device exists, and report what is on them."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no CUDA device. NLE training is not configured for CPU.")
    if not gpus:
        raise SystemExit("ABORT: --gpus selected no devices.")
    count = torch.cuda.device_count()
    for gpu in gpus:
        if not 0 <= gpu < count:
            raise SystemExit(f"ABORT: --gpus names device {gpu} but torch sees {count}.")
        free, total = torch.cuda.mem_get_info(gpu)
        # memory in use is a heuristic for contention and not a measure of it: a
        # process can hold memory and no SMs, or SMs and little memory. A GPU
        # already busy measured 487 primitive steps per second against 2,230
        print(
            f"GPU {gpu}: {(total - free) / 2**30:.1f} of {total / 2**30:.1f} GiB in use",
            flush=True,
        )


def preflight(cells: Sequence[Cell]) -> None:
    """Print each option cell's catalogue, and abort if one asks for too many rows.

    The interacting count is printed because a catalogue of pure movement cannot
    solve anything, and that is invisible in a learning curve until the curve is
    already flat.
    """
    from envs import make_nle
    from options import GROUP_MOVE, _catalogue, catalogue_digest, select_options

    action_sets: Dict[str, List[object]] = {}
    print("Checking option catalogues...", flush=True)
    for cell in cells:
        if cell.args.condition == "action":
            continue
        env_id = cell.args.env_id
        if env_id not in action_sets:
            env = make_nle(env_id, cell.args.max_episode_steps)
            action_sets[env_id] = list(env.unwrapped.actions)
            env.close()
        rows = _catalogue(action_sets[env_id])
        if cell.args.n_options == FULL_CATALOGUE and len(rows) != FULL_CATALOGUE:
            raise SystemExit(
                f"ABORT: {cell.name} asks for FULL_CATALOGUE={FULL_CATALOGUE} rows "
                f"but {env_id} admits {len(rows)}, so its top point is a prefix and "
                "no longer the whole catalogue. Update FULL_CATALOGUE."
            )
        chosen = select_options(
            rows, cell.args.n_options, cell.args.option_family, cell.args.option_seed
        )
        interacting = sum(1 for row in chosen if row.group != GROUP_MOVE)
        print(
            f"  {cell.name}\n"
            f"    {len(chosen)} of {len(rows)} rows, {interacting} interacting, "
            f"digest {catalogue_digest(chosen)[:12]}",
            flush=True,
        )


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
        "--n-options",
        str(args.n_options),
        "--option-family",
        args.option_family,
        "--option-seed",
        str(args.option_seed),
        "--reward-delay",
        str(args.reward_delay),
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
        "--checkpoint-keep",
        "all" if args.tag == CHECKPOINT_KEEP_ALL_TAG else "endpoints",
        "--trace-flush-iterations",
        str(args.trace_flush_iterations),
    ]


def seed_log(job: Job) -> pathlib.Path:
    return job.cell.directory / f"seed{job.seed}.log"


def launch(job: Job, gpu: int) -> Running:
    """Start one seed pinned to one GPU, its output going to its own log.

    Pinned by `CUDA_VISIBLE_DEVICES` rather than by a torch device index, so the
    child sees exactly one device and needs no argument for it. Without this
    every cell defaults to device 0 and contends there.
    """
    job.cell.directory.mkdir(parents=True, exist_ok=True)
    log = seed_log(job).open("w")
    process = subprocess.Popen(
        ppo_command(job.cell, job.seed),
        cwd=str(HERE),
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            # a pipe or a file makes python block-buffer stdout, so without this
            # a seed's log stays empty until the process exits, which on a 10M
            # frame run is hours after the point of reading it
            "PYTHONUNBUFFERED": "1",
        },
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return Running(process=process, job=job, gpu=gpu, log=log)


def log_progress(path: pathlib.Path) -> Tuple[Optional[int], Optional[int]]:
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    frames, sps = _FRAMES_RE.findall(text), _SPS_RE.findall(text)
    return (
        int(frames[-1]) if frames else None,
        int(sps[-1]) if sps else None,
    )


def remaining_seconds(
    budget: int, frames: Optional[int], sps: Optional[int]
) -> Optional[float]:
    if frames is None or sps is None or sps <= 0:
        return None
    return max(budget - frames, 0) / sps


def pool_eta(
    running_remaining: Sequence[Optional[float]],
    pending_durations: Sequence[Optional[float]],
    capacity: int,
) -> Optional[float]:
    if None in running_remaining or None in pending_durations:
        return None
    slots = sorted(t for t in running_remaining if t is not None)
    slots.extend([0.0] * max(capacity - len(slots), 0))
    for duration in pending_durations:
        assert duration is not None
        slots[0] += duration
        slots.sort()
    return max(slots)


def print_status(
    n_jobs: int,
    running: Sequence[Running],
    pending: Sequence[Job],
    capacity: int,
    rates: Dict[Condition, int],
) -> None:
    n_done = n_jobs - len(pending) - len(running)
    line = f"{n_done}/{n_jobs} done  {len(running)} run  {len(pending)} queued"
    if running or pending:
        eta = pool_eta(
            [
                remaining_seconds(e.job.cell.args.budget, *log_progress(seed_log(e.job)))
                for e in running
            ],
            [
                remaining_seconds(j.cell.args.budget, 0, rates.get(j.cell.args.condition))
                for j in pending
            ],
            capacity,
        )
        if eta is None:
            line += "  eta —"
        elif eta >= 3600:
            line += f"  eta {eta / 3600:.1f}h"
        else:
            line += f"  eta {int(round(eta / 60.0))}m"
    print(f"\n{line}", flush=True)


def run_pool(
    jobs: Sequence[Job], gpus: Sequence[int], per_gpu: int, stagger: float
) -> List[Job]:
    """Run every job, at most `per_gpu` on each GPU, and return the ones that failed.

    Sleeps `stagger` after every launch, the first wave included: a burst of
    simultaneous starts is what fails, and the first wave is a burst.
    """
    pending = list(jobs)
    running: List[Running] = []
    load = {gpu: 0 for gpu in gpus}
    failures: List[Job] = []
    capacity = per_gpu * len(gpus)
    n_jobs = len(jobs)
    last_status = time.time()
    rates: Dict[Condition, int] = {}

    def note_sps(job: Job) -> None:
        sps = log_progress(seed_log(job))[1]
        if sps is not None:
            rates[job.cell.args.condition] = sps

    while pending or running:
        while pending and len(running) < capacity:
            # least loaded, so a GPU whose cells finished early is refilled
            # first; below capacity, some GPU is always under `per_gpu`
            gpu = min(gpus, key=lambda index: (load[index], index))
            job = pending.pop(0)
            load[gpu] += 1
            running.append(launch(job, gpu))
            print(
                f"  start  gpu{gpu} {job.cell.name} seed {job.seed} "
                f"({len(pending)} queued)",
                flush=True,
            )
            time.sleep(stagger)

        done = [entry for entry in running if entry.process.poll() is not None]
        for entry in done:
            note_sps(entry.job)
            running.remove(entry)
            entry.log.close()
            load[entry.gpu] -= 1
            code = entry.process.returncode
            print(
                f"  finish gpu{entry.gpu} {entry.job.cell.name} "
                f"seed {entry.job.seed} "
                f"({'ok' if code == 0 else f'exit {code}'})",
                flush=True,
            )
            if code != 0:
                failures.append(entry.job)
        now = time.time()
        if done or now - last_status >= STATUS_EVERY_SECONDS:
            for entry in running:
                note_sps(entry.job)
            print_status(n_jobs, running, pending, capacity, rates)
            last_status = now
        if not done:
            time.sleep(POLL_SECONDS)

    return failures


def finish_cell(cell: Cell, elapsed: float) -> None:
    """Write the `meta.json` that marks the cell complete.

    The cell-level aggregate, reached only when every seed succeeded. It is not
    the provenance of record: `ppo.py` writes `provenance_seed{N}.json` per seed
    unconditionally, which is the only thing a partially-failed cell leaves.
    """
    import nle

    from ppo import LOG_SCHEMA_VERSION, git_sha

    meta = {
        "cell": cell.name,
        "group": cell.group,
        "seeds": list(cell.seeds),
        "args": dataclasses.asdict(cell.args),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        # of the group, not the cell: the pool interleaves seeds across cells, so
        # a cell has no wall-clock span of its own
        "group_elapsed_seconds": round(elapsed, 1),
        "git_sha": git_sha(),
        "nle_version": nle.__version__,
        "host": socket.gethostname(),
        "log_schema_version": LOG_SCHEMA_VERSION,
    }
    (cell.directory / "meta.json").write_text(json.dumps(meta, indent=2))


def write_failures(group_directory: pathlib.Path, failures: Sequence[Job]) -> None:
    """Record every failed run and the exact command that reruns it.

    The cell directory is left alone. Deleting it would take the output of the
    seeds that succeeded, and under the pool it could take a seed still running.
    """
    group_directory.mkdir(parents=True, exist_ok=True)
    (group_directory / "failures.json").write_text(
        json.dumps(
            [
                {
                    "cell": job.cell.name,
                    "seed": job.seed,
                    "log": str(seed_log(job)),
                    "command": ppo_command(job.cell, job.seed),
                }
                for job in failures
            ],
            indent=2,
        )
    )


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="exp1", choices=sorted(SWEEPS))
    parser.add_argument("--cell", action="append", help="run only these cells")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--seeds",
        type=int,
        default=len(PILOT_SEEDS),
        help=f"seeds per cell, as range(n). {len(PILOT_SEEDS)} is a pilot; "
        f"{len(ANALYSIS_SEEDS)} is what an interval needs",
    )
    parser.add_argument(
        "--gpus", help="comma-separated device indices, default every visible device"
    )
    parser.add_argument("--per-gpu", type=int, default=DEFAULT_PER_GPU)
    parser.add_argument("--stagger", type=float, default=LAUNCH_STAGGER_SECONDS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the selected cells, returning a process exit status."""
    options = parse_arguments(argv)
    sweep = dataclasses.replace(
        SWEEPS[options.sweep], seeds=tuple(range(options.seeds))
    )
    cells = select(list(expand(sweep)), options.cell)
    if not cells:
        print("no cells match those names")
        return 1

    if options.dry_run:
        print(f"{len(cells)} cells in sweep {options.sweep!r}:")
        for cell in cells:
            print(f"  {cell.name} seeds={list(cell.seeds)}")
        return 0

    gpus = resolve_gpus(options.gpus)
    assert_gpu(gpus)
    preflight(cells)

    jobs = [Job(cell, seed) for cell in cells for seed in cell.seeds]
    if options.seeds < len(ANALYSIS_SEEDS):
        print(
            f"\nPILOT: {options.seeds} seeds is below plot.py's MIN_SEEDS_FOR_IQM, "
            "so intervals will be median-and-range, not bootstrapped.",
            flush=True,
        )
    print(
        f"{len(jobs)} runs over {len(cells)} cells on GPUs {gpus}, "
        f"{options.per_gpu} per GPU, {options.stagger:.0f}s apart\n",
        flush=True,
    )

    start = time.time()
    failures = run_pool(jobs, gpus, options.per_gpu, options.stagger)
    elapsed = time.time() - start

    failed = {job.cell.name for job in failures}
    for cell in cells:
        if cell.name not in failed:
            finish_cell(cell, elapsed)
    if failures:
        write_failures(cells[0].group_directory, failures)

    print(
        f"\n{len(cells) - len(failed)}/{len(cells)} cells complete "
        f"in {elapsed / 3600:.2f}h."
    )
    for job in failures:
        print(f"  failed: {job.cell.name} seed {job.seed}")
    if failures:
        print(f"  see {cells[0].group_directory / 'failures.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
