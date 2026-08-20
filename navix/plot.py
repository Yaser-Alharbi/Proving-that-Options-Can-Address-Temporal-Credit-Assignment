import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
PLOTS = pathlib.Path(__file__).resolve().parent / "plots"


def label_for(run):
    parts = run.split("__")
    condition = parts[1] if len(parts) > 1 else run
    tag = parts[-1] if len(parts) > 4 else ""
    return f"{condition} {tag}".strip()


def default_name(runs, x):
    env = runs[0].split("__")[0]
    conditions = sorted({run.split("__")[1] for run in runs if "__" in run})
    tags = sorted({run.split("__")[-1] for run in runs if len(run.split("__")) > 4})
    parts = [env, "-".join(conditions) or "runs", *tags, x]
    return "__".join(parts) + ".png"


def group_for(run):
    """Everything but the seed and timestamp, so seeds of one setting collapse."""
    parts = run.split("__")
    keep = [p for p in parts if not p.startswith("seed")]
    return "__".join(keep[1:2] + keep[3:]) or run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--x", default="primitive_step",
                   choices=["primitive_step", "decision_step"])
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--group-seeds", action="store_true",
                   help="average runs that differ only by seed")
    p.add_argument("--bins", type=int, default=200)
    p.add_argument("--out", default=None,
                   help="bare filenames land in navix/plots/; a path is used as given")
    args = p.parse_args()

    out = pathlib.Path(args.out or default_name(args.runs, args.x))
    if out.parent == pathlib.Path("."):
        out = PLOTS / out
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = {run: pd.read_csv(RESULTS / run / "episodes.csv") for run in args.runs}
    frames = {run: df for run, df in frames.items() if len(df)}
    if not frames:
        raise SystemExit("no episodes logged in any of those runs")

    x_max = min(df[args.x].max() for df in frames.values())
    print(f"truncating all runs at {args.x}={x_max}")

    curves = {}
    for run, df in frames.items():
        df = df[df[args.x] <= x_max]
        y = df["episodic_return"].rolling(args.window, min_periods=1).mean()
        curves.setdefault(group_for(run) if args.group_seeds else run, []).append(
            (df[args.x].to_numpy(), y.to_numpy())
        )

    grid = np.linspace(0, x_max, args.bins)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, seeds in curves.items():
        stacked = np.stack([np.interp(grid, x, y) for x, y in seeds])  # (S, bins)
        mean = stacked.mean(axis=0)
        ax.plot(grid, mean, label=f"{label_for(name)} (n={len(seeds)})")
        if len(seeds) > 1:
            err = stacked.std(axis=0) / np.sqrt(len(seeds))
            ax.fill_between(grid, mean - err, mean + err, alpha=0.2)

    ax.set_xlabel("primitive environment steps" if args.x == "primitive_step"
                  else "agent decisions")
    ax.set_ylabel(f"episodic return ({args.window}-episode moving average)")
    ax.set_title(args.runs[0].split("__")[0])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
