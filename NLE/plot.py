import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = pathlib.Path("results")


def label_for(run):
    parts = run.split("__")
    condition = parts[1] if len(parts) > 1 else run
    tag = parts[-1] if len(parts) > 4 else ""
    return f"{condition} {tag}".strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--x", default="primitive_step",
                   choices=["primitive_step", "decision_step"])
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--out", default="comparison.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in args.runs:
        df = pd.read_csv(RESULTS / run / "episodes.csv")
        smoothed = df["episodic_return"].rolling(args.window, min_periods=1).mean()
        ax.plot(df[args.x], smoothed, label=label_for(run))

    ax.set_xlabel("primitive environment steps" if args.x == "primitive_step"
                  else "agent decisions")
    ax.set_ylabel(f"episodic return ({args.window}-episode moving average)")
    ax.set_title(args.runs[0].split("__")[0])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()