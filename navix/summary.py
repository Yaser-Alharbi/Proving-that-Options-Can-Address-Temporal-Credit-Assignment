import argparse
import fnmatch
import json
import pathlib

import pandas as pd

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def summarise(run_dir, tail_frac, min_episodes):
    config = json.loads((run_dir / "config.json").read_text())
    try:
        episodes = pd.read_csv(run_dir / "episodes.csv")
    except pd.errors.EmptyDataError:  # run died before the header was flushed
        episodes = pd.DataFrame(columns=["primitive_step", "episodic_return"])
    is_options = config["action_space"] != "action"

    row = {
        "env": config["env_id"].replace("Navix-", "").replace("-v0", ""),
        "cond": config["action_space"],
        "family": config["option_family"] if is_options else "-",
        "n_act": config["n_actions"],
        "dur": round(config["mean_option_len"], 2),
        "fwd": config["max_forward"] if is_options else 0,
        "disc": config["discount"],
        "budget": config["budget"],
        "seed": config["seed"],
        "tag": config["tag"] or "-",
        "episodes": len(episodes),
        "steps": 0,
        "ret_all": float("nan"),
        "ret_final": float("nan"),
        "len_final": float("nan"),
        "run": run_dir.name,
    }
    if len(episodes) < min_episodes:
        return row

    reached = int(episodes.primitive_step.max())
    tail = episodes[episodes.primitive_step > (1 - tail_frac) * reached]
    solved = tail[tail["terminated"] == 1] if "terminated" in tail else tail[tail.episodic_return > 0]
    row.update(
        steps=reached,
        ret_all=round(episodes.episodic_return.mean(), 3),
        ret_final=round(tail.episodic_return.mean(), 3),
        len_final=round(solved.episodic_length.median(), 1) if len(solved) else float("nan"),
    )
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("patterns", nargs="*", default=["*"],
                   help="run name globs, e.g. '*__20m' (default: every run)")
    p.add_argument("--tail", type=float, default=0.1,
                   help="fraction of the run to average for the final columns")
    p.add_argument("--min-episodes", type=int, default=1)
    p.add_argument("--sort", default="cond,n_act,seed")
    p.add_argument("--csv", default=None, help="also write the table to this path")
    args = p.parse_args()

    runs = [
        d for d in sorted(RESULTS.iterdir())
        if (d / "config.json").exists() and (d / "episodes.csv").exists()
        and any(fnmatch.fnmatch(d.name, pattern) for pattern in args.patterns)
    ]
    if not runs:
        raise SystemExit(f"no runs in {RESULTS} matching {args.patterns}")

    table = pd.DataFrame([summarise(d, args.tail, args.min_episodes) for d in runs])
    keys = [k for k in args.sort.split(",") if k in table.columns]
    table = table.sort_values(keys) if keys else table

    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    print(f"\n{len(table)} runs; ret_final and len_final average the last "
          f"{args.tail:.0%} of each run by primitive step")
    if args.csv:
        pathlib.Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
