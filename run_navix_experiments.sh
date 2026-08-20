#!/usr/bin/env bash
# Runs the remaining Navix investigations: option-count sweep, harder
# environment, longer baseline.
#
# Usage: ./run_navix_experiments.sh
#
# Per-run CSVs are written by train.py to navix/results/<run_name>/.
# Stdout for each run goes to logs/<name>.log here.

set -uo pipefail  # no -e: one failed run should not kill the rest

export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache

cd "$(dirname "$0")"
LOGDIR="logs"
mkdir -p "$LOGDIR"

SEEDS=(1 2 3)
MAX_PARALLEL=2   # bump to 3 if the first batch completes without OOM

run() {
  local desc="$1"; shift
  local name
  name="$(date +%m%d_%H%M%S)_$(echo "$desc" | tr ' /=' '___')"
  echo "=== $desc ==="
  python navix/train.py "$@" > "$LOGDIR/${name}.log" 2>&1 &
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n; done
}

# ---------------------------------------------------------------------------
# 1. Option-count sweep, both families
#    64/random seeds 1-3 and 64/grammar seed 1 are already run; skipped here.
# ---------------------------------------------------------------------------
SWEEP_N=(8 16 32 64 128)

for n in "${SWEEP_N[@]}"; do
  for s in "${SEEDS[@]}"; do
    if [[ "$n" != "64" ]]; then
      run "sweep random n=$n seed=$s" \
        --action-space option --option-family random \
        --n-options "$n" --max-forward 4 --seed "$s" \
        --tag "sweep-random-n$n"
    fi
  done
done

for n in "${SWEEP_N[@]}"; do
  for s in "${SEEDS[@]}"; do
    if [[ "$n" == "64" && "$s" == "1" ]]; then
      continue
    fi
    run "sweep grammar n=$n seed=$s" \
      --action-space option --option-family grammar \
      --n-options "$n" --max-forward 4 --seed "$s" \
      --tag "sweep-grammar-n$n"
  done
done

wait  # let the sweep finish before switching environments

# ---------------------------------------------------------------------------
# 2. Harder / longer-horizon environment
#    Confirm this env id is registered before running the full set.
# ---------------------------------------------------------------------------
HARD_ENV="Navix-DoorKey-16x16-v0"

for s in "${SEEDS[@]}"; do
  run "hard action seed=$s" \
    --env-id "$HARD_ENV" --action-space action --seed "$s" \
    --tag "hard-action"
done

for s in "${SEEDS[@]}"; do
  run "hard option-random seed=$s" \
    --env-id "$HARD_ENV" --action-space option --option-family random \
    --n-options 64 --max-forward 4 --seed "$s" \
    --tag "hard-option-random"
done

for s in "${SEEDS[@]}"; do
  run "hard option-grammar seed=$s" \
    --env-id "$HARD_ENV" --action-space option --option-family grammar \
    --n-options 64 --max-forward 4 --seed "$s" \
    --tag "hard-option-grammar"
done

wait

# ---------------------------------------------------------------------------
# 3. Longer baseline runs
#    Run the 3M set first. Only run 10M if the baseline is still climbing.
# ---------------------------------------------------------------------------
for s in "${SEEDS[@]}"; do
  run "long-baseline 3M seed=$s" \
    --action-space action --budget 3000000 --seed "$s" \
    --tag "long-baseline-3M"
done

wait

# Uncomment once the 3M curves have been inspected.
# for s in "${SEEDS[@]}"; do
#   run "long-baseline 10M seed=$s" \
#     --action-space action --budget 10000000 --seed "$s" \
#     --tag "long-baseline-10M"
# done
# wait

echo "All runs complete. CSVs in navix/results/<run_name>/episodes.csv"