#!/usr/bin/env bash
# Matched-budget MoE position ablation. Same corpus, data order, steps, batch, context as the
# dense baseline runs (runs/dense/seed{42,1,2}). Two workers, one thread each.
set -euo pipefail
ROOT=/home/claude/language-benchmark
CODE=/home/claude/wt-stage2
COMMIT=$(git -C "$CODE" rev-parse HEAD)
jobs=()
for seed in 42 1 2; do
  for pos in fourier sinusoidal learned none; do
    jobs+=("$pos $seed")
  done
done
run() {
  local pos=$1 seed=$2 out="$ROOT/runs/moe-$1/seed$2"
  mkdir -p "$ROOT/runs/moe-$pos"
  echo "$COMMIT" > "$ROOT/runs/moe-$pos/seed$seed.commit"
  python -I "$CODE/experiments/corpus_benchmark.py" --data "$ROOT/corpus" --output "$out" \
    --model moe --position "$pos" --seed "$seed" --epochs 1 --steps 5000 --threads 1 \
    --capacity-mode causal --capacity-factor 1.25 --aux-weight 0.01 \
    > "$ROOT/runs/moe-$pos/seed$seed.log" 2>&1 && echo "done $pos $seed" || echo "FAILED $pos $seed"
}
export -f run; export ROOT CODE COMMIT
printf '%s\n' "${jobs[@]}" | xargs -P 2 -L 1 bash -c 'run $0 $1'
echo ALL_DONE
