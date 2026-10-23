#!/usr/bin/env bash
# Re-run every repo check on CPU and write raw consoles to checks/results/.
# Usage: bash checks/run_all.sh [/path/to/corpus-bench]   (corpus-bench holds runs/seed*/best.pt and corpus/)
set -u
cd "$(dirname "$0")/.."
R=checks/results; mkdir -p $R
BENCH=${1:-/workspace/artemis-corpus-bench}
run() { local out=$1; shift; { echo "\$ $*"; "$@" 2>&1; echo "exit=$?"; } > "$R/$out"; tail -3 "$R/$out"; }
run env_console.txt python3 -c "import sys,torch,numpy,platform,pytest;print('python',sys.version.split()[0],'torch',torch.__version__,'numpy',numpy.__version__,'pytest',pytest.__version__,platform.platform(),'cpus',__import__('os').cpu_count())"
run check1_historical_console.txt python3 -m pytest -v -s checks/test_check1_historical.py
run check2_checkpoints_console.txt python3 -m pytest -v -s checks/test_check2_checkpoints.py
run check2_real_checkpoints_console.txt python3 checks/check2_real_checkpoints.py --runs "$BENCH/runs" --data "$BENCH/corpus"
run check3_moe_dispatch_console.txt python3 -m pytest -v -s checks/test_check3_moe_dispatch.py
run check3_bench_console.txt python3 checks/check3_bench.py
run check4_solver_console.txt python3 -m pytest -v -s checks/test_check4_solver.py
run check4_pinn_train_console.txt python3 checks/check4_pinn_train.py --seconds 180 --seed 567885
run check5_causal_console.txt python3 -m pytest -v -s checks/test_check5_causal.py
run full_suite_console.txt python3 -m pytest -q tests checks
