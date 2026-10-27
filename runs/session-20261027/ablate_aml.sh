#!/bin/bash
# Replicate recipe ablation on Azure ML (NC8as_T4_v3): arms A (original), B (cosine+warmup+Adam carry-over),
# C (B + batch 64), each with data_seed 1 and 2, 10 supervisor cycles from the same cycle-114 checkpoint
# (data_seed 0 was measured on Vast). Plus arm G = C on the T4 GPU (seed 1) for device timing.
set -u
O=$1; mkdir -p $O; SRC=$(pwd); W=/tmp/w; mkdir -p $W
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $O/status.txt; }
nvidia-smi > $O/nvidia_smi.txt 2>&1; lscpu > $O/lscpu.txt 2>&1
pip install -q pytest==8.3.3 pyjwt==2.9.0 > $O/pip.txt 2>&1
python -c 'import torch,sys,numpy,platform,os;print("torch",torch.__version__,"cuda",torch.version.cuda,torch.cuda.is_available(),"py",sys.version.split()[0],"numpy",numpy.__version__,platform.platform(),"cpus",os.cpu_count())' | tee $O/env.txt
cp -r $SRC/base $W/base
mk(){ name=$1; shift; rm -rf $W/$name; cp -r $W/base $W/$name
  python - $W/$name/config.json "$@" <<'PY'
import json, sys
p, kv = sys.argv[1], sys.argv[2:]; c = json.load(open(p))
for item in kv:
    k, v = item.split("=", 1); c[k] = json.loads(v)
json.dump(c, open(p, "w"), indent=2); open(p, "a").write("\n")
PY
}
COMMON='ema_decay=0.99 eval_extra=true num_threads=2'
RB='lr_schedule="cosine" warmup_steps=30 lr_min=1e-5 carry_optimizer_on_growth=true'
for s in 1 2; do
  mk A_s$s $COMMON data_seed=$s
  mk B_s$s $COMMON $RB data_seed=$s
  mk C_s$s $COMMON $RB batch_size=64 data_seed=$s
done
mk G_s1 $COMMON $RB batch_size=64 data_seed=1 device='"cuda"'
run(){ (cd $W/$1; s=$(date +%s); timeout 2400 python supervisor.py --cycles 10 --timeout 900 > $O/$1.txt 2>&1; log "arm $1 rc=$? secs=$(( $(date +%s)-s ))"
        python $W/base/evaluate.py --root $W/$1 --device cpu --out $O/eval_$1.json > /dev/null 2>$O/eval_$1.err) ; }
run A_s1 & run B_s1 & run C_s1 & run G_s1 & wait
run A_s2 & run B_s2 & run C_s2 & wait
log DONE
