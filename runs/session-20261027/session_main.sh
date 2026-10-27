#!/bin/bash
# Artemis 567885 continued training on an Azure ML NC8as_T4_v3 node (the box only orchestrates).
# Continues the measured ablation winner (arm C: cosine lr + warmup, Adam carry-over on growth,
# batch 64, EMA 0.99; caps 16 layers / CF 3.0) from cycle 125 in 10-cycle supervisor chunks.
set -u
O=$1; mkdir -p $O; T0=$(date +%s); DEADLINE=$(( T0 + 5*3600 + 25*60 ))
left(){ echo $(( DEADLINE - $(date +%s) )); }
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $O/status.txt; }
SRC=$(pwd); W=/tmp/w; mkdir -p $W; cp -r $SRC/main $W/main
log "start left=$(left)"
nvidia-smi > $O/nvidia_smi.txt 2>&1; lscpu > $O/lscpu.txt 2>&1
pip install -q pytest==8.3.3 pyjwt==2.9.0 > $O/pip.txt 2>&1
python -c 'import torch,sys,numpy,pytest,platform,os;print("torch",torch.__version__,"cuda",torch.version.cuda,torch.cuda.is_available(),"py",sys.version.split()[0],"numpy",numpy.__version__,"pytest",pytest.__version__,platform.platform(),"cpus",os.cpu_count())' | tee $O/env.txt
cd $W/main && timeout 1200 python -m pytest -q tests checks -p no:cacheprovider > $O/pytest.txt 2>&1; log "pytest rc=$? $(tail -1 $O/pytest.txt)"
python evaluate.py --root $W/main --device cpu --out $O/eval_start.json > /dev/null 2>$O/eval_start.err; log "eval_start rc=$?"
# one timed cycle per device on throwaway copies, keep the faster device for the long run
for d in cpu cuda; do rm -rf $W/t_$d; cp -r $W/main $W/t_$d
  python - $W/t_$d/config.json $d <<'PY'
import json,sys; p,d=sys.argv[1:]; c=json.load(open(p)); c["device"]=d; c["num_threads"]=8
json.dump(c,open(p,"w"),indent=2); open(p,"a").write("\n")
PY
  s=$(date +%s); (cd $W/t_$d && timeout 900 python supervisor.py --cycles 1 --timeout 900 > $O/timing_$d.txt 2>&1); echo $(( $(date +%s)-s )) > $W/secs_$d; log "timing $d secs=$(cat $W/secs_$d)"; done
DEV=$( [ $(cat $W/secs_cuda) -lt $(cat $W/secs_cpu) ] && echo cuda || echo cpu ); log "device=$DEV"
python - $W/main/config.json $DEV <<'PY'
import json,sys; p,d=sys.argv[1:]; c=json.load(open(p)); c["device"]=d; c["num_threads"]=8
json.dump(c,open(p,"w"),indent=2); open(p,"a").write("\n")
PY
cd $W/main; last=600; n=0
while [ $(left) -gt $(( last * 3 / 2 + 120 )) ]; do
  s=$(date +%s); timeout $(( $(left) - 60 )) python supervisor.py --cycles 10 --timeout 900 >> $O/main.txt 2>&1; rc=$?
  last=$(( $(date +%s) - s )); n=$((n+1))
  python evaluate.py --root $W/main --device cpu --out $O/eval_latest.json > /dev/null 2>>$O/eval.err
  python -c 'import json,sys;r=json.load(open(sys.argv[2]));print("CURVE "+json.dumps({"chunk":int(sys.argv[1]),"cycle":r["cycle"],"layers":r["arch"]["num_layers"],"raw":r["raw"]["heldout_mse"],"ema":r.get("ema",{}).get("heldout_mse"),"ood":r["raw"]["ood_mse"]}))' $n $O/eval_latest.json | tee -a $O/curve.jsonl
  cyc=$(python -c 'import json;print(json.load(open("config.json"))["cycle"])')
  mkdir -p $O/ckpt; cp runs/checkpoint.pt $O/ckpt/checkpoint_cycle$cyc.pt; cp config.json $O/ckpt/config_cycle$cyc.json
  cp runs/expand_log.jsonl runs/supervisor_log.jsonl $O/ 2>/dev/null
  log "chunk=$n rc=$rc secs=$last left=$(left) next_cycle=$cyc"
  [ $rc -ne 0 ] && { log "supervisor failed rc=$rc, stopping"; break; }
done
cp config.json $O/final_config.json; cp runs/checkpoint.pt $O/final_checkpoint.pt; cp runs/*.jsonl runs/runtime_errors.log $O/ 2>/dev/null
python evaluate.py --root $W/main --device cpu --out $O/eval_after.json > /dev/null 2>$O/eval_after.err
log "DONE"
