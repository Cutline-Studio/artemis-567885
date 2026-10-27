#!/bin/bash
# Artemis 567885 continued-training session (runs ON the Vast instance; the box only orchestrates).
# Phases: env + tests -> eval before -> 4-arm recipe ablation (10 cycles each, same seeds) -> pick winner
# on held-out MSE -> continue the winner with the supervisor in 10-cycle chunks until the deadline.
set -u
R=$(cat /w/R); DEADLINE=$(( R + 5*3600 + 45*60 ))   # training stops by rent+5h45m; instance dies at rent+6h
O=/w/out; mkdir -p $O; cd /w
left(){ echo $(( DEADLINE - $(date +%s) )); }
log(){ echo "$(date -u +%FT%TZ) $*" >> $O/status.txt; }
log "start left=$(left)"
nvidia-smi > $O/nvidia_smi.txt 2>&1; lscpu > $O/lscpu.txt 2>&1
pip install -q pytest==8.3.3 > $O/pip.txt 2>&1
python -c 'import torch,sys,numpy,pytest,platform;print("torch",torch.__version__,"cuda",torch.version.cuda,"py",sys.version.split()[0],"numpy",numpy.__version__,"pytest",pytest.__version__,platform.platform(),"cpus",__import__("os").cpu_count())' > $O/env.txt 2>&1
cd /w/base && timeout 1200 python -m pytest -q tests checks -p no:cacheprovider > $O/pytest.txt 2>&1; log "pytest rc=$?"
cd /w/base && python evaluate.py --root /w/base --device cpu --out $O/eval_before.json > /dev/null 2>$O/eval_before.err; log "eval_before rc=$?"

mkarm(){ name=$1; shift; rm -rf /w/arm_$name; cp -r /w/base /w/arm_$name
  python - "$name" "$@" <<'PY'
import json, sys
name, kv = sys.argv[1], sys.argv[2:]
p = f"/w/arm_{name}/config.json"; c = json.load(open(p))
for item in kv:
    k, v = item.split("=", 1); c[k] = json.loads(v)
json.dump(c, open(p, "w"), indent=2); open(p, "a").write("\n")
PY
}
COMMON='ema_decay=0.99 eval_extra=true num_threads=4'
mkarm A $COMMON                                                                  # original recipe (caps raised only)
mkarm B $COMMON lr_schedule='"cosine"' warmup_steps=30 lr_min=1e-5 carry_optimizer_on_growth=true
mkarm C $COMMON lr_schedule='"cosine"' warmup_steps=30 lr_min=1e-5 carry_optimizer_on_growth=true batch_size=64
mkarm G $COMMON lr_schedule='"cosine"' warmup_steps=30 lr_min=1e-5 carry_optimizer_on_growth=true batch_size=64 device='"cuda"'
for a in A B C G; do ( cd /w/arm_$a; s=$(date +%s); timeout $(left) python supervisor.py --cycles 10 --timeout 900 > $O/arm_$a.txt 2>&1; echo "arm $a rc=$? secs=$(( $(date +%s)-s ))" >> $O/status.txt ) & done
wait
for a in A B C G; do python /w/base/evaluate.py --root /w/arm_$a --device cpu --out $O/eval_arm_$a.json > /dev/null 2>$O/eval_arm_$a.err; done
python - <<'PY' > $O/winner.json
import json
arms = {}
for a in "ABCG":
    try: arms[a] = json.load(open(f"/w/out/eval_arm_{a}.json"))
    except Exception: pass
def best(r): return min(r[k]["heldout_mse"] for k in ("raw", "ema") if k in r)
w = min(arms, key=lambda a: best(arms[a]))
st = {l.split()[1]: int(l.split("secs=")[1]) for l in open("/w/out/status.txt") if l.startswith("arm ")}
print(json.dumps({"winner": w, "best_heldout_mse": {a: best(r) for a, r in arms.items()}, "arm_secs": st}))
PY
W=$(python -c 'import json;print(json.load(open("/w/out/winner.json"))["winner"])')
# C (cpu) and G (cuda) are the same recipe; keep whichever device ran faster for the long run
if [ "$W" = C ] || [ "$W" = G ]; then
  W=$(python -c 'import json;s=json.load(open("/w/out/winner.json"))["arm_secs"];print("G" if s.get("G",9e9)<s.get("C",9e9) else "C")'); fi
log "winner=$W"
cp -r /w/arm_$W /w/main
python - <<'PY'
import json; p="/w/main/config.json"; c=json.load(open(p)); c["num_threads"]=16
json.dump(c,open(p,"w"),indent=2); open(p,"a").write("\n")
PY
cd /w/main; last=600; n=0
while [ $(left) -gt $(( last * 3 / 2 + 120 )) ]; do
  s=$(date +%s); timeout $(( $(left) - 60 )) python supervisor.py --cycles 10 --timeout 900 >> $O/main.txt 2>&1; rc=$?
  last=$(( $(date +%s) - s )); n=$((n+1))
  python evaluate.py --root /w/main --device cpu --out $O/eval_main_latest.json > /dev/null 2>>$O/eval_main.err
  python -c 'import json,sys;r=json.load(open("/w/out/eval_main_latest.json"));print(json.dumps({"chunk":int(sys.argv[1]),"cycle":r["cycle"],"arch":r["arch"],"raw":r["raw"]["heldout_mse"],"ema":r.get("ema",{}).get("heldout_mse")}))' $n >> $O/curve.jsonl
  cp runs/checkpoint.pt $O/main_checkpoint.pt.tmp && mv $O/main_checkpoint.pt.tmp $O/main_checkpoint.pt
  log "main chunk=$n rc=$rc secs=$last left=$(left)"
  [ $rc -ne 0 ] && { log "main supervisor failed rc=$rc, stopping"; break; }
done
cp /w/main/config.json $O/main_config.json
python evaluate.py --root /w/main --device cpu --out $O/eval_after.json > /dev/null 2>$O/eval_after.err
log "DONE"; touch $O/DONE
