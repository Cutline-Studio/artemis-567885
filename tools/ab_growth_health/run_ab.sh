#!/bin/bash
# Growth-health A/B (runs ON a rented instance; never on the orchestration box).
# Layout prepared by the box: /w/old = expand-loop @ 7be7914 (pre-fix code), /w/new = this branch,
# /w/ck365.pt = draft-release checkpoint_cycle365.pt (sha256 c268feec...), /w/R = rent epoch.
# Arms (same seeds, same recipe = the 2026-10-27 ablation winner "C": cosine+warmup30, lr_min 1e-5,
# Adam carry-over on growth, batch 64, EMA 0.99, eval_extra; caps 16 layers / 16 experts / CF 3.0):
#   F_old / F_new / F_new_ml : fresh init (seed 567885), NF cycles; F_new_ml = new rules but legacy
#                              most-loaded clone init (isolates fix 4)
#   C_old / C_new            : continue the 2026-10-27 Vast run from cycle 365 (growth was still active
#                              there; at cycle 624 every layer is at its cap so neither code can grow)
set -u
NF=${NF:-150}; NC=${NC:-15}
# AS RUN on 2026-10-27: NF was 40, then each F arm was continued in place with
# `expand.py loop --cycles 110` (state is persisted per cycle, so 40+110 = 150 cycles from the same seeds),
# and pytest was re-run after `pip install pyjwt==2.9.0` (checks/test_check1 imports jwt).
R=$(cat /w/R); DEADLINE=$(( R + 47*60 ))
O=/w/out; mkdir -p $O; cd /w
left(){ echo $(( DEADLINE - $(date +%s) )); }
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $O/status.txt; }
log "start left=$(left) NF=$NF NC=$NC"
nvidia-smi > $O/nvidia_smi.txt 2>&1; lscpu > $O/lscpu.txt 2>&1
pip install -q pytest==8.3.3 pyjwt==2.9.0 > $O/pip.txt 2>&1
python -c 'import torch,sys,numpy,pytest,platform,os;print("torch",torch.__version__,"cuda",torch.version.cuda,torch.cuda.is_available(),"py",sys.version.split()[0],"numpy",numpy.__version__,"pytest",pytest.__version__,platform.platform(),"cpus",os.cpu_count())' | tee $O/env.txt
( cd /w/old && timeout 900 python -m pytest -q tests checks -p no:cacheprovider > $O/pytest_old.txt 2>&1; echo "rc=$?" >> $O/pytest_old.txt ) &
( cd /w/new && timeout 900 python -m pytest -q tests checks -p no:cacheprovider > $O/pytest_new.txt 2>&1; echo "rc=$?" >> $O/pytest_new.txt ) &
wait; log "pytest old: $(tail -2 $O/pytest_old.txt | tr '\n' ' ') | new: $(tail -2 $O/pytest_new.txt | tr '\n' ' ')"

GPU=$(python -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")')
RECIPE='{"batch_size":64,"lr_schedule":"cosine","warmup_steps":30,"lr_min":1e-05,"carry_optimizer_on_growth":true,"ema_decay":0.99,"eval_extra":true,"num_threads":4,"caps":{"capacity_factor_max":3.0,"num_experts_max":16,"alpha_max":1.0,"num_layers_max":16}}'
mkfresh(){ name=$1 code=$2 extra=$3; rm -rf /w/arm_$name; mkdir -p /w/arm_$name
  cp /w/$code/*.py /w/arm_$name/; (cd /w/arm_$name && python expand.py init --root . --force > /dev/null)
  python - /w/arm_$name/config.json "$RECIPE" "$extra" <<'PY'
import json, sys
p, rec, extra = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
c = json.load(open(p)); c.update(rec); c["device"] = "cpu"
c["rules"].update(extra.pop("rules", {})); c.update(extra)
json.dump(c, open(p, "w"), indent=2); open(p, "a").write("\n")
PY
}
mkcont(){ name=$1 code=$2; rm -rf /w/arm_$name; mkdir -p /w/arm_$name/runs
  cp /w/$code/*.py /w/arm_$name/; cp /w/ck365.pt /w/arm_$name/runs/checkpoint.pt
  python - /w/arm_$name /w/new/runs/session-20261027/vast-main/final_config.json $GPU <<'PY'
import json, sys, torch
root, src, dev = sys.argv[1:]
c = json.load(open(src)); ck = torch.load(root + "/runs/checkpoint.pt", map_location="cpu", weights_only=False)
h = c["history"][:365]; assert h[-1]["cycle"] == 364
c.update(history=h, cycle=365, pending_mutation=None, num_layers=ck["arch"]["num_layers"],
         num_experts=ck["arch"]["num_experts"], capacity_factor=h[-1]["capacity_factor"], alpha=h[-1]["alpha"],
         device=dev, num_threads=4)
assert ck["arch"]["num_experts"] == h[-1]["num_experts"], (ck["arch"], h[-1]["num_experts"])
json.dump(c, open(root + "/config.json", "w"), indent=2); open(root + "/config.json", "a").write("\n")
PY
}
mkfresh F_old old '{}'; mkfresh F_new new '{}'; mkfresh F_new_ml new '{"rules":{"expert_init":"most_loaded"}}'
mkcont C_old old; mkcont C_new new
# baseline eval of the cycle-365 checkpoint (identical for both continuation arms)
python /w/new/evaluate.py --root /w/arm_C_old --device cpu --out $O/eval_C_start365.json > /dev/null 2>$O/eval_C_start.err
log "arms ready gpu=$GPU"
for a in F_old F_new F_new_ml; do ( cd /w/arm_$a; s=$(date +%s); timeout $(left) python expand.py loop --cycles $NF --root . > $O/loop_$a.txt 2>&1; echo "$(date -u +%FT%TZ) arm $a rc=$? secs=$(( $(date +%s)-s ))" >> $O/status.txt ) & done
for a in C_old C_new; do ( cd /w/arm_$a; s=$(date +%s); timeout $(left) python expand.py loop --cycles $NC --root . > $O/loop_$a.txt 2>&1; echo "$(date -u +%FT%TZ) arm $a rc=$? secs=$(( $(date +%s)-s ))" >> $O/status.txt ) & done
wait
for a in F_old F_new F_new_ml C_old C_new; do
  python /w/new/evaluate.py --root /w/arm_$a --device cpu --out $O/eval_$a.json > /dev/null 2>$O/eval_$a.err
  cp /w/arm_$a/runs/expand_log.jsonl $O/expand_log_$a.jsonl 2>/dev/null
done
python /w/new/tools/ab_growth_health/summarize.py /w $O > $O/summary.json 2> $O/summary.err
log "DONE"
