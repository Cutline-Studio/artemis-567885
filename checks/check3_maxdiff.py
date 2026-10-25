"""Print max |loop - vectorized| for outputs, input grads and parameter grads (check 3 fix).
Usage: python3 checks/check3_maxdiff.py"""
import sys, torch; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checks.test_check3_moe_dispatch import compare, CONFIGS
rows=[]
for cfg in CONFIGS:
  for dt in (torch.float32, torch.float64):
    for tr in (False, True):
      r = compare(*cfg, dt, tr, assert_param_grads=False); rows.append(r["param_grad_max_abs"])
for E in (4,16,64):
  for dt in (torch.float32, torch.float64):
    seeds = [4] if (E==4 and dt==torch.float32) else []
    for s in seeds+[E]:
      r = compare(E,2,1.0,4096,128,512,dt,True,seed=s,assert_param_grads=False)
      print("bench E",E,str(dt),"seed",s,"param_grad_max_abs",r["param_grad_max_abs"],"out",r["out_max_abs"],"xgrad",r["xgrad_max_abs"])
print("small configs max param grad diff:", max(rows))
