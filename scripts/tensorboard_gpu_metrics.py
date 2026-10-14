#!/usr/bin/env python3
import argparse, subprocess, time
from torch.utils.tensorboard import SummaryWriter
p=argparse.ArgumentParser(); p.add_argument("--logdir",required=True); p.add_argument("--run-id",required=True); a=p.parse_args()
w=SummaryWriter(a.logdir,flush_secs=1)
w.add_text("run/id",a.run_id,0); w.flush()
try:
 while True:
  raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],text=True)
  step=int(time.time())
  for line in raw.splitlines():
   idx,used,total,util=[int(x.strip()) for x in line.split(",")]
   w.add_scalar(f"gpu/{idx}/memory_used_mib",used,step)
   w.add_scalar(f"gpu/{idx}/memory_total_mib",total,step)
   w.add_scalar(f"gpu/{idx}/utilization_percent",util,step)
  w.flush(); time.sleep(1)
finally:
 w.flush(); w.close()
