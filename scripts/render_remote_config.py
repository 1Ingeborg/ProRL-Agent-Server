#!/usr/bin/env python3
import json,os,time
from pathlib import Path
import yaml
ROOT=Path(os.environ.get("POLAR_HOME",str(Path(__file__).resolve().parents[1])))
def main():
    source=json.loads((ROOT/"datasets/slime/manifest.json").read_text())
    cfg={"polar_rollout_url":"http://127.0.0.1:18080","polar_gateway_url":"http://127.0.0.1:18100",
         "polar_reward_key":"score","polar_max_async_level":int(os.environ.get("POLAR_MAX_ASYNC_LEVEL","1")),"polar_request_timeout":7500,
         "polar_callback_host":"127.0.0.1","polar_min_complete_accept_fraction":float(os.environ.get("POLAR_MIN_COMPLETE_ACCEPT_FRACTION","1.0")),
         "polar_run_id":os.environ.get("POLAR_RUN_ID") or source.get("run_id") or time.strftime("%Y%m%dT%H%M%S"),
         "polar_task_id_template":"smith-{args.polar_run_id}-{rollout_id}-{sample.group_index}",
         "polar_task_template":{"timeout_seconds":7200,
             "runtime":"{sample.metadata.runtime}","agent":"{sample.metadata.agent}",
             "builder":"{sample.metadata.builder}","evaluator":"{sample.metadata.evaluator}",
             "metadata":"{sample.metadata.task_metadata}"}}
    target=ROOT/"config/remote-slime.yaml";target.write_text(yaml.safe_dump(cfg,sort_keys=False))
    print(target)
if __name__=="__main__":main()
