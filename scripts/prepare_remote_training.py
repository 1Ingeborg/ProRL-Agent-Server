#!/usr/bin/env python3
"""Export Slime prompts referencing execution-host-only native Polar assets."""
import hashlib, json, os, sys
from pathlib import Path
ROOT=Path(os.environ.get("POLAR_HOME",str(Path(__file__).resolve().parents[1])))
sys.path.insert(0,str(ROOT/"adapters"))
from covered_dataset import load_covered
from swe_agent_tasks import build_task, save, select, CONFIG, digest
def main():
    tasks,manifest,_=load_covered()
    (CONFIG/"hidden-eval").mkdir(parents=True,exist_ok=True)
    outputs={"train":[],"val":[]}
    for task in tasks:
        key=digest(task["instance_id"].encode())[:12]
        save(CONFIG/"test-specs"/(key+".json"),{k:task[k] for k in ["instance_id","FAIL_TO_PASS","PASS_TO_PASS"]})
        payload=build_task(task,batch="rl").model_dump(mode="json")
        payload["metadata"]["dataset_split"]=task["split"]
        # 73000 input + 8192 output stays under the 81920-token per-sample training cap.
        payload["agent"]["settings"].update(call_limit=80,max_input_tokens=73000,max_output_tokens=8192,temperature=0.7)
        # Template data includes no reference repair or bug injection patch.
        assert "patch" not in payload["metadata"]
        outputs["train" if task["split"]=="train" else "val"].append({
            "prompt":[{"role":"user","content":task["problem_statement"]}],"label":0,
            "metadata":{"instance_id":task["instance_id"],"dataset_split":task["split"],
                        "runtime":payload["runtime"],"agent":payload["agent"],
                        "builder":payload["builder"],"evaluator":payload["evaluator"],
                        "task_metadata":payload["metadata"]}})
    first={t["instance_id"] for t in select(tasks)[:2]}
    outputs["smoke"]=[row for row in outputs["train"] if row["metadata"]["instance_id"] in first]
    assert len(outputs["smoke"])==2
    directory=ROOT/"datasets/slime"; directory.mkdir(parents=True,exist_ok=True)
    report={"source_datasets":{k:{"rows":v["rows"],"sha256":v["sha256"]} for k,v in manifest["datasets"].items()},
            "image_count":len(manifest["images"]),"files":{},"reference_patches_exported":False}
    for name,rows in outputs.items():
        raw=("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows)).encode()
        path=directory/(name+".jsonl"); temp=path.with_suffix(".tmp");temp.write_bytes(raw);temp.replace(path)
        report["files"][name]={"rows":len(rows),"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}
    save(directory/"manifest.json",report)
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
