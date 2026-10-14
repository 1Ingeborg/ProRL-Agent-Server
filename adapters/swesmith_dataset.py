"""Load the SWE-smith tasks shipped in data/ (300 train, 50 validation)."""
import json, os
from pathlib import Path

ROOT = Path(os.environ.get("POLAR_HOME", str(Path(__file__).resolve().parents[1])))
DATA = ROOT / "data"

def image_ref(image_name):
    # Rows name images as jyangballin/swesmith.x86_64.<repo>; Docker Hub serves them as swebench/.
    return "swebench/" + image_name.split("/", 1)[1] + ":latest"

def load_tasks():
    tasks = []
    for split, name in [("train", "train.jsonl"), ("validation", "val.jsonl")]:
        for line in (DATA / name).read_text().splitlines():
            row = json.loads(line)
            tasks.append(dict(row, split=split, image=image_ref(row["image_name"])))
    ids = [t["instance_id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate instance_id across splits"
    return tasks
