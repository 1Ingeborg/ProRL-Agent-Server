"""Load the current covered dataset and its compact, immutable verification evidence."""
import os
import hashlib, json, subprocess
from pathlib import Path

ROOT = Path(os.environ.get("POLAR_HOME", str(Path(__file__).resolve().parents[1])))
DATA = ROOT / "datasets"
EVIDENCE = ROOT / "runs/dataset-verification"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def empty(value):
    if value is None or isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)):
        return not value or any(empty(x) for x in (value.values() if isinstance(value, dict) else value))
    return False

def load_covered(inspect_images=True):
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    raw_evidence = (EVIDENCE / "verification.jsonl").read_bytes()
    assert sha(raw_evidence) == manifest["verification_sha256"], "verification evidence hash mismatch"
    records = [json.loads(line) for line in raw_evidence.splitlines()]
    evidence = {row["instance_id"]: row for row in records}
    assert len(records) == len(evidence) == 350
    tasks, ids = [], {}
    for split, name in [("train", "train.jsonl"), ("validation", "val.jsonl")]:
        raw = (DATA / name).read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        assert sha(raw) == manifest["datasets"][split]["sha256"]
        assert len(rows) == manifest["datasets"][split]["rows"] == {"train": 300, "validation": 50}[split]
        ids[split] = {row["instance_id"] for row in rows}
        assert len(ids[split]) == len(rows)
        for row in rows:
            assert not empty(row), "empty field: " + row["instance_id"]
            record = evidence[row["instance_id"]]
            assert record["split"] == split
            assert sha(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()) == record["task_sha256"]
            assert row["repo"] not in manifest["excluded_repositories"]
            tests = row["FAIL_TO_PASS"] + row["PASS_TO_PASS"]
            assert len(tests) == len(set(tests))
            assert record["status"] == "passed"
            assert [record["cases"][k]["reward"] for k in ["empty", "unrelated_patch", "reference_repair"]] == [0, 0, 1]
            for case in ["unrelated_patch", "reference_repair"]:
                result = record["cases"][case]
                assert result["expected_test_count"] == result["observed_test_count"] == len(tests)
                assert not result["missing_count"] and not result["unexpected_count"]
                assert not result["runner"]["timed_out"] and not result["runner"]["parse_error"]
                assert set(record["test_statuses"][case]) == set(tests)
            assert all(v == "PASSED" for v in record["test_statuses"]["reference_repair"].values())
            tasks.append(dict(row, split=split))
    assert not ids["train"] & ids["validation"]
    assert ids["train"] | ids["validation"] == set(evidence)
    for name, digest in manifest["upstream_file_hashes"].items():
        assert sha((ROOT / name).read_bytes()) == digest
        assert sha(subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:" + name])) == digest
    assert sha((ROOT / "adapters/swesmith_polar_verifier_runner.py").read_bytes()) == manifest["runner_sha256"]
    preparation = Path(manifest["preparation_file"]).read_bytes()
    assert sha(preparation) == manifest["preparation_sha256"]
    patches = json.loads(preparation)["bug_injection_patches"]
    assert set(patches) == set(evidence)
    images = manifest["images"]
    assert len({t["image_name"] for t in tasks}) == len(images) == 39
    if inspect_images:
        inspected = json.loads(subprocess.check_output(
            ["docker", "image", "inspect", *{i["image_id"] for i in images.values()}], text=True))
        by_id = {item["Id"]: item for item in inspected}
        for image in images.values():
            item = by_id[image["image_id"]]
            assert item["Os"] == "linux" and item["Architecture"] == "amd64"
            assert any(d.endswith("@" + image["digest"]) for d in item["RepoDigests"])
    patch_ids = {}
    for t in tasks:
        t["patch"] = patches[t["instance_id"]]
        digest = sha(t["patch"].encode())
        assert digest == evidence[t["instance_id"]]["bug_patch_sha256"]
        patch_ids.setdefault(t["split"], set()).add(digest)
        t["image"] = images[t["image_name"]]
    assert not patch_ids["train"] & patch_ids["validation"]
    test_rows = load_test()
    assert not ({row["instance_id"] for row in test_rows} & set(evidence))
    return tasks, manifest, evidence


def load_test():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    raw = (DATA / "test.jsonl").read_bytes()
    assert sha(raw) == manifest["datasets"]["test"]["sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == len({row["instance_id"] for row in rows}) == 500
    for row in rows:
        assert all(key in row for key in ["instance_id", "repo", "base_commit", "problem_statement", "version", "patch", "test_patch"])
        assert isinstance(row["FAIL_TO_PASS"], list) and isinstance(row["PASS_TO_PASS"], list)
    return rows
