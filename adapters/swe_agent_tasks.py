"""Build native Polar task requests for SWE-smith tasks."""
import os
import hashlib, json, shlex
from pathlib import Path
from polar.rollout.models import TaskRequest

ROOT = Path(os.environ.get("POLAR_HOME", str(Path(__file__).resolve().parents[1])))
CONFIG = ROOT / "config/swe-agent"
BRIDGE = ROOT / "runs/swe-agent/bridge"
PYTHON = str(ROOT / "conda/swe/bin/python")

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() == data:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data); temporary.replace(path)

def select(tasks):
    ordered = sorted((t for t in tasks if t["split"] == "train"),
        key=lambda t: (len(t["FAIL_TO_PASS"]) + len(t["PASS_TO_PASS"]),
                       len(t["problem_statement"]), t["instance_id"]))
    selected, repos = [], set()
    for task in ordered:
        if task["repo"] not in repos:
            selected.append(task); repos.add(task["repo"])
        if len(selected) == 3:
            break
    assert len(selected) == len(repos) == 3
    return selected

def runtime_environment():
    return {
        "PATH": "/opt/miniconda3/envs/testbed/bin:" + str(ROOT / "conda/swe/bin") +
                ":/usr/local/bin:/usr/bin:/bin",
        "HOME": "/polar/session", "PYTHONPATH": "", "POLAR_HOME": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "UV_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/polar/session/cache", "TMPDIR": "/tmp",
        "HTTP_PROXY": "", "HTTPS_PROXY": "", "ALL_PROXY": "",
        "http_proxy": "", "https_proxy": "", "all_proxy": "",
        "NO_PROXY": "*", "no_proxy": "*",
    }

def build_task(task, model="Qwen/Qwen3.5-4B", batch="pilot", offline=False):
    key = digest(task["instance_id"].encode())[:12]
    test_spec = CONFIG / "test-specs" / (key + ".json")
    prepare = ROOT / "adapters/swe_task_prepare.py"
    verifier = ROOT / "adapters/swesmith_polar_verifier_runner.py"
    agent_ref = "refs/remotes/origin/" + task["instance_id"]
    env = runtime_environment()
    expected = {"node_" + digest(n.encode()): "PASSED"
                for n in task["FAIL_TO_PASS"] + task["PASS_TO_PASS"]}
    expected["verifier_run_complete"] = "PASSED"
    volumes = [str(ROOT / "conda") + ":" + str(ROOT / "conda") + ":ro",
               str(ROOT / "repos/SWE-agent") + ":" + str(ROOT / "repos/SWE-agent") + ":ro",
               str(ROOT / "config/tokenizer") + ":/polar/session/tokenizer:ro",
               str(CONFIG / "hidden-eval") + ":/polar/session/eval_runtime:ro"]
    if not offline:
        volumes.append(str(BRIDGE) + ":/polar/gateway:ro")
    payload = {
        "task_id": "swe-agent-" + batch + "-" + key,
        "instruction": task["problem_statement"],
        "num_samples": 1, "timeout_seconds": 7200,
        "runtime": {"backend": "docker", "import_path": "swe_agent_runtime:SWEAgentDockerRuntime",
            "image": task["image"],
            "network": "none", "allow_internet": False, "gpus": 0,
            "cpus": 1, "memory_mb": 3072, "workdir": "/testbed",
            "kwargs": {"volumes": volumes}, "env": env,
            "prepare": [
                {"type": "upload_file", "source": str(prepare), "target": "/polar/session/swe_task_prepare.py"},
                {"type": "exec", "command": PYTHON + " /polar/session/swe_task_prepare.py --instance-id " +
                    shlex.quote(task["instance_id"]), "cwd": "/testbed"}],
            "eval_prepare": [
                {"type": "exec", "command": "git checkout --detach " + shlex.quote(agent_ref + "~1"),
                 "cwd": "/testbed"},
                {"type": "upload_file", "source": str(verifier), "target": "/polar/session/run_verifier_tests.py"},
                {"type": "upload_file", "source": str(test_spec), "target": "/polar/session/test_spec.json"}]},
        "agent": {"import_path": "swe_agent_harness:SWEAgentHarness", "model_name": model,
                  "settings": {"call_limit": 30, "offline_gateway_bridge": not offline}},
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {"strategy": "test_on_output", "refresh_runtime": True,
            "config": {"repo_dir": "/testbed",
                "patch_command": "cd /testbed && git add -A && git diff --cached --binary HEAD",
                "test_command": "/opt/miniconda3/envs/testbed/bin/python /polar/session/run_verifier_tests.py",
                "expected_output_json": expected, "test_timeout": 180, "apply_timeout": 60,
                "exclude_patterns": ["tests/**", "**/tests/**", "test_*.py", "**/test_*.py",
                                     "conftest.py", "**/conftest.py"]}},
        "metadata": {"instance_id": task["instance_id"], "dataset_split": "train",
                     "repo": task["repo"], "image": task["image"]}
    }
    return TaskRequest.model_validate(payload)
