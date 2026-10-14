"""Prepare an isolated buggy Git tree. Run only inside the task container."""
import argparse, json, os, pathlib, shutil, subprocess, tarfile, tempfile

def git(*args):
    return subprocess.check_output(["git", "-C", "/testbed", *args])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instance-id", required=True)
    args = p.parse_args()
    if not pathlib.Path("/.dockerenv").exists():
        raise RuntimeError("Task preparation is allowed only inside Docker")
    root = pathlib.Path("/testbed")
    if root.resolve() != pathlib.Path("/testbed") or not (root / ".git").is_dir():
        raise RuntimeError("Expected original /testbed Git repository")
    ref = "refs/remotes/origin/" + args.instance_id
    commit = git("rev-parse", ref + "^{commit}").decode().strip()
    with tempfile.TemporaryDirectory(prefix="swe-bug-tree-", dir="/tmp") as temp:
        archive = pathlib.Path(temp) / "source.tar"
        git("archive", "--format=tar", "--output=" + str(archive), commit)
        # The old image repository includes reference fixes in its history.
        # Replace all files, refs and Git objects with exactly the buggy tree.
        for child in root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        with tarfile.open(archive) as bundle:
            bundle.extractall(root, filter="data")
    git("init", "-q")
    git("config", "user.name", "SWE task")
    git("config", "user.email", "task@localhost")
    git("add", "-A", "-f")
    git("commit", "-q", "-m", "Task starting state")
    assert git("rev-list", "--all", "--count").decode().strip() == "1"
    assert not git("remote").strip()
    assert not git("status", "--porcelain").strip()
    pathlib.Path("/polar/session/logs/agent").mkdir(parents=True, exist_ok=True)
    pathlib.Path("/polar/session/cache").mkdir(exist_ok=True)
    pathlib.Path("/polar/session/task-preparation.json").write_text(json.dumps({
        "instance_id": args.instance_id, "buggy_source_commit": commit,
        "starting_tree": git("rev-parse", "HEAD^{tree}").decode().strip(),
        "history_commits": 1, "remote_count": 0, "reference_patch_uploaded": False
    }, indent=2))
    print("Isolated buggy tree prepared; no original Git history or remotes")
if __name__ == "__main__":
    main()
