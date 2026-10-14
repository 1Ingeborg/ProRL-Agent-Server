#!/usr/bin/env python3
"""nohup job manager with start-tick and command-line ownership checks."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(os.environ.get("POLAR_HOME",str(Path(__file__).resolve().parents[1])))
def proc(pid):
    try:
        p=Path("/proc")/str(pid);s=(p/"stat").read_text().rsplit(")",1)[1].split()
        if s[0]=="Z":return None
        return s[19],[x.decode() for x in (p/"cmdline").read_bytes().split(b"\0") if x]
    except (FileNotFoundError,ProcessLookupError):return None
def main():
    p=argparse.ArgumentParser();p.add_argument("role",choices=["training","rl-bridge"]);p.add_argument("action",choices=["start","stop","status"])
    args=p.parse_args();directory=ROOT/"runs"/args.role;directory.mkdir(parents=True,exist_ok=True)
    command=(["bash",str(ROOT/"scripts/autodl_train.sh"),"_run"] if args.role=="training"
             else [str(ROOT/"conda/polar/bin/python"),str(ROOT/"adapters/swe_gateway_bridge.py"),"--mode","unix","--socket",str(ROOT/"runs/swe-agent/bridge/model.sock")])
    def owned(record):
        current=proc(record.get("pid",-1))
        return bool(current and current[0]==record.get("start_ticks") and current[1]==command)
    with (directory/"control.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=directory/"process.json";record=json.loads(path.read_text()) if path.exists() else {}
        if args.action=="status":print("running PID="+str(record["pid"]) if owned(record) else "stopped");return
        if args.action=="stop":
            if owned(record):
                if os.getpgid(record["pid"])!=record["pid"]:raise RuntimeError("Unexpected process group")
                os.killpg(record["pid"],signal.SIGTERM)
                for _ in range(300):
                    if not owned(record):break
                    time.sleep(.1)
                if owned(record):os.killpg(record["pid"],signal.SIGKILL)
            path.unlink(missing_ok=True);print("stopped");return
        if owned(record):print("already running PID="+str(record["pid"]));return
        if args.role=="training":
            check=subprocess.run(["bash",str(ROOT/"scripts/autodl_train.sh"),"preflight"])
            if check.returncode:raise SystemExit(check.returncode)
        else:
            cfg=(ROOT/"config/model.env").read_text()
            if not any(x.startswith("MODEL_BASE_URL=") and x.split("=",1)[1].strip().strip("\"'") for x in cfg.splitlines()):
                raise SystemExit("MODEL_BASE_URL is empty; no service or bridge started")
            subprocess.run(["bash",str(ROOT/"scripts/services.sh"),"start"],check=True)
            socket=ROOT/"runs/swe-agent/bridge/model.sock";socket.parent.mkdir(parents=True,exist_ok=True)
            if socket.exists():
                import socket as socklib
                with socklib.socket(socklib.AF_UNIX) as s:
                    try:s.connect(str(socket))
                    except ConnectionRefusedError:socket.unlink()
                    else:raise RuntimeError("An existing bridge owns the socket")
        log_path=directory/"job.log"
        if args.role=="training":
            manifest=ROOT/"datasets/slime/manifest.json"
            run_id=os.environ.get("RUN_ID") or (json.loads(manifest.read_text()).get("run_id") if manifest.exists() else None)
            if run_id:
                log_path=directory/run_id/"job.log"
                log_path.parent.mkdir(parents=True,exist_ok=True)
        with log_path.open("ab") as log:
            child=subprocess.Popen(["nohup",*command],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(200):
            info=proc(child.pid)
            if child.poll() is not None:raise RuntimeError("Job exited; inspect "+str(directory/"job.log"))
            if info and info[1]==command:break
            time.sleep(.05)
        else:raise RuntimeError("Unexpected job identity")
        path.write_text(json.dumps({"pid":child.pid,"start_ticks":info[0],"command":command,
            "run_id":os.environ.get("RUN_ID"),"log_path":str(log_path),
            "training_profile":os.environ.get("TRAIN_PROFILE")},indent=2)+"\n")
        print("started with nohup PID="+str(child.pid)+" log="+str(log_path))
if __name__=="__main__":main()
