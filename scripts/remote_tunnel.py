#!/usr/bin/env python3
"""Manage the execution-host SSH tunnel without forwarding shell access."""
import argparse, json, os, signal, subprocess, time
from pathlib import Path
ROOT = Path(os.environ.get("POLAR_HOME", str(Path(__file__).resolve().parents[1])))
OUT = ROOT / "runs/remote-tunnel"
def identity(pid):
    try:
        p = Path("/proc") / str(pid)
        fields = (p/"stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z": return None
        return fields[19], (p/"cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError): return None
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=["start","stop","status"])
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/"control.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=OUT/"process.json"; record=json.loads(path.read_text()) if path.exists() else {}
        item=identity(record.get("pid",-1))
        alive=bool(item and item[0]==record.get("start_ticks") and item[1]==[x.encode() for x in record.get("command",[])]+[b""])
        if args.action=="status": print("running PID="+str(record["pid"]) if alive else "stopped"); return
        if args.action=="stop":
            if alive:
                if os.getpgid(record["pid"])!=record["pid"]: raise RuntimeError("Unexpected process group")
                os.killpg(record["pid"],signal.SIGTERM)
                for _ in range(50):
                    if not identity(record["pid"]): break
                    time.sleep(.1)
                if identity(record["pid"]): raise RuntimeError("SSH tunnel did not exit")
            path.unlink(missing_ok=True); print("stopped"); return
        if alive: print("already running PID="+str(record["pid"])); return
        cfg=json.loads((ROOT/"config/remote-tunnel.json").read_text())
        for key in ["host","port","user","identity_file","known_hosts"]:
            if not cfg.get(key): raise RuntimeError("Tunnel configuration missing: "+key)
        cmd=["ssh","-N","-T","-p",str(cfg["port"]),"-i",cfg["identity_file"],
             "-o","BatchMode=yes","-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=yes",
             "-o","UserKnownHostsFile="+cfg["known_hosts"],"-o","ExitOnForwardFailure=yes",
             "-o","ConnectTimeout=15","-o","ServerAliveInterval=15","-o","ServerAliveCountMax=3",
             "-L","127.0.0.1:18900:127.0.0.1:9000",
             "-R","127.0.0.1:18080:127.0.0.1:8080",
             "-R","127.0.0.1:18100:127.0.0.1:8100",cfg["user"]+"@"+cfg["host"]]
        with (OUT/"ssh.log").open("ab") as log:
            child=subprocess.Popen(["nohup",*cmd],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        # Allow nohup to exec ssh before storing process identity.
        for _ in range(200):
            item=identity(child.pid)
            if child.poll() is not None: raise RuntimeError("SSH exited; inspect "+str(OUT/"ssh.log"))
            if item and item[1]==[x.encode() for x in cmd]+[b""]: break
            time.sleep(.05)
        else: raise RuntimeError("Unexpected SSH identity")
        time.sleep(2)
        if child.poll() is not None: raise RuntimeError("SSH forwarding failed; inspect "+str(OUT/"ssh.log"))
        path.write_text(json.dumps({"pid":child.pid,"start_ticks":item[0],"command":cmd},indent=2)+"\n")
        print("started with nohup PID="+str(child.pid))
if __name__=="__main__": main()
