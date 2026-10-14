#!/usr/bin/env bash
set -Eeuo pipefail
export POLAR_HOME="${POLAR_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Task specs import adapters by module path (e.g. swe_agent_harness:SWEAgentHarness).
export PYTHONPATH=$POLAR_HOME/adapters TMPDIR=$POLAR_HOME/tmp LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$POLAR_HOME/runs"
action=${1:-status}
case "$action" in start|stop|status) ;; *) echo 'Usage: services.sh start|stop|status'; exit 2 ;; esac
if [[ $action == start ]]; then
  [[ -f "$POLAR_HOME/config/model.env" ]] || { echo 'MODEL_BASE_URL configuration missing; no service started'; exit 2; }
  set -a; source "$POLAR_HOME/config/model.env"; set +a
  [[ -n ${MODEL_BASE_URL:-} ]] || { echo "MODEL_BASE_URL is required in $POLAR_HOME/config/model.env; no service started"; exit 2; }
fi
exec 9>"$POLAR_HOME/runs/services.lock"
flock -n 9 || { echo 'Another service operation is running'; exit 1; }
"$POLAR_HOME/conda/polar/bin/python" - "$action" <<'PY'
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ["POLAR_HOME"])
OUT=ROOT/'runs/services'
CLI=str(ROOT/'conda/polar/bin/polar')
CONFIG=str(ROOT/'config/topology.yaml')
COMMANDS={name:[CLI,'serve_'+name,'-c',CONFIG] for name in ['rollout','gateway']}
def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        args=(p/'cmdline').read_bytes().split(b'\0')
        return fields[19],[x.decode() for x in args if x]
    except (FileNotFoundError,PermissionError,ProcessLookupError):
        return None
def owned(name,record):
    item=proc(record.get('pid',-1))
    if item is None or item[0]!=record.get('start_ticks'):return False
    expected=COMMANDS[name]
    return any(item[1][i:i+len(expected)]==expected for i in range(len(item[1])))
def read(name):
    p=OUT/(name+'.json')
    try:return json.loads(p.read_text())
    except FileNotFoundError:return {}
def save(name,record):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/(name+'.json');q=p.with_suffix('.tmp')
    q.write_text(json.dumps(record,indent=2)+'\n');q.replace(p)
def stop(name,record):
    if not owned(name,record):
        (OUT/(name+'.json')).unlink(missing_ok=True)
        return
    pid=record['pid']
    if os.getpgid(pid)!=pid:raise RuntimeError('Unexpected process group: '+name)
    os.killpg(pid,signal.SIGTERM)
    deadline=time.monotonic()+5
    while owned(name,record) and time.monotonic()<deadline:time.sleep(0.1)
    if owned(name,record):os.killpg(pid,signal.SIGKILL)
    (OUT/(name+'.json')).unlink(missing_ok=True)
action=sys.argv[1]
if action=='start':
    import yaml
    from polar.config import TopologyConfig
    text=(ROOT/'config/topology.template.yaml').read_text()
    for key in ['MODEL_BASE_URL','MODEL_NAME','INFERENCE_ENGINE','POLAR_HOME']:
        value=os.environ.get(key)
        if not value:raise SystemExit(key+' is required; no service started')
        text=text.replace('${'+key+'}',value)
    data=yaml.safe_load(text);TopologyConfig.model_validate(data)
    Path(CONFIG).write_text(yaml.safe_dump(data))
    started=[]
    try:
        for name,cmd in COMMANDS.items():
            old=read(name)
            if owned(name,old):
                print(name+' already running PID='+str(old['pid']));continue
            with (ROOT/'runs'/(''+name+'-service.log')).open('ab') as log:
                child=subprocess.Popen(['nohup',*cmd],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True,close_fds=True)
            time.sleep(0.2)
            item=proc(child.pid)
            if child.poll() is not None or item is None:
                raise RuntimeError(name+' exited; inspect '+str(ROOT/'runs'/(name+'-service.log')))
            record={'pid':child.pid,'start_ticks':item[0],'command':cmd}
            if not owned(name,record):raise RuntimeError('Unexpected service command: '+name)
            save(name,record);started.append((name,record))
            print(name+' started PID='+str(child.pid))
    except Exception:
        for name,record in started:stop(name,record)
        raise
elif action=='stop':
    for name in COMMANDS:
        stop(name,read(name));print(name+' stopped')
else:
    for name in COMMANDS:
        record=read(name)
        print(name+(' running PID='+str(record['pid']) if owned(name,record) else ' stopped'))
PY
