#!/usr/bin/env python3
"""SWE-smith input/output adapter for Polar's unmodified test_on_output verifier.
Runs real pytest tests and exports one stable identifier per complete node ID.
Does not compute an RL reward; Polar owns grading and patch replay.
"""
import collections,hashlib,json,os,pathlib,signal,subprocess,sys,time
import xml.etree.ElementTree as ET
ROOT=pathlib.Path("/polar/session")
CFG=json.loads((ROOT/"test_spec.json").read_text())
def alias(node):return "node_"+hashlib.sha256(node.encode()).hexdigest()
tests=CFG["FAIL_TO_PASS"]+CFG["PASS_TO_PASS"]
assert len(tests)==len(set(tests))
plugin=r'''
import signal,os
class PolarTestTimeout(TimeoutError):pass
def deadline(signum,frame):
    raise PolarTestTimeout("POLAR_INDIVIDUAL_TEST_TIMEOUT")
def pytest_collection_modifyitems(items):
    for item in items:
        item.user_properties.append(("polar_full_nodeid",item.nodeid))
def pytest_runtest_logstart(nodeid,location):
    signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,20)
def pytest_runtest_logfinish(nodeid,location):
    signal.setitimer(signal.ITIMER_REAL,0)
'''
(ROOT/"polar_nodeid_plugin.py").write_text(plugin)
env=os.environ.copy()
env.update({"PATH":str(pathlib.Path(sys.executable).parent)+os.pathsep+env.get("PATH",""),
    "PYTHONPATH":str(ROOT)+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
    "PYTHONPYCACHEPREFIX":str(ROOT/"bytecode"),"PYTHONDONTWRITEBYTECODE":"1",
    "PYTHONHASHSEED":"0","PIP_NO_INDEX":"1","HF_HUB_OFFLINE":"1","NO_PROXY":"*","no_proxy":"*"})
for key in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):env.pop(key,None)
xml=ROOT/"pytest-results.xml";log=ROOT/"pytest-output.log"
cmd=[sys.executable,"-m","pytest","-q","--tb=short","--color=no","-p","no:cacheprovider",
    "-p","polar_nodeid_plugin","--junitxml="+str(xml),*tests]
start=time.monotonic();timed_out=False
with log.open("w") as f:
    proc=subprocess.Popen(cmd,cwd="/testbed",env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    try:rc=proc.wait(timeout=170)
    except subprocess.TimeoutExpired:
        timed_out=True
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait();rc=124
observed={};extra=[];parse_error=None
try:
    tree=ET.parse(xml)
    for case in tree.iter("testcase"):
        node=None
        for prop in case.findall("./properties/property"):
            if prop.attrib.get("name")=="polar_full_nodeid":node=prop.attrib.get("value")
        status="ERROR" if case.find("error") is not None else ("FAILED" if case.find("failure") is not None else ("SKIPPED" if case.find("skipped") is not None else "PASSED"))
        if not node or node not in tests:
            extra.append({"name":case.attrib.get("name"),"status":status,"nodeid":node});continue
        previous=observed.get(node)
        if previous is None or {"PASSED":0,"SKIPPED":1,"FAILED":2,"ERROR":3}[status] > {"PASSED":0,"SKIPPED":1,"FAILED":2,"ERROR":3}[previous]:observed[node]=status
except Exception as e:parse_error=repr(e)
missing=[n for n in tests if n not in observed]
detail={"pytest_exit":rc,"timed_out":timed_out,"seconds":round(time.monotonic()-start,3),
    "parse_error":parse_error,"extra_cases":extra,"missing":missing,"statuses":observed,
    "counts":dict(collections.Counter(observed.values()))}
(ROOT/"runner-report.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2))
for node in tests:print(observed.get(node,"ERROR")+" "+alias(node))
complete=not timed_out and parse_error is None and not missing and not extra and rc==0
print(("PASSED" if complete else "ERROR")+" verifier_run_complete")
# The guard stops partial, crashed or incomplete runs from matching an all-pass map.
sys.exit(0 if complete else 1)
