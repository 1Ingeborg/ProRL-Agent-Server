"""Launch upstream SWE-agent; keep all reasoning and tool logic upstream."""
import os
import shlex
from pathlib import Path
from polar.agent.base import BaseHarness
from polar.runtime.models import ExecInput

ROOT = Path(os.environ.get("POLAR_HOME", str(Path(__file__).resolve().parents[1])))
class SWEAgentHarness(BaseHarness):
    async def setup(self, runtime):
        for name in ["swe_agent_runner.py", "swe_gateway_bridge.py"]:
            await runtime.upload_file(str(Path(__file__).with_name(name)),
                                      runtime.runtime_session_dir + "/" + name)
    def run_steps(self, instruction):
        command = str(ROOT / "conda/swe/bin/python") + " /polar/session/swe_agent_runner.py"
        command += " --model " + shlex.quote(self.model_name or "Qwen/Qwen3.5-4B")
        command += " --instruction " + shlex.quote(instruction)
        command += " --call-limit " + str(int(self.settings.get("call_limit", 80)))
        for key,default in [("max_input_tokens",122880),("max_output_tokens",8192),("temperature",0.7)]:
            command += " --" + key.replace("_","-") + " " + str(self.settings.get(key,default))
        command += " --run-id " + shlex.quote(str(self.settings.get("run_id", "unknown")))
        if self.settings.get("offline_gateway_bridge"):
            command += " --gateway-socket /polar/gateway/model.sock"
        return [ExecInput(command=command, cwd="/testbed", env=self.env)]
    async def postprocess(self, runtime, result):
        # Preserve staged, unstaged, and newly created files before teardown.
        response = await runtime.exec(
            "git add -A && git diff --cached --binary HEAD > /polar/session/logs/agent/candidate.patch",
            cwd="/testbed", timeout_sec=30)
        if response.return_code:
            raise RuntimeError("Could not collect SWE-agent candidate patch")
