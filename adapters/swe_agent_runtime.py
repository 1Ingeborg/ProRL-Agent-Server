"""Native DockerRuntime with durable session files before Polar removes temp dirs."""
import os
import asyncio, hashlib, json, shutil
from pathlib import Path
from polar.runtime.docker import DockerRuntime

ROOT = Path(os.environ.get("POLAR_HOME", str(Path(__file__).resolve().parents[1])))
class SWEAgentDockerRuntime(DockerRuntime):
    async def stop(self):
        try:
            await super().stop()
        finally:
            # The source session includes its eval_runtime directory.
            if not self.session_id.endswith("-eval") and self.session_dir.exists():
                key = hashlib.sha256(self.session_id.encode()).hexdigest()[:20]
                kind = "offline-artifacts" if self.session_id.startswith("swecheck-") else "artifacts"
                destination = ROOT / "runs/swe-agent" / kind / key
                destination.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(shutil.copytree, self.session_dir, destination,
                                        dirs_exist_ok=True, symlinks=True)
                (destination / "session-identity.json").write_text(
                    json.dumps({"session_id": self.session_id, "network": "none", "gpu": 0}, indent=2))
