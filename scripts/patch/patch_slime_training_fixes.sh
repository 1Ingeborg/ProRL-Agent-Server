#!/usr/bin/env bash
# Training-side Slime fixes used by the online GRPO runs:
# - divide logits by the rollout temperature in place, so get_log_probs_and_entropy
#   does not hold a second full [T, V] fp32 copy (bitwise-identical result);
# - flush TensorBoard after every scalar batch so long steps show up immediately.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"

if [ ! -d "${SLIME_DIR}/.git" ]; then
    echo "ERROR: Slime git checkout not found at ${SLIME_DIR}" >&2
    exit 1
fi

PATCH_FILE="$(mktemp)"
cleanup() {
    rm -f "${PATCH_FILE}"
}
trap cleanup EXIT

cat > "${PATCH_FILE}" <<'PATCH'
diff --git a/slime/backends/megatron_utils/loss.py b/slime/backends/megatron_utils/loss.py
index 9131101..a3cfe6a 100644
--- a/slime/backends/megatron_utils/loss.py
+++ b/slime/backends/megatron_utils/loss.py
@@ -419,7 +419,8 @@ def get_log_probs_and_entropy(
     # Apply rollout temperature scaling to logits to match rollout-time log-probs.
     rollout_temperature = getattr(args, "rollout_temperature", 1.0)
     if rollout_temperature != 1.0:
-        logits = logits / rollout_temperature
+        # In-place: avoids a second full [T, V] fp32 copy; bitwise-identical result.
+        logits.div_(rollout_temperature)
     logits = logits.contiguous()
     T = logits.size(0)
     device = logits.device
diff --git a/slime/utils/tensorboard_utils.py b/slime/utils/tensorboard_utils.py
index 3c384ec..ea13a45 100644
--- a/slime/utils/tensorboard_utils.py
+++ b/slime/utils/tensorboard_utils.py
@@ -55,6 +55,7 @@ class _TensorboardAdapter(metaclass=SingletonMeta):
         """
         for key in data:
             self._writer.add_scalar(key, data[key], step)
+        self._writer.flush()
 
     def finish(self):
         """Close the tensorboard writer"""
PATCH

if git -C "${SLIME_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "Slime training fixes already applied."
    exit 0
fi

git -C "${SLIME_DIR}" apply --check "${PATCH_FILE}"
git -C "${SLIME_DIR}" apply "${PATCH_FILE}"
echo "Applied Slime training fixes to ${SLIME_DIR}."
