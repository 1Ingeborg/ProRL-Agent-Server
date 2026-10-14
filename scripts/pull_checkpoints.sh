#!/usr/bin/env bash
# Runs on the Aliyun host. Pulls each AutoDL checkpoint once its post-save eval has been
# logged, verifies the file manifest, then frees the AutoDL copy (the final iteration is
# kept on both sides). Connection settings come from config/remote-tunnel.json.
# Usage: scripts/pull_checkpoints.sh <RUN_ID> <FINAL_ITER> [REMOTE_SAVE_ROOT] [DEST]
set -uo pipefail
POLAR_HOME="${POLAR_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${1:?run id}" FINAL_ITER="${2:?final iteration}"
REMOTE_HOME="${REMOTE_HOME:-/root/autodl-tmp/ProRL-Agent-Serve}"
REMOTE_SAVE="${3:-$REMOTE_HOME/checkpoints/full300}"
DEST="${4:-$POLAR_HOME/checkpoints/$RUN_ID}"
read -r HOST PORT IDENTITY KNOWN_HOSTS < <("$POLAR_HOME/conda/polar/bin/python" -c '
import json,sys; c=json.load(open(sys.argv[1]))
print(c["user"]+"@"+c["host"], c["port"], c["identity_file"], c["known_hosts"])' "$POLAR_HOME/config/remote-tunnel.json")
SSH_OPTS="-p $PORT -i $IDENTITY -o UserKnownHostsFile=$KNOWN_HOSTS -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=30"
r() { ssh $SSH_OPTS "$HOST" "$@"; }
log() { echo "[$(date +%F_%T)] $*"; }
manifest() { find "$1" -type f -printf "%P %s\n" | sort | md5sum | cut -d" " -f1; }
mkdir -p "$DEST"
pull() {
  local name=$1 n=$2
  log "pulling $name"
  rsync -a --partial -e "ssh $SSH_OPTS" "$HOST:$REMOTE_SAVE/$name/" "$DEST/$name.partial/" || { log "rsync failed for $name"; return 1; }
  local want; want=$(r "cd $REMOTE_SAVE/$name && find . -type f -printf \"%P %s\n\" | sort | md5sum | cut -d\" \" -f1") || return 1
  [[ $(manifest "$DEST/$name.partial") == "$want" ]] || { log "manifest mismatch for $name"; return 1; }
  mv "$DEST/$name.partial" "$DEST/$name"
  rsync -a -e "ssh $SSH_OPTS" --exclude "iter_*" "$HOST:$REMOTE_SAVE/" "$DEST/" || true
  echo "$n" > "$DEST/latest_checkpointed_iteration.txt"
  if (( n != FINAL_ITER )); then r "rm -rf $REMOTE_SAVE/$name" && log "removed AutoDL copy of $name"; fi
  log "pulled $name ($(du -sh "$DEST/$name" | cut -f1))"
}
while true; do
  state=$(r "cat $REMOTE_SAVE/latest_checkpointed_iteration.txt 2>/dev/null || echo -1; echo ---; ls -d $REMOTE_SAVE/iter_* 2>/dev/null; echo ---; grep -aoE \"rollout.py:[0-9]+ - eval [0-9]+:\" $REMOTE_HOME/runs/training/$RUN_ID/job.readable.log | grep -oE \"[0-9]+:\$\" | tr -d :; echo ---; $REMOTE_HOME/conda/train/bin/python $REMOTE_HOME/scripts/managed_job.py training status") || { log "ssh failed; retrying"; sleep 60; continue; }
  tracker=$(awk "NR==1" <<<"$state")
  dirs=$(awk "/^---\$/{s++;next} s==1" <<<"$state")
  evals=" $(awk "/^---\$/{s++;next} s==2" <<<"$state" | tr "\n" " ") "
  running=$(awk "/^---\$/{s++;next} s==3" <<<"$state" | grep -c running)
  for d in $dirs; do
    name=$(basename "$d"); n=$((10#${name#iter_}))
    [[ -d $DEST/$name ]] && continue
    (( n <= tracker )) || continue
    # A crashed run never logs the eval, so pull whatever was saved before exiting.
    if [[ $evals == *" $n "* ]] || (( running == 0 )); then pull "$name" "$n"; fi
  done
  if (( running == 0 )); then log "training stopped; exiting"; break; fi
  sleep 120
done
