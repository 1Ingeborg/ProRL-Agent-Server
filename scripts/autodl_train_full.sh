#!/usr/bin/env bash
set -Eeuo pipefail
export POLAR_HOME="${POLAR_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TRAIN_PROFILE="${TRAIN_PROFILE:-$POLAR_HOME/config/full300-8a800.env}"
export RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)-full300-8a800}"
exec bash "$POLAR_HOME/scripts/autodl_train.sh" "${1:-plan}"
