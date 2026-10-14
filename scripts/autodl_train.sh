#!/usr/bin/env bash
set -Eeuo pipefail
export POLAR_HOME="${POLAR_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -n ${TRAIN_PROFILE:-} ]]; then
 set -a
 source "$TRAIN_PROFILE"
 set +a
fi
PY="$POLAR_HOME/conda/train/bin/python"
export CONDA_PREFIX="$POLAR_HOME/conda/train"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/root/autodl-tmp/models/Qwen3.5-4B}"
export SLIME_DIR="$POLAR_HOME/repos/slime" MEGATRON_DIR="$POLAR_HOME/repos/Megatron-LM"
export REF_LOAD="${REF_LOAD:-$POLAR_HOME/checkpoints/reference}" SAVE_ROOT="${SAVE_ROOT:-$POLAR_HOME/checkpoints/smoke}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" CUDA_DEVICE_MAX_CONNECTIONS=1
export PATH="$POLAR_HOME/conda/train/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$POLAR_HOME/adapters:$POLAR_HOME/src:$SLIME_DIR:$MEGATRON_DIR"
export TMPDIR="$POLAR_HOME/tmp" HF_HOME="$POLAR_HOME/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$POLAR_HOME/cache/torch-extensions" TRITON_CACHE_DIR="$POLAR_HOME/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$POLAR_HOME/cache/torchinductor" CUDA_CACHE_PATH="$POLAR_HOME/cache/cuda"
export SLIME_HOST_IP=127.0.0.1
# Ray appends /ray/session_*; keep Unix socket paths below AF_UNIX's limit.
export WANDB_MODE=offline RAY_TMPDIR="$POLAR_HOME"
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=2147483648
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export FLA_TILELANG=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TRAIN_GPUS="${TRAIN_GPUS:-2}" ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export COLOCATE="${COLOCATE:-1}" TRAIN_TP="${TRAIN_TP:-1}" ROLLOUT_TP="${ROLLOUT_TP:-1}"
export TRAIN_CP="${TRAIN_CP:-1}"
export OFFLOAD_TRAIN="${OFFLOAD_TRAIN:-0}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}" N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export START_ROLLOUT_ID="${START_ROLLOUT_ID:-0}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-131072}"
export SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}" SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-}" EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-}" N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export PROMPT_DATA="${PROMPT_DATA:-$POLAR_HOME/datasets/slime/smoke.jsonl}"
export RUN_ID="${RUN_ID:-$($PY -c 'import json,os; print(json.load(open(os.path.join(os.environ["POLAR_HOME"],"datasets/slime/manifest.json"))).get("run_id") or "")')}"
export TENSORBOARD_DIR="$POLAR_HOME/runs/tensorboard/$RUN_ID"
mkdir -p "$TENSORBOARD_DIR" "$POLAR_HOME/runs/training"
export POLAR_RUN_ID="$RUN_ID"
for value in "$TRAIN_GPUS" "$ROLLOUT_GPUS" "$TRAIN_TP" "$TRAIN_CP" "$ROLLOUT_TP" "$ROLLOUT_BATCH_SIZE" "$N_SAMPLES_PER_PROMPT" "$NUM_ROLLOUT"; do
 [[ $value =~ ^[1-9][0-9]*$ ]] || { echo 'GPU and batch counts must be positive integers'; exit 2; }
done
(( TRAIN_GPUS % (TRAIN_TP * TRAIN_CP) == 0 && ROLLOUT_GPUS % ROLLOUT_TP == 0 )) || { echo 'GPU counts must be divisible by TP'; exit 2; }
if [[ $COLOCATE == 1 ]]; then
 [[ "$TRAIN_GPUS" == "$ROLLOUT_GPUS" ]] || { echo 'Colocated training and rollout must use the same GPU count'; exit 2; }
 export EXPECTED_GPUS="$TRAIN_GPUS"
else
 export EXPECTED_GPUS="$((TRAIN_GPUS + ROLLOUT_GPUS))"
fi
source "$POLAR_HOME/examples/swegym_slime_grpo/model_args.sh"
ACTION="${1:-plan}"
command=("$PY" "$SLIME_DIR/train.py"
 --actor-num-nodes 1 --actor-num-gpus-per-node "$TRAIN_GPUS" --num-gpus-per-node "$EXPECTED_GPUS"
 --rollout-num-gpus "$ROLLOUT_GPUS" --rollout-num-gpus-per-engine "$ROLLOUT_TP"
 "${MODEL_ARGS[@]}" --hf-checkpoint "$HF_CHECKPOINT" --ref-load "$REF_LOAD"
 --load "$REF_LOAD" --no-save-optim --no-save-rng
 --rollout-function-path remote_sync_rollout.generate
 --custom-rm-path slime_bridge.reward.reward_func
 --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards
 --custom-config-path "$POLAR_HOME/config/remote-slime.yaml"
 --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer
 --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --metadata-key metadata
 --reward-key score --num-rollout "$NUM_ROLLOUT" --start-rollout-id "$START_ROLLOUT_ID" --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
 --rollout-max-response-len 8192 --rollout-max-prompt-len "$CONTEXT_LENGTH"
 --rollout-temperature 0.7 --rollout-top-p 1.0 --rollout-top-k -1
 --num-steps-per-rollout 1
 --save-debug-rollout-data "$POLAR_HOME/runs/training/$RUN_ID/samples-{rollout_id}.pt"
 --tensor-model-parallel-size "$TRAIN_TP" --pipeline-model-parallel-size 1 --context-parallel-size "$TRAIN_CP"
 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
 --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" --log-probs-chunk-size 256
 --advantage-estimator grpo --normalize-advantages --use-tis
 --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl
 --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28
 --use-precision-aware-optimizer
 --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
 --attention-dropout 0 --hidden-dropout 0
 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
 --attention-backend auto --no-gradient-accumulation-fusion
 --sglang-mem-fraction-static 0.85 --sglang-context-length "$CONTEXT_LENGTH" --sglang-max-running-requests 128
 --sglang-tool-call-parser qwen3_coder --sglang-router-port 9000
 --router-policy cache_aware --use-tensorboard --tb-project-name ProRL-Agent-Serve --tb-experiment-name "$RUN_ID")
if [[ $COLOCATE == 1 ]]; then
 command+=(--colocate)
elif [[ $OFFLOAD_TRAIN == 1 ]]; then
 command+=(--offload-train)
fi
if [[ $SAVE_CHECKPOINT == 1 ]]; then
 command+=(--save "$SAVE_ROOT" --save-interval "$SAVE_INTERVAL")
fi
# Eval runs after the weight update, so the epoch-boundary eval scores the fully trained model.
if [[ -n $EVAL_INTERVAL ]]; then
 command+=(--eval-interval "$EVAL_INTERVAL" --eval-prompt-data val "$EVAL_PROMPT_DATA" --n-samples-per-eval-prompt "$N_SAMPLES_PER_EVAL_PROMPT")
 [[ ${SKIP_EVAL_BEFORE_TRAIN:-0} == 1 ]] && command+=(--skip-eval-before-train)
fi
preflight() {
 "$PY" - <<'PY'
import hashlib,json,os,shutil,torch,urllib.request
from pathlib import Path
root=Path(os.environ["POLAR_HOME"])
if not torch.cuda.is_available():raise SystemExit("GPU is not visible; training was not started")
if torch.cuda.device_count()!=int(os.environ['EXPECTED_GPUS']):raise SystemExit("Visible GPU count does not match training/rollout allocation")
prompt=Path(os.environ['PROMPT_DATA'])
manifest=json.loads((root/'datasets/slime/manifest.json').read_text())
entry=manifest['files'][prompt.stem]
if hashlib.sha256(prompt.read_bytes()).hexdigest()!=entry['sha256']:raise SystemExit('Prompt data hash does not match manifest')
rows=sum(1 for line in prompt.open() if line.strip())
if os.environ.get('TRAIN_SCOPE')=='full300':
 if rows!=300:raise SystemExit('Full profile requires exactly the original 300 training prompts')
 if int(os.environ['NUM_ROLLOUT'])*int(os.environ['ROLLOUT_BATCH_SIZE'])!=rows:raise SystemExit('Full profile must cover exactly one epoch')
if not (Path(os.environ["REF_LOAD"])/"latest_checkpointed_iteration.txt").exists():raise SystemExit("Convert the HF checkpoint first; reference torch_dist is missing")
size=sum(p.stat().st_size for p in Path(os.environ["REF_LOAD"]).rglob("*") if p.is_file())
free=shutil.disk_usage(root).free
reserve=int(os.environ.get('DISK_RESERVE_GB','10'))*1_000_000_000
if os.environ['SAVE_CHECKPOINT']=='1' and free<reserve+size*int(os.environ.get('SAVE_DISK_COPIES','2')):raise SystemExit("Insufficient disk for a weights-only checkpoint, staging and disk reserve")
if free<10_000_000_000:raise SystemExit("10 GB disk reserve violated")
urllib.request.urlopen("http://127.0.0.1:18080/health",timeout=10).close()
PY
}
case "$ACTION" in
 plan) printf '%q ' "${command[@]}"; printf '\n' ;;
 preflight) preflight ;;
 convert)
   "$PY" -c 'import torch; assert torch.cuda.is_available(), "GPU is not visible"'
   # The native converter writes one torch_dist checkpoint directly; HF remains in place.
   "$PY" -c 'import shutil,os,pathlib; p=pathlib.Path(os.environ["HF_CHECKPOINT"]); s=sum(x.stat().st_size for x in p.glob("*.safetensors")); assert shutil.disk_usage(os.environ["POLAR_HOME"]).free>=11000000000+s, "Insufficient checkpoint-conversion disk budget"'
   export PYTHON_BIN="$PY" TORCH_DIST_DIR="$REF_LOAD"
   bash "$POLAR_HOME/examples/swegym_slime_grpo/convert_weights.sh" ;;
 start|stop|status) "$PY" "$POLAR_HOME/scripts/managed_job.py" training "$ACTION" ;;
 _run)
   exec 8>"$POLAR_HOME/runs/training/worker.lock";flock -n 8 || exit 2
   preflight
   "$PY" "$POLAR_HOME/scripts/render_remote_config.py"
   "$PY" "$POLAR_HOME/scripts/tensorboard_gpu_metrics.py" --logdir "$TENSORBOARD_DIR" --run-id "$RUN_ID" &
   GPU_METRICS_PID=$!
   "$PY" "$POLAR_HOME/scripts/follow_readable_log.py" "$POLAR_HOME/runs/training/$RUN_ID/job.log" "$POLAR_HOME/runs/training/$RUN_ID/job.readable.log" &
   READABLE_LOG_PID=$!
   trap 'kill "$GPU_METRICS_PID" "$READABLE_LOG_PID" 2>/dev/null || true; wait "$GPU_METRICS_PID" "$READABLE_LOG_PID" 2>/dev/null || true' EXIT
   "${command[@]}" ;;
 *) echo 'Usage: autodl_train.sh plan|preflight|convert|start|stop|status'; exit 2 ;;
esac
