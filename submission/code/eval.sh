#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model=${1:-$here/../03_模型权重/checkpoint}
model=$(cd -- "$model" && pwd)
export LINGBOTVLA_TRAINING_CONFIG="$model/lingbotvla_cli.yaml"
export AITER_TRITON_ONLY=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTHONUNBUFFERED=1
export TMPDIR=${TMPDIR:-/var/tmp/cucu-eval-cache}
mkdir -p "$TMPDIR"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
exec /opt/robotwin-env/bin/python "$here/scripts/run_fp32_benchmark.py" --gpu-count 8 --episodes "${EPISODES:-10}" --task-config "${TASK_CONFIG:-both}" --run-name "${RUN_NAME:-cucu_6k_fp32_reproduce}" --model-path "$model" --base-port "${BASE_PORT:-13820}" --expert-check --accept-expert-info-on-failure --video
