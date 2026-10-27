#!/usr/bin/env bash
set -euo pipefail
base=/var/tmp/robotwin-official-weak5-20261004
mkdir -p "$base/eval_cache"
export TMPDIR="$base/eval_cache"
export LINGBOTVLA_TRAINING_CONFIG=/models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml
export AITER_TRITON_ONLY=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTHONUNBUFFERED=1
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
cd /RoboTwin
set +e
/opt/robotwin-env/bin/python -u "$base/run_selected_benchmark.py" \
  --gpu-count 5 --episodes 10 --task-config demo_clean \
  --tasks put_object_cabinet stack_blocks_three handover_block move_stapler_pad pick_dual_bottles \
  --run-name official_robotwin_weak5_bf16_10x_20261004 \
  --runtime-dir /workspace/runtime \
  --model-path /models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt \
  --base-port 13520 --expert-check --accept-expert-info-on-failure --video
status=$?
printf '%s\n' "$status" > "$base/exit_code"
exit "$status"
