#!/usr/bin/env bash
set -euo pipefail
base=/workspace/runtime/experiments/randomized_fp32_20261026
export AITER_TRITON_ONLY=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE PYTHONUNBUFFERED=1
export LINGBOTVLA_TRAINING_CONFIG=/workspace/runtime/models/full_sft_6000/lingbotvla_cli.yaml
export TMPDIR=/var/tmp/robotwin-randomized-fp32-20261026
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
cd /RoboTwin
/opt/robotwin-env/bin/python -u "$base/gpu_monitor.py" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
set +e
/opt/robotwin-env/bin/python -u "$base/verify_model.py"
status=$?
if [ "$status" -eq 0 ]; then
 /opt/robotwin-env/bin/python -u "$base/run_fp32_benchmark.py" --gpu-count 8 --episodes 10 --task-config demo_randomized --run-name full_sft_6000_randomized50x10_fp32_8gpu_video_20261026 --model-path /workspace/runtime/models/full_sft_6000 --base-port 13820 --expert-check --accept-expert-info-on-failure --video
 status=$?
fi
printf '%s\n' "$status" > "$base/exit_code"
exit "$status"
