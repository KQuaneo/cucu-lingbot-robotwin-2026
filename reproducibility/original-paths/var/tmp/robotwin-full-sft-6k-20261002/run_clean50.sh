#!/usr/bin/env bash
export AITER_TRITON_ONLY=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE PYTHONUNBUFFERED=1
export LINGBOTVLA_TRAINING_CONFIG=/workspace/runtime/models/full_sft_6000/lingbotvla_cli.yaml
export TMPDIR=/var/tmp/robotwin-full-sft-6k-20261002/eval_cache TMP=/var/tmp/robotwin-full-sft-6k-20261002/eval_cache TEMP=/var/tmp/robotwin-full-sft-6k-20261002/eval_cache
mkdir -p "$TMPDIR"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
cd /RoboTwin
/opt/robotwin-env/bin/python experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py --gpu-count 8 --episodes 10 --task-config demo_clean --run-name full_sft_6000_clean50x10_8gpu_video --model-path /workspace/runtime/models/full_sft_6000 --base-port 13420 --expert-check --accept-expert-info-on-failure --video
status=$?
printf "%s\n" "$status" > /var/tmp/robotwin-full-sft-6k-20261002/clean50_exit_code
exit "$status"
