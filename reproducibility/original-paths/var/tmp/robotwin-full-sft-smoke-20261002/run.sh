#!/usr/bin/env bash
export AITER_TRITON_ONLY=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export TMPDIR=/var/tmp/robotwin-full-sft-smoke-20261002/cache TMP=/var/tmp/robotwin-full-sft-smoke-20261002/cache TEMP=/var/tmp/robotwin-full-sft-smoke-20261002/cache
export TRITON_CACHE_DIR=/var/tmp/robotwin-full-sft-smoke-20261002/cache/triton TORCHINDUCTOR_CACHE_DIR=/var/tmp/robotwin-full-sft-smoke-20261002/cache/inductor
export GPU_COUNT=8 MAX_STEPS=10 SAVE_STEPS=10 TEACHER_MODE=full OPTIMIZER=adamw MICRO_BATCH_SIZE=16 GLOBAL_BATCH_SIZE=256
export OUTPUT_DIR=/var/tmp/robotwin-full-sft-smoke-20261002/output LOG_FILE=/workspace/runtime/outputs/logs/full_sft_8gpu_10steps_20261002.log
bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/train_full_sft.sh
status=$?
printf "%s\n" "$status" > /var/tmp/robotwin-full-sft-smoke-20261002/exit_code
exit "$status"
