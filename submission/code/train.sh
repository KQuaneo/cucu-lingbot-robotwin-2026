#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export AITER_TRITON_ONLY=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export TMPDIR=${TMPDIR:-/var/tmp/cucu-training-cache}
mkdir -p "$TMPDIR"
cd /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
exec /opt/robotwin-env/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 -m tasks.vla.train_lingbotvla "$here/configs/实际训练配置.yaml" --train.output_dir "${OUTPUT_DIR:-/workspace/runtime/outputs/cucu_full_sft_6000_reproduce}" --train.enable_resume false --train.max_steps 6000
