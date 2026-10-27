#!/usr/bin/env bash
# PREPARED ONLY. Does not execute training without explicit --execute.
set -euo pipefail
if [[ "${1:-}" != "--execute" ]]; then
  echo '未启动。仅在获得训练授权并完成空间/路径检查后，传入 --execute CONFIG_ABSOLUTE_PATH'
  exit 0
fi
cfg=${2:?Provide an absolute YAML path}
[[ "$cfg" = /* && -f "$cfg" ]] || { echo '配置必须是存在的绝对路径'; exit 2; }
export AITER_TRITON_ONLY=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export TMPDIR=/var/tmp/cucu-augmentation-cache
mkdir -p "$TMPDIR"
cd /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
exec /opt/robotwin-env/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 -m tasks.vla.train_lingbotvla "$cfg"
