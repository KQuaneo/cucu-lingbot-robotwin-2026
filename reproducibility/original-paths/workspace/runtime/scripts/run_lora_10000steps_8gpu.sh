#!/usr/bin/env bash
set -euo pipefail

cd /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
source /opt/robotwin-env/bin/activate

mkdir -p /workspace/runtime/outputs/logs
mkdir -p /root/robotwin-tmp

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TMPDIR=/root/robotwin-tmp
export TMP=/root/robotwin-tmp
export TEMP=/root/robotwin-tmp

python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  -m tasks.vla.train_lingbotvla \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/lingbotvla_cli.yaml \
  --train.data_parallel_shard_size 8 \
  --train.gradient_accumulation_steps 1 \
  --train.global_batch_size 8 \
  --train.max_steps 10000 \
  --train.save_steps 10000 \
  --train.output_dir /workspace/runtime/outputs/lora_10000steps_8gpu
