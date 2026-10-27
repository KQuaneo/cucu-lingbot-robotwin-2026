#!/usr/bin/env bash
cd /RoboTwin
export HIP_VISIBLE_DEVICES=0 ROBOTWIN_DISABLE_CUROBO=1 ROBOTWIN_EE_PLANNER=mplib PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export AITER_TRITON_ONLY=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
/opt/robotwin-env/bin/python scripts/eval_policy_xpolicylab.py --task_name adjust_bottle --task_config demo_clean --policy_name LingBot-VLA-v2 --protocol lingbot_vla_v2 --host 127.0.0.1 --port 13410 --device_id 0 --seed 0 --test_num 10 --expert_check true --accept_expert_info_on_failure true --episode_info_output /workspace/runtime/outputs/logs/full_sft_6000_adjust_bottle_episodes.jsonl --eval_batch false --additional_info eval_video_log=false
status=$?
printf "%s\n" "$status" > /var/tmp/robotwin-full-sft-6k-20261002/eval_exit_code
exit "$status"
