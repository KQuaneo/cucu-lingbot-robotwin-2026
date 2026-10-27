from pathlib import Path
import ast
import difflib
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import yaml
from safetensors import safe_open

base = Path('/var/tmp/robotwin-official-weak5-20261004')
base.mkdir(parents=True, exist_ok=True)
source = Path('/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py')
original = source.read_text()
modified = original.replace('choices=(4, 8)', 'choices=(4, 5, 8)', 1)
old = '    parser.add_argument("--episodes", type=int, default=10)'
new = old + '\n    parser.add_argument("--tasks", nargs="+", default=None, help="Select tasks after validating the complete 50-task catalog.")'
assert modified.count(old) == 1
modified = modified.replace(old, new, 1)
old = '        raise RuntimeError(f"Expected 50 unique tasks, got {len(tasks)}")'
new = old + '''
    if args.tasks is not None:
        if len(set(args.tasks)) != len(args.tasks):
            raise ValueError("--tasks contains duplicates")
        unknown_tasks = sorted(set(args.tasks) - set(tasks))
        if unknown_tasks:
            raise ValueError(f"Unknown tasks: {unknown_tasks}")
        tasks = list(args.tasks)'''
assert modified.count(old) == 1
modified = modified.replace(old, new, 1)
ast.parse(modified)
wrapper = base / 'run_selected_benchmark.py'
if wrapper.exists():
    assert wrapper.read_text() == modified, 'Existing wrapper differs; refusing overwrite'
else:
    wrapper.write_text(modified)

tasks = ['put_object_cabinet', 'stack_blocks_three', 'handover_block', 'move_stapler_pad', 'pick_dual_bottles']
all_tasks = yaml.safe_load(Path('/RoboTwin/env_cfg/eval/all_tasks.yml').read_text())['tasks']
assert len(all_tasks) == len(set(all_tasks)) == 50
assert all(x in all_tasks for x in tasks)
official = Path('/models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b-robotwin')
model = official / 'checkpoints/global_step_50000/hf_ckpt'
config = official / 'lingbotvla_cli.yaml'
source_root = Path('/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2')
index = json.loads((model / 'model.safetensors.index.json').read_text())
shards = sorted(set(index['weight_map'].values()))
weights = []
all_keys = set()
for name in shards:
    p = model / name
    assert p.is_file() and p.stat().st_size > 0
    with safe_open(p, framework='pt', device='cpu') as f:
        keys = set(f.keys())
        assert not (keys & all_keys)
        all_keys.update(keys)
        dtypes = sorted({str(f.get_slice(k).get_dtype()) for k in keys})
    weights.append({'name': name, 'bytes': p.stat().st_size, 'keys': len(keys), 'storage_dtypes': dtypes})
assert all_keys == set(index['weight_map'])

launcher = '''#!/usr/bin/env bash
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
/opt/robotwin-env/bin/python -u "$base/run_selected_benchmark.py" \\
  --gpu-count 5 --episodes 10 --task-config demo_clean \\
  --tasks put_object_cabinet stack_blocks_three handover_block move_stapler_pad pick_dual_bottles \\
  --run-name official_robotwin_weak5_bf16_10x_20261004 \\
  --runtime-dir /workspace/runtime \\
  --model-path /models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt \\
  --base-port 13520 --expert-check --accept-expert-info-on-failure --video
status=$?
printf '%s\\n' "$status" > "$base/exit_code"
exit "$status"
'''
run_sh = base / 'run.sh'
if run_sh.exists():
    assert run_sh.read_text() == launcher, 'Existing launcher differs; refusing overwrite'
else:
    run_sh.write_text(launcher)
    run_sh.chmod(0o755)
subprocess.run(['bash', '-n', str(run_sh)], check=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

baseline_successes = dict(zip(tasks, [0, 0, 1, 1, 1]))
baseline_durations = dict(zip(tasks, [1358, 1882, 2571, 1040, 4310]))
cfg = yaml.safe_load(config.read_text())
norm = source_root / 'assets/norm_stats/robotwin.json'
qwen = Path('/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/Qwen3-VL-4B-Instruct-config-tokenizer')
assert norm.is_file()
assert (qwen / 'config.json').is_file() and (qwen / 'tokenizer.json').is_file()
manifest = {
    'schema_version': 1,
    'status': 'prepared_not_started',
    'prepared_utc': datetime.now(timezone.utc).isoformat(),
    'experiment_id': 'official_robotwin_weak5_bf16_10x_20261004',
    'model_name': 'robbyant/lingbot-vla-v2-6b-robotwin',
    'checkpoint_step': 50000,
    'model_path': str(model),
    'training_config_path': str(config),
    'training_config_sha256': sha(config),
    'source_script': str(source),
    'source_script_sha256': sha(source),
    'wrapper_path': str(wrapper),
    'wrapper_sha256': sha(wrapper),
    'launcher_sha256': sha(run_sh),
    'loader_sha256': sha(source_root / 'deploy/lingbot_vla_v2_policy.py'),
    'server_launcher_sha256': sha(source.parent / 'launch_official_server.sh'),
    'eval_script_sha256': sha(Path('/RoboTwin/scripts/eval_policy_xpolicylab.py')),
    'wrapper_changes': ['GPU choices allow 5', '--tasks list argument', 'validate 50-task catalog then check duplicates/unknown names and select supplied tasks'],
    'task_catalog_count': len(all_tasks),
    'tasks': tasks,
    'evaluation': {'gpu_count': 5, 'gpu_ids': [0, 1, 2, 3, 4], 'ports': [13520, 13521, 13522, 13523, 13524], 'episodes_per_task': 10, 'task_config': 'demo_clean', 'instruction_type': 'seen', 'seed_argument': 0, 'expected_valid_seeds': list(range(100000, 100010)), 'expert_check': True, 'accept_expert_info_on_failure': True, 'video': True, 'eval_batch': False, 'use_length': 25, 'use_compile': False, 'attention_implementation': 'eager', 'use_bf16': True, 'use_fp32': False, 'valid_episode_retry_patch': True},
    'official_training_enable_fp32': cfg['train']['enable_fp32'],
    'precision_control_note': 'CLI enables whole-model BF16 conversion after strict weight loading; official enable_fp32 is a training/save flag and is not read by inference modules. Stored official weights are F32.',
    'norm_path': str(norm),
    'norm_sha256': sha(norm),
    'norm_resolution_note': 'Official data.norm_stats_file=null; FeatureTransform resolves configs/robot_configs/robotwin.yaml norm_stats to source-root assets/norm_stats/robotwin.json.',
    'tokenizer_override_path': str(qwen.resolve()),
    'weight_files': weights,
    'weight_tensor_count': len(all_keys),
    'weight_tensor_bytes': index['metadata']['total_size'],
    'baseline_full_sft_6000': [{'task': t, 'successes': baseline_successes[t], 'episodes': 10, 'actual_seeds': list(range(100000, 100010)), 'elapsed_seconds': baseline_durations[t]} for t in tasks],
    'output_dir': '/workspace/runtime/outputs/official_robotwin_weak5_bf16_10x_20261004',
    'log_dir': '/workspace/runtime/outputs/official_robotwin_weak5_bf16_10x_20261004/logs',
    'launch_log': '/workspace/runtime/outputs/logs/official_robotwin_weak5_bf16_10x_20261004.launch.log',
    'command': 'nohup bash /var/tmp/robotwin-official-weak5-20261004/run.sh > /workspace/runtime/outputs/logs/official_robotwin_weak5_bf16_10x_20261004.launch.log 2>&1 < /dev/null &',
    'verification': 'AST syntax, bash -n, catalog uniqueness, task presence, weight shard header/index match and dependency paths checked. No GPU initialization or inference performed.',
}
(base / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'wrapper_sha256': sha(wrapper), 'launcher_sha256': sha(run_sh), 'tasks': tasks, 'weight_headers': len(all_keys), 'prepared': True}, indent=2))
print(''.join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True), fromfile=str(source), tofile=str(wrapper))))
