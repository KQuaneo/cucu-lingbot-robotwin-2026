from pathlib import Path
import ast
import datetime
import hashlib
import json
import subprocess

BASE = Path('/var/tmp/robotwin-6k-fp32-20261004')
EXP = Path('/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin')
RUN = 'full_sft_6000_clean50x10_fp32_8gpu_video_20261004'
BASE.mkdir(parents=True, exist_ok=True)
original = (EXP / 'scripts/run_clean_benchmark.py').read_text()
script = original
old = 'server_script = root / "experiments/lingbot_vla_v2_6b_robotwin/scripts/launch_official_server.sh"'
assert script.count(old) == 1
script = script.replace(old, 'server_script = Path("/var/tmp/robotwin-6k-fp32-20261004/launch_fp32.sh")')
old = '    work_items = [(task_config, task) for task_config in task_configs for task in tasks]'
assert script.count(old) == 1
script = script.replace(old, '''    previous_times = {}
    prior = Path('/workspace/runtime/outputs/full_sft_6000_clean50x10_8gpu_video/task_times.tsv')
    for line in prior.read_text().splitlines():
        fields = line.split('\\t')
        previous_times[fields[1]] = int(fields[5])
    tasks.sort(key=lambda task: -previous_times.get(task, 0))
    work_items = [(task_config, task) for task_config in task_configs for task in tasks]''')
script = script.replace('    started_perf = time.perf_counter()', """    def deadline_stop(signum, frame):
        print('FP32 deadline/interruption: saving completed tasks and stopping only this run', flush=True)
        cleanup()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, deadline_stop)
    deadline = datetime.fromisoformat('2026-10-04T09:35:00+08:00')
    timer = threading.Timer(max(1, (deadline - utc_now()).total_seconds()), lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True
    timer.start()
    started_perf = time.perf_counter()""")
ast.parse(script)
(BASE / 'run_fp32_benchmark.py').write_text(script)
launcher = (EXP / 'scripts/launch_official_server.sh').read_text()
launcher = launcher.replace('root=$(cd -- "$script_dir/.." && pwd)', 'root=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin')
launcher = launcher.replace('--use_bf16 True', '--use_bf16 False').replace('--use_fp32 False', '--use_fp32 True')
launcher = launcher.replace('exec "$python_bin" -m deploy.lingbot_vla_v2_policy', 'exec "$python_bin" /var/tmp/robotwin-6k-fp32-20261004/serve_fp32.py')
(BASE / 'launch_fp32.sh').write_text(launcher)
audit = '''import collections, runpy, torch, sys
sys.path.insert(0, '/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2')
original_cuda = torch.nn.Module.cuda
def audited_cuda(self, *args, **kwargs):
    result = original_cuda(self, *args, **kwargs)
    if type(self).__name__ == 'LingBotVlaV2InferencePolicy':
        parameters = collections.Counter(str(p.dtype) for p in self.parameters() if p.is_floating_point())
        buffers = collections.Counter(str(b.dtype) for b in self.buffers() if b.is_floating_point())
        print('FP32_DTYPE_AUDIT parameters=' + str(dict(parameters)) + ' buffers=' + str(dict(buffers)), flush=True)
        assert parameters and set(parameters) == {'torch.float32'}, parameters
        assert not (set(buffers) - {'torch.float32'}), buffers
    return result
torch.nn.Module.cuda = audited_cuda
runpy.run_module('deploy.lingbot_vla_v2_policy', run_name='__main__')
'''
ast.parse(audit)
(BASE / 'serve_fp32.py').write_text(audit)
run = '''#!/usr/bin/env bash
set -euo pipefail
base=/var/tmp/robotwin-6k-fp32-20261004
export AITER_TRITON_ONLY=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE PYTHONUNBUFFERED=1
export LINGBOTVLA_TRAINING_CONFIG=/workspace/runtime/models/full_sft_6000/lingbotvla_cli.yaml
export TMPDIR="$base/eval_cache" TMP="$base/eval_cache" TEMP="$base/eval_cache"
mkdir -p "$TMPDIR"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
cd /RoboTwin
set +e
/opt/robotwin-env/bin/python -u "$base/run_fp32_benchmark.py" --gpu-count 8 --episodes 10 --task-config demo_clean --run-name full_sft_6000_clean50x10_fp32_8gpu_video_20261004 --model-path /workspace/runtime/models/full_sft_6000 --base-port 13620 --expert-check --accept-expert-info-on-failure --video
status=$?
printf '%s\\n' "$status" > "$base/exit_code"
exit "$status"
'''
(BASE / 'run.sh').write_text(run)
subprocess.run(['bash', '-n', str(BASE / 'run.sh')], check=True)
subprocess.run(['bash', '-n', str(BASE / 'launch_fp32.sh')], check=True)
subprocess.run(['/opt/robotwin-env/bin/python', str(BASE / 'run_fp32_benchmark.py'), '--help'], check=True, stdout=subprocess.DEVNULL)
manifest = {'run_name': RUN, 'model': 'our_full_sft_6000', 'model_path': '/workspace/runtime/models/full_sft_6000', 'precision': 'FP32', 'gpu_count': 8, 'episodes_per_task': 10, 'task_count': 50, 'seed': 0, 'task_config': 'demo_clean', 'seen': True, 'expert_check': True, 'accept_expert_info_on_failure': True, 'video': True, 'use_length': 25, 'eager': True, 'use_compile': False, 'gpu_scheduling': 'all 8 GPUs start immediately; official BF16 cancelled by user', 'task_order': 'descending previous BF16 task time', 'hard_stop_hong_kong': '2026-10-04T09:35:00+08:00', 'prepared_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'sha256': {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in ['run_fp32_benchmark.py', 'launch_fp32.sh', 'serve_fp32.py', 'run.sh']}}
(BASE / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest))
