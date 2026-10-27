#!/usr/bin/env python3
"""有限只读采集官方 BF16 五任务对照；不启动、停止或改写远端评测。

默认采集一次；--watch 每150秒采集，完整结果或截止时间到达后结束。
--offline 用本地最近快照重新生成报告。录像仅建立准确清单，不下载。
"""
import argparse
import csv
import datetime as dt
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time

RUN_NAME = 'full_sft_6000_clean50x10_fp32_8gpu_video_20261004'
REMOTE_RUN = '/workspace/runtime/outputs/' + RUN_NAME
TASKS = ['adjust_bottle', 'beat_block_hammer', 'blocks_ranking_rgb', 'blocks_ranking_size', 'click_alarmclock', 'click_bell', 'dump_bin_bigbin', 'grab_roller', 'handover_block', 'handover_mic', 'hanging_mug', 'lift_pot', 'move_can_pot', 'move_pillbottle_pad', 'move_playingcard_away', 'move_stapler_pad', 'open_laptop', 'open_microwave', 'pick_diverse_bottles', 'pick_dual_bottles', 'place_a2b_left', 'place_a2b_right', 'place_bread_basket', 'place_bread_skillet', 'place_burger_fries', 'place_can_basket', 'place_cans_plasticbox', 'place_container_plate', 'place_dual_shoes', 'place_empty_cup', 'place_fan', 'place_mouse_pad', 'place_object_basket', 'place_object_scale', 'place_object_stand', 'place_phone_stand', 'place_shoe', 'press_stapler', 'put_bottles_dustbin', 'put_object_cabinet', 'rotate_qrcode', 'scan_object', 'shake_bottle', 'shake_bottle_horizontally', 'stack_blocks_three', 'stack_blocks_two', 'stack_bowls_three', 'stack_bowls_two', 'stamp_seal', 'turn_switch']
HK = dt.timezone(dt.timedelta(hours=8))
DEFAULT_DEADLINE = '2026-10-04T08:45:00+08:00'
HERE = Path(__file__).resolve().parent
BACKUP_ROOT = HERE.parent.parent
RESULTS = BACKUP_ROOT / '结果' / 'FP32全任务'
ANSI = re.compile(r'\x1b\[[0-9;]*m')

REMOTE = r'''
import datetime, gzip, hashlib, json, re, subprocess, sys
from pathlib import Path
run=Path('/workspace/runtime/outputs/full_sft_6000_clean50x10_fp32_8gpu_video_20261004')
tasks=['adjust_bottle', 'beat_block_hammer', 'blocks_ranking_rgb', 'blocks_ranking_size', 'click_alarmclock', 'click_bell', 'dump_bin_bigbin', 'grab_roller', 'handover_block', 'handover_mic', 'hanging_mug', 'lift_pot', 'move_can_pot', 'move_pillbottle_pad', 'move_playingcard_away', 'move_stapler_pad', 'open_laptop', 'open_microwave', 'pick_diverse_bottles', 'pick_dual_bottles', 'place_a2b_left', 'place_a2b_right', 'place_bread_basket', 'place_bread_skillet', 'place_burger_fries', 'place_can_basket', 'place_cans_plasticbox', 'place_container_plate', 'place_dual_shoes', 'place_empty_cup', 'place_fan', 'place_mouse_pad', 'place_object_basket', 'place_object_scale', 'place_object_stand', 'place_phone_stand', 'place_shoe', 'press_stapler', 'put_bottles_dustbin', 'put_object_cabinet', 'rotate_qrcode', 'scan_object', 'shake_bottle', 'shake_bottle_horizontally', 'stack_blocks_three', 'stack_blocks_two', 'stack_bowls_three', 'stack_bowls_two', 'stamp_seal', 'turn_switch']
files={}; videos=[]; warnings=[]
all_done=all((run/'done'/('demo_clean__'+task+'.done')).is_file() for task in tasks)
for name in ['logs','episode_info','done']:
 p=run/name
 if p.is_dir():
  for f in sorted(p.rglob('*')):
   if f.is_file() and not f.is_symlink():
    files[str(f.relative_to(run))]=f.read_text(errors='replace')
for name in ['events.log','task_times.tsv','run_start_iso','failures.tsv']:
 p=run/name
 if p.is_file():files[name]=p.read_text(errors='replace')
for task in tasks:
 text=re.sub(r'\x1b\[[0-9;]*m','',files.get('logs/eval_demo_clean__'+task+'.log','')).replace('\r','\n')
 paths=re.findall(r'Data has been saved to ([^\n]+/_result\.txt)',text)
 row={'task':task,'selection_method':'final_result_path_only','result_path':None,'directory':None,'mp4_count':0,'mp4_bytes':0,'files':[],'warnings':[]}
 if paths:
  result=Path(paths[-1].strip())
  if not result.is_absolute():result=Path('/workspace/runtime')/result
  result=result.resolve()
  expected=(Path('/workspace/runtime/eval_result')/task/'LingBot-VLA-v2/demo_clean/default').resolve()
  if result.name!='_result.txt' or expected not in result.parents:
   row['warnings'].append('Logged result path outside expected task subtree; not collected')
  else:
   row['result_path']=str(result);row['directory']=str(result.parent)
   if result.is_file():files['eval_result/'+str(result.relative_to('/workspace/runtime/eval_result'))]=result.read_text(errors='replace')
   else:row['warnings'].append('Logged result file missing')
   if result.parent.is_dir():
    for f in sorted(result.parent.glob('*.mp4')):
     stat=f.stat();entry={'name':f.name,'remote_path':str(f),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':None}
     if (run/'done'/('demo_clean__'+task+'.done')).is_file():
      digest=hashlib.sha256()
      with f.open('rb') as handle:
       for block in iter(lambda:handle.read(1048576),b''):digest.update(block)
      entry['sha256']=digest.hexdigest()
      after=f.stat()
      if after.st_size!=stat.st_size or after.st_mtime_ns!=stat.st_mtime_ns:
       entry['sha256']=None;row['warnings'].append('Video changed while hashing: '+f.name)
     row['files'].append(entry)
    row['mp4_count']=len(row['files']);row['mp4_bytes']=sum(f['bytes'] for f in row['files'])
 videos.append(row)
prepare=Path('/var/tmp/robotwin-6k-fp32-20261004')
for name in ['manifest.json','run.sh','launch_receipt.json','exit_code','run_fp32_benchmark.py','launch_fp32.sh','serve_fp32.py']:
 p=prepare/name
 if p.is_file():files['preparation/'+name]=p.read_text(errors='replace')
try:
 ps=subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,pmem,args'],text=True,timeout=10)
 lines=[ps.splitlines()[0]]
 for line in ps.splitlines()[1:]:
  if 'full_sft_6000_clean50x10_fp32_8gpu_video_20261004' in line or 'robotwin-6k-fp32-20261004' in line or (('deploy.lingbot_vla_v2_policy' in line or 'eval_policy_xpolicylab.py' in line) and (any(str(p) in line for p in range(13620,13628)) or any(t in line for t in tasks))):lines.append(line)
 processes='\n'.join(lines)
except Exception as exc:processes='';warnings.append('process inventory: '+str(exc))
now=datetime.datetime.now(datetime.timezone.utc)
data={'schema_version':1,'run_name':run.name,'run_path':str(run),'run_exists':run.exists(),'snapshot_utc':now.isoformat(),'snapshot_hong_kong':now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'files':files,'video_inventory':videos,'processes':processes,'warnings':warnings}
sys.stdout.buffer.write(gzip.compress(json.dumps(data,ensure_ascii=False).encode(),compresslevel=6))
'''


def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)


def atomic_text(path, content):
    atomic_bytes(path, content.encode('utf-8'))


def save_json(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def capture():
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
               '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
               '-p', '31227', 'root@YOUR_SERVER_IP', '/opt/robotwin-env/bin/python -']
    result = subprocess.run(command, input=REMOTE.encode(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=100, check=False)
    if result.returncode:
        raise RuntimeError(f'SSH exit {result.returncode}: ' + result.stderr.decode(errors='replace')[-1600:])
    data = json.loads(gzip.decompress(result.stdout))
    if data.get('run_name') != RUN_NAME:
        raise ValueError('Unexpected run name in remote snapshot')
    atomic_bytes(RESULTS / '原始快照.json.gz', result.stdout)
    raw_root = RESULTS / '原始日志'
    for name, content in data['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe snapshot path: ' + name)
        atomic_text(raw_root / relative, content)
    atomic_text(RESULTS / '运行进程.txt', data['processes'] + '\n')
    save_json(RESULTS / '录像目录清单.json', {
        'run_name': RUN_NAME, 'run_path': REMOTE_RUN,
        'snapshot_utc': data['snapshot_utc'], 'tasks': data['video_inventory'],
        'notes': 'Only exact final result paths logged by this run; no directory fallback. Videos remain on server.'})
    return data


def parse_task(task, snapshot):
    files = snapshot['files']
    log_name = 'logs/eval_demo_clean__' + task + '.log'
    text = ANSI.sub('', files.get(log_name, '')).replace('\r', '\n')
    segments = []
    current = []
    previous = 0
    anomalies = []
    for match in re.finditer(r'Success rate:\s*(\d+)/(\d+)[^\n]*?current seed:\s*(\d+)', text):
        count, total, seed = map(int, match.groups())
        if total == 1 and current:
            segments.append(current)
            current = []
            previous = 0
        delta = count - previous
        success = bool(delta) if delta in (0, 1) else None
        if total != len(current) + 1 or success is None or count > total:
            anomalies.append(f'Unexpected cumulative rollout record {count}/{total} seed={seed}')
        current.append({'index': total - 1, 'seed': seed, 'success': success,
                        'cumulative_successes': count, 'cumulative_rollouts': total})
        previous = count
    if current:
        segments.append(current)
    episodes = segments[-1] if segments else []
    if len(segments) > 1:
        anomalies.append(f'{len(segments)} rollout segments found; report uses last segment')
    final_matches = list(re.finditer(r'Final success rate:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%', text))
    final = {'successes': int(final_matches[-1][1]), 'rollouts': int(final_matches[-1][2]),
             'logged_percent': float(final_matches[-1][3])} if final_matches else None
    done = 'done/demo_clean__' + task + '.done' in files
    successful_parse = len(episodes) == 10 and all(e['success'] is not None for e in episodes)
    complete = bool(done and final and final['rollouts'] == 10 and successful_parse
                    and sum(e['success'] for e in episodes) == final['successes'])
    if done and not complete:
        anomalies.append('Done marker exists but final/episode records do not yet agree')
    if final and episodes and (final['successes'] != episodes[-1]['cumulative_successes']
                              or final['rollouts'] != episodes[-1]['cumulative_rollouts']):
        anomalies.append('Final score and last cumulative rollout record differ')
    seeds = [e['seed'] for e in episodes]
    if len(seeds) != len(set(seeds)):
        anomalies.append('Duplicate seeds in current rollout segment; paired seed summary omitted')
    if complete and seeds != BASELINE[task]['actual_seeds']:
        anomalies.append('Actual valid seeds differ from prior BF16; compare only logged overlapping seeds')
    total = episodes[-1]['cumulative_rollouts'] if episodes else 0
    count = episodes[-1]['cumulative_successes'] if episodes else None
    if complete:
        count, total = final['successes'], final['rollouts']
    if complete:
        status = 'completed'
    elif log_name in files:
        status = 'running_or_partial'
    elif snapshot['run_exists']:
        status = 'waiting_for_task'
    else:
        status = 'not_started'
    prechecks = []
    for line in files.get('episode_info/demo_clean__' + task + '.jsonl', '').splitlines():
        try:
            prechecks.append(json.loads(line))
        except json.JSONDecodeError:
            anomalies.append('Incomplete/unparseable expert precheck JSON line')
    timing = None
    for line in files.get('task_times.tsv', '').splitlines():
        values = line.split('\t')
        if len(values) == 7 and values[1] == task:
            try:
                timing = {'gpu': int(values[2]), 'start_utc': values[3], 'end_utc': values[4],
                          'elapsed_seconds': int(values[5]), 'exit_code': int(values[6])}
            except ValueError:
                anomalies.append('Unparseable task timing row')
    return {'task': task, 'status': status, 'completed': complete,
            'successes': count, 'policy_rollouts': total, 'success_rate': count / total if total else None,
            'episodes': episodes, 'logged_rollout_segments': len(segments), 'final_log_result': final,
            'done_marker': done, 'actual_seeds': seeds, 'task_time': timing,
            'remote_log_file': REMOTE_RUN + '/' + log_name, 'expert_prechecks': prechecks,
            'errors': {key: text.count(key) for key in ['Policy rollout error', 'Traceback (most recent call last)',
                        'OutOfMemoryError', 'CUDA out of memory', 'No space left on device']},
            'anomalies': anomalies}


BASELINE = {}
with (BACKUP_ROOT / '报告/per_task_results.csv').open(encoding='utf-8-sig') as handle:
    for row in csv.DictReader(handle):
        seeds = [int(x) for x in row['actual_seeds'].split(';') if x]
        failed = {int(x) for x in row['failed_seeds'].split(';') if x}
        BASELINE[row['task']] = {'successes': int(row['successes']), 'episodes': 10,
                                'actual_seeds': seeds, 'outcomes': {seed: seed not in failed for seed in seeds}}

def render(snapshot):
    rows = [parse_task(task, snapshot) for task in TASKS]
    for row in rows:
        baseline = BASELINE[row['task']]
        row['bf16_baseline'] = baseline
        row['matched_seeds'] = [{'seed': e['seed'], 'bf16_success': baseline['outcomes'][e['seed']],
                                 'fp32_success': e['success']} for e in row['episodes']
                                if e['seed'] in baseline['outcomes'] and e['success'] is not None]
        row['final_difference_percentage_points'] = ((row['successes'] - baseline['successes']) * 10
                                                      if row['completed'] else None)
    finished = [r for r in rows if r['completed']]
    successes = sum(r['successes'] for r in finished)
    episodes = sum(r['policy_rollouts'] for r in finished)
    exit_text = snapshot['files'].get('preparation/exit_code', '').strip()
    exit_code = int(exit_text) if re.fullmatch(r'-?\d+', exit_text) else None
    complete = len(finished) == 50 and episodes == 500 and exit_code == 0
    ended = complete or exit_code is not None
    state = 'completed' if complete else 'ended_incomplete' if ended else 'running'
    report = {'run_name': RUN_NAME, 'status': state, 'complete': complete,
              'snapshot_utc': snapshot['snapshot_utc'], 'snapshot_hong_kong': snapshot['snapshot_hong_kong'],
              'completed_tasks': len(finished), 'planned_tasks': 50, 'completed_task_successes': successes,
              'completed_task_rollouts': episodes, 'all_logged_rollouts': sum(r['policy_rollouts'] for r in rows),
              'success_rate': successes / episodes if episodes else None,
              'bf16_on_completed_tasks': {'successes': sum(BASELINE[r['task']]['successes'] for r in finished), 'episodes': episodes},
              'original_bf16_all_tasks': {'successes': 276, 'episodes': 500, 'rate': 0.552},
              'precision': 'FP32', 'exit_code': exit_code, 'tasks': rows,
              'conditions': 'same 6k HF checkpoint/config, demo_clean, seen, seed arg0, valid retry, 10/task, expert+accept info, video, eager, use_length25, no compile',
              'limits': ['Actual valid seeds can differ after setup retries; preserve them and compare overlapping seeds.',
                         'Instruction sampling and policy noise were not fixed in the prior BF16 run.',
                         'FP32 casts the existing exported checkpoint; it cannot recover precision previously lost when saving weights.',
                         'Ten episodes per task; descriptive results do not prove causality.'],
              'server_anomalies': {name: {key: content.count(key) for key in ['Traceback (most recent call last)', 'OutOfMemoryError', 'CUDA out of memory']}
                                   for name, content in snapshot['files'].items() if name.startswith('logs/server_gpu')}}
    save_json(HERE / '结果.json', report)
    columns = ['task', 'status', 'bf16_successes', 'fp32_successes', 'fp32_rollouts', 'final_difference_percentage_points', 'actual_seeds', 'bf16_actual_seeds']
    output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=columns); writer.writeheader()
    table = []
    for row in rows:
        writer.writerow({'task': row['task'], 'status': row['status'], 'bf16_successes': BASELINE[row['task']]['successes'],
                         'fp32_successes': row['successes'] if row['successes'] is not None else '',
                         'fp32_rollouts': row['policy_rollouts'], 'final_difference_percentage_points': row['final_difference_percentage_points'],
                         'actual_seeds': ';'.join(map(str, row['actual_seeds'])), 'bf16_actual_seeds': ';'.join(map(str, BASELINE[row['task']]['actual_seeds']))})
        score = f"{row['successes']}/{row['policy_rollouts']}" if row['policy_rollouts'] else '—'
        table.append(f"| {row['task']} | {BASELINE[row['task']]['successes']}/10 | {score} | {'完成' if row['completed'] else '等待' if row['status']=='waiting_for_task' else '部分结果'} |")
    atomic_text(HERE / 'per_task_fp32_vs_bf16.csv', '\ufeff' + output.getvalue())
    rate = f'{successes/episodes*100:.2f}%' if episodes else '尚无完整任务'
    text = f'''# 6k 模型 FP32 全任务评测

更新时间：{snapshot['snapshot_hong_kong']}。状态：{state}。完整任务 **{len(finished)}/50**；完整任务合计 **{successes}/{episodes}（{rate}）**。全部已记录回合 {report['all_logged_rollouts']}/500。退出码：{exit_code}。

原 BF16 全任务：276/500，55.20%。进行中的分数仍会变化；未完成任务不作为0分。比较总体成绩需要两轮都完整结束。

| 任务 | 原 BF16 | 当前 FP32 | 状态 |
|---|---:|---:|---|
{chr(10).join(table)}

## 条件与限制

使用同一6k导出checkpoint和模型配置，仅将推理设置改为FP32；50个demo_clean任务，各10个有效回合，seen指令、seed参数0、expert预检与失败信息接纳、valid retry patch、录像、eager、use_length25、不编译。优先执行原BF16耗时最长的任务。

实际有效种子可能因初始化重试不同；逐任务CSV及JSON保留两轮种子与重叠种子结果。指令抽样、策略噪声和规划随机性可能不同。FP32转换已有权重无法恢复保存时损失的精度。每项仅10次，暂不做显著性或因果判断。

原始日志和本次准确录像目录位于 ../../结果/FP32全任务/。录像按已完成任务持续备份到 ../../录像/FP32全任务/。旧BF16结果与录像保留。

服务器预计10:10回收，评测设置09:35（香港时间）截止，给最终备份预留时间；提前完成则正常退出。未完成时明确记录部分结果。

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-04
- Verification Status: {'VERIFIED (execution/log consistency only)' if complete else 'UNVERIFIED'}
- Version Label: full-sft-6000-fp32-clean50x10-v1

## ExperimentResult

- ID: {RUN_NAME}
- Status: {state}
- Working Directory: /RoboTwin
- Exit Code: {exit_code}
- Output: 本报告、结果.json、CSV、原始日志、已闭合录像。
'''
    atomic_text(HERE / '评测报告.md', text)
    return report, ended

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--watch', action='store_true'); parser.add_argument('--interval', type=int, default=180)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    deadline = dt.datetime.fromisoformat('2026-10-04T10:00:00+08:00')
    with (RESULTS / '监控.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while dt.datetime.now(dt.timezone.utc) < deadline:
            try:
                snapshot = capture(); report, ended = render(snapshot)
                print(json.dumps({k: report[k] for k in ['status', 'completed_tasks', 'all_logged_rollouts', 'completed_task_successes', 'completed_task_rollouts', 'snapshot_hong_kong']}, ensure_ascii=False), flush=True)
                if ended:
                    archive = RESULTS / '最终原始快照.tar.gz'
                    with tarfile.open(archive, 'w:gz') as handle:
                        for p in [RESULTS / '原始日志', RESULTS / '录像目录清单.json', RESULTS / '原始快照.json.gz']:
                            handle.add(p, arcname=p.name)
                    save_json(RESULTS / '已结束.json', {'run_name': RUN_NAME, 'complete': report['complete'], 'status': report['status'], 'exit_code': report['exit_code'], 'snapshot_utc': report['snapshot_utc'], 'archive': str(archive), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
                    if report['complete']: save_json(RESULTS / '采集完成.json', report)
                    return
            except Exception as exc:
                print('Capture error: ' + repr(exc), flush=True)
                save_json(RESULTS / '采集错误.json', {'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'error': repr(exc)})
            if not args.watch: return
            time.sleep(min(args.interval, max(1, (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds())))
        save_json(RESULTS / '监控截止.json', {'status': 'deadline_reached', 'time_hong_kong': dt.datetime.now(HK).isoformat()})

if __name__ == '__main__': main()
