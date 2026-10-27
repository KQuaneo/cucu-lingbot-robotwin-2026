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

RUN_NAME = 'official_robotwin_weak5_bf16_10x_20261004'
REMOTE_RUN = '/workspace/runtime/outputs/' + RUN_NAME
TASKS = ['put_object_cabinet', 'stack_blocks_three', 'handover_block',
         'move_stapler_pad', 'pick_dual_bottles']
HK = dt.timezone(dt.timedelta(hours=8))
DEFAULT_DEADLINE = '2026-10-04T08:45:00+08:00'
HERE = Path(__file__).resolve().parent
BACKUP_ROOT = HERE.parent.parent
RESULTS = BACKUP_ROOT / '结果' / '官方对照'
ANSI = re.compile(r'\x1b\[[0-9;]*m')

REMOTE = r'''
import datetime, gzip, hashlib, json, re, subprocess, sys
from pathlib import Path
run=Path('/workspace/runtime/outputs/official_robotwin_weak5_bf16_10x_20261004')
tasks=['put_object_cabinet','stack_blocks_three','handover_block','move_stapler_pad','pick_dual_bottles']
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
     if all_done:
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
prepare=Path('/var/tmp/robotwin-official-weak5-20261004')
for name in ['manifest.json','run.sh','launch_receipt.json','exit_code']:
 p=prepare/name
 if p.is_file():files['preparation/'+name]=p.read_text(errors='replace')
try:
 ps=subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,pmem,args'],text=True,timeout=10)
 lines=[ps.splitlines()[0]]
 for line in ps.splitlines()[1:]:
  if 'official_robotwin_weak5_bf16_10x_20261004' in line or 'robotwin-official-weak5-20261004' in line or (('deploy.lingbot_vla_v2_policy' in line or 'eval_policy_xpolicylab.py' in line) and (any(str(p) in line for p in range(13520,13525)) or any(t in line for t in tasks))):lines.append(line)
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
                            stderr=subprocess.PIPE, timeout=55, check=False)
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
    if complete and seeds != list(range(100000, 100010)):
        anomalies.append('Actual seeds differ from expected 100000..100009')
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


def build_report(snapshot):
    baseline = json.loads((HERE.parent / '实验结果.json').read_text())
    reference = json.loads((HERE / '官方参考.json').read_text())
    our_tasks = {row['task']: row for row in baseline['tasks']}
    rows = []
    for task in TASKS:
        row = parse_task(task, snapshot)
        ours = our_tasks[task]
        row['ours'] = {key: ours[key] for key in ['successes', 'policy_rollouts', 'success_rate', 'episodes']}
        row['ours']['source_run'] = 'full_sft_6000_clean50x10_8gpu_video'
        row['official_published_clean_percent'] = reference['published_clean_percent'][task]
        ours_by_seed = {e['seed']: e['success'] for e in ours['episodes']}
        if len(row['actual_seeds']) == len(set(row['actual_seeds'])):
            pairs = [{'seed': e['seed'], 'ours_success': ours_by_seed[e['seed']],
                      'official_success': e['success']} for e in row['episodes']
                     if e['seed'] in ours_by_seed and e['success'] is not None]
        else:
            pairs = []
        row['matched_seed_records'] = pairs
        row['paired_counts'] = {name: sum(condition(p) for p in pairs) for name, condition in {
            'both_success': lambda p: p['ours_success'] and p['official_success'],
            'both_failure': lambda p: not p['ours_success'] and not p['official_success'],
            'ours_failure_official_success': lambda p: not p['ours_success'] and p['official_success'],
            'ours_success_official_failure': lambda p: p['ours_success'] and not p['official_success'],
        }.items()}
        row['same_seed_sequence_as_ours'] = row['actual_seeds'] == [e['seed'] for e in ours['episodes']] if row['completed'] else None
        row['final_difference_percentage_points'] = (row['success_rate'] - ours['success_rate']) * 100 if row['completed'] else None
        rows.append(row)
    finished = [row for row in rows if row['completed']]
    complete = len(finished) == 5
    official_successes = sum(row['successes'] for row in finished)
    official_rollouts = sum(row['policy_rollouts'] for row in finished)
    ours_matched_successes = sum(row['ours']['successes'] for row in finished)
    ours_matched_rollouts = sum(row['ours']['policy_rollouts'] for row in finished)
    report = {'schema_version': 1, 'run_name': RUN_NAME, 'run_path': REMOTE_RUN,
              'snapshot_utc': snapshot['snapshot_utc'], 'snapshot_hong_kong': snapshot['snapshot_hong_kong'],
              'status': 'final' if complete else ('partial' if snapshot['run_exists'] else 'not_started'),
              'completed_tasks': len(finished), 'planned_tasks': 5, 'planned_episodes_per_task': 10,
              'completed_task_official_successes': official_successes if finished else None,
              'completed_task_official_rollouts': official_rollouts,
              'completed_task_official_rate': official_successes / official_rollouts if official_rollouts else None,
              'ours_on_completed_tasks': {'successes': ours_matched_successes if finished else None,
                                          'rollouts': ours_matched_rollouts,
                                          'rate': ours_matched_successes / ours_matched_rollouts if ours_matched_rollouts else None},
              'ours_selected_five': {'successes': sum(our_tasks[t]['successes'] for t in TASKS),
                                    'rollouts': sum(our_tasks[t]['policy_rollouts'] for t in TASKS)},
              'configured_conditions': {'model': 'robbyant/lingbot-vla-v2-6b-robotwin',
                  'checkpoint': '/models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt',
                  'precision': 'BF16', 'task_config': 'demo_clean', 'instruction_type': 'seen',
                  'seed_argument': 0, 'expected_actual_seeds': list(range(100000, 100010)),
                  'expert_check': True, 'accept_expert_info_on_failure': True,
                  'valid_episode_retry_patch': True, 'eval_batch': False, 'use_length': 25,
                  'use_compile': False, 'video': True},
              'reference': reference, 'tasks': rows, 'remote_processes': snapshot['processes'],
              'snapshot_warnings': snapshot['warnings'],
              'comparison_limits': ['Same scenario seed does not guarantee identical instruction sampling, policy noise or simulator/planner trajectories.',
                  'Ten episodes per task is a diagnostic sample; no significance test or causal claim is made.',
                  'Published reference uses FP32 and 100 episodes per task; local BF16/10 results are not a direct reproduction.',
                  'Selected tasks were chosen for low performance of our model, so results cannot estimate all-task performance.']}
    save_json(HERE / '对照结果.json', report)
    fields = ['task', 'status', 'ours_successes', 'ours_rollouts', 'official_bf16_successes',
              'official_bf16_rollouts', 'official_bf16_percent', 'final_difference_percentage_points',
              'official_published_clean_fp32_percent', 'actual_seeds', 'same_final_seed_sequence',
              'ours_failed_official_succeeded', 'both_failed', 'official_failed_ours_succeeded', 'both_succeeded']
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        pairs = row['paired_counts']
        writer.writerow({'task': row['task'], 'status': row['status'], 'ours_successes': row['ours']['successes'],
            'ours_rollouts': row['ours']['policy_rollouts'],
            'official_bf16_successes': row['successes'] if row['successes'] is not None else '',
            'official_bf16_rollouts': row['policy_rollouts'],
            'official_bf16_percent': round(row['success_rate'] * 100, 2) if row['success_rate'] is not None else '',
            'final_difference_percentage_points': row['final_difference_percentage_points'] if row['completed'] else '',
            'official_published_clean_fp32_percent': row['official_published_clean_percent'],
            'actual_seeds': ';'.join(map(str, row['actual_seeds'])),
            'same_final_seed_sequence': row['same_seed_sequence_as_ours'] if row['completed'] else '',
            'ours_failed_official_succeeded': pairs['ours_failure_official_success'],
            'both_failed': pairs['both_failure'], 'official_failed_ours_succeeded': pairs['ours_success_official_failure'],
            'both_succeeded': pairs['both_success']})
    atomic_bytes(HERE / 'per_task_comparison.csv', b'\xef\xbb\xbf' + buffer.getvalue().encode('utf-8'))
    labels = {'completed': '完成', 'running_or_partial': '进行中/部分',
              'waiting_for_task': '等待启动', 'not_started': '未开始'}
    table = []
    for row in rows:
        score = f"{row['successes']}/{row['policy_rollouts']}" if row['policy_rollouts'] else '—'
        percent = f"{row['success_rate']*100:.0f}%" if row['policy_rollouts'] else '—'
        difference = f"{row['final_difference_percentage_points']:+.0f}" if row['completed'] else '—'
        table.append(f"| {row['task']} | {labels[row['status']]} | {row['ours']['successes']}/10 | {score} ({percent}) | {difference} | {row['official_published_clean_percent']}% |")
    aggregate = (f"已完成任务合计：官方 **{official_successes}/{official_rollouts}（{official_successes/official_rollouts*100:.1f}%）**；我们在相同已完成任务上 **{ours_matched_successes}/{ours_matched_rollouts}（{ours_matched_successes/ours_matched_rollouts*100:.1f}%）**。" if official_rollouts else '官方对照尚无完整任务结果；未开始/未完成任务不作0分处理。')
    episode_lines = []
    for row in rows:
        pairs = row['paired_counts']
        episode_lines.append(f"- `{row['task']}`：记录{len(row['episodes'])}/10个rollout；实际seed为{','.join(map(str,row['actual_seeds'])) or '尚无'}。相同seed记录中，我们失败而官方成功{pairs['ours_failure_official_success']}次；双方失败{pairs['both_failure']}次；我们成功而官方失败{pairs['ours_success_official_failure']}次；双方成功{pairs['both_success']}次。")
    anomalies = [f"- `{row['task']}`：{message}" for row in rows for message in row['anomalies']]
    anomalies += [f"- 采集：{message}" for message in snapshot['warnings']]
    error_count = sum(sum(row['errors'].values()) for row in rows)
    source = reference['sources']
    markdown = f'''# 官方模型 BF16：低分五任务对照

快照：{snapshot['snapshot_hong_kong']}（香港时间）。运行：`{RUN_NAME}`。进度：**{len(finished)}/5任务完成**。我们的五任务基线为 **3/50（6%）**。

{aggregate}

## 任务结果

| 任务 | 官方本地状态 | 我们6k BF16 | 官方本地BF16 | 最终差值（百分点） | 官方发布Clean FP32 |
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

官方本地分母为实际已完成policy rollout；进行中的分数仍可变化。汇总只计已完整完成的任务。官方发布分数来自[模型卡]({source['scores']})，官方说明使用[FP32验证]({source['precision']})和[每任务100次评测]({source['episode_count_and_pinned_revision']})；本地为BF16、每任务10次，不能直接视为论文条件的等价复现。

## 对照条件与限制

- 官方权重：`robbyant/lingbot-vla-v2-6b-robotwin`的`global_step_50000/hf_ckpt`；6k基线来自基础6b模型Full-SFT。
- 本地双方按同一设置比较：BF16、demo_clean、seen指令、seed参数0、10次有效policy rollout、expert预检、允许expert失败时使用已生成信息、valid-episode retry patch、录像、use_length=25、eager、不编译。实际启动配置保留在方案与原始日志中。
- 相同场景seed不保证指令抽样、推理噪声或仿真/规划器轨迹完全相同。逐seed统计仅描述日志对应记录，不作严格配对试验的绝对等同断言。
- 五任务因我们的低分而被选中，不能用其平均值估计全部任务成绩。每任务只有10次，一次成功改变10个百分点；未做显著性检验或因果归因。
- 本地RoboTwin版本与官方部署固定版本不同。官方高分支持优先检查训练配方；官方低分提示继续检查FP32与评测环境，单次对照不能证明唯一原因。
- `episode_info`的plan_success/task_success来自expert预检，不能作为模型成功率。

## 实际seed与描述性对应结果

{chr(10).join(episode_lines)}

## 日志检查与保存

策略异常关键词计数：{error_count}。expert预检失败不计作策略异常。

{chr(10).join(anomalies) if anomalies else '当前未发现分数解析/seed一致性异常。'}

原始日志、done、events、task_times、episode_info、结果文本与启动清单保存在`硬盘/结果/官方对照/原始日志`；机器可读快照为`原始快照.json.gz`。录像仅记录本轮日志中的准确最终目录到`录像目录清单.json`，尚未由本脚本下载。完整结果的最终小文件归档与`采集完成.json`只在五任务最终分数、逐rollout记录和done全部一致后写入。

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-04
- Verification Status: {'VERIFIED (execution/log consistency only)' if complete else 'UNVERIFIED'}
- Version Label: official-weak5-bf16-10x-v1

## ExperimentResult

- ID: {RUN_NAME}
- Type: matched-local-checkpoint-diagnostic
- Status: {'COMPLETED' if complete else 'NOT_STARTED' if not snapshot['run_exists'] else 'RUNNING_OR_PARTIAL'}
- Command: 按`评测方案.md`及原始`preparation/run.sh`执行；本脚本仅只读采集。
- Working Directory: /RoboTwin
- Exit Code: 各task_times记录可查；本报告不代替启动器总退出码。
- Output: 本报告、对照结果.json、per_task_comparison.csv、原始日志和录像目录清单。
- Anomalies: 见日志检查。
'''
    atomic_text(HERE / '对照报告.md', markdown)
    return report


def finalize(report):
    inventory = json.loads((RESULTS / '录像目录清单.json').read_text())
    video_ready = len(inventory['tasks']) == 5 and all(
        row['directory'] and row['mp4_count'] == 10 and len(row['files']) == 10
        and all(re.fullmatch(r'[0-9a-f]{64}', f.get('sha256') or '') for f in row['files'])
        for row in inventory['tasks'])
    if not video_ready:
        raise ValueError('Scores complete; awaiting 50 closed videos with exact paths and SHA256')
    temporary = RESULTS / '最终原始小文件快照.tar.gz.partial'
    archive = RESULTS / '最终原始小文件快照.tar.gz'
    with tarfile.open(temporary, 'w:gz') as tar:
        for path in [RESULTS / '原始日志', RESULTS / '原始快照.json.gz',
                     RESULTS / '录像目录清单.json', RESULTS / '运行进程.txt']:
            if path.exists():
                tar.add(path, arcname=path.name, recursive=True)
    temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    save_json(RESULTS / '采集完成.json', {
        'complete': True, 'run_name': RUN_NAME, 'snapshot_utc': report['snapshot_utc'],
        'completed_tasks': 5, 'official_successes': report['completed_task_official_successes'],
        'policy_rollouts': report['completed_task_official_rollouts'],
        'archive': str(archive), 'archive_sha256': digest,
        'exact_video_directories': [row['directory'] for row in inventory['tasks']],
        'video_inventory_path': str(RESULTS / '录像目录清单.json'),
        'video_inventory_ready': video_ready,
        'video_count': sum(row['mp4_count'] for row in inventory['tasks']),
        'videos_downloaded_by_this_script': False})


def log_error(exc):
    now = dt.datetime.now(HK).isoformat()
    with (RESULTS / '采集错误.log').open('a', encoding='utf-8') as out:
        out.write(now + ' ' + type(exc).__name__ + ': ' + str(exc) + '\n')
    save_json(RESULTS / '监控状态.json', {'status': 'retrying_after_read_error',
               'time_hong_kong': now, 'error': str(exc), 'run_name': RUN_NAME})
    print(json.dumps({'read_error': str(exc), 'time_hong_kong': now}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--interval', type=int, default=150)
    parser.add_argument('--deadline', default=DEFAULT_DEADLINE)
    args = parser.parse_args()
    if args.offline and args.watch:
        parser.error('--offline cannot be combined with --watch')
    if not 120 <= args.interval <= 180:
        parser.error('--interval must be 120..180 seconds')
    deadline = dt.datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error('--deadline must include its timezone')
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / '.collector.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another official-comparison collector is already active.', flush=True)
            return 2
        while True:
            if args.watch and dt.datetime.now(dt.timezone.utc) >= deadline:
                save_json(RESULTS / '监控状态.json', {'status': 'deadline_reached_incomplete',
                    'time_hong_kong': dt.datetime.now(HK).isoformat(), 'deadline': args.deadline,
                    'run_name': RUN_NAME, 'note': 'Collector stopped; no remote process was stopped.'})
                return 0
            try:
                snapshot = (json.loads(gzip.decompress((RESULTS / '原始快照.json.gz').read_bytes()))
                            if args.offline else capture())
                report = build_report(snapshot)
                if report['completed_tasks'] == 5:
                    finalize(report)
                save_json(RESULTS / '监控状态.json', {'status': report['status'],
                    'snapshot_hong_kong': report['snapshot_hong_kong'],
                    'completed_tasks': report['completed_tasks'], 'run_name': RUN_NAME,
                    'deadline': args.deadline, 'watch': args.watch})
                print(json.dumps({'status': report['status'], 'completed_tasks': report['completed_tasks'],
                    'official_successes': report['completed_task_official_successes'],
                    'official_rollouts': report['completed_task_official_rollouts'],
                    'snapshot_hong_kong': report['snapshot_hong_kong']}, ensure_ascii=False), flush=True)
                if report['completed_tasks'] == 5 or not args.watch:
                    return 0
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
                log_error(exc)
                if not args.watch:
                    return 1
            remaining = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
            # This finite background collector sleeps; it never blocks the interactive agent.
            if remaining > 0:
                time.sleep(min(args.interval, remaining))


if __name__ == '__main__':
    raise SystemExit(main())
