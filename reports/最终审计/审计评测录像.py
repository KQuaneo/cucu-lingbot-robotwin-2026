#!/usr/bin/env python3
"""Read-only local audit of the two complete evaluations and their video backups."""
import csv
import hashlib
import json
import re
import tarfile
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
START = time.monotonic()
ISSUES = []

def load(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def require(ok, message):
    if not ok:
        ISSUES.append(message)
    return bool(ok)

def file_record(p):
    if not require(p.is_file(), f'Missing file: {p}'):
        return {'path': str(p), 'exists': False}
    return {'path': str(p), 'exists': True, 'bytes': p.stat().st_size, 'sha256': sha(p)}

def clean(s):
    return re.sub(r'\x1b\[[0-9;]*m', '', s.replace('\r', '\n'))

FINAL = re.compile(r'Final success rate:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%')
ROLL = re.compile(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)')

def audit_run(label, run, report_path, raw, manifests, video_root, expected_successes):
    report = load(report_path)
    tasks = report['tasks']
    task_names = {t['task'] for t in tasks}
    require(len(tasks) == len(task_names) == 50, f'{label}: expected 50 unique tasks')
    require(sum(t['successes'] for t in tasks) == expected_successes, f'{label}: success sum mismatch')
    require(sum(t['policy_rollouts'] for t in tasks) == 500, f'{label}: episode sum mismatch')
    require(report['completed_tasks'] == 50 and report['completed_task_successes'] == expected_successes and report['completed_task_rollouts'] == 500, f'{label}: top-level counts mismatch')
    done_paths = set(raw.joinpath('done').glob('*.done'))
    require(len(done_paths) == 50, f'{label}: done count {len(done_paths)}')
    eval_paths = set(raw.joinpath('logs').glob('eval_demo_clean__*.log'))
    server_paths = set(raw.joinpath('logs').glob('server_gpu*.log'))
    require(len(eval_paths) == 50, f'{label}: evaluation log count {len(eval_paths)}')
    require(len(server_paths) == 8, f'{label}: server log count {len(server_paths)}')
    times = [line.split('\t') for line in (raw/'task_times.tsv').read_text().splitlines() if line.strip()]
    require(len(times) == 50 and {t[1] for t in times} == task_names, f'{label}: task timing count/task mismatch')
    require(all(t[-1] == '0' for t in times), f'{label}: nonzero task exit code')
    audits = []
    for t in tasks:
        name = t['task']
        p = raw/'logs'/f'eval_demo_clean__{name}.log'
        require(p.is_file(), f'{label}: missing log {name}')
        require((raw/'done'/f'demo_clean__{name}.done').is_file(), f'{label}: missing done {name}')
        require(t['completed'] and t['status'] == 'completed' and t['done_marker'], f'{label}: incomplete structured status {name}')
        text = clean(p.read_text(errors='replace'))
        finals = FINAL.findall(text)
        require(len(finals) == 1, f'{label}: final log result count {name}: {len(finals)}')
        if finals:
            su, ep, pct = finals[-1]
            require(int(su) == t['successes'] and int(ep) == t['policy_rollouts'] == 10 and abs(float(pct)-int(su)*10) < 0.001, f'{label}: final score mismatch {name}')
        rolls = ROLL.findall(text)
        require(len(rolls) == 10, f'{label}: rollout log count {name}: {len(rolls)}')
        previous = 0
        episodes = []
        for su, ep, seed in rolls:
            su, ep, seed = int(su), int(ep), int(seed)
            episodes.append({'index': ep-1, 'seed': seed, 'success': su > previous})
            previous = su
        require([(x['index'],x['seed'],x['success']) for x in episodes] == [(x['index'],x['seed'],x['success']) for x in t['episodes']], f'{label}: episode metadata mismatch {name}')
        require(sum(x['success'] for x in episodes) == t['successes'], f'{label}: episode success sum mismatch {name}')
        for error in ['Policy rollout error','OutOfMemoryError','CUDA out of memory','No space left on device']:
            require(error not in text, f'{label}: {error} in {name}')
        audits.append({'task': name, 'successes': t['successes'], 'episodes':10, 'actual_seeds':[x['seed'] for x in episodes], 'log':file_record(p), 'done_marker':str(raw/'done'/f'demo_clean__{name}.done')})

    entries=[]
    archive_audits=[]
    for mp in manifests:
        manifest=load(mp)
        require(manifest['run'] == run, f'{label}: wrong run in manifest {mp}')
        require(sum(f['kind']=='video' for f in manifest['files']) == manifest['video_count'], f'{label}: manifest video count mismatch {mp}')
        entries.extend(manifest['files'])
        stem=mp.name.removesuffix('_manifest.json')
        completion=load(ROOT/'传输'/f'{stem}_complete.json')
        tar=Path(completion['archive_path'])
        record=file_record(tar)
        record['expected_bytes']=completion['archive_bytes']
        record['expected_sha256']=completion['archive_sha256']
        record['sha256_matches']=record.get('sha256') == completion['archive_sha256']
        require(completion['complete'] and record.get('bytes') == completion['archive_bytes'] and record['sha256_matches'], f'{label}: archive integrity mismatch {tar}')
        # Inspect archive headers, without extracting or hashing each member twice.
        with tarfile.open(tar) as tf:
            members={m.name:m.size for m in tf.getmembers() if m.isfile()}
            embedded=json.load(tf.extractfile('video_manifest.json'))
        require(embedded==manifest, f'{label}: embedded video manifest differs {tar}')
        content_members={k:v for k,v in members.items() if k!='video_manifest.json'}
        require(content_members == {f['archive_path']:f['bytes'] for f in manifest['files']}, f'{label}: tar member list/size mismatch {tar}')
        record.update({'manifest':str(mp),'file_members':len(members),'video_members':manifest['video_count'],'header_only_inspection':True})
        archive_audits.append(record)
    require(len({f['archive_path'] for f in entries})==len(entries)==550, f'{label}: duplicate/missing manifest entries')
    require({f['task'] for f in entries}==task_names, f'{label}: video tasks mismatch')
    verified=[]
    original_dirs={}
    for f in entries:
        p=video_root/f['archive_path']
        record=file_record(p)
        ok=record.get('bytes')==f['bytes'] and record.get('sha256')==f['sha256']
        require(ok, f'{label}: saved file mismatch {p}')
        expected_original=f"/workspace/runtime/eval_result/{f['task']}/LingBot-VLA-v2/demo_clean/default/{f['timestamp']}/{f['file']}"
        require(f['original_path']==expected_original, f'{label}: original path mapping mismatch {p}')
        original_dirs[f['task']]=str(Path(f['original_path']).parent)
        if f['kind']=='video':
            with p.open('rb') as h:
                require(b'ftyp' in h.read(64), f'{label}: invalid MP4 header {p}')
        else:
            task=next(t for t in tasks if t['task']==f['task'])
            rate=float(p.read_text().splitlines()[-1])
            require(abs(rate-task['successes']/10)<1e-8, f'{label}: result text score mismatch {p}')
            original=Path(f['original_path'])
            if label=='BF16':
                raw_result=ROOT/'结果'/'eval_result'/Path(*original.parts[4:])
            else:
                raw_result=raw/'eval_result'/Path(*original.parts[4:])
            # /workspace/runtime/eval_result prefix has four path components including root.
            require(raw_result.is_file(), f'{label}: missing saved raw result text {raw_result}')
            if raw_result.is_file():
                require(sha(raw_result)==f['sha256'], f'{label}: raw/ video result text mismatch {p}')
        record.update({'task':f['task'],'kind':f['kind'],'original_path':f['original_path'],'matches_manifest':ok})
        verified.append(record)
    counts=Counter(f['task'] for f in entries if f['kind']=='video')
    require(all(n==10 for n in counts.values()) and len(counts)==50, f'{label}: each task must have 10 videos')
    for task in task_names:
        require({f['file'] for f in entries if f['task']==task and f['kind']=='video'} == {f'episode{i}.mp4' for i in range(10)}, f'{label}: video episode filenames incomplete {task}')
    finder_files=[str(p) for p in video_root.rglob('.DS_Store')]
    actual=set(str(p.relative_to(video_root)) for p in video_root.rglob('*') if p.is_file() and p.name!='.DS_Store')
    expected={f['archive_path'] for f in entries}
    require(actual==expected, f'{label}: unexpected/missing extracted files: {actual.symmetric_difference(expected)}')
    # Exact original result directory also must agree with this run's log save path.
    for task,path in original_dirs.items():
        text=clean((raw/'logs'/f'eval_demo_clean__{task}.log').read_text(errors='replace'))
        rel=path.removeprefix('/workspace/runtime/')+'/_result.txt'
        require(rel in text, f'{label}: video timestamp differs from log saved path {task}')
    server_audits=[]
    for p in sorted(server_paths):
        text=p.read_text(errors='replace')
        lines=[x for x in text.splitlines() if 'FP32_DTYPE_AUDIT' in x]
        if label=='FP32':
            require(len(lines)==1 and "parameters={'torch.float32': 1672}" in lines[0] and "buffers={'torch.float32': 147}" in lines[0], f'FP32: dtype audit missing/mismatch {p}')
        server_audits.append({'path':str(p),'sha256':sha(p),'dtype_audit':lines,'traceback_count':text.count('Traceback (most recent call last)')})
    extra=[]
    for p in [raw/'events.log',raw/'task_times.tsv',raw/'run_start_iso']:
        extra.append(file_record(p))
    epinfo=sorted((raw/'episode_info').glob('*.jsonl'))
    require(len(epinfo)==50, f'{label}: episode info file count {len(epinfo)}')
    return {'precision':label,'run':run,'report':file_record(report_path),'completed_tasks':len(tasks),'successes':expected_successes,'episodes':500,'success_rate':expected_successes/500,'evaluation_logs':len(eval_paths),'done_markers':len(done_paths),'task_exit_codes_all_zero':all(t[-1]=='0' for t in times),'expert_info_files':len(epinfo),'expert_info_note':'Expert precheck JSONL entries are metadata, not policy episode counts; setup retries can change valid seeds.','server_logs':server_audits,'task_audits':audits,'video_count':sum(f['kind']=='video' for f in entries),'result_text_count':sum(f['kind']=='result' for f in entries),'extracted_root':str(video_root),'verified_file_bytes':sum(f['bytes'] for f in entries),'verified_files':verified,'archives':archive_audits,'auxiliary_files':extra,'ignored_finder_metadata_files':finder_files}

def main():
    bf=audit_run('BF16','/workspace/runtime/outputs/full_sft_6000_clean50x10_8gpu_video',ROOT/'报告/实验结果.json',ROOT/'结果/full_sft_6000_clean50x10_8gpu_video',sorted((ROOT/'录像').glob('video_snapshot_*_manifest.json')),ROOT/'录像/全任务测评',276)
    fp=audit_run('FP32','/workspace/runtime/outputs/full_sft_6000_clean50x10_fp32_8gpu_video_20261004',ROOT/'报告/FP32全任务/结果.json',ROOT/'结果/FP32全任务/原始日志',sorted((ROOT/'录像/FP32全任务').glob('video_snapshot_fp32_*_manifest.json')),ROOT/'录像/FP32全任务/任务录像',263)
    require({x['task'] for x in bf['task_audits']}=={x['task'] for x in fp['task_audits']}, 'BF16 and FP32 task sets differ')
    training_log=ROOT/'日志/full_sft_8gpu_6000steps_20261002.log'
    train_text=clean(training_log.read_text(errors='replace'))
    steps=[int(x) for x in re.findall(r'INFO - __main__ - Step (\d+)/\d+, Epoch',train_text)]
    require(steps==list(range(1,6001)), 'Training log does not contain exactly ordered steps 1..6000')
    csv_path=ROOT/'日志/training_metrics_6000steps.csv'
    with csv_path.open(encoding='utf-8-sig',newline='') as h:
        rows=list(csv.DictReader(h))
    require([int(x['step']) for x in rows]==list(range(1,6001)), 'Metrics CSV steps incomplete/duplicated')
    require(float(rows[-1]['Loss'])==0.0283 and float(rows[-1]['VLA_Loss'])==0.0243, 'Training CSV final losses mismatch')
    require('Reached max_steps=6000, stopping training.' in train_text and 'global_step_6000 successfully!' in train_text, 'Training completion/save evidence missing')
    training_files=[file_record(p) for p in [training_log,csv_path,ROOT/'日志/loss_curve_6000steps.png',ROOT/'日志/训练日志与曲线.tar.gz']]
    require((ROOT/'日志/loss_curve_6000steps.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n'), 'Training PNG signature invalid')
    cancellation=ROOT/'结果/官方对照/已取消.json'
    official=load(ROOT/'报告/官方对照/对照结果.json')
    require(cancellation.is_file() and official['status']=='cancelled_by_user','Official partial evaluation cancellation evidence missing')
    cancelled=load(cancellation)
    otherfiles=[file_record(p) for p in [ROOT/'报告/FP32全任务/启动验证.json',ROOT/'报告/FP32全任务/manifest.json',ROOT/'结果/FP32全任务/已结束.json',cancellation,ROOT/'报告/官方对照/对照结果.json',ROOT/'传输/最终评测备份完成.json',ROOT/'传输/FP32录像备份完成.json']]
    raw_archives=[]
    for p in [ROOT/'结果/全任务评测快照.tar.gz',ROOT/'结果/FP32全任务/最终原始快照.tar.gz']:
        record=file_record(p)
        if 'FP32全任务' in str(p):
            expected=load(ROOT/'结果/FP32全任务/已结束.json')['archive_sha256']
            require(record['sha256']==expected,'FP32 raw archive checksum mismatch')
            record['expected_sha256']=expected
        with tarfile.open(p) as tf:
            record['file_members']=sum(m.isfile() for m in tf.getmembers())
        raw_archives.append(record)
    bfseeds={x['task']:x['actual_seeds'] for x in bf['task_audits']}
    seed_pairs=[{'task':x['task'],'identical_seed_sequence':x['actual_seeds']==bfseeds[x['task']],'common_seed_count':len(set(x['actual_seeds']) & set(bfseeds[x['task']]))} for x in fp['task_audits']]
    output={'audit_utc':datetime.now(timezone.utc).isoformat(),'audit_hong_kong':datetime.now(timezone(timedelta(hours=8))).isoformat(),'scope':'Read-only local audit; only complete BF16 and FP32 50-task evaluations counted. Official cancelled partial run excluded. No SSH or external-drive writes.','passed':not ISSUES,'issues':ISSUES,'duration_seconds':round(time.monotonic()-START,3),'evaluations':[bf,fp],'total_complete_video_count':bf['video_count']+fp['video_count'],'total_result_text_count':bf['result_text_count']+fp['result_text_count'],'training':{'ordered_log_steps':len(steps),'csv_rows':len(rows),'last_step':rows[-1],'files':training_files},'raw_evaluation_archives':raw_archives,'official_cancelled_partial':{'status':official['status'],'cancellation':cancelled,'included_in_1000_video_count':False},'seed_pairing':seed_pairs,'additional_evidence':otherfiles,'coverage_limits':['This audit verifies saved evaluation outputs and full 6000-step training log/metrics/curve. Model and DCP integrity are audited separately.','This audit does not assert full original training datasets, simulator assets, Docker/container environment, or all historical LoRA/teammate checkpoints are backed up; a separate remote inventory is required.','File SHA256 checks establish that saved videos equal the transfer manifests; full video decoding was not repeated.','The evaluation uses valid-episode retry: two runs can admit different actual seeds when setup fails. Exact seed comparisons are stored per task.']}
    (OUT/'评测录像审计.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    lines=['# 两轮评测、录像与训练日志最终审计','',f"审计时间：{output['audit_hong_kong']}。耗时 {output['duration_seconds']} 秒。",'',f"结论：{'通过' if output['passed'] else '存在问题'}。本次重新读取原始日志，逐文件重新计算 SHA256。",'','| 评测 | 任务 | 有效回合 | 成功数 | 成功率 | 录像 | 结果文本 |','|---|---:|---:|---:|---:|---:|---:|']
    for ev in [bf,fp]:
        lines.append(f"| {ev['precision']} | 50 | 500 | {ev['successes']} | {ev['success_rate']:.1%} | {ev['video_count']} | {ev['result_text_count']} |")
    lines+=['','## 核对内容','','- 两轮共 100 份评测日志、100 个 done 标记和 100 个任务退出码均已逐任务核对；日志最终结果、逐回合成功/种子记录、结构化报告、录像中的结果文本一致。','- 每轮 50 个任务各有 episode0.mp4 至 episode9.mp4，共 1000 段完整任务录像；官方取消的部分对照没有混入。','- 1000 段录像和 100 份结果文本已重新核对大小、SHA256、原服务器路径及日志中的实际运行时间目录。','- 7 个录像 tar 归档已重新计算 SHA256、核对大小，并读取成员清单与大小；没有重复解压。','- FP32 的 8 份服务日志均包含实际参数与缓冲区为 torch.float32 的审计记录；2 处历史握手 Traceback 已保留在原日志与启动验证说明中，未被当作策略回合失败。','- 训练日志与 CSV 均有按顺序的 1–6000 步；终值 Loss=0.0283、VLA_Loss=0.0243；正常停止与 6000 步存档成功证据存在。loss PNG 和训练日志归档存在。','- 官方 BF16 部分对照的用户取消标记、原始日志和部分结构化结果存在。','','## 本地路径','']
    for ev in [bf,fp]:
        lines.append(f"- {ev['precision']} 录像：`{ev['extracted_root']}`")
        lines.append(f"- {ev['precision']} 报告：`{ev['report']['path']}`")
    lines+=['',f'- 训练日志与曲线：`{ROOT / "日志"}`',f'- 逐文件审计清单：`{OUT / "评测录像审计.json"}`','','## 范围与限制','','模型与完整续训 DCP 由另一份审计核对。本审计不等于完整原始训练数据、仿真资源、整个容器环境或所有历史实验模型均已下载；这些须以远端资产清单确认。录像进行了传输清单一致性校验与 MP4 文件头检查，没有重新解码全部视频。',f"本次两轮评测核对为 {sum(x['identical_seed_sequence'] for x in seed_pairs)}/50 个任务种子序列完全一致，共 {sum(x['common_seed_count'] for x in seed_pairs)}/500 个回合同任务同种子；指令选择和模型采样等其他随机性不由本审计保证。"]
    if ISSUES:
        lines+=['','## 问题','']+[f'- {x}' for x in ISSUES]
    (OUT/'评测录像审计.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'passed':output['passed'],'issues':ISSUES,'videos':output['total_complete_video_count'],'result_texts':output['total_result_text_count'],'train_steps':len(steps),'duration_seconds':output['duration_seconds'],'report':str(OUT/'评测录像审计.md')},ensure_ascii=False))

if __name__=='__main__':
    main()
