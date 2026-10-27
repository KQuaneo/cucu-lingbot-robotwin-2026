from pathlib import Path
import subprocess,hashlib,datetime,json,time,io,tarfile,csv,os,concurrent.futures,shlex,fcntl
base=Path(__file__).resolve().parent
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-p','35051','root@YOUR_SERVER_IP']
remote='/workspace/runtime/experiments/randomized_fp32_20261026'
deadline=datetime.datetime(2026,10,26,23,25,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
lock=(base/'采集.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
    return h.hexdigest()
def log(s):print(datetime.datetime.now().isoformat(),s,flush=True)
def save_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def download(row):
    dst=base/row['local_relative'];dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() and dst.stat().st_size==row['bytes'] and sha(dst)==row['sha256']:return True
    temp=dst.with_suffix('.mp4.downloading')
    with temp.open('wb') as f:
        p=subprocess.run(ssh+['cat -- '+shlex.quote(row['remote_path'])],stdout=f,stderr=subprocess.PIPE,timeout=90)
    if p.returncode or temp.stat().st_size!=row['bytes'] or sha(temp)!=row['sha256']:
        log('VIDEO_PENDING '+row['remote_path']);return False
    temp.replace(dst);return True
failures=0
while time.time()<deadline:
    try:
        p=subprocess.run(ssh+['/opt/robotwin-env/bin/python '+remote+'/snapshot_remote.py'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=100)
        if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-1500:])
        entries={}
        with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:gz') as tar:
            for m in tar:
                assert m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts,m.name
                entries[m.name]=tar.extractfile(m).read()
        snap=json.loads(entries['snapshot.json']);videos=json.loads(entries['video_manifest.json'])
        for name,content in entries.items():
            dst=base/name;dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix(dst.suffix+'.tmp');tmp.write_bytes(content);tmp.replace(dst)
        (base/'最新原始快照.tar.gz').write_bytes(p.stdout)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:verified=sum(pool.map(download,videos))
        failures=0
        status={'captured_utc':snap['captured_utc'],'run_name':snap['run_name'],'completed_tasks':snap['completed_tasks'],'all_logged_rollouts':snap['all_logged_rollouts'],'completed_task_successes':snap['completed_task_successes'],'completed_task_rollouts':snap['completed_task_rollouts'],'videos_verified':verified,'finalized_videos_expected':len(videos),'benchmark_complete':snap['complete'],'backup_complete':snap['complete'] and verified==500,'exit_code':snap['exit_code'],'issues':snap['issues']}
        save_json(base/'采集状态.json',status)
        lines=['# 6k模型随机场景FP32评测','',f"更新：{snap['captured_utc']}。任务完成 {snap['completed_tasks']}/50；已记录有效回合 {snap['all_logged_rollouts']}/500。",f"已完整完成任务：{snap['completed_task_successes']}/{snap['completed_task_rollouts']} 成功；其余任务未完成，不能按失败计入。",f"Mac已核验录像 {verified} 段。状态：{'完成' if status['backup_complete'] else '进行中或部分结果'}。",'', '50个demo_randomized任务，每任务10次自测，FP32、unseen指令、MPLib、valid retry、use_length=25、不编译。测试次数不是官方100次。','','|任务|状态|成功/实际回合|','|---|---|---:|']
        lines += ['|%s|%s|%s/%s|'%(r['task'],r['status'],r['successes'],r['policy_rollouts']) for r in snap['tasks']]
        lines += ['','## Material Passport','- Origin Skill: experiment-agent','- Origin Mode: run','- Origin Date: 2026-10-26','- Verification Status: '+('VERIFIED (log and transfer consistency)' if status['backup_complete'] else 'IN_PROGRESS'),'','## ExperimentResult','- Status: '+('COMPLETED' if status['backup_complete'] else 'RUNNING_OR_PARTIAL'),'- ID: '+snap['run_name'],'- Exit Code: '+str(snap['exit_code']),'- Output: snapshot.json, raw/, 录像/, video_manifest.json, 采集状态.json']
        (base/'评测报告.md').write_text('\n'.join(lines)+'\n')
        with (base/'逐任务结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,['task','status','successes','policy_rollouts','actual_seeds']);w.writeheader()
            for r in snap['tasks']:w.writerow({k:r[k] for k in ['task','status','successes','policy_rollouts']}|{'actual_seeds':';'.join(str(e['seed']) for e in r['episodes'])})
        log(json.dumps(status,ensure_ascii=False))
        if status['backup_complete']:
            save_json(base/'采集完成.json',status);break
        if snap['exit_code'] is not None and snap['exit_code']!=0:
            save_json(base/'评测异常.json',status);log('BENCHMARK_FAILED; retained actual completed results, no automatic rerun');break
    except Exception as exc:
        failures+=1;log('SNAPSHOT_ERROR '+repr(exc))
        save_json(base/'采集连接状态.json',{'time':datetime.datetime.now().isoformat(),'consecutive_errors':failures,'error':repr(exc),'benchmark_status':'last saved snapshot remains authoritative; no completion claimed'})
    time.sleep(180)
else:log('COLLECTOR_DEADLINE; use latest saved snapshot, not a complete-result claim')
