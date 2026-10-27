from pathlib import Path
import re,json,datetime,hashlib,tarfile,io,sys,os
import yaml
base=Path('/workspace/runtime/experiments/randomized_fp32_20261026')
run=Path('/workspace/runtime/outputs/full_sft_6000_randomized50x10_fp32_8gpu_video_20261026')
ansi=re.compile(r'\x1b\[[0-9;]*m')
tasks=yaml.safe_load(Path('/RoboTwin/env_cfg/eval/all_tasks.yml').read_text())['tasks']
start=datetime.datetime.fromisoformat((run/'run_start_iso').read_text().strip()).timestamp()
rows=[];videos=[];issues=[]
for task in tasks:
    p=run/'logs'/('eval_demo_randomized__'+task+'.log')
    text=ansi.sub('',p.read_text(errors='replace')).replace('\r','\n') if p.exists() else ''
    episodes=[];previous=0
    for m in re.finditer(r'Success rate:\s*(\d+)/(\d+)[^\n]*?current seed:\s*(\d+)',text):
        success,total,seed=map(int,m.groups())
        if total!=len(episodes)+1 or success-previous not in (0,1):
            issues.append(task+': non-contiguous policy outcomes');continue
        episodes.append({'index':total-1,'seed':seed,'success':success>previous});previous=success
    final=re.findall(r'Final success rate:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%',text)
    done=(run/'done'/('demo_randomized__'+task+'.done')).exists()
    complete=done and bool(final) and int(final[-1][1])==10 and len(episodes)==10 and int(final[-1][0])==previous
    row={'task':task,'completed':complete,'status':'completed' if complete else ('running' if p.exists() else 'pending'),'successes':previous,'policy_rollouts':len(episodes),'episodes':episodes,'done_marker':done,'final':final[-1] if final else None,'video_directory':None,'errors':{k:text.count(k) for k in ['Policy rollout error','Traceback (most recent call last)','OutOfMemoryError','No space left on device']}}
    paths=re.findall(r'Data has been saved to ([^\n]+/_result\.txt)',text)
    directories=[]
    if paths:
        result_path=Path(paths[-1])
        if not result_path.is_absolute():result_path=Path('/RoboTwin')/result_path
        directories=[result_path.parent.resolve()]
    else:
        for root in [Path('/RoboTwin/eval_result'),Path('/workspace/runtime/eval_result')]:
            parent=root/task/'LingBot-VLA-v2/demo_randomized/default'
            if parent.exists():
                for d in parent.iterdir():
                    if not d.is_dir():continue
                    try:stamp=datetime.datetime.strptime(d.name.split('_')[0]+'_'+d.name.split('_')[1],'%Y-%m-%d_%H-%M-%S').replace(tzinfo=datetime.timezone.utc).timestamp()
                    except ValueError:continue
                    if stamp>=start-30:directories.append(d.resolve())
        directories=list({str(d.resolve()):d for d in directories}.values())
    if len(directories)==1:
        d=directories[0];row['video_directory']=str(d)
        for ep in episodes:
            src=d/('episode%d.mp4'%ep['index'])
            if not src.exists():issues.append(task+': missing finalized video '+src.name);continue
            h=hashlib.sha256(src.read_bytes()).hexdigest()
            videos.append({'task':task,'episode':ep['index'],'seed':ep['seed'],'success':ep['success'],'remote_path':str(src),'local_relative':'录像/'+task+'/'+src.name,'bytes':src.stat().st_size,'sha256':h})
    elif episodes:issues.append(task+': ambiguous or missing video directory '+str(len(directories)))
    rows.append(row)
exitcode=int((base/'exit_code').read_text()) if (base/'exit_code').exists() else None
finished=[r for r in rows if r['completed']]
result={'run_name':run.name,'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'team':'cucu','checkpoint_step':6000,'precision':'FP32','task_config':'demo_randomized','instruction_type':'unseen','episodes_per_task':10,'completed_tasks':len(finished),'completed_task_successes':sum(r['successes'] for r in finished),'completed_task_rollouts':sum(r['policy_rollouts'] for r in finished),'all_logged_rollouts':sum(r['policy_rollouts'] for r in rows),'exit_code':exitcode,'complete':len(finished)==50 and exitcode==0 and not issues,'tasks':rows,'issues':issues,'conditions':{'use_length':25,'compile':False,'expert_check':True,'accept_expert_info_on_failure':True,'valid_retry_patch':True,'seed_argument':0,'planner':'mplib','gpu_count':8,'video':True},'note':'10 episodes/task self-test; never represent as 100 official trials.'}
entries={}
for p in run.rglob('*'):
    if p.is_file() and p.suffix in ['.log','.tsv','.jsonl','.done'] or (p.is_file() and p.name=='run_start_iso'):
        entries['raw/'+str(p.relative_to(run))]=p.read_bytes()
for name in ['launcher.log','launcher.pid','exit_code','gpu_monitor.jsonl','model_verified.json','expected_model_manifest.json','environment_current.json','actual_RoboTwin.patch','actual_XPolicyLab.patch','actual_lingbot-vla-v2.patch']:
    p=base/name
    if p.is_file():entries['preparation/'+name]=p.read_bytes()
for row in rows:
    if row['completed'] and row['video_directory']:
        p=Path(row['video_directory'])/'_result.txt'
        if p.exists():entries['raw/result_text/'+row['task']+'/_result.txt']=p.read_bytes()
entries['snapshot.json']=json.dumps(result,ensure_ascii=False,indent=2).encode()
entries['video_manifest.json']=json.dumps(videos,ensure_ascii=False,indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
    for name,content in entries.items():
        info=tarfile.TarInfo(name);info.size=len(content);tar.addfile(info,io.BytesIO(content))
