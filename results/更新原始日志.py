"""只读更新本轮评测原始日志、结果文本和录像目录索引，不下载录像或修改评测。"""
import gzip, json, subprocess, tarfile
from pathlib import Path

REMOTE = r'''
import datetime, io, json, re, sys, tarfile
from pathlib import Path
run=Path('/workspace/runtime/outputs/full_sft_6000_clean50x10_8gpu_video')
starts={}
for line in (run/'events.log').read_text().splitlines():
 m=re.match(r'(\S+) gpu=(\d+) config=(\S+) task=(\S+) start$',line)
 if m:starts[m[4]]=datetime.datetime.fromisoformat(m[1])
rows=[];results=[]
for task,start in starts.items():
 p=run/'logs'/('eval_demo_clean__'+task+'.log')
 s=re.sub(r'\x1b\[[0-9;]*m','',p.read_text(errors='replace'))
 result=re.findall(r'Data has been saved to (.+/_result.txt)',s)
 root=Path('/workspace/runtime/eval_result')/task/'LingBot-VLA-v2/demo_clean/default'
 candidates=[]
 if result:
  f=Path('/workspace/runtime')/result[-1]
  if f.is_file():results.append(f)
  if f.parent.is_dir():candidates=[f.parent]
 elif root.exists():
  for d in root.iterdir():
   if not d.is_dir():continue
   try:stamp=datetime.datetime.strptime(d.name,'%Y-%m-%d_%H-%M-%S_%f').replace(tzinfo=datetime.timezone.utc)
   except ValueError:continue
   if stamp>=start:candidates.append(d)
 final=re.findall(r'Final success rate:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%',s)
 rows.append({'task':task,'start_utc':start.isoformat(),'selection_method':'final_result_path' if result else 'timestamp_not_before_event_task_start','ambiguous':len(candidates)!=1,'directories':[str(d) for d in candidates],'mp4_count':sum(len(list(d.glob('*.mp4'))) for d in candidates),'mp4_bytes':sum(v.stat().st_size for d in candidates for v in d.glob('*.mp4')),'final_policy_result':list(final[-1]) if final else None,'errors':{k:s.count(k) for k in ['Policy rollout error','Traceback (most recent call last)','OutOfMemoryError','No space left on device','CUDA out of memory']}})
inventory={'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':str(run),'notes':'Completed task exact directories from final result log; unfinished task directory must be unique and timestamp >= current task start. In-progress MP4 may still be writing.','tasks':rows}
blob=json.dumps(inventory,ensure_ascii=False,indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for n in ['logs','episode_info','diagnostics','done','events.log','task_times.tsv','run_start_iso']:
  p=run/n
  if p.exists():tar.add(p,arcname='full_sft_6000_clean50x10_8gpu_video/'+n)
 for f in results:tar.add(f,arcname=str(f.relative_to('/workspace/runtime')))
 info=tarfile.TarInfo('录像目录清单.json');info.size=len(blob);tar.addfile(info,io.BytesIO(blob))
'''

if __name__ == '__main__':
    folder=Path(__file__).resolve().parent
    temporary=folder/'全任务评测快照.tar.gz.partial'
    archive=folder/'全任务评测快照.tar.gz'
    with temporary.open('wb') as out:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-p','31227','root@YOUR_SERVER_IP','/opt/robotwin-env/bin/python -'],input=REMOTE.encode(),stdout=out,check=True)
    with gzip.open(temporary,'rb') as f:
        while f.read(1048576):pass
    with tarfile.open(temporary) as t:
        for item in t.getmembers():
            if item.name.startswith('/') or '..' in Path(item.name).parts or item.issym() or item.islnk():
                raise ValueError('Unexpected archive member: '+item.name)
        t.extractall(folder)
    temporary.replace(archive)
    data=json.loads((folder/'录像目录清单.json').read_text())
    print(json.dumps({'snapshot_utc':data['snapshot_utc'],'final_tasks':sum(bool(t['final_policy_result']) for t in data['tasks']),'videos':sum(t['mp4_count'] for t in data['tasks']),'bytes':sum(t['mp4_bytes'] for t in data['tasks']),'errors':sum(sum(t['errors'].values()) for t in data['tasks']),'ambiguous_directories':sum(t['ambiguous'] for t in data['tasks'])},ensure_ascii=False))
