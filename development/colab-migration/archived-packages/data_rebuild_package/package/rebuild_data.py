import concurrent.futures, json, os, shutil, subprocess
from pathlib import Path
local=Path('/content/lingbot-stage2');package=local/'package';project=package/'rebuild/RoboTwin'
model_python='/content/lingbot-venv/bin/python'
conversion=Path('/content/lerobot-convert-env/bin/python')
if not conversion.is_file():subprocess.run(['python','-m','uv','venv','--python','3.12',str(conversion.parent.parent)],check=True)
subprocess.run(['python','-m','uv','pip','install','--python',str(conversion),'lerobot[dataset]==0.6.0','h5py==3.14.0','torch==2.8.0','torchvision==0.23.0','av==15.0.0'],check=True)
assert shutil.disk_usage('/content').free > 100*2**30,'Insufficient space for dataset reconstruction'
manifest=json.loads((package/'data_manifest.json').read_text())
tasks=sorted({Path(x['path']).parts[0].removesuffix('_joint_v30') for x in manifest['files']});assert len(tasks)==50
logs=local/'data_rebuild_logs';logs.mkdir(exist_ok=True)
env=os.environ.copy();env.update(PATH=str(conversion.parent)+':'+env['PATH'],HF_REVISION='a967b852afa21a9cbf19a198f7e653109042e87c',ROBOTWIN_DATA_ROOT=str(project/'data'),HF_ARCHIVE_CACHE=str(local/'raw_archive_cache'),HF_KEEP_ARCHIVES='0',HF_MAX_WORKERS='4',HF_EXTRACT_WORKERS='2',HF_LEROBOT_HOME=str(local/'lerobot'),LEROBOT_VIDEO_BACKEND='pyav',LEROBOT_VIDEO_CODEC='h264')
subprocess.run(['bash',str(project/'scripts/download_xpolicylab_data.sh'),*tasks],cwd=project,env=env,check=True)
def convert(task):
    out=local/'lerobot'/(task+'_joint_v30');info=out/'meta/info.json'
    if info.is_file() and ((out/'.conversion_complete').is_file() or task=='adjust_bottle'):
        value=json.loads(info.read_text())
        if value['total_episodes']==50 and value['fps']==15:
            return task,value
    log=logs/(task+'.log')
    with log.open('w') as f:
        result=subprocess.run([str(conversion),str(project/'XPolicyLab/scripts/transform_lerobot_v30_format.py'),'demo_clean.'+task+'.aloha_agilex','--repo_id',task+'_joint_v30','--max_episode','50'],cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT)
    assert result.returncode==0 and 'Failed files:' not in log.read_text(),f'Conversion failed: {log}'
    value=json.loads(info.read_text());assert value['total_episodes']==50 and value['fps']==15
    (out/'.conversion_complete').write_text('verified converter completion\n')
    return task,value
counts={}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for future in concurrent.futures.as_completed([pool.submit(convert,t) for t in tasks]):
        task,info=future.result();counts[task]={'episodes':info['total_episodes'],'frames':info['total_frames']};print('CONVERTED',task,counts[task],flush=True)
assert sum(v['episodes'] for v in counts.values())==2500
assert sum(v['frames'] for v in counts.values())==549787
record={'source_repo':'TianxingChen/RoboTwin2.0','revision':env['HF_REVISION'],'converter':'XPolicyLab c37109c500be67d0dea6b36bf7337bbd26e763cd + archived patch','lerobot':'0.6.0','counts':counts,'source':'public reconstruction + SHA-verified recovered adjust_bottle'}
(local/'data_rebuild_audit.json').write_text(json.dumps(record,indent=2))
drive=Path('/content/drive/MyDrive/LingBot-Colab-Stage2-20261008')
if drive.is_dir():
    shutil.copytree(local/'lerobot',drive/'lerobot',dirs_exist_ok=True)
    shutil.copytree(logs,drive/'data_rebuild_logs',dirs_exist_ok=True)
    shutil.copy2(local/'data_rebuild_audit.json',drive/'data_rebuild_audit.json')
print('FULL_50_TASK_DATA_PASS',flush=True)
