# coding: utf-8
from pathlib import Path
import json,subprocess,sys,os,shutil,hashlib,zipfile,datetime,csv
base=Path(__file__).resolve().parents[3];run=base/'报告/随机场景_20261026';out=base/'提交/cucu_初赛提交材料'
backup=json.loads((run/'采集状态.json').read_text());snapshot=json.loads((run/'snapshot.json').read_text())
assert backup['backup_complete'] and backup['videos_verified']==500 and snapshot['complete'] and snapshot['exit_code']==0,'Randomized evaluation and verified backup must finish first'
for name in ['build_official_results.py','build_reproduction_report.py']:
    subprocess.run([sys.executable,str(Path(__file__).parent/name)],check=True)
def link_or_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():return
    try:os.link(src,dst)
    except OSError:shutil.copy2(src,dst)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()
evidence=out/'04_复现与调优说明/评测证据';evidence.mkdir(exist_ok=True)
shutil.copytree(run/'raw',evidence/'randomized/原始日志',dirs_exist_ok=True)
for name in ['video_manifest.json','采集状态.json','snapshot.json','逐任务结果.csv','评测报告.md']:
    shutil.copy2(run/name,evidence/'randomized'/name)
for name in ['gpu_monitor.jsonl','model_verified.json','environment_current.json']:
    shutil.copy2(run/'preparation'/name,evidence/'randomized'/name)
videos=json.loads((run/'video_manifest.json').read_text());assert len(videos)==500
for r in videos:
    src=run/r['local_relative'];assert src.stat().st_size==r['bytes'] and sha(src)==r['sha256']
    link_or_copy(src,evidence/'randomized/录像'/r['task']/src.name)
clean=json.loads((base/'报告/FP32全任务/结果.json').read_text())
clean_dirs={r['task']:Path(r['directory']).name for r in json.loads((base/'结果/FP32全任务/录像目录清单.json').read_text())['tasks']}
shutil.copytree(base/'结果/FP32全任务/原始日志',evidence/'clean/原始日志',dirs_exist_ok=True)
index=[]
for t in clean['tasks']:
    d=base/'录像/FP32全任务/任务录像'/t['task']/clean_dirs[t['task']]
    for ep in t['episodes']:
        src=d/('episode%d.mp4'%ep['index']);dst=evidence/'clean/录像'/t['task']/src.name
        assert src.is_file();link_or_copy(src,dst)
        index.append({'setting':'clean','task':t['task'],'episode':ep['index'],'seed':ep['seed'],'success':ep['success'],'video':str(dst.relative_to(out))})
for r in videos:
    index.append({'setting':'randomized','task':r['task'],'episode':r['episode'],'seed':r['seed'],'success':r['success'],'video':str((evidence/'randomized/录像'/r['task']/Path(r['local_relative']).name).relative_to(out))})
assert len(index)==1000
with (evidence/'全部回合录像索引.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,list(index[0]));w.writeheader();w.writerows(index)
readme=out/'02_代码材料/README.md';text=readme.read_text();text=text.replace('这是6k Full-SFT候选模型的材料准备目录。官方results.json模板尚待已报名账号下载，随机场景评测仍在进行；请以提交状态.json为准。','这是cucu队6k Full-SFT候选模型的复现代码。正式results.json按官网下载模板生成，clean和randomized均填写真实10次/任务；完整评测日志与1000段录像在04_复现与调优说明/评测证据。')
readme.write_text(text)
expected=json.loads((out/'03_模型权重/checkpoint/文件校验清单.json').read_text())
for row in expected:
    p=out/'03_模型权重/checkpoint'/row['file'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
excluded={'提交状态.json','README_提交前须知.md','文件校验清单.json','results_draft.json','.DS_Store'}
files=[]
for p in sorted(out.rglob('*')):
    rel=p.relative_to(out)
    if any(x in rel.parts for x in ['.git','__pycache__','官方模板参考']):continue
    if not p.is_file() or p.name in excluded:continue
    if p.is_symlink():assert p.resolve().is_relative_to(out.resolve()),'External symlink '+str(p)
    files.append(p)
manifest={'team':'cucu','team_id':'1382038','captured_hong_kong':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'results':json.loads((out/'01_评测结果/results.json').read_text()),'files':[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
manifest_path=out/'文件校验清单.json';manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');files.append(manifest_path)
archive=out.parent/(out.name+'.zip');temp=archive.with_suffix('.zip.assembling')
with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,strict_timestamps=False,allowZip64=True) as pack:
    for p in files:
        pack.write(p,str(p.relative_to(out.parent)),compress_type=zipfile.ZIP_STORED if p.suffix in ['.safetensors','.mp4'] else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(temp) as pack:
    assert len(pack.namelist())==len(files) and pack.testzip() is None
    for r in manifest['files']:
        h=hashlib.sha256()
        with pack.open(out.name+'/'+r['path']) as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==r['sha256'],r['path']
temp.replace(archive)
receipt={'complete':True,'file':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files),'all_member_sha256_verified':True,'zip_crc_verified':True,'team_id':'1382038','submission_uploaded':False}
(out.parent/'cucu_正式提交包校验.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
p=out/'提交状态.json';state=json.loads(p.read_text());state.update({'final_zip_created':True,'final_zip_receipt':receipt,'randomized_result':{'tasks':50,'attempts':500,'successes':sum(r['successes'] for r in snapshot['tasks'])},'required_before_submission':['Upload verified ZIP via authenticated cucu Tianchi captain account; save submission receipt']});p.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
