from pathlib import Path
import json,os,shutil,subprocess,tarfile,time,concurrent.futures
D=Path('/content/drive/MyDrive/LingBot-Colab-Stage2-20261008');L=Path('/content/lingbot-stage2');P=L/'package'
def state(phase,**extra):
    (D/'micro_benchmark_status.json').write_text(json.dumps(dict(phase=phase,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),formal_training_started=False,precision='FP32',use_lora=False,global_batch_size=256,**extra),indent=2))
    print(phase,extra,flush=True)
def run(cmd): subprocess.run(cmd,check=True)
try:
    assert D.is_dir(),'Drive not mounted'
    L.mkdir(exist_ok=True)
    lock=L/'benchmark_restore.started';assert not lock.exists(),'Restore already launched';lock.write_text(str(os.getpid()))
    state('benchmark_restoring_package')
    with tarfile.open(D/'full_sft_package_patched_20261010.tar.gz') as f:f.extractall(L,filter='data')
    p=P/'prepare.py';p.write_text(p.read_text().replace('num_workers=1','num_workers=0'))
    p=P/'lingbot/lingbotvla/checkpoint/checkpointer.py';p.write_text(p.read_text().replace('sync_files=False','sync_files=True'))
    state('benchmark_restoring_cuda')
    run(['bash',str(P/'setup_cuda.sh')])
    py='/content/lingbot-venv/bin/python'
    run(['python','-m','uv','pip','install','--python',py,'--no-deps','utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183','pipeline @ git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6'])
    state('benchmark_caching_inputs')
    def cache(name):
        src=D/name;dst=L/name
        shutil.copytree(src,dst,dirs_exist_ok=True)
        files=[p for p in src.rglob('*') if p.is_file()]
        assert all((dst/p.relative_to(src)).stat().st_size==p.stat().st_size for p in files)
        print('CACHE_VERIFIED',name,len(files),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(cache,['lerobot','full_sft_6000']))
    state('benchmark_restoring_teachers')
    run([py,'-c',"from huggingface_hub import snapshot_download; snapshot_download('robbyant/lingbot-vla-v2-6b',revision='11c703bf6a5c1f45b3b69168482da11fdbba53d7',allow_patterns=['depth/model.pt','dino_video/teacher_step_10000.pth','dino_video/config.yaml'],local_dir='/content/lingbot-stage2/teachers/base'); snapshot_download('Ruicheng/moge-2-vitb-normal',revision='ca5f0e07ff01d3e5a364c1d954ed12ee1814b368',allow_patterns=['model.pt'],local_dir='/content/lingbot-stage2/teachers/moge')"])
    state('benchmark_validating_inputs')
    run([py,str(P/'prepare.py'),'--drive',str(D),'--local',str(L),'--mode','train'])
    shutil.copyfile(D/'benchmark_micro.py',L/'benchmark_micro.py')
    run([py,'-u',str(L/'benchmark_micro.py')])
except Exception as e:
    if D.is_dir():state('benchmark_restore_failed',error=repr(e))
    raise
