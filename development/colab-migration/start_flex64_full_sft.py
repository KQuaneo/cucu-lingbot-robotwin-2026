"""Restore and launch a new, bounded Full-SFT run from the verified HF 6k snapshot."""
from pathlib import Path
import concurrent.futures, fcntl, json, os, re, shutil, subprocess, sys, tarfile, time, traceback
D=Path('/content/drive/MyDrive/LingBot-Colab-Stage2-20261008'); L=Path('/content/lingbot-stage2'); P=L/'package'
assert D.is_dir(), 'Drive is not mounted'
L.mkdir(exist_ok=True)
lock=(L/'formal_flex64.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
started=time.time(); deadline=started+24*3600
R=D/'runs'/('full_sft_flex_cached_micro64_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())); R.mkdir(parents=True)
status=dict(phase='restoring_package',formal_training_started=False,precision='BF16 mixed precision (FP32 master/optimizer)',use_lora=False,micro_batch_size=64,gradient_accumulation_steps=4,global_batch_size=256,base_step=6000,additional_step=0,cumulative_step=6000,target_cumulative_step=10000,optimizer_resume=False,run_directory=str(R),log=str(R/'training.log'),deadline_unix=deadline)
def state(phase,**kw):
    status.update(phase=phase,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    for dest in [D/'pipeline_status.json',R/'status.json']:
        tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(status,ensure_ascii=False,indent=2));tmp.replace(dest)
    print(phase,kw,flush=True)
def run(cmd): subprocess.run(cmd,check=True)
def cache():
    files=[]
    root=D/'lerobot'
    for p in root.rglob('*'):
        if p.is_file(): files.append((p,L/'lerobot'/p.relative_to(root),p.stat().st_size))
    manifest=json.loads((D/'full_sft_6000/transfer_manifest.json').read_text())
    for x in manifest['files']:
        assert Path(x['file']).name==x['file']
        files.append((D/'full_sft_6000'/x['file'],L/'full_sft_6000'/x['file'],x['bytes']))
    q=D/'full_sft_6000/transfer_manifest.json';files.append((q,L/'full_sft_6000/transfer_manifest.json',q.stat().st_size))
    total=sum(x[2] for x in files); done=0; count=0; last=0
    def cp(item):
        src,dst,n=item;assert src.stat().st_size==n, str(src);dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists() or dst.stat().st_size!=n:
            tmp=dst.with_suffix(dst.suffix+'.copying');shutil.copyfile(src,tmp);assert tmp.stat().st_size==n;tmp.replace(dst)
        return n
    state('caching_inputs',cache_total_bytes=total,cache_completed_bytes=0,cache_files_total=len(files))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for f in concurrent.futures.as_completed([ex.submit(cp,x) for x in files]):
            done+=f.result();count+=1
            if time.time()-last>10 or count==len(files):
                state('caching_inputs',cache_completed_bytes=done,cache_files_completed=count);last=time.time()
try:
    state('restoring_package')
    with tarfile.open(D/'full_sft_package_patched_20261010.tar.gz') as a:a.extractall(L,filter='data')
    p=P/'prepare.py';p.write_text(p.read_text().replace('num_workers=1','num_workers=0'))
    p=P/'lingbot/lingbotvla/checkpoint/checkpointer.py';p.write_text(p.read_text().replace('sync_files=False','sync_files=True'))
    state('restoring_cuda_and_cache')
    with (R/'setup.log').open('w') as setup:
        process=subprocess.Popen(['bash',str(P/'setup_cuda.sh')],stdout=setup,stderr=subprocess.STDOUT)
        cache()
        state('waiting_for_cuda');assert process.wait()==0,'CUDA setup failed; see setup.log'
    py='/content/lingbot-venv/bin/python'
    run(['python','-m','uv','pip','install','--python',py,'--no-deps','utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183','pipeline @ git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6'])
    state('restoring_teachers')
    run([py,'-c',"from huggingface_hub import snapshot_download; snapshot_download('robbyant/lingbot-vla-v2-6b',revision='11c703bf6a5c1f45b3b69168482da11fdbba53d7',allow_patterns=['depth/model.pt','dino_video/teacher_step_10000.pth','dino_video/config.yaml'],local_dir='/content/lingbot-stage2/teachers/base'); snapshot_download('Ruicheng/moge-2-vitb-normal',revision='ca5f0e07ff01d3e5a364c1d954ed12ee1814b368',allow_patterns=['model.pt'],local_dir='/content/lingbot-stage2/teachers/moge')"])
    state('validating_inputs')
    run([py,str(P/'prepare.py'),'--drive',str(D),'--local',str(L),'--mode','train'])
    # The environment has PyYAML; the launcher itself uses the system interpreter.
    config_script=R/'configure.py'
    config_script.write_text('''from pathlib import Path
import yaml
L=Path('/content/lingbot-stage2')
c=yaml.safe_load((L/'train.yaml').read_text())
c['train'].update(attention_implementation='flex_cached',enable_fp32=False,enable_mixed_precision=True,use_lora=False,train_expert_only=False,freeze_vit=False,micro_batch_size=64,gradient_accumulation_steps=4,global_batch_size=256,max_steps=4000,save_steps=0,save_epochs=0,save_hf_weights=False,enable_resume=False,load_checkpoint_path=None,output_dir=OUT)
assert c['train']['enable_fsdp_offload'] and c['data']['num_workers']==0
(L/'formal_flex64.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
'''.replace('OUT',repr(str(R/'output'))))
    run([py,str(config_script)]);shutil.copyfile(L/'formal_flex64.yaml',R/'train.yaml')
    T=P/'lingbot/tasks/vla/train_lingbotvla.py'
    assert 'LINGBOT_CHECKPOINT_INTERVAL_SECONDS' in T.read_text()
    env=os.environ.copy()
    for k in list(env):
        if k.startswith('LINGBOT_') or k in ['HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','AITER_TRITON_ONLY','FLASH_ATTENTION_TRITON_AMD_ENABLE']:env.pop(k,None)
    env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',MALLOC_ARENA_MAX='2',QWEN3VL_PATH=str(P/'qwen'),PYTHONPATH=str(P/'lingbot/lingbotvla/models/vla/vision_models/morgbd_clean/3rd/utils3d'),LINGBOT_CHECKPOINT_INTERVAL_SECONDS='10800',LINGBOT_TRAIN_DEADLINE_UNIX=str(deadline))
    state('loading_formal_model')
    command=[py,'-m','torch.distributed.run','--standalone','--nproc-per-node=1','-m','tasks.vla.train_lingbotvla',str(L/'formal_flex64.yaml')]
    proc=subprocess.Popen(command,cwd=P/'lingbot',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    state('loading_formal_model',trainer_pid=proc.pid)
    with (R/'training.log').open('w') as log:
        for line in proc.stdout:
            log.write(line);log.flush()
            m=re.search(r'Step (\d+)/[^\r\n]+',line)
            if m:
                step=int(m[1]);state('running_train',formal_training_started=True,additional_step=step,cumulative_step=6000+step,last_step_log=m[0])
            if 'Distributed checkpoint saved at ' in line and 'successfully!' in line:
                state('running_train',last_checkpoint_step=status['additional_step'],last_checkpoint_log=line.strip())
    code=proc.wait();assert code==0,f'Training exited {code}; see training.log'
    assert status['additional_step']>0, 'No optimizer update completed'
    state('training_complete' if status['additional_step']>=4000 else 'training_stopped_at_budget',exit_code=code)
except Exception as e:
    state('failed',error=repr(e));traceback.print_exc();raise
