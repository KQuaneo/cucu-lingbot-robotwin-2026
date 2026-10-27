import argparse, json, math, os, re, subprocess
from pathlib import Path
import yaml
p=argparse.ArgumentParser()
p.add_argument('--drive',required=True)
p.add_argument('--local',default='/content/lingbot-stage2')
p.add_argument('--mode',choices=['smoke','resume','train'],required=True)
a=p.parse_args();drive=Path(a.drive);local=Path(a.local);package=local/'package'
assert (drive/'full_sft_6000').is_dir(), 'Drive not mounted or weights missing'
mode='smoke' if a.mode=='resume' else a.mode
subprocess.run([os.sys.executable,str(package/'prepare.py'),'--drive',str(drive),'--local',str(local),'--mode',mode],check=True)
config_file=local/(mode+'.yaml')
c=yaml.safe_load(config_file.read_text())
assert not c['train']['use_lora'] and c['train']['enable_fp32']
assert c['train']['enable_fsdp_offload'] and not c['train']['freeze_vit']
if a.mode=='resume':
    checkpoint=drive/'runs/full_sft_smoke/checkpoints/global_step_2'
    assert (checkpoint/'model/.metadata').is_file() and (checkpoint/'optimizer/.metadata').is_file()
    c['train'].update(load_checkpoint_path=str(checkpoint),enable_resume=True,max_steps=3,save_steps=3)
    config_file=local/'resume.yaml';config_file.write_text(yaml.safe_dump(c,sort_keys=False))
if a.mode=='train':
    gate=drive/'runs/full_sft_validation.json'
    assert gate.is_file() and json.loads(gate.read_text())['resume_to_3'], 'Full-SFT smoke/resume gate not passed'
    assert c['train']['max_steps']==4000 and c['train']['global_batch_size']==256
    checkpoints=drive/'runs/full_sft_train/checkpoints'
    assert not (checkpoints/'global_step_4000').exists(), 'Finished output already exists'
    c['train']['enable_resume'] = checkpoints.is_dir() and any(checkpoints.glob('global_step_*'))
    config_file.write_text(yaml.safe_dump(c,sort_keys=False))
env=os.environ.copy()
for key in ['HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','AITER_TRITON_ONLY','FLASH_ATTENTION_TRITON_AMD_ENABLE']:env.pop(key,None)
env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',LINGBOT_FP32_TRAIN='1',QWEN3VL_PATH=str(package/'qwen'),PYTHONPATH=str(package/'lingbot/lingbotvla/models/vla/vision_models/morgbd_clean/3rd/utils3d'))
if a.mode=='train':env['LINGBOT_CHECKPOINT_INTERVAL_SECONDS']='7200'
command=[os.sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=1','-m','tasks.vla.train_lingbotvla',str(config_file)]
log=drive/'runs'/('full_sft_'+a.mode+'.log');log.parent.mkdir(parents=True,exist_ok=True)
with log.open('w') as output:
    process=subprocess.Popen(command,cwd=package/'lingbot',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:print(line,end='',flush=True);output.write(line);output.flush()
    assert process.wait()==0, f'Training failed: {log}'
text=log.read_text()
assert 'Full-SFT parameters:' in text and 'LoRA enabled:' not in text
step=3 if a.mode=='resume' else 2 if a.mode=='smoke' else 4000
out=drive/'runs'/('full_sft_'+mode)/'checkpoints'/f'global_step_{step}'
assert (out/'model/.metadata').is_file() and (out/'optimizer/.metadata').is_file()
if a.mode=='resume':
    assert 'Starting training from scratch' not in text and 'Skip loading Optimizer' not in text
    # Successful state restore and extra_state-derived Step 3 prove the checkpoint was used.
    assert 'Verified non-empty restored CPU AdamW moments.' in text
    assert re.search(r'Step 3/',text), 'Resume did not reach step 3'
    (drive/'runs/full_sft_validation.json').write_text(json.dumps({'resume_to_3':True,'checkpoint':str(out),'precision':'FP32','use_lora':False},indent=2))
print(json.dumps({'mode':a.mode,'new_optimizer_step':step,'cumulative_label':6000+step,'checkpoint':str(out)}))
