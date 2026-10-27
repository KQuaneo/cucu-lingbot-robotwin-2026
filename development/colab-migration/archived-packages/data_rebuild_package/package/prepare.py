import json, hashlib, argparse
from pathlib import Path
import yaml
p=argparse.ArgumentParser()
p.add_argument('--drive',required=True)
p.add_argument('--local',default='/content/lingbot-stage2')
p.add_argument('--mode',choices=['smoke','train'],default='smoke')
p.add_argument('--target',action='append',default=[])
p.add_argument('--repeat',type=int,default=1)
a=p.parse_args(); drive=Path(a.drive); local=Path(a.local); package=local/'package'
hf=local/'full_sft_6000'; data=local/'lerobot'
manifest=json.loads((hf/'transfer_manifest.json').read_text())
for f in manifest['files']:
    q=hf/f['file']; assert q.is_file(), f'Missing HF file: {q}'
    assert q.stat().st_size==f['bytes'], f'Size mismatch: {q}'
    h=hashlib.sha256()
    with q.open('rb') as s:
        for b in iter(lambda:s.read(8*1024*1024),b''): h.update(b)
    assert h.hexdigest()==f['sha256'], f'Hash mismatch: {q}'
expected=json.loads((package/'data_manifest.json').read_text())
names=sorted({Path(x['path']).parts[0] for x in expected['files']})
if a.mode=='smoke': names=['adjust_bottle_joint_v30']
counts={}; lines=[]
for name in names:
    d=data/name; info=json.loads((d/'meta/info.json').read_text())
    assert list((d/'data').rglob('*.parquet')), f'No parquet: {d}'
    assert list((d/'videos').rglob('*.mp4')), f'No videos: {d}'
    counts[name]={'episodes':info['total_episodes'],'frames':info['total_frames']}
    assert info['fps']==15 and info['total_episodes']==50, f'Dataset mismatch: {d}'
    lines.append(f'robotwin {d}')
if a.mode=='train':
    assert sum(x['episodes'] for x in counts.values())==2500
    assert sum(x['frames'] for x in counts.values())==549787
for target in a.target:
    name=target+'_joint_v30'; assert name in counts
    lines += [f'robotwin {data/name}'] * max(0,a.repeat-1)
local.mkdir(exist_ok=True,parents=True)
listing=local/(a.mode+'_data.txt'); listing.write_text('\n'.join(lines)+'\n')
c=yaml.safe_load((package/'baseline.yaml').read_text())
c['model'].update(model_path=str(hf),config_path=str(hf),tokenizer_path=str(package/'qwen'))
c['model'].update(attn_implementation='sdpa',vit_attn_implementation='sdpa')
c['data'].update(train_path=str(listing),robot_config_root=str(package/'lingbot/configs/robot_configs'),norm_stats_file=str(package/'lingbot/assets/norm_stats/robotwin.json'),num_workers=1,prefetch_factor=2,train_size=1000000)
t=c['train']; t.update(attention_implementation='eager',use_lora=False,train_expert_only=False,freeze_vit=False,init_device='cpu',enable_fsdp_offload=True,enable_activation_offload=False,enable_fp32=True,enable_mixed_precision=True,data_parallel_shard_size=1,data_parallel_replicate_size=1,micro_batch_size=1,gradient_accumulation_steps=2 if a.mode=='smoke' else 256,global_batch_size=2 if a.mode=='smoke' else 256,dyn_bsz=False,enable_gradient_checkpointing=True,use_compile=False,enable_resume=False,load_checkpoint_path=None,max_steps=2 if a.mode=='smoke' else 4000,save_steps=2 if a.mode=='smoke' else 500,save_hf_weights=False,output_dir=str(drive/'runs'/('full_sft_'+a.mode)),lr=2e-5,lr_min=1e-5,lr_warmup_ratio=0.02)
base=local/'teachers/base'; moge=local/'teachers/moge'
t['align_params']['depth']['moge_path']=str(moge/'model.pt')
t['align_params']['depth']['morgbd_path']=str(base/'depth/model.pt')
t['align_params']['video']['ckpt_path']=str(base/'dino_video/teacher_step_10000.pth')
t['align_params']['video']['config_path']=str(base/'dino_video/config.yaml')
for file in [t['align_params']['depth']['moge_path'],t['align_params']['depth']['morgbd_path'],t['align_params']['video']['ckpt_path'],t['align_params']['video']['config_path']]:
    assert Path(file).is_file(), f'Missing teacher: {file}'
out=local/(a.mode+'.yaml'); out.write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False))
(drive/'runs').mkdir(parents=True,exist_ok=True)
(drive/'runs'/(a.mode+'_data_audit.json')).write_text(json.dumps(counts,indent=2))
print(out)

(drive/'runs'/('full_sft_'+a.mode+'_provenance.json')).write_text(json.dumps({'initial_weights':'full_sft_6000','base_step':6000,'target_cumulative_step':10000,'additional_optimizer_steps':2 if a.mode=='smoke' else 4000,'optimizer_resume':False,'global_batch_size':t['global_batch_size'],'use_lora':False},indent=2))
