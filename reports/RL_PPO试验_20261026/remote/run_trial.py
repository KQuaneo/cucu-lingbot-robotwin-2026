import os,sys,time,json,signal,subprocess,concurrent.futures,datetime,traceback
from pathlib import Path
import numpy as np
from update_ppo import update,self_test
ROOT=Path(__file__).resolve().parent;PY=sys.executable;SOURCE='/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2'
DEADLINE=datetime.datetime(2026,10,26,19,56,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
procs=[];handles=[]
def save(name,obj):
 p=ROOT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def stop(p):
 if p.poll() is None:
  try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
  except ProcessLookupError:pass
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
def env(g):
 e=os.environ.copy();e.update(HIP_VISIBLE_DEVICES=str(g),AITER_TRITON_ONLY='1',FLASH_ATTENTION_TRITON_AMD_ENABLE='TRUE',QWEN3VL_PATH='/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/Qwen3-VL-4B-Instruct-config-tokenizer',LINGBOTVLA_TRAINING_CONFIG='/workspace/runtime/models/full_sft_6000/lingbotvla_cli.yaml',PYTHONUNBUFFERED='1',ROBOTWIN_DISABLE_CUROBO='1',ROBOTWIN_EE_PLANNER='mplib',PYOPENGL_PLATFORM='egl',PYTHONPATH=SOURCE+':/RoboTwin:/RoboTwin/scripts:/RoboTwin/description/utils')
 e.pop('CUDA_VISIBLE_DEVICES',None);e.pop('ROCR_VISIBLE_DEVICES',None);return e
def phase(name,mean,seed_start,task,fixed=None):
 assert time.time()<DEADLINE-180,'Insufficient time for another phase'
 save('phase.json',{'phase':name,'mean':mean})
 def worker(g):
  f=(ROOT/f'eval_{name}_gpu{g}.log').open('wb');handles.append(f)
  cmd=[PY,str(ROOT/'eval_rl.py'),'--task_name',task,'--task_config','demo_clean','--policy_name','cucu-residual-rl-research','--protocol','lingbot_vla_v2','--host','127.0.0.1','--port',str(13920+g),'--device_id','0','--seed_start',str(fixed[g]['seed'] if fixed else seed_start+g*100),'--test_num','1','--expert_check','true','--accept_expert_info_on_failure','true','--eval_batch','false','--episode_info_output',str(ROOT/f'expert_{name}_gpu{g}.jsonl'),'--additional_info','eval_video_log=true']
  e=env(g)
  if fixed:e['CUCU_RL_FIXED_INSTRUCTION']=fixed[g]['instruction']
  p=subprocess.Popen(cmd,cwd='/RoboTwin',env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);procs.append(p)
  try:code=p.wait(timeout=max(1,min(780,DEADLINE-time.time())))
  except subprocess.TimeoutExpired:stop(p);raise RuntimeError('Rollout deadline exceeded')
  f.flush();assert code==0,(name,g,code)
  text=(ROOT/f'eval_{name}_gpu{g}.log').read_text(errors='replace');assert 'Policy rollout error' not in text
  records=list((ROOT/'trajectories'/name).glob(f'worker{g}_seed*/episode.json'));assert len(records)==1,(name,g,len(records))
  row=json.loads(records[0].read_text());assert row['terminal'] and row['chunks']>0
  return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(worker,range(8)))
 save(f'{name}_outcomes.json',rows);print(name,'reward',sum(r['reward'] for r in rows),'/',len(rows),flush=True);return rows
def main():
 import hashlib
 prior=Path('/workspace/runtime/experiments/residual_rl_multitask_20261026');tasks=['beat_block_hammer','rotate_qrcode','press_stapler'];provenance=[];updates={}
 save('self_test.json',self_test())
 for task in tasks:
  path=prior/(task+'_train0_outcomes.json');data=path.read_bytes();rows=json.loads(data);assert len(rows)==8 and all(r['task_name']==task and r['mean']==[0.]*12 for r in rows)
  save(task+'_reused_train_outcomes.json',rows);provenance.append({'task':task,'source':str(path),'sha256':hashlib.sha256(data).hexdigest(),'episodes':8,'validation_not_used_for_update':True})
  updates[task]=update(rows);save(task+'_ppo_update.json',updates[task])
  save(task+'_control_reference.json',json.loads((prior/(task+'_control_outcomes.json')).read_text()))
 save('training_provenance.json',provenance);save('status.json',{'status':'STARTING','method':'clipped PPO residual actor','new_training_rollouts':0,'reused_on_policy_rollouts':24,'base_frozen':True})
 save('phase.json',{'phase':'startup','mean':[0.]*12})
 for g in range(8):
  f=(ROOT/f'server_gpu{g}.log').open('wb');handles.append(f);p=subprocess.Popen([PY,str(ROOT/'residual_server.py'),'--root',str(ROOT),'--worker',str(g),'--port',str(13920+g)],env=env(g),cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);procs.append(p)
 started=time.time()
 while True:
  assert time.time()<DEADLINE and time.time()-started<600,'Model readiness timeout'
  assert all(p.poll() is None for p in procs),'Server exited'
  if all('BASE_FROZEN_FP32_READY' in (ROOT/f'server_gpu{g}.log').read_text(errors='replace') for g in range(8)):break
  time.sleep(5)
 time.sleep(2);results=[]
 for i,task in enumerate(tasks):
  save('status.json',{'status':'RUNNING','method':'clipped PPO residual actor','current_task':task,'completed_tasks':results})
  u=updates[task];assert u['nonzero_update'];control=json.loads((ROOT/(task+'_control_reference.json')).read_text());fixed={r['worker']:r for r in control}
  save(task+'_ppo_checkpoint.json',{'mean':u['mean_after'],'std':1.,'scale_radians':.004,'transform':'scale*tanh(latent)','joint_indices':[0,1,2,3,4,5,7,8,9,10,11,12],'base_model':'/workspace/runtime/models/full_sft_6000','base_frozen':True,'optimization_epochs':4,'training_batches':1,'method':'clipped PPO actor'})
  after=phase(task+'_ppo_after',u['mean_after'],410000+i*10000,task,fixed=fixed)
  paired=[{'worker':r['worker'],'seed':r['seed'],'instruction_equal':r['instruction']==fixed[r['worker']]['instruction'],'seed_equal':r['seed']==fixed[r['worker']]['seed'],'noise_equal':r['epsilon']==fixed[r['worker']]['epsilon'],'control_success':fixed[r['worker']]['success'],'ppo_success':r['success']} for r in after];save(task+'_paired_validation.json',paired)
  old_result=next(r for r in json.loads((prior/'task_results.json').read_text()) if r['task']==task)
  results.append({'task':task,'control_successes':sum(int(r['success']) for r in control),'reinforce_successes':old_result['after_successes'],'ppo_successes':sum(int(r['success']) for r in after),'validation_rollouts_per_policy':8,'all_pairs_matched':all(p['instruction_equal'] and p['seed_equal'] and p['noise_equal'] for p in paired),'mean_change_norm':u['parameter_change_norm'],'latent_KL':u['latent_KL_old_new'],'ppo_epochs':4,'new_training_rollouts':0,'base_model_updated':False})
  save('task_results.json',results)
 save('status.json',{'status':'COMPLETED','method':'clipped PPO residual actor','task_results':results,'new_validation_rollouts':24,'reused_training_rollouts':24,'base_model_updated':False,'performance_improvement_claim':False,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
if __name__=='__main__':
 code=0
 try:main()
 except BaseException as e:
  code=1;traceback.print_exc();save('failure.json',{'error':repr(e),'traceback':traceback.format_exc()});save('status.json',{'status':'FAILED','error':repr(e)})
 finally:
  for p in reversed(procs):stop(p)
  for f in handles:f.close()
  (ROOT/'exit_code').write_text(str(code)+'\n')
 sys.exit(code)
