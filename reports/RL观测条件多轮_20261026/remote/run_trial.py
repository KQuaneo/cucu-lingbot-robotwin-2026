import os,sys,time,json,signal,subprocess,concurrent.futures,datetime,traceback
from pathlib import Path
import numpy as np
from update_conditional import update
from conditional_model import initial_model
import torch
ROOT=Path(__file__).resolve().parent;PY=sys.executable;SOURCE='/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2'
DEADLINE=datetime.datetime(2026,10,26,20,20,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
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
def phase(name,version,seed_start,task='click_bell',fixed=None):
 assert time.time()<DEADLINE-120,'Insufficient time for another phase'
 save('phase.json',{'phase':name,'version':version})
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
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(worker,range(4)))
 save(f'{name}_outcomes.json',rows);print(name,'reward',sum(r['reward'] for r in rows),'/',len(rows),flush=True);return rows
def main():
 actor=initial_model();torch.save(actor.state_dict(),ROOT/'actor_v0.pt');save('status.json',{'status':'STARTING','conditional':True,'rounds_requested':3,'task':'click_bell','base_frozen':True})
 save('phase.json',{'phase':'startup','version':0})
 for g in range(4):
  f=(ROOT/f'server_gpu{g}.log').open('wb');handles.append(f);p=subprocess.Popen([PY,str(ROOT/'conditional_server.py'),'--root',str(ROOT),'--worker',str(g),'--port',str(13920+g)],env=env(g),cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);procs.append(p)
 started=time.time()
 while True:
  assert time.time()<DEADLINE and time.time()-started<600,'Model readiness timeout'
  assert all(p.poll() is None for p in procs),'Server exited'
  if all('BASE_FROZEN_FP32_READY' in (ROOT/f'server_gpu{g}.log').read_text(errors='replace') for g in range(4)):break
  time.sleep(3)
 time.sleep(1);results=[]
 for version in range(3):
  save('status.json',{'status':'RUNNING','round':version,'completed_rounds':results,'conditional':True})
  rows=phase('online_round'+str(version),version,510000+version*10000)
  result=update(ROOT,rows,version);save(f'update_round{version}.json',result);results.append({'round':version,'episodes':len(rows),'successes':sum(int(r['success']) for r in rows),'version_before':version,'version_after':version+1,'parameter_change_norm':result['parameter_change_norm'],'nonzero_update':result['nonzero_update'],'conditional_feature_probe_difference':result['conditional_feature_probe_difference']});save('round_results.json',results)
 validation=phase('validation_after',3,610000)
 save('status.json',{'status':'COMPLETED','conditional':True,'task':'click_bell','round_results':results,'fresh_online_episodes':12,'validation_episodes':4,'validation_successes':sum(int(r['success']) for r in validation),'nonzero_updates':sum(int(r['nonzero_update']) for r in results),'base_model_updated':False,'performance_improvement_claim':False,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
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
