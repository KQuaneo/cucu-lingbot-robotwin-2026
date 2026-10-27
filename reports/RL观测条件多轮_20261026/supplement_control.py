import sys,time,json,subprocess,traceback
from pathlib import Path
import run_trial as trial
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 code=0
 try:
  assert int((ROOT/'exit_code').read_text())==0
  trial.save('supplement_status.json',{'status':'STARTING','purpose':'paired initial conditional actor v0 control; no training'})
  for g in range(4):
   f=(ROOT/f'control_server_gpu{g}.log').open('wb');trial.handles.append(f);p=subprocess.Popen([sys.executable,str(ROOT/'conditional_server.py'),'--root',str(ROOT),'--worker',str(g),'--port',str(13920+g)],env=trial.env(g),cwd=trial.SOURCE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);trial.procs.append(p)
  started=time.time()
  while True:
   assert time.time()<trial.DEADLINE-120 and time.time()-started<450
   assert all(p.poll() is None for p in trial.procs)
   if all('BASE_FROZEN_FP32_READY' in (ROOT/f'control_server_gpu{g}.log').read_text(errors='replace') for g in range(4)):break
   time.sleep(3)
  time.sleep(1);after=json.loads((ROOT/'validation_after_outcomes.json').read_text());fixed={r['worker']:r for r in after}
  control=trial.phase('validation_control_v0',0,610000,fixed=fixed)
  paired=[]
  for c in control:
   r=fixed[c['worker']];paired.append({'worker':c['worker'],'seed':c['seed'],'seed_equal':c['seed']==r['seed'],'instruction_equal':c['instruction']==r['instruction'],'control_success':c['success'],'trained_success':r['success']})
  trial.save('conditional_paired_validation.json',paired);trial.save('supplement_status.json',{'status':'COMPLETED','control_episodes':4,'control_successes':sum(int(r['success']) for r in control),'trained_episodes':4,'trained_successes':sum(int(r['success']) for r in after),'matched_seed_instruction_pairs':sum(p['seed_equal'] and p['instruction_equal'] for p in paired),'base_frozen':True,'new_training_episodes':0})
 except BaseException as e:
  code=1;traceback.print_exc();trial.save('supplement_status.json',{'status':'FAILED','error':repr(e)})
 finally:
  for p in reversed(trial.procs):trial.stop(p)
  for f in trial.handles:f.close()
  (ROOT/'supplement_exit_code').write_text(str(code)+'\n')
 sys.exit(code)
