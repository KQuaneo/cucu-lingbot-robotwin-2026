from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,os,time,json,urllib.request,re,signal,traceback
ROOT=Path('/RoboTwin')
RUN=Path('/workspace/runtime/outputs/full_sft_6000_cross3_video')
MODEL=Path('/workspace/runtime/models/full_sft_6000')
PYTHON='/opt/robotwin-env/bin/python'
SERVER=ROOT/'experiments/lingbot_vla_v2_6b_robotwin/scripts/launch_official_server.sh'
RUN.mkdir(exist_ok=True)
assert (MODEL/'transfer_manifest.json').is_file()
assert not (RUN/'started.json').exists(), 'This run was already started'
(RUN/'started.json').write_text(json.dumps({'started_unix':time.time(),'model':str(MODEL)}))

def worker(gpu,jobs):
 port=13420+gpu
 env=os.environ.copy();env.update(HIP_VISIBLE_DEVICES=str(gpu),ROBOTWIN_DISABLE_CUROBO='1',ROBOTWIN_EE_PLANNER='mplib',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',AITER_TRITON_ONLY='1',FLASH_ATTENTION_TRITON_AMD_ENABLE='TRUE',LINGBOTVLA_TRAINING_CONFIG=str(MODEL/'lingbotvla_cli.yaml'))
 for key in ['ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']:env.pop(key,None)
 serverlog=RUN/f'server_gpu{gpu}.log'
 with serverlog.open('wb') as handle:
  server=subprocess.Popen(['bash',str(SERVER),str(gpu),str(port),str(serverlog),'False',str(MODEL)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   ready=False
   for _ in range(300):
    if server.poll() is not None:raise RuntimeError(f'Server {gpu} exited: {server.returncode}')
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as response:
      if response.read()==b'OK':ready=True;break
    except Exception:time.sleep(2)
   if not ready:raise RuntimeError(f'Server {gpu} failed readiness')
   for task,n,seed,label in jobs:
    cmd=[PYTHON,str(ROOT/'scripts/eval_policy_xpolicylab.py'),'--task_name',task,'--task_config','demo_clean','--policy_name','LingBot-VLA-v2','--protocol','lingbot_vla_v2','--host','127.0.0.1','--port',str(port),'--device_id','0','--seed','0','--test_num',str(n),'--expert_check','true','--accept_expert_info_on_failure','true','--episode_info_output',str(RUN/f'{label}_episodes.jsonl'),'--eval_batch','false','--additional_info','eval_video_log=true']
    if seed is not None:cmd+=['--seed_start',str(seed)]
    start=time.time()
    with (RUN/f'{label}.log').open('wb') as log:
     process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     try:status=process.wait(timeout=7200)
     except BaseException:
      os.killpg(process.pid,signal.SIGTERM);process.wait();raise
    s=(RUN/f'{label}.log').read_text(errors='ignore')
    match=re.findall(r'Final success rate:\s*(\d+)/(\d+)\s*=\s*([0-9.]+)%',s)
    result={'task':task,'gpu':gpu,'status':status,'elapsed_s':time.time()-start,'diagnostic':seed is not None,'seed_start':seed,'result':match[-1] if match else None}
    (RUN/f'{label}_done.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    assert status==0 and match and int(match[-1][1])==n,result
  finally:
   if server.poll() is None:
    os.killpg(server.pid,signal.SIGTERM)
    try:server.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()

jobs=[[(t,10,None,t)] for t in ['click_bell','open_microwave','stack_blocks_three']]
jobs.append([('adjust_bottle',1,100002,'adjust_bottle_seed100002'),('adjust_bottle',1,100005,'adjust_bottle_seed100005')])
try:
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(worker,g,j) for g,j in enumerate(jobs)]
  for f in futures:f.result()
 (RUN/'exit_code').write_text('0\n')
except Exception:
 (RUN/'exit_code').write_text('1\n');traceback.print_exc();raise
