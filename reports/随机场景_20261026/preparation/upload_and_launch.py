from pathlib import Path
import subprocess,io,tarfile,json,datetime
base=Path(__file__).resolve().parent
workspace=base.parents[2]
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-p','35051','root@YOUR_SERVER_IP']
remote='/workspace/runtime/experiments/randomized_fp32_20261026'
patch=workspace/'配置与代码/原路径/RoboTwin/scripts/eval_policy_xpolicylab.py'
assert patch.is_file(),str(patch)
payload=io.BytesIO()
with tarfile.open(fileobj=payload,mode='w:gz') as tar:
    for name in ['run.sh','run_fp32_benchmark.py','serve_fp32.py','launch_fp32.sh','verify_model.py','gpu_monitor.py','expected_model_manifest.json']:
        tar.add(base/name,arcname=name)
    tar.add(patch,arcname='eval_policy_valid_retry.py')
subprocess.run(ssh+['mkdir -p '+remote+' && tar --no-same-owner -xzf - -C '+remote],input=payload.getvalue(),check=True,timeout=90)
code="""
from pathlib import Path
import hashlib,shutil,subprocess,json,os
base=Path('/workspace/runtime/experiments/randomized_fp32_20261026')
script=Path('/RoboTwin/scripts/eval_policy_xpolicylab.py')
current=hashlib.sha256(script.read_bytes()).hexdigest()
assert current in ['b44e44855846bda01ce1f83a173b851b8531cb406731940f9ffe914f8abab999','8f1722bfb25e60ebae17c5bd43a75733e239b63f95e3bd47a7cd434cee224b97'],current
if current!='8f1722bfb25e60ebae17c5bd43a75733e239b63f95e3bd47a7cd434cee224b97':
 shutil.copy2(script,base/'eval_policy_before_valid_retry.py')
 shutil.copy2(base/'eval_policy_valid_retry.py',script)
assert hashlib.sha256(script.read_bytes()).hexdigest()=='8f1722bfb25e60ebae17c5bd43a75733e239b63f95e3bd47a7cd434cee224b97'
for name in ['run_fp32_benchmark.py','serve_fp32.py','verify_model.py','gpu_monitor.py']:
 compile((base/name).read_text(),str(base/name),'exec')
assert not (base/'launcher.pid').exists(),'Already launched; inspect PID rather than duplicate'
for port in range(13820,13828):
 import socket
 sock=socket.socket();sock.bind(('127.0.0.1',port));sock.close()
with (base/'launcher.log').open('ab') as log:
 p=subprocess.Popen(['bash',str(base/'run.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(base/'launcher.pid').write_text(str(p.pid)+'\\n')
print(json.dumps({'pid':p.pid,'remote_preparation':str(base),'run_name':'full_sft_6000_randomized50x10_fp32_8gpu_video_20261026','precision':'FP32','task_config':'demo_randomized','episodes_per_task':10,'gpu_count':8,'model_verification_then_benchmark':True}))
"""
p=subprocess.run(ssh+['/opt/robotwin-env/bin/python -'],input=code,text=True,capture_output=True,timeout=90)
print(p.stdout,p.stderr);p.check_returncode()
receipt=json.loads(p.stdout.strip().splitlines()[-1]);receipt['launched_from_mac_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(base/'launch_receipt.json').write_text(json.dumps(receipt,indent=2))
