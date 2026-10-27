import subprocess,json,datetime,time,os
from pathlib import Path
base=Path(__file__).resolve().parent
while True:
    try:
        p=subprocess.run(['rocm-smi','--showuse','--showmemuse','--showmeminfo','vram','--json'],capture_output=True,text=True,timeout=40)
        record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':p.returncode,'stderr':p.stderr[-2000:]}
        try:record['gpus']=json.loads(p.stdout)
        except ValueError:record['raw']=p.stdout
        with (base/'gpu_monitor.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    except Exception as exc:
        with (base/'gpu_monitor.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':repr(exc)})+'\n')
    time.sleep(60)
