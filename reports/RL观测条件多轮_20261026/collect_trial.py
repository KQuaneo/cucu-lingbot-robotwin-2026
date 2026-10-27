from pathlib import Path
import subprocess,tarfile,io,json,hashlib
BASE=Path(__file__).resolve().parent
SCRIPT=r'''from pathlib import Path
import tarfile,sys,json,hashlib,io,re
b=Path('/workspace/runtime/experiments/conditional_rl_online_20261026');files=list(b.glob('*.py'))+list(b.glob('*.json'))+list(b.glob('*.jsonl'))+list(b.glob('*.log'))+list(b.glob('*.pt'))+list(b.glob('*exit_code'))
# Only copy completed episode arrays; no partially written active chunk.
for p in b.glob('trajectories/*/*/episode.json'):
 r=json.loads(p.read_text())
 if r.get('terminal'):files+=list(p.parent.glob('*'))
videos=[]
for p in b.glob('eval_*.log'):
 t=re.sub(r'\x1b\[[0-9;]*m','',p.read_text(errors='replace'));paths=re.findall(r'Data has been saved to\s+([^\r\n]+)',t)
 if paths:
  d=Path(paths[-1].strip());d=d if d.is_absolute() else Path('/RoboTwin')/d
  for v in d.parent.glob('*.mp4'):videos.append(v)
manifest=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as a:
 for p in sorted(set(files)):
  if not p.is_file():continue
  rel='remote/'+str(p.relative_to(b));data=p.read_bytes();m=tarfile.TarInfo(rel);m.size=len(data);a.addfile(m,io.BytesIO(data));manifest.append({'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 for p in sorted(set(videos)):
  rel='videos/'+str(p.relative_to('/RoboTwin/eval_result'));data=p.read_bytes();m=tarfile.TarInfo(rel);m.size=len(data);a.addfile(m,io.BytesIO(data));manifest.append({'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 data=json.dumps(manifest).encode();m=tarfile.TarInfo('transfer_manifest.json');m.size=len(data);a.addfile(m,io.BytesIO(data))
'''
def collect():
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-p','35051','root@YOUR_SERVER_IP','/opt/robotwin-env/bin/python -'],input=SCRIPT.encode(),capture_output=True,timeout=160);assert p.returncode==0,p.stderr.decode()
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:gz') as a:
  entries={m.name:a.extractfile(m).read() for m in a if m.isfile()}
 manifest=json.loads(entries['transfer_manifest.json'])
 for r in manifest:assert len(entries[r['path']])==r['bytes'] and hashlib.sha256(entries[r['path']]).hexdigest()==r['sha256']
 for name,data in entries.items():
  assert not name.startswith('/') and '..' not in Path(name).parts
  dst=BASE/name;dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix(dst.suffix+'.collecting');tmp.write_bytes(data);tmp.replace(dst)
 status=json.loads(entries.get('remote/status.json',b'{}'));print(json.dumps({'files_sha_verified':len(manifest),'bytes':sum(r['bytes'] for r in manifest),'status':status},ensure_ascii=False))
if __name__=='__main__':collect()
