#!/usr/bin/env python3
"""Verify release transport files and reconstruct exact original ZIP bytes (stdlib only)."""
from pathlib import Path
import argparse,json,hashlib

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
 return h.hexdigest()
def restore(directory,manifest):
 for archive in manifest['archives']:
  name=archive['name'];assert Path(name).name==name
  target=directory/name;parts=archive.get('parts',[])
  if target.exists():
   if target.stat().st_size==archive['bytes'] and digest(target)==archive['sha256']:
    print('VERIFIED',name);continue
   raise RuntimeError('Existing ZIP does not match; move it aside before retrying: '+str(target))
  if not parts:raise FileNotFoundError('Missing direct ZIP: '+str(target))
  tmp=directory/(name+'.assembling');assert not tmp.exists(),'Temporary assembly exists; review/remove it first: '+str(tmp)
  total=0;whole=hashlib.sha256()
  try:
   with tmp.open('xb') as out:
    for part in parts:
     assert Path(part['name']).name==part['name'];source=directory/part['name'];h=hashlib.sha256();n=0
     with source.open('rb') as f:
      for data in iter(lambda:f.read(8*1024*1024),b''):out.write(data);h.update(data);whole.update(data);n+=len(data);total+=len(data)
     assert n==part['bytes'] and h.hexdigest()==part['sha256'],'Part checksum failed: '+part['name']
   assert total==archive['bytes'] and whole.hexdigest()==archive['sha256'],'Reconstructed ZIP checksum failed: '+name
   tmp.replace(target);print('RESTORED_AND_VERIFIED',name)
  except BaseException:
   tmp.unlink(missing_ok=True);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('evidence-downloads'));p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'manifests/release_assets.json');a=p.parse_args();assert a.directory.is_dir();restore(a.directory,json.loads(a.manifest.read_text()))
