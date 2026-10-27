"""Reconstruct connector-uploaded chunks, with hashes, without modifying Drive sources."""
from pathlib import Path
import hashlib, json, os, shutil

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024), b''): h.update(b)
    return h.hexdigest()

def assemble(source, target):
    source,target=Path(source),Path(target)
    target.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.is_file() and '.part' not in p.name and '.chunk96.' not in p.name and p.suffix!='.safetensors':
            shutil.copy2(p,target/p.name)
    index=json.loads((source/'model.safetensors.index.json').read_text())
    rows=json.loads((source/'chunks_manifest.json').read_text()) if (source/'chunks_manifest.json').exists() else []
    for name in sorted(set(index['weight_map'].values())):
        if (source/name).is_file():
            shutil.copy2(source/name,target/name)
            continue
        parts=sorted((r for r in rows if r['original']==name),key=lambda r:r['part'])
        assert parts and [r['part'] for r in parts]==list(range(len(parts))), f'Missing parts for {name}'
        for r in parts:
            p=source/r['name']
            assert p.is_file() and p.stat().st_size==r['bytes'] and digest(p)==r['sha256'], f'Incomplete chunk: {p.name}'
        temp=target/(name+'.assembling')
        with temp.open('wb') as out:
            for r in parts:
                with (source/r['name']).open('rb') as f: shutil.copyfileobj(f,out,16*1024*1024)
        os.replace(temp,target/name)
        print('ASSEMBLED',name,flush=True)
    # prepare.py validates each final shard against the original transfer manifest.

if __name__=='__main__':
    import sys
    assemble(*sys.argv[1:])
