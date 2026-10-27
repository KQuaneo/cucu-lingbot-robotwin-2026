from pathlib import Path
import hashlib, json, datetime, time, subprocess
base=Path(__file__).resolve().parent
model=Path('/workspace/runtime/models/full_sft_6000')
expected=json.loads((base/'expected_model_manifest.json').read_text())
checked=[];start=time.time()
for row in expected['files']:
    path=model/row['file'];assert path.stat().st_size==row['bytes'],str(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==row['sha256'],str(path)
    checked.append(row)
    print('MODEL_FILE_VERIFIED',path.name,flush=True)
receipt={'complete':True,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':checked,'seconds':time.time()-start,'source':'Mac independently verified 6k inference checkpoint manifest'}
(base/'model_verified.json').write_text(json.dumps(receipt,indent=2))
print('MODEL_VERIFICATION_COMPLETE',len(checked),flush=True)
