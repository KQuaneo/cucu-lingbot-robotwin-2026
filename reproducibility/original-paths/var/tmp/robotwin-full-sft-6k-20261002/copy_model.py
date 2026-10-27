from pathlib import Path
import shutil,hashlib,json,datetime
src=Path("/var/tmp/robotwin-full-sft-6k-20261002/hf_ckpt_6000")
dst=Path("/workspace/runtime/models/full_sft_6000")
assert not dst.exists(), dst
stage=dst.with_name(dst.name+".copying")
shutil.copytree(src,stage)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(16*1024*1024),b""):h.update(b)
 return h.hexdigest()
records=[]
for p in sorted(src.iterdir()):
 if not p.is_file():continue
 a=sha(p);b=sha(stage/p.name)
 assert a==b,p.name
 records.append({"file":p.name,"bytes":p.stat().st_size,"sha256":a})
shutil.copy2("/var/tmp/robotwin-full-sft-6k-20261002/output/lingbotvla_cli.yaml",stage/"lingbotvla_cli.yaml")
(stage/"transfer_manifest.json").write_text(json.dumps({"source":str(src),"verified_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"files":records},indent=2))
stage.rename(dst)
print("VERIFIED",len(records),"FILES",sum(r["bytes"] for r in records),"BYTES",dst,flush=True)
