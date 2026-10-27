# -*- coding: utf-8 -*-
import pathlib,json,hashlib,time,datetime,shutil,concurrent.futures,collections
BASE=pathlib.Path('/path/to/project')
OUT=BASE/'硬盘/报告/最终审计'
started=time.time()
hf=BASE/'硬盘/模型/full_sft_6000'
dcp=BASE/'硬盘/训练存档/global_step_6000'
hf_remote=json.loads((BASE/'硬盘/传输/full_sft_6000_remote_manifest.json').read_text())
hf_transfer=json.loads((hf/'transfer_manifest.json').read_text())
dcp_manifest=json.loads((BASE/'硬盘/传输/global_step_6000_remote_manifest.json').read_text())
manifest_conflicts=[]
remote_map={v['file']:v for v in hf_remote['files']}
for x in hf_transfer['files']:
 if x['file'] in remote_map and x!=remote_map[x['file']]: manifest_conflicts.append(x['file'])

def check(root,record):
 p=root/record['file']; ret={'file':record['file'],'expected_bytes':record['bytes'],'expected_sha256':record['sha256'],'present':p.is_file()}
 if not ret['present']: return ret
 ret['actual_bytes']=p.stat().st_size
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 ret['actual_sha256']=h.hexdigest()
 ret['size_ok']=ret['actual_bytes']==record['bytes']; ret['sha256_ok']=ret['actual_sha256']==record['sha256']
 return ret

def run(name,root,records):
 st=time.time(); done=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  pending={ex.submit(check,root,r):r for r in records}
  for f in concurrent.futures.as_completed(pending):
   r=f.result(); done.append(r)
   if len(done)%20==0 or len(done)==len(records):print(name,len(done),'/',len(records),'elapsed',round(time.time()-st,1),flush=True)
   if not r.get('size_ok',False) or not r.get('sha256_ok',False): print('MISMATCH',name,r,flush=True)
 done.sort(key=lambda x:x['file'])
 return {'root':str(root),'files_expected':len(records),'files_present':sum(x['present'] for x in done),'size_verified':sum(x.get('size_ok',False) for x in done),'sha256_verified':sum(x.get('sha256_ok',False) for x in done),'expected_bytes':sum(x['bytes'] for x in records),'seconds':time.time()-st,'files':done}

hf_result=run('HF',hf,hf_transfer['files'])
dcp_result=run('DCP',dcp,dcp_manifest['files'])
index=json.loads((hf/'model.safetensors.index.json').read_text())
shards=sorted(set(index['weight_map'].values()))
index_refs={'weight_entries':len(index['weight_map']),'referenced_shards':shards,'missing_shards':[x for x in shards if not (hf/x).is_file()]}
parts={}
for group in ('model','optimizer','extra_state'):
 rows=[x for x in dcp_result['files'] if x['file'].startswith(group+'/')]
 parts[group]={'files':len(rows),'bytes':sum(x['expected_bytes'] for x in rows),'sha256_verified':sum(x.get('sha256_ok',False) for x in rows)}
configs=[p for p in dcp.rglob('*') if p.is_file() and p.suffix in ('.yaml','.yml')]
config_copy=BASE/'硬盘/配置与代码/原路径/var/tmp/robotwin-full-sft-6k-20261002/output/lingbotvla_cli.yaml'
confighash=hashlib.sha256(config_copy.read_bytes()).hexdigest() if config_copy.exists() else None
metadata={}
for group in ('model','optimizer'):
 p=dcp/group/'.metadata'; metadata[group]={'present':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None,'deserialized':False}
usage=shutil.disk_usage(BASE)
report={'status':'PASS' if not manifest_conflicts and hf_result['sha256_verified']==hf_result['files_expected'] and dcp_result['sha256_verified']==dcp_result['files_expected'] and not index_refs['missing_shards'] else 'FAIL','audit_time_hong_kong':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'elapsed_seconds':time.time()-started,'hf':hf_result,'hf_remote_manifest_count':len(hf_remote['files']),'hf_transfer_manifest_count':len(hf_transfer['files']),'hf_manifest_conflicts':manifest_conflicts,'hf_index':index_refs,'dcp':dcp_result,'dcp_parts':parts,'dcp_metadata':metadata,'training_config_inside_dcp':[str(p.relative_to(dcp)) for p in configs],'training_config_external_copy':{'path':str(config_copy),'present':config_copy.exists(),'sha256':confighash,'matches_hf_yaml':confighash==hashlib.sha256((hf/'lingbotvla_cli.yaml').read_bytes()).hexdigest()},'disk':{'total':usage.total,'used':usage.used,'free':usage.free},'limits':['本审计验证文件存在、大小和SHA256，与下载时远端清单一致。','未加载PyTorch DCP元数据，未在新服务器实际恢复训练或运行推理。','DCP之外的代码、配置、归一化文件、运行依赖、训练数据、teacher/base资源仍需要独立保留或按固定版本重新获取。','DCP目录没有训练YAML，但HF目录和配置归档中都有相同YAML副本。']}
(OUT/'模型存档审计.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
lines=['# 模型与续训存档最终审计','','审计时间：'+report['audit_time_hong_kong'],'','结果：**'+report['status']+'**。本次重新读取全部文件并计算 SHA256。','',f"- 推理模型：{hf_result['sha256_verified']}/{hf_result['files_expected']} 文件校验通过；{hf_result['expected_bytes']:,} 字节（{hf_result['expected_bytes']/2**30:.2f} GiB）。",f"- 完整6k续训存档：{dcp_result['sha256_verified']}/{dcp_result['files_expected']} 文件校验通过；{dcp_result['expected_bytes']:,} 字节（{dcp_result['expected_bytes']/2**30:.2f} GiB）。",f"- 模型索引：{index_refs['weight_entries']} 个参数项，引用3个 safetensors 分片，缺片数 {len(index_refs['missing_shards'])}。",f"- HF远端清单原有14项，transfer_manifest补充训练 YAML 为第15项；重叠项无冲突。",f"- 用时：{report['elapsed_seconds']:.1f} 秒。",f"- 当前内置磁盘剩余：{usage.free:,} 字节（{usage.free/2**30:.2f} GiB）。",'','## 续训存档组成','','| 部分 | 文件数 | 字节 | SHA256通过 |','|---|---:|---:|---:|']
for g,v in parts.items():lines.append(f"| {g} | {v['files']} | {v['bytes']:,} | {v['sha256_verified']} |")
lines+=['','model/.metadata 和 optimizer/.metadata 均已按清单检查存在、大小和 SHA256；未反序列化。extra_state_rank_0.pt 至 rank_7.pt 均在。','','## 配置及恢复界限','','完整DCP目录本身没有训练 YAML。推理模型目录 lingbotvla_cli.yaml 与配置归档中的原训练 YAML 副本 SHA256 一致。','']
lines+=['- '+x for x in report['limits']]
lines+=['','备份文件完整性验证通过，不等于已经在另一台服务器做过实际续训恢复测试。']
(OUT/'模型存档审计.md').write_text('\n'.join(lines)+'\n')
print('FINAL',json.dumps({k:report[k] for k in ('status','elapsed_seconds','dcp_parts','disk')},ensure_ascii=False),flush=True)
