from pathlib import Path
import json,hashlib,zipfile,shutil
b=Path(__file__).resolve().parent;r=b/'remote';s=json.loads((r/'status.json').read_text());a=json.loads((r/'final_audit.json').read_text());assert s['status']=='COMPLETED' and (r/'exit_code').read_text().strip()=='0'
manifest=json.loads((b/'transfer_manifest.json').read_text())
for x in manifest:
 p=b/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
videos=list((b/'videos').rglob('*.mp4'));assert len(videos)==a['new_validation_episodes']==24
ref=b.parent/'RL多任务试验_20261026/基础6k权重校验清单.json'
if ref.exists():shutil.copy2(ref,b/'基础6k权重校验清单.json')
lines=['# PPO残差策略对照结果','',f"完成时间：{s['finished_utc']}；exit0，全部本试验服务与采样进程已退出。",'', '## 配对验证','', '|任务|未更新随机残差|REINFORCE一次更新|PPO裁剪目标4个epoch|PPO相对control：改善/退步|','|---|---:|---:|---:|---:|']
for row in s['task_results']:
 audit=next(x for x in a['task_audits'] if x['task']==row['task']);lines.append(f"|{row['task']}|{row['control_successes']}/8|{row['reinforce_successes']}/8|{row['ppo_successes']}/8|{audit['improved_pairs']}/{audit['regressed_pairs']}|")
lines+=['','## 实际运行与验收','','- 每任务复用旧策略mu=0采集的8个真实训练回合，共24回合；没有新增训练采样，不将旧回合重复计为新采样。训练源路径和SHA在training_provenance.json。','- 三任务各独立从mu=0计算PPO clipped actor更新，clip=0.2、Adam lr=0.03、4个优化epoch；Monte Carlo终局reward减leave-one-out baseline，无value critic。','- 固定std=1，mu更新范数限制0.2，latent Gaussian KL约0.02；概率比裁剪确实触发。每回合只计一次latent probability。','- 新增24个真实验证回合，验证seed与训练分离；与旧control按实际seed、指令、潜变量epsilon、基础模型torch seed配对。','- 原6k FP32主模型冻结，只更新每任务12维残差均值；动作偏移仍为0.004*tanh(z)，夹爪未改。',f"- 新增{a['chunks']}个动作块、实际执行{a['actual_executed_steps']}步，24段录像。",f"- {len(manifest)}个远端证据文件逐文件SHA校验通过，共{sum(x['bytes'] for x in manifest):,}字节。",'- 初始梯度解析核验通过，clip正负优势自检通过，恒定奖励时零更新；旧log probability复算通过，所有实际终局reward与环境日志一致。']
for x in a['task_audits']:lines.append(f"- {x['task']}：8对seed/指令/epsilon匹配，初始状态完全一致{x['initial_state_exact_matches']}对、图像完全一致{x['initial_images_exact_matches']}对；clip fraction依次为{x['clip_fractions']}。")
lines+=['','## 解释边界','','control是未学习的随机残差策略，不是完全无残差的原6k模型。PPO与REINFORCE除了裁剪目标，也改变了优化器、学习率和优化epoch数，因此不能把差异单独归因于clip。这里只是同一小训练批次的受限策略方法对照，不是多轮在线PPO训练或6B主模型RL微调。','', '每项验证仅8对，均为已观察过的开发场景，不能当作新独立测试集或用来宣称50任务提升；需要更多预注册seed和任务复核。若成功数不变或下降，应如实报告。','', '## 保存位置','','- remote/<task>_ppo_checkpoint.json：PPO残差checkpoint（12维），需搭配原6k权重。','- remote/<task>_ppo_update.json：每个epoch loss、gradient、ratio、clip fraction、KL和mean。','- remote/<task>_reused_train_outcomes.json、training_provenance.json：更新的真实训练来源。','- remote/<task>_paired_validation.json：逐回合对照结果。','- remote/trajectories/、videos/：新验证回合完整观测、状态、动作、执行步数、reward和24录像。','- remote/final_audit.json、transfer_manifest.json：独立审计和SHA清单。','- 此前REINFORCE控制组完整轨迹在同级RL多任务试验_20261026；本包不重复嵌入旧轨迹或12.77GB基础权重。','', '原6k权重、原评测脚本及已提交包未修改，未重交比赛结果。未来重跑须改控制器硬截止日期、独立目录及端口，不能覆盖已有证据。']
(b/'PPO方法对照结果.md').write_text('\n'.join(lines)+'\n')
out=b.parents[1]/'分享/RL_PPO方法对照_20261026.zip';out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for p in sorted(b.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(b.parent)))
with zipfile.ZipFile(out) as z:assert z.testzip() is None
receipt={'complete':True,'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'new_videos':len(videos),'remote_files_sha_verified':len(manifest),'zip_crc_verified':True,'task_results':s['task_results']};(b/'本地备份完成.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));print(json.dumps(receipt,ensure_ascii=False))
