from pathlib import Path
import json,hashlib,zipfile,datetime
b=Path(__file__).resolve().parent;r=b/'remote';status=json.loads((r/'status.json').read_text());audit=json.loads((r/'final_audit.json').read_text());assert status['status']=='COMPLETED' and (r/'exit_code').read_text().strip()=='0';results=status['task_results'];assert len(results)==3
manifest=json.loads((b/'transfer_manifest.json').read_text())
for row in manifest:
 p=b/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
videos=list((b/'videos').rglob('*.mp4'));assert len(videos)==audit['episodes']==72
lines=['# 多任务残差RL试验结果','',f"最终核验：{audit['verified_utc']}；exit0；全部本试验进程已退出。",'', '## 结果','', '|任务|训练成功/实际回合|独立验证：未更新残差|独立验证：更新残差|配对：改善/退步|','|---|---:|---:|---:|---:|']
for task in results:
 a=next(x for x in audit['task_audits'] if x['task']==task['task']);lines.append(f"|{task['task']}|{task['train_successes']}/{task['train_rollouts']}|{task['control_successes']}/8|{task['after_successes']}/8|{a['improved_pairs']}/{a['regressed_pairs']}|")
lines+=['', '每个任务均独立从mu=0开始，基于真实成功/失败奖励完成一次非零REINFORCE更新。control是未学习的随机残差分布(mu=0,std=1)，不是完全无残差的原6k。验证回合与训练seed分离，验证条件在更新前后按实际seed、完整指令、潜变量噪声及基础模型torch种子匹配。','', '## 审计与备份','',f"- 共{audit['episodes']}回合、{audit['chunks']}个动作块，实际执行{audit['actual_executed_steps']}步；72段录像。",f"- 本地{len(manifest)}个远端证据文件逐成员SHA核验通过，合计{sum(x['bytes'] for x in manifest):,}字节。",'- 三项任务的解析梯度均与autograd一致；更新后的worker均加载各自checkpoint参数。','- 三项任务的训练种子与验证种子不重叠，验证各8对seed/指令/噪声匹配。','- 所有终局reward与环境最终成功日志一致；夹爪保持不变，最大手臂关节偏移小于0.004001弧度。','- 原评测脚本SHA不变，6k模型冻结，未修改原权重或已提交包。']
for a in audit['task_audits']:lines.append(f"- {a['task']}：8对中初始状态完全一致{a['initial_state_exact_matches']}对、三路图像完全一致{a['initial_images_exact_matches']}对。")
lines+=['','## 解释边界','','本次只学习每任务12个全局残差均值，没有更新6B主模型、teacher或引入观测条件的网络，也不是已适配LingBot的ReinFlow/PPO。每项只做一次更新、验证仅8对，不能外推为50任务性能提升；也不能把训练成功率与独立验证成功率混在一起比较。实际收益应以表中独立配对验证为准。','', '若成功数未变，结论是本设置未观察到成功率收益，不等于证明所有RL无效。可能需要更多rollout、多次更新、适当的探索尺度和有状态条件的残差策略；这些均未在本次验证。原始首次click_bell的5/8→7/8使用不同种子，没有本次这样的配对控制，不能把它当作可靠增益。','', '## 文件','','- remote/<task>_residual_policy_checkpoint.json：每任务独立12维残差参数，须与原6k模型配套。','- remote/<task>_policy_update.json：真实梯度/优势值/均值/变化范数/KL。','- remote/<task>_paired_validation.json：逐回合对照。','- remote/trajectories/：逐chunk图像、状态、基础动作/叠加后动作、执行步数回执和真实奖励。','- videos/：72段录像。','- remote/final_audit.json、transfer_manifest.json：审计和SHA清单。','- 基础6k权重校验清单.json：基础模型引用；本包不重复放入12.77GB基础权重。','', '实验已结束，无后台RL服务。未来重跑必须改独立目录、采样种子以及控制器硬截止时间，不能直接覆盖现有证据。']
(b/'多任务RL试验结果.md').write_text('\n'.join(lines)+'\n')
out=b.parents[1]/'分享/RL多任务闭环试验_20261026.zip';out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as pack:
 for p in sorted(b.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:pack.write(p,str(p.relative_to(b.parent)))
with zipfile.ZipFile(out) as pack:assert pack.testzip() is None
receipt={'complete':True,'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'videos':len(videos),'remote_files_sha_verified':len(manifest),'zip_crc_verified':True,'task_results':results};(b/'本地备份完成.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False))
