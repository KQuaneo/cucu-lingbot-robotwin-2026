#!/usr/bin/env python3
"""只读 SSH 抓取结果并更新本目录报告。离线使用：python3 更新报告.py --offline。"""
import argparse, csv, json, subprocess
from pathlib import Path

def atomic_text(path, content):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(content,encoding='utf-8')
    temporary.replace(path)

REMOTE = r'''
import json,re,datetime,subprocess,statistics,importlib.metadata
from pathlib import Path
import yaml
run=Path('/workspace/runtime/outputs/full_sft_6000_clean50x10_8gpu_video')
log=Path('/workspace/runtime/outputs/logs/full_sft_8gpu_6000steps_20261002.log')
config=Path('/var/tmp/robotwin-full-sft-6k-20261002/output/lingbotvla_cli.yaml')
official=Path('/models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml')
ansi=re.compile(r'\x1b\[[0-9;]*m')
tasks=[Path(x.split()[-1]).name.removesuffix('_joint_v30') for x in Path('/workspace/runtime/robotwin_demo_clean_joint_v30.txt').read_text().splitlines() if x.strip()]
time_rows={}
for x in (run/'task_times.tsv').read_text().splitlines():
 v=x.split('\t')
 if len(v)==7:time_rows[v[1]]={'gpu':int(v[2]),'start_utc':v[3],'end_utc':v[4],'elapsed_seconds':int(v[5]),'exit_code':int(v[6])}
rows=[]
for task in tasks:
 p=run/'logs'/('eval_demo_clean__'+task+'.log')
 s=ansi.sub('',p.read_text(errors='replace')) if p.exists() else ''
 matches=list(re.finditer(r'Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%, current seed:\s*(\d+)',s))
 final=list(re.finditer(r'Final success rate:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%',s))
 episodes=[];prev=0
 for m in matches:
  n=int(m[1]);episodes.append({'index':int(m[2])-1,'seed':int(m[4]),'success':n>prev});prev=n
 done=(run/'done'/('demo_clean__'+task+'.done')).exists()
 finished=bool(final and int(final[-1][2])==10 and len(episodes)==10 and done)
 n=int(final[-1][1]) if final else (int(matches[-1][1]) if matches else 0)
 total=int(final[-1][2]) if final else (int(matches[-1][2]) if matches else 0)
 result_match=re.findall(r'Data has been saved to (.+/_result.txt)',s)
 result_path='/workspace/runtime/'+result_match[-1] if result_match else None
 # episode_info 的 plan_success/task_success 属于 expert 预检，不能用作 policy 成功率。
 prechecks=[];pinfo=run/'episode_info'/('demo_clean__'+task+'.jsonl')
 if pinfo.exists():
  for line in pinfo.read_text().splitlines():
   try:prechecks.append(json.loads(line))
   except json.JSONDecodeError:pass
 timing=[{'seed':int(m[1]),'calls':int(m[2]),'mean_seconds':float(m[3]),'total_seconds':float(m[4])} for m in re.finditer(r'\[benchmark_timing\] seed=(\d+) calls=(\d+) mean_s=([\d.]+).*?total_s=([\d.]+)',s)]
 rows.append({'task':task,'status':'completed' if finished else ('running_or_partial' if p.exists() else 'not_started'),'completed':finished,'successes':n,'policy_rollouts':total,'success_rate':n/total if total else None,'episodes':episodes,'done_marker':done,'result_path':result_path,'result_file_exists':Path(result_path).exists() if result_path else False,'log_path':str(p),'timings':timing,'task_time':time_rows.get(task),'policy_rollout_errors':s.count('Policy rollout error'),'tracebacks':s.count('Traceback (most recent call last)'),'expert_prechecks':prechecks})
s=ansi.sub('',log.read_text(errors='replace'))
train=[]
for line in s.splitlines():
 m=re.search(r' - Step (\d+)/\d+, Epoch (\d+), (.*)',line)
 if not m:continue
 vals={k:float(v) for k,v in re.findall(r'([A-Za-z_]+)\s+([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)',m[3])}
 train.append({'step':int(m[1]),'epoch':int(m[2]),**vals})
checkpoint_info=[]
for step in [5000,6000]:
 p=Path('/var/tmp/robotwin-full-sft-6k-20261002/output/checkpoints')/('global_step_'+str(step))
 files=[f for f in p.rglob('*') if f.is_file()]
 checkpoint_info.append({'step':step,'remote_path':str(p),'bytes':sum(f.stat().st_size for f in files),'file_count':len(files),'model_metadata':(p/'model/.metadata').exists(),'optimizer_metadata':(p/'optimizer/.metadata').exists()})
versions={}
for package in ['torch','triton','flash-attn','sapien','mplib','lerobot']:
 try:versions[package]=importlib.metadata.version(package)
 except importlib.metadata.PackageNotFoundError:versions[package]=None
commits={}
for name,p in [('RoboTwin','/RoboTwin'),('XPolicyLab','/RoboTwin/XPolicyLab'),('lingbot-vla-v2','/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2')]:
 try:commits[name]=subprocess.check_output(['git','-C',p,'rev-parse','HEAD'],text=True).strip()
 except:commits[name]=None
infos=[]
for task in tasks:
 p=Path('/RoboTwin/data/lerobot')/(task+'_joint_v30')/'meta/info.json'
 if p.exists():
  d=json.loads(p.read_text());infos.append({'task':task,'total_frames':d.get('total_frames'),'total_episodes':d.get('total_episodes')})
model=Path('/workspace/runtime/models/full_sft_6000')
model_files=[{'name':p.name,'bytes':p.stat().st_size} for p in model.iterdir() if p.is_file()]
index=model/'model.safetensors.index.json'
model_index=json.loads(index.read_text()) if index.exists() else {}
now=datetime.datetime.now(datetime.timezone.utc)
data={'schema_version':1,'snapshot_utc':now.isoformat(),'snapshot_hong_kong':now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'run_path':str(run),'training_log_path':str(log),'training_config_path':str(config),'training_config':yaml.safe_load(config.read_text()),'official_model_config':yaml.safe_load(official.read_text()),'tasks':rows,'training_metrics':train,'training_completed':('Reached max_steps=6000, stopping training.' in s and 'global_step_6000 successfully!' in s),'training_log_error_counts':{k:s.count(k) for k in ['Traceback (most recent call last)','OutOfMemoryError','No space left on device']},'checkpoints':checkpoint_info,'versions':versions,'commits':commits,'datasets':infos,'model_files':model_files,'model_weight_index_count':len(model_index.get('weight_map',{})),'diagnostics':json.loads((run/'diagnostics/early_failure_analysis.json').read_text()),'benchmark_command':'python experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py --gpu-count 8 --episodes 10 --task-config demo_clean --run-name full_sft_6000_clean50x10_8gpu_video --model-path /workspace/runtime/models/full_sft_6000 --base-port 13420 --expert-check --accept-expert-info-on-failure --video'}
print(json.dumps(data,ensure_ascii=False))
'''

def create_report(data,out):
    allrows=data['tasks'];finished=[x for x in allrows if x['completed']]
    successes=sum(x['successes'] for x in finished);rollouts=sum(x['policy_rollouts'] for x in finished)
    complete=len(finished)==len(allrows)==50
    rate=successes/rollouts if rollouts else 0
    train=data['training_metrics'];tail=train[-100:]
    means={k:sum(x[k] for x in tail)/len(tail) for k in ['Loss','VLA_Loss','Depth_Loss','Future_Depth_Loss','FutureVideo_Loss','GradNorm','StepTime'] if tail and all(k in x for x in tail)}
    cfg=data['training_config'];t=cfg['train'];a=t['align_params'];oa=data['official_model_config']['train']['align_params']
    differences=[{'parameter':'depth.block_future_depth_to_action','ours':a['depth']['block_future_depth_to_action'],'official':oa['depth']['block_future_depth_to_action']},{'parameter':'video.block_suffix_to_future_video','ours':a['video']['block_suffix_to_future_video'],'official':oa['video']['block_suffix_to_future_video']},{'parameter':'video.use_cosine_loss','ours':a['video']['use_cosine_loss'],'official':oa['video']['use_cosine_loss']}]
    summary={'schema_version':1,'snapshot_utc':data['snapshot_utc'],'snapshot_hong_kong':data['snapshot_hong_kong'],'evaluation_status':'final' if complete else 'partial','planned_tasks':50,'planned_episodes_per_task':10,'completed_tasks':len(finished),'completed_task_successes':successes,'completed_task_rollouts':rollouts,'completed_task_success_rate':rate,'evaluation_conditions':{'checkpoint_step':6000,'precision':'BF16','task_config':'demo_clean','instruction_type':'seen','expert_check':True,'accept_expert_info_on_failure':True,'valid_episode_retry_patch':True,'eval_batch':False,'use_length':25,'training_chunk_size':50,'seed_argument':0,'actual_seeds_in_task_records':True,'video':True},'training':{'completed':data['training_completed'],'last_step':train[-1] if train else None,'last_100_steps_mean':means,'config':cfg,'checkpoints':data['checkpoints'],'dataset_frames':sum(x['total_frames'] or 0 for x in data['datasets']),'dataset_episodes':sum(x['total_episodes'] or 0 for x in data['datasets']),'teacher_alignment_differences_from_published_model_config':differences,'error_counts':data['training_log_error_counts']},'historical_adjust_bottle_tests':[{'model':'10k LoRA','successes':0,'rollouts':10,'precision':'BF16'},{'model':'official RoboTwin posttrained checkpoint, local test','successes':10,'rollouts':10,'precision':'BF16'},{'model':'6k Full-SFT dedicated initial test','successes':8,'rollouts':10,'precision':'BF16'},{'model':'6k Full-SFT clean50 repeated test','successes':next((x['successes'] for x in finished if x['task']=='adjust_bottle'),None),'rollouts':10,'precision':'BF16'}],'official_published_results':{'clean_success_rate':0.9352,'randomized_success_rate':0.9280,'published_validation_precision':'FP32','directly_comparable':False,'sources':['https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin','https://github.com/Robbyant/lingbot-vla-v2#robotwin-deployment']},'versions':data['versions'],'commits':data['commits'],'tasks':allrows,'failure_analysis':data['diagnostics']}
    atomic_text(out/'实验结果.json',json.dumps(summary,ensure_ascii=False,indent=2))
    fields=['task','status','successes','policy_rollouts','success_rate_percent','actual_seeds','failed_seeds','elapsed_seconds','gpu','policy_rollout_errors','tracebacks','remote_result_file','remote_log_file']
    with (out/'per_task_results.csv.tmp').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for x in allrows:
            timing=x['task_time'] or {};writer.writerow({'task':x['task'],'status':x['status'],'successes':x['successes'],'policy_rollouts':x['policy_rollouts'],'success_rate_percent':round(x['success_rate']*100,2) if x['success_rate'] is not None else '', 'actual_seeds':';'.join(str(e['seed']) for e in x['episodes']),'failed_seeds':';'.join(str(e['seed']) for e in x['episodes'] if not e['success']),'elapsed_seconds':timing.get('elapsed_seconds',''),'gpu':timing.get('gpu',''),'policy_rollout_errors':x['policy_rollout_errors'],'tracebacks':x['tracebacks'],'remote_result_file':x['result_path'] or '', 'remote_log_file':x['log_path']})
    (out/'per_task_results.csv.tmp').replace(out/'per_task_results.csv')
    def tf(x):return '开启' if x else '关闭'
    table='\n'.join('| '+x['task']+' | '+('完成' if x['completed'] else ('进行中/部分' if x['status']=='running_or_partial' else '尚未开始'))+' | '+str(x['successes'])+'/'+str(x['policy_rollouts'])+' | '+(f"{x['success_rate']*100:.1f}%" if x['success_rate'] is not None else '—')+' |' for x in allrows)
    difftable='\n'.join(f"| `{x['parameter']}` | `{str(x['ours']).lower()}` | `{str(x['official']).lower()}` |" for x in differences)
    dataset_frames=summary['training']['dataset_frames'];dataset_episodes=summary['training']['dataset_episodes']
    checks='\n'.join(f"| {x['step']} | {x['bytes']/1024**3:.2f} GiB | {x['file_count']} | {'存在' if x['model_metadata'] else '缺失'} | {'存在' if x['optimizer_metadata'] else '缺失'} |" for x in data['checkpoints'])
    report=f'''# LingBot-VLA-v2：6k Full-SFT 实验与评测报告（{'最终结果' if complete else '阶段快照'}）

快照时间：**{data['snapshot_hong_kong']}（香港时间 UTC+8）**。本报告从服务器只读采集。若全任务尚未完成，以下整体数值仅统计已完整完成的任务，不包含部分任务，也不代表最终全任务成绩。

## 1. 核心结果

- 正式 Full-SFT：完成 **6000 步**，日志确认到达 max_steps 并保存最终 DCP checkpoint。
- 全任务评测计划：50 个 `demo_clean` 任务，每任务 10 个有效 policy episode，BF16 推理、seen 指令。
- 快照进度：**{len(finished)}/50 个任务完成**，这些任务共成功 **{successes}/{rollouts}（{rate*100:.2f}%）**。
- 状态：**{'50 个任务的最终结果' if complete else '部分结果，剩余任务仍须刷新'}**。
- 5k 未做评测，用户选择保留存档并将评测资源用于 6k；不能据此判断 5k 与 6k 优劣。

## 2. 训练配方与复现信息

| 项目 | 本次实际设置 |
|---|---|
| 初始模型 | 基础 `robbyant/lingbot-vla-v2-6b`，不是官方 RoboTwin 后训练模型 |
| 模式 | Full-SFT；`use_lora=false`，`train_expert_only=false` |
| 视觉编码器 | `freeze_vision_encoder=false`，`freeze_vit=false`，可训练 |
| GPU | 8 × AMD Radeon PRO W7900D，48 GiB/卡 |
| 优化器 | AdamW，weight_decay=0 |
| 批量 | micro batch 16；global batch 256；梯度累积 2 |
| 并行与显存 | FSDP2；replicate 1 / shard 8；full shard；gradient checkpointing |
| 训练精度 | mixed precision=true，enable_fp32=false |
| 学习率 | 1e-4 → 5e-5，cosine，6000 步；warmup ratio=0 |
| 视觉学习率 | 1e-6；MoE expert 学习率由代码自动缩放 |
| 最大梯度范数 | 1.0 |
| 种子 | 42 |
| 数据 | 50 个 demo_clean 转换数据集；{dataset_episodes:,} 条轨迹，{dataset_frames:,} 帧 |
| 图像/动作 | 三相机；256 像素；chunk_size=50；模型动作维度上限55，机器人关节动作14维 |
| 保存策略 | 只保存 global_step_5000 与 global_step_6000 |
| 运行时间 | 训练日志进度计时约 31 小时 57 分；不含导出和评测 |

累计抽取样本量约为 6000 × 256 = **1,536,000**。这是批次样本数，不能直接当作不重复数据量。最后一个日志样本总 loss 为 {train[-1]['Loss']:.4f}，动作 VLA loss 为 {train[-1]['VLA_Loss']:.4f}；最近100步总 loss 均值 {means.get('Loss',0):.4f}，动作 loss 均值 {means.get('VLA_Loss',0):.4f}。下降的 loss 说明训练目标得到优化，不能代替闭环成功率。

### Teacher 与对齐目标

本次 `TEACHER_MODE=full`，`post_training=true`、`data.use_future_image=true`。当前深度与未来深度目标开启；`depth.use_future_depth=true`；`align_params.use_future_video=true`；视频当前 patch 目标 `video.use_current_patch_loss=true`。深度、未来深度和未来视频 loss 权重均为 **0.004**，视频 MSE 开启，CLS/cosine loss 关闭。日志中的 `Depth_Loss`、`Future_Depth_Loss`、`FutureVideo_Loss` 均有非零值。

Teacher：MoGe `moge-2-vitb-normal/model.pt`，MoRGBD `robbyant_lingbot-vla-v2-6b/depth/model.pt`，视频 `dino_video/teacher_step_10000.pth` 及其 `config.yaml`。这些 teacher 用于训练，推理通常不需要加载它们。

**本次不是官方配方的逐项等价复现。** 与官方发布 RoboTwin 模型随附的 `lingbotvla_cli.yaml` 对照，以下对齐开关不同：

| 参数 | 本次 | 官方发布配置 |
|---|---|---|
{difftable}

此外官方发布配置是 50k 步、global batch 1024、micro batch 32、MuON、32 路数据 shard；本次是 6k、256、16、AdamW、8 shard。官方配置的 `enable_fp32=true`、full shard 与 gradient checkpointing 关闭，本次亦不同。开关名说明本次对 future depth → action、suffix → future video 的注意力联系施加阻断；不能仅用“full teacher”认定与官方所有细节相同。原始配置完整保存在本备份的“配置与代码”目录及报告 JSON 快照中。

## 3. 存档、导出与本机备份

| DCP 存档 | 远程大小 | 文件数 | model/.metadata | optimizer/.metadata |
|---|---:|---:|---|---|
{checks}

远程 DCP 路径：`/var/tmp/robotwin-full-sft-6k-20261002/output/checkpoints/global_step_5000` 和 `global_step_6000`。之前的检查确认保存链路完整；本报告再次确认大小与 metadata 存在。**这不等于这两个约71 GiB的训练存档已经下载到本机。** 本机实际备份范围与校验状态以“传输”目录的清单为准。

6k 已导出为推理 HF/safetensors（约12 GiB），服务器可读取目录 `/workspace/runtime/models/full_sft_6000`。导出后的模型已用于本次评测。推理冒烟成功：加载/健康检查正常，单条输出 `[25,14]`、batch 2 `[2,25,14]`，没有 NaN/Inf。模型的本机目标目录为 `模型/full_sft_6000`，是否全部下载并校验完成请查传输记录。

恢复推理需保留：

1. 全部 safetensors 分片与 `model.safetensors.index.json`；模型配置文件及 `lingbotvla_cli.yaml`。
2. Qwen3-VL tokenizer/processor 配置，Robotwin normalization stats 与机器人配置。
3. 这次使用的 LingBot/RoboTwin/XPolicyLab 源码版本、部署与评测脚本、有效 episode retry patch 与原脚本备份。
4. 运行环境/依赖信息。新机器上的路径需改到当地目录；Radeon 运行还需 AITER/FlashAttention AMD 的环境变量。
5. 闭环仿真还需 RoboTwin assets；只有导出权重不能独立运行完整仿真。

继续训练则还需要完整 DCP 的模型、优化器和相关训练状态，以及基础模型/teacher/训练数据等。**仅约12 GiB推理权重不能等价恢复原优化器训练状态。** `/var/tmp` 和当前实例的 `/workspace` 均不能代替本机或外部持久备份，实例回收前必须完成转存。

## 4. 评测条件与结果口径

- 8 个推理服务并行；当前 BF16，`use_length=25`，`use_compile=false`。
- `demo_clean`、seen 指令、每任务10个实际 policy rollout。
- `expert_check=true`，`accept_expert_info_on_failure=true`；后者允许利用 expert 失败时已生成的任务信息，这与严格仅接受 expert 成功场景的口径可能不同。
- 保留 valid-episode retry patch：`UnStableError` 场景初始化失败不占 policy episode 配额；继续换seed，直到实际完成10个episode。
- runner seed 参数是0，评测代码生成的实际seed从100000附近开始，可能跳过初始化失败seed；每个任务的实际seed和失败seed已写入 CSV/JSON。
- 结果来自日志 `Final success rate` 与逐episode `Success rate` 记录，辅以done标记。`episode_info.jsonl` 中 `plan_success`/`task_success` 是 expert 预检字段，**不能据其false值判断模型失败**。
- 每任务只有10个episode，结果会受抽样、随机噪声、数值精度与仿真轨迹影响。若需要严谨比较模型，应统一精度、环境和更多episode。

| 任务 | 状态 | 成功/实际rollout | 成功率 |
|---|---|---:|---:|
{table}

机器可读逐episode结果、实际seed、耗时、推理异常计数等请见 `实验结果.json`；表格软件可打开 `per_task_results.csv`（UTF-8 BOM）。进行中任务的当前值不计入上述已完成任务汇总。

## 5. 历史单任务结果与官方成绩

| 测试 | adjust_bottle |
|---|---:|
| 10k LoRA，当前环境 BF16 | 0/10 |
| 官方 RoboTwin 后训练模型，本机 BF16单任务对照 | 10/10 |
| 6k Full-SFT，最初专项测试 | 8/10 |
| 同一6k权重，本轮全任务中的重复测试 | {next((str(x['successes'])+'/10' for x in finished if x['task']=='adjust_bottle'),'尚未完成')} |

专项8/10与重复测试9/10属于两次运行；不要覆盖成一个成绩，也不能将一次提升直接归因于训练变化，两次使用同一6k权重。小样本闭环结果存在波动，评测重复与并行环境亦可影响轨迹。

官方发布平均成功率为 **Clean 93.52%、Randomized 92.80%**。[官方模型卡](https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin) 官方发布验证使用 **FP32**，本次用 BF16；模型训练配方、后训练时长、硬件/规划器、每任务样本数和场景接受条件也不同，因此数字不能直接当作同条件模型差距。[官方部署说明](https://github.com/Robbyant/lingbot-vla-v2#robotwin-deployment)

本轮只评测 clean；没有完成本模型 randomized 全任务结果，也没有进行本机官方模型全任务对照。

## 6. 失败案例：初步录像分析

### adjust_bottle，seed 100002，episode 1

初始接触似乎将瓶子移到了桌后方，未建立稳定抓取；之后机械臂仍在原区域运动，没有重新定位并抓起瓶子，最终400步失败。更像抓取与恢复失败，不能仅解释为最终位置阈值偏差。

### click_alarmclock，seed 100000，episode 0

机械臂接近并闭合夹爪，但多次悬停和微调，没有在采样画面中观察到持续建立所需按钮接触；最终400步失败。更像接触位置问题。

这两项分析来自单个头部相机、8张采样帧，不能确认3D接触距离、力或帧间短暂接触。它们是观察与推断，不能作为所有失败的统一原因。“结果/full_sft_6000_clean50x10_8gpu_video/diagnostics”保留原始结构化分析，录像/失败拼图可用于复查。

## 7. 后续建议

1. 先保证6k推理模型与本轮结果完整保存在本机并通过校验，再考虑训练优化。
2. 待本轮完成后，将50任务结果按失败类型分类，优先选择低分任务录像复查。
3. 下一轮公平对照可用官方权重、相同评测口径和相同精度；时间有限时先做低分任务对照。
4. 若继续训练10k/30k，需重新规划batch、学习率日程、teacher开关和正式保存位置；当前模型与结果是6k基线，不是30k效果预测。

## 8. 源码与环境

| 仓库 | commit |
|---|---|
{chr(10).join('| '+k+' | `'+str(v)+'` |' for k,v in data['commits'].items())}

依赖版本：{json.dumps(data['versions'],ensure_ascii=False)}。AITER flags：`AITER_TRITON_ONLY=1`、`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`。原始环境清单及具体启动脚本请见“配置与代码”。

## 9. 更新与查阅

`原始快照.json` 保存本报告的数据依据。`更新报告.py` 只读抓取远程结果，可在实例可访问时重新运行；`python3 更新报告.py --offline` 使用既有快照重新生成。它不启动/停止评测，不修改远程文件。离线可查“模型、日志、配置与代码、结果、录像、传输”目录。只有传输清单和hash校验成功的文件才能确认已经保留到本机。
'''
    atomic_text(out/'实验报告.md',report)
    print(json.dumps({'completed_tasks':len(finished),'successes':successes,'rollouts':rollouts,'rate':rate,'status':summary['evaluation_status'],'snapshot':data['snapshot_hong_kong'],'files':['实验报告.md','per_task_results.csv','实验结果.json','原始快照.json','更新报告.py']},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--offline',action='store_true');parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    raw=args.out/'原始快照.json'
    if args.offline:data=json.loads(raw.read_text(encoding='utf-8'))
    else:
        p=subprocess.run(['ssh','-o','BatchMode=yes','-C','-o','ConnectTimeout=12','-p','31227','root@YOUR_SERVER_IP','/opt/robotwin-env/bin/python -'],input=REMOTE,text=True,capture_output=True,check=True)
        data=json.loads(p.stdout);atomic_text(raw,json.dumps(data,ensure_ascii=False,indent=2))
    create_report(data,args.out)
