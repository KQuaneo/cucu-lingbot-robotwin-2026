# coding: utf-8
from pathlib import Path
import shutil,subprocess,tempfile,os,json,hashlib,datetime
base=Path(__file__).resolve().parents[3]
out=base/'提交/cucu_初赛提交材料'
out.mkdir(parents=True,exist_ok=True)
git='/Users/LOCAL_USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git'
repo_material=base/'配置与代码/补充交接_20261004/解包/repositories'
src=out/'02_代码材料/src/RoboTwin'
state=out/'提交状态.json'
if state.exists():previous=json.loads(state.read_text())
else:previous={}
repo_versions={'RoboTwin':'266f3aadf505a4f7fe9af0faa41a20f5f47cd123','XPolicyLab':'c37109c500be67d0dea6b36bf7337bbd26e763cd','lingbot-vla-v2':'951475ae1b1d87553e7dc47c97b53a3d695c0d13'}
targets={'RoboTwin':src,'XPolicyLab':src/'XPolicyLab','lingbot-vla-v2':src/'experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2'}
env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
if not previous.get('source_snapshots_prepared'):
    for name,target in targets.items():
        temp=Path(tempfile.mkdtemp(prefix='cucu-source-'))/'repo'
        subprocess.run([git,'clone','--quiet','--no-checkout',str(repo_material/(name+'.bundle')),str(temp)],check=True,env=env)
        subprocess.run([git,'-C',str(temp),'-c','filter.lfs.process=','-c','filter.lfs.smudge=','-c','filter.lfs.required=false','checkout','--quiet','--detach',repo_versions[name]],check=True,env=env)
        patch=repo_material/(name+'.working.patch')
        if patch.stat().st_size:subprocess.run([git,'-C',str(temp),'apply',str(patch)],check=True,env=env)
        shutil.copytree(temp,target,dirs_exist_ok=True,symlinks=True,ignore=shutil.ignore_patterns('.git','.DS_Store'))
        shutil.rmtree(temp.parent)
    for original in [base/'配置与代码/原路径/RoboTwin',base/'配置与代码/补充交接_20261004/解包/original_paths/RoboTwin']:
        shutil.copytree(original,src,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.DS_Store','__pycache__'))
def copy(source,dest):
    dest=out/dest;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
copy(base/'模型/full_sft_6000/lingbotvla_cli.yaml','02_代码材料/configs/实际训练配置.yaml')
copy(base/'日志/full_sft_8gpu_6000steps_20261002.log','04_复现与调优说明/训练证据/6000步训练日志.log')
copy(base/'日志/training_metrics_6000steps.csv','04_复现与调优说明/训练证据/训练指标.csv')
copy(base/'日志/loss_curve_6000steps.png','04_复现与调优说明/训练证据/loss曲线.png')
copy(base/'报告/实验报告.md','04_复现与调优说明/历史报告/BF16_实验报告.md')
copy(base/'报告/FP32全任务/评测报告.md','04_复现与调优说明/历史报告/FP32_clean_评测报告.md')
copy(base/'报告/FP32全任务/结果.json','01_评测结果/原始证据/clean_fp32_结果.json')
copy(base/'报告/FP32全任务/per_task_fp32_vs_bf16.csv','01_评测结果/原始证据/clean_fp32_vs_bf16.csv')
copy(base/'报告/最终审计/恢复说明.md','04_复现与调优说明/恢复说明.md')
copy(base/'报告/最终审计/恢复资源清单.json','04_复现与调优说明/恢复资源清单.json')
for name in ['serve_fp32.py','launch_fp32.sh','run_fp32_benchmark.py']:
    text=Path(__file__).parent.joinpath(name).read_text()
    text=text.replace("Path(\"/workspace/runtime/experiments/randomized_fp32_20261026/launch_fp32.sh\")","Path(__file__).resolve().parent / 'launch_fp32.sh'")
    text=text.replace('/workspace/runtime/experiments/randomized_fp32_20261026/serve_fp32.py','"$script_dir/serve_fp32.py"')
    text=text.replace("deadline = datetime.fromisoformat('2026-10-26T23:15:00+08:00')","deadline = utc_now() + __import__('datetime').timedelta(hours=72)")
    text=text.replace('for line in prior.read_text().splitlines():',"for line in (prior.read_text().splitlines() if prior.exists() else []):")
    dest=out/'02_代码材料/scripts'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
# Preserve the original training launcher as evidence; the reproducible launcher uses the actual YAML.
copy(base/'配置与代码/原路径/var/tmp/robotwin-full-sft-6k-20261002/run.sh','02_代码材料/scripts/历史训练启动记录.sh')
qwen=base/'配置与代码/补充交接_20261004/解包/original_paths/models/robotwin-persistent/models/Qwen3-VL-4B-Instruct-config-tokenizer'
shutil.copytree(qwen,out/'02_代码材料/configs/Qwen3-VL-4B-Instruct-config-tokenizer',dirs_exist_ok=True)
recipe=base/'配置与代码/补充交接_20261004/Robotwin-radeon-cloud'
shutil.copytree(recipe,out/'02_代码材料/环境构建配方',dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','.DS_Store'))
model=out/'03_模型权重/checkpoint';model.mkdir(parents=True,exist_ok=True)
expected=json.loads(Path(__file__).parent.joinpath('expected_model_manifest.json').read_text())['files']
for row in expected:
    original=base/'模型/full_sft_6000'/row['file'];target=model/row['file']
    assert original.stat().st_size==row['bytes']
    if not target.exists():
        try:os.link(original,target)
        except OSError:shutil.copy2(original,target)
(model/'文件校验清单.json').write_text(json.dumps(expected,ensure_ascii=False,indent=2)+'\n')
train='''#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export AITER_TRITON_ONLY=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export TMPDIR=${TMPDIR:-/var/tmp/cucu-training-cache}
mkdir -p "$TMPDIR"
cd /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
exec /opt/robotwin-env/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 -m tasks.vla.train_lingbotvla "$here/configs/实际训练配置.yaml" --train.output_dir "${OUTPUT_DIR:-/workspace/runtime/outputs/cucu_full_sft_6000_reproduce}" --train.enable_resume false --train.max_steps 6000
'''
(out/'02_代码材料/train.sh').write_text(train)
evaluation='''#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model=${1:-$here/../03_模型权重/checkpoint}
model=$(cd -- "$model" && pwd)
export LINGBOTVLA_TRAINING_CONFIG="$model/lingbotvla_cli.yaml"
export AITER_TRITON_ONLY=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTHONUNBUFFERED=1
export TMPDIR=${TMPDIR:-/var/tmp/cucu-eval-cache}
mkdir -p "$TMPDIR"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
exec /opt/robotwin-env/bin/python "$here/scripts/run_fp32_benchmark.py" --gpu-count 8 --episodes "${EPISODES:-10}" --task-config "${TASK_CONFIG:-both}" --run-name "${RUN_NAME:-cucu_6k_fp32_reproduce}" --model-path "$model" --base-port "${BASE_PORT:-13820}" --expert-check --accept-expert-info-on-failure --video
'''
(out/'02_代码材料/eval.sh').write_text(evaluation)
readme='''# cucu：代码与复现入口\n\n这是6k Full-SFT候选模型的材料准备目录。官方results.json模板尚待已报名账号下载，随机场景评测仍在进行；请以提交状态.json为准。\n\n## 环境与路径\n\n使用保存的“环境构建配方”和固定版本来源准备ROCm环境，或采用原赛事实例robotwin镜像。Python为/opt/robotwin-env/bin/python，RoboTwin源码放在/RoboTwin；src/RoboTwin包含三套实际源码及当时的修改。完整RoboTwin仿真资产、clean训练数据与base/teacher大权重需按固定来源取得，未嵌入本包。\n\nQwen配置与tokenizer在configs/Qwen3-VL-4B-Instruct-config-tokenizer。原路径/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/Qwen3-VL-4B-Instruct-config-tokenizer应指向此目录。其他模型、teacher、数据路径按恢复说明映射；不要用HF导出config.json替代Qwen配置。\n\n## 推理与评测\n\n执行bash eval.sh，默认8卡、FP32、clean+randomized各50任务，每任务10次并录像；完整100次可设EPISODES=100。TASK_CONFIG=demo_randomized可仅跑随机场景，使用新的RUN_NAME避免混入旧结果。脚本不复用本次竞赛截止时间。评测前资源和路径必须已准备好。部署固定use_length=25、不编译，MPLib、expert check、接纳expert失败信息、有效回合retry补丁；配置默认clean为seen、randomized为unseen。\n\n## 训练\n\n在公开clean训练数据、base/teacher权重和路径准备后执行bash train.sh；使用实际保存的YAML进行8卡6000步Full-SFT，输出新目录。不要运行历史启动记录来续训。完整恢复6001步需要另存的6k DCP及AdamW状态（未放入本推理模型提交包），详见恢复说明。\n\n模型来自基础lingbot-vla-v2-6b；无randomized训练数据，无官方RoboTwin后训练权重初始化。源代码中的许可证与来源文件保留，三套Git commit见源码版本.json。包中的历史报告有当时快照描述，最终提交状态和评测证据优先。\n'''
(out/'02_代码材料/README.md').write_text(readme)
(out/'02_代码材料/源码版本.json').write_text(json.dumps(repo_versions,indent=2)+'\n')
status={'team':'cucu','prepared_hong_kong':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'source_snapshots_prepared':True,'model_files_staged':len(expected),'model_bytes':sum(r['bytes'] for r in expected),'checkpoint_step':6000,'clean_result':{'precision':'FP32','tasks':50,'trials':500,'successes':263},'randomized_result':'running; actual snapshot in report directory','official_template_obtained':False,'team_captain_authenticated':False,'submission_uploaded':False,'final_zip_created':False,'required_before_submission':['Obtain official results/report templates from registered captain account','Finish and verify randomized evaluation','Fill official template with actual trial counts, not fabricated 100','Verify final package and submit via authenticated Tianchi entry; retain receipt']}
state.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(status,ensure_ascii=False))
