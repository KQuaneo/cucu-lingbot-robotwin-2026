# cucu：代码与复现入口

这是cucu队6k Full-SFT候选模型的复现代码。正式results.json按官网下载模板生成，clean和randomized均填写真实10次/任务；完整评测日志与1000段录像在04_复现与调优说明/评测证据。

## 环境与路径

使用保存的“环境构建配方”和固定版本来源准备ROCm环境，或采用原赛事实例robotwin镜像。Python为/opt/robotwin-env/bin/python，RoboTwin源码放在/RoboTwin；src/RoboTwin包含三套实际源码及当时的修改。完整RoboTwin仿真资产、clean训练数据与base/teacher大权重需按固定来源取得，未嵌入本包。

Qwen配置与tokenizer在configs/Qwen3-VL-4B-Instruct-config-tokenizer。原路径/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/Qwen3-VL-4B-Instruct-config-tokenizer应指向此目录。其他模型、teacher、数据路径按恢复说明映射；不要用HF导出config.json替代Qwen配置。

## 推理与评测

执行bash eval.sh，默认8卡、FP32、clean+randomized各50任务，每任务10次并录像；完整100次可设EPISODES=100。TASK_CONFIG=demo_randomized可仅跑随机场景，使用新的RUN_NAME避免混入旧结果。脚本不复用本次竞赛截止时间。评测前资源和路径必须已准备好。部署固定use_length=25、不编译，MPLib、expert check、接纳expert失败信息、有效回合retry补丁；配置默认clean为seen、randomized为unseen。

## 训练

在公开clean训练数据、base/teacher权重和路径准备后执行bash train.sh；使用实际保存的YAML进行8卡6000步Full-SFT，输出新目录。不要运行历史启动记录来续训。完整恢复6001步需要另存的6k DCP及AdamW状态（未放入本推理模型提交包），详见恢复说明。

模型来自基础lingbot-vla-v2-6b；无randomized训练数据，无官方RoboTwin后训练权重初始化。源代码中的许可证与来源文件保留，三套Git commit见源码版本.json。包中的历史报告有当时快照描述，最终提交状态和评测证据优先。

训练数据清单副本在configs/robotwin_demo_clean_joint_v30.txt。按原数据布局准备完50个转换数据集后，将其放到/workspace/runtime/robotwin_demo_clean_joint_v30.txt；清单每行顺序和路径保持一致。
