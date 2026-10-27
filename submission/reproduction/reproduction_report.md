# 复现与调优说明

## 1. 基础信息

- 团队名称：cucu；团队ID：1382038。
- 使用模型版本：robbyant/lingbot-vla-v2-6b，固定修订11c703bf6a5c1f45b3b69168482da11fdbba53d7，由基础模型Full-SFT获得本候选；没有使用官方RoboTwin后训练权重初始化。
- 代码版本 / Commit：RoboTwin 266f3aadf505a4f7fe9af0faa41a20f5f47cd123；XPolicyLab c37109c500be67d0dea6b36bf7337bbd26e763cd；LingBot-VLA-v2 951475ae1b1d87553e7dc47c97b53a3d695c0d13。实际源码与兼容改动在02_代码材料/src，环境与关键源码一致性证据另存。
- 提交checkpoint名称：full_sft_6000，HF推理权重15文件、3个safetensors分片，位于03_模型权重/checkpoint。对应训练为6000步，非10k或30k模型。

## 2. 训练复现

- 使用数据集：RoboTwin 2.0 Aloha-AgileX 50项demo_clean，每项50条轨迹，共2500条、549787帧。50项数据清单在02_代码材料/configs/robotwin_demo_clean_joint_v30.txt，norm stats及robot config随代码提供。
- 是否使用randomized数据训练：否。
- 一键训练脚本：环境与公共数据/teacher路径准备完成后，执行`bash 02_代码材料/train.sh`；输出新目录，不会自动加载6k续训存档。
- 训练配置文件：02_代码材料/configs/实际训练配置.yaml；robot config：configs/robot_configs与源码相应目录。
- 训练参数：8×AMD Radeon PRO W7900D、Full-SFT、AdamW、weight_decay0，micro batch16、global batch256、累积2，FSDP2 full shard、gradient checkpointing，视觉编码器参与训练。6000步，seed42，warmup0，cosine主学习率1e-4至5e-5，视觉学习率1e-6，训练混合精度。
- 对齐：full teacher，MoGe、MoRGBD与DINO video；当前/未来深度和未来视频目标开启，对应loss权重0.004。teacher大权重与公共仿真资产不嵌入提交包，固定来源与路径见恢复资源清单。
- 训练证据：日志确认6000步正常结束，墙钟约31小时57分；末步总loss0.0283、VLA loss0.0243，末100步均值0.0332/0.0291。日志、曲线、指标CSV在训练证据目录。loss下降不能代替闭环成功率。

## 3. 评测复现

- 一键评测脚本：`bash 02_代码材料/eval.sh`；默认FP32、两种setting、每任务10次有效回合。设置EPISODES=100可运行100次，需另行安排算力时间。
- checkpoint加载：eval.sh默认加载03_模型权重/checkpoint，训练结构从checkpoint中的lingbotvla_cli.yaml读取，Qwen配置/tokenizer、Robotwin归一化与机器人配置配套使用。FP32审计检查所有浮点参数和buffers；8个服务各记录1672个float32参数项与147个float32 buffer项。将已保存权重转为FP32不会恢复保存时的舍入精度。
- clean命令：`TASK_CONFIG=demo_clean EPISODES=10 RUN_NAME=cucu_clean_fp32 bash 02_代码材料/eval.sh`。
- randomized命令：`TASK_CONFIG=demo_randomized EPISODES=10 RUN_NAME=cucu_randomized_fp32 bash 02_代码材料/eval.sh`。
- 评测环境：ROCm7.2.1、PyTorch2.9.1、Triton3.5.1、FlashAttention2.8.4、SAPIEN3.0.3、MPLib0.2.1；完整包版本与差异见环境证据。设置AITER_TRITON_ONLY=1、FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE、ROBOTWIN_DISABLE_CUROBO=1、ROBOTWIN_EE_PLANNER=mplib、PYOPENGL_PLATFORM=egl。
- 条件：8个FP32推理服务，use_length25、eager、不编译、eval_batch=false、expert_check=true、accept_expert_info_on_failure=true，保留valid-episode retry及录像。clean使用seen指令，randomized使用unseen指令；随机化开启桌面杂物、背景、光照与桌高。seed参数0，实际种子从100000附近开始，可因初始化失败跳过，逐回合保留实际seed。
- clean已完成结果：50任务×10回合，263/500成功（52.60%），exit0；500录像已保存核验。另一次BF16诊断276/500单独留档，不混入FP32次数。
- randomized结果：50任务、500个有效policy回合，成功177/500（35.40%），exit code 0，500段录像SHA256校验通过。
- 本队实际自测规模为每任务10次，按官方JSON模板填写真实attempts、successes，不宣称完成100次。每任务小样本结果有波动；指令与策略采样随机性未完全固定，不能仅以BF16/FP32分数差异作精度因果归因。本地ROCm/MPLib、长度25等条件也与官方发布环境不同。

## 4. 代码改动说明

- 是否修改LingBot-VLA 2.0官方代码：是。deploy/lingbot_vla_v2_policy.py适配训练配置发现、Qwen路径、变长batch与ROCm推理，禁用单形状ViT缓存；数据加载、参数与LoRA辅助、训练入口的兼容修改保留在实际源码中。主要文件见环境证据的修改文件与SHA清单。
- 是否修改RoboTwin官方评测代码：是。scripts/eval_policy_xpolicylab.py增加/保留外部策略协议、expert预检信息处理、记录与有效回合retry。UnStableError初始化失败不算policy回合；修正为无条件回退该回合配额，换seed补足10个实际policy回合。原脚本备份及修改后文件在代码材料和原始配置证据中保留。
- envs/_base_task.py、envs/robot/planner.py、envs/robot/robot.py保留ROCm/MPLib适配；XPolicyLab/scripts/transform_lerobot_v30_format.py适配数据转换。MPLib替代CUDA专用CuRobo，因此明确记录规划器差异。
- 未改变任务成功判定来提高分数；结果来自真实policy累计成功和最终日志，episode_info里的plan_success/task_success为expert预检，不当作policy结果。指令/随机化与样本限制均如实报告。

## 5. 调优说明

- 训练参数调整：受算力时间限制，选择8卡AdamW full-shard、micro16/global256及checkpointing，完成6000步Full-SFT；与官方50k步、global1024、MuON配方不同。
- 数据处理方式：仅clean50×50轨迹，LeRobot v3转换、三相机256像素、训练chunk_size50、机器人关节动作14维，使用随包Robotwin归一化与配置。训练数据的视频/parquet不在本包，固定来源和转换配方另有说明。
- 模型结构改动：没有声明新的网络架构贡献。训练alignment head只用于训练目标，DCP导出时排除训练专用head，其余推理权重严格匹配。
- 对齐开关差异：本次depth.block_future_depth_to_action=true、video.block_suffix_to_future_video=true、video.use_cosine_loss=false，官方对应false、false、true；完整teacher不代表逐项等价复现官方配方。
- 推理策略：FP32、动作执行长度25、不编译，单卡一个服务，通过8卡动态队列覆盖50任务，完整保留日志与录像。
- 主要提升点与限制：得到可恢复、可审计的6k Full-SFT候选及闭环基线；没有同条件的基础模型全任务对照，不能量化训练净提升。低分任务及少量录像提示抓取、接触、失败后恢复值得验证，尚未证明统一原因。后续需控制训练量、优化器、学习率日程和对齐开关分别比较。

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-26
- Verification Status: VERIFIED (execution/log/transfer consistency)
- Version Label: cucu-6k-submission
