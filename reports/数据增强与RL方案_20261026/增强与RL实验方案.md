# 数据增强与RL实验准备（未执行）

按用户2026-10-26最新指令，仅准备方案，不启动训练、RL或新评测，不更新已提交包。

## 当前可用条件与事实

18:55香港时间远端仍可SSH；8张W7900D利用率和显存占用均为0；/workspace约56.6GB可用。50个clean数据集全部存在，meta/info.json合计2500条轨迹、549787帧；MoGe、MoRGBD和DINO-video三份teacher大权重存在。数据和teacher在当前镜像内，未来换实例须重新确认。

6k结果：clean263/500=52.6%，randomized177/500=35.4%，各50任务×10次。randomized还包含unseen指令，下降不能全部归因于视觉域偏移。测试规模小，不能按某任务7/10差距直接下原因结论。

## 推荐顺序：先增强对照，再接RL

已有增强入口：data.image_augment。现有6k配置为false。本地及当前远端实现仅作亮度0.7–1.3、对比度0.6–1.4、饱和度0.5–1.5扰动；同一训练样本的多视角和当前/未来图像重放相同参数。没有在本方案中加入裁剪、旋转、水平翻转或动作标签变换。

第一轮只改变image_augment，降低变量数量。默认增强强度可能过强，先检查真实训练帧预览；若需弱增强，应复制独立实验源码后修改范围，并在两组记录源码差异，不能悄悄改当前提交源文件。颜色扰动只覆盖视觉变化的一部分，不能解决语言泛化、几何变化或抓取恢复问题。

三组对照：A=原6k；B=6k权重初始化后继续相同步数、不增强；C=同样初始化和步数、开启增强。B/C固定数据次序、seed、batch、优化器、学习率和teacher；至少多seed重复后再下结论。只对比A/C不能区分额外训练和增强的作用。

已准备4个YAML：增强/对照各10步和200步。10步用于检查dataloader、增强、多teacher、反向传播和checkpoint链路；200步只是后续pilot，不等于正式有效训练量。均为8卡Full-SFT、AdamW、micro16/global256、梯度累积2、FSDP2 full shard、gradient checkpointing、full teacher、视觉不冻结；候选LR固定1e-5（lr与lr_min相同），这是保守试验值，未证明最优。

重要：这些配置从HF 6k推理权重初始化，enable_resume=false。优化器和训练专用alignment heads不会继承；它不是DCP的精确step6001续训。若要严格续训，需恢复已备份的原6k DCP及原参数并实测恢复，不能把HF初始化写成精确续训。

配置输出均在独立/var/tmp目录。当前save_hf_weights=false，训练到max_steps仍会自动保存DCP（约76.6GB量级），不是只有轻量日志；必须检查/var/tmp所在文件系统空间，不能放到只有56.6GB剩余的/workspace。要闭环推理，需另做HF导出和分片加载核验；不能假定生成DCP就能直接被现有推理服务读取。

## 时间预算

历史6000步约31小时57分，平均约19.2秒/步，含当时运行开销；仅作估计，换镜像和I/O会变化。10步纯训练约3.2分钟；50步约16分钟；200步约64分钟，均另加模型/teacher加载、首步编译、存档、HF导出、传输与评测。两组200步加可靠评测显然超出一小时。本次不执行这些操作。

下一次充足算力：先10步检查loss/grad finite和真实增强图；通过后分别做B/C各200步；导出后用固定小任务集pilot，再决定是否扩展训练和50任务×100次评测。不要把10步loss变化当作闭环提升。

## 闭环任务与验收

优先观察5项：move_playingcard_away(9→2)、open_laptop(9→2)、place_bread_skillet(8→1)、place_object_scale(8→1)、place_can_basket(7→0)，括号为当前clean/randomized各10次成功数。它们是按已观测差距挑出的开发集，存在选择偏差；另外预留未用于选参的任务作验证，并检查clean回退。

B/C比较用相同FP32、use_length25、规划器、成功判定、场景seed和指令清单；当前评测默认指令/策略随机性未完全固定，需要补充记录和回放，而非只设置一个seed就宣称严格配对。完整保存每回合动作、reward/成功、seed、指令、视频与原始日志。初始化失败仍按valid-episode规则处理；policy失败应保留。训练样本不能来自randomized测试集。

## RL接入路线（设计，非已实现）

当前LingBot源码没有PPO/GRPO/ReinFlow训练入口；现有训练是流匹配模仿学习。建议以ReinFlow为首个工程调研候选：其论文通过给确定性流采样路径引入可学习噪声，使策略概率可计算，再做在线策略优化。作者仓库提供若干VLA接入示例，但未在本次核实LingBot原生支持或ROCm兼容，因此不能直接执行仓库命令替代现有训练。

需要新增的模块：
1. RoboTwin向量环境/采样worker，独立clean训练seed与保留评测seed。
2. LingBot actor适配器：返回执行动作及采样过程所需信息，处理chunk_size50与use_length25的对应关系、14维有效动作和归一化；避免对未执行动作错误计入回报。
3. 算法要求的概率/噪声参数和value/advantage计算，old-policy快照、终止/超时bootstrap语义。
4. rollout数据缓冲、分布式更新、训练恢复、吞吐与显存监控。
5. 回归检查：冻结actor时动作/行为与原服务一致；更新前后概率比和梯度有限；reward来自真实环境成功判定，不使用expert预检成功冒充policy reward。

第一个任务选clean基线中等且回合较短的任务：先从逐任务表筛选3–7/10，再计时后确定；不选成功始终为0的长任务做首次调试。先只验证采样→reward→一次更新→新策略采样的闭环。初始主奖励只用任务成功0/1，不同时加入一堆未经验证的距离/接触奖励；需要时单独消融reward shaping。保留原SFT策略并考虑算法兼容的BC/参考策略约束，逐项核验。

离线成功轨迹筛选后再做SFT属于筛选模仿/自训练，不应冒称已经跑了在线RL。已有评测录像和累计成功日志也不足以直接训练RL：还缺逐步动作、原始观测、奖励、终止和采样概率等轨迹字段，不能从MP4还原这些关键信息。

比赛规则仍限定clean50任务×50条训练数据；在线RL新增环境交互是否允许需主办方明确，尤其不能使用randomized评测rollout训练后仍宣称遵循原设置。本方案不修改已提交材料或生成新的比赛成绩。

## 执行与验证状态

YAML已在远端通过yaml.safe_load，并核验8卡batch/FSDP/teacher开关及数据元信息。尚未做模型构建、数据读取完整性、增强图像视觉检查、前后向、存档、HF导出或推理测试。没有启动训练或RL，没有把未运行配置写成实验结果。

## 一手参考

- ReinFlow论文：https://arxiv.org/abs/2505.22094 （v7，2026-01-08）
- 作者代码：https://github.com/ReinFlow/ReinFlow （仅查阅，未安装或执行）
- 官方比赛要求：https://tianchi.aliyun.com/competition/entrance/532514/information
- 本地代码：02_代码材料/src/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2/lingbotvla/data/vla_data/{transform.py,utils.py}，train_lingbotvla.py及实际训练YAML。
