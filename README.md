# cucu · LingBot-VLA 2.0 / RoboTwin 2026 比赛实验归档

cucu队在蚂蚁灵波具身大模型挑战赛中的训练、评测、失败分析及RL探索记录。此仓库同时保留有效结果和没有改善的实验；不会将冒烟、取消或仅准备的工作写成已完成训练。

**归档日期：2026-10-27。** 队伍ID：1382038。基础模型为 `robbyant/lingbot-vla-v2-6b`，6000步Full-SFT得到比赛候选。没有用官方RoboTwin后训练权重初始化本候选。

## 主结果

| 模型/设置 | 实际任务 × 回合 | 成功数 | 成功率 |
|---|---:|---:|---:|
| Full-SFT 6k · clean · BF16（诊断运行） | 50 × 10 | 276/500 | 55.2% |
| Full-SFT 6k · clean · FP32（提交结果） | 50 × 10 | 263/500 | 52.6% |
| Full-SFT 6k · randomized · FP32（提交结果） | 50 × 10 | 177/500 | 35.4% |

每个任务实际自测10次，**没有完成官方要求的每任务100次**。提交JSON如实填写 `attempts=10`，没有把10次改成100次。clean默认seen指令，randomized默认unseen指令；规划器为MPLib、动作执行长度25。运行条件与官方发布环境存在差异，详见复现报告；BF16/FP32分差不能单独作为精度因果结论。

正式包于 **2026-10-26 18:51:41（香港时间）** 被天池接收，当时显示“待评测”。这是上传接收记录，**不是审核通过或最终排名**。此归档没有再次提交研究用RL模型。

## 训练与后续实验

- 6k Full-SFT：8×AMD Radeon PRO W7900D、AdamW、micro batch16/global batch256、FSDP2 full shard、梯度检查点、视觉参与训练、full teacher。约31小时57分，末步总loss0.0283/VLA loss0.0243。
- 历史LoRA训练、专项模型准入与单任务评测：完整保留日志，并区分本项目与队友的不同LoRA运行。
- 固定残差REINFORCE：仅更新12个残差均值，3任务配对验证未观察到净提升。
- 固定残差PPO裁剪目标：复用同一批真实训练回合，24个新验证回合；两项不变、一项少成功1次。
- 有观测条件残差网络：171维图像/状态/基础动作特征、5900参数，3轮新在线采样，**两次非零更新**；训练回合4/4、3/4、2/4，最后新种子验证1/4。最后补充control未同步到本地，未认定完成。
- 数据增强方案、柜子左右增强审计、Colab迁移和RLinf适配核查：分别标注实际完成范围，不把静态检查当作训练或闭环增益。

完整19项记录、证据和限制见 **[实验总索引](docs/EXPERIMENT_INDEX.md)** 与 [机器可读目录](docs/experiment_catalog.json)。

## 文件在哪里

| 目录 | 内容 |
|---|---|
| [`reports/`](reports/) | 原报告、逐任务结果、原始运行状态、统计与审计 |
| [`results/`](results/) | BF16/FP32原始评测日志、回合/录像目录清单 |
| [`training/`](training/) | 6k完整训练日志、metrics CSV、loss曲线 |
| [`submission/evaluation/`](submission/evaluation/) | 按官方模板填写的真实clean/randomized结果 |
| [`submission/code/`](submission/code/) | 对应代码、训练/评测脚本和配置，含固定版本源码快照 |
| [`submission/reproduction/`](submission/reproduction/) | 官方五节格式复现报告及训练证据 |
| [`reproducibility/`](reproducibility/) | 补丁、历史配置与日志、模型/数据备份清单 |
| [`development/`](development/) | 数据增强审计、Colab迁移、RL适配检查代码和结果 |
| [`docs/`](docs/) | 实验索引、公开范围与下载/核验说明 |

**[全部录像、逐chunk轨迹和小型RL参数下载](https://github.com/KQuaneo/cucu-lingbot-robotwin-2026/releases/tag/results-2026-10-27)**。Release分为clean、randomized、训练复现、RL和开发实验五个证据包；其中三个以32MiB分块上传，提供一键重组校验脚本。下载和SHA256校验见 [ARTIFACTS.md](docs/ARTIFACTS.md)。

## 复现入口

比赛候选的配置和命令见 [原复现说明](submission/reproduction/reproduction_report.md) 和 [代码README](submission/code/README.md)。这些是已保存实验的配置/源码快照，包含原平台路径；更换机器需先准备匹配环境、公开数据和teacher权重并调整路径。未宣称在另一台全新机器上重新验证了完整6000步训练。

基础模型revision：`11c703bf6a5c1f45b3b69168482da11fdbba53d7`。

| 上游 | 固定commit |
|---|---|
| RoboTwin | `266f3aadf505a4f7fe9af0faa41a20f5f47cd123` |
| XPolicyLab | `c37109c500be67d0dea6b36bf7337bbd26e763cd` |
| LingBot-VLA-v2 | `951475ae1b1d87553e7dc47c97b53a3d695c0d13` |

## 公开范围与重要限制

本次按队伍决定公开**结果、代码、录像和小型RL网络/残差参数**，不上传12.77GB的6k主模型权重、76.59GB续训DCP/优化器、训练数据搬运缓存，或含这些内容的13.3GB原提交ZIP。配置和校验清单保留，不能据此误以为大权重也已公开。

公开副本去除了账号核对、SSH地址、本机用户名/绝对项目路径、临时上传信息和下载签名。原始本地材料未修改。`source_sha256`对应未脱敏原文件，`published_sha256`对应公开副本；不同哈希并不表示传输损坏。见 [公开范围说明](docs/PUBLICATION_SCOPE.md)。

固定/条件残差RL均冻结6k主模型，不能称为6B主模型RL微调。小样本下没有可靠性能提升结论；相同成功数也可能包含个别回合改善与退步。

## 许可证与引用

上游源码、模型配置和附带依赖按各自许可证发布，保留原LICENSE/NOTICE及版权头；**整个仓库不能统一视为MIT或可商用**。相关说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本归档没有为原创实验脚本另行指定新的开源许可证。

官方参考：[比赛说明](https://tianchi.aliyun.com/competition/entrance/532514/information)、[复现指南](https://github.com/ZiguanWang/Robotwin-radeon-cloud/blob/main/Reproduce_Guide.md)。
