# 实验与结果总索引

截至2026-10-27，按本地已保存证据整理的19项记录。完整机器可读字段、指标与边界见[experiment_catalog.json](experiment_catalog.json)。这不是新一轮复测，也不宣称所有历史计划均已运行。

| ID | 实验 | 状态 | 实际结果/规模 |
|---|---|---|---|
| `sft-full-6000` | [Full-SFT 正式训练 6000 步](../training/full_sft_8gpu_6000steps_20261002.log) | 已完成 | 6000步；末步loss 0.0283 / VLA 0.0243 |
| `sft-smoke-10` | [AMD Full-SFT 10 步及推理冒烟](../reproducibility/original-paths/workspace/runtime/outputs/logs/full_sft_8gpu_10steps_20261002.log) | 已完成 | 10步；6次有限值推理调用（不是回合） |
| `sft-6000-adjust-bottle` | [6k Full-SFT 独立 adjust_bottle 评测](../reproducibility/original-paths/workspace/runtime/outputs/logs/full_sft_6000_adjust_bottle_valid10.log) | 已完成 | 8/10；与全任务9/10分开 |
| `clean-bf16-50x10` | [6k clean 全任务 BF16](../reports/实验结果.json) | 已完成 | 276/500（55.2%），50任务×10 |
| `clean-fp32-50x10` | [6k clean 全任务 FP32](../reports/FP32全任务/结果.json) | 已完成 | 263/500（52.6%），50任务×10 |
| `randomized-fp32-50x10` | [6k randomized 全任务 FP32](../reports/随机场景_20261026/采集状态.json) | 已完成 | 177/500（35.4%），50任务×10 |
| `official-robotwin-adjust-bottle` | [官方 RoboTwin 后训练模型 adjust_bottle 复现](../reproducibility/original-paths/workspace/runtime/outputs/logs/robotwin_checkpoint_adjust_bottle_valid10.log) | 完成；保留修复前诊断 | 修复后10/10；早期5/10分母不可信 |
| `official-weak5-bf16` | [官方模型低分五任务 BF16 对照](../reports/官方对照/对照结果.json) | 用户取消 | 0个完整任务，不给最终成功率 |
| `lora-local-10000` | [本项目 LoRA 10000 步训练](../reproducibility/original-paths/workspace/runtime/outputs/logs/lora_10000steps_8gpu.log) | 已完成 | 10000步完成，约8h21m49s |
| `lora-pb-1001-adjust-bottle` | [队友 lora_10k_PB_1001 的 adjust_bottle 评测](../reproducibility/original-paths/workspace/runtime/outputs/logs/lora_10k_PB_1001_adjust_bottle_valid10_retry1.log) | 评测完成 | PB checkpoint评测0/10 |
| `augmentation-plan-oct26` | [颜色增强与对照配置准备](../reports/数据增强与RL方案_20261026/状态.json) | 准备；未运行 | 4个YAML，未训练/未评测 |
| `cabinet-augmentation-audit` | [柜子任务左右重采集增强审计](../development/cabinet_augmentation_audit/审计结论.md) | 静态审计 | 代码审计；无增强后闭环成绩 |
| `colab-migration-oct10` | [Colab 迁移、FP32 Full-SFT 冒烟与暂停](../development/colab-migration/migration_status.json) | 用户暂停 | 2步smoke+恢复第3步；正式0步 |
| `rlinf-v2-stage1-oct17` | [RLinf V2 接口核查与第一阶段准备](../development/lingbot_v2_rl_stage1/verification.json) | CPU准备；未GPU运行 | 记录13项CPU测试；未实现RL训练 |
| `rl-fixed-first` | [固定残差 REINFORCE 首次闭环](../reports/RL首次试验_20261026/首次RL试验结果.md) | 已完成 | 训练5/8，更新后新种子7/8；16新回合 |
| `rl-fixed-multitask` | [固定残差 REINFORCE 三任务对照](../reports/RL多任务试验_20261026/多任务RL试验结果.md) | 已完成 | 配对4→4、4→4、5→5（均/8）；72新回合 |
| `rl-fixed-ppo` | [固定残差 PPO 裁剪目标方法对照](../reports/RL_PPO试验_20261026/PPO方法对照结果.md) | 已完成 | 配对PPO 4/8、3/8、5/8；24新验证回合 |
| `rl-conditional-online` | [观测条件残差网络三轮在线采样](../reports/RL观测条件多轮_20261026/remote/status.json) | 主流程有完成记录；补充对照未核验 | 3轮4/4、3/4、2/4；2次非零更新；验证1/4 |
| `tianchi-submission` | [天池初赛提交回执](../submission/evaluation/results.json) | 回执为待评测 | 2026-10-26 18:51:41，保存回执为待评测 |

## 必须保留的比较边界

- **主结果**：6k模型的clean BF16为276/500、clean FP32为263/500、randomized FP32为177/500。每项都是50任务各10次，不是各100次；三次运行独立保存。
- randomized使用unseen指令，clean使用seen指令，因此二者差距不只是视觉随机化。BF16/FP32虽有相同场景种子记录，指令抽样与策略噪声未完全固定，不能做严格精度因果归因。
- 独立adjust_bottle的8/10与全任务中的9/10是不同运行。已保存的BF16全任务种子序列与独立运行相同，不把差异解释成“种子集不同”。
- 修复前官方adjust_bottle日志最终报5/10，但只实际记录5/5；修复后10个有效policy回合为10/10。保留该诊断历史，不能只摘最终分母当有效成绩。
- 官方weak5 BF16对照已取消，完成任务数为0。官方已发布的FP32/100次分数属于外部参考，不是这次本地测量。
- 本项目LoRA 10000步与队友`lora_10k_PB_1001`分别记录，PB的0/10不能归给另一训练运行。
- 数据增强配置、柜子增强静态审计及RLinf第一阶段准备不是增强/RL性能提升证据。Colab正式训练在optimizer step1完成前暂停，不存在新完成的10k训练。

## RL记录如何计数

1. 首次固定残差：16个真实新回合；5/8→7/8使用不同种子，不据此宣称提升。
2. 三任务REINFORCE：72个新回合，每任务训练8次、独立control验证8次、更新后验证8次。control为未学习的随机残差策略。
3. PPO对照：复用前项24个训练回合，只新增24个验证回合；4个优化epoch不等于4轮新在线采样。
4. 观测条件网络：本地状态记录12个新训练回合与4个新验证回合，5900参数、171维输入，两次非零更新。第一轮4/4回报相同，真实零更新；版本号增加不等于发生有效更新。
5. 所有这些RL试验都冻结6k主模型；固定12维残差与有观测条件MLP分别列出，不称为6B主模型或动作专家微调。CPU/合成自检、expert预检查不加入policy成功率。

## 观测条件RL补充control：未核验

现有本地备份未找到`supplement_status.json`、`conditional_paired_validation.json`或最终`final_audit.json`。已有补充采样脚本不证明4个对照回合完成；当前235条transfer manifest也不含control结果。因此**只登记已保存的16个闭合回合，不把计划中的额外4回合计为已完成**。最终验证1/4缺已核验同种子初始网络对照，不能与三轮不同种子训练成功率直接比较。

## 提交与发布范围

天池保存回执表明2026-10-26 18:51:41接收了材料，当时状态“待评测”；不代表审核通过、已得分或晋级。提交的clean/randomized采用真实每任务10次，复用上表既有结果，不构成额外1000个新回合。

本公开归档包括结果、代码、复现配置、报告、审计与录像。按本次发布范围不上传大模型权重、完整DCP、大优化器存档、原始训练数据或含权重的原始13.3GB提交ZIP。原6k训练与模型的版本/完整性清单可保留，但清单不等于权重本身已上传。5k及早期LoRA大权重此前已按用户指令取消备份，不声称本地或GitHub提供完整恢复。

## 目录与证据落点

- `reports/`：实验报告、结构化结果、RL状态与审计。
- `results/`：全任务原始评测日志、done和时序记录。
- `training/`：正式训练日志、指标与loss曲线。
- `submission/code/`、`submission/evaluation/`、`submission/reproduction/`：正式提交的代码、结果模板与复现说明。
- `reproducibility/original-paths/`：历史日志/配置的原目录相对结构。
- `development/`：迁移、增强审计与RL接口准备。

录像、NPZ轨迹与轻量RL checkpoint等大体积证据存放于Release附件，附件内部保留上述相对路径；仓库和Release的精确映射以归档manifest为准。索引不把分享ZIP、解包目录、提交目录或迁移分块中的同一文件当成新的实验结果。
