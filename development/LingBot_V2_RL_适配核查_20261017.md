# LingBot-VLA V2 6B → RLinf / RoboTwin 适配核查

日期：2026-10-17（香港时间）。范围：本地静态代码与配置审查、上游接口对照；没有启动 GPU、仿真 rollout 或 RL 训练，也没有修改已有训练源码和模型。

## 结论

可以开展 V2 6B 的 RL 适配，但现有 RLinf LingBot 接口不能直接加载本项目的 6k 权重运行。需要新增 V2 wrapper，而不是仅修改 model_path。当前完成的是适配核查，不是可运行的 RL 实现。

本地已具备：V2 源码、6k HF checkpoint、机器人配置与归一化文件、RoboTwin 任务源码。它们可支持接口开发和静态检查。真实数值等价、梯度、环境交互及性能验证需要后续 Linux/NVIDIA 环境。

## 已核实的差异

| 环节 | 本项目 V2 | RLinf 现有 LingBot wrapper | 处理要求 |
|---|---|---|---|
| 模型类 | LingbotVlaV2Policy / FlowMatchingV2 | 从 pi0.modeling_lingbot_vla 导入 LingbotVlaPolicy | 单独注册 V2 模型，保留旧版接口 |
| 视觉语言骨干 | Qwen3-VL，image_grid_thw、三轴位置及 DeepStack 视觉输入 | 旧版前缀与位置处理，文档使用 Qwen2.5-VL | 复用 V2 embed_prefix、位置构建与 predict_velocity，不复制旧版前向 |
| 动作与状态 | config.max_action_dim=55；max_state_dim 以配置为准；机器人映射由配置解释 | wrapper 有 75 维 padded-state 重排和固定动作维度索引 | 从 V2 机器人配置和规范特征顺序派生映射，禁止直接截前14维或照搬旧索引 |
| 采样 | 初始高斯噪声 + Euler 更新 x_t += dt*v_t，返回动作张量 | 随机 flow_sde 等采样，保存 chains、prev_logprobs、denoise_inds | 增加可回放的随机转移；保留原 ODE 推理作为等价基准 |
| 策略更新 | forward 返回 Flow Matching 及辅助损失 | default_forward 重算采样链的 logprobs，交给 RL loss | 新增与 rollout 一致的可微概率重算入口 |
| 动作时域 | 预测50步，部署支持 use_length 截取；旧评测记录用25步 | wrapper 按 action_chunk 采样并传递环境动作 | 明确预测长度与执行长度，记录实际执行 mask，统一奖励归属 |
| MoE 与分布式 | 32个专家、top-k=4，另有共享专家与既有并行计划 | wrapper 默认 action_expert 训练范围，切分规则针对旧类 | 审查专家、路由器、投影层是否被正确训练和同步；先单卡再多卡 |
| 环境 | 本地 RoboTwin 普通评测接口，含 check_success | RLinf 需要 RLinf_support 分支的 robotwin.envs.vector_env.VectorEnv | 对照 reset/step/chunk_step、相机键、状态、seed、done/truncation 语义 |

“初始噪声是随机的”不等于已有可直接用于 GRPO 的动作概率。RLinf 使用的是特定随机采样链及其转移概率；不能给最终动作随意添加高斯噪声后沿用旧 logprob。

## 四个具体阻塞点

1. **Qwen3-VL 前向不兼容**：V2 embed_prefix 返回六项（含 prefix_position_ids、visual_pos_masks、deepstack_visual_embeds）；RLinf 旧采样入口按三项解包。V2 predict_velocity 明确要求 prefix_position_ids。
2. **模型维度映射不兼容**：RLinf 固定状态重排引用索引73/74，模型动作索引将夹爪放在14/15。本项目55维规范空间必须从 V2 的机器人配置解析。实际环境仍为14维，不能混淆内部规范空间与环境动作。
3. **缺少 RL 采样与重算接口**：V2 sample_actions 只有最终动作；需要保存去噪链、选中随机步、旧概率、观测与版本信息，并用同一观测/处理结果重算概率。评估路径必须支持原 ODE 行为。
4. **权重加载需要严格检查**：构造正确 V2 类、Qwen配置与 tokenizer；核对 missing/unexpected keys、MoE 张量、参数数量和 dtype。禁止通过 strict=False 静默跳过不匹配，把随机初始化模块当成恢复成功。

## 奖励与任务选择

本地 adjust_bottle.check_success 已有基于瓶子目标侧和高度的成功判定，可以作为稀疏奖励的来源。RLinf wrapper 也已有环境奖励/custom reward、相对奖励、成功指标和 timeout 管理接口，但需要验证 RLinf_support 分支的任务实现与本地版本一致。

第一版实验设计建议：任务成功奖励一次，未成功为0；timeout 单独记录。不要把所有 termination 都当成功，也不要在成功后每步重复发奖励。若用阶段奖励，必须先审计其是否鼓励刷分，再与稀疏奖励对照。

优先从已有50任务评测结果中选“有成功也有失败”的任务。adjust_bottle 先用于接口冒烟；历史10/10仅对应当时专项模型和配置，不能证明6k模型在新协议下100%，也不能据此断言它不适合RL。

GRPO 同一初始状态下需要有回报差异。全成功或全失败的组可能没有有效相对优势；应报告非零优势组占比、成功率和组内回报方差，不能只看 RL loss。

## 实施顺序与验收

| 顺序 | 内容 | 通过标准 | 运行环境 |
|---|---|---|---|
| 1 | V2 wrapper 的加载、观测与动作映射 | 参数无异常漏载；三视角、tokenizer与归一化一致；动作14维往返可核对 | 本地可开发；模型运行需GPU |
| 2 | 原 ODE 采样等价 | 相同图像、状态、指令和固定噪声，与现有推理接口数值一致；dtype与误差阈值预先记录 | Linux/GPU |
| 3 | 随机采样与概率回放 | 同一批链和同一策略重算 logprob 一致；更新前概率比约1；选定随机步 sigma>0且概率有限 | Linux/GPU |
| 4 | 14维有效动作与时域mask | 非控制维、padding和未执行动作不误计入目标；chunk奖励和done边界正确 | 单元检查 + 环境 |
| 5 | 奖励、reset与seed | 成功奖励不重复、超时不误报成功；同组初始状态一致，组间与评测种子独立 | RoboTwin/GPU |
| 6 | 一次RL更新与保存恢复 | 有限loss/梯度；目标模块参数确实更新；actor/rollout权重一致；保存重载后输出可复现 | Linux/GPU |
| 7 | 小规模对照 | 6k基线 vs RL后，同样评测种子、动作执行长度、精度、规划器；记录成功率和遗忘 | Linux/GPU |

在第1—6项完成前，不启动长训练。先少量环境与一个任务；上游示例的128环境、8卡和global batch 1024不是本项目可直接照搬的硬件要求。所需卡数和预算应由真实显存与步速测量决定。

RL 本身不强制继续 SFT 的全部 depth/video teacher 辅助损失。第一版是否保留辅助项是实验选择，应记录并对照；不能未经检查删掉模型内部的感知 query、mask 或已有分支。SFT+RL混合若引入teacher则另计资源。

## 本地证据

- `outputs/colab-migration/package/lingbot/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`：FlowMatchingV2.embed_prefix / sample_actions / predict_velocity，LingbotVlaV2Policy.forward / get_parallel_plan。
- `硬盘/模型/full_sft_6000/config.json`：V2模型类、55维动作、50步预测、10次去噪、Qwen3-VL与MoE参数。此轮只读配置，没有重新校验大权重全部字节。
- `outputs/colab-migration/package/lingbot/configs/robot_configs/robotwin.yaml`：环境14维状态/动作中的左右臂和夹爪来源，以及三路图像键。
- `outputs/colab-migration/package/lingbot/deploy/lingbot_vla_v2_policy.py`：_infer_batch、归一化反变换与use_length截取。
- `outputs/colab-migration/RoboTwin/envs/adjust_bottle.py`：check_success。

## 上游来源与复核边界

查阅日 RLinf main HEAD：`6b7853bcb493bef7b5c194ed4811207c7097bee5`。接口文件通过 GitHub 连接读取；未完整克隆RLinf或执行其依赖环境。文档和文件可能继续变化，实现前应固定完整提交。

- [RLinf LingBot文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/lingbotvla.html)：旧4B权重、GRPO与flow_sde配置。
- [RLinf LingBot wrapper](https://github.com/RLinf/RLinf/blob/6b7853bcb493bef7b5c194ed4811207c7097bee5/rlinf/models/embodiment/lingbotvla/lingbotvla_action_model.py)：本轮读取对应文件内容；目录查询返回blob SHA `7985fce7de77903d082bad2d91b9d82b8d6cce35`。
- [模型工厂](https://github.com/RLinf/RLinf/blob/6b7853bcb493bef7b5c194ed4811207c7097bee5/rlinf/models/embodiment/lingbotvla/__init__.py)。
- [RoboTwin环境wrapper](https://github.com/RLinf/RLinf/blob/6b7853bcb493bef7b5c194ed4811207c7097bee5/rlinf/envs/sim/robotwin/robotwin_env.py)。
- [GRPO示例](https://github.com/RLinf/RLinf/blob/6b7853bcb493bef7b5c194ed4811207c7097bee5/examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla.yaml)。

以上是已读代码的静态判断，不代表已排除所有框架或仿真兼容问题，也不代表RL一定提升成功率。
