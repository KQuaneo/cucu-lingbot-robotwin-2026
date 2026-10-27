# 首次RL闭环实验：冻结6k的残差策略

这是研究冒烟，不是6B主模型的ReinFlow微调，不修改或重新提交比赛包。任务为click_bell/demo_clean，原模型历史clean成功6/10；本实验用新训练seed210000起，必要时220000起，更新后seed230000起。每批8个并行实际policy回合，初始化不稳定的场景按原valid-episode retry规则换seed。没有使用randomized评测轨迹训练。

可训练对象是12维高斯潜变量分布的均值mu，初始0、固定单位标准差。每个回合采样一次z~N(mu,I)，整个回合的12个手臂关节目标叠加0.004*tanh(z)弧度的偏移；两个夹爪维度不变。6k模型完全冻结，FP32、动作块25步。偏移有界，但它不具备观测条件，也不能代表复杂的接触/恢复策略。

环境终局任务成功为reward=1，否则0。对一个回合只计算一次潜变量log probability，不因反复执行相同偏移而重复计算。score-function REINFORCE loss=-mean(advantage*log N(z;mu,I))，advantage使用其它回合奖励的均值作为leave-one-out baseline。CPU上计算梯度，一次SGD更新，LR0.2，均值更新L2范数上限0.2，对应潜变量高斯KL不超过0.02。若两批所有奖励相同，明确记录无有效更新，不伪造梯度或更改奖励。

输入三路图像/机器人状态、原模型动作、叠加后动作、采样潜变量和均值保存在逐chunk NPZ及episode.json。执行回执executed_chunks.jsonl逐chunk记录实际执行步数（成功可能提前结束），不能把输出的25步都算已执行。真实终局奖励通过独立eval副本的WebSocket事件传回；原评测脚本和成功判定不修改。

controller 19:50香港时间前停止，仅清理自己启动的进程组；持久化status.json、policy_update.json、12维residual_policy_checkpoint.json和退出码。原6k HF文件保持只读使用。collect_trial.py将已闭合轨迹、脚本、日志和录像打包下载，逐成员SHA256验证；不复制未完成的NPZ。

验收要求：真实采样有图像/状态/动作；每回合真实终局reward与最终日志一致；mean梯度有限且非零；checkpoint均值发生有限变化；更新后重新完成8回合采样；所有本地证据SHA通过。真实训练/更新后成功率只能描述本次样本，不能用8–16次训练和8次验证声称性能提高。

算法自检已验证：成功潜变量方向提高概率，全部等回报时更新为0。实际成败以remote/status.json和退出码为准。仅有脚本创建或服务启动不代表试验完成。
