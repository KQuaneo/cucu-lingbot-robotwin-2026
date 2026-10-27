# 多任务残差RL试验（研究，独立于已提交模型）

任务：beat_block_hammer、rotate_qrcode、press_stapler，全部demo_clean。8GPU分别运行冻结的6k FP32模型；每任务独立初始化12维Gaussian latent mean为0，固定std=1，关节偏移0.004*tanh(z)，不改夹爪。任务间不继承残差参数。

每任务：8回合在线采样→真实终局奖励0/1→一次REINFORCE更新（leave-one-out baseline，LR0.2，参数更新范数上限0.2）。若8回合奖励全相同，再采另一批8回合；仍无奖励差异则记录无有效更新，不伪造梯度。

有更新后：先以mu=0的未更新随机残差策略，在8个独立验证seed上采样，随后用更新mu在同一组实际seed、完全相同指令、同一潜变量epsilon和相同基础模型torch随机种子上采样。验证回合不参与梯度更新。注意control并不是无残差的原6k模型；它是未学习的随机残差策略。对应关系会逐回合核验，失败配对不冒称严格配对。环境GPU数值与物理仿真仍可能有非确定性，因此不能声称位级相同行为。

采样初始seed按worker间隔100分配，减少初始化retry造成不同worker种子重复的机会。所有实际seed与指令保存。每任务最多32回合，通常24；3任务通常72。不同任务的12维checkpoint和梯度分别保存，不合并为一个多任务策略，也不更新6B主模型或teacher。

硬截止19:50香港时间；停止仅针对本试验进程组。使用独立远端目录/workspace/runtime/experiments/residual_rl_multitask_20261026和本地本目录。原评测脚本只读，新增实验副本在build_instruction处接收验证指令覆盖，用于匹配；真实成功判定不变。保存完整逐chunk图像/状态/动作、执行步数回执、终局奖励、录像、梯度和参数。collect_trial.py只备份闭合回合，并逐成员SHA校验。

脚本创建或服务开始不代表完成，结果以remote/status.json、各task_results和退出码为准。样本小且只做一次更新，即使验证成功数增加，也不声明训练已普遍提升。
