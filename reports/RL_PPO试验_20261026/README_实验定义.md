# PPO残差策略方法对照

本次试验不更新6B主模型，而是使用与REINFORCE试验相同的12维episode-level Gaussian latent residual。每个任务从mu=0、std=1重新初始化，关节偏移仍为0.004*tanh(z)弧度、夹爪不变。

方法：PPO clipped actor objective，clip epsilon=0.2、Adam lr=0.03、每批4个优化epoch、梯度范数上限1、mu变化范数上限0.2（latent KL<=0.02）。advantage为真实终局成功0/1回报减去其它回合均值，没有训练value critic。每回合只计一次高斯潜变量概率；变换的Jacobian在新旧策略比率中抵消。

每任务复用上一轮从mu=0采集的8个真实训练回合，其实际reward和初始策略已在此前审计。旧策略对应mu=0，不加载REINFORCE更新后的mu。共复用24个on-policy训练样本，未新增训练采样；不把这些已有样本冒称新采样，也不把4个优化epoch称为4轮在线采样训练。输入来源SHA另存training_provenance.json。

三个任务beat_block_hammer、rotate_qrcode、press_stapler，每项新增8个PPO验证回合，与前轮control的实际seed、完整指令、epsilon和基础模型torch seed匹配。验证reward不用于参数更新。与REINFORCE除了clipping也有优化器、学习率和优化epoch差异，不能把结果变化纯粹归因于clipping。

算法自检：初始梯度与解析式一致；裁剪正/负advantage的符号正确；恒定奖励时更新为零；latent KL受限。实际结果以remote/status.json、退出码、逐回合核验为准。硬截止2026-10-26 19:56香港时间，保留备份时间；未完成任务不伪报完成。

这不是LingBot/ReinFlow原生完整RL训练器，只是对受限残差策略的PPO目标试验；没有teacher更新、没有randomized训练、没有替换或重新提交比赛包。
