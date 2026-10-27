# LingBot V2 RL 第一阶段准备包

2026-10-17。此包实现权重结构检查、严格加载工具、观测和动作映射、原ODE推理接口及GPU一致性验收脚本。**尚未实现GRPO/PPO、随机采样链和RL概率重算；不是可直接训练的RLinf模型插件。**

没有改写原始备份、模型、数据或原训练代码。适配入口放在独立文件中。

## 已运行

- 本地真实6k checkpoint的3个分片头部、张量区间和index结构检查：1708个BF16张量、6375907511个张量元素。包含训练辅助分支，不等于可训练策略参数数量。
- 从实际训练配置、robotwin.yaml推导状态与动作映射：`[0,1,2,3,4,5,28,6,7,8,9,10,11,29]`。
- 13项CPU测试：不同joint组顺序、左右夹爪哨兵往返、缺失/额外/形状错误权重、非有限权重、截断分片、相机顺序、非法输入、固定噪声回放和执行时域。
- 从真实adjust_bottle数据抽取episode 0的第0和30帧，保存三视角RGB、14维状态和对应语言。视频时间戳与数据行对齐，来源见provenance文件。样本只用于推理一致性检查，不是新采集的RL经验。

**未运行**：真实6B模型加载、GPU前向、与原推理数值比较、仿真交互、策略梯度与多卡训练。分片头部检查不替代完整内容SHA256校验；CPU测试用小模型/合成张量，不能证明6B GPU数值正确。

## 文件

- `stage1.py`：checkpoint_inventory、derive_indices、strict_load、raw_observations、V2ODEBridge。
- `test_stage1.py`：CPU验收测试。
- `local_audit.json`：真实本地checkpoint静态检查结果。
- `export_fixture.py`：从现有LeRobot v3首episode导出两帧同步输入的小工具。
- `adjust_bottle_observations.npz` / `.provenance.json`：真实输入与来源记录。
- `gpu_equivalence.py`：在配置完毕的Linux/CUDA机器上，严格加载并与原server推理比较。只在通过后写入passed结果。

## 本地检查

需要Python 3.10+、PyTorch、NumPy、PyYAML、safetensors；导出真实样本另需PyArrow、PyAV。此轮复用了已有lerobot环境，没有新增或升级依赖。

```bash
python -m unittest -v test_stage1
python stage1.py --checkpoint /path/full_sft_6000 \
  --robot-config /path/lingbot/configs/robot_configs/robotwin.yaml \
  --output local_audit.json
```

derive_indices仅适用于本项目absolute joint-position映射。它不做归一化；有relative/derived action会拒绝，不能拿它代替完整FeatureTransform。

strict_load先检查全部key/shape再逐分片加载，避免拼接整份大权重；默认不允许额外key。GPU验收仅明确允许与原推理server相同的训练辅助分支前缀。它不负责构建Qwen3模型、tokenizer或训练优化器；中途内容读取错误应销毁该模型后重新加载。

## GPU验收

先安装已有LingBot源码配套依赖（原迁移方案为PyTorch2.8.0+cu128、FlashAttention2.8.3），准备Qwen3-VL配置/tokenizer和归一化文件。无需此时启动RoboTwin仿真。原server使用CUDA，不能以Mac/MPS运行替代这个检查。

```bash
python gpu_equivalence.py \
  --source /path/lingbot \
  --checkpoint /path/full_sft_6000 \
  --qwen-config /path/Qwen3-VL-4B-Instruct-config-tokenizer \
  --norm-stats /path/lingbot/assets/norm_stats/robotwin.json \
  --observations ./adjust_bottle_observations.npz \
  --output ./gpu_equivalence.json \
  --precision fp32 --execute-steps 25
```

默认误差阈值atol=1e-5、rtol=1e-4，失败不能静默放宽。BF16应单独验证并预先确定合理阈值，不与FP32结果混记。固定噪声会先clone，因为上游Euler循环原地修改噪声；当前bridge仅支持预处理后的原ODE采样，不返回任何虚构logprob。

GPU脚本从原部署模块导入server，子类只替换权重加载检查；图像预处理、语言处理、FeatureTransform与动作反归一化仍走原路径。验证对象是这条路径的一致性，不代表环境state与控制器的端到端语义已经验证。

## 接下来

通过真实GPU一致性检查后，才接入flow-SDE采样和概率回放，验证旧策略概率重算一致、更新前概率比约1、有效14维/实际执行步mask正确。随后接RoboTwin奖励与一次更新。暂时不应按本包启动长RL训练或估算所需卡数。
