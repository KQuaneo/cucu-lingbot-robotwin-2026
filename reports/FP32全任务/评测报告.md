# 6k 模型 FP32 全任务评测

更新时间：2026-10-04T06:43:29.340109+08:00。状态：completed。完整任务 **50/50**；完整任务合计 **263/500（52.60%）**。全部已记录回合 500/500。退出码：0。

原 BF16 全任务：276/500，55.20%。进行中的分数仍会变化；未完成任务不作为0分。比较总体成绩需要两轮都完整结束。

| 任务 | 原 BF16 | 当前 FP32 | 状态 |
|---|---:|---:|---|
| adjust_bottle | 9/10 | 9/10 | 完成 |
| beat_block_hammer | 8/10 | 6/10 | 完成 |
| blocks_ranking_rgb | 5/10 | 6/10 | 完成 |
| blocks_ranking_size | 3/10 | 2/10 | 完成 |
| click_alarmclock | 6/10 | 6/10 | 完成 |
| click_bell | 9/10 | 6/10 | 完成 |
| dump_bin_bigbin | 6/10 | 7/10 | 完成 |
| grab_roller | 10/10 | 10/10 | 完成 |
| handover_block | 1/10 | 0/10 | 完成 |
| handover_mic | 9/10 | 9/10 | 完成 |
| hanging_mug | 4/10 | 4/10 | 完成 |
| lift_pot | 2/10 | 4/10 | 完成 |
| move_can_pot | 2/10 | 1/10 | 完成 |
| move_pillbottle_pad | 2/10 | 3/10 | 完成 |
| move_playingcard_away | 9/10 | 9/10 | 完成 |
| move_stapler_pad | 1/10 | 2/10 | 完成 |
| open_laptop | 9/10 | 9/10 | 完成 |
| open_microwave | 5/10 | 1/10 | 完成 |
| pick_diverse_bottles | 5/10 | 3/10 | 完成 |
| pick_dual_bottles | 1/10 | 4/10 | 完成 |
| place_a2b_left | 4/10 | 5/10 | 完成 |
| place_a2b_right | 8/10 | 7/10 | 完成 |
| place_bread_basket | 8/10 | 10/10 | 完成 |
| place_bread_skillet | 7/10 | 8/10 | 完成 |
| place_burger_fries | 9/10 | 9/10 | 完成 |
| place_can_basket | 6/10 | 7/10 | 完成 |
| place_cans_plasticbox | 5/10 | 3/10 | 完成 |
| place_container_plate | 8/10 | 8/10 | 完成 |
| place_dual_shoes | 3/10 | 0/10 | 完成 |
| place_empty_cup | 9/10 | 9/10 | 完成 |
| place_fan | 3/10 | 3/10 | 完成 |
| place_mouse_pad | 3/10 | 1/10 | 完成 |
| place_object_basket | 8/10 | 9/10 | 完成 |
| place_object_scale | 9/10 | 8/10 | 完成 |
| place_object_stand | 7/10 | 5/10 | 完成 |
| place_phone_stand | 5/10 | 3/10 | 完成 |
| place_shoe | 4/10 | 3/10 | 完成 |
| press_stapler | 4/10 | 4/10 | 完成 |
| put_bottles_dustbin | 4/10 | 0/10 | 完成 |
| put_object_cabinet | 0/10 | 1/10 | 完成 |
| rotate_qrcode | 5/10 | 5/10 | 完成 |
| scan_object | 3/10 | 4/10 | 完成 |
| shake_bottle | 10/10 | 10/10 | 完成 |
| shake_bottle_horizontally | 10/10 | 10/10 | 完成 |
| stack_blocks_three | 0/10 | 0/10 | 完成 |
| stack_blocks_two | 6/10 | 5/10 | 完成 |
| stack_bowls_three | 3/10 | 6/10 | 完成 |
| stack_bowls_two | 9/10 | 7/10 | 完成 |
| stamp_seal | 2/10 | 3/10 | 完成 |
| turn_switch | 8/10 | 9/10 | 完成 |

## 条件与限制

使用同一6k导出checkpoint和模型配置，仅将推理设置改为FP32；50个demo_clean任务，各10个有效回合，seen指令、seed参数0、expert预检与失败信息接纳、valid retry patch、录像、eager、use_length25、不编译。优先执行原BF16耗时最长的任务。

实际有效种子可能因初始化重试不同；逐任务CSV及JSON保留两轮种子与重叠种子结果。指令抽样、策略噪声和规划随机性可能不同。FP32转换已有权重无法恢复保存时损失的精度。每项仅10次，暂不做显著性或因果判断。

原始日志和本次准确录像目录位于 ../../结果/FP32全任务/。录像按已完成任务持续备份到 ../../录像/FP32全任务/。旧BF16结果与录像保留。

服务器预计10:10回收，评测设置09:35（香港时间）截止，给最终备份预留时间；提前完成则正常退出。未完成时明确记录部分结果。

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-04
- Verification Status: VERIFIED (execution/log consistency only)
- Version Label: full-sft-6000-fp32-clean50x10-v1

## ExperimentResult

- ID: full_sft_6000_clean50x10_fp32_8gpu_video_20261004
- Status: completed
- Working Directory: /RoboTwin
- Exit Code: 0
- Output: 本报告、结果.json、CSV、原始日志、已闭合录像。
