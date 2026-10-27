# RoboTwin 数据池生成期数据增强框架

本目录是一套可迁移到同类型 RoboTwin 实例的补丁。它把数据增强分为通用事务层和任务插件层：通用层负责锁、编号、来源快照、staging、journal、manifest、恢复和数据完整性；插件只负责生成一个临时 episode，并进行任务级物理验收。

## 一、现有功能

内置插件：

```text
put_object_cabinet_symmetry
```

该插件基于 clean seed 重建场景，将任务侧转换到对侧，交换双臂职责，在仿真中重新规划并执行完整操作。它同步生成图像、关节状态、末端位姿、夹爪状态和 action，不是只做图像水平翻转。

## 二、安装到另一个实例

先只检查将要覆盖的文件：

```bash
python install_augmentation_framework.py \
  --root /RoboTwin \
  --dry-run
```

确认目标实例目录和文件清单后安装：

```bash
python install_augmentation_framework.py \
  --root /RoboTwin \
  --install
```

安装器会先备份所有将被覆盖的文件，并输出 backup 路径。回滚：

```bash
python install_augmentation_framework.py \
  --root /RoboTwin \
  --rollback /RoboTwin/.augmentation_backup_YYYYMMDD_HHMMSS
```

## 三、实例自检

只检查依赖、RoboTwin commit、插件注册和池状态，不写数据：

```bash
cd /RoboTwin

/opt/robotwin-env/bin/python -m scripts.augmentations.check \
  --root /RoboTwin \
  --pool /RoboTwin/data/demo_clean/put_object_cabinet/aloha_agilex \
  --seed-file /RoboTwin/data/demo_clean/put_object_cabinet/aloha_agilex/seed.txt \
  --plugin put_object_cabinet_symmetry \
  --task put_object_cabinet \
  --config demo_clean \
  --base-count 50 \
  --target-count 10 \
  --dry-run
```

`--dry-run` 严格只读：不会创建 lock、补采基础数据、迁移 manifest、恢复 journal 或生成 episode。如果发现 pending journal，只报告拟恢复动作。

显式恢复 journal 并校验池，但绝不生成新 episode：

```bash
/opt/robotwin-env/bin/python -m scripts.augmentations.check \
  --root /RoboTwin \
  --pool <pool_path> \
  --seed-file <pool_path>/seed.txt \
  --base-count 50 \
  --target-count 10 \
  --repair
```

## 四、生成命令

通用新命令：

```bash
cd /RoboTwin

bash collect_data.sh \
  put_object_cabinet demo_clean 0 \
  --data-augmentation put_object_cabinet_symmetry \
  --augmentation-count 10
```

旧命令仍兼容：

```bash
bash collect_data.sh \
  put_object_cabinet demo_clean 0 \
  --object-cabinet-sym-aug \
  --sym-aug-count 10
```

也兼容先前误拼写：

```text
--object-cabinet-sysm-aug
```

不传任何增强参数时，不会导入 registry 或插件，普通采集行为不变。普通采集和增强提交共用 `.augmentation.lock`，同一个已更新实例上的两个采集进程不会并发写同一数据池。

## 五、核心文件

```text
RoboTwin/
├── collect_data.sh
└── scripts/
    ├── collect_data.py
    ├── mirror_put_object_cabinet.py
    ├── append_put_object_cabinet_symmetry.py   # 旧API兼容层
    └── augmentations/
        ├── api.py
        ├── registry.py
        ├── validators.py
        ├── appender.py
        ├── check.py
        └── plugins/
            ├── put_object_cabinet_symmetry.py
            └── template.py
```

## 六、职责边界

### `registry.py`

只保存插件名称、版本、支持的 task/config 和惰性 factory 字符串。注册时不导入仿真器或插件实现。

### `api.py`

定义 `AugmentationPlugin` 和 `GenerationResult`。插件返回受控状态：

```text
success
planning_failure
collection_failure
validation_failure
exception
```

### `appender.py`

负责所有正式池写入：

- 池级进程锁；
- `augmentation_manifest.json` 全局编号所有权；
- source fingerprint 与代码 SHA-256；
- staging 和通用 bundle 校验；
- commit journal；
- 多文件提交和中断恢复；
- attempt 记录；
- legacy 写集合的原子提交。

### task plugin

只负责：

- 判断 task/config 是否适用；
- 根据一个 clean source 在 staging 中生成 `episode_0000000`；
- 返回 provenance、scene entry 和物理成功指标；
- 执行只读的任务级额外验证；
- 声明 legacy JSON 的目标内容，不直接写正式池。

## 七、池级状态文件

```text
augmentation_manifest.json
augmentation_<plugin>_attempts.json
augmentation_<plugin>_sources.json
.augmentation_commit_<episode>.json
.augmentation.lock
```

`augmentation_manifest.json` 是所有增强插件共享的唯一正式编号所有权来源。增强区必须为稠密前缀；未知文件、编号空洞、dangling scene 或未登记 artifact 均会停止运行，而不是猜测归属。

当前 cabinet 插件继续同步：

```text
symmetry_seed_mapping.json
symmetry_generation_report.json
```

首次遇到旧格式数据时，会验证 HDF5、视频、scene 和 source 映射后幂等导入全局 manifest。

## 八、提交顺序

```text
生成 staging episode
→ 通用 HDF5/MP4 校验
→ 插件物理校验
→ 写完整 commit journal
→ 移动 HDF5/MP4/instruction/trajectory
→ 更新 scene_info
→ 更新全局 ownership manifest
→ 更新全局 attempt report
→ 执行 appender 管理的 legacy JSON writes
→ 最后删除 journal
```

任一步异常都保留 journal。普通 append 或显式 `--repair` 会按 journal 中明确列出的 destination 和旧 JSON 内容精确回滚。

## 九、实现新的增强插件

1. 复制 `scripts/augmentations/plugins/template.py`。
2. 设置唯一 `name`、递增的 `version`、支持的 tasks/configs。
3. 实现 `generate()`，只在传入的 staging 目录生成编号 0 的 bundle。
4. 返回 `GenerationResult`，只有完成任务级物理判定时才能返回 `success`。
5. 在 `validate_result()` 中加入任务特有的只读检查。
6. 在 `registry.py` 增加惰性 factory 元数据。
7. 升级代码时必须升级插件版本；框架同时把代码 SHA 写入 source fingerprint 和 manifest，防止新旧生成范式混用。

当前 source provider 明确限定为 RoboTwin clean `seed.txt` + episode 索引。如果未来增强不基于 seed，应把 source enumeration 抽象为 `SourceSpec` provider；不要在 appender 中伪造 seed。

伪代码：

```text
class NewPlugin(AugmentationPlugin):
    name = "new_plugin"
    version = "0.1.0"
    supported_tasks = ("target_task",)

    generate(staging, source_episode, source_seed, context):
        source_id = hash(source fingerprint + plugin version + code hash)
        rebuild clean source
        apply state/action-consistent augmentation
        replan or transform all required modalities
        execute in simulator

        if physical task failed:
            return planning_failure or collection_failure

        export bundle to staging/episode_0000000
        return success + scene_entry + metrics + provenance
```

## 十、已完成测试

- 默认无 flag 时 registry/plugin 不进入 `sys.modules`。
- 旧 CLI 参数解析和兼容 wrapper 可 resume。
- 通用 CLI/dry-run 识别已提交数据和下一编号。
- 旧 symmetry mapping/report 可幂等迁移。
- 两个插件共享 manifest 时编号依次为 50、51、52，无冲突。
- journal 覆盖 artifact、scene、global manifest、attempt 和 legacy JSON 的半提交恢复。
- strict dry-run 前后递归目录清单和所有文件 SHA 完全一致。
- `--repair` 在 target 未满足时不生成任何 episode。
- source snapshot、plugin version 和代码 SHA 防止来源或实现漂移。
- HDF5 双臂 joint/gripper/EE pose、四路 RGB、finite、长度与 MP4 可解码检查通过。

## 十一、注意事项

- 同一个池只能由已包含共享锁修改的采集入口写入；不要同时运行旧版未加锁 collector。
- `exception` 被视为可重试，不会永久占用 source；物理规划/采集失败作为终态 attempt 跳过。
- 若代码 SHA 或插件版本变化，旧 generic source snapshot 会明确报错。应保留旧代码完成原批次，或使用新的空数据池/新插件版本；不要手工删除 snapshot 强行混用。
- 规则要求输入为 clean 数据时，只能把对应 clean pool 的 seed/source 交给插件。
