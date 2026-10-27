# 下载全部录像和证据

[Release下载页](https://github.com/KQuaneo/cucu-lingbot-robotwin-2026/releases/tag/results-2026-10-27)。为绕开当前大请求上传限速，三个视频较多的ZIP按32MiB字节分块上传，另外两个ZIP直接上传。重组后与原验证ZIP逐字节相同，不重新编码或删减录像。

## 一次下载并重组

在仓库根目录执行（需要GitHub CLI和可用的Python3）：

```bash
gh release download results-2026-10-27 --repo KQuaneo/cucu-lingbot-robotwin-2026 --dir evidence-downloads
python3 scripts/restore_release_archives.py --directory evidence-downloads
```

也可从Release页手工把所有`.partNNN`、两个直接ZIP和元数据下载到同一文件夹，再运行重组脚本。脚本逐块校验SHA256，并校验最终ZIP；不会覆盖不匹配的现有ZIP或自动执行/解压其中代码。

|原始证据包|总字节|传输文件数|录像数|原始ZIP SHA256|
|---|---:|---:|---:|---|
|cucu-clean-evaluation-evidence.zip|462,988,728|14|1000|`ba4c02f76df2774d7f867cf445e2964e7b51c98beed521f78a657f05ae0c58fc`|
|cucu-randomized-evaluation-evidence.zip|215,319,975|7|500|`8a5f04d1729882c64cb2a86ba3a32670e955719c2b09e0d6d5d7b4a6c130d2df`|
|cucu-training-code-reproduction.zip|64,089,347|1|0|`3f68d6cf8a29661821d7bc8657361d766bd3cd6e310f5a883872025491982a1f`|
|cucu-rl-experiments-evidence.zip|459,983,170|14|128|`231689c105f73a2329a393210761bc3c376b40f222123a71bc49d6e8fc3bfe1b`|
|cucu-development-experiments.zip|39,934,504|1|0|`c82025a212e76e0d8a2b5d0230360d1fd1fd4920372ea270ab7edfe10fcb5927`|

`manifests/release_assets.json`含每块的下载URL、长度和SHA；`SHA256SUMS`核验下载的块和直接ZIP。

```bash
cd evidence-downloads
sha256sum -c SHA256SUMS
# macOS: shasum -a 256 -c SHA256SUMS
```

每个重组后的ZIP含PUBLICATION_MANIFEST.json，列出来源路径、公开路径、原文件SHA与脱敏后SHA。全部ZIP成员SHA及CRC已在发布前核验。

共1628段：clean BF16/FP32各500，randomized500，RL128。大模型权重、续训存档与13.3GB原提交包不上传。

GitHub附件限制参考：[Release说明](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。本次分块是上传通道兼容措施，并非原ZIP超出GitHub限制。
