# 全部录像和证据下载

Release：[results-2026-10-27](https://github.com/KQuaneo/cucu-lingbot-robotwin-2026/releases/tag/results-2026-10-27)。

|附件|文件大小|录像数|SHA256|
|---|---:|---:|---|
|[cucu-clean-evaluation-evidence.zip](https://github.com/KQuaneo/cucu-lingbot-robotwin-2026/releases/download/results-2026-10-27/cucu-clean-evaluation-evidence.zip)|462,988,728 B|1000|`ba4c02f76df2774d7f867cf445e2964e7b51c98beed521f78a657f05ae0c58fc`|
|[cucu-randomized-evaluation-evidence.zip](https://github.com/KQuaneo/cucu-lingbot-robotwin-2026/releases/download/results-2026-10-27/cucu-randomized-evaluation-evidence.zip)|215,319,975 B|500|`8a5f04d1729882c64cb2a86ba3a32670e955719c2b09e0d6d5d7b4a6c130d2df`|
|[cucu-training-code-reproduction.zip](https://github.com/KQuaneo/cucu-lingbot-robotwin-2026/releases/download/results-2026-10-27/cucu-training-code-reproduction.zip)|64,089,347 B|0|`3f68d6cf8a29661821d7bc8657361d766bd3cd6e310f5a883872025491982a1f`|
|[cucu-rl-experiments-evidence.zip](https://github.com/KQuaneo/cucu-lingbot-robotwin-2026/releases/download/results-2026-10-27/cucu-rl-experiments-evidence.zip)|459,983,170 B|128|`231689c105f73a2329a393210761bc3c376b40f222123a71bc49d6e8fc3bfe1b`|
|[cucu-development-experiments.zip](https://github.com/KQuaneo/cucu-lingbot-robotwin-2026/releases/download/results-2026-10-27/cucu-development-experiments.zip)|39,934,504 B|0|`c82025a212e76e0d8a2b5d0230360d1fd1fd4920372ea270ab7edfe10fcb5927`|

每个ZIP均包含PUBLICATION_MANIFEST.json，列出原始相对路径、公开路径、原文件SHA和脱敏后SHA。归档全部成员已SHA核验，ZIP CRC通过。

下载全部附件到同一目录后运行：

```bash
sha256sum -c SHA256SUMS
# macOS也可使用：shasum -a 256 -c SHA256SUMS
```

clean附件包含BF16与FP32各500段，randomized附件500段，RL附件128段（含逐chunk NPZ和小型网络参数）。其余两个附件为训练/复现和开发实验代码、配置与证据。

不含6k主模型或续训存档；13.3GB原提交ZIP因含大权重未上传。官方要求每任务100次，本队实际10次，结果如实保留。

GitHub普通Git文件100MiB限制，Release每附件小于2GiB。参考：[大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[Release说明](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。
