# 完整舞蹈录像的原始轨迹

这份轨迹来自 `artifacts/checkpoints/model_018000.pt`，与附带的完整舞蹈视频对应，不是最新的 30k 快照。

单个实例、seed 42，50 Hz，共 11283 次真实仿真转移和 11284 个状态。首次失败发生在 193.76 秒；之后策略和物理继续运行，不重置、不冻结，直到源动作末尾 225.66 秒。标准成功率仍为 0，失败后的片段不计入标准追踪评分。包含首尾状态的视频为 225.68 秒。

`rollout.npz` 的 78 个轨迹、指标和名称数组与原记录完全一致，仅 `metadata_json` 的路径改成仓库相对路径。三个 JSON 信息文件也只做了路径整理，系统字体记录保留字体文件名。

从仓库根目录重新渲染已有轨迹（不会重新运行策略或物理；需要可用的渲染环境）：

```bash
.venv/bin/python -m opensprite_tracker.supermimic.render \
  --rollout artifacts/recordings/dance-model018000/rollout.npz \
  --output outputs/rerender-dance-model018000 \
  --model assets/opensprite/opensprite.xml \
  --env-index 0 --width 1920 --height 1080 --fps 50 \
  --include-post-failure
```

输出目录必须尚不存在。元数据会自动选择完整记录，包括首次失败之后的真实状态；不会添加失败停帧。
