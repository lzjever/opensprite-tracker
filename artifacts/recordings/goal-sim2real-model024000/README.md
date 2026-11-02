# goal-sim2real 24k 完整舞蹈录像

轨迹、视频与 `artifacts/checkpoints/goal-sim2real_model_024000.pt` 对应。Actor使用99维本体观测，保留IMU重力投影；本包与旧`goal-sim`示例分开。

单实例、seed 42，50 Hz，共11283次真实仿真转移、11284个状态。首次失败为193.76秒，标准成功率为0；之后真实策略与物理继续运行，不重置、不循环参考、不添加停帧，直至源末尾225.66秒。失败后的状态不计入标准评分。包含首尾状态的视频为225.68秒，1920×1080、50 fps。

视频：[`goal-sim2real-model024000-full-1080p50.mp4`](../../videos/goal-sim2real-model024000-full-1080p50.mp4)。`preview.png`、`qa-postfailure.png`、`qa-last.png`分别展示开场、失败后和末帧。

`rollout.npz`的78个非元数据数组与原记录逐值、dtype和shape一致，仅`metadata_json`的机器路径改为仓库相对路径。发布checkpoint也只重定位路径元数据，模型、优化器、归一化和采样张量及训练数值保持不变；文件哈希因此与原快照不同。

`verification.json`的80项检查及原始哈希属于原始录像任务。发布副本的路径重定位、张量和数组等价另行核验，见[发布等价证明](publication.json)；不表示重新执行80项检查或完整舞蹈成功。

从仓库根目录重新渲染已有轨迹（需要可用渲染环境，输出目录必须尚不存在）：

```bash
.venv/bin/python -m opensprite_tracker.supermimic.render \
  --rollout artifacts/recordings/goal-sim2real-model024000/rollout.npz \
  --output outputs/rerender-goal-sim2real-model024000 \
  --model assets/opensprite/opensprite.xml \
  --env-index 0 --width 1920 --height 1080 --fps 50 \
  --include-post-failure
```

此命令只渲染保存的完整状态序列，不重新运行策略或物理。
