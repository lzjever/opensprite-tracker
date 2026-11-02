# 独立复现

[返回主README](../README.md)

当前数据为单条 `dance2_subject1`，225.66秒。以下命令在仓库根目录执行。代码、原始BVH、Geno、机器人、对应资产和示例产物都在仓库内；Python依赖仍需按requirements安装。

## 环境

Linux、Python3.12、CUDA12.8；训练使用支持CUDA的GPU。准备Git LFS、Node.js≥18、FFmpeg和EGL运行库。Ubuntu可安装 `git-lfs nodejs ffmpeg libegl1 libgl1 fonts-noto-cjk`；发行版Node过旧时另装新版。

```bash
git lfs install --local
git lfs pull
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv venv --python 3.12 .venv-umr
uv pip install --python .venv-umr/bin/python -r requirements-umr.txt
.venv/bin/python scripts/demo.py check --profile all
```

两个环境分别用于训练/视频和原生UMR，NumPy、MuJoCo版本不同。Node解析器及所需Three.js文件已经包含，不需要外部项目或全局npm包。检查命令验证文件、依赖与CPU接口，不执行GPU仿真。

## 原始动作到训练动作

原始数据位于 `data/sources/lafan1-resolved/`，人体为 `assets/geno/Geno.fbx`，机器人为 `assets/opensprite/`，已有对应关系位于 `assets/umr_correspondence/geno_opensprite0825/`。

复用已有对应关系，对完整原始BVH运行原生UMR优化并转换：

```bash
.venv/bin/python scripts/demo.py retarget --motions dance2_subject1
.venv/bin/python scripts/demo.py build-motion
```

输出为 `outputs/rebuild-dance/native/dance2_subject1/native.npz`，单动作及打包库在 `outputs/rebuild-dance/motions/`。训练库为其中的 `lafan_dance.npz`。

从表面采样、对应网络训练、原生优化到转换全部重建：

```bash
.venv/bin/python scripts/demo.py retarget --stage all --motions dance2_subject1 \
  --correspondence-dir outputs/rebuild-dance/new-correspondence
```

对应训练默认CPU；需要使用GPU时显式增加 `--correspondence-device cuda`。原生轨迹优化使用CPU。完整重建耗时独立于tracker八小时预算。新训练对应关系存在随机性，不承诺与随仓库提供的结果逐位一致。

分阶段可用 `--stage build`、`--stage train` 和 `--stage retarget`，三阶段保持相同的 `--correspondence-dir`、输出和cache参数。仅做解析/优化小检查可传 `--max-frames 3`，该产物不能当完整舞蹈训练库；正式转换要求完整源时间轴。

## 训练

使用随仓库的动作库，从零训练：

```bash
.venv/bin/python scripts/demo.py train --output outputs/my-dance --hours 8

# 独立的传感器Actor配方，重新冷启动八小时
.venv/bin/python scripts/demo.py train --preset goal-sim2real \
  --output outputs/my-dance-sim2real --hours 8
```

使用自己重新retarget的动作库：

```bash
.venv/bin/python scripts/demo.py train --output outputs/my-retarget-dance --hours 8 \
  --motion outputs/rebuild-dance/motions/lafan_dance.npz
```

默认`--preset goal-sim`使用`configs/demo/dance-8h.json`，`goal-sim2real`使用`configs/demo/dance-goal-sim2real-8h.json`。命令生成独立`demo-training.json`，记录当前Git提交，由monitor维护累计预算和故障恢复。目录必须全新；新配方不继承旧模型、优化器或账本。正式启动及恢复要求HEAD匹配配置中的`source_commit`，且已跟踪源码无未提交修改；`outputs/`中的产物不影响检查。修改后先提交，再生成正式配置；`--dry-run`无需干净源码。

新Actor为99维传感器观测，Critic仍可使用全局特权且主网络与Actor同构。WM为训练专用。两种配方的物理、数据、奖励及高度/重力失败门槛保持一致，详见[配方说明](plan/goal-sim2real.md)。八小时含设置、编译及预热，soft28620秒、hard28800秒，预留180秒停止保存。启动前GPU验收若属于同一训练任务，应计入该任务的累计账本，不能重新获得八小时。正常恢复不刷新时钟；monitor恢复同一run时可再次运行：

```bash
.venv/bin/python scripts/monitor_training.py \
  --config outputs/my-dance/demo-training.json
```

恢复前确认该run的monitor和trainer均已退出。checkpoint恢复模型、优化器和统计；replay、RNG与仿真状态重建，不保证逐位续跑。`metrics.csv`、`supervisor.json`和checkpoint位于训练输出目录。

CLI每小时审查为可选工具，新配方使用[独立审查任务](plan/hourly-audit-goal-sim2real-8h-task.md)，从传入配置读取run和预算。常规审查使用CPU，必要的GPU评估串行安排。最终对clip0、seed42/43/44各N3请求完整源轨迹，完整录像与成功率分别报告。

## 评估和完整录像

新录像对应`goal-sim2real / model24000`。使用同一快照重新评估或录制时，显式指定checkpoint：

```bash
.venv/bin/python scripts/demo.py evaluate \
  --checkpoint artifacts/checkpoints/goal-sim2real_model_024000.pt \
  --num-envs 3 --seed 42 --output outputs/my-sim2real-evaluation
.venv/bin/python scripts/demo.py record \
  --checkpoint artifacts/checkpoints/goal-sim2real_model_024000.pt \
  --num-envs 1 --seed 42 --output outputs/my-sim2real-recording
```

默认CLI未变：不传`--checkpoint`时，评估仍为`goal-sim/model30000`、seed42、N3；录像仍为`goal-sim/model18000`、seed42、N1。也可指定自己的模型路径。模型接口与执行器模式从checkpoint读取。

新[完整MP4](../artifacts/videos/goal-sim2real-model024000-full-1080p50.mp4)、[同一快照](../artifacts/checkpoints/goal-sim2real_model_024000.pt)和[真实rollout](../artifacts/recordings/goal-sim2real-model024000/rollout.npz)均随仓库提供。[预览GIF](media/goal-sim2real-model024000-preview.gif)拼接0–4、192–196、221–225秒三个片段，[静态海报](media/goal-sim2real-model024000-poster.png)可单独查看。model24000是首次检查这次录像请求时选定的最新已保存中间快照，不是八小时最终checkpoint，也不表示当前训练仍以它为最新版本。原[B站视频](https://b23.tv/aMs7Vw3)对应`goal-sim/model18000`，不是新录像。

评估从第一帧请求全源，失败后仍以首次失败计分；录像在失败后继续真实控制至源结尾，并在HUD标注。新录像首次失败193.76秒，此后同一策略继续至225.66秒，无重置或参考循环；完整视频不代表完整跟踪成功。

已有rollout无需重新执行策略或物理，直接生成视频：

```bash
MUJOCO_GL=egl .venv/bin/python -m opensprite_tracker.supermimic.render \
  --rollout artifacts/recordings/goal-sim2real-model024000/rollout.npz \
  --model assets/opensprite/opensprite.xml \
  --output outputs/my-rerender --env-index 0 \
  --width 1920 --height 1080 --fps 50 \
  --include-post-failure --failure-hold-seconds 0
```

显式指定仓库内模型可覆盖录像元数据中的机器路径。地面为格子，蓝色半透明机器人表示参考，HUD包含源时间、误差、力矩/速度和失败状态。新录像为1080p、50fps、11284帧，帧状态覆盖0–225.66秒；末状态也占一帧，容器时长为225.68秒。关节effort限幅与速度软约束的边界见[硬件限制核验](hardware-limits.md)；这些仿真结果不构成实机验证。

## 已执行的独立性检查

隔离副本已完成CPU模型加载、动作库/FK检查，以及只使用随仓库素材的Node人体解析和3帧原生UMR优化。单动作重新打包与提供的动作库逐字段一致。CLI命令完成dry-run及针对性CPU测试。未在全新机器上重新跑完对应训练、八小时tracker训练和完整GPU录像；硬件、驱动与随机性可能影响结果。
