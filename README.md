# OpenSprite 三动作统一 Tracker

一个共享的 SuperMimic actor、critic 和世界模型，同时学习三条完整 OrangeDuck LAFAN 动作。网络与训练主干直接参考本机 `beyondmimic-umr` 已成功的最新 SuperMimic 分支，按 OpenSprite 重新适配。机器人为官方 Sprite0825，31 个驱动关节；仿真为 MuJoCo Warp/mjlab，物理 200 Hz、策略 50 Hz。

| 动作 | 源数据时长 | 用途 |
| --- | ---: | --- |
| `dance2_subject1` | 225.69 秒 | 长舞蹈 |
| `fallAndGetUp3_subject1` | 102.19 秒 | 历史测试的倒地起身 |
| `walk4_subject1` | 163.92 秒 | 较慢步行 |

计划、选择依据、算法差异及交叉审查见 [docs/plan](docs/plan/README.md)。训练和结果记录见 [training-review](docs/plan/training-review.md)。

## 安装与数据

需要 Linux、兼容 CUDA 12.8 的 NVIDIA GPU 和 Python 3.12。所有项目代码、资产及选定源数据均在仓库内；`.reference/` 是可移除的调研材料。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
PYTHONPATH=. .venv/bin/python scripts/validate_opensprite_model.py --seconds 60
PYTHONPATH=. .venv/bin/python -m pytest -q tests
```

UMR 使用独立数值环境；FBX 解析需要 Node.js 18 或更新版本。公开版保留原始源数据；Geno/OpenSprite correspondence 和正式三动作库须在本地生成。重建命令：

```bash
uv venv --python 3.12 .venv-umr
uv pip install --python .venv-umr/bin/python -r requirements-umr.txt
node --version
PYTHONPATH=. .venv/bin/python scripts/build_opensprite_model.py
PYTHONPATH=. .venv-umr/bin/python scripts/retarget_lafan.py --stage all
PYTHONPATH=. .venv/bin/python scripts/inspect_retarget_geometry.py \
  data/motions/dance2_subject1.npz \
  data/motions/fallAndGetUp3_subject1.npz \
  data/motions/walk4_subject1.npz --render --self-collision
```

正式 retarget 运行仓库内的原生 UMR NR 蒙皮表面优化器，使用 OpenSprite 32 个官方网格和全新 Geno/OpenSprite 4096-slot correspondence；不沿用 X2 的 correspondence、姿态、关节限位或机器人参数。`--stage build/train/retarget/convert` 可分别运行各阶段；已存在兼容 slots 时直接复用，retarget 总是按当前配置重新计算。原始 BVH 不改，完整保留三源时轴，逐源 FK/速度后合库。历史 IK 原型已隔离在忽略的诊断目录，不进入正式训练。

官方位置/速度约束及最终地面检查通过；实际小幅 root-Z 补偿最大约 13–16 mm。原生 self-penetration 约束有 slack，舞蹈少量帧仍有 wrist/pelvis 穿插（最深 43.7 mm，超过 30 mm 的帧占 0.124%），因此三条均保留 `needs_review` 标签并如实记录几何残差，不宣称完美物理可行。详见 `data/motions/*.json`。

## 训练、监控和评估

正式配置与确切命令保存在 [configs/training.json](configs/training.json)。 当前用户已授权按官方论文规模重新训练：WM2048×3、策略2048/1024/512/256、4096环境、WM10×2048窗口、策略4096轨迹、replay1000步/环境、TF32。16GB适配采用CPU replay与完整逻辑batch内的梯度累积；详见[重新训练方案](docs/plan/paper-scale-restart.md)。本轮重新获得独立12小时预算，旧轮结果保留。本轮初始化/编译、GPU训练验证及正式训练/恢复共用新授权的12小时单卡预算，恢复不得刷新。完整 checkpoint 保存actor/critic/世界模型、双成本预测head、全部优化器、采样状态和累计时钟；恢复不能重置预算。

训练入口支持 `--resume <checkpoint>`；监控入口在 tmux 中启动单个子任务，每小时保存 TensorBoard/GPU 遥测、在同配置异常退出后从最近完整快照恢复，并对累计时间设上限：

```bash
.venv/bin/python scripts/monitor_training.py --config configs/training.json
```

SIGTERM/SIGINT 在完整训练 outer 边界保存退出。监控只控制它创建的进程组。查看 `outputs/<run>/supervisor.json`、`train-segment-*.log`、`reviews/` 和 TensorBoard；不修改或停止其他项目。

小时智能审计按用户要求使用 CLI 后台进程，每 3600 秒串行调用一次 `codex exec`，立即执行首审。完整授权、预算核算和最终交付流程见 [hourly-audit-task.md](docs/plan/hourly-audit-task.md)。在独立 tmux 会话中运行：

```bash
tmux new-session -d -s opensprite-hourly-audit '.venv/bin/python scripts/hourly_audit.py --config configs/training.json'
```

审计有独占锁、执行日志与超时记录，API 故障不停止训练；不使用系统 service、timer、cron 或 Desktop。查看运行目录 `hourly/runner-state.json` 的状态/下次时间和 `hourly/latest.md` 的实际结论。需要机器持续开机及后台进程存活，重启后手动重启脚本。

训练结束后，任务负责同一共享检查点的三动作×三 seed 完整请求评估和 ghost 视频，整理到 `outputs/delivery/opensprite_lafan3_tracker/`，披露真实未通过项；runner 核实交付完整且 trainer 停止后自动退出。

逐动作评估同一 checkpoint，默认从每条第一帧运行至真实结尾，无自动 reset：

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_opensprite.py \
  --checkpoint outputs/<run>/model_XXXXXX.pt \
  --clip 0 --num-envs 3 --seed 42 \
  --output outputs/<run>/eval/clip0-seed42
```

对clip1/2和seed43/44同样执行。评估目录包含`summary.json`及实际rollout；`python -m opensprite_tracker.supermimic.render --help`提供reference ghost视频入口。

主动截短只用于诊断，不能作为全程验收。正式参考分支使用相对姿态追踪配方，评估逐动作完整完成率、身体/关节/末端误差、全局漂移、最差片段和真实接触/力矩/速度峰值；完整物理成功与质量分开呈现。不能用平均回报或窗口成功代替整条长动作表现。

## 来源和许可

模型：[BeijingDynamics/open_sprite](https://github.com/BeijingDynamics/open_sprite)，AGPL-3.0。训练算法移植自最新 X2 SuperMimic 分支，保留第三方 notices；未沿用 X2 的关节轴、惯量或扭矩。LAFAN 来源及非商业研究条款保存在 `data/sources/lafan1-resolved/README.md` 和 `license.txt`。代码/资产许可与动作数据许可分别适用，详见 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 公开分发范围

LAFAN原始数据的CC BY-NC-ND 4.0许可包含禁止分发衍生材料的条件。公开版保留未修改的原始源文件、许可证与归属说明，预计算重定向动作、采样/对应关系及其模型权重不进入公开Git历史。使用上文的原生UMR命令可在本仓库内生成这些本地研究资产，无需其他项目checkout。OpenSprite官方资产与第三方代码分别遵循仓库内保留的许可证。

公开main使用独立的代码快照，避免旧历史带入受限衍生数据；本地研究历史、正在运行的训练及固定训练分支保持原样。
