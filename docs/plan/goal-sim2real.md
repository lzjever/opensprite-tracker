# 单舞蹈 goal-sim2real 配方

`goal-sim2real`是独立可选的Actor传感器输入配方。默认`goal-sim`及已有快照继续可用。两者仅训练`dance2_subject1`：50Hz、11284状态、11283转移、225.66秒。

| 项目 | goal-sim | goal-sim2real |
|---|---|---|
| Actor单帧观测 | 105维，含依赖实际heading的rel6 | 99维，移除rel6 |
| Actor保留输入 | 关节位置/速度、IMU gyro/重力投影、前次动作 | 相同 |
| Actor参考 | 770维局部参考特征 | 相同，仅依赖参考，可离线生成 |
| Actor主网络 | 编码器1024/512/256→32 tanh；解码器1024/512/256→31；ELU | 相同；10帧历史缩为990维，解码器输入1022维 |
| Critic | 1268→256→128→1 | 770维参考经1024/512/256→32 tanh；498维特权特征与latent拼接，经1024/512/256→1；ELU |
| Critic/WM真值 | 训练专用，可用全局状态、执行器记忆和物理参数 | 相同；Actor和Critic权重各自独立 |
| 奖励和失败判定 | 全局跟踪奖励；参考高度误差0.17m、gravity平方误差1.6 | 相同，包括hard termination和soft survival |
| 启动模板 | `configs/demo/dance-8h.json` | `configs/demo/dance-goal-sim2real-8h.json` |

该配方限定Actor的实际观测来源，并使Critic主结构与Actor一致。Critic允许全局特权；WM是训练模型。全局位置和世界姿态不输入新Actor，训练中的奖励、价值估计及参考失稳判定仍可使用真值。高度/重力门槛不因接口变化而放宽；匹配参考的躺倒、翻转及合理身体触地仍保留。

实际部署需要关节编码器与速度估计、IMU gyro、可靠的重力方向估计，以及一致的IMU外参、单位、轴向、默认关节零位、历史窗口和前次动作约定。动态加速度不能直接等同重力；仿真中可读取理想重力投影，实机需经过标定与估计。此接口不需要实际世界heading或全局位置，但尚未验证实机跟踪能力。

其余配方保持768env、WM2048×3、WM4×1024/K8、Actor512/H32、Critic5×4、500outer仅WM及成本head预热。控制50Hz、物理200Hz；关闭一阶力矩迟滞，保留0/1个5ms tick命令延迟、hold .9及每4tick刷新。评估从checkpoint读取接口及执行器模式，旧checkpoint缺新标识时使用原接口。

```bash
.venv/bin/python scripts/demo.py train --preset goal-sim2real --dry-run
.venv/bin/python scripts/demo.py train --preset goal-sim2real \
  --output outputs/my-dance-sim2real --hours 8
```

正式启动使用干净的已提交源码，并记录`source_commit`；启动和恢复校验HEAD一致。每次新训练有独立配置、输出与账本，从零初始化所有网络、优化器、归一化和采样状态。八小时soft28620秒、hard28800秒，启动前GPU验收、编译、预热和恢复均计入同一任务预算；普通resume不刷新。源码固定后不得移动训练分支。

验收覆盖传感器输入的平移及世界Z旋转不变性、旧模式回归、checkpoint接口拒绝错配、Critic归一化/梯度及固定λ目标。正式训练前还需生产尺寸的真实joint update和checkpoint恢复检查，记录显存、耗时与有限梯度。CPU测试或短窗误差不能证明完整舞蹈成功。

完整评估从源首帧开始，不重置、不循环，clip0使用seed42/43/44各N3。首次失败保持，失败后继续录像的状态不进入标准评分。运行到最后一帧、八小时预算结束及交付完整均不等于质量通过。每小时审查使用[独立CLI任务](hourly-audit-goal-sim2real-8h-task.md)。
