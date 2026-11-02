# OpenSprite Demo Tracker

- 当前训练仅含 `dance2_subject1`：50Hz，11284状态、11283转移、225.66秒。使用仓库内 `data/motions/lafan_dance.npz`。
- 本仓库必须独立运行。代码、机器人、原始动作、人体资产、表面对应和配置由仓库提供。`.reference/` 和 `outputs/` 不参与发布，不作为运行依赖。
- UMR使用OpenSprite自己的31关节、32 link、表面采样、对应关系和约束。以官方机器人轴、惯量、限位和effort为准，保留第三方许可。
- 两套独立可选配方：`goal-sim`保留105维Actor及critic256/128；`goal-sim2real`使用99维Actor，去掉依赖实际世界heading的rel6，保留关节位置/速度、IMU gyro/重力投影和前次动作。770维reference-only目标不变。Critic允许全局特权，使用与Actor同宽的编码器/解码器及latent32，参数独立；WM训练专用可用真值。
- 两种配方的actor编码器/解码器各1024/512/256、latent32；WM2048×3；768env；WM4×1024/K8、actor512/H32、critic5×4。先500个outer仅训练WM和成本head，再联合训练。
- 一阶力矩迟滞关闭，命令延迟保留0/1个5ms tick、hold .9、每4tick刷新。控制50Hz，物理200Hz。评估读取checkpoint中的执行器模式。
- 已有`goal-sim`运行目录为`outputs/train/opensprite-dance-no-torque-lag-cold-8h-20261101`。新的`goal-sim2real`必须使用独立目录、配置和冷启动账本；从本次传入配置读取run。八小时累计预算hard28800秒、soft28620秒，包含启动前GPU验收、编译、预热和恢复；普通resume不刷新。固定训练分支不可移动；有`source_commit`的配置在启动和恢复前校验HEAD及已跟踪源码。
- 训练保持当前物理、数据、奖励及判定，包括参考高度误差0.17m与gravity平方误差1.6的hard/soft失败边界。去除Actor观测特权不等于删除训练/评估真值判定。合理身体触地应被保留。模型和短窗误差有限不等于完整舞蹈成功。
- 每小时审查沿用CLI后台runner；`goal-sim`使用`docs/plan/hourly-audit-dance-8h-task.md`，`goal-sim2real`使用`docs/plan/hourly-audit-goal-sim2real-8h-task.md`。常规审查仅CPU，按需GPU评估串行安排。不创建系统服务、timer或cron，不启动第二个trainer。明确故障先保留现场，再在原预算内恢复。
- 完整评估从源首帧开始，不重置、不循环参考。录像允许失败后继续真实策略和物理，保留首次失败和未评分标记。
- 用户授权向公开仓库提交本demo代码、所选原始动作、资产、快照和视频。发布说明仅描述当前单舞蹈配方，不披露内部参考项目和历史实验。保留必要版权和许可证。
- 文档简练、用陈述句；中文主README链接英文版。发布前检查文件完整性、相对路径、LFS和独立复现入口。不要更改运行中的训练配置来满足文档。
