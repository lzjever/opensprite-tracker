# OpenSprite 资产调研与仿真适配

2026-10-31。官方来源：[BeijingDynamics/open_sprite](https://github.com/BeijingDynamics/open_sprite)，main 版本提交 `35afc2126fc31fa4cf21a2e6061c8bfa6c69726a`（2026-08-28）。完整调研副本在 `.reference/open_sprite`，运行所需模型及许可证在 `assets/opensprite/vendor`；训练与 retarget 不访问调研副本。

## 实际机器人契约

- Sprite0825 sanitized-v4：总质量 **14.651386 kg**，31 个转动关节，浮动根；MuJoCo `nq=38, nv=37`。约 0.95 m 高，V 形髋座。官方 external-PD XML 的根部初始 z=0.65 m 实际悬空约 0.188 m，不能当成落地默认姿态。
- 腿左右各 6，腰 **roll/yaw 两轴，无 pitch**，臂左右各 7，头 **pitch/roll/yaw 三轴**。31 DoF 总数恰好与 X2 相同，但身体与关节语义不同，不能拷贝 X2 索引、额定扭矩、PD、惯量或碰撞尺寸。
- 根体 `pelvis_link`，胸部 `waist_yaw_link`，脚末端左右 `ankle_roll_link`，手末端左右 `wrist_roll_link`，头末端 `head_yaw_link`。最终跟踪 **全部 32 个实体 body**（不含 world），由本地 Sprite MJCF 的真实 DFS 顺序读取，确保两腰、三头、三腕、髋肩各级链条都有完整位置、旋转与速度参考；不保留 X2 的 15-body 子集。
- 右髋 pitch、右膝均绕 -Y，左侧对应 +Y；两踝 pitch 均为 -Y。两侧膝屈曲值一正一负。V 型骨盆的 ±25° 固定变换已经包含在官方 MJCF 中，所有 FK/IK 应直接读模型。
- 官方 URDF 限位与 external-PD XML 一致：腿上部 effort 12 Nm、速度 10 rad/s；踝 6 Nm、12.56 rad/s；腰 roll 30 Nm、yaw 12 Nm，两者 3.8 rad/s；肩/肘/腕 yaw 3 Nm；腕 pitch/roll 与头全部 0.8 Nm；臂头速度均 3.8 rad/s。硬件踝为并联耦合机构，仿真用广义 pitch/roll 坐标，本工程不实现硬件电机映射。

## 独立资产生成

`scripts/build_opensprite_model.py` 从 repo 内 unmodified vendor XML 与 mesh 生成 `assets/opensprite/opensprite.xml`，不联网，不需要 `.reference`。保留官方质量、惯性张量、质心、关节轴、关节限制及 armature，逐值验证一致。只改：

1. timestep 5 ms、implicitfast；policy decimation 4，即 50 Hz。
2. 保留 mesh 作为 visual，不用于大量环境碰撞。每个非脚 body 生成 CAD AABB 内接 capsule；骨盆/胸部用 0.9 倍 AABB box。半径为最小 AABB 半长的 0.85 倍，最长轴端点取 maxhalf-radius。足部完整保留官方 104×44×10 mm sole box，底面距 ankle-roll 原点 39 mm。碰撞外壳是明确的计算近似，不声称与 CAD 完全一致。
3. 排除父子、祖孙关节壳体的接触，保留其他非局部身体接触与全身地面接触。neutral 姿态没有非局部自碰撞。
4. 几何落地根部高度为 z=0.46224576 m；训练重置依 retarget 轨迹。
5. 不在 XML 中添加 actuator。mjlab task 的外部 PD actuator 会添加严格 effort-cap motor。必须让 actuator `armature=None, frictionloss=None, viscous_damping=None`，不能用原 X2 的统一 0.03 armature/0.3 frictionloss 覆盖官方参数。

## 控制参数及实测

官方没有发布 tuned gains 与可识别的执行器动态。因此以下均为本工程仿真实现选择，不能称为官方硬件控制器：腿与踝 kp=60；腿 kd=2，踝 kd=1.5；腰 kp/kd=20/0.7；上臂与腕 yaw=8/0.3；腕 pitch/roll 和头=2/0.12。所有 torque 仍严格裁剪到官方 effort。肩 roll 默认左右 ±0.12 rad，其他默认零。**用户明确要求与 X2 相同的 command delay 与 torque lag**，因此最终使用 0–1 个 5 ms tick 命令延迟（20 ms cadence，hold 概率 0.9）、5–10 ms 时间常数的力矩迟滞，31 个关节独立参数；这是用户指定的仿真执行器模型，不声称是 Sprite 已识别的硬件动态。之前考虑的 0-lag 官方控制模式只用于诊断。

早期低 ankle kp=16、leg kp=30 时零位 PD 站立约 2 s 向前倒；14.65 kg 的倒立摆重力刚度大于这组脚踝弹簧刚度。这已通过调节仿真增益纠正，没有放大额定扭矩。`scripts/validate_opensprite_model.py --seconds 60` 加入地面，以真实 effort 限制和 **0 lag 官方控制模式** 验证：

- 所有物理参数与 vendor 逐值一致。
- 60 s 全过程状态有限，pelvis 最低/最终 z=0.4594/0.4604 m。
- 最大实际 effort/额定 effort=0.416，无 nominal 非局部自碰撞。

用户指定范围最差 τ 的验证 `--lag-seconds 0.01 --seconds 60` 同样通过：最低/最终 z=0.4592/0.4604 m，最大 effort 比 0.446。这个 CPU PD 测试包含 10 ms torque lag，但不包含随机 0–1 tick command queue；完整 delay+lag 的实际 mjlab 步进/恢复由训练 smoke 验证。把 assets、robot.py 和 validator 独立复制到随机临时目录后运行，无 `.reference` 和相邻源码依赖，物理一致性与 10 s 站立也通过。

这证明模型和控制参数可以站立，并不证明所有 retarget 动作动态可行；舞蹈与起身的真实接触/力矩边界必须由动作 QA 和策略训练后的完整回放继续检查。不得用质量缩放、加大 torque 或关闭所有身体接触掩盖失败。

## 与 tracker / retarget 的交叉质询

- **31 维网络能直接复用吗？** 物理状态维度可复用，关节/部位语义必须来自本模型；头必须包含三轴，手臂范围从 index14 开始，不继承 X2 两轴头和三轴腰。
- **能沿用 Capsule7 足奖励吗？** 不能。本足是官方单 box。SuperMimic 六项追踪奖励不需要 X2 脚载荷/sole-normal 等专用奖励；相关旧 helper 不得在 import 或训练中强制读取 foot1..7。
- **可否把同一 MJCF 用于 IK 与训练？** 是。两者使用 `assets/opensprite/opensprite.xml`；所有 source trajectories 的 FK 和 body ordering 保持一致。
- **执行器分组是否沿用 X2？** 不沿用。31 个独立关节，group 数为 31；原先为了保留 X2 label 宽度拼接腰与头的 12 分组已移除。世界模型读取当前机器人的实际参数上下文，而不是为其他机器人固定 label 维度。
- **相同 31 关节是否证明 flat141 数值适用？** flat141 的物理/执行器记忆维度可由本机器人 31 关节重新计算，但不能以维度相同保留 X2 关节语义。已对 20 个随机 Sprite 位姿和速度比较可微 FK 与 MuJoCo：最大位置/旋转误差约 1e-15，速度与 1e-7 s 积分差分误差小于 2e-6。

## 真正 UMR 的新机器人配置

用户明确要求原生 UMR；此前任务空间 IK 只保留为失败诊断 baseline，不作为最终 retarget。`configs/umr/opensprite.json` 定义新 `opensprite0825` 模型，使用 repo-owned 原生 UMR `build_robot_sample` 对 **全部 32 个官方 STL mesh** 做外表面采样，重训 `geno` 与 `opensprite0825` 两个点云的 template-residual correspondence AE；不读取任何 X2 mesh、slot 或 correspondence 权重。

中心选 Geno `Hips` 和 Sprite `pelvis_link`，而不是继承 X2 的腰中心或 G1 的 bbox 高度比。机器人到人体 T-pose 坐标为 `(x_human,y_human,z_human)=(y_robot,z_robot,x_robot)`，逆矩阵在配置中明确给出；矩阵正交、行列式 +1。

T-pose 只令 shoulder-roll 左 +π/2、右 -π/2，其余全部为官方零位。MuJoCo 实测相对 pelvis：肩 `(0, ±0.1225, 0.300)` m、肘 `(0.020, ±0.33375, 0.3015)` m、腕末端 `(0.0255, ±0.51975, 0.289)` m、踝 `(0.015, ±0.08709, -0.42325)` m。左右臂接近水平且向人体左右伸展，所有关节在官方限制内；不使用求解位置式 T-pose 导致的 null-space 分支。

解剖语义审查按新拓扑逐级确认：pelvis 对 Hips，腰 roll/yaw 链和胸部对 Spine，肩 pitch/roll/yaw 和肘对 Arm/ForeArm，三腕与手末端对 Hand，三轴头链对 Head，髋三轴/膝/踝两轴对 UpLeg/Leg/Foot。真正 UMR 的稠密对应由 Geno skin 和 Sprite 官方外表面训练获得，此语义说明用于左右/部位 QA，不能替换原生 surface correspondence 算法。

`scripts/validate_opensprite_umr.py --build-sample` 使用独立 `.venv-umr` 校验 T-pose、32body 与全部官方 visual mesh、native UMR 的 4096 点采样以及按真实 mesh face 反查的每 body 可见采样数。碰撞约束使用本机器人从官方 CAD 派生的 primitive 模型，与最终训练相同，表面对应始终采样完整官方 mesh，而非用碰撞 capsule 代替人体/机器人表面。

原生 `mujoco_collision` 地面硬约束要求场景中真的存在 plane。UMR 使用 `assets/opensprite/retarget_scene.xml`，它只 include 裸机器人模型并添加 z=0 plane，所有机器人质量、惯量、关节及 body 顺序逐值保持；plane 不参与表面采样。配置选择原生支持的该 ground mode，避免 `surface_slots_all` 的稀疏优化子集漏掉真实脚角/手端。没有为此改写 UMR 算法。

最新 plain-MLP SuperMimic 移植使用 `opensprite_tracker/supermimic/`。机器人观测使用 waist-yaw 坐标系的 **虚拟 IMU**（orientation identity），因为官方未给硬件 IMU site；这不是沿用不存在的 `imu_in_torso`。新观测 105 维、目标 77 维、raw context 221 维、31 关节 memory 66 维；这些都由 Sprite 自由度重算。转移指标为 97 维（31 speed、31 requested torque、31 applied torque、2 mean load、2 peak load），full-body impact 增加 1 维。真实四个 5 ms 子步提供监督；想象过程使用原训练逻辑的双头保守代价，不从 kinematic pose 伪造接触力。

许可证：官方模型 AGPL-3.0 完整文本在 `assets/opensprite/vendor/LICENSE`，公开来源未包含可直接使用的官方 V38 策略或训练代码。
