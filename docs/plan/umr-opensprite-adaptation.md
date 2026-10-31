# OpenSprite UMR 严格适配

2026-10-31，用户再次明确：UMR retarget 和训练都必须适配全新型号，X2 只是算法参考。

## 修正当前实现范围

最初原型仅移植了 YY 的 BVH 输入/FK、重采样和部分后处理，求解器为任务空间 IK。它不等于 UMR，不能作为正式 retarget 交付；已停止这条原型的全量迭代，保留作诊断对照。正式训练尚未启动。

## 正式流程

1. 复制 YY 中官方 UMR 最小运行代码与许可证到本 repo，不在运行时引用 `.reference/` 或外部 `.runtime`。
2. 使用 OrangeDuck 的 `Geno.fbx` 原生蒙皮和精确 BVH 全通道输入，沿用 UMR NR FBX adapter，不套用不匹配的 LaFAN SMPL-X 模板。
3. 根据 OpenSprite 官方完整 STL/URDF/MJCF 新建 robot config、合法 Tpose、外表面点云、point-cloud center 与 root 偏移。不要因为 X2 用 waist_yaw 就照抄它。
4. 为 Geno 与 OpenSprite 两个模板重新学习 surface correspondence。不得加载 X2 slots、对应网络或其网格采样缓存。GPU correspondence 训练也计入总 12 小时训练预算。
5. 沿用 UMR NR surface-vector optimizer，重新定义本机器人实际硬限位、速度边界、完整碰撞壳与 ground/self-penetration 约束。保留原算法，机器人适配通过本地配置实现；确需更改核心计算时先说明原因并验证。
6. 先做独立短段与代表性倒地/起身测试，再完整求解三条源。检查身体部位 correspondence 是否串侧，手支撑/脚底是否正确，腰两轴和头三轴是否受控。
7. 训练入口以新任务/数据 revision 从头开始，不能续用 IK 原型或旧机器人语义的检查点。

## 新机器人检查表

| 项目 | OpenSprite 依据 |
| --- | --- |
| 型号 | Sprite0825，约0.95m、14.65kg |
| 关节 | 腿12、腰2、双臂14、头3；31受控关节及32link frames |
| 运动学 | 官方关节轴与固定变换，含镜像膝、V形髋，随机姿态/速度对MuJoCo独立验证 |
| 动力学/边界 | 官方质量、惯量、armature、effort/velocity/hard limits，不能取X2数值 |
| 执行器 | 用户随后明确要求与X2相同命令延迟及一阶力矩迟滞：沿用其参数与tick时序，映射到OpenSprite逐关节状态/context及官方effort；原零延迟提案作废 |
| 接触 | 官方足底尺寸及本机器人网格生成primitive，完整身体地面接触，局部机械排除单独检查 |
| 跟踪 | 全32body，腰2和头3显式参与；状态/输出尺寸由实际模型导出 |
| 奖励/终止 | Sprite体型尺度、完整参考相对姿态，真实/想象/评估共享配置 |

## 分工与复核

retarget 成员负责原生 UMR vendor、Geno staging、独立环境及运行入口；资产成员负责 OpenSprite 网格/新配置/Tpose/correspondence 几何；tracker 成员清理 X2 死路径、逐关节执行器契约与全body跟踪。主代理检查三方接口、独立性、预算和训练结果。每一阶段都要用最终生成文件做独立复核，不能从当前源码推断旧进程已经采用了新实现。

用户后续明确指定延迟和力矩迟滞与X2相同。该指令优先于团队提出的零延迟名义基线；正式仿真与想象模型必须共同实现源X2的0–1 tick命令延迟、每控制周期更新及5–10ms一阶力矩时间常数，并核对tick真实单位。没有授权复用X2关节映射、限位、惯量、扭矩或碰撞。
