# 仿真后端设计与交叉审查

## 交付范围

`MujocoBackend` 在 CPU 上执行真实 MuJoCo 物理。它读取包内模型、网格和 profile，直接使用统一的 `LowCmd_`、`LowState_`。子工程安装后不需要父项目、`.reference`、训练框架或 GPU。

`legacy_v4` 保留 v4 刚体模型、31关节顺序、关节约束和训练用 PD 参数。物理步长为5ms，策略周期为20ms，solver iterations/ls_iterations为50/25。一阶力矩迟滞为零。命令延迟为0/1个物理tick，保持概率0.9，每4tick刷新，随机相位和种子明确记录。CPU NumPy 与训练用 GPU RNG 不保证逐随机数一致；相同 CPU 种子的运行可以复现。

`runtime_timing_v4` 是独立实验 profile。它使用2ms物理步长和pelvis IMU，保留旧版刚体惯量和effort。它不代表换装J4340P后的真实机器人，也不自动接收旧模型的部署指令。

## 公共接口

`start/read/write/step/close` 实现仿真生命周期。`enable/disable` 单独控制使能。写入命令不会隐式使能。仿真支持手动确定性步进，实时服务负责按墙钟节拍调用相同接口。

所有指令都采用关节坐标和SI单位。完整控制力矩为 `Kp*(q_des-q)+Kd*(dq_des-dq)+tau_ff`，先计算再按每关节effort显式限幅，最后写入 `qfrc_applied`。模型没有MuJoCo actuator；仅设置 `actuatorfrcrange` 无法限制外部施加的力矩。

位置目标不截断为关节机械限位，速度也不被人为硬截断。机械位置限制交给物理约束求解，速度限制用于诊断。这样保持已训练策略的执行语义；真机需要自己的电机包络和目标投影。

`LowState_` 提供关节反馈、IMU gyro/重力投影/加速度及有效性。`tau_est` 在仿真中表示已施加力矩，`tau_est_valid=True`。全局root位置、姿态和速度只由独立的 `debug_truth()` 提供。它们不进入策略公共状态。

## 时钟与指令归属

加速仿真消息使用 `clock_domain=sim`。传感器时间来自物理采样时刻，`sample_time_kind=simulation`。重复读取静止仿真会得到新的发布序号，但不会伪造新的采样时间。

命令绑定源状态序号与源状态session。写入者必须提供正序号和非空controller session；另一控制器不能接管已有命令流。重复序号、过期/未来命令和过期源状态被拒绝。reset会换状态epoch。disable会清空已有源状态记录，保留发布流session、写入者身份和命令序号上界；重新enable后旧命令不能重放，控制器身份只能通过reset重新建立。超过100ms没有有效指令就失能并进入fault；重复enable不会刷新watchdog。

## IMU交叉质询

现有舞蹈策略的IMU训练位置是 `waist_yaw_link`。实际机器人IMU安装在pelvis，不能直接用pelvis gyro冒充torso gyro。因此profile与消息都明确IMU所属link。

显式的pelvis到torso变换需要腰部关节位置和速度。重力通过相对旋转转换，gyro还必须加入腰部运动产生的相对角速度。交叉测试用30组随机根姿态、腰部位置和关节速度，将SDK变换与MuJoCo torso传感器输出比较，误差低于1e-12。这里验证的是几何和速度变换，未验证真机反馈延迟下的估计误差。

## 单元与集成检查

当前CPU测试覆盖模型/关节顺序/包内SHA、十秒站立、完整PD加前馈力矩限幅、显式使能、watchdog、重放/归属/epoch、时间戳语义、yaw不变的重力投影、腰部运动对IMU的影响、确定性延迟、非法输入和关闭行为。

OSMesa子进程生成真实仿真图像，验证headless渲染，不初始化CUDA。`outputs/sim-preview.png` 已作人工检查，机器人和格子地面正常。`examples/sim_robot.py` 启动与客户端分离的SDK服务；独立sim服务与客户端的两进程例程已经通过：1秒内发送48条命令，双方正常退出，最终状态为disabled。原Actor与导出前FrozenActor在32组输入上的输出最大差为0；reference goal在141帧（含片尾）对照的最大差为9.54e-7。详细验收记录位于本地outputs/acceptance。

这些检查证明SDK仿真链路可运行，不代表整个225.66秒舞蹈已成功，也不代表真机控制已完成现场验收。

## 独立迁移验收

`tools/check_standalone.py` 将发行文件复制到新的目录，排除参考仓库、输出、虚拟环境和构建缓存；对复制内容生成逐文件SHA256清单。随后在新目录构建wheel，使用已有第三方依赖安装到隔离目录，移除源码路径，并断言父项目包无法导入。

已安装wheel中的42份模型资产通过哈希核验，真实物理站立两秒的最低root高度为0.459355m。两个独立进程通过已安装SDK完成命令和反馈交换，正常失能并退出。搬迁后的native源码独立编译出CAN控制器、ICM读取器和三项测试程序，三项离线测试全部通过；没有启动硬件程序。本机未安装CMake，因此该项明确采用cc/c++编译回退路径。

这次验收复用开发机已有第三方Python依赖，运行架构是x86_64；它证明代码和资产可从父项目迁移，不替代Jetson aarch64安装及真机验收。
