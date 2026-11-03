# OpenSprite SDK 实施计划

目标是在本目录交付可独立切出的 SDK、MuJoCo 仿真和 Jetson Orin Nano Super 部署例程。上层控制程序使用 Unitree SDK2 风格的初始化、发布命令和订阅状态；消息采用 OpenSprite 的31关节顺序。第一版提供 Python 客户端、本地进程内/Unix 通信和原生 C++ CAN 执行层。接口风格兼容不表示与 Unitree DDS 消息二进制互通。

## 来源与边界

- Unitree SDK2、SDK2 Python 和 MuJoCo bridge 只作公开接口参考，下载到 `.reference/`，不作为安装依赖。
- 官方 `open_sprite_runtime` 的固定版本为 `4515206c24404e18a2d5c168eccdb7468aea6ab7`。移植必要原生代码时保留 AGPL 和来源，移除现场绝对路径依赖。
- 硬件以 NVIDIA Jetson Orin Nano Super Developer Kit、Kunhong 四路 USB CAN FD 和官方 BOM 为研究对象。逐机零位、方向、通信范围、电流/温度限制和 IMU 安装必须由标定配置提供。
- 独立复制模型、网格、配置和许可证。SDK 不导入父项目代码，不依赖 `.reference/` 或父目录的运行文件。策略导出工具允许显式输入外部 checkpoint/动作文件，导出后的 bundle 独立运行。
- 保留现有 tracker、运行目录、物理配置和预算。本次不操作机器人、不启动新训练、不把离线验证写成实机验证。

## 公共契约

`ChannelFactoryInitialize`、`ChannelPublisher.Init/Write`、`ChannelSubscriber.Init/Read/Close` 使用熟悉的发布订阅习惯。LowCmd/LowState 的公共数组按31个关节排序。`motor_cmd`/`motor_state` 只作关节空间的兼容别名，原始电机状态另行诊断，禁止混入 CAN 排序。

命令包含 q、dq、kp、kd、tau 和全身 mode。第一版只支持全身一致的启用模式，拒绝部分关节隐式切换。发送命令不隐式取得硬件控制权。消息包含 profile、顺序 hash、session、序号、源状态序号、时间戳及 sim/monotonic 时钟域。状态保留电机和 IMU 各自的采样时刻、有效性和传感器所在 link。默认队列采用最新状态语义；旧包、非有限数、错误顺序及过期数据不得继续执行。

公共观测只有可测关节、IMU 和诊断数据。全局姿态/位置真值只由独立仿真评估接口提供。IMU 重力投影是估计重力，不能用动态加速度直接归一化代替。

## 交叉质询与决策

1. **IMU 所在 link 不同。** 现舞蹈策略使用 waist_yaw_link，官方真机使用骨盆 IMU。部署显式检查所在 link；只有经过测试的关节运动学转换才能把骨盆 gyro/重力转为上身输入，不能固定旋转冒充。
2. **关节和电机不同。** 左右脚踝和头部 pitch/roll 是耦合机构。转换位置/速度时使用标定矩阵，转换力矩时使用逆转置，验证功率守恒。
3. **官方状态 IPC 不是关节状态。** 528字节包是 motor order 的聚合快照，1280字节目标包是 joint order；各有独立顺序 hash。IMU 使用独立协议。后端负责转换和汇合。
4. **官方 CAN 循环需要逐电机新鲜度审查。** 仅保留启动时 seen 标记不足以处理运行中断流；移植层补逐电机反馈 age 检查，避免重复发布旧反馈。
5. **硬件版本不一致。** 当前 tracker 使用旧版 v4；后续官方四肩已换 J4340P。旧版仿真 profile 保留，硬件 profile 独立声明。只修改电机参数不能宣称 CAD 惯量已更新。
6. **时间不能混用。** 加速仿真使用 sim time；真机使用 monotonic。2kHz 调度循环不等于2kHz电机反馈；脚踝500Hz、其他电机50Hz分别调度和检查 age。
7. **统一接口不保证同等动力学。** 自由基仿真仍显式写 qfrc_applied 并限幅。延迟、物理频率、执行器和模型版本进入 profile，部署禁止静默换参。

## 实施单元与负责人

| 单元 | 实施 | 交叉审查 |
|---|---|---|
| 消息、Channel、Unix broker、顺序/时间契约 | contract agent | hardware / sim |
| 独立机器人资产、MuJoCo、观测、调度与渲染 | sim agent | contract / root |
| CAN 原生层、标定验证、IPC/IMU、真实后端 | hardware agent | contract / root |
| 策略导出、IMU 转换、统一部署例程、安装与文档 | root | sim / hardware |

## 验收顺序

1. 单元测试：消息深拷贝/有限性/顺序、跨进程通信、断连、时钟/序号、watchdog、MIT 编码、耦合映射及功率守恒、原生 ABI。
2. CPU 真实 MuJoCo：31关节、限幅、IMU、全身使能/失能、确定性步进、超时和故障注入。编译 C++ 原生层并运行离线测试。
3. 导出已发布 goal-sim2real checkpoint，比较独立实现与原策略的输入、目标和输出；ONNX 与 PyTorch 做数值比较。仿真例程由同一公共接口执行，完整运行且保存首次失败，不把继续运行解释为成功。
4. 真机例程通过 fake native peer 和真实 SocketCAN 编码链路离线验收；缺失标定、模型/profile 不匹配、旧状态和 IMU 断流必须拒绝继续。只编译和回放，不接管本机设备。
5. 从单独复制的子目录安装并运行，检查 wheel/源码包资产、相对路径、许可证和 `.reference` 排除。最后交叉审查逐项关门，记录已测平台及剩余实机验收。

交付包括中英文使用入口、兼容性说明、Jetson 配置与构建步骤、sim2sim/硬件部署例程、可重复测试命令和验收记录。成熟度以这些检查的实际结果为准，未连接 Jetson 的项目继续明确标为未实测。

## 软件验收状态

接口、仿真、硬件适配与策略部署已实现并完成交叉审查。发现并修复了控制者session切换、reset重放、读后缓存老化、IMU所在link、动作±20限幅、MIT范围静默忽略和仿真正常失能握手等问题。

实际结果见 [validation.md](../validation.md)。SDK已通过独立目录搬迁、wheel安装、父项目import阻断、真实物理、跨进程和原生源码离线测试。硬件软件入口已具备，但Jetson、USB IMU和上电验收仍需实际设备，不标作已完成。
