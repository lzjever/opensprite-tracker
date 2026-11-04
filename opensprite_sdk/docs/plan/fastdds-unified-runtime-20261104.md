# Fast DDS 与统一执行层实施计划

本次在独立 SDK 内增加真实 Fast DDS 通信。策略、遥操作和 ROS 2 程序发送相同 OpenSprite 关节消息，统一执行进程选择 MuJoCo 或 Jetson 后端。保留本地传输与现有训练配置，不启动训练或连接真实电机，不录制视频。

## Team 共识与交叉质询

| 问题 | 结论 |
|---|---|
| 是否直接复用 Unitree 的机器人消息布局 | 参考公开 API 和 DDS/ROS 2 使用习惯。协议采用 OpenSprite 自己的31关节、单位、顺序和机构。 |
| Python 是否绑定全部 Fast DDS API | 采用原生 C++ Fast DDS、生成的类型和窄 C ABI，Python 通过 ctypes 调用。进程内转换可使用 JSON，DDS 线上必须是有类型的 CDR 消息。 |
| ROS 2 兼容是否只要相同字段 | 不足够。以 `.msg` 为协议来源，确定生成 IDL、类型名、Topic 和 QoS；用真实 ROS 2 进程双向通信验证。 |
| 多个 DDS writer 是否等于多个控制者 | 不等于。执行层显式授予 controller session，拒绝其他控制者、重放和跨机器人 session 的命令。DDS 发现不承担授权。 |
| 远程主机如何判断消息年龄 | 来源状态的序号、传感器时间和命令时间均保留机器人时钟。执行端凭自己的状态缓存判断年龄；客户端用本地等待期限处理断流，不把两台主机的 monotonic 时间直接相减。 |
| 收到管理应答是否代表电机执行完成 | 应答明确区分请求接受与执行验证。真机关闭连接只能报告失能路径已请求，实际结果仍由原生退出报告确认。 |
| 是否让 DDS 回调驱动电机周期 | 回调只交付有界消息。执行循环和原生电机调度独立运行，网络断流必须触发本地保护。 |
| 是否为低延迟放宽反馈新鲜度 | 不放宽。现有50Hz原生聚合反馈与脚踝10ms检查必须保留；异步重构需要验证，不能宣称获得训练中的5ms端到端延迟。 |
| 仿真和真机是否完全使用同一设备代码 | 共用协议、权限、生命周期和执行框架，设备后端分别实现物理步进或 CAN/IMU。模型/profile 和标定必须匹配。 |

## 实施分工

1. 协议与传输单元：ROS 2 `opensprite_msgs`、确定性生成器、IDL、原生 Fast DDS 库和 ctypes 包装。
2. 执行单元：统一 runtime、显式 claim/enable/disable/release、来源检查、故障退出、独立策略客户端和 MuJoCo 实时入口。
3. 验证单元：真实 DDS 跨进程、ROS 2 双向互通、慢读者、断流、权限与重放、迁移安装验收；独立交叉审查。
4. 主代理：现有 Channel 和消息兼容集成、中英文说明、完整策略验收、最终修复及交付记录。

协议单元和执行单元相互质询控制权、时间和应答语义；验证单元以独立进程和故障注入检查这些承诺。每个单元完成后交叉审查，未关闭的严重问题阻止软件交付。

## 协议和运行约定

- `LowCmd`、`LowState` 保留现有公共字段和31关节顺序，新增 `ControlRequest` 和 `ControlReply`。消息使用 OpenSprite 命名空间和有界字段。
- DDS Topic 为 `rt/lowcmd`、`rt/lowstate`、`rt/control_request`、`rt/control_reply`；ROS 2 对应去掉 DDS 的 `rt` 前缀。显式 domain/interface 隔离实例，默认本机接口。
- 高频状态和命令采用有界最新值语义；管理请求/应答采用有界可靠传输。管理请求指定目标机器人 session、控制者和 request ID，重复请求只能得到同一结果，不能再次执行。
- claim 不隐式 enable。真机操作保留本地许可、逐机标定、profile 和策略验收门槛。硬件默认 shadow，不自动启动或接管 CAN。
- 公共状态只含传感器可得信息；仿真真值只供独立诊断和本地显示。执行时间、状态时间和传感器时间分别记录，异步通信不改写命令来源。
- 首次仿真轨迹从源首帧开始，失败后继续真实策略和物理，不自动重置或循环参考。当前旧版舞蹈模型不能静默换为新版真机 profile。
- 原生通信库独立构建，生成器和源码可随 SDK 发布；`.reference/`、`outputs/`、本机标定不作为运行依赖。明确支持和实测的 Fast DDS 版本。

## 严格验收

1. 保留已有 SDK 和原生测试；增加消息边界、schema 漂移、非有限值、错误关节顺序、时间和 session 测试。
2. 使用真实 Fast DDS 的独立发布/订阅进程验证两种数据消息和两种管理消息、domain 隔离、重连及释放资源。
3. 模拟慢诊断读者、控制进程退出、命令断流、旧包、错误控制者和篡改的重复请求。验证没有无界队列，没有用发送成功伪造执行确认。
4. 同一 runtime 启动 MuJoCo，另一进程运行真实 CPU ONNX 策略，完成225.66秒参考轨迹。单独报告首次失败和完整执行，不把完整执行等同跟踪成功。
5. fake native/CAN/IMU 对端检查硬件路径，保留逐传感器时间和缺失反馈保护，不访问实际设备。
6. 用官方 ROS 2 容器构建本工程消息包，进行 ROS 2 与原生 SDK 的真实双向通信。不能用同进程 serializer 对照冒充互通。
7. 搬迁整个 SDK、构建 wheel 和 DDS 原生库，隔离父项目与参考目录后运行。检查生成入口、许可证、安装路径和缺失依赖的明确错误。
8. 独立 review 实现与测试，修复发现的问题，再运行相关回归和发布文件检查。交付文档明确 x86_64 软件测试、Jetson ARM64 和真实硬件实测的界限。

## 当前状态

研究、共识、实施和交叉审查已完成。250项Python测试、3项原生CTest、真实ROS 2双向互通、完整225.66秒DDS策略链路及搬迁安装验收通过。软件范围内无未关闭的阻断发现。详细结果与修复见 [验收记录](../validation-dds-20261104.md)。Jetson和真实硬件保持待实测状态。

## 公开资料

- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2)
- [Unitree ROS 2 消息与例程](https://github.com/unitreerobotics/unitree_ros2)
- [Fast DDS 与 ROS 2](https://fast-dds.docs.eprosima.com/en/v3.3.1/fastdds/ros2/ros2.html)
- [ROS 2 Topic 与 DDS 名称映射](https://design.ros2.org/articles/topic_and_service_names.html)
