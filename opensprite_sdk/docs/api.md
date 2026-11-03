# 接口与时间契约

SDK 提供 Python 客户端和 Linux 原生执行层。`ChannelFactoryInitialize(domain, interface=None, transport="memory"|"unix", endpoint=...)` 选择本地通信域；同一进程内配置一次后再创建各通道。当前不提供 DDS 网络发现、ROS2 或 G1 二进制消息兼容。

```python
from opensprite_sdk.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from opensprite_sdk.messages import LowCmd_, LowState_

ChannelFactoryInitialize(1, transport="unix", endpoint="/tmp/opensprite-sdk.sock")
publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
subscriber = ChannelSubscriber("rt/lowstate", LowState_)
subscriber.Init(queueLen=1)
publisher.Init()
state = subscriber.Read(timeout=1.0)
# 填写完整31关节指令后调用 publisher.Write(command)。
publisher.Close()
subscriber.Close()
```

`ChannelSubscriber.Init(handler, queueLen=1)` 支持回调。默认只保留最新状态；显式使用更大队列时是有界FIFO，满时丢弃最旧消息。回调异常保留在 `last_error`，不能当作控制程序仍然健康。`RecurrentThread` 提供常见定周期调用习惯和超时统计，不保证Python硬实时。

## 命令和反馈

| 对象 | 字段和含义 |
|---|---|
| `LowCmd_.joint_cmd[31]` | `mode,q,dq,kp,kd,tau`，rad、rad/s、Nm；tau为前馈力矩 |
| `LowState_.joint_state[31]` | `q,dq,tau_est,valid,tau_est_valid,sample_ns,sample_time_kind` |
| `LowState_.imu_state` | gyro、估计单位重力、加速度、附着link、读取时刻、有效性 |
| 消息头 | `profile_id,joint_order_hash,seq,stamp_ns,clock_domain,session_id` |
| 命令来源 | `source_state_seq,source_state_session_id`，必须引用后端提供的新鲜状态 |

`motor_cmd/motor_state` 是对应关节数组的别名，不能按CAN枚举顺序解释。公共指令始终以 `constants.JOINT_NAMES` 为序。物理驱动顺序及其hash仅属于硬件后端。

`constants.JointIndex` 提供 `LeftHipPitch`、`LeftAnkleRoll`、`WaistYaw`、`HeadYaw` 等枚举，方便使用SDK2风格的索引写法；取值来自OpenSprite顺序。

阻抗语义是 `tau_total = kp*(q_target-q) + kd*(dq_target-dq) + tau_ff`。仿真显式限制完整力矩；真机通过标定机构映射交给原生执行器检查。第一版支持全身一致的mode0或mode1，拒绝混合使能。关节目标的硬件合法范围由逐机配置提供，不由G1或仿真参数推断。

`valid` 表示位置/速度可用；力矩估计另有 `tau_est_valid`。当前硬件状态ABI不包含实际力矩和使能反馈，不能把零占位解释成实测值，也不能把 `ready` 当作已经测得电机使能。

## 时钟、权限与退出

加速仿真采用 `clock_domain="sim"`；硬件使用同主机monotonic时钟。时间戳不可跨时钟域比较。IMU和电机保留各自时间来源：`simulation`、`host_receive`、`host_read` 或 `unknown`。新聚合包不会刷新旧传感器的时间。

每个topic只能有一个publisher，后端进一步锁定controller session并拒绝重放。单个私有Unix socket不是跨用户身份认证系统；真实部署使用受控目录。原生CAN进程另有主机级独占锁。SDK不会修改网口、驱动权限或启动系统服务。

仿真和硬件的 `enable()` 都是显式生命周期操作。`Write` 不隐式使能。正常仿真失能能返回状态确认；真实硬件关闭IPC走原生失能路径，最终结果以原生报告为准。

显式仿真reset产生新的state session；旧命令不能跨reset复用。策略适配器收到会话或时钟变化会拒绝继续，必须重新初始化历史。真机没有 `reset()` 或 `step()`。

## 传感器与策略

公共状态不提供实际全局位置或world heading。训练/评估可通过仿真后端独立的 `debug_truth()` 获取真值。IMU字段采用声明link坐标系；重力投影不能由动态加速度直接归一化替代。

现舞蹈actor在 `waist_yaw_link` 上观察IMU，真机模板使用 `pelvis_link`。独立策略bundle包含骨盆到上身的关节运动学：重力随相对旋转变换，gyro还要加腰关节运动贡献。该计算只使用关节q/dq及IMU测量，不需要实际全局姿态；不匹配且无转换时拒绝部署。

SDK策略仅执行 `goal-sim2real` actor，保留99维观测、10帧时间顺序历史、770维参考目标、冻结归一化和动作±20限幅。previous action是限幅后发送的策略动作，位于隐含执行器延迟之前。参考轨迹只前进、不循环、不自动重置。
