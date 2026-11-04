# 接口与时间契约

SDK 提供 Python 客户端和 Linux 原生执行层。`ChannelFactoryInitialize(domain, interface=None, transport="memory"|"unix"|"dds", endpoint=...)` 选择通信方式；同一进程内配置一次后再创建各通道。DDS 使用 domain/interface，Unix 使用 endpoint。原生 DDS 和 ROS 2 使用本工程的 OpenSprite 消息，见 [DDS 协议](dds.md)。

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

Unix发布沿用SDK2中“写入通信层”和“消费者执行”分开的习惯：`Write()`成功表示broker已校验并接受消息，不表示每个订阅程序已经收到或执行。每个订阅者都有独立发送worker和一个有界latest待发槽；暂停的诊断进程只合并自己的旧待发帧，不会让健康控制读者和发布者等待它。已进入内核接收队列的帧按原顺序读取，恢复后再接最新待发帧。显式设置更大的`queueLen`只改变接收端已收到帧的本地FIFO，不承诺网络逐帧可靠投递。

关闭broker时，对健康订阅者做一次非阻塞末帧flush，再取消所有待发送和接收线程；不会等待满队列的慢读者。控制程序仍应根据状态序号、时间和明确的失能确认判断执行结果，不能把`Write()`的返回值当作电机执行回执。当前本地传输错误以异常报告，这一点与SDK2 Python的部分`False`返回路径不同。

## 命令和反馈

| 对象 | 字段和含义 |
|---|---|
| `LowCmd_.joint_cmd[31]` | `mode,q,dq,kp,kd,tau`，rad、rad/s、Nm；tau为前馈力矩 |
| `LowState_.joint_state[31]` | `q,dq,tau_est,valid,tau_est_valid,sample_ns,sample_time_kind` |
| `LowState_.imu_state` | gyro、估计单位重力、加速度、附着link、读取时刻、有效性 |
| 消息头 | `profile_id,joint_order_hash,seq,stamp_ns,clock_domain,session_id` |
| 命令来源 | `source_state_seq,source_state_session_id`，必须引用后端提供的新鲜状态 |
| 命令接受关联 | `LowState.accepted_command_seq,accepted_controller_session`，统一runtime成功提交到后端的命令；不是电机实测执行回执 |
| 管理请求 | `ControlRequest` 的 request ID、controller session、robot session 和 claim/enable/disable/release 操作 |
| 管理应答 | `ControlReply.accepted,status,detail,execution_verified`，区分请求接受与执行验证 |

`motor_cmd/motor_state` 是对应关节数组的别名，不能按CAN枚举顺序解释。公共指令始终以 `constants.JOINT_NAMES` 为序。物理驱动顺序及其hash仅属于硬件后端。

`constants.JointIndex` 提供 `LeftHipPitch`、`LeftAnkleRoll`、`WaistYaw`、`HeadYaw` 等枚举，方便使用SDK2风格的索引写法；取值来自OpenSprite顺序。

阻抗语义是 `tau_total = kp*(q_target-q) + kd*(dq_target-dq) + tau_ff`。仿真显式限制完整力矩；真机通过标定机构映射交给原生执行器检查。第一版支持全身一致的mode0或mode1，拒绝混合使能。关节目标的硬件合法范围由逐机配置提供，不由G1或仿真参数推断。

MuJoCo与主机脚踝PD对完整关节力矩做饱和。其他电机保留内置PD，完整请求超过关节effort时明确拒绝，不会偷偷改成全身前馈控制。原生层同时检查量化后的MIT命令及其他电机仍保持的旧命令，通过标定矩阵转回完整关节力矩；耦合电机按实际发送顺序检查，不能只检查全部更新后的终态。这是基于最新反馈的估算约束，不是两次采样之间瞬时机械力矩的独立测量。

`valid` 表示位置/速度可用；力矩估计另有 `tau_est_valid`。当前硬件状态ABI不包含实际力矩和使能反馈，不能把零占位解释成实测值，也不能把 `ready` 当作已经测得电机使能。

## 时钟、权限与退出

加速仿真采用 `clock_domain="sim"`；硬件使用同主机monotonic时钟。时间戳不可跨时钟域比较。IMU和电机保留各自时间来源：`simulation`、`host_receive`、`host_read` 或 `unknown`。新聚合包不会刷新旧传感器的时间。

硬件 `LowState.stamp_ns` 是机器人完成关节与IMU观测汇合的时刻，能够覆盖稍晚取得的IMU样本；原生状态包的时间另在后端校验。各传感器 `sample_ns` 保持原值。远程策略以机器人消息头时刻判断样本相对年龄，执行端再以自身当前时钟校验真实新鲜度。

硬件后端缓存最多64份已发布来源快照。异步命令可以引用仍新鲜的上一帧，实际来源序号保持不变，不会在发送时改成最新帧。来源关节/IMU分别检查60ms/30ms；当前执行反馈另按脚踝10ms、其他电机60ms、IMU30ms检查。未知、过期、未来或其他session的来源都会被拒绝。公开可变消息不是缓存的权威副本。

本地传输每个topic只能有一个publisher；DDS允许多个writer，统一执行层显式分配控制权，后端锁定controller session并拒绝重放。session字符串不是身份认证。单个私有Unix socket也不是跨用户身份认证系统；部署使用受控目录和网络。原生CAN进程另有主机级独占锁。SDK不会修改网口、驱动权限或启动系统服务。

仿真和硬件的 `enable()` 都是显式生命周期操作。`Write` 不隐式使能。正常仿真失能能返回状态确认；真实硬件关闭IPC走原生失能路径，最终结果以原生报告为准。

显式仿真reset产生新的state session；旧命令不能跨reset复用。策略适配器收到会话或时钟变化会拒绝继续，必须重新初始化历史。真机没有 `reset()` 或 `step()`。

仿真接受mode0时同样提交控制者和序号高水位；重新enable不清除此记录，只有reset开启新epoch。仿真延迟在初始化/reset时从配置的最小tick起步，缺少历史时补齐首帧，然后按配置刷新；固定正延迟不会退回零延迟。

当前跨进程硬件桥接服务在读取下一帧native反馈期间可能额外排队约一个50Hz周期。来源缓存使这种异步命令合法，但没有消除这段延迟。直接监督部署入口不经过该桥接。控制频率、传感器年龄和总延迟需要分别测量，不能把该桥接视为与训练的5ms延迟完全相同。

## 传感器与策略

公共状态不提供实际全局位置或world heading。训练/评估可通过仿真后端独立的 `debug_truth()` 获取真值。IMU字段采用声明link坐标系；重力投影不能由动态加速度直接归一化替代。

现舞蹈actor在 `waist_yaw_link` 上观察IMU，真机模板使用 `pelvis_link`。独立策略bundle包含骨盆到上身的关节运动学：重力随相对旋转变换，gyro还要加腰关节运动贡献。该计算只使用关节q/dq及IMU测量，不需要实际全局姿态；不匹配且无转换时拒绝部署。

SDK策略仅执行 `goal-sim2real` actor，保留99维观测、10帧时间顺序历史、770维参考目标、冻结归一化和动作±20限幅。previous action是限幅后发送的策略动作，位于隐含执行器延迟之前。参考轨迹只前进、不循环、不自动重置。
