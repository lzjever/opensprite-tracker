# C++ SDK

C++ 与 Python 使用同一组 OpenSprite 消息和 Fast DDS 通道。C++ 客户端与原生网关直接传递生成类型，不经过 JSON、Python 或策略网络。公开接口参考 Unitree SDK2 的初始化、Publisher、Subscriber 和回调使用习惯，保留 OpenSprite 自己的31关节协议。

## 构建与安装

在 SDK 根目录执行：

```bash
bash tools/build_dds.sh
cmake --install native-dds/build --prefix "$PWD/outputs/sdk-install"
cmake -S examples/cpp -B outputs/cpp-examples \
  -DCMAKE_PREFIX_PATH="$PWD/outputs/sdk-install"
cmake --build outputs/cpp-examples --parallel 2
outputs/cpp-examples/opensprite_low_level --help
```

安装提供公共头文件、生成的消息类型、动态库和 CMake 包。其他工程通过 `find_package(OpenSpriteSDK CONFIG REQUIRED)` 及 `target_link_libraries(app PRIVATE OpenSpriteSDK::dds)` 使用，不需要包含源码目录或 JSON 转换头文件。跨架构需要在相应平台重新构建。

## 发布与订阅

```cpp
#include <opensprite_sdk/core/channel.hpp>

using namespace opensprite_sdk;
ChannelFactory::Instance()->Init(42, "lo");
ChannelSubscriber<msg::LowState> states("rt/lowstate");
states.InitChannel();
msg::LowState state;
if (states.Read(state, 1000)) {
    const double position = state.joint_state()[0].q();
    // 使用该测量及其机器人时钟；这里没有实际全局位姿。
}
states.CloseChannel();
```

`Read` 超时单位为毫秒。回调模式使用 `InitChannel(handler, queue_length)`，handler 接受 `const T&`；消息仅在调用期间借用，需长期使用时复制。回调与轮询读取不能同时消费同一订阅者。回调队列有界，满时丢最旧数据；回调异常可通过 `RethrowLastError()` 获取。

也提供 `opensprite_sdk/robot/channel/channel_factory.hpp`、`channel_publisher.hpp`、`channel_subscriber.hpp` 入口及 `opensprite_sdk::robot` 名称，方便沿用 SDK2 的 C++ 文件组织。订阅者支持 `const void*` 回调写法，指针仍只在回调期间有效。

回调可以调用自身订阅者的 `CloseChannel()`。外部关闭或析构会等待工作线程退出；回调应及时返回。回调内销毁自身订阅者时，工作线程由受控回收器接管并等待退出，不把 DDS 清理线程遗留到进程退出之后。

默认接收历史为1，保持最新值语义。需要按时间选择真实观测的客户端可用 `InitChannel(64)` 显式保留有界FIFO；DDS历史最多64条，写端仍为KEEP_LAST(1)。历史保留用于承受短暂调度抖动，不恢复真实丢包，不改写样本时间，也不放宽执行端的反馈年龄和watchdog。

显式历史reader关闭 Fast DDS Data-sharing，使用独立传输拷贝；否则其观测可能随着写端深度1的共享样本回收而失效。默认深度1不变。相关行为见 [Fast DDS历史耦合说明](https://fast-dds.docs.eprosima.com/en/2.6.x/fastdds/transport/datasharing.html)。

`ChannelPublisher<T>::Write` 返回值只表示传输层接受。可选第二参数是发现读者的等待时间，单位为微秒；默认不等待。实际管理消息写入阻塞由 DDS QoS 有限约束。命令执行结果通过状态中的后端接受序号和单独的执行验证判断。

## 控制客户端

`<opensprite_sdk/client.hpp>` 提供 `LowLevelClient`、`ManagementClient` 和 `JointCommand`。`JointCommand` 从实际状态保留机器人 profile、时钟、source session 和序号，按关节顺序填充 q、dq、kp、kd、tau。`ManagementClient::Request` 单独完成 claim、enable、disable、release，并关联请求及应答。

```bash
# 默认只读状态，不申请控制权。
outputs/cpp-examples/opensprite_low_level --domain 42 --interface lo --duration 2

# 显式申请当前测量位姿的低增益保持，需执行端允许使能。
outputs/cpp-examples/opensprite_low_level --domain 42 --interface lo \
  --hold --kp 2 --kd 0.12 --rate-hz 50 --duration 2
```

低增益保持用于接口演示，不是自由站立或舞蹈配方。SDK 不自动启动设备、申请电机权限或加载模型。真机与仿真由相同上层客户端连接不同执行端；命令调度为500Hz，MuJoCo物理保持200Hz，Actor按50Hz生成目标。状态时间与传感器采样时间分别保留，不把500Hz发布写成500Hz独立传感器采样。

C++ 保持例程的写入时钟独立于状态读取。`--rate-hz 500` 每2ms发布保持目标，使用最近真实状态的原始source序号与时间；200Hz仿真不会限制发送循环为200Hz。状态断流100ms会退出，调度错过期限不补发突发命令；实际发送数、独立观测数和超期数在结束时报告。

C++ 的生成类型保留 Fast DDS 有界字符串行为。写入长字符串前应检查 UTF-8 字节长度；原生类型可能在赋值时截断。SDK 校验有限浮点、序列边界和 UTF-8，执行端继续校验关节顺序、来源、权限和反馈年龄。会话标识不是身份认证。
