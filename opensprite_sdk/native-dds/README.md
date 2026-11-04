# OpenSprite Fast DDS 适配层

公共消息以 `../ros2/opensprite_msgs/msg/*.msg` 为准。`tools/generate_dds_schema.py` 从本项目使用的固定和有界字段子集生成 IDL 及本地 JSON 转换代码。提交的 C++ 类型由 Fast DDS Gen 2.3.0 的 `-typeros2` 生成，保留生成器的 Apache-2.0 许可。

Ubuntu 24.04 开发依赖：

```bash
sudo apt install cmake g++ libfastrtps-dev libfastcdr-dev nlohmann-json3-dev
bash tools/build_dds.sh
```

以上命令从 SDK 根目录执行。原生接口要求 Fast DDS 2.6 或以上的2.x版本；已在 Ubuntu 24.04 x86_64 验证 Fast DDS 2.11.2 / Fast CDR 1.1.0，也在官方 ROS 2 Humble Ubuntu 22.04 x86_64 容器中编译并验证 Fast DDS 2.6.12 / Fast CDR 1.0.29 的类型消息往返。Jetson ARM64仍需目标机验收。使用 ROS 安装的库时先加载对应 `setup.bash`，使 CMake 找到其前缀。

编译使用提交的类型，不要求 Java 或生成器。修改消息后安装 `fastddsgen`、Java，执行 `bash tools/regenerate_dds.sh` 并重新编译；`python tools/generate_dds_schema.py --check` 同时检查 IDL、转换代码和生成 C++ 的SHA清单。

`core.dds.DdsEndpoint` 通过窄 C ABI 调用 C++。JSON 只用于 Python/C++ 进程内参数转换，线上始终是具名生成类型的 CDR。控制命令和状态使用 best effort、keep last 1、volatile；管理请求和回复使用 reliable、keep last 1、volatile，原生写入最长阻塞10ms。`write(timeout=...)` 接受 SDK 的调用参数，原生写入上限固定10ms，不提供每次调用单独修改 DDS QoS。

默认只绑定127.0.0.1。指定网卡名或本机 IPv4 地址时只绑定该地址，不静默开放所有接口；ROS 2 对端也需要匹配所选网卡。一个进程的同域同网卡 endpoint 共享 participant 和引用计数 topic。DDS允许多个 writer，运控所有权由执行层明确管理，不用发现时的匹配数量代替权限授予。

独立安装可执行 `cmake --install native-dds/build --prefix /path/to/prefix`。Python按 `OPENSPRITE_DDS_LIBRARY`、虚拟环境 `lib`、源码开发目录和系统库路径查找适配层。C ABI头文件和消息定义随原生安装提供；跨架构应在对应平台重新构建，Python wheel不包含架构相关动态库。
