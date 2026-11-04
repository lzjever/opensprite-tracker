# ROS 2 互通验收

在 SDK 目录安装测试依赖并构建原生 DDS 库后执行：

```bash
bash tools/build_dds.sh
python tools/check_dds_acceptance.py --ros2 \
  --output outputs/dds-acceptance/acceptance.json
```

ROS 2 验收需要 Docker，以及包含 `colcon`、`rclpy`、`rosidl_default_generators` 和 `rmw_fastrtps_cpp` 的镜像。默认使用官方 `ros:humble-ros-base`；首次使用可先执行 `docker pull ros:humble-ros-base`。`--ros-image` 可指定其他已具备这些依赖的镜像，其兼容性由当次结果判断。

验收在容器内独立编译 `ros2/opensprite_msgs`，通过生成的 ROS 2 类型支持和 `rclpy` 通信。宿主机使用 SDK 原生 Fast DDS。两边双向交换 `LowCmd`、`LowState`、`ControlRequest`、`ControlReply`，逐字段比较31关节、IMU、中文字符串、超过 `2**53` 的整数，以及后端接受命令的关联字段。该关联字段表示后端接受命令，不表示实际电机完成执行。

默认 domain 为173，Topic 使用随机前缀。容器使用 host network；本机测试设置 `ROS_LOCALHOST_ONLY=1`，原生端仅选择 loopback。DDS 发现匹配数、消息接收数、镜像 ID、消息包 SHA256 和结果写入 JSON。完整数据和构建日志保存在同级独立工作目录，均不参与发布。

单独运行 ROS 2 互通验收：

```bash
python tools/check_ros2_interop.py \
  --output outputs/dds-acceptance/ros2-interop.json
```

`interop_peer.py` 是容器侧程序，使用真实 ROS 2 中间件，不导入 SDK 通信实现。验收不访问 CAN、IMU 或真实机器人，不启动训练，也不录制视频。
