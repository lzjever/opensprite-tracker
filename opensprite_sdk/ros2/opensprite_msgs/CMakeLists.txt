cmake_minimum_required(VERSION 3.8)
project(opensprite_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/JointCmd.msg" "msg/JointState.msg" "msg/IMUState.msg"
  "msg/LowCmd.msg" "msg/LowState.msg"
  "msg/ControlRequest.msg" "msg/ControlReply.msg")
ament_export_dependencies(rosidl_default_runtime)
ament_package()
