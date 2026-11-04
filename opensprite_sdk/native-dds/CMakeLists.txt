cmake_minimum_required(VERSION 3.16)
project(opensprite_dds LANGUAGES CXX)
find_package(fastrtps 2.6 REQUIRED)
find_package(fastcdr REQUIRED)
find_package(nlohmann_json 3.9 REQUIRED)
find_package(Threads REQUIRED)
add_library(opensprite_dds SHARED src/endpoint.cpp
  generated/OpenSprite.cxx generated/OpenSpritePubSubTypes.cxx)
target_compile_features(opensprite_dds PRIVATE cxx_std_17)
target_compile_options(opensprite_dds PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(opensprite_dds PRIVATE generated src)
target_link_libraries(opensprite_dds PRIVATE fastrtps fastcdr nlohmann_json::nlohmann_json Threads::Threads)
set_target_properties(opensprite_dds PROPERTIES CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
install(TARGETS opensprite_dds LIBRARY DESTINATION lib)
install(FILES include/opensprite_dds.h DESTINATION include)
install(FILES idl/OpenSprite.idl DESTINATION share/opensprite_sdk/idl)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../ros2/opensprite_msgs/msg/
        DESTINATION share/opensprite_sdk/msg)
