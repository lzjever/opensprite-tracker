cmake_minimum_required(VERSION 3.16)
project(opensprite_dds VERSION 0.3.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(fastrtps 2.6 REQUIRED)
find_package(fastcdr REQUIRED)
find_package(nlohmann_json 3.9 REQUIRED)
find_package(Threads REQUIRED)
add_library(opensprite_dds SHARED src/endpoint.cpp
  generated/OpenSprite.cxx generated/OpenSpritePubSubTypes.cxx)
target_compile_features(opensprite_dds PRIVATE cxx_std_17)
target_compile_options(opensprite_dds PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(opensprite_dds PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generated>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE src)
target_link_libraries(opensprite_dds PUBLIC fastrtps fastcdr Threads::Threads
  PRIVATE nlohmann_json::nlohmann_json)
# Generated message constructors and typed factories are public C++ SDK ABI.
set_target_properties(opensprite_dds PROPERTIES EXPORT_NAME dds)
add_library(OpenSpriteSDK::dds ALIAS opensprite_dds)
install(TARGETS opensprite_dds EXPORT OpenSpriteSDKTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/opensprite_dds.h DESTINATION include)
install(DIRECTORY include/opensprite_sdk DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES generated/OpenSprite.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES generated/OpenSprite.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opensprite_sdk/idl)
configure_package_config_file(cmake/OpenSpriteSDKConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/OpenSpriteSDKConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenSpriteSDK)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/OpenSpriteSDKConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(EXPORT OpenSpriteSDKTargets NAMESPACE OpenSpriteSDK::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenSpriteSDK)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/OpenSpriteSDKConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/OpenSpriteSDKConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenSpriteSDK)
install(FILES idl/OpenSprite.idl DESTINATION share/opensprite_sdk/idl)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../ros2/opensprite_msgs/msg/
        DESTINATION share/opensprite_sdk/msg)
install(FILES ../LICENSE ../THIRD_PARTY_NOTICES.md DESTINATION share/licenses/opensprite_sdk)
install(DIRECTORY ../licenses/ DESTINATION share/licenses/opensprite_sdk)
