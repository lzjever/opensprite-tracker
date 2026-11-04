// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <array>
#include <cstdint>

namespace opensprite_sdk {
inline constexpr std::size_t JointCount = 31;
inline constexpr std::uint64_t JointOrderHash = 17325335851961314422ULL;
inline constexpr const char* LowCmdTopic = "rt/lowcmd";
inline constexpr const char* LowStateTopic = "rt/lowstate";
inline constexpr const char* ControlRequestTopic = "rt/control_request";
inline constexpr const char* ControlReplyTopic = "rt/control_reply";
inline constexpr std::array<const char*, JointCount> JointNames{{
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint",
    "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "head_pitch_joint", "head_roll_joint", "head_yaw_joint"
}};
}  // namespace opensprite_sdk
