// Generated from authoritative .msg; do not edit.
#pragma once
#include "OpenSprite.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace opensprite_msgs { namespace msg {
inline void from_json(const nlohmann::json& j, JointCmd& v) {
  if (!j.is_object() || j.size() != 6) throw std::invalid_argument("invalid JointCmd fields");
  {
    if (!(j.at("mode").is_number_unsigned() || (j.at("mode").is_number_integer() && j.at("mode").get<std::int64_t>() >= 0)) || j.at("mode").get<std::uint64_t>() > 255ULL) throw std::invalid_argument("invalid unsigned integer");
    v.mode() = j.at("mode").get<std::uint8_t>();
  }
  {
    if (!j.at("q").is_number() || !std::isfinite(j.at("q").get<double>())) throw std::invalid_argument("nonfinite or nonnumeric value");
    v.q() = j.at("q").get<double>();
  }
  {
    if (!j.at("dq").is_number() || !std::isfinite(j.at("dq").get<double>())) throw std::invalid_argument("nonfinite or nonnumeric value");
    v.dq() = j.at("dq").get<double>();
  }
  {
    if (!j.at("kp").is_number() || !std::isfinite(j.at("kp").get<double>())) throw std::invalid_argument("nonfinite or nonnumeric value");
    v.kp() = j.at("kp").get<double>();
  }
  {
    if (!j.at("kd").is_number() || !std::isfinite(j.at("kd").get<double>())) throw std::invalid_argument("nonfinite or nonnumeric value");
    v.kd() = j.at("kd").get<double>();
  }
  {
    if (!j.at("tau").is_number() || !std::isfinite(j.at("tau").get<double>())) throw std::invalid_argument("nonfinite or nonnumeric value");
    v.tau() = j.at("tau").get<double>();
  }
}
inline void to_json(nlohmann::json& j, const JointCmd& v) {
  j = nlohmann::json::object();
  j["mode"] = v.mode();
  j["q"] = v.q();
  j["dq"] = v.dq();
  j["kp"] = v.kp();
  j["kd"] = v.kd();
  j["tau"] = v.tau();
}
inline void from_json(const nlohmann::json& j, JointState& v) {
  if (!j.is_object() || j.size() != 7) throw std::invalid_argument("invalid JointState fields");
  {
    if (!j.at("q").is_number() || !std::isfinite(j.at("q").get<double>())) throw std::invalid_argument("nonfinite or nonnumeric value");
    v.q() = j.at("q").get<double>();
  }
  {
    if (!j.at("dq").is_number() || !std::isfinite(j.at("dq").get<double>())) throw std::invalid_argument("nonfinite or nonnumeric value");
    v.dq() = j.at("dq").get<double>();
  }
  {
    if (!j.at("tau_est").is_number() || !std::isfinite(j.at("tau_est").get<double>())) throw std::invalid_argument("nonfinite or nonnumeric value");
    v.tau_est() = j.at("tau_est").get<double>();
  }
  {
    if (!j.at("valid").is_boolean()) throw std::invalid_argument("invalid boolean");
    v.valid() = j.at("valid").get<bool>();
  }
  {
    if (!(j.at("sample_ns").is_number_unsigned() || (j.at("sample_ns").is_number_integer() && j.at("sample_ns").get<std::int64_t>() >= 0)) || j.at("sample_ns").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.sample_ns() = j.at("sample_ns").get<std::uint64_t>();
  }
  {
    if (!j.at("sample_time_kind").is_string() || j.at("sample_time_kind").get_ref<const std::string&>().size() > 16) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.sample_time_kind() = j.at("sample_time_kind").get<std::string>();
  }
  {
    if (!j.at("tau_est_valid").is_boolean()) throw std::invalid_argument("invalid boolean");
    v.tau_est_valid() = j.at("tau_est_valid").get<bool>();
  }
}
inline void to_json(nlohmann::json& j, const JointState& v) {
  j = nlohmann::json::object();
  j["q"] = v.q();
  j["dq"] = v.dq();
  j["tau_est"] = v.tau_est();
  j["valid"] = v.valid();
  j["sample_ns"] = v.sample_ns();
  j["sample_time_kind"] = v.sample_time_kind();
  j["tau_est_valid"] = v.tau_est_valid();
}
inline void from_json(const nlohmann::json& j, IMUState& v) {
  if (!j.is_object() || j.size() != 8) throw std::invalid_argument("invalid IMUState fields");
  {
    if (!j.at("gyroscope").is_array() || j.at("gyroscope").size() != 3) throw std::invalid_argument("invalid gyroscope length");
    for (size_t i = 0; i < j.at("gyroscope").size(); ++i) {
      if (!j.at("gyroscope")[i].is_number() || !std::isfinite(j.at("gyroscope")[i].get<double>())) throw std::invalid_argument("nonfinite or nonnumeric value");
      v.gyroscope()[i] = j.at("gyroscope")[i].get<double>();
    }
  }
  {
    if (!j.at("projected_gravity").is_array() || j.at("projected_gravity").size() != 3) throw std::invalid_argument("invalid projected_gravity length");
    for (size_t i = 0; i < j.at("projected_gravity").size(); ++i) {
      if (!j.at("projected_gravity")[i].is_number() || !std::isfinite(j.at("projected_gravity")[i].get<double>())) throw std::invalid_argument("nonfinite or nonnumeric value");
      v.projected_gravity()[i] = j.at("projected_gravity")[i].get<double>();
    }
  }
  {
    if (!j.at("accelerometer").is_array() || j.at("accelerometer").size() != 3) throw std::invalid_argument("invalid accelerometer length");
    for (size_t i = 0; i < j.at("accelerometer").size(); ++i) {
      if (!j.at("accelerometer")[i].is_number() || !std::isfinite(j.at("accelerometer")[i].get<double>())) throw std::invalid_argument("nonfinite or nonnumeric value");
      v.accelerometer()[i] = j.at("accelerometer")[i].get<double>();
    }
  }
  {
    if (!j.at("valid").is_boolean()) throw std::invalid_argument("invalid boolean");
    v.valid() = j.at("valid").get<bool>();
  }
  {
    if (!(j.at("sample_ns").is_number_unsigned() || (j.at("sample_ns").is_number_integer() && j.at("sample_ns").get<std::int64_t>() >= 0)) || j.at("sample_ns").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.sample_ns() = j.at("sample_ns").get<std::uint64_t>();
  }
  {
    if (!j.at("frame").is_string() || j.at("frame").get_ref<const std::string&>().size() > 16) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.frame() = j.at("frame").get<std::string>();
  }
  {
    if (!j.at("attached_link").is_string() || j.at("attached_link").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.attached_link() = j.at("attached_link").get<std::string>();
  }
  {
    if (!j.at("sample_time_kind").is_string() || j.at("sample_time_kind").get_ref<const std::string&>().size() > 16) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.sample_time_kind() = j.at("sample_time_kind").get<std::string>();
  }
}
inline void to_json(nlohmann::json& j, const IMUState& v) {
  j = nlohmann::json::object();
  j["gyroscope"] = v.gyroscope();
  j["projected_gravity"] = v.projected_gravity();
  j["accelerometer"] = v.accelerometer();
  j["valid"] = v.valid();
  j["sample_ns"] = v.sample_ns();
  j["frame"] = v.frame();
  j["attached_link"] = v.attached_link();
  j["sample_time_kind"] = v.sample_time_kind();
}
inline void from_json(const nlohmann::json& j, LowCmd& v) {
  if (!j.is_object() || j.size() != 9) throw std::invalid_argument("invalid LowCmd fields");
  {
    if (!j.at("profile_id").is_string() || j.at("profile_id").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.profile_id() = j.at("profile_id").get<std::string>();
  }
  {
    if (!(j.at("joint_order_hash").is_number_unsigned() || (j.at("joint_order_hash").is_number_integer() && j.at("joint_order_hash").get<std::int64_t>() >= 0)) || j.at("joint_order_hash").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.joint_order_hash() = j.at("joint_order_hash").get<std::uint64_t>();
  }
  {
    if (!(j.at("seq").is_number_unsigned() || (j.at("seq").is_number_integer() && j.at("seq").get<std::int64_t>() >= 0)) || j.at("seq").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.seq() = j.at("seq").get<std::uint64_t>();
  }
  {
    if (!(j.at("stamp_ns").is_number_unsigned() || (j.at("stamp_ns").is_number_integer() && j.at("stamp_ns").get<std::int64_t>() >= 0)) || j.at("stamp_ns").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.stamp_ns() = j.at("stamp_ns").get<std::uint64_t>();
  }
  {
    if (!j.at("clock_domain").is_string() || j.at("clock_domain").get_ref<const std::string&>().size() > 16) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.clock_domain() = j.at("clock_domain").get<std::string>();
  }
  {
    if (!j.at("session_id").is_string() || j.at("session_id").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.session_id() = j.at("session_id").get<std::string>();
  }
  {
    if (!j.at("joint_cmd").is_array() || j.at("joint_cmd").size() != 31) throw std::invalid_argument("invalid joint_cmd length");
    for (size_t i = 0; i < j.at("joint_cmd").size(); ++i) {
      from_json(j.at("joint_cmd")[i], v.joint_cmd()[i]);
    }
  }
  {
    if (!(j.at("source_state_seq").is_number_unsigned() || (j.at("source_state_seq").is_number_integer() && j.at("source_state_seq").get<std::int64_t>() >= 0)) || j.at("source_state_seq").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.source_state_seq() = j.at("source_state_seq").get<std::uint64_t>();
  }
  {
    if (!j.at("source_state_session_id").is_string() || j.at("source_state_session_id").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.source_state_session_id() = j.at("source_state_session_id").get<std::string>();
  }
}
inline void to_json(nlohmann::json& j, const LowCmd& v) {
  j = nlohmann::json::object();
  j["profile_id"] = v.profile_id();
  j["joint_order_hash"] = v.joint_order_hash();
  j["seq"] = v.seq();
  j["stamp_ns"] = v.stamp_ns();
  j["clock_domain"] = v.clock_domain();
  j["session_id"] = v.session_id();
  j["joint_cmd"] = v.joint_cmd();
  j["source_state_seq"] = v.source_state_seq();
  j["source_state_session_id"] = v.source_state_session_id();
}
inline void from_json(const nlohmann::json& j, LowState& v) {
  if (!j.is_object() || j.size() != 12) throw std::invalid_argument("invalid LowState fields");
  {
    if (!j.at("profile_id").is_string() || j.at("profile_id").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.profile_id() = j.at("profile_id").get<std::string>();
  }
  {
    if (!(j.at("joint_order_hash").is_number_unsigned() || (j.at("joint_order_hash").is_number_integer() && j.at("joint_order_hash").get<std::int64_t>() >= 0)) || j.at("joint_order_hash").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.joint_order_hash() = j.at("joint_order_hash").get<std::uint64_t>();
  }
  {
    if (!(j.at("seq").is_number_unsigned() || (j.at("seq").is_number_integer() && j.at("seq").get<std::int64_t>() >= 0)) || j.at("seq").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.seq() = j.at("seq").get<std::uint64_t>();
  }
  {
    if (!(j.at("stamp_ns").is_number_unsigned() || (j.at("stamp_ns").is_number_integer() && j.at("stamp_ns").get<std::int64_t>() >= 0)) || j.at("stamp_ns").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.stamp_ns() = j.at("stamp_ns").get<std::uint64_t>();
  }
  {
    if (!j.at("clock_domain").is_string() || j.at("clock_domain").get_ref<const std::string&>().size() > 16) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.clock_domain() = j.at("clock_domain").get<std::string>();
  }
  {
    if (!j.at("session_id").is_string() || j.at("session_id").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.session_id() = j.at("session_id").get<std::string>();
  }
  {
    if (!j.at("joint_state").is_array() || j.at("joint_state").size() != 31) throw std::invalid_argument("invalid joint_state length");
    for (size_t i = 0; i < j.at("joint_state").size(); ++i) {
      from_json(j.at("joint_state")[i], v.joint_state()[i]);
    }
  }
  {
    from_json(j.at("imu_state"), v.imu_state());
  }
  {
    if (!j.at("status").is_string() || j.at("status").get_ref<const std::string&>().size() > 16) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.status() = j.at("status").get<std::string>();
  }
  {
    if (!j.at("faults").is_array() || j.at("faults").size() > 64) throw std::invalid_argument("invalid faults length");
    v.faults().resize(j.at("faults").size());
    for (size_t i = 0; i < j.at("faults").size(); ++i) {
      if (!j.at("faults")[i].is_string() || j.at("faults")[i].get_ref<const std::string&>().size() > 1024) throw std::invalid_argument("invalid bounded UTF-8 string");
      v.faults()[i] = j.at("faults")[i].get<std::string>();
    }
  }
  {
    if (!(j.at("accepted_command_seq").is_number_unsigned() || (j.at("accepted_command_seq").is_number_integer() && j.at("accepted_command_seq").get<std::int64_t>() >= 0)) || j.at("accepted_command_seq").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.accepted_command_seq() = j.at("accepted_command_seq").get<std::uint64_t>();
  }
  {
    if (!j.at("accepted_controller_session").is_string() || j.at("accepted_controller_session").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.accepted_controller_session() = j.at("accepted_controller_session").get<std::string>();
  }
}
inline void to_json(nlohmann::json& j, const LowState& v) {
  j = nlohmann::json::object();
  j["profile_id"] = v.profile_id();
  j["joint_order_hash"] = v.joint_order_hash();
  j["seq"] = v.seq();
  j["stamp_ns"] = v.stamp_ns();
  j["clock_domain"] = v.clock_domain();
  j["session_id"] = v.session_id();
  j["joint_state"] = v.joint_state();
  j["imu_state"] = v.imu_state();
  j["status"] = v.status();
  j["faults"] = v.faults();
  j["accepted_command_seq"] = v.accepted_command_seq();
  j["accepted_controller_session"] = v.accepted_controller_session();
}
inline void from_json(const nlohmann::json& j, ControlRequest& v) {
  if (!j.is_object() || j.size() != 10) throw std::invalid_argument("invalid ControlRequest fields");
  {
    if (!j.at("profile_id").is_string() || j.at("profile_id").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.profile_id() = j.at("profile_id").get<std::string>();
  }
  {
    if (!(j.at("joint_order_hash").is_number_unsigned() || (j.at("joint_order_hash").is_number_integer() && j.at("joint_order_hash").get<std::int64_t>() >= 0)) || j.at("joint_order_hash").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.joint_order_hash() = j.at("joint_order_hash").get<std::uint64_t>();
  }
  {
    if (!(j.at("seq").is_number_unsigned() || (j.at("seq").is_number_integer() && j.at("seq").get<std::int64_t>() >= 0)) || j.at("seq").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.seq() = j.at("seq").get<std::uint64_t>();
  }
  {
    if (!(j.at("stamp_ns").is_number_unsigned() || (j.at("stamp_ns").is_number_integer() && j.at("stamp_ns").get<std::int64_t>() >= 0)) || j.at("stamp_ns").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.stamp_ns() = j.at("stamp_ns").get<std::uint64_t>();
  }
  {
    if (!j.at("clock_domain").is_string() || j.at("clock_domain").get_ref<const std::string&>().size() > 16) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.clock_domain() = j.at("clock_domain").get<std::string>();
  }
  {
    if (!j.at("session_id").is_string() || j.at("session_id").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.session_id() = j.at("session_id").get<std::string>();
  }
  {
    if (!j.at("request_id").is_string() || j.at("request_id").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.request_id() = j.at("request_id").get<std::string>();
  }
  {
    if (!j.at("controller_session").is_string() || j.at("controller_session").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.controller_session() = j.at("controller_session").get<std::string>();
  }
  {
    if (!j.at("robot_session").is_string() || j.at("robot_session").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.robot_session() = j.at("robot_session").get<std::string>();
  }
  {
    if (!j.at("operation").is_string() || j.at("operation").get_ref<const std::string&>().size() > 16) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.operation() = j.at("operation").get<std::string>();
  }
}
inline void to_json(nlohmann::json& j, const ControlRequest& v) {
  j = nlohmann::json::object();
  j["profile_id"] = v.profile_id();
  j["joint_order_hash"] = v.joint_order_hash();
  j["seq"] = v.seq();
  j["stamp_ns"] = v.stamp_ns();
  j["clock_domain"] = v.clock_domain();
  j["session_id"] = v.session_id();
  j["request_id"] = v.request_id();
  j["controller_session"] = v.controller_session();
  j["robot_session"] = v.robot_session();
  j["operation"] = v.operation();
}
inline void from_json(const nlohmann::json& j, ControlReply& v) {
  if (!j.is_object() || j.size() != 13) throw std::invalid_argument("invalid ControlReply fields");
  {
    if (!j.at("profile_id").is_string() || j.at("profile_id").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.profile_id() = j.at("profile_id").get<std::string>();
  }
  {
    if (!(j.at("joint_order_hash").is_number_unsigned() || (j.at("joint_order_hash").is_number_integer() && j.at("joint_order_hash").get<std::int64_t>() >= 0)) || j.at("joint_order_hash").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.joint_order_hash() = j.at("joint_order_hash").get<std::uint64_t>();
  }
  {
    if (!(j.at("seq").is_number_unsigned() || (j.at("seq").is_number_integer() && j.at("seq").get<std::int64_t>() >= 0)) || j.at("seq").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.seq() = j.at("seq").get<std::uint64_t>();
  }
  {
    if (!(j.at("stamp_ns").is_number_unsigned() || (j.at("stamp_ns").is_number_integer() && j.at("stamp_ns").get<std::int64_t>() >= 0)) || j.at("stamp_ns").get<std::uint64_t>() > UINT64_MAX) throw std::invalid_argument("invalid unsigned integer");
    v.stamp_ns() = j.at("stamp_ns").get<std::uint64_t>();
  }
  {
    if (!j.at("clock_domain").is_string() || j.at("clock_domain").get_ref<const std::string&>().size() > 16) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.clock_domain() = j.at("clock_domain").get<std::string>();
  }
  {
    if (!j.at("session_id").is_string() || j.at("session_id").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.session_id() = j.at("session_id").get<std::string>();
  }
  {
    if (!j.at("request_id").is_string() || j.at("request_id").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.request_id() = j.at("request_id").get<std::string>();
  }
  {
    if (!j.at("controller_session").is_string() || j.at("controller_session").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.controller_session() = j.at("controller_session").get<std::string>();
  }
  {
    if (!j.at("robot_session").is_string() || j.at("robot_session").get_ref<const std::string&>().size() > 128) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.robot_session() = j.at("robot_session").get<std::string>();
  }
  {
    if (!j.at("accepted").is_boolean()) throw std::invalid_argument("invalid boolean");
    v.accepted() = j.at("accepted").get<bool>();
  }
  {
    if (!j.at("status").is_string() || j.at("status").get_ref<const std::string&>().size() > 64) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.status() = j.at("status").get<std::string>();
  }
  {
    if (!j.at("detail").is_string() || j.at("detail").get_ref<const std::string&>().size() > 1024) throw std::invalid_argument("invalid bounded UTF-8 string");
    v.detail() = j.at("detail").get<std::string>();
  }
  {
    if (!j.at("execution_verified").is_boolean()) throw std::invalid_argument("invalid boolean");
    v.execution_verified() = j.at("execution_verified").get<bool>();
  }
}
inline void to_json(nlohmann::json& j, const ControlReply& v) {
  j = nlohmann::json::object();
  j["profile_id"] = v.profile_id();
  j["joint_order_hash"] = v.joint_order_hash();
  j["seq"] = v.seq();
  j["stamp_ns"] = v.stamp_ns();
  j["clock_domain"] = v.clock_domain();
  j["session_id"] = v.session_id();
  j["request_id"] = v.request_id();
  j["controller_session"] = v.controller_session();
  j["robot_session"] = v.robot_session();
  j["accepted"] = v.accepted();
  j["status"] = v.status();
  j["detail"] = v.detail();
  j["execution_verified"] = v.execution_verified();
}
}} // namespace opensprite_msgs::msg
