// Generated from authoritative .msg; do not edit.
#pragma once
#include "OpenSprite.h"
#include <cmath>
#include <stdexcept>
#include <string>
namespace opensprite_msgs { namespace msg {
template<class String> inline void validate_wire_string(const String& value, size_t bound) {
  if (value.size()>bound) throw std::invalid_argument("invalid bounded string");
  const char* bytes=value.c_str();
  for (size_t i=0; i<value.size();) {
    unsigned c=static_cast<unsigned char>(bytes[i++]), n=0, minimum=0;
    if (c==0) throw std::invalid_argument("embedded NUL in string");
    if (c<128) continue;
    if ((c&0xe0)==0xc0) {c&=0x1f;n=1;minimum=0x80;}
    else if ((c&0xf0)==0xe0) {c&=0x0f;n=2;minimum=0x800;}
    else if ((c&0xf8)==0xf0) {c&=0x07;n=3;minimum=0x10000;}
    else throw std::invalid_argument("invalid UTF-8 string");
    if (i+n>value.size()) throw std::invalid_argument("truncated UTF-8 string");
    while (n--) {unsigned b=static_cast<unsigned char>(bytes[i++]); if ((b&0xc0)!=0x80) throw std::invalid_argument("invalid UTF-8 continuation"); c=(c<<6)|(b&0x3f);}
    if (c<minimum || c>0x10ffff || (c>=0xd800 && c<=0xdfff)) throw std::invalid_argument("invalid UTF-8 scalar");
  }
}
inline void validate_wire(const JointCmd& v) {
  if (!std::isfinite(v.q())) throw std::invalid_argument("nonfinite wire value");
  if (!std::isfinite(v.dq())) throw std::invalid_argument("nonfinite wire value");
  if (!std::isfinite(v.kp())) throw std::invalid_argument("nonfinite wire value");
  if (!std::isfinite(v.kd())) throw std::invalid_argument("nonfinite wire value");
  if (!std::isfinite(v.tau())) throw std::invalid_argument("nonfinite wire value");
}
inline void validate_wire(const JointState& v) {
  if (!std::isfinite(v.q())) throw std::invalid_argument("nonfinite wire value");
  if (!std::isfinite(v.dq())) throw std::invalid_argument("nonfinite wire value");
  if (!std::isfinite(v.tau_est())) throw std::invalid_argument("nonfinite wire value");
  validate_wire_string(v.sample_time_kind(), 16);
}
inline void validate_wire(const IMUState& v) {
  for (const auto& item : v.gyroscope()) { if (!std::isfinite(item)) throw std::invalid_argument("nonfinite wire value"); }
  for (const auto& item : v.projected_gravity()) { if (!std::isfinite(item)) throw std::invalid_argument("nonfinite wire value"); }
  for (const auto& item : v.accelerometer()) { if (!std::isfinite(item)) throw std::invalid_argument("nonfinite wire value"); }
  validate_wire_string(v.frame(), 16);
  validate_wire_string(v.attached_link(), 128);
  validate_wire_string(v.sample_time_kind(), 16);
}
inline void validate_wire(const LowCmd& v) {
  validate_wire_string(v.profile_id(), 128);
  validate_wire_string(v.clock_domain(), 16);
  validate_wire_string(v.session_id(), 128);
  for (const auto& item : v.joint_cmd()) { validate_wire(item); }
  validate_wire_string(v.source_state_session_id(), 128);
}
inline void validate_wire(const LowState& v) {
  validate_wire_string(v.profile_id(), 128);
  validate_wire_string(v.clock_domain(), 16);
  validate_wire_string(v.session_id(), 128);
  for (const auto& item : v.joint_state()) { validate_wire(item); }
  validate_wire(v.imu_state());
  validate_wire_string(v.status(), 16);
  if (v.faults().size()>64) throw std::invalid_argument("wire sequence exceeds bound");
  for (const auto& item : v.faults()) { validate_wire_string(item, 1024); }
  validate_wire_string(v.accepted_controller_session(), 128);
}
inline void validate_wire(const ControlRequest& v) {
  validate_wire_string(v.profile_id(), 128);
  validate_wire_string(v.clock_domain(), 16);
  validate_wire_string(v.session_id(), 128);
  validate_wire_string(v.request_id(), 128);
  validate_wire_string(v.controller_session(), 128);
  validate_wire_string(v.robot_session(), 128);
  validate_wire_string(v.operation(), 16);
}
inline void validate_wire(const ControlReply& v) {
  validate_wire_string(v.profile_id(), 128);
  validate_wire_string(v.clock_domain(), 16);
  validate_wire_string(v.session_id(), 128);
  validate_wire_string(v.request_id(), 128);
  validate_wire_string(v.controller_session(), 128);
  validate_wire_string(v.robot_session(), 128);
  validate_wire_string(v.status(), 64);
  validate_wire_string(v.detail(), 1024);
}
}} // namespace opensprite_msgs::msg
