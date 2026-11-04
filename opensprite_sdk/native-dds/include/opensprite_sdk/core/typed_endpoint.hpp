// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <OpenSprite.h>
#include <memory>
#include <string>
#include <type_traits>

namespace opensprite_sdk {

namespace msg {
using JointCmd = opensprite_msgs::msg::JointCmd;
using JointState = opensprite_msgs::msg::JointState;
using IMUState = opensprite_msgs::msg::IMUState;
using LowCmd = opensprite_msgs::msg::LowCmd;
using LowState = opensprite_msgs::msg::LowState;
using ControlRequest = opensprite_msgs::msg::ControlRequest;
using ControlReply = opensprite_msgs::msg::ControlReply;
using LowCmd_ = LowCmd;
using LowState_ = LowState;
using ControlRequest_ = ControlRequest;
using ControlReply_ = ControlReply;
}  // namespace msg

template<class T> struct MessageTraits { static constexpr bool supported = false; };
#define OPENSPRITE_MESSAGE_TRAITS(TYPE) \
    template<> struct MessageTraits<msg::TYPE> { \
        static constexpr bool supported = true; \
        static constexpr const char* name = #TYPE "_"; \
    }
OPENSPRITE_MESSAGE_TRAITS(LowCmd);
OPENSPRITE_MESSAGE_TRAITS(LowState);
OPENSPRITE_MESSAGE_TRAITS(ControlRequest);
OPENSPRITE_MESSAGE_TRAITS(ControlReply);
#undef OPENSPRITE_MESSAGE_TRAITS

// Native robot gateways and C++ clients exchange generated objects directly.
// This interface never serializes through the local Python JSON C ABI.
template<class T>
class TypedEndpoint {
public:
    virtual ~TypedEndpoint() = default;
    virtual void Write(const T& message) = 0;
    virtual bool Take(T& message, unsigned timeout_ms = 0) = 0;
    virtual int MatchedCount() = 0;
};

namespace detail {
template<class T>
std::shared_ptr<TypedEndpoint<T>> CreateEndpointImpl(unsigned domain,
    const std::string& interface, const std::string& topic, const std::string& role);
template<class T>
std::shared_ptr<TypedEndpoint<T>> CreateEndpointWithHistoryImpl(unsigned domain,
    const std::string& interface, const std::string& topic, const std::string& role,
    unsigned reader_history);
}

template<class T>
std::shared_ptr<TypedEndpoint<T>> CreateEndpoint(unsigned domain,
    const std::string& interface, const std::string& topic, const std::string& role) {
    static_assert(MessageTraits<T>::supported, "unsupported OpenSprite topic message");
    return detail::CreateEndpointImpl<T>(domain, interface, topic, role);
}

template<class T>
std::shared_ptr<TypedEndpoint<T>> CreateEndpoint(unsigned domain,
    const std::string& interface, const std::string& topic, const std::string& role,
    unsigned reader_history) {
    static_assert(MessageTraits<T>::supported, "unsupported OpenSprite topic message");
    return detail::CreateEndpointWithHistoryImpl<T>(domain, interface, topic, role, reader_history);
}

}  // namespace opensprite_sdk
