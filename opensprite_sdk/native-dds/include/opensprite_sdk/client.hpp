// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <opensprite_sdk/core/channel.hpp>
#include <opensprite_sdk/constants.hpp>
#include <array>
#include <cmath>

namespace opensprite_sdk {

template<class Target>
inline void CopyRobotHeader(const msg::LowState& state, Target& target) {
    target.profile_id(state.profile_id());
    target.joint_order_hash(state.joint_order_hash());
    target.clock_domain(state.clock_domain());
    target.stamp_ns(state.stamp_ns());
}

inline msg::LowCmd JointCommand(const msg::LowState& state,
    const std::array<double, JointCount>& q,
    const std::array<double, JointCount>& kp,
    const std::array<double, JointCount>& kd,
    const std::array<double, JointCount>& dq = {},
    const std::array<double, JointCount>& tau = {}) {
    if (state.joint_order_hash() != JointOrderHash || state.session_id().size() == 0 || state.seq() == 0)
        throw std::invalid_argument("command needs a valid OpenSprite source state");
    msg::LowCmd command;
    CopyRobotHeader(state, command);
    command.source_state_seq(state.seq());
    command.source_state_session_id(state.session_id());
    for (std::size_t i=0; i<JointCount; ++i) {
        if (!std::isfinite(q[i]) || !std::isfinite(dq[i]) || !std::isfinite(kp[i])
            || !std::isfinite(kd[i]) || !std::isfinite(tau[i]) || kp[i]<0 || kd[i]<0)
            throw std::invalid_argument("joint command has invalid values");
        auto& target = command.joint_cmd()[i];
        target.mode(1); target.q(q[i]); target.dq(dq[i]); target.kp(kp[i]); target.kd(kd[i]); target.tau(tau[i]);
    }
    return command;
}

class LowLevelClient {
public:
    explicit LowLevelClient(std::string controller_session = {})
        : controller_(controller_session.empty() ? NewSessionId() : std::move(controller_session)),
          publisher_(LowCmdTopic), subscriber_(LowStateTopic) {
        if (controller_.size()>128 || controller_.find('\0')!=std::string::npos)
            throw std::invalid_argument("invalid controller session");
    }
    ~LowLevelClient() { try { Close(); } catch (...) {} }
    void Init() {
        subscriber_.InitChannel();
        try { publisher_.InitChannel(); } catch (...) { subscriber_.CloseChannel(); throw; }
    }
    bool Read(msg::LowState& state, unsigned timeout_ms=1000) { return subscriber_.Read(state, timeout_ms); }
    bool Write(msg::LowCmd command) {
        if (command.session_id().size() && command.session_id()!=controller_)
            throw std::invalid_argument("command controller differs from this client");
        command.session_id(controller_);
        if (command.seq()==0) command.seq(last_sequence_+1);
        if (!publisher_.Write(command)) return false;
        last_sequence_ = command.seq();
        return true;
    }
    void Close() { publisher_.CloseChannel(); subscriber_.CloseChannel(); }
    const std::string& ControllerSession() const { return controller_; }
    std::uint64_t LastCommandSequence() const { return last_sequence_; }
    ChannelPublisher<msg::LowCmd>& Publisher() { return publisher_; }
    ChannelSubscriber<msg::LowState>& Subscriber() { return subscriber_; }
private:
    std::string controller_;
    std::uint64_t last_sequence_=0;
    ChannelPublisher<msg::LowCmd> publisher_;
    ChannelSubscriber<msg::LowState> subscriber_;
};

class ManagementClient {
public:
    explicit ManagementClient(std::string controller_session = {})
        : robot_(std::move(controller_session)), requests_(ControlRequestTopic), replies_(ControlReplyTopic) {}
    ~ManagementClient() { try { Close(); } catch (...) {} }
    void Init() {
        try { robot_.Init(); replies_.InitChannel(); requests_.InitChannel(); }
        catch (...) { Close(); throw; }
    }
    bool Read(msg::LowState& state, unsigned timeout_ms=1000) {
        if (!robot_.Read(state, timeout_ms)) return false;
        state_=state; have_state_=true; return true;
    }
    bool Write(const msg::LowCmd& command) { return robot_.Write(command); }
    msg::ControlReply Request(const std::string& operation, unsigned timeout_ms=3000) {
        if (!have_state_) {
            msg::LowState state;
            if (!Read(state, timeout_ms)) throw std::runtime_error("robot state timeout");
        }
        return Request(operation, state_, timeout_ms);
    }
    msg::ControlReply Request(const std::string& operation, const msg::LowState& state,
                              unsigned timeout_ms=3000) {
        if (timeout_ms==0 || timeout_ms>10000) throw std::invalid_argument("management timeout must be in1..10000ms");
        if (operation!="claim" && operation!="enable" && operation!="disable" && operation!="release")
            throw std::invalid_argument("unsupported control operation");
        if (state.session_id().size()==0 || state.joint_order_hash()!=JointOrderHash)
            throw std::invalid_argument("management requires a valid robot source state");
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
        if (operation=="enable") {
            while (robot_.Publisher().MatchedCount()==0 || robot_.Subscriber().MatchedCount()==0
                    || requests_.MatchedCount()==0 || replies_.MatchedCount()==0) {
                if (std::chrono::steady_clock::now()>=deadline)
                    throw std::runtime_error("DDS endpoints did not match before enable");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        msg::ControlRequest request;
        CopyRobotHeader(state, request);
        request.session_id(robot_.ControllerSession());
        request.request_id(NewSessionId());
        request.controller_session(robot_.ControllerSession());
        request.robot_session(state.session_id());
        request.operation(operation);
        auto retry_at=std::chrono::steady_clock::time_point::min();
        do {
            if (std::chrono::steady_clock::now()>=retry_at) {
                if (!requests_.Write(request)) throw std::runtime_error("control publisher is closed");
                retry_at=std::chrono::steady_clock::now()+std::chrono::milliseconds(200);
            }
            msg::ControlReply reply;
            const auto remaining=std::chrono::duration_cast<std::chrono::milliseconds>(deadline-std::chrono::steady_clock::now()).count();
            if (!replies_.Read(reply, static_cast<unsigned>(std::max<std::int64_t>(0,std::min<std::int64_t>(20,remaining)))))
                continue;
            if (reply.request_id()!=request.request_id() || reply.controller_session()!=robot_.ControllerSession()
                || reply.robot_session()!=state.session_id()) continue;
            if (!reply.accepted()) throw std::runtime_error(operation+" rejected: "+reply.detail().to_string());
            return reply;
        } while (std::chrono::steady_clock::now()<deadline);
        throw std::runtime_error("no correlated "+operation+" acknowledgement");
    }
    void Close() { requests_.CloseChannel(); replies_.CloseChannel(); robot_.Close(); }
    const std::string& ControllerSession() const { return robot_.ControllerSession(); }
    std::uint64_t LastCommandSequence() const { return robot_.LastCommandSequence(); }
private:
    LowLevelClient robot_;
    ChannelPublisher<msg::ControlRequest> requests_;
    ChannelSubscriber<msg::ControlReply> replies_;
    bool have_state_=false;
    msg::LowState state_;
};

}  // namespace opensprite_sdk
