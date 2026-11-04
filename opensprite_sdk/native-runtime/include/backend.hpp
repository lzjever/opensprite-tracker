#pragma once
// OpenSprite device interface. SPDX-License-Identifier: AGPL-3.0-only
#include <OpenSprite.h>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <memory>
#include <string>

namespace opensprite::runtime {
namespace wire = opensprite_msgs::msg;
constexpr std::uint64_t kJointOrderHash = 17325335851961314422ULL;
constexpr std::size_t kJointCount = 31;

class Backend {
public:
    virtual ~Backend() = default;
    // Poll never waits for network/device I/O. Sequence increases only for new
    // device snapshots; raw sensor timestamps remain in the robot clock.
    virtual bool Poll(wire::LowState& state) = 0;
    virtual void Enable() = 0;
    // Returns after the device backend accepts the command, not motor execution.
    virtual void Submit(const wire::LowCmd& command) = 0;
    virtual void Tick() = 0;
    virtual void Disable() noexcept = 0;
    virtual bool IsSimulation() const = 0;
    virtual bool IsShadow() const = 0;
    virtual std::uint64_t PeriodNs() const = 0;
    // Command dispatch and simulator integration have independent clocks.
    virtual std::uint64_t PhysicsPeriodNs() const { return 0; }
    virtual void PhysicsStep() {}
    virtual std::string Profile() const = 0;
    virtual std::string ClockDomain() const = 0;
    virtual nlohmann::json Diagnostics() const { return nlohmann::json::object(); }
};

std::uint64_t MonotonicNs();
std::unique_ptr<Backend> MakeHardwareBackend(const std::string& config, bool actuate);
// Implemented by the independent MuJoCo adapter.
std::unique_ptr<Backend> MakeMujocoBackend(const std::string& config);
} // namespace opensprite::runtime
