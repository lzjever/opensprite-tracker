#pragma once
// SPDX-License-Identifier: AGPL-3.0-only
#include "backend.hpp"
#include <atomic>
#include <string>

namespace opensprite::runtime {
struct RuntimeOptions {
    unsigned domain=1;
    std::string interface="lo";
    bool allow_enable=false;
    double duration=240.;
    std::string truth_output;
};
nlohmann::json RunRobot(std::unique_ptr<Backend> backend, const RuntimeOptions& options,
                        const std::atomic<bool>* stop=nullptr);
} // namespace opensprite::runtime
