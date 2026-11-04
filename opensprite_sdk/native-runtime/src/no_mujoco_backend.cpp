// SPDX-License-Identifier: AGPL-3.0-only
#include "backend.hpp"
#include <stdexcept>
namespace opensprite::runtime {
std::unique_ptr<Backend> MakeMujocoBackend(const std::string&) {
    throw std::runtime_error("this hardware-only executable was built without MuJoCo");
}
}
