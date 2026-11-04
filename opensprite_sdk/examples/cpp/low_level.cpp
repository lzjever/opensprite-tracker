// SPDX-License-Identifier: AGPL-3.0-only
#include <opensprite_sdk/client.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    unsigned domain=1, rate=50;
    std::string interface="lo";
    double duration=2., kp=2., kd=.12;
    bool hold=false;
    try {
        const auto unsigned_value=[](const std::string& value) {
            if (value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)
                throw std::invalid_argument("expected an unsigned integer");
            const auto result=std::stoull(value);
            if (result>1000000) throw std::invalid_argument("integer argument is too large");
            return static_cast<unsigned>(result);
        };
        const auto numeric_value=[](const std::string& value) {
            std::size_t end=0;
            const auto result=std::stod(value,&end);
            if (end!=value.size()) throw std::invalid_argument("invalid numeric argument");
            return result;
        };
        for (int i=1; i<argc; ++i) {
            const std::string option=argv[i];
            if (option=="--hold") {hold=true; continue;}
            if (option=="--help") {
                std::cout << "opensprite_low_level [--domain N] [--interface NIC] [--duration SECONDS]\n"
                             "  Default: read sensor state only. --hold explicitly claims and enables\n"
                             "  a measured-pose hold using --kp GAIN --kd GAIN --rate-hz 50|500.\n";
                return 0;
            }
            if (i+1>=argc) throw std::invalid_argument("missing value for "+option);
            const std::string value=argv[++i];
            if (option=="--domain") domain=unsigned_value(value);
            else if (option=="--interface") interface=value;
            else if (option=="--duration") duration=numeric_value(value);
            else if (option=="--rate-hz") rate=unsigned_value(value);
            else if (option=="--kp") kp=numeric_value(value);
            else if (option=="--kd") kd=numeric_value(value);
            else throw std::invalid_argument("unknown option: "+option);
        }
        if (domain>232 || (rate!=50 && rate!=500) || !std::isfinite(duration) || duration<=0 || duration>10
            || !std::isfinite(kp) || kp<0 || !std::isfinite(kd) || kd<0)
            throw std::invalid_argument("invalid domain, rate, duration or gains");
        opensprite_sdk::ChannelFactory::Instance()->Init(static_cast<int>(domain),interface);
        opensprite_sdk::ManagementClient robot;
        robot.Init();
        opensprite_sdk::msg::LowState state;
        if (!robot.Read(state,10000)) throw std::runtime_error("robot state timeout");
        std::array<double,opensprite_sdk::JointCount> target{}, stiffness{}, damping{};
        for (std::size_t i=0;i<target.size();++i) target[i]=state.joint_state()[i].q();
        stiffness.fill(kp); damping.fill(kd);
        bool enabled=false;
        std::uint64_t commands=0, observations=1, deadline_misses=0;
        const auto robot_session=state.session_id().to_string();
        const auto robot_clock=state.clock_domain().to_string();
        try {
            if (hold) {robot.Request("claim"); robot.Request("enable"); enabled=true;}
            const auto period=std::chrono::nanoseconds(1000000000ULL/rate);
            auto next=std::chrono::steady_clock::now();
            auto last_received=next;
            const auto end=next+std::chrono::duration<double>(duration);
            while (std::chrono::steady_clock::now()<end) {
                // The cyclic writer has its own clock. A200Hz simulator may
                // legitimately supply no new sample on some500Hz write ticks.
                // Preserve the latest actual source sequence and sample times.
                opensprite_sdk::msg::LowState incoming;
                if (robot.Read(incoming,0)) {
                    if (incoming.session_id()!=robot_session || incoming.clock_domain()!=robot_clock)
                        throw std::runtime_error("robot session/clock changed");
                    if (incoming.seq()<state.seq()) throw std::runtime_error("robot state sequence regressed");
                    if (incoming.seq()>state.seq()) ++observations;
                    state=std::move(incoming);
                    last_received=std::chrono::steady_clock::now();
                }
                if (std::chrono::steady_clock::now()-last_received>=std::chrono::milliseconds(100))
                    throw std::runtime_error("robot state stream stopped");
                if (hold) {
                    if (!robot.Write(opensprite_sdk::JointCommand(state,target,stiffness,damping)))
                        throw std::runtime_error("command publisher closed");
                    ++commands;
                }
                next+=period;
                if (std::chrono::steady_clock::now()>next+period) {
                    ++deadline_misses;
                    next=std::chrono::steady_clock::now()+period;
                }
                std::this_thread::sleep_until(next);
            }
            if (enabled) {
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
                while (state.accepted_command_seq()<robot.LastCommandSequence()
                       || state.accepted_controller_session()!=robot.ControllerSession()) {
                    if (!robot.Read(state,100) || std::chrono::steady_clock::now()>deadline)
                        throw std::runtime_error("last command acceptance timeout");
                }
                const auto reply=robot.Request("disable");
                std::cout << "disable accepted=" << reply.accepted()
                          << " execution_verified=" << reply.execution_verified() << '\n';
                enabled=false;
            }
        } catch (...) {
            if (enabled) {try {robot.Request("disable",1000);} catch (...) {}}
            throw;
        }
        std::cout << "profile=" << state.profile_id().to_string() << " state_seq=" << state.seq()
                  << " commands=" << commands << " observations=" << observations
                  << " deadline_misses=" << deadline_misses
                  << " clock=" << state.clock_domain().to_string() << '\n';
        robot.Close();
    } catch (const std::exception& error) {
        std::cerr << "OpenSprite SDK: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
