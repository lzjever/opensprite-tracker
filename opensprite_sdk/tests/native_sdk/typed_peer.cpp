// Independent consumer of installed public headers. SPDX-License-Identifier: AGPL-3.0-only
#include <opensprite_sdk/core/channel.hpp>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace sdk = opensprite_sdk;
namespace msg = opensprite_sdk::msg;
using Clock = std::chrono::steady_clock;

template<class T>
struct Echo {
    sdk::ChannelSubscriber<T> reader;
    sdk::ChannelPublisher<T> writer;
    unsigned received = 0;
    T latest;
    bool sent = false;

    Echo(const std::string& prefix, const std::string& name, bool ros)
        : reader("rt/" + prefix + (ros ? "/ros/" : "/to_cpp/") + name),
          writer("rt/" + prefix + (ros ? "/native/" : "/from_cpp/") + name) {
        reader.InitChannel();
        writer.InitChannel();
    }

    void poll() {
        T value;
        if (reader.Read(value, 0)) {
            latest = std::move(value);
            ++received;
        }
        if (received && !sent && writer.MatchedCount() > 0) {
            if (!writer.Write(latest)) throw std::runtime_error("typed echo write rejected");
            sent = true;
        }
    }

    ~Echo() { reader.CloseChannel(); writer.CloseChannel(); }
};

int echo(const std::string& prefix, bool ros = false) {
    Echo<msg::LowCmd> commands(prefix, "LowCmd", ros);
    Echo<msg::LowState> states(prefix, "LowState", ros);
    Echo<msg::ControlRequest> requests(prefix, "ControlRequest", ros);
    Echo<msg::ControlReply> replies(prefix, "ControlReply", ros);
    std::cout << "ready" << std::endl;
    const auto end = Clock::now() + std::chrono::seconds(12);
    while (Clock::now() < end) {
        commands.poll(); states.poll(); requests.poll(); replies.poll();
        if (commands.sent && states.sent && requests.sent && replies.sent) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            std::cout << "{\"passed\":true,\"message_types\":4}" << std::endl;
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("typed echo did not receive all four message types");
}

int callback(const std::string& prefix, bool throws) {
    sdk::ChannelSubscriber<msg::LowCmd> reader("rt/" + prefix + "/callback");
    sdk::ChannelPublisher<msg::LowCmd> writer("rt/" + prefix + "/callback");
    std::atomic<unsigned> called{0};
    reader.InitChannel([&](const msg::LowCmd&) {
        ++called;
        if (throws) throw std::runtime_error("intentional callback failure");
        reader.CloseChannel();
    }, 1);
    writer.InitChannel();
    if (!writer.WaitForMatch(3000)) throw std::runtime_error("callback endpoints did not match");
    msg::LowCmd value;
    value.seq(0);
    value.profile_id("legacy_v4");
    value.joint_order_hash(17325335851961314422ULL);
    value.session_id("installed-cpp-callback");
    const auto end = Clock::now() + std::chrono::seconds(3);
    while (!called && Clock::now() < end) {
        if (!writer.Write(value)) throw std::runtime_error("callback write failed");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!called) throw std::runtime_error("callback did not run");
    if (throws) {
        bool caught = false;
        while (!caught && Clock::now() < end) {
            try { reader.RethrowLastError(); }
            catch (const std::runtime_error&) { caught = true; }
            if (!caught) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!caught) throw std::runtime_error("callback exception was lost");
    }
    reader.CloseChannel();
    writer.CloseChannel();
    std::cout << "{\"passed\":true,\"callback_calls\":" << called.load() << "}" << std::endl;
    return 0;
}

int destroy_from_callback(const std::string& prefix) {
    auto reader = std::make_unique<sdk::ChannelSubscriber<msg::LowCmd>>("rt/"+prefix+"/destroy");
    sdk::ChannelPublisher<msg::LowCmd> writer("rt/"+prefix+"/destroy");
    std::atomic<bool> destroyed{false};
    reader->InitChannel([&](const msg::LowCmd&) {
        reader.reset();
        destroyed = true;
    }, 1);
    writer.InitChannel();
    if (!writer.WaitForMatch(3000)) throw std::runtime_error("self-destruction endpoints did not match");
    msg::LowCmd value;
    value.profile_id("legacy_v4");
    value.joint_order_hash(17325335851961314422ULL);
    value.session_id("installed-cpp-self-destroy");
    const auto end = Clock::now()+std::chrono::seconds(3);
    while (!destroyed && Clock::now()<end) {
        writer.Write(value);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!destroyed) throw std::runtime_error("callback self-destruction did not finish");
    writer.CloseChannel();
    std::cout << "{\"passed\":true,\"destroyed_from_callback\":true}" << std::endl;
    return 0;
}

int state_stream(const std::string& topic) {
    sdk::ChannelPublisher<msg::LowState> writer(topic);
    writer.InitChannel();
    if (!writer.WaitForMatch(3000)) throw std::runtime_error("state stream has no matched reader");
    msg::LowState state;
    state.profile_id("legacy_v4"); state.joint_order_hash(17325335851961314422ULL);
    state.session_id("native-history-source"); state.clock_domain("sim"); state.status("disabled");
    state.imu_state().valid(true); state.imu_state().frame("body");
    state.imu_state().attached_link("waist_yaw_link");
    state.imu_state().sample_time_kind("simulation");
    state.imu_state().projected_gravity()[2]=-1.;
    const auto begin = Clock::now();
    for (unsigned frame=0; frame<120; ++frame) {
        std::this_thread::sleep_until(begin+std::chrono::milliseconds(5*frame));
        state.seq(frame+1); state.stamp_ns(static_cast<std::uint64_t>(frame)*5000000ULL);
        state.imu_state().sample_ns(state.stamp_ns());
        for (auto& joint : state.joint_state()) {
            joint.valid(true); joint.sample_ns(state.stamp_ns()); joint.sample_time_kind("simulation");
        }
        state.joint_state()[0].q(frame*.001);
        if (!writer.Write(state)) throw std::runtime_error("state stream write failed");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::cout << "{\"passed\":true,\"samples\":120}" << std::endl;
    // Keep the writer alive while the consumer inspects its receive history.
    // Fast DDS may purge unread data when a volatile writer is removed.
    std::string release;
    std::getline(std::cin, release);
    writer.CloseChannel();
    return 0;
}

int state_history(const std::string& topic) {
    sdk::ChannelSubscriber<msg::LowState> reader(topic);
    reader.InitChannel({}, 64);
    std::cout << "ready" << std::endl;
    if (!reader.WaitForMatch(3000)) throw std::runtime_error("history reader has no matched source");
    std::this_thread::sleep_for(std::chrono::milliseconds(650));
    msg::LowState state;
    unsigned count=0;
    std::uint64_t expected=57;
    while (reader.Read(state, 0)) {
        if (state.seq()!=expected || state.stamp_ns()!=(expected-1)*5000000ULL)
            throw std::runtime_error("installed C++ history is not the real bounded FIFO");
        ++expected; ++count;
    }
    if (count!=64) throw std::runtime_error("installed C++ history lost observations");
    reader.CloseChannel();
    std::cout << "{\"passed\":true,\"retained\":64}" << std::endl;
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::invalid_argument("usage: typed_peer DOMAIN PREFIX MODE");
        sdk::ChannelFactory::Instance()->Init(std::stoi(argv[1]), "lo");
        const std::string mode(argv[3]);
        if (mode == "echo") return echo(argv[2]);
        if (mode == "ros-echo") return echo(argv[2], true);
        if (mode == "callback-close") return callback(argv[2], false);
        if (mode == "callback-throw") return callback(argv[2], true);
        if (mode == "callback-destroy") return destroy_from_callback(argv[2]);
        if (mode == "state-stream") return state_stream(argv[2]);
        if (mode == "state-history") return state_history(argv[2]);
        throw std::invalid_argument("unknown test mode");
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 2;
    }
}
