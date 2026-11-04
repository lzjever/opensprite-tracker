// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <opensprite_sdk/core/typed_endpoint.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace opensprite_sdk {

namespace detail {
// A callback may destroy its own subscriber. Join those workers from an owned
// reaper, including at process exit, rather than detaching DDS cleanup.
class ThreadReaper {
public:
    static ThreadReaper& Instance() { static ThreadReaper reaper; return reaper; }
    void Retire(std::thread worker) {
        { std::lock_guard<std::mutex> lock(mutex_); workers_.push_back(std::move(worker)); }
        condition_.notify_one();
    }
    ~ThreadReaper() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_=true; }
        condition_.notify_all();
        worker_.join();
    }
private:
    ThreadReaper() : worker_([this] {
        for (;;) {
            std::thread retired;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock,[this] {return stopping_ || !workers_.empty();});
                if (workers_.empty()) return;
                retired=std::move(workers_.front()); workers_.pop_front();
            }
            retired.join();
        }
    }) {}
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::thread> workers_;
    bool stopping_=false;
    std::thread worker_;
};
}  // namespace detail

inline std::uint64_t MonotonicNanoseconds() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline std::string NewSessionId() {
    std::random_device random;
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned i = 0; i < 4; ++i) output << std::setw(8) << random();
    return output.str();
}

class ChannelFactory {
public:
    struct Config { unsigned domain = 0; std::string interface = "lo"; };
    static ChannelFactory* Instance() { static ChannelFactory factory; return &factory; }
    void Init(int domain, const std::string& interface = "lo") {
        if (domain < 0 || domain > 232) throw std::invalid_argument("DDS domain must be in0..232");
        if (interface.size() > 128 || interface.find('\0') != std::string::npos)
            throw std::invalid_argument("invalid DDS interface");
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = {static_cast<unsigned>(domain), interface.empty() ? "lo" : interface};
        initialized_ = true;
    }
    Config GetConfig() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) throw std::logic_error("initialize ChannelFactory before creating channels");
        return config_;
    }
    void Release() { std::lock_guard<std::mutex> lock(mutex_); initialized_ = false; }
private:
    ChannelFactory() = default;
    mutable std::mutex mutex_;
    bool initialized_ = false;
    Config config_;
};

inline void ChannelFactoryInitialize(int domain = 0, const std::string& interface = "lo") {
    ChannelFactory::Instance()->Init(domain, interface);
}

template<class T>
class ChannelPublisher {
    static_assert(MessageTraits<T>::supported, "unsupported OpenSprite message");
public:
    explicit ChannelPublisher(std::string topic) : topic_(std::move(topic)), session_(NewSessionId()) {}
    ~ChannelPublisher() { CloseChannel(); }
    ChannelPublisher(const ChannelPublisher&) = delete;
    ChannelPublisher& operator=(const ChannelPublisher&) = delete;

    void InitChannel() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (endpoint_) return;
        const auto config = ChannelFactory::Instance()->GetConfig();
        endpoint_ = CreateEndpoint<T>(config.domain, config.interface, topic_, "publisher");
    }
    void Init() { InitChannel(); }
    bool Write(const T& message, std::int64_t wait_microseconds = 0) {
        if (wait_microseconds < 0 || wait_microseconds>10000000)
            throw std::invalid_argument("publication match wait must be in0..10000000us");
        if (wait_microseconds && !WaitForMatch(static_cast<unsigned>((wait_microseconds+999)/1000)))
            return false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (!endpoint_) return false;
        T outgoing = message;
        if (outgoing.seq() == 0) outgoing.seq(sequence_+1);
        if (outgoing.session_id().size() == 0) outgoing.session_id(session_);
        if (outgoing.clock_domain().size() == 0) outgoing.clock_domain("monotonic");
        if (outgoing.stamp_ns() == 0 && outgoing.clock_domain() == "monotonic")
            outgoing.stamp_ns(MonotonicNanoseconds());
        const bool epoch_changed = std::is_same<T, msg::LowState>::value && outgoing.session_id() != session_;
        if (outgoing.seq() <= sequence_ && !epoch_changed)
            throw std::invalid_argument("publisher sequence must increase");
        if (outgoing.session_id() != session_ && sequence_ && !epoch_changed)
            throw std::invalid_argument("publisher controller session changed");
        endpoint_->Write(outgoing);
        sequence_ = outgoing.seq();
        session_ = outgoing.session_id().to_string();
        // DDS acceptance does not acknowledge receiver or motor execution.
        // Native reliable publication has a fixed bounded10ms QoS wait.
        return true;
    }
    void CloseChannel() { std::lock_guard<std::mutex> lock(mutex_); endpoint_.reset(); }
    void Close() { CloseChannel(); }
    int MatchedCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return endpoint_ ? endpoint_->MatchedCount() : 0;
    }
    bool WaitForMatch(unsigned timeout_ms = 2000) const {
        const auto deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
        do {
            if (MatchedCount() > 0) return true;
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (true);
    }
    const std::string& GetChannelName() const { return topic_; }
private:
    std::string topic_;
    mutable std::mutex mutex_;
    std::shared_ptr<TypedEndpoint<T>> endpoint_;
    std::uint64_t sequence_ = 0;
    std::string session_;
};

template<class T>
class ChannelSubscriber {
    static_assert(MessageTraits<T>::supported, "unsupported OpenSprite message");
    struct State {
        std::shared_ptr<TypedEndpoint<T>> endpoint;
        std::atomic<bool> active{true};
        std::mutex mutex;
        std::condition_variable condition;
        std::deque<T> queue;
        std::size_t capacity = 1;
        std::function<void(const T&)> handler;
        std::exception_ptr error;
        std::atomic<std::int64_t> last_available{-1};
        std::thread receiver, dispatcher;
        std::mutex join_mutex;
    };
public:
    explicit ChannelSubscriber(std::string topic) : topic_(std::move(topic)) {}
    ~ChannelSubscriber() {
        try {
            auto state=GetState();
            if (state && current_worker_==state.get()) {
                state->active=false; state->condition.notify_all();
                for (auto* worker : {&state->receiver,&state->dispatcher})
                    if (worker->joinable()) detail::ThreadReaper::Instance().Retire(std::move(*worker));
            } else CloseChannel();
        } catch (...) {}
    }
    ChannelSubscriber(const ChannelSubscriber&) = delete;
    ChannelSubscriber& operator=(const ChannelSubscriber&) = delete;

    void InitChannel(std::function<void(const T&)> handler = {}, std::int64_t queue_length = 1) {
        if (queue_length < 0 || queue_length > 1024) throw std::invalid_argument("subscriber queue must be in0..1024");
        std::lock_guard<std::mutex> lifecycle(lifecycle_);
        if (state_) throw std::logic_error("subscriber is already initialized");
        const auto config = ChannelFactory::Instance()->GetConfig();
        auto state = std::make_shared<State>();
        state->capacity = static_cast<std::size_t>(std::max<std::int64_t>(1, queue_length));
        state->endpoint = CreateEndpoint<T>(config.domain, config.interface, topic_, "subscriber",
            static_cast<unsigned>(std::min<std::size_t>(state->capacity,64)));
        state->handler = std::move(handler);
        state_ = state;
        if (state->handler) {
            try {
                state->receiver = std::thread([state] { Receive(state); });
                state->dispatcher = std::thread([state] { Dispatch(state); });
            } catch (...) {
                state->active = false;
                state->condition.notify_all();
                if (state->receiver.joinable()) state->receiver.join();
                state_.reset();
                throw;
            }
        }
    }
    void Init(std::function<void(const T&)> handler = {}, std::int64_t queue_length = 1) {
        InitChannel(std::move(handler), queue_length);
    }
    void InitChannel(std::int64_t queue_length) {
        InitChannel(std::function<void(const T&)>{}, queue_length);
    }
    template<class Handler, std::enable_if_t<std::is_invocable_v<Handler,const void*>
        && !std::is_invocable_v<Handler,const T&>,int> = 0>
    void InitChannel(Handler handler, std::int64_t queue_length = 1) {
        std::function<void(const void*)> callback=std::move(handler);
        if (!callback) throw std::invalid_argument("subscriber callback is empty");
        InitChannel(std::function<void(const T&)>([callback=std::move(callback)](const T& value) {
            callback(static_cast<const void*>(&value));
        }), queue_length);
    }
    bool Read(T& message, unsigned timeout_ms = 0) {
        if (timeout_ms > 10000) throw std::invalid_argument("read timeout exceeds10 seconds");
        auto state = GetState();
        if (!state) return false;
        if (state->handler) throw std::logic_error("Read and callback mode cannot consume the same subscriber");
        const auto deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
        do {
            if (!state->active) return false;
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline-std::chrono::steady_clock::now()).count();
            const auto slice = static_cast<unsigned>(std::max<std::int64_t>(0, std::min<std::int64_t>(100, remaining)));
            if (state->endpoint->Take(message, slice)) {
                state->last_available = static_cast<std::int64_t>(MonotonicNanoseconds());
                return state->active.load();
            }
        } while (state->active && std::chrono::steady_clock::now() < deadline);
        return false;
    }
    void CloseChannel() {
        auto state=GetState();
        if (!state) return;
        state->active = false;
        state->condition.notify_all();
        // Keep the worker handle owned until a non-callback caller joins it.
        // Taking join_mutex here from the callback could deadlock an external
        // Close already joining this callback.
        if (current_worker_==state.get()) return;
        std::lock_guard<std::mutex> joining(state->join_mutex);
        for (auto* worker : {&state->receiver, &state->dispatcher}) {
            if (!worker->joinable()) continue;
            worker->join();
        }
        std::lock_guard<std::mutex> lifecycle(lifecycle_);
        if (state_==state) state_.reset();
    }
    void Close() { CloseChannel(); }
    int MatchedCount() const { auto state = GetState(); return state && state->active ? state->endpoint->MatchedCount() : 0; }
    bool WaitForMatch(unsigned timeout_ms = 2000) const {
        const auto deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
        do {
            if (MatchedCount() > 0) return true;
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (true);
    }
    std::int64_t GetLastDataAvailableTime() const {
        auto state = GetState(); return state ? state->last_available.load() : -1;
    }
    std::exception_ptr LastError() const {
        auto state = GetState();
        if (!state) return {};
        std::lock_guard<std::mutex> lock(state->mutex);
        return state->error;
    }
    void RethrowLastError() const { if (auto error = LastError()) std::rethrow_exception(error); }
    const std::string& GetChannelName() const { return topic_; }
private:
    static void Fail(const std::shared_ptr<State>& state) {
        { std::lock_guard<std::mutex> lock(state->mutex); state->error = std::current_exception(); }
        state->active = false;
        state->condition.notify_all();
    }
    static void Receive(const std::shared_ptr<State>& state) {
        current_worker_=state.get();
        try {
            while (state->active) {
                T message;
                if (!state->endpoint->Take(message, 100)) continue;
                state->last_available = static_cast<std::int64_t>(MonotonicNanoseconds());
                std::lock_guard<std::mutex> lock(state->mutex);
                if (state->queue.size() == state->capacity) state->queue.pop_front();
                state->queue.push_back(std::move(message));
                state->condition.notify_all();
            }
        } catch (...) { Fail(state); }
        current_worker_=nullptr;
    }
    static void Dispatch(const std::shared_ptr<State>& state) {
        current_worker_=state.get();
        try {
            while (state->active) {
                T message;
                {
                    std::unique_lock<std::mutex> lock(state->mutex);
                    state->condition.wait(lock, [&] { return !state->active || !state->queue.empty(); });
                    if (!state->active) break;
                    message = std::move(state->queue.front());
                    state->queue.pop_front();
                }
                state->handler(message);
            }
        } catch (...) { Fail(state); }
        current_worker_=nullptr;
    }
    std::shared_ptr<State> GetState() const { std::lock_guard<std::mutex> lock(lifecycle_); return state_; }
    std::string topic_;
    mutable std::mutex lifecycle_;
    std::shared_ptr<State> state_;
    inline static thread_local const State* current_worker_=nullptr;
};

template<class T> using ChannelPublisherPtr = std::shared_ptr<ChannelPublisher<T>>;
template<class T> using ChannelSubscriberPtr = std::shared_ptr<ChannelSubscriber<T>>;

}  // namespace opensprite_sdk
