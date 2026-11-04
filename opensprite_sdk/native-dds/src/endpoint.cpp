// OpenSprite typed Fast DDS adapter. SPDX-License-Identifier: AGPL-3.0-only
#include "OpenSpritePubSubTypes.h"
#include "wire_json.hpp"
#include "wire_validation.hpp"
#include <opensprite_sdk/core/typed_endpoint.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/core/status/PublicationMatchedStatus.hpp>
#include <fastdds/dds/core/status/SubscriptionMatchedStatus.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>
#include <fastrtps/utils/IPLocator.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace fdds = eprosima::fastdds::dds;
namespace wire = opensprite_msgs::msg;
using Json = nlohmann::json;
using Return = eprosima::fastrtps::types::ReturnCode_t;

namespace {
thread_local std::string last_error;

std::string interface_address(const std::string& interface) {
    if (interface.empty() || interface == "lo" || interface == "localhost") return "127.0.0.1";
    in_addr addr{};
    const bool literal = inet_pton(AF_INET, interface.c_str(), &addr) == 1;
    ifaddrs* addresses = nullptr;
    if (getifaddrs(&addresses) != 0) throw std::runtime_error("getifaddrs failed");
    std::unique_ptr<ifaddrs, decltype(&freeifaddrs)> guard(addresses, freeifaddrs);
    for (auto* item = addresses; item; item = item->ifa_next) {
        if (!item->ifa_addr || item->ifa_addr->sa_family != AF_INET) continue;
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(item->ifa_addr)->sin_addr, ip, sizeof(ip));
        if ((literal && interface == ip) || (!literal && interface == item->ifa_name)) return ip;
    }
    throw std::invalid_argument("DDS interface must name an existing IPv4 interface/address");
}

struct Context {
    fdds::DomainParticipant* participant = nullptr;
    std::mutex mutex;
    std::map<std::string, fdds::TypeSupport> types;
    struct TopicRef { fdds::Topic* topic; unsigned count; };
    std::map<std::string, TopicRef> topics;

    Context(unsigned domain, const std::string& ip) {
        fdds::DomainParticipantQos qos;
        qos.name("opensprite-sdk");
        qos.transport().use_builtin_transports = false;
        auto udp = std::make_shared<eprosima::fastdds::rtps::UDPv4TransportDescriptor>();
        udp->interfaceWhiteList.push_back(ip);
        qos.transport().user_transports.push_back(udp);
        // Unicast peers make loopback discovery work without multicast support.
        // No implicit network-wide transport is installed.
        if (ip == "127.0.0.1") {
            eprosima::fastrtps::rtps::Locator_t peer;
            peer.kind = LOCATOR_KIND_UDPv4;
            eprosima::fastrtps::rtps::IPLocator::setIPv4(peer, ip);
            qos.wire_protocol().builtin.initialPeersList.push_back(peer);
        }
        participant = fdds::DomainParticipantFactory::get_instance()->create_participant(domain, qos);
        if (!participant) throw std::runtime_error("Fast DDS participant creation failed");
    }

    ~Context() {
        if (participant) {
            participant->delete_contained_entities();
            fdds::DomainParticipantFactory::get_instance()->delete_participant(participant);
        }
    }

    template<class Support> std::string register_type() {
        fdds::TypeSupport support(new Support());
        const std::string name = support.get_type_name();
        if (!types.count(name)) {
            if (support.register_type(participant) != Return::RETCODE_OK)
                throw std::runtime_error("Fast DDS type registration failed");
            types.emplace(name, support);
        }
        return name;
    }

    fdds::Topic* acquire_topic(const std::string& name, const std::string& type) {
        auto found = topics.find(name);
        if (found != topics.end()) {
            if (found->second.topic->get_type_name() != type)
                throw std::invalid_argument("DDS topic already uses a different message type");
            ++found->second.count;
            return found->second.topic;
        }
        auto* created = participant->create_topic(name, type, fdds::TOPIC_QOS_DEFAULT);
        if (!created) throw std::runtime_error("Fast DDS topic creation failed");
        topics.emplace(name, TopicRef{created, 1});
        return created;
    }

    void release_topic(fdds::Topic* topic) {
        auto found = topics.find(topic->get_name());
        if (found != topics.end() && --found->second.count == 0) {
            participant->delete_topic(topic);
            topics.erase(found);
        }
    }
};

std::shared_ptr<Context> context(unsigned domain, const std::string& interface) {
    static std::mutex mutex;
    static std::map<std::string, std::weak_ptr<Context>> contexts;
    const auto ip = interface_address(interface);
    const auto key = std::to_string(domain) + ":" + ip;
    std::lock_guard<std::mutex> lock(mutex);
    auto shared = contexts[key].lock();
    if (!shared) {
        shared = std::make_shared<Context>(domain, ip);
        contexts[key] = shared;
    }
    return shared;
}

struct EndpointBase {
    virtual ~EndpointBase() = default;
    virtual void write(const Json&) = 0;
    virtual std::string take(unsigned) = 0;
    virtual int matches() = 0;
};

template<class T, class Support>
class Endpoint final : public EndpointBase, public opensprite_sdk::TypedEndpoint<T> {
    std::shared_ptr<Context> context_;
    std::string message_type_;
    fdds::Topic* topic_ = nullptr;
    fdds::Publisher* publisher_ = nullptr;
    fdds::Subscriber* subscriber_ = nullptr;
    fdds::DataWriter* writer_ = nullptr;
    fdds::DataReader* reader_ = nullptr;

    void cleanup() {
        auto* participant = context_->participant;
        if (publisher_) {
            if (writer_) publisher_->delete_datawriter(writer_);
            participant->delete_publisher(publisher_);
        }
        if (subscriber_) {
            if (reader_) subscriber_->delete_datareader(reader_);
            participant->delete_subscriber(subscriber_);
        }
        if (topic_) context_->release_topic(topic_);
        writer_ = nullptr; reader_ = nullptr; publisher_ = nullptr; subscriber_ = nullptr; topic_ = nullptr;
    }

public:
    Endpoint(unsigned domain, const std::string& interface, const std::string& topic,
             const std::string& message_type, const std::string& role, unsigned reader_history=1)
        : context_(context(domain, interface)), message_type_(message_type) {
        std::lock_guard<std::mutex> lock(context_->mutex);
        try {
            if (reader_history<1 || reader_history>64 || (role=="publisher" && reader_history!=1))
                throw std::invalid_argument("subscriber history must be in1..64; publisher history is1");
            const auto type_name = context_->template register_type<Support>();
            topic_ = context_->acquire_topic(topic, type_name);
            const bool management = message_type == "ControlRequest_" || message_type == "ControlReply_";
            if (role == "publisher") {
                publisher_ = context_->participant->create_publisher(fdds::PUBLISHER_QOS_DEFAULT);
                if (!publisher_) throw std::runtime_error("Fast DDS publisher creation failed");
                fdds::DataWriterQos qos;
                qos.history().kind = fdds::KEEP_LAST_HISTORY_QOS;
                qos.history().depth = 1;
                qos.durability().kind = fdds::VOLATILE_DURABILITY_QOS;
                qos.reliability().kind = management ? fdds::RELIABLE_RELIABILITY_QOS : fdds::BEST_EFFORT_RELIABILITY_QOS;
                qos.reliability().max_blocking_time = eprosima::fastrtps::Duration_t(0, 10000000);
                qos.resource_limits().max_samples = 8;
                qos.resource_limits().max_instances = 1;
                qos.resource_limits().max_samples_per_instance = 8;
                qos.resource_limits().allocated_samples = 1;
                qos.endpoint().history_memory_policy = eprosima::fastrtps::rtps::PREALLOCATED_WITH_REALLOC_MEMORY_MODE;
                writer_ = publisher_->create_datawriter(topic_, qos);
                if (!writer_) throw std::runtime_error("Fast DDS writer creation failed");
            } else if (role == "subscriber") {
                subscriber_ = context_->participant->create_subscriber(fdds::SUBSCRIBER_QOS_DEFAULT);
                if (!subscriber_) throw std::runtime_error("Fast DDS subscriber creation failed");
                fdds::DataReaderQos qos;
                qos.history().kind = fdds::KEEP_LAST_HISTORY_QOS;
                // Default remains latest1. Timed observation consumers may
                // explicitly retain a bounded chronological receive history.
                const auto samples = std::max<unsigned>(8,reader_history);
                qos.history().depth = static_cast<std::int32_t>(reader_history);
                // Fast DDS Data-sharing loans the writer's history and couples
                // reader retention to writer depth1. A history reader needs
                // its own transport copies; disabling built-in transports alone
                // does not disable Data-sharing delivery.
                if (reader_history>1) qos.data_sharing().off();
                qos.durability().kind = fdds::VOLATILE_DURABILITY_QOS;
                qos.reliability().kind = management ? fdds::RELIABLE_RELIABILITY_QOS : fdds::BEST_EFFORT_RELIABILITY_QOS;
                qos.resource_limits().max_samples = samples;
                qos.resource_limits().max_instances = 1;
                qos.resource_limits().max_samples_per_instance = samples;
                qos.resource_limits().allocated_samples = 1;
                qos.endpoint().history_memory_policy = eprosima::fastrtps::rtps::PREALLOCATED_WITH_REALLOC_MEMORY_MODE;
                reader_ = subscriber_->create_datareader(topic_, qos);
                if (!reader_) throw std::runtime_error("Fast DDS reader creation failed");
            } else throw std::invalid_argument("role must be publisher or subscriber");
        } catch (...) { cleanup(); throw; }
    }

    ~Endpoint() override {
        std::lock_guard<std::mutex> lock(context_->mutex);
        cleanup();
    }

    void write(const Json& envelope) override {
        if (!writer_) throw std::runtime_error("endpoint is not a publisher");
        if (!envelope.is_object() || envelope.size() != 3 || envelope.at("schema") != 1 || envelope.at("type") != message_type_)
            throw std::invalid_argument("invalid SDK message envelope");
        T typed = envelope.at("data").get<T>();
        Write(typed);
    }

    void Write(const T& message) override {
        if (!writer_) throw std::runtime_error("endpoint is not a publisher");
        wire::validate_wire(message);
        T typed = message;
        // Fast DDS 2.x's single-argument convenience overload returns bool.
        if (!writer_->write(&typed)) throw std::runtime_error("Fast DDS write failed or exceeded bounded blocking time");
    }

    std::string take(unsigned timeout_ms) override {
        T typed;
        return Take(typed, timeout_ms)
            ? Json{{"schema", 1}, {"type", message_type_}, {"data", typed}}.dump() : std::string{};
    }

    bool Take(T& typed, unsigned timeout_ms) override {
        if (timeout_ms > 10000) throw std::invalid_argument("DDS take timeout exceeds10 seconds");
        if (!reader_) throw std::runtime_error("endpoint is not a subscriber");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        do {
            fdds::SampleInfo info;
            const auto result = reader_->take_next_sample(&typed, &info);
            if (result == Return::RETCODE_OK && info.valid_data) {
                wire::validate_wire(typed);
                return true;
            }
            if (result != Return::RETCODE_OK && result != Return::RETCODE_NO_DATA)
                throw std::runtime_error("Fast DDS take failed");
            if (std::chrono::steady_clock::now() >= deadline) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (true);
        return false;
    }

    int MatchedCount() override { return matches(); }

    int matches() override {
        if (writer_) {
            fdds::PublicationMatchedStatus status;
            if (writer_->get_publication_matched_status(status) != Return::RETCODE_OK)
                throw std::runtime_error("Fast DDS publication match status failed");
            return status.current_count;
        }
        fdds::SubscriptionMatchedStatus status;
        if (reader_->get_subscription_matched_status(status) != Return::RETCODE_OK)
            throw std::runtime_error("Fast DDS subscription match status failed");
        return status.current_count;
    }
};
} // namespace

namespace opensprite_sdk { namespace detail {
#define OPENSPRITE_TYPED_FACTORY(TYPE) \
template<> std::shared_ptr<TypedEndpoint<msg::TYPE>> CreateEndpointImpl<msg::TYPE>( \
    unsigned domain, const std::string& interface, const std::string& topic, const std::string& role) { \
    if (domain>232 || topic.empty() || topic.size()>128 || topic.find('\0')!=std::string::npos) \
        throw std::invalid_argument("invalid typed DDS endpoint arguments"); \
    return std::make_shared<Endpoint<wire::TYPE, wire::TYPE##PubSubType>>( \
        domain, interface, topic, #TYPE "_", role); \
}
OPENSPRITE_TYPED_FACTORY(LowCmd)
OPENSPRITE_TYPED_FACTORY(LowState)
OPENSPRITE_TYPED_FACTORY(ControlRequest)
OPENSPRITE_TYPED_FACTORY(ControlReply)
#undef OPENSPRITE_TYPED_FACTORY
#define OPENSPRITE_HISTORY_FACTORY(TYPE) \
template<> std::shared_ptr<TypedEndpoint<msg::TYPE>> CreateEndpointWithHistoryImpl<msg::TYPE>( \
    unsigned domain, const std::string& interface, const std::string& topic, const std::string& role, \
    unsigned reader_history) { \
    if (domain>232 || topic.empty() || topic.size()>128 || topic.find('\0')!=std::string::npos) \
        throw std::invalid_argument("invalid typed DDS endpoint arguments"); \
    return std::make_shared<Endpoint<wire::TYPE, wire::TYPE##PubSubType>>( \
        domain, interface, topic, #TYPE "_", role, reader_history); \
}
OPENSPRITE_HISTORY_FACTORY(LowCmd)
OPENSPRITE_HISTORY_FACTORY(LowState)
OPENSPRITE_HISTORY_FACTORY(ControlRequest)
OPENSPRITE_HISTORY_FACTORY(ControlReply)
#undef OPENSPRITE_HISTORY_FACTORY
}} // namespace opensprite_sdk::detail

#define API extern "C" __attribute__((visibility("default")))
API const char* osdds_error() { return last_error.c_str(); }
API void* osdds_open_with_history(unsigned domain, const char* interface, const char* topic,
                     const char* message_type, const char* role, unsigned reader_history) {
    try {
        if (domain > 232 || !topic || !message_type || !role) throw std::invalid_argument("invalid endpoint arguments");
        const std::string type(message_type), name(topic), iface(interface ? interface : "lo");
        if (name.empty() || name.size() > 128) throw std::invalid_argument("invalid topic length");
        last_error.clear();
#define MAKE(TYPE) if (type == #TYPE "_") return new Endpoint<wire::TYPE, wire::TYPE##PubSubType>(domain, iface, name, type, role, reader_history)
        MAKE(LowCmd);
        MAKE(LowState);
        MAKE(ControlRequest);
        MAKE(ControlReply);
#undef MAKE
        throw std::invalid_argument("unsupported OpenSprite DDS message type");
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
API void* osdds_open(unsigned domain, const char* interface, const char* topic,
                     const char* message_type, const char* role) {
    return osdds_open_with_history(domain,interface,topic,message_type,role,1);
}
API int osdds_write(void* handle, const char* payload, size_t size) {
    try {
        if (!handle || !payload || size > 1048576) throw std::invalid_argument("invalid DDS bridge payload");
        static_cast<EndpointBase*>(handle)->write(Json::parse(payload, payload + size));
        return 1;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
API int osdds_take(void* handle, char* output, size_t capacity, unsigned timeout_ms) {
    try {
        if (!handle || !output || timeout_ms > 10000) throw std::invalid_argument("invalid DDS take arguments");
        const auto data = static_cast<EndpointBase*>(handle)->take(timeout_ms);
        if (data.size() > capacity) throw std::length_error("DDS bridge receive buffer too small");
        std::memcpy(output, data.data(), data.size());
        return static_cast<int>(data.size());
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
API int osdds_matches(void* handle) {
    try {
        if (!handle) throw std::invalid_argument("closed DDS endpoint");
        return static_cast<EndpointBase*>(handle)->matches();
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
API void osdds_close(void* handle) { delete static_cast<EndpointBase*>(handle); }
