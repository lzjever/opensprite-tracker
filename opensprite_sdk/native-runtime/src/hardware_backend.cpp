// OpenSprite native device gateway. SPDX-License-Identifier: AGPL-3.0-only
#include "backend.hpp"
#include "policy_imu_client.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace opensprite::runtime {
namespace {
using Json = nlohmann::json;
using Vector = std::array<double, kJointCount>;
using Matrix = std::array<Vector, kJointCount>;
#pragma pack(push, 1)
struct StatePacket {
    char magic[4]; std::uint16_t version, kind;
    std::uint64_t sequence; std::int64_t stamp; std::uint64_t order;
    double position[kJointCount], velocity[kJointCount];
    std::int64_t sample[kJointCount];
};
struct TargetPacket {
    char magic[4]; std::uint16_t version, kind;
    std::uint64_t sequence; std::int64_t stamp; std::uint64_t order, source;
    double q[kJointCount], dq[kJointCount], kp[kJointCount], kd[kJointCount], tau[kJointCount];
};
#pragma pack(pop)
static_assert(sizeof(StatePacket)==776 && sizeof(TargetPacket)==1280, "native IPC ABI drift");

void Require(bool value, const std::string& why) {
    if (!value) throw std::runtime_error(why);
}
void Fresh(std::uint64_t stamp, std::uint64_t now, std::uint64_t age, const char* label) {
    Require(stamp && stamp<=now && now-stamp<=age, std::string(label)+": stale/future timestamp");
}
std::uint64_t NameHash(const std::vector<std::string>& names) {
    std::uint64_t h=0xcbf29ce484222325ULL;
    for (const auto& name:names) {
        Require(!name.empty() && name.size()<=128 && name.find('\0')==std::string::npos, "invalid device name");
        for (unsigned char c:name) {h^=c;h*=0x100000001b3ULL;}
        h*=0x100000001b3ULL;
    }
    Require(names.size()==kJointCount && std::set<std::string>(names.begin(),names.end()).size()==kJointCount,
            "expected31 unique device names");
    return h;
}
Vector ReadVector(const Json& input) {
    Require(input.is_array() && input.size()==kJointCount,"expected finite31-vector");
    Vector result{};
    for (std::size_t i=0;i<kJointCount;++i) {
        Require(input[i].is_number() && !input[i].is_boolean(),"invalid vector component");
        result[i]=input[i].get<double>();
        Require(std::isfinite(result[i]),"nonfinite vector component");
    }
    return result;
}
Matrix ReadMatrix(const Json& input) {
    Require(input.is_array() && input.size()==kJointCount,"expected31x31 matrix");
    Matrix result{};
    for (std::size_t i=0;i<kJointCount;++i) result[i]=ReadVector(input[i]);
    return result;
}
std::string Bounded(const Json& input, const char* label, std::size_t maximum=128) {
    Require(input.is_string(),std::string(label)+" requires string");
    auto value=input.get<std::string>();
    Require(!value.empty() && value.size()<=maximum && value.find('\0')==std::string::npos,
            std::string("invalid ")+label);
    return value;
}
struct JointLimit { double lo, hi, velocity, kp, kd, effort; };

class HardwareBackend final : public Backend {
    std::string profile_, imu_link_;
    Matrix inverse_{};
    Vector offset_{};
    std::array<std::array<double,3>,3> rotation_{};
    std::array<JointLimit,kJointCount> limits_{};
    std::uint64_t motor_hash_=0, native_max_age_=60'000'000, imu_max_age_=30'000'000;
    bool actuate_=false, synthetic_=false, enabled_=false;
    std::atomic<int> fd_{-1};
    std::unique_ptr<sprite_policy::NativeImuClient> imu_;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    mutable std::mutex mutex_;
    std::optional<wire::LowState> latest_;
    StatePacket latest_packet_{};
    std::string error_;
    std::uint64_t last_polled_=0, target_sequence_=0, started_ns_=0;

    void CurrentFreshLocked() const {
        Require(latest_.has_value(),"native device feedback is not ready");
        const auto now=MonotonicNs();
        Fresh(static_cast<std::uint64_t>(latest_packet_.stamp),now,native_max_age_,"native aggregate");
        Fresh(latest_->stamp_ns(),now,native_max_age_,"fused observation");
        for (auto stamp:latest_packet_.sample)
            Fresh(static_cast<std::uint64_t>(stamp),now,10'000'000,"500Hz motor feedback");
        Fresh(latest_->imu_state().sample_ns(),now,imu_max_age_,"IMU sample");
    }
    void CheckErrorLocked() const { Require(error_.empty(),"native feedback worker: "+error_); }

    void Run() noexcept {
        try {
            std::uint64_t previous=0;
            while (!stop_) {
                pollfd descriptor{fd_.load(),POLLIN,0};
                const auto ready=::poll(&descriptor,1,10);
                if (ready<0 && errno==EINTR) continue;
                Require(ready>=0,"native feedback poll failed");
                if (!ready) continue;
                if(stop_||fd_.load()!=descriptor.fd)break;
                Require((descriptor.revents&POLLIN)!=0,"native IPC disconnected");
                StatePacket packet{};
                const auto bytes=::recv(descriptor.fd,&packet,sizeof(packet),MSG_DONTWAIT|MSG_TRUNC);
                Require(bytes==static_cast<ssize_t>(sizeof(packet)),"native state ABI2 packet size/disconnect");
                Require(std::memcmp(packet.magic,"SPRT",4)==0 && packet.version==2 &&
                        (packet.kind==1 || packet.kind==3) && packet.order==motor_hash_,
                        "native state ABI/order mismatch");
                Require(packet.sequence>previous,"native state sequence regression/replay");
                previous=packet.sequence;
                const auto now=MonotonicNs();
                Fresh(static_cast<std::uint64_t>(packet.stamp),now,native_max_age_,"native aggregate");
                for (std::size_t i=0;i<kJointCount;++i) {
                    Require(std::isfinite(packet.position[i]) && std::isfinite(packet.velocity[i]),"nonfinite native feedback");
                    Fresh(static_cast<std::uint64_t>(packet.sample[i]),now,10'000'000,"500Hz motor feedback");
                    Require(packet.sample[i]<=packet.stamp,"motor sample follows native aggregate");
                }
                Require(!actuate_ || packet.kind==1,"actuating gateway connected to shadow-only native");
                Require(actuate_ || packet.kind==3,"shadow gateway requires shadow-only native");
                sprite_policy::ImuSnapshot imu;
                if (!imu_->snapshot(imu)) continue;
                Fresh(imu.start,MonotonicNs(),imu_max_age_,"IMU sample");
                wire::LowState state;
                state.profile_id(profile_);state.joint_order_hash(kJointOrderHash);
                state.seq(packet.sequence);state.clock_domain("monotonic");
                state.status(actuate_?"ready":"shadow");
                auto& joints=state.joint_state();
                for (std::size_t j=0;j<kJointCount;++j) {
                    double q=0,dq=0;
                    std::uint64_t oldest=UINT64_MAX;
                    for (std::size_t m=0;m<kJointCount;++m) {
                        const auto coefficient=inverse_[j][m];
                        q+=coefficient*(packet.position[m]-offset_[m]);dq+=coefficient*packet.velocity[m];
                        if (std::abs(coefficient)>1e-12)
                            oldest=std::min(oldest,static_cast<std::uint64_t>(packet.sample[m]));
                    }
                    Require(std::isfinite(q) && std::isfinite(dq) && oldest!=UINT64_MAX,"invalid mapped joint feedback");
                    joints[j].q(q);joints[j].dq(dq);joints[j].tau_est(0.);
                    joints[j].valid(true);joints[j].tau_est_valid(false);
                    joints[j].sample_ns(oldest);joints[j].sample_time_kind("host_receive");
                }
                auto& sensor=state.imu_state();
                for (std::size_t i=0;i<3;++i) {
                    double accel=0,gyro=0,gravity=0;
                    for (std::size_t j=0;j<3;++j) {
                        accel+=rotation_[i][j]*imu.values[j];
                        gyro+=rotation_[i][j]*imu.values[3+j];
                        gravity-=rotation_[i][j]*imu.values[10+j]/9.80665;
                    }
                    sensor.accelerometer()[i]=accel;sensor.gyroscope()[i]=gyro;
                    sensor.projected_gravity()[i]=gravity;
                }
                sensor.valid(true);sensor.frame("body");sensor.attached_link(imu_link_);
                sensor.sample_ns(imu.start);sensor.sample_time_kind("host_read");
                // The header anchors the completed fusion; it never rewrites
                // either device's actual sampling/publication timestamps.
                state.stamp_ns(MonotonicNs());
                std::lock_guard<std::mutex> lock(mutex_);
                latest_packet_=packet;latest_=std::move(state);
                CurrentFreshLocked();
            }
        } catch (const std::exception& error) {
            if (!stop_) {std::lock_guard<std::mutex> lock(mutex_);error_=error.what();}
        }
    }

    void SendLocked(const wire::LowCmd& command) {
        CurrentFreshLocked();
        TargetPacket packet{};
        std::memcpy(packet.magic,"SPRT",4);packet.version=1;packet.kind=2;
        packet.sequence=++target_sequence_;packet.stamp=static_cast<std::int64_t>(MonotonicNs());
        packet.order=kJointOrderHash;packet.source=command.source_state_seq();
        for (std::size_t i=0;i<kJointCount;++i) {
            const auto& target=command.joint_cmd()[i];
            packet.q[i]=target.q();packet.dq[i]=target.dq();packet.kp[i]=target.kp();
            packet.kd[i]=target.kd();packet.tau[i]=target.tau();
        }
        Require(::send(fd_.load(),&packet,sizeof(packet),MSG_DONTWAIT|MSG_NOSIGNAL)==sizeof(packet),
                "native target IPC send failed");
    }

public:
    HardwareBackend(const std::string& path,bool actuate):actuate_(actuate) {
        Require(__BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__,"native IPC requires little-endian host");
        std::ifstream input(path);Require(bool(input),"cannot read native hardware config");
        const auto config=Json::parse(input);
        Require(config.at("schema")=="opensprite-native-hardware-v1" &&
                config.at("configured")==true && config.at("estop_configured")==true,
                "native hardware requires measured configuration and estop");
        synthetic_=config.at("synthetic").get<bool>();
        Require(!actuate_ || !synthetic_,"synthetic hardware must never actuate");
        Bounded(config.at("hardware_identity"),"hardware identity");
        profile_=Bounded(config.at("profile_id"),"profile");
        Require(profile_=="physical_v5","hardware profile must be physical_v5");
        const auto joint_names=config.at("joint_names").get<std::vector<std::string>>();
        Require(NameHash(joint_names)==kJointOrderHash && config.at("joint_order_hash").get<std::uint64_t>()==kJointOrderHash,
                "OpenSprite joint order mismatch");
        const auto motor_names=config.at("motor_names").get<std::vector<std::string>>();
        motor_hash_=NameHash(motor_names);
        Require(config.at("motor_order_hash").get<std::uint64_t>()==motor_hash_,"motor order hash mismatch");
        const auto forward=ReadMatrix(config.at("joint_to_motor"));
        inverse_=ReadMatrix(config.at("motor_to_joint"));offset_=ReadVector(config.at("motor_offset"));
        for (std::size_t i=0;i<kJointCount;++i) for(std::size_t j=0;j<kJointCount;++j) {
            double product=0;
            for(std::size_t k=0;k<kJointCount;++k)product+=forward[i][k]*inverse_[k][j];
            Require(std::isfinite(product) && std::abs(product-(i==j?1.:0.))<1e-8,
                    "calibration matrices are not mutual inverses");
        }
        const auto limits=config.at("joint_limits");
        Require(limits.is_array() && limits.size()==kJointCount,"31 joint envelopes required");
        for(std::size_t i=0;i<kJointCount;++i) {
            const auto& limit=limits[i];const auto range=limit.at("position_rad");
            Require(range.is_array()&&range.size()==2,"joint position range required");
            auto& value=limits_[i];
            value={range[0].get<double>(),range[1].get<double>(),limit.at("velocity_rad_s").get<double>(),
                limit.at("kp").get<double>(),limit.at("kd").get<double>(),limit.at("effort_nm").get<double>()};
            Require(std::isfinite(value.lo)&&std::isfinite(value.hi)&&value.lo<value.hi,"invalid joint range");
            for(double v:{value.velocity,value.kp,value.kd,value.effort})Require(std::isfinite(v)&&v>0,"invalid joint envelope");
            Require(value.kd<=5.,"native Kd envelope exceeds5");
        }
        imu_link_=Bounded(config.at("imu_link"),"IMU link");
        const auto rotation=config.at("sensor_to_body_rotation");
        Require(rotation.is_array()&&rotation.size()==3,"IMU rotation3x3 required");
        for(std::size_t i=0;i<3;++i) {
            Require(rotation[i].is_array()&&rotation[i].size()==3,"IMU rotation3x3 required");
            for(std::size_t j=0;j<3;++j) {rotation_[i][j]=rotation[i][j].get<double>();Require(std::isfinite(rotation_[i][j]),"nonfinite IMU rotation");}
        }
        for(std::size_t i=0;i<3;++i)for(std::size_t j=0;j<3;++j) {
            double dot=0;for(std::size_t k=0;k<3;++k)dot+=rotation_[k][i]*rotation_[k][j];
            Require(std::abs(dot-(i==j?1.:0.))<1e-9,"IMU rotation not orthogonal");
        }
        const double determinant=rotation_[0][0]*(rotation_[1][1]*rotation_[2][2]-rotation_[1][2]*rotation_[2][1])
            -rotation_[0][1]*(rotation_[1][0]*rotation_[2][2]-rotation_[1][2]*rotation_[2][0])
            +rotation_[0][2]*(rotation_[1][0]*rotation_[2][1]-rotation_[1][1]*rotation_[2][0]);
        Require(std::abs(determinant-1)<1e-9,"IMU rotation must be proper");
        imu_max_age_=config.at("imu_max_age_ns").get<std::uint64_t>();
        native_max_age_=config.at("native_max_age_ns").get<std::uint64_t>();
        Require(imu_max_age_>0&&imu_max_age_<=30'000'000&&native_max_age_>0&&native_max_age_<=60'000'000,
                "native/IMU freshness may not be weakened");
        Require(config.at("motor_max_age_ns")==10'000'000&&config.at("source_max_age_ns")==60'000'000&&
                config.at("command_timeout_ns")==100'000'000&&config.at("joint_execution_hz")==500&&config.at("state_hz")==500,
                "500Hz execution/source/watchdog contract mismatch");
        const auto native_path=Bounded(config.at("native_socket"),"native socket",107);
        const auto imu_path=Bounded(config.at("imu_socket"),"IMU socket",107);
        try {
            started_ns_=MonotonicNs();
            const auto descriptor=::socket(AF_UNIX,SOCK_SEQPACKET|SOCK_CLOEXEC|SOCK_NONBLOCK,0);
            Require(descriptor>=0,"cannot create native IPC socket");fd_=descriptor;
            sockaddr_un address{};address.sun_family=AF_UNIX;
            std::memcpy(address.sun_path,native_path.c_str(),native_path.size()+1);
            Require(::connect(descriptor,reinterpret_cast<sockaddr*>(&address),sizeof(address))==0,"native IPC connect failed");
            imu_=std::make_unique<sprite_policy::NativeImuClient>(imu_path,true,true);
            worker_=std::thread([this]{Run();});
        } catch (...) {Disable();throw;}
    }
    ~HardwareBackend() override {Disable();}
    bool Poll(wire::LowState& state) override {
        std::lock_guard<std::mutex> lock(mutex_);CheckErrorLocked();
        if(!latest_||latest_->seq()==last_polled_)return false;
        CurrentFreshLocked();state=*latest_;last_polled_=state.seq();return true;
    }
    void Enable() override {
        std::lock_guard<std::mutex> lock(mutex_);CheckErrorLocked();CurrentFreshLocked();
        if(enabled_)return;
        if(actuate_) {
            wire::LowCmd startup;startup.source_state_seq(latest_->seq());
            for(std::size_t i=0;i<kJointCount;++i) {startup.joint_cmd()[i].mode(1);startup.joint_cmd()[i].q(latest_->joint_state()[i].q());}
            SendLocked(startup);
        }
        enabled_=true;
    }
    void Submit(const wire::LowCmd& command) override {
        std::lock_guard<std::mutex> lock(mutex_);CheckErrorLocked();CurrentFreshLocked();
        Require(enabled_,"native gateway must enable explicitly");
        for(std::size_t i=0;i<kJointCount;++i) {
            const auto& t=command.joint_cmd()[i];const auto& l=limits_[i];
            Require(t.mode()==1,"native Submit expects whole-body command mode1");
            for(double v:{t.q(),t.dq(),t.kp(),t.kd(),t.tau()})Require(std::isfinite(v),"nonfinite native target");
            Require(t.q()>=l.lo&&t.q()<=l.hi&&std::abs(t.dq())<=l.velocity&&t.kp()>=0&&t.kp()<=l.kp&&
                    t.kd()>=0&&t.kd()<=l.kd&&std::abs(t.tau())<=l.effort,"target exceeds measured joint envelope");
        }
        SendLocked(command);
    }
    void Tick() override {
        std::lock_guard<std::mutex> lock(mutex_);CheckErrorLocked();
        if(latest_)CurrentFreshLocked();
        else Require(MonotonicNs()-started_ns_<1'000'000'000,"native device startup feedback timeout");
    }
    void Disable() noexcept override {
        stop_=true;enabled_=false;
        const auto descriptor=fd_.exchange(-1);
        if(descriptor>=0)::shutdown(descriptor,SHUT_RDWR);
        if(worker_.joinable()&&worker_.get_id()!=std::this_thread::get_id())worker_.join();
        if(descriptor>=0)::close(descriptor);
        imu_.reset();
    }
    bool IsSimulation() const override {return false;}
    bool IsShadow() const override {return !actuate_;}
    std::uint64_t PeriodNs() const override {return 2'000'000;}
    std::string Profile() const override {return profile_;}
    std::string ClockDomain() const override {return "monotonic";}
    Json Diagnostics() const override {return {{"joint_execution_hz",500},{"state_hz",500},{"native_motor_disable_verified",false},{"synthetic",synthetic_}};}
};
} // namespace
std::unique_ptr<Backend> MakeHardwareBackend(const std::string& config,bool actuate) {
    return std::make_unique<HardwareBackend>(config,actuate);
}
} // namespace opensprite::runtime
