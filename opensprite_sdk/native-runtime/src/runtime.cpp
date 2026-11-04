// Shared simulation/hardware robot endpoint. SPDX-License-Identifier: AGPL-3.0-only
#include "native_runtime.hpp"
#include <opensprite_sdk/core/typed_endpoint.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <iomanip>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace opensprite::runtime {
std::uint64_t MonotonicNs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
namespace {
using Json=nlohmann::json;
template<class T>using Endpoint=opensprite_sdk::TypedEndpoint<T>;
std::string Session() {
    std::random_device random;
    std::ostringstream value;
    for(int i=0;i<4;++i)value<<std::hex<<std::setw(8)<<std::setfill('0')<<random();
    return value.str();
}
void Require(bool condition,const std::string& reason) {
    if(!condition)throw std::runtime_error(reason);
}
void Fresh(std::uint64_t stamp,std::uint64_t now,std::uint64_t maximum,const char* what,bool zero=false) {
    Require((stamp||zero)&&stamp<=now&&now-stamp<=maximum,std::string(what)+": stale/future timestamp");
}

class Runtime {
    std::unique_ptr<Backend> backend_;
    RuntimeOptions options_;
    std::shared_ptr<Endpoint<wire::LowCmd>> commands_;
    std::shared_ptr<Endpoint<wire::LowState>> states_;
    std::shared_ptr<Endpoint<wire::ControlRequest>> requests_;
    std::shared_ptr<Endpoint<wire::ControlReply>> replies_;
    std::string session_=Session(),owner_,fault_;
    std::optional<wire::LowState> state_;
    std::map<std::uint64_t,wire::LowState> sources_;
    bool enabled_=false,moving_=false,stopping_=false;
    std::uint64_t accepted_=0,rejected_=0,rejected_management_=0,last_sequence_=0,reply_sequence_=0;
    std::uint64_t enabled_at_=0,last_command_wall_=0,execution_ticks_=0,deadline_misses_=0;
    struct CachedRequest {std::string identity;wire::ControlReply reply;};
    std::map<std::string,CachedRequest> cache_;
    std::deque<std::string> cache_order_;
    std::map<std::string,std::uint64_t> management_highwater_;
    std::thread reply_worker_;
    std::mutex reply_mutex_;
    std::condition_variable reply_condition_;
    std::deque<wire::ControlReply> reply_queue_;
    std::atomic<bool> reply_stop_{false};
    std::atomic<std::uint64_t> reply_errors_{0},reply_drops_{0};
    std::uint64_t last_idle_publication_=0,last_truth_stamp_=UINT64_MAX;
    std::ofstream truth_;
    std::thread truth_worker_;
    std::mutex truth_mutex_;
    std::condition_variable truth_condition_;
    std::deque<Json> truth_queue_;
    std::atomic<bool> truth_stop_{false};
    std::atomic<std::uint64_t> truth_drops_{0},truth_errors_{0};

    std::uint64_t RobotNow() const {
        return backend_->IsSimulation()?(state_?state_->stamp_ns():0):MonotonicNs();
    }
    void SendReplies() noexcept {
        while(!reply_stop_) {
            wire::ControlReply reply;
            {
                std::unique_lock<std::mutex> lock(reply_mutex_);
                reply_condition_.wait_for(lock,std::chrono::milliseconds(20),[this]{return reply_stop_||!reply_queue_.empty();});
                if(reply_stop_)return;
                if(reply_queue_.empty())continue;
                reply=std::move(reply_queue_.front());reply_queue_.pop_front();
            }
            try {replies_->Write(reply);}catch(const std::exception&){++reply_errors_;}
        }
    }
    void SendTruth() noexcept {
        while(true) {
            Json row;
            {
                std::unique_lock<std::mutex> lock(truth_mutex_);
                truth_condition_.wait_for(lock,std::chrono::milliseconds(20),[this]{return truth_stop_||!truth_queue_.empty();});
                if(truth_queue_.empty()) {if(truth_stop_)break;continue;}
                row=std::move(truth_queue_.front());truth_queue_.pop_front();
            }
            try {truth_<<row.dump()<<'\n';if(!truth_)++truth_errors_;}
            catch(const std::exception&){++truth_errors_;}
        }
        truth_.flush();if(!truth_)++truth_errors_;
    }
    void QueueReply(wire::ControlReply reply) {
        reply.seq(++reply_sequence_);reply.session_id(session_);
        std::lock_guard<std::mutex> lock(reply_mutex_);
        if(reply_queue_.size()>=32){reply_queue_.pop_front();++reply_drops_;}
        reply_queue_.push_back(std::move(reply));reply_condition_.notify_one();
    }
    void ValidateState(const wire::LowState& state) {
        Require(state.seq()>0,"backend state sequence must be positive");
        Require(state.status().to_string()!="fault"&&state.faults().empty(),"backend reported fault");
        Require(state.profile_id().to_string()==backend_->Profile()&&state.clock_domain().to_string()==backend_->ClockDomain()
                &&state.joint_order_hash()==kJointOrderHash,"backend profile/clock/order mismatch");
        if(!backend_->IsSimulation())Fresh(state.stamp_ns(),MonotonicNs(),60'000'000,"backend fusion");
        for(const auto& joint:state.joint_state()) {
            for(double v:{joint.q(),joint.dq(),joint.tau_est()})Require(std::isfinite(v),"nonfinite joint state");
            Require(joint.valid(),"invalid joint measurement");
            Fresh(joint.sample_ns(),state.stamp_ns(),60'000'000,"joint sample",backend_->IsSimulation());
            Require(joint.sample_time_kind().to_string()==(backend_->IsSimulation()?"simulation":"host_receive"),"unknown joint sample clock");
        }
        const auto& imu=state.imu_state();
        Require(imu.valid()&&imu.frame().to_string()=="body"&&imu.attached_link().size()>0,"invalid IMU measurement");
        Fresh(imu.sample_ns(),state.stamp_ns(),30'000'000,"IMU sample",backend_->IsSimulation());
        Require(imu.sample_time_kind().to_string()==(backend_->IsSimulation()?"simulation":"host_read"),"unknown IMU sample clock");
        double norm=0;
        for(std::size_t i=0;i<3;++i) {
            Require(std::isfinite(imu.gyroscope()[i])&&std::isfinite(imu.accelerometer()[i])&&std::isfinite(imu.projected_gravity()[i]),"nonfinite IMU");
            norm+=imu.projected_gravity()[i]*imu.projected_gravity()[i];
        }
        Require(std::abs(std::sqrt(norm)-1.)<=.02,"IMU projected gravity is not unit length");
    }
    bool PollAndPublish() {
        wire::LowState incoming;
        if(!backend_->Poll(incoming))return false;
        ValidateState(incoming);
        if(state_)Require(incoming.seq()>state_->seq()&&incoming.stamp_ns()>=state_->stamp_ns(),"backend sequence/time regression");
        incoming.session_id(session_);incoming.accepted_command_seq(last_sequence_);
        incoming.accepted_controller_session(last_sequence_?owner_:"");
        state_=incoming;sources_[incoming.seq()]=incoming;
        while(sources_.size()>64)sources_.erase(sources_.begin());
        states_->Write(incoming);last_idle_publication_=MonotonicNs();
        if(truth_.is_open()&&moving_&&incoming.stamp_ns()>0&&incoming.stamp_ns()%20'000'000==0&&incoming.stamp_ns()!=last_truth_stamp_) {
            auto row=backend_->Diagnostics();row["time"]=incoming.stamp_ns()/1e9;
            row["accepted_command_seq"]=last_sequence_;
            {std::lock_guard<std::mutex> lock(truth_mutex_);
             if(truth_queue_.size()>=32){truth_queue_.pop_front();++truth_drops_;}
             truth_queue_.push_back(std::move(row));truth_condition_.notify_one();}
            last_truth_stamp_=incoming.stamp_ns();
        }
        return true;
    }
    wire::ControlReply Process(const wire::ControlRequest& request) {
        wire::ControlReply reply;
        reply.profile_id(backend_->Profile());reply.joint_order_hash(kJointOrderHash);
        reply.clock_domain(backend_->ClockDomain());reply.stamp_ns(RobotNow());
        reply.request_id(request.request_id());reply.controller_session(request.controller_session());
        reply.robot_session(session_);reply.accepted(false);reply.execution_verified(false);reply.status("rejected");
        const auto controller=request.controller_session().to_string(),id=request.request_id().to_string();
        const auto operation=request.operation().to_string();
        const auto key=Json::array({controller,id}).dump();
        // Header sequence/time may change on a retry; intent may not.
        const auto identity=Json{{"profile",request.profile_id().to_string()},
            {"order",request.joint_order_hash()},{"clock",request.clock_domain().to_string()},
            {"publisher",request.session_id().to_string()},{"controller",controller},
            {"robot",request.robot_session().to_string()},{"operation",operation}}.dump();
        const auto found=cache_.find(key);
        if(found!=cache_.end()) {
            if(found->second.identity==identity) {
                auto water=management_highwater_.find(controller);
                if(water!=management_highwater_.end())water->second=std::max(water->second,request.seq());
                return found->second.reply;
            }
            reply.detail("request id reused with different intent");return reply;
        }
        try {
            Require(state_.has_value(),"device feedback is not ready");
            Require(!controller.empty()&&!id.empty()&&request.session_id().to_string()==controller,
                    "request publisher must declare controller session");
            Require(request.profile_id().to_string()==backend_->Profile()&&request.joint_order_hash()==kJointOrderHash&&
                    request.clock_domain().to_string()==backend_->ClockDomain()&&request.robot_session().to_string()==session_,
                    "robot session/profile/clock/order mismatch");
            const auto previous=management_highwater_.find(controller);
            Require(request.seq()>(previous==management_highwater_.end()?0:previous->second),"management replay/sequence regression");
            Require(!stopping_&&fault_.empty(),"runtime is stopping or faulted");
            if(operation=="claim") {
                Require(owner_.empty()||owner_==controller,"another controller owns this robot");
                Require(management_highwater_.count(controller)||management_highwater_.size()<256,"controller history is full; restart robot runtime");
                owner_=controller;reply.status("claimed");
            } else {
                Require(owner_==controller,"requesting controller does not own robot");
                if(operation=="enable") {
                    Require(options_.allow_enable,"local startup permission does not allow enable");
                    if(!enabled_) {backend_->Enable();enabled_=true;enabled_at_=MonotonicNs();last_command_wall_=0;}
                    reply.status(backend_->IsShadow()?"shadow_ready":"enable_requested");
                    reply.execution_verified(backend_->IsSimulation());
                } else if(operation=="disable") {
                    backend_->Disable();enabled_=false;stopping_=true;
                    reply.status(backend_->IsSimulation()?"disabled":"disable_requested");
                    reply.execution_verified(backend_->IsSimulation());
                } else if(operation=="release") {
                    Require(!enabled_&&!moving_,"release requires idle never-started controller");
                    owner_.clear();reply.status("released");
                } else throw std::runtime_error("unsupported management operation");
            }
            management_highwater_[controller]=request.seq();reply.accepted(true);
        } catch(const std::exception& error) {reply.detail(std::string(error.what()).substr(0,1024));}
        if(cache_.size()>=256) {cache_.erase(cache_order_.front());cache_order_.pop_front();}
        cache_order_.push_back(key);cache_.emplace(key,CachedRequest{identity,reply});return reply;
    }
    void ValidateCommand(const wire::LowCmd& command) {
        Require(command.profile_id().to_string()==backend_->Profile()&&command.clock_domain().to_string()==backend_->ClockDomain()
                &&command.joint_order_hash()==kJointOrderHash,"command profile/clock/order mismatch");
        Require(command.seq()>last_sequence_,"command sequence regression/replay");
        Require(command.source_state_session_id().to_string()==session_,"command source robot epoch mismatch");
        const auto source=sources_.find(command.source_state_seq());
        Require(source!=sources_.end(),"unknown/expired source state sequence");
        const auto now=RobotNow();
        Fresh(command.stamp_ns(),now,100'000'000,"command",backend_->IsSimulation());
        Fresh(source->second.stamp_ns(),now,60'000'000,"source state",backend_->IsSimulation());
        for(const auto& joint:source->second.joint_state())
            Fresh(joint.sample_ns(),now,60'000'000,"source joint",backend_->IsSimulation());
        Fresh(source->second.imu_state().sample_ns(),now,30'000'000,"source IMU",backend_->IsSimulation());
        const auto mode=command.joint_cmd()[0].mode();
        Require(mode==0||mode==1,"unknown joint mode");
        for(const auto& target:command.joint_cmd()) {
            Require(target.mode()==mode,"mixed whole-body modes are unsupported");
            for(double v:{target.q(),target.dq(),target.kp(),target.kd(),target.tau()})Require(std::isfinite(v),"nonfinite command");
            Require(target.kp()>=0&&target.kd()>=0,"negative impedance gain");
        }
    }
public:
    Runtime(std::unique_ptr<Backend> backend,const RuntimeOptions& options):backend_(std::move(backend)),options_(options) {
        Require(backend_!=nullptr,"backend required");
        Require(std::isfinite(options_.duration)&&options_.duration>0&&options_.duration<=86400,"finite positive duration required");
        if(!options.truth_output.empty()) {
            Require(backend_->IsSimulation(),"truth trace is simulation-only");
            truth_.open(options.truth_output);Require(bool(truth_),"cannot open simulation diagnostic trace");
        }
        commands_=opensprite_sdk::CreateEndpoint<wire::LowCmd>(options.domain,options.interface,"rt/lowcmd","subscriber");
        states_=opensprite_sdk::CreateEndpoint<wire::LowState>(options.domain,options.interface,"rt/lowstate","publisher");
        requests_=opensprite_sdk::CreateEndpoint<wire::ControlRequest>(options.domain,options.interface,"rt/control_request","subscriber");
        replies_=opensprite_sdk::CreateEndpoint<wire::ControlReply>(options.domain,options.interface,"rt/control_reply","publisher");
        try {
            reply_worker_=std::thread([this]{SendReplies();});
            if(truth_.is_open())truth_worker_=std::thread([this]{SendTruth();});
        } catch(...) {Close();throw;}
    }
    ~Runtime() {Close();}
    void Close() noexcept {
        backend_->Disable();reply_stop_=true;reply_condition_.notify_all();
        if(reply_worker_.joinable())reply_worker_.join();
        truth_stop_=true;truth_condition_.notify_all();
        if(truth_worker_.joinable())truth_worker_.join();
    }
    Json Run(const std::atomic<bool>* stop) {
        const auto started=MonotonicNs(),period=backend_->PeriodNs();
        const auto physics_period=backend_->PhysicsPeriodNs();
        Require(period==2'000'000,"joint execution period must be2ms");
        Require(!backend_->IsSimulation()||physics_period==5'000'000,"simulation physics period must be5ms");
        const auto base_period=physics_period?std::gcd(period,physics_period):period;
        const auto execution_stride=period/base_period;
        const auto physics_stride=physics_period?physics_period/base_period:0;
        std::uint64_t iteration=0,motion_origin=0;
        auto next=std::chrono::steady_clock::now();
        try {
            while(MonotonicNs()-started<static_cast<std::uint64_t>(options_.duration*1e9)&&!stopping_&&(!stop||!stop->load())) {
                PollAndPublish();
                const bool execute=iteration%execution_stride==0;
                if(execute) {
                    wire::ControlRequest request;
                    if(requests_->Take(request,0)) {
                        // An uncorrelatable request cannot receive a valid
                        // shared-channel reply. Do not poison healthy readers.
                        if(request.request_id().size()==0||request.controller_session().size()==0||
                                request.robot_session().size()==0||request.session_id().size()==0)++rejected_management_;
                        else QueueReply(Process(request));
                    }
                    if(stopping_)break;
                    wire::LowCmd command;
                    if(commands_->Take(command,0)) {
                        if(owner_.empty()||command.session_id().to_string()!=owner_||!enabled_)++rejected_;
                        else {
                            ValidateCommand(command);
                            if(command.joint_cmd()[0].mode()==0){backend_->Disable();enabled_=false;stopping_=true;}
                            else backend_->Submit(command);
                            if(!moving_)motion_origin=iteration;
                            last_sequence_=command.seq();++accepted_;last_command_wall_=MonotonicNs();moving_=true;
                        }
                    }
                    if(stopping_)break;
                    if(!backend_->IsSimulation()||(moving_&&iteration>motion_origin)) {
                        backend_->Tick();++execution_ticks_;
                    }
                }
                if(enabled_) {
                    const auto last=last_command_wall_?last_command_wall_:enabled_at_;
                    Require(MonotonicNs()-last<100'000'000,"controller_stream_watchdog");
                }
                // Actuator latch/dispatch runs at500Hz. The independent5ms
                // physics clock advances only after the first accepted target.
                // No interpolated500Hz sensor observations are fabricated.
                if(physics_stride&&moving_&&iteration>motion_origin&&
                        (iteration-motion_origin)%physics_stride==0)backend_->PhysicsStep();
                PollAndPublish();
                if(backend_->IsSimulation()&&!moving_&&state_&&MonotonicNs()-last_idle_publication_>=20'000'000) {
                    states_->Write(*state_);last_idle_publication_=MonotonicNs();
                }
                ++iteration;next+=std::chrono::nanoseconds(base_period);
                if(std::chrono::steady_clock::now()>next+std::chrono::nanoseconds(base_period)){
                    ++deadline_misses_;next=std::chrono::steady_clock::now();
                }
                std::this_thread::sleep_until(next);
            }
        } catch(const std::exception& error) {fault_=error.what();}
        if(fault_.empty()&&!state_)fault_="device feedback never became ready";
        backend_->Disable();enabled_=false;
        if(stopping_&&fault_.empty())std::this_thread::sleep_for(std::chrono::milliseconds(50));
        Close();
        if(fault_.empty()&&(truth_drops_||truth_errors_))fault_="simulation diagnostic trace lost data";
        Json result={{"schema","opensprite-native-runtime-v1"},{"backend",backend_->IsSimulation()?"mujoco":"hardware"},
            {"profile_id",backend_->Profile()},{"device_ready",state_.has_value()},{"robot_session",session_},{"accepted_commands",accepted_},
            {"accepted_command_seq",last_sequence_},{"rejected_commands",rejected_},
            {"rejected_management_requests",rejected_management_},
            {"execution_ticks",execution_ticks_},{"execution_hz",1e9/period},
            {"physics_hz",physics_period?Json(1e9/physics_period):Json(nullptr)},
            {"fault",fault_.empty()?Json(nullptr):Json(fault_)},
            {"deadline_misses",deadline_misses_},{"management_reply_errors",reply_errors_.load()},
            {"management_reply_drops",reply_drops_.load()},{"truth_output_drops",truth_drops_.load()},
            {"truth_output_errors",truth_errors_.load()},{"wall_seconds",(MonotonicNs()-started)/1e9},
            {"hardware_tested",false},{"video_recorded",false}};
        const auto diagnostics=backend_->Diagnostics();
        for(auto it=diagnostics.begin();it!=diagnostics.end();++it)if(!result.contains(it.key()))result[it.key()]=it.value();
        return result;
    }
};
} // namespace
Json RunRobot(std::unique_ptr<Backend> backend,const RuntimeOptions& options,const std::atomic<bool>* stop) {
    Runtime runtime(std::move(backend),options);return runtime.Run(stop);
}
} // namespace opensprite::runtime
