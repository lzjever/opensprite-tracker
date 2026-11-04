// SPDX-License-Identifier: AGPL-3.0-only
#include "backend.hpp"
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

namespace opensprite::runtime {
namespace {
using Json=nlohmann::json;
class MujocoBackend final: public Backend {
    mjModel* model_=nullptr; mjData* data_=nullptr;
    Json profile_; std::array<int,31> qids_{},vids_{};
    std::array<double,31> effort_{},tau_{},raw_{};
    std::uint64_t tick_=0,seq_=0,period_=0,dispatches_=0;
    std::uint64_t last_polled_tick_=UINT64_MAX;
    bool last_polled_enabled_=false;
    bool enabled_=false,has_command_=false;
    wire::LowCmd command_;
    wire::LowCmd actuator_command_;
    std::deque<wire::LowCmd> delay_;
    int lag_=0,phase_=0,min_delay_=0,max_delay_=0,refresh_=4;
    double hold_=.9;
    std::mt19937_64 rng_;
    int imu_body_=0,gyro_=0,accel_=0;
    static bool finite(const std::vector<double>& v) {
        return std::all_of(v.begin(),v.end(),[](double x){return std::isfinite(x);});
    }
public:
    explicit MujocoBackend(const std::string& filename) {
        try {
            std::ifstream stream(filename);if(!stream)throw std::runtime_error("cannot open sim scene");
            const auto scene=Json::parse(stream);
            if(scene.at("schema")!="opensprite-native-sim-v1")throw std::runtime_error("sim scene schema");
            if(scene.value("joint_execution_hz",500)!=500 || scene.value("physics_hz",200)!=200)
                throw std::runtime_error("scene requires500Hz command dispatch and200Hz physics");
            profile_=scene.at("profile");
            if(profile_.at("schema")!="opensprite_sdk_sim_profile_v1" ||
               profile_.at("torque_lag_seconds").get<double>()!=0.)throw std::runtime_error("sim execution contract");
            const double dt=profile_.at("physics_dt");
            if(!std::isfinite(dt) || std::abs(dt-.005)>1e-12)
                throw std::runtime_error("simulation physics must remain200Hz; joint command dispatch is500Hz");
            if(std::abs(profile_.at("policy_dt").get<double>()-.02)>1e-12)
                throw std::runtime_error("tracker policy period must be20ms");
            period_=std::llround(dt*1e9);
            const auto path=std::filesystem::path(filename).parent_path()/scene.at("model").get<std::string>();
            char error[1024]{};model_=mj_loadXML(path.c_str(),nullptr,error,sizeof(error));
            if(!model_)throw std::runtime_error(std::string("MuJoCo model: ")+error);
            if(model_->nq!=38 || model_->nv!=37 || model_->nu!=0)throw std::runtime_error("OpenSprite31joint free-base model required");
            model_->opt.timestep=dt;
            model_->opt.iterations=profile_.at("solver_iterations");
            model_->opt.ls_iterations=profile_.at("solver_ls_iterations");
            data_=mj_makeData(model_);if(!data_)throw std::runtime_error("MuJoCo data allocation");
            const auto names=profile_.at("joint_names").get<std::vector<std::string>>();
            if(names.size()!=31)throw std::runtime_error("joint order length");
            std::uint64_t hash=0xcbf29ce484222325ULL;
            for(std::size_t i=0;i<31;++i) {
                for(unsigned char c:names[i]){hash^=c;hash*=0x100000001b3ULL;}hash^=0;hash*=0x100000001b3ULL;
                int id=mj_name2id(model_,mjOBJ_JOINT,names[i].c_str());
                if(id<0)throw std::runtime_error("sim joint absent");
                qids_[i]=model_->jnt_qposadr[id];vids_[i]=model_->jnt_dofadr[id];
                effort_[i]=profile_.at("effort_limit").at(i).get<double>();
                if(!std::isfinite(effort_[i])||effort_[i]<=0 ||
                   std::abs(model_->jnt_range[2*id]-profile_.at("joint_lower").at(i).get<double>())>1e-9 ||
                   std::abs(model_->jnt_range[2*id+1]-profile_.at("joint_upper").at(i).get<double>())>1e-9)
                    throw std::runtime_error("sim joint limit/effort contract");
                data_->qpos[qids_[i]]=profile_.at("default_q").at(i).get<double>();
            }
            if(hash!=17325335851961314422ULL)throw std::runtime_error("OpenSprite joint order hash");
            auto pose=scene.at("initial_qpos").get<std::vector<double>>();
            auto velocity=scene.at("initial_qvel").get<std::vector<double>>();
            if(pose.size()!=38||velocity.size()!=37||!finite(pose)||!finite(velocity))
                throw std::runtime_error("sim initial state dimensions/finite");
            double norm=0;for(int i=3;i<7;++i)norm+=pose[i]*pose[i];
            if(std::abs(norm-1)>1e-5)throw std::runtime_error("sim initial quaternion");
            std::copy(pose.begin(),pose.end(),data_->qpos);std::copy(velocity.begin(),velocity.end(),data_->qvel);
            imu_body_=mj_name2id(model_,mjOBJ_BODY,profile_.at("imu_body_name").get<std::string>().c_str());
            const int gyro=mj_name2id(model_,mjOBJ_SENSOR,"sdk_gyro"),accel=mj_name2id(model_,mjOBJ_SENSOR,"sdk_accelerometer");
            if(imu_body_<0||gyro<0||accel<0||mj_name2id(model_,mjOBJ_GEOM,"sdk_ground")<0)
                throw std::runtime_error("standalone scene must include floor and IMU sensors");
            gyro_=model_->sensor_adr[gyro];accel_=model_->sensor_adr[accel];
            auto delay=profile_.at("command_delay");min_delay_=delay.at("min_ticks");max_delay_=delay.at("max_ticks");
            refresh_=delay.at("refresh_ticks");hold_=delay.at("hold_probability");
            if(min_delay_<0||max_delay_<min_delay_||max_delay_>100||refresh_<1||!std::isfinite(hold_)||hold_<0||hold_>1)
                throw std::runtime_error("sim delay bounds");
            rng_.seed(scene.value("seed",42ULL));lag_=min_delay_;
            if(delay.at("random_phase").get<bool>())phase_=std::uniform_int_distribution<int>(0,refresh_-1)(rng_);
            mj_forward(model_,data_);
        } catch(...) {if(data_)mj_deleteData(data_);if(model_)mj_deleteModel(model_);throw;}
    }
    ~MujocoBackend() override {if(data_)mj_deleteData(data_);if(model_)mj_deleteModel(model_);}
    bool Poll(wire::LowState& state) override {
        if(last_polled_tick_==tick_ && last_polled_enabled_==enabled_)return false;
        last_polled_tick_=tick_;last_polled_enabled_=enabled_;
        state=wire::LowState();state.profile_id(Profile());state.joint_order_hash(17325335851961314422ULL);
        state.seq(++seq_);state.stamp_ns(tick_*period_);state.clock_domain("sim");
        state.status(enabled_?"enabled":"disabled");
        for(int i=0;i<31;++i) {
            auto& j=state.joint_state()[i];j.q(data_->qpos[qids_[i]]);j.dq(data_->qvel[vids_[i]]);
            j.tau_est(tau_[i]);j.valid(true);j.tau_est_valid(true);j.sample_ns(state.stamp_ns());j.sample_time_kind("simulation");
        }
        auto& imu=state.imu_state();imu.valid(true);imu.sample_ns(state.stamp_ns());
        imu.sample_time_kind("simulation");imu.frame("body");imu.attached_link(profile_.at("imu_body_name").get<std::string>());
        for(int i=0;i<3;++i){imu.gyroscope()[i]=data_->sensordata[gyro_+i];
            imu.accelerometer()[i]=data_->sensordata[accel_+i];imu.projected_gravity()[i]=-data_->xmat[imu_body_*9+6+i];}
        return true;
    }
    void Enable() override {enabled_=true;}
    void Submit(const wire::LowCmd& command) override {
        if(!enabled_)throw std::runtime_error("sim not enabled");
        for(const auto& j:command.joint_cmd())if(j.kp()>profile_.at("maximum_kp").get<double>()||
            j.kd()>profile_.at("maximum_kd").get<double>())throw std::runtime_error("sim gain envelope");
        command_=command;has_command_=true;
    }
    void Tick() override {
        if(!enabled_||!has_command_)return;
        actuator_command_=command_;++dispatches_;
    }
    void PhysicsStep() override {
        if(!enabled_||!has_command_||dispatches_==0)return;
        delay_.push_back(actuator_command_);while(int(delay_.size())>max_delay_+1)delay_.pop_front();
        if((tick_+phase_)%refresh_==0) {
            int candidate=std::uniform_int_distribution<int>(min_delay_,max_delay_)(rng_);
            if(std::generate_canonical<double,53>(rng_)>=hold_)lag_=candidate;
        }
        const auto& cmd=delay_[delay_.size()-1-std::min<std::size_t>(lag_,delay_.size()-1)];
        std::fill(data_->qfrc_applied,data_->qfrc_applied+model_->nv,0.);
        for(int i=0;i<31;++i){const auto& j=cmd.joint_cmd()[i];
            raw_[i]=j.kp()*(j.q()-data_->qpos[qids_[i]])+j.kd()*(j.dq()-data_->qvel[vids_[i]])+j.tau();
            tau_[i]=std::clamp(raw_[i],-effort_[i],effort_[i]);data_->qfrc_applied[vids_[i]]=tau_[i];}
        mj_step(model_,data_);++tick_;
        for(int i=0;i<model_->nq;++i)if(!std::isfinite(data_->qpos[i]))throw std::runtime_error("nonfinite sim pose");
        for(int i=0;i<model_->nv;++i)if(!std::isfinite(data_->qvel[i]))throw std::runtime_error("nonfinite sim velocity");
        mj_forward(model_,data_);
    }
    void Disable() noexcept override {enabled_=false;has_command_=false;delay_.clear();tau_.fill(0);if(data_)std::fill(data_->qfrc_applied,data_->qfrc_applied+model_->nv,0.);}
    bool IsSimulation()const override{return true;} bool IsShadow()const override{return false;}
    std::uint64_t PeriodNs()const override{return 2'000'000;}
    std::uint64_t PhysicsPeriodNs()const override{return period_;}
    std::string Profile()const override{return profile_.at("profile_id");}
    std::string ClockDomain()const override{return "sim";}
    Json Diagnostics()const override {
        return {{"simulated_seconds",tick_*period_/1e9},{"physics_steps",tick_},
            {"physics_hz",1e9/period_},{"joint_command_dispatches",dispatches_},{"command_lag_ticks",lag_},
            {"qpos",std::vector<double>(data_->qpos,data_->qpos+38)},
            {"qvel",std::vector<double>(data_->qvel,data_->qvel+37)},
            {"applied_torque",tau_},{"requested_torque",raw_}};
    }
};
} // namespace
std::unique_ptr<Backend> MakeMujocoBackend(const std::string& config){return std::make_unique<MujocoBackend>(config);}
} // namespace opensprite::runtime
