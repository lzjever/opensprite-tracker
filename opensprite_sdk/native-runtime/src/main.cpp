// SPDX-License-Identifier: AGPL-3.0-only
#include "native_runtime.hpp"
#include <atomic>
#include <cmath>
#include <csignal>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
std::atomic<bool> stop{false};
void Signal(int) {stop=true;}
}
int main(int argc,char** argv) {
    std::string backend="mujoco",config,report;
    opensprite::runtime::RuntimeOptions options;
    bool actuate=false,acknowledge=false;
    // Preserve an explicitly supplied report destination even when another
    // option fails before the normal parser reaches --report.
    for(int i=1;i+1<argc;++i) {
        const std::string key=argv[i],next=argv[i+1];
        if((key=="--report"||key=="--output")&&next.rfind("--",0)!=0)report=next;
    }
    auto write_report=[&](const nlohmann::json& body) {
        if(report.empty())return;
        const auto parent=std::filesystem::path(report).parent_path();
        if(!parent.empty())std::filesystem::create_directories(parent);
        std::ofstream output(report);if(!output)throw std::runtime_error("cannot write report");
        output<<body.dump(2)<<'\n';
    };
    try {
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            auto value=[&](){if(i+1>=argc||std::string(argv[i+1]).rfind("--",0)==0)throw std::runtime_error("missing value for "+arg);return std::string(argv[++i]);};
            if(arg=="--backend")backend=value();
            else if(arg=="--config")config=value();
            else if(arg=="--domain") {
                const auto text=value();std::size_t consumed=0;
                if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("invalid DDS domain");
                const auto domain=std::stoull(text,&consumed);
                if(consumed!=text.size()||domain>232)throw std::runtime_error("DDS domain must be0..232");
                options.domain=static_cast<unsigned>(domain);
            }
            else if(arg=="--interface")options.interface=value();
            else if(arg=="--duration") {
                const auto text=value();std::size_t consumed=0;options.duration=std::stod(text,&consumed);
                if(consumed!=text.size())throw std::runtime_error("invalid duration");
            }
            else if(arg=="--report"||arg=="--output")report=value();
            else if(arg=="--truth-output")options.truth_output=value();
            else if(arg=="--allow-enable"||arg=="--enable-sim")options.allow_enable=true;
            else if(arg=="--actuate")actuate=true;
            else if(arg=="--acknowledge-actuation")acknowledge=true;
            else if(arg=="--help") {
                std::cout<<"opensprite_robot --backend mujoco|hardware --config FILE --domain N --interface lo "
                    "--allow-enable --duration SECONDS --report FILE [--actuate --acknowledge-actuation]\n";
                return 0;
            } else throw std::runtime_error("unknown argument: "+arg);
        }
        if(config.empty())throw std::runtime_error("--config is required");
        if(backend!="mujoco"&&backend!="hardware")throw std::runtime_error("unknown backend");
        if(backend=="hardware"&&!options.truth_output.empty())throw std::runtime_error("truth trace is simulation-only");
        if(options.domain>232)throw std::runtime_error("DDS domain must be0..232");
        if(!std::isfinite(options.duration)||options.duration<=0||options.duration>86400)throw std::runtime_error("duration must be finite and positive");
        if(backend=="hardware"&&options.duration>10)throw std::runtime_error("hardware commissioning is bounded to10 seconds");
        if(backend=="mujoco"&&(actuate||acknowledge))throw std::runtime_error("hardware actuation flags cannot be used in simulation");
        if(actuate&&(!acknowledge||!options.allow_enable))throw std::runtime_error("actuation needs acknowledgement and local enable permission");
        if(acknowledge&&!actuate)throw std::runtime_error("actuation acknowledgement requires --actuate");
        std::signal(SIGINT,Signal);std::signal(SIGTERM,Signal);
        auto device=backend=="mujoco"?opensprite::runtime::MakeMujocoBackend(config):
            opensprite::runtime::MakeHardwareBackend(config,actuate);
        auto result=opensprite::runtime::RunRobot(std::move(device),options,&stop);
        write_report(result);
        std::cout<<result.dump(2)<<'\n';
        return result.at("fault").is_null()?0:1;
    } catch(const std::exception& error) {
        const nlohmann::json failure={{"schema","opensprite-native-runtime-v1"},{"fault",error.what()},
            {"hardware_tested",false},{"video_recorded",false}};
        try {write_report(failure);}catch(const std::exception& report_error){std::cerr<<"report error: "<<report_error.what()<<'\n';}
        std::cerr<<failure.dump(2)<<'\n';return 1;
    }
}
