// Offline regression tests against the exact compiled deployment implementation.
#define main opensprite_vendor_controller_main
#include "../vendor/sprite_can_shadow.cpp"
#undef main
#include <cassert>
#include <atomic>
#include <thread>

struct PolicyFixture {
  std::vector<Motor> motors = std::vector<Motor>(31);
  NativeKinematics kinematics;
  IpcSocket ipc;
  PolicyFixture() {
    ipc.has_target = true;
    for (std::size_t i=0; i<31; ++i) {
      auto& m=motors[i];m.name="motor"+std::to_string(i);
      m.position_min=-12.5;m.position_max=12.5;m.velocity_min=-20;m.velocity_max=20;
      m.torque_min=-40;m.torque_max=40;m.soft_position_min=-8;m.soft_position_max=8;
      m.hard_position_min=-10;m.hard_position_max=10;m.deployment_velocity_max=20;
      m.mechanical_peak_torque=40;m.commissioning_torque_cap=5;
      m.poll_rate_hz=50;m.poll_phase=0;
      kinematics.joint_names[i]="joint"+std::to_string(i);
      kinematics.joint_effort_limit[i]=4;
      kinematics.joint_to_motor[i][i]=1;
      kinematics.motor_to_joint[i][i]=1;
    }
  }
  void differential(std::size_t index) {
    auto& a=kinematics.joint_to_motor;auto& b=kinematics.motor_to_joint;
    a[index][index]=1;a[index][index+1]=1;a[index+1][index]=1;a[index+1][index+1]=-1;
    b[index][index]=.5;b[index][index+1]=.5;b[index+1][index]=.5;b[index+1][index+1]=-.5;
    kinematics.joint_names[index]="head_pitch_joint";
    kinematics.joint_names[index+1]="head_roll_joint";
  }
};

template<class Function> bool rejects(Function function) {
  try {function();} catch(const std::runtime_error&) {return true;}
  return false;
}

void test_joint_effort_contract() {
  for (bool head: {false,true}) {
    PolicyFixture f;if(head)f.differential(0);
    f.kinematics.joint_effort_limit[0]=1;
    f.ipc.latest_target.position_rad[0]=.2;f.ipc.latest_target.kp[0]=10;
    assert(rejects([&]{preview_final_motor_commands(f.motors,f.ipc,f.kinematics);}));
  }
  // All impedance contributions, not just feedforward, share the joint cap.
  for (int term=0;term<3;++term) {
    PolicyFixture f;f.kinematics.joint_effort_limit[0]=1;
    if(term==0){f.ipc.latest_target.position_rad[0]=.2;f.ipc.latest_target.kp[0]=10;}
    if(term==1){f.ipc.latest_target.velocity_rad_s[0]=2;f.ipc.latest_target.kd[0]=1;}
    if(term==2)f.ipc.latest_target.feedforward_torque_nm[0]=2;
    assert(rejects([&]{preview_final_motor_commands(f.motors,f.ipc,f.kinematics);}));
  }
  // Reject a now-invalid held internal-PD command even between 50 Hz refreshes.
  {
    PolicyFixture f;f.kinematics.joint_effort_limit[0]=.5;
    f.ipc.latest_target.position_rad[0]=.01;f.ipc.latest_target.kp[0]=10;
    preview_final_motor_commands(f.motors,f.ipc,f.kinematics);
    f.motors[0].last_position=-.1;assert(f.ipc.motor_preview_count==1);
    assert(rejects([&]{preview_final_motor_commands(f.motors,f.ipc,f.kinematics);}));
  }
  // Wire q quantization can increase magnitude even when raw PD is within cap.
  // Exercise identity, signed reduction and non-unit transmission explicitly.
  for (double ratio: {1., .5, -2.}) {
    PolicyFixture f;f.kinematics.joint_effort_limit[0]=.1;
    f.kinematics.joint_to_motor[0][0]=ratio;f.kinematics.motor_to_joint[0][0]=1/ratio;
    f.ipc.latest_target.position_rad[0]=ratio<0 ? -.00001 : .00001;
    f.ipc.latest_target.kp[0]=100;
    f.motors[0].last_position=ratio*(ratio<0 ? -.001 : .001);
    preview_final_motor_commands(f.motors,f.ipc,f.kinematics);
    std::array<bool,31> due{};due.fill(true);
    assert(rejects([&]{(void)checked_policy_slot(f.motors,f.kinematics,due);}));
  }
  // Zero-gain, not-yet-updated motors have finite-resolution wire residuals.
  {
    PolicyFixture f;std::array<bool,31> due{};
    auto payloads=checked_policy_slot(f.motors,f.kinematics,due);
    assert(payloads[0]==zero_gain_payload(f.motors[0]));
    f.kinematics.joint_effort_limit[0]=1e-5;
    assert(rejects([&]{(void)checked_policy_slot(f.motors,f.kinematics,due);}));
  }
  // Quantized held frames are checked against newest measured q/dq at every slot.
  {
    PolicyFixture f;f.kinematics.joint_effort_limit[0]=.5;
    f.ipc.latest_target.position_rad[0]=.01;f.ipc.latest_target.kp[0]=10;
    preview_final_motor_commands(f.motors,f.ipc,f.kinematics);
    std::array<bool,31> due{};due.fill(true);
    auto payloads=checked_policy_slot(f.motors,f.kinematics,due);
    for(std::size_t i=0;i<31;++i){f.motors[i].last_policy_payload=payloads[i];f.motors[i].last_policy_payload_valid=true;}
    f.motors[0].last_position=-.1;due.fill(false);
    assert(rejects([&]{(void)checked_policy_slot(f.motors,f.kinematics,due);}));
  }
  // Head A/B updates occupy separate slots. Both complete targets are safe,
  // but updating A while B holds its old command would overtorque head pitch.
  {
    PolicyFixture f;f.differential(0);f.kinematics.joint_effort_limit[0]=.2;
    f.kinematics.joint_effort_limit[1]=2;
    f.ipc.latest_target.position_rad[1]=.1;f.ipc.latest_target.kp[0]=10;f.ipc.latest_target.kp[1]=10;
    preview_final_motor_commands(f.motors,f.ipc,f.kinematics);
    std::array<bool,31> due{};due.fill(true);
    // An existing valid held state, reachable by small successive updates.
    for(std::size_t i=0;i<31;++i){f.motors[i].last_policy_payload=protected_policy_payload(f.motors[i]);f.motors[i].last_policy_payload_valid=true;}
    std::array<bool,31> held{};
    (void)checked_policy_slot(f.motors,f.kinematics,held);
    f.ipc.latest_target.position_rad[1]=-.1;f.ipc.motor_preview_count=10;
    preview_final_motor_commands(f.motors,f.ipc,f.kinematics);
    assert(rejects([&]{(void)checked_policy_slot(f.motors,f.kinematics,due);}));
    due.fill(false);due[0]=true;
    assert(rejects([&]{(void)checked_policy_slot(f.motors,f.kinematics,due);}));
  }
  {
    PolicyFixture f;f.differential(0);f.kinematics.joint_effort_limit[0]=.2;
    f.ipc.latest_target.position_rad[1]=.01;
    f.ipc.latest_target.kp[0]=f.ipc.latest_target.kp[1]=10;
    preview_final_motor_commands(f.motors,f.ipc,f.kinematics);
    std::array<bool,31> due{};due.fill(true);
    (void)checked_policy_slot(f.motors,f.kinematics,due);
  }
  std::cout<<"PASS complete joint effort, wire quantization, transmission and staggered head limits\n";
}

void test_best_effort_shutdown(FeedbackDrain drain) {
  std::vector<Socket> sockets{{"broken",-1}};
  std::vector<Motor> motors(4);
  std::map<std::string,std::size_t> indices{{"broken",0}};
  std::map<std::pair<std::string,int>,std::size_t> feedback;
  std::array<std::array<int,2>,3> peers{};
  std::array<std::atomic<int>,3> disable_counts{};
  for(std::size_t i=0;i<4;++i) {
    auto& m=motors[i];m.name="shutdown_motor"+std::to_string(i);
    m.interface=i==0?"broken":"healthy"+std::to_string(i);
    m.can_id=1;m.master_id=17;m.poll_rate_hz=50;m.poll_phase=0;
    m.position_min=-12.5;m.position_max=12.5;m.velocity_min=-20;m.velocity_max=20;
    m.torque_min=-40;m.torque_max=40;m.mos_temperature_limit=80;m.rotor_temperature_limit=80;
    m.final_disabled=true; // Stale success must not survive a fresh failed shutdown.
    if(i>0) {
      assert(socketpair(AF_UNIX,SOCK_DGRAM|SOCK_NONBLOCK,0,peers[i-1].data())==0);
      sockets.push_back({m.interface,peers[i-1][0]});indices[m.interface]=i;
      feedback[{m.interface,m.master_id}]=i;
    }
  }
  std::atomic<bool> stop{false};
  std::thread responder([&] {
    while(!stop.load()) {
      for(std::size_t i=0;i<3;++i) {
        canfd_frame command{};
        while(recv(peers[i][1],&command,sizeof(command),MSG_DONTWAIT)>0) {
          if(command.data[7]==0xFD && command.data[0]==0xFF){++disable_counts[i];continue;}
          canfd_frame reply{};reply.can_id=17;reply.len=8;reply.flags=CANFD_BRS;
          reply.data[0]=1;reply.data[1]=0x7F;reply.data[2]=0xFF;
          reply.data[3]=0x7F;reply.data[4]=0xF7;reply.data[5]=0xFF;
          reply.data[6]=20;reply.data[7]=20;
          assert(send(peers[i][1],&reply,sizeof(reply),0)==sizeof(reply));
        }
      }
      usleep(100);
    }
  });
  const auto failure=best_effort_disable(sockets,motors,indices,feedback,drain);
  stop=true;responder.join();
  assert(!failure.empty() && failure.find("broken")!=std::string::npos);
  assert(!motors[0].final_disabled);
  for(std::size_t i=0;i<3;++i) {
    assert(disable_counts[i]==3 && motors[i+1].final_disabled);
    close(peers[i][0]);close(peers[i][1]);
  }
  std::cout<<"PASS shutdown: faulty bus isolated, every healthy motor disabled and verified\n";
}

void test_full_policy_failure_cleanup(std::vector<Motor> motors,
                                     const NativeKinematics& kinematics,
                                     const std::vector<JointSafety>& limits,
                                     const std::string& report_path) {
  const std::string bad_bus=motors.front().interface;
  std::vector<Socket> sockets{{bad_bus,-1}};
  std::map<std::string,std::size_t> indices{{bad_bus,0}};
  std::map<std::pair<std::string,int>,std::size_t> feedback;
  std::vector<std::array<int,2>> peers;
  std::vector<std::string> healthy_names;
  std::array<std::atomic<int>,31> disable_counts{};
  for(std::size_t i=0;i<motors.size();++i) {
    const auto& m=motors[i];feedback[{m.interface,m.master_id}]=i;
    if(indices.count(m.interface))continue;
    std::array<int,2> fds{};assert(socketpair(AF_UNIX,SOCK_DGRAM|SOCK_NONBLOCK,0,fds.data())==0);
    indices[m.interface]=sockets.size();sockets.push_back({m.interface,fds[0]});
    peers.push_back(fds);healthy_names.push_back(m.interface);
  }
  std::atomic<bool> stop{false};
  std::thread responder([&]{
    while(!stop.load()) {
      for(std::size_t bus=0;bus<peers.size();++bus) {
        canfd_frame command{};
        while(recv(peers[bus][1],&command,sizeof(command),MSG_DONTWAIT)>0) {
          auto motor=std::find_if(motors.begin(),motors.end(),[&](const Motor& m){
            return m.interface==healthy_names[bus] && m.can_id==static_cast<int>(command.can_id);
          });
          assert(motor!=motors.end());
          const auto index=static_cast<std::size_t>(motor-motors.begin());
          if(command.data[0]==0xFF && command.data[7]==0xFD){++disable_counts[index];continue;}
          canfd_frame reply{};reply.can_id=static_cast<canid_t>(motor->master_id);
          reply.len=8;reply.flags=CANFD_BRS;reply.data[0]=static_cast<std::uint8_t>(motor->can_id);
          reply.data[1]=0x7F;reply.data[2]=0xFF;reply.data[3]=0x7F;reply.data[4]=0xF7;reply.data[5]=0xFF;
          reply.data[6]=20;reply.data[7]=20;
          assert(send(peers[bus][1],&reply,sizeof(reply),0)==sizeof(reply));
        }
      }
      usleep(100);
    }
  });
  Options options;options.duration_s=.01;options.output=report_path;
  IpcSocket ipc;
  const int result=run_native_protected_policy(options,sockets,motors,indices,feedback,
      kinematics,limits,ipc,123,false);
  stop=true;responder.join();
  assert(result==2);
  for(std::size_t i=0;i<motors.size();++i) {
    if(motors[i].interface==bad_bus){assert(!motors[i].final_disabled);continue;}
    assert(disable_counts[i]==3 && motors[i].final_disabled);
  }
  std::ifstream report(report_path);std::stringstream text;text<<report.rdbuf();
  assert(text.str().find("\"passed\": false")!=std::string::npos);
  for(const auto& pair:peers){close(pair[0]);close(pair[1]);}
  std::cout<<"PASS complete native failure path: healthy bus disable, fault report retained\n";
}

int main(int argc, char** argv) {
  test_joint_effort_contract();
  test_best_effort_shutdown(drain_policy_feedback);
  test_best_effort_shutdown(drain_hold_feedback);
  static_assert(sizeof(StatePacket)==776,"state ABI2");
  static_assert(sizeof(TargetPacket)==1280,"target ABI1");
  Motor codec;codec.position_min=-12.5;codec.position_max=12.5;
  codec.velocity_min=-20;codec.velocity_max=20;codec.torque_min=-10;codec.torque_max=10;
  codec.last_position=0;codec.soft_position_min=-1;codec.soft_position_max=1;
  const std::array<std::uint8_t,8> zero_golden{0x7f,0xff,0x7f,0xf0,0,0,0x07,0xff};
  assert(zero_gain_payload(codec)==zero_golden);
  const std::array<std::uint8_t,8> hold_golden{0x7f,0xff,0x7f,0xf0,1,2,0x87,0xff};
  assert(hold_payload(codec,0,.2,.05,0,1)==hold_golden);
  bool invalid_hold=false;
  try {hold_payload(codec,2,.2,.05,0,1);} catch(const std::runtime_error&) {invalid_hold=true;}
  assert(invalid_hold);
  std::vector<Motor> motors(31);
  for(std::size_t i=0;i<motors.size();++i) {
    auto& m=motors[i];m.name="motor"+std::to_string(i);m.poll_rate_hz=i<4?500:50;
    m.seen=true;m.last_feedback_ns=1'000'000'000LL;
    m.last_position=0.1*static_cast<double>(i);m.last_velocity=-0.1*static_cast<double>(i);
  }
  require_motor_feedback_fresh(motors,1'005'000'000LL);
  bool stale=false;
  try {require_motor_feedback_fresh(motors,1'010'000'001LL);} catch(const std::runtime_error&) {stale=true;}
  assert(stale);
  for(auto& m:motors)m.last_feedback_ns=1'100'000'000LL;
  bool future=false;
  try {require_motor_feedback_fresh(motors,1'005'000'000LL);} catch(const std::runtime_error&) {future=true;}
  assert(future);
  for(auto& m:motors)m.last_feedback_ns=1'000'000'000LL;
  int fds[2];assert(socketpair(AF_UNIX,SOCK_SEQPACKET|SOCK_NONBLOCK,0,fds)==0);
  IpcSocket ipc;ipc.client_fd=fds[0];ipc.actuation_allowed=true;
  send_state_packet(ipc,motors,123,1'005'000'000LL);
  StatePacket packet{};
  assert(recv(fds[1],&packet,sizeof(packet),0)==sizeof(packet));
  assert(packet.version==2 && packet.sequence==1 && packet.motor_order_hash==123);
  assert(packet.host_receive_ns[30]==1'000'000'000LL);
  assert(packet.position_rad[30]==3.0);
  ipc.actuation_allowed=false;
  send_state_packet(ipc,motors,123,1'005'000'000LL);
  StatePacket shadow{};
  assert(recv(fds[1],&shadow,sizeof(shadow),0)==sizeof(shadow));
  assert(shadow.kind==3 && shadow.version==2);

  // Queue zero startup and an active target together. The gate must observe
  // the first packet, then the normal control loop may consume the next.
  TargetPacket first{};std::memcpy(first.magic,"SPRT",4);first.version=1;first.kind=2;
  first.sequence=1;first.monotonic_ns=1'005'000'000LL;first.joint_order_hash=456;
  // A normal 50 Hz asynchronous client can use the previous still-fresh state.
  first.source_state_sequence=ipc.state_sequence-1;
  TargetPacket second=first;second.sequence=2;second.position_rad[0]=.2;
  assert(send(fds[1],&first,sizeof(first),0)==sizeof(first));
  assert(send(fds[1],&second,sizeof(second),0)==sizeof(second));
  drain_target_packets(ipc,456,{},1'005'000'000LL,true);
  assert(ipc.target_count==1 && ipc.latest_target.position_rad[0]==0.);
  drain_target_packets(ipc,456,{},1'005'000'000LL,true);
  assert(ipc.target_count==1 && ipc.latest_target.position_rad[0]==0.);
  drain_target_packets(ipc,456,{},1'005'000'000LL);
  assert(ipc.target_count==2 && ipc.latest_target.position_rad[0]==.2);
  TargetPacket future_target=second;future_target.sequence=3;
  future_target.source_state_sequence=ipc.state_sequence+1;
  assert(send(fds[1],&future_target,sizeof(future_target),0)==sizeof(future_target));
  assert(rejects([&]{drain_target_packets(ipc,456,{},1'005'000'000LL);}));
  assert(ipc.target_count==2);
  future_target.sequence=4;future_target.source_state_sequence=0;
  assert(send(fds[1],&future_target,sizeof(future_target),0)==sizeof(future_target));
  assert(rejects([&]{drain_target_packets(ipc,456,{},1'005'000'000LL);}));
  assert(ipc.target_count==2);
  // At500Hz a normal50Hz controller legitimately references >5 older states.
  // Trust emitted sequence+time, not a rate-dependent frame-count allowance.
  for(int i=0;i<20;++i) {
    send_state_packet(ipc,motors,123,1'005'000'000LL);
    StatePacket observed{};assert(recv(fds[1],&observed,sizeof(observed),0)==sizeof(observed));
  }
  TargetPacket historical=first;historical.sequence=5;historical.source_state_sequence=1;
  assert(send(fds[1],&historical,sizeof(historical),0)==sizeof(historical));
  drain_target_packets(ipc,456,{},1'005'000'000LL);
  assert(ipc.target_count==3);
  historical.sequence=6;historical.monotonic_ns=1'066'000'000LL;
  assert(send(fds[1],&historical,sizeof(historical),0)==sizeof(historical));
  assert(rejects([&]{drain_target_packets(ipc,456,{},1'066'000'000LL);}));
  {
    PolicyFixture fullbody;
    fullbody.ipc.latest_target.position_rad[0]=.01;
    fullbody.ipc.latest_target.kp[0]=1;
    preview_final_motor_commands(fullbody.motors,fullbody.ipc,fullbody.kinematics);
    fullbody.ipc.latest_target.position_rad[0]=.02;
    preview_final_motor_commands(fullbody.motors,fullbody.ipc,fullbody.kinematics);
    assert(fullbody.motors[0].preview_position==.02);
  }
  if(argc>=2) {
    std::ofstream output(argv[1],std::ios::binary);
    output.write(reinterpret_cast<const char*>(&packet),sizeof(packet));
    assert(output.good());
  }
  if(argc>=2) {
    const std::string lock_path=std::string(argv[1])+".lock";
    {
      ControllerOwnerLock first(lock_path);bool duplicate=false;
      try {ControllerOwnerLock second(lock_path);} catch(const std::runtime_error&) {duplicate=true;}
      assert(duplicate);
    }
    {ControllerOwnerLock released(lock_path);}
    unlink(lock_path.c_str());
  }
  if(argc==5) {
    const auto loaded=load_motors(argv[2]);
    const auto map=load_kinematics(argv[3],loaded);
    const auto limits=load_joint_safety(argv[4]);
    assert(loaded.size()==31 && limits.size()==31);
    assert(!map.joint_names[0].empty());
    std::map<std::string,std::array<int,4>> per_bus;
    for(const auto& motor:loaded) {
      assert(motor.poll_rate_hz==500 && motor.poll_phase>=0 && motor.poll_phase<4);
      int count=0;
      for(std::size_t slot=0;slot<2000;++slot)count+=opensprite_can::Due(slot,motor.poll_phase);
      assert(count==500);
      ++per_bus[motor.interface][motor.poll_phase];
    }
    assert(per_bus.size()==4);
    for(const auto& bus:per_bus)for(int count:bus.second)assert(count<=2);
    const auto budget=opensprite_can::CheckBudget(8,1'000'000,5'000'000);
    assert(std::abs(budget.utilization-.7168)<1e-12);
    assert(budget.worst_slot_seconds<.0004);
    assert(rejects([]{opensprite_can::CheckBudget(8,1'000'000,2'000'000);}));
    assert(rejects([]{opensprite_can::CheckBudget(8,500'000,5'000'000);}));
    assert(rejects([]{opensprite_can::CheckLink("lo",8);}));
    for(unsigned mode:{CAN_CTRLMODE_LISTENONLY,CAN_CTRLMODE_LOOPBACK}) {
      opensprite_can::Link bad{1'000'000,5'000'000,CAN_CTRLMODE_FD|mode,true,true};
      assert(rejects([&]{opensprite_can::CheckLinkConfiguration(bad,8);}));
    }
    std::cout<<"PASS all31 motors500Hz,balanced4slot schedule,FD wire budget andlink gate\n";
    test_full_policy_failure_cleanup(loaded,map,limits,std::string(argv[1])+".cleanup.json");
  }
  close(fds[0]);close(fds[1]);
  std::cout<<"PASS state ABI2/target ABI1, native wire, per-motor stale/future guard\n";
  return 0;
}
