// Offline regression tests against the exact compiled deployment implementation.
#define main opensprite_vendor_controller_main
#include "../vendor/sprite_can_shadow.cpp"
#undef main
#include <cassert>
int main(int argc, char** argv) {
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
  first.source_state_sequence=ipc.state_sequence;
  TargetPacket second=first;second.sequence=2;second.position_rad[0]=.2;
  assert(send(fds[1],&first,sizeof(first),0)==sizeof(first));
  assert(send(fds[1],&second,sizeof(second),0)==sizeof(second));
  drain_target_packets(ipc,456,{},1'005'000'000LL,true);
  assert(ipc.target_count==1 && ipc.latest_target.position_rad[0]==0.);
  drain_target_packets(ipc,456,{},1'005'000'000LL,true);
  assert(ipc.target_count==1 && ipc.latest_target.position_rad[0]==0.);
  drain_target_packets(ipc,456,{},1'005'000'000LL);
  assert(ipc.target_count==2 && ipc.latest_target.position_rad[0]==.2);
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
  }
  close(fds[0]);close(fds[1]);
  std::cout<<"PASS state ABI2/target ABI1, native wire, per-motor stale/future guard\n";
  return 0;
}
