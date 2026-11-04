# OpenSprite joint coordinates, SI units. AGPL-3.0-only.
string<=128 profile_id
uint64 joint_order_hash
uint64 seq
uint64 stamp_ns
string<=16 clock_domain
string<=128 session_id
JointState[31] joint_state
IMUState imu_state
string<=16 status
string<=1024[<=64] faults
uint64 accepted_command_seq
string<=128 accepted_controller_session
