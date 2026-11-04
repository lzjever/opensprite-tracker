# OpenSprite joint coordinates, SI units. AGPL-3.0-only.
string<=128 profile_id
uint64 joint_order_hash
uint64 seq
uint64 stamp_ns
string<=16 clock_domain
string<=128 session_id
JointCmd[31] joint_cmd
uint64 source_state_seq
string<=128 source_state_session_id
