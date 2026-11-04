"""Jetson client for the SDK's native CAN daemon and native ICM IMU reader.

This module does not open CAN, enable motors at import, start a controller, or
claim measured latency. The native supervisor is launched explicitly by users.
"""
import copy
from collections import OrderedDict
from pathlib import Path
import socket
import tempfile
import time
import threading
from functools import wraps
import uuid
import numpy as np
from ..constants import JOINT_NAMES, JOINT_ORDER_HASH
from ..messages import LowCmd_, LowState_, JointState, IMUState
from .hardware_config import HardwareConfig
from .native_ipc import StatePacket, decode_icm, fresh, pack_target


def _state_synchronized(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)
    return locked


class HardwareBackend:
    # Actor sources can legitimately lag the newest 50 Hz native snapshot.
    # This is separate from the fresher feedback used by the native controller.
    source_cache_size = 64
    source_max_age_ns = 60_000_000
    capabilities = {'joint_coordinates':True,'joint_sample_timestamps':True,
                    'joint_sample_time_kind':'host_receive','torque_feedback':False,
                    'motor_enable_feedback':False,'global_pose':False,
                    'step':False,'native_state_abi':2,'native_target_abi':1}

    def __init__(self,config,*,profile_id='physical_v5',allow_actuation=False,
                 clock=time.monotonic_ns,state_socket=None,imu_socket=None):
        self.config=config if isinstance(config,HardwareConfig) else HardwareConfig.load(config)
        if profile_id != self.config.data.get('profile_id', 'physical_v5'):
            raise ValueError('hardware profile must match the native robot configuration')
        self.profile_id=profile_id;self.allow_actuation=allow_actuation;self.clock=clock
        self._state_socket=state_socket;self._imu_socket=imu_socket;self._temporary=None
        self._started=False;self._closed=False;self._enabled=False;self._fault=None
        self._previous_state=0;self._previous_imu=0;self._native_target=0;self._last_command=0
        self._state=None;self._packet=None;self._owner=None;self.session_id=uuid.uuid4().hex
        self._source_states = OrderedDict()
        self._state_lock = threading.RLock()

    def start(self):
        if self._closed:
            raise RuntimeError('hardware backend is closed; create a new session')
        if self._started:return self
        try:
            if self.config.data.get('synthetic') and (self._state_socket is None or self._imu_socket is None):
                raise RuntimeError('synthetic CPU fixtures require injected fake-peer sockets')
            if self._state_socket is None:
                self._state_socket=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET)
                self._state_socket.settimeout(.5)
                self._state_socket.connect(self.config.data['native']['socket_path'])
            self._state_socket.settimeout(.5)
            if self._imu_socket is None:
                self._temporary=tempfile.TemporaryDirectory(prefix='opensprite-sdk-imu-')
                self._imu_socket=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM)
                self._imu_socket.bind(str(Path(self._temporary.name)/'rx.sock'))
                self._imu_socket.settimeout(.03)
                self._imu_socket.connect(self.config.data['imu']['socket_path'])
            self._imu_socket.settimeout(.03)
            self._started=True
            return self
        except BaseException:
            self.close();raise

    def _require_open(self):
        if not self._started or self._closed or self._fault:
            raise RuntimeError(f'hardware backend unavailable: {self._fault or "not started/closed"}')

    def read(self):
        self._require_open()
        try:
            data=self.config.data
            payload=self._state_socket.recv(777)
            if not payload:
                raise EOFError('native controller closed state IPC')
            now=self.clock()
            packet=StatePacket.unpack(payload,expected_hash=self.config.motor_order_hash,
                previous_sequence=self._previous_state,now=now,
                max_age_ns=int(data['controller'].get('native_state_max_age_ms',60)*1e6))
            if self.allow_actuation and not packet.actuation_allowed:
                raise RuntimeError('actuation backend connected to zero-gain-only shadow daemon')
            self._previous_state=packet.sequence
            # The daemon provides actual per-motor host receive stamps. Verify
            # each endpoint independently; an aggregate stamp cannot mask freezes.
            ankle_names=set()
            for key in ('left_ankle','right_ankle'):
                ankle_names.update(data['differentials'][key]['motor_names'])
            for name,stamp in zip(self.config.motor_names,packet.host_receive_ns):
                fresh(stamp,now,10_000_000 if name in ankle_names else 60_000_000,name)
            if self._imu_socket.send(b'GETIMU01')!=8:
                raise RuntimeError('short ICM IPC request')
            imu=decode_icm(self._imu_socket.recv(193),now=self.clock(),
                rotation=data['imu']['sensor_to_body_rotation'],previous_sequence=self._previous_imu,
                max_age_ns=int(data['imu']['max_age_ms']*1e6))
            self._previous_imu=imu.sequence
            q,dq=self.config.joint_state(packet.position,packet.velocity)
            # The state combines independently sampled native and IMU data.
            # Its header is robot-local collection time, which can anchor
            # remote observation-age checks without comparing machine clocks.
            # Per-sensor timestamps remain their original measurement times.
            state=LowState_(profile_id=self.profile_id,seq=packet.sequence,stamp_ns=self.clock(),
                session_id=self.session_id,clock_domain='monotonic',
                status='ready' if packet.actuation_allowed else 'shadow')
            for i in range(31):
                dependencies=np.flatnonzero(np.abs(self.config.inverse[i])>1e-12)
                stamp=min(packet.host_receive_ns[j] for j in dependencies)
                state.joint_state[i]=JointState(q=float(q[i]),dq=float(dq[i]),tau_est=0.,valid=True,
                    sample_ns=int(stamp),sample_time_kind='host_receive')
            state.imu_state=IMUState(gyroscope=imu.gyro,projected_gravity=imu.projected_gravity,
                accelerometer=imu.accelerometer,valid=True,sample_ns=imu.sample_ns,
                sample_time_kind='host_read',frame='body',attached_link=data['imu']['attached_link'])
            state.validate()
            # Blocking native/IMU acquisition above never holds the command
            # lock. Commit the matching packet, state and source cache together.
            with self._state_lock:
                self._require_open()
                self._state=state;self._packet=packet
                self._require_fresh_state()
                self._remember_source(state)
                return copy.deepcopy(state)
        except BaseException as error:
            self._fault=str(error);self.close();raise

    def _require_fresh_state(self):
        if self._state is None or self._packet is None:
            raise RuntimeError('read a fresh joint and IMU state first')
        now=self.clock()
        fresh(self._state.stamp_ns,now,int(self.config.data['controller'].get('native_state_max_age_ms',60)*1e6),'source state')
        fresh(self._packet.stamp_ns,now,int(self.config.data['controller'].get('native_state_max_age_ms',60)*1e6),'native aggregate state')
        fresh(self._state.imu_state.sample_ns,now,int(self.config.data['imu']['max_age_ms']*1e6),'source IMU')
        ankle_names=set()
        for name in ('left_ankle','right_ankle'):
            ankle_names.update(self.config.data['differentials'][name]['motor_names'])
        for name,stamp in zip(self.config.motor_names,self._packet.host_receive_ns):
            fresh(stamp,now,10_000_000 if name in ankle_names else 60_000_000,name)

    def _remember_source(self, state):
        # Public callers receive mutable SDK messages. Never use their copy as
        # the authority for a later command's causal source or sample ages.
        self._source_states[state.seq] = copy.deepcopy(state)
        now = self.clock()
        while self._source_states:
            sequence, snapshot = next(iter(self._source_states.items()))
            if (len(self._source_states) <= self.source_cache_size
                    and now-snapshot.stamp_ns <= self.source_max_age_ns):
                break
            del self._source_states[sequence]

    def _require_fresh_source(self, command):
        if command.source_state_session_id != self.session_id:
            raise ValueError('target source session mismatch')
        source = self._source_states.get(command.source_state_seq)
        if source is None:
            raise ValueError('target source sequence is unknown, expired or in the future')
        now = self.clock()
        fresh(source.stamp_ns, now, self.source_max_age_ns, 'command source state')
        for joint in source.joint_state:
            if not joint.valid or joint.sample_time_kind == 'unknown':
                raise RuntimeError('command source joint measurement is invalid')
            fresh(joint.sample_ns, now, self.source_max_age_ns, 'command source joint')
        if not source.imu_state.valid or source.imu_state.sample_time_kind == 'unknown':
            raise RuntimeError('command source IMU measurement is invalid')
        fresh(source.imu_state.sample_ns, now,
              int(self.config.data['imu']['max_age_ms']*1e6), 'command source IMU')

    def _send(self,command):
        self._require_fresh_state()
        self._require_fresh_source(command)
        self._native_target+=1
        payload=pack_target(command,self.clock(),JOINT_ORDER_HASH,self._native_target,
                            command.source_state_seq)
        if self._state_socket.send(payload)!=len(payload):
            raise RuntimeError('short native target IPC write')

    @_state_synchronized
    def enable(self):
        """Explicit permission: release the first zero-gain startup target.

        The independently launched native process must also have --policy-actuation.
        No mode telemetry exists in ABI2; SDK status remains ready because motor enable feedback is not in this ABI.
        Check the native final enable/disable report.
        """
        self._require_open()
        if self._enabled:return
        if not self.allow_actuation:
            raise RuntimeError('backend constructed without explicit actuation permission')
        if self._state is None:
            raise RuntimeError('read a fresh joint and IMU state before enabling')
        if not self._packet.actuation_allowed:
            raise RuntimeError('zero-gain-only shadow daemon cannot enable motors')
        self._require_fresh_state()
        command=LowCmd_(profile_id=self.profile_id,source_state_seq=self._state.seq,
                        source_state_session_id=self.session_id)
        for target,measured in zip(command.joint_cmd,self._state.joint_state):
            target.mode=1;target.q=measured.q
        try:self._send(command);self._enabled=True
        except BaseException:
            self.close();raise

    def write(self,command):
        return self._accept(command,preview=False)

    def preview(self,command):
        """Send a fully validated policy target to a proven zero-gain-only daemon.

        Shadow native code computes motor commands but transmits only disabled
        zero-gain echo frames. Mode is declared by ABI2 kind, never inferred from
        the client flag or filename.
        """
        return self._accept(command,preview=True)

    @_state_synchronized
    def _accept(self,command,*,preview):
        self._require_open();command.validate()
        if self._packet is None:
            raise RuntimeError('read native state before sending a command')
        if preview and self._packet.actuation_allowed:
            raise RuntimeError('preview requires a zero-gain-only shadow daemon')
        if not preview and not self._packet.actuation_allowed:
            raise RuntimeError('shadow daemon requires preview(), not actuating write()')
        modes={x.mode for x in command.joint_cmd}
        if len(modes)!=1:
            raise ValueError('hardware supports whole-body mode only; mixed per-joint modes rejected')
        if command.profile_id!=self.profile_id or command.clock_domain!='monotonic':
            raise ValueError('hardware command profile/clock mismatch')
        if not command.session_id or (self._owner is not None and command.session_id!=self._owner):
            raise ValueError('hardware command controller session/owner mismatch')
        self._require_fresh_source(command)
        if command.seq<=self._last_command:
            raise ValueError('command sequence regression/replay')
        fresh(command.stamp_ns,self.clock(),100_000_000,'command')
        self._require_fresh_state()
        # Disable follows exactly the same ownership/sequence/freshness gate.
        if modes=={0}:
            self._owner=command.session_id;self._last_command=command.seq
            if self._enabled:self.disable()
            return
        if not preview and not self._enabled:
            raise RuntimeError('call enable() explicitly before writing an actuating command')
        # Reject out-of-contract targets rather than silently altering a policy.
        for name,target in zip(JOINT_NAMES,command.joint_cmd):
            limits=self.config.data['joint_limits'][name];low,high=limits['position_rad']
            if not low<=target.q<=high or abs(target.dq)>limits['velocity_rad_s'] or target.kp>limits['kp'] or target.kd>limits['kd'] or abs(target.tau)>limits['effort_nm']:
                raise ValueError(f'target exceeds configured joint envelope: {name}')
        try:
            self._send(command);self._last_command=command.seq;self._owner=command.session_id
        except BaseException as error:
            self._fault=str(error);self.close();raise

    def disable(self):
        # Disconnect is the native watchdog/disable path. Do not manufacture a
        # successful disable acknowledgement: inspect the native report on exit.
        self.close()

    @_state_synchronized
    def close(self):
        self._enabled=False;self._closed=True
        self._source_states.clear()
        for sock in (self._state_socket,self._imu_socket):
            if sock is not None:
                try:sock.close()
                except OSError:pass
        if self._temporary is not None:
            self._temporary.cleanup();self._temporary=None

    def __enter__(self):
        return self.start()

    def __exit__(self,*_):
        self.close()
