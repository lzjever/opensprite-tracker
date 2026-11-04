"""Measured hardware configuration and portable native-controller inputs.

No runtime import from the research checkout or another repository is required.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import numpy as np
from ..constants import JOINT_NAMES, JOINT_ORDER_HASH, ordered_name_hash

NATIVE_FIELDS = ('motor_name','interface','can_id','master_id','position_min_rad','position_max_rad','velocity_min_rad_s','velocity_max_rad_s','torque_min_nm','torque_max_nm','soft_position_min_rad','soft_position_max_rad','hard_position_min_rad','hard_position_max_rad','deployment_velocity_max_rad_s','rated_torque_nm','mechanical_peak_torque_nm','commissioning_torque_cap_nm','feedback_torque_cap_nm','mos_temperature_limit_c','rotor_temperature_limit_c','filtered_mos_temperature_limit_c','filtered_rotor_temperature_limit_c','temperature_filter_time_constant_s','poll_rate_hz')
JOINT_FIELDS = ('joint_name','position_min_rad','position_max_rad','velocity_max_rad_s','kp_max','kd_max','feedforward_torque_max_nm')


def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
        raise ValueError(f'{name}: measured finite numeric value required')
    if positive and value <= 0:
        raise ValueError(f'{name}: must be positive')
    return float(value)


def pair(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f'{name}: measured range required')
    result = tuple(number(x,name) for x in value)
    if result[0] >= result[1]:
        raise ValueError(f'{name}: range must increase')
    return result


@dataclass
class HardwareConfig:
    data: dict
    motor_names: tuple
    matrix: np.ndarray
    offset: np.ndarray
    inverse: np.ndarray

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != 'opensprite-sdk-hardware-v1':
            raise ValueError('hardware schema must be opensprite-sdk-hardware-v1')
        if data.get('configured') is not True or not data.get('hardware_identity'):
            raise ValueError('hardware requires completed configuration and individual hardware_identity')
        if data.get('estop',{}).get('configured') is not True:
            raise ValueError('independent emergency-stop chain must be configured')
        interfaces=data.get('can_adapter',{}).get('interfaces',[])
        if len(interfaces)!=4 or any(not isinstance(x,str) or not x or '/' in x for x in interfaces) or len(set(interfaces))!=4:
            raise ValueError('four distinct SocketCAN interface names required')
        imu=data.get('imu',{})
        if imu.get('mounting_verified') is not True or imu.get('packet_format') != 'SPIMU003':
            raise ValueError('verified ICM45686 SPIMU003 mounting required')
        if not imu.get('attached_link') or not imu.get('socket_path'):
            raise ValueError('IMU attached link and IPC socket required')
        rotation=np.asarray(imu.get('sensor_to_body_rotation'),dtype=float)
        if rotation.shape!=(3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-9) or not np.isclose(np.linalg.det(rotation),1):
            raise ValueError('IMU sensor_to_body_rotation must be a proper rotation')
        if number(imu.get('max_age_ms'),'IMU max age',positive=True)>30:
            raise ValueError('ICM maximum sample age may not exceed 30 ms')
        if number(data.get('controller',{}).get('native_state_max_age_ms',60),'native state max age',positive=True)>60:
            raise ValueError('native aggregate state maximum age may not exceed 60 ms')
        if data.get('controller',{}).get('native_target_timeout_ms',100)!=100:
            raise ValueError('native target watchdog is fixed at 100 ms')
        for rate in ('joint_execution_hz','native_state_hz'):
            value=data.get('controller',{}).get(rate,500)
            if type(value) is not int or value!=500:
                raise ValueError(f'{rate}: all-body hardware rate must be500Hz')
        records=data.get('motor_map',{})
        if len(records)!=31:
            raise ValueError('31 physical motor records required')
        identities=set()
        for name,motor in records.items():
            for field in ('can_channel','can_id','master_id'):
                if type(motor.get(field)) is not int:
                    raise ValueError(f'{name}.{field}: measured integer required')
            channel,can_id,master=motor['can_channel'],motor['can_id'],motor['master_id']
            if not 0<=channel<4 or not 1<=can_id<=8 or master!=can_id+0x10:
                raise ValueError(f'{name}: invalid CAN endpoint')
            if (channel,master) in identities:
                raise ValueError('duplicate per-bus Master_ID')
            identities.add((channel,master))
            if motor.get('encoder_sign') not in (-1,1) or type(motor.get('encoder_sign')) is not int:
                raise ValueError(f'{name}: measured encoder sign required')
            number(motor.get('motor_zero_rad'),f'{name}.motor_zero_rad')
            ranges=motor.get('mit_ranges',{})
            if ranges.get('source') != 'motor_register_readback':
                raise ValueError(f'{name}: MIT ranges must come from per-drive register readback')
            for field in ('position_rad','velocity_rad_s','kp','kd','torque_nm'):
                pair(ranges.get(field),f'{name}.MIT {field}')
            if ranges['kp'] != [0.,500.] or ranges['kd'] != [0.,5.]:
                raise ValueError(f'{name}: native MIT codec requires fixed Kp [0,500], Kd [0,5] encoding ranges')
            if not isinstance(motor.get('firmware'),str) or not motor['firmware']:
                raise ValueError(f'{name}: measured firmware identity required')
            soft=pair(motor.get('soft_limit_rad'),f'{name}.soft limit')
            hard=pair(motor.get('hard_limit_rad'),f'{name}.hard limit')
            if not hard[0]<soft[0]<soft[1]<hard[1]:
                raise ValueError(f'{name}: soft limits outside hard limits')
            for field in ('rated_torque_nm','peak_torque_nm','max_speed_rad_s','rated_current_a','peak_current_a','temperature_limit_c','commissioning_torque_cap_nm','feedback_torque_cap_nm'):
                number(motor.get(field),f'{name}.{field}',positive=True)
            cap=min(motor['peak_torque_nm'],min(map(abs,ranges['torque_nm'])))
            if not 0<motor['commissioning_torque_cap_nm']<=motor['feedback_torque_cap_nm']<=cap:
                raise ValueError(f'{name}: motor cap exceeds protocol/mechanical bound')
        names=tuple(sorted(records,key=lambda n:(interfaces[records[n]['can_channel']],records[n]['can_id'])))
        if len({(records[n]['can_channel'],records[n]['can_id']) for n in names})!=31:
            raise ValueError('duplicate per-bus CAN_ID')
        matrix=np.zeros((31,31));offset=np.zeros(31);covered=set()
        indices={name:i for i,name in enumerate(names)}
        for name,motor in records.items():
            joint=motor.get('policy_joint')
            if joint is None:
                continue
            if joint not in JOINT_NAMES or joint in covered:
                raise ValueError('invalid/duplicate direct joint')
            sign=motor.get('policy_to_motor_sign')
            if type(sign) is not int or sign not in (-1,1):
                raise ValueError(f'{name}: measured policy_to_motor_sign required')
            ratio=number(motor.get('reduction_ratio'),name,positive=True)*number(motor.get('linkage_ratio'),name,positive=True)
            matrix[indices[name],JOINT_NAMES.index(joint)]=sign*motor['encoder_sign']*ratio
            offset[indices[name]]=motor['motor_zero_rad'];covered.add(joint)
        differential_names={'left_ankle','right_ankle','head'}
        if set(data.get('differentials',{}))!=differential_names:
            raise ValueError('left/right ankle and head differential calibrations required')
        for label,p in data['differentials'].items():
            if p.get('calibrated') is not True or not p.get('source') or 'PLACEHOLDER' in p.get('source',''):
                raise ValueError(f'{label}: measured differential calibration required')
            joints=p.get('joint_order',[]);motors=p.get('motor_names',[])
            expected={'left_ankle':['left_ankle_pitch_joint','left_ankle_roll_joint'],'right_ankle':['right_ankle_pitch_joint','right_ankle_roll_joint'],'head':['head_pitch_joint','head_roll_joint']}[label]
            if joints!=expected or len(motors)!=2 or len(set(motors))!=2 or any(x not in records for x in motors):
                raise ValueError(f'{label}: differential ordering mismatch')
            a=np.asarray(p.get('joint_to_motor_matrix'),dtype=float);zero=np.asarray(p.get('motor_zero_rad'),dtype=float)
            if a.shape!=(2,2) or zero.shape!=(2,) or not np.isfinite(a).all() or not np.isfinite(zero).all() or abs(np.linalg.det(a))<1e-8 or np.linalg.cond(a)>100:
                raise ValueError(f'{label}: invalid calibration matrix/offset')
            for i,motor in enumerate(motors):
                row=indices[motor];record=records[motor]
                matrix[row,[JOINT_NAMES.index(j) for j in joints]]=record['encoder_sign']*a[i]
                offset[row]=record['motor_zero_rad']+record['encoder_sign']*zero[i]
            if any(j in covered for j in joints):
                raise ValueError('differential/direct joint overlap')
            covered.update(joints)
        if covered!=set(JOINT_NAMES) or np.linalg.matrix_rank(matrix)!=31 or np.linalg.cond(matrix)>100:
            raise ValueError('motor map must cover all 31 joints with an invertible transmission')
        limits=data.get('joint_limits',{})
        if set(limits)!=set(JOINT_NAMES):
            raise ValueError('joint_limits must contain every SDK joint')
        for joint,lim in limits.items():
            lo,hi=pair(lim.get('position_rad'),joint)
            for field in ('velocity_rad_s','kp','kd','effort_nm'):
                number(lim.get(field),f'{joint}.{field}',positive=True)
            if lim['kd']>5:
                raise ValueError('native joint safety Kd guard requires <=5')
        return cls(data,names,matrix,offset,np.linalg.inv(matrix))

    @property
    def motor_order_hash(self):
        return ordered_name_hash(self.motor_names)

    def joint_state(self,position,velocity):
        return self.inverse@(np.asarray(position)-self.offset), self.inverse@np.asarray(velocity)

    def motor_state(self,position,velocity):
        return self.matrix@np.asarray(position)+self.offset,self.matrix@np.asarray(velocity)

    def joint_to_motor_torque(self,torque):
        return self.inverse.T@np.asarray(torque)

    def generate(self,directory):
        self.can_wire_budget()
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        interfaces=self.data['can_adapter']['interfaces'];ankles=set()
        for label in ('left_ankle','right_ankle'):
            ankles.update(self.data['differentials'][label]['motor_names'])
        rows=[]
        for name in self.motor_names:
            m=self.data['motor_map'][name];r=m['mit_ranges'];temperature=m['temperature_limit_c']
            row=dict(zip(NATIVE_FIELDS,[name,interfaces[m['can_channel']],m['can_id'],m['master_id'],*r['position_rad'],*r['velocity_rad_s'],*r['torque_nm'],*m['soft_limit_rad'],*m['hard_limit_rad'],m['max_speed_rad_s'],m['rated_torque_nm'],m['peak_torque_nm'],m['commissioning_torque_cap_nm'],m['feedback_torque_cap_nm'],temperature,temperature,min(.9*temperature,90),min(.9*temperature,80),2.0,500]))
            rows.append(row)
        def write(name,fields,rows):
            path=directory/name
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(rows)
            return path
        motors=write('motors.tsv',NATIVE_FIELDS,rows)
        fields=('kind','name','offset_or_effort')+tuple(f'c{i}' for i in range(31));rows=[]
        for i,name in enumerate(self.motor_names):
            rows.append(dict(zip(fields,('motor',name,self.offset[i],*self.matrix[i]))))
        for i,name in enumerate(JOINT_NAMES):
            rows.append(dict(zip(fields,('joint',name,self.data['joint_limits'][name]['effort_nm'],*self.inverse[i]))))
        kinematics=write('kinematics.tsv',fields,rows)
        rows=[]
        for name in JOINT_NAMES:
            lim=self.data['joint_limits'][name]
            rows.append(dict(zip(JOINT_FIELDS,(name,*lim['position_rad'],lim['velocity_rad_s'],lim['kp'],lim['kd'],lim['effort_nm']))))
        safety=write('joint-safety.tsv',JOINT_FIELDS,rows)
        return {'motors':str(motors),'kinematics':str(kinematics),'joint_safety':str(safety)}

    def can_wire_budget(self):
        """Conservative FD+BRS model; the native driver checks actual link data."""
        adapter=self.data['can_adapter']
        arbitration=number(adapter.get('arbitration_bitrate',1_000_000),'CAN arbitration bitrate',positive=True)
        data=number(adapter.get('data_bitrate',5_000_000),'CAN data bitrate',positive=True)
        if data<arbitration:
            raise ValueError('CAN data bitrate must not be below arbitration bitrate')
        frame=64/arbitration+128/data
        result=[]
        for channel,interface in enumerate(adapter['interfaces']):
            motors=sum(m['can_channel']==channel for m in self.data['motor_map'].values())
            utilization=motors*500*2*frame
            slot=((motors+3)//4)*2*frame
            if not 0<motors<=8 or utilization>.8 or slot>.0004:
                raise ValueError(f'{interface}: all-body500Hz CAN FD budget exceeds80%')
            result.append(dict(interface=interface,motors=motors,execution_hz=500,
                frames_per_second=motors*1000,utilization=utilization,
                worst_slot_seconds=slot,arbitration_bitrate=arbitration,data_bitrate=data,
                hardware_measured=False))
        return result

    def launch_command(self,directory,*,actuate=False,acknowledge=False):
        if actuate and self.data.get('synthetic'):
            raise ValueError('synthetic CPU fixture must never launch actuating hardware')
        if actuate and not acknowledge:
            raise ValueError('actuation requires explicit operator acknowledgement')
        files=self.generate(directory);native=self.data['native'];duration=number(native['duration_s'],'duration',positive=True)
        if duration>10:
            raise ValueError('SDK initial native deployments are bounded to 10 seconds; no acknowledgement bypass')
        ack='ENABLE_NATIVE_PROTECTED_POLICY_ACTUATION' if actuate else 'ZERO_GAIN_NATIVE_SHADOW'
        cmd=[native['binary'],'--config',files['motors'],'--output',str(Path(directory)/'native-report.json'),'--duration',str(duration),'--cpu',str(native['cpu']),'--acknowledge-hardware-tx',ack,'--all-motors-disabled-confirmed','--ipc-socket',native['socket_path'],'--kinematics-config',files['kinematics'],'--joint-safety-config',files['joint_safety'],'--policy-joint-hash',str(JOINT_ORDER_HASH)]
        if actuate:cmd.append('--policy-actuation')
        if native.get('realtime_priority',0):cmd+=['--realtime-priority',str(native['realtime_priority'])]
        return cmd
