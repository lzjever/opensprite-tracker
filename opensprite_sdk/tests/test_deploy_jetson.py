"""Finite native-supervisor integration with injected IPC/process clocks only."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from opensprite_sdk.constants import JOINT_NAMES
from opensprite_sdk.messages import LowCmd_,LowState_,JointState,IMUState

SPEC=importlib.util.spec_from_file_location('sdk_deploy_jetson_example',Path(__file__).resolve().parents[1]/'examples/deploy_jetson.py')
deploy=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(deploy)


class Clock:
    def __init__(self):self.now=10.;self.hooks=[]
    def __call__(self):return self.now
    def sleep(self,value):
        self.now+=value
        for hook in self.hooks:hook(self.now)


def scenario(tmp_path,*,actuate=False,qualified=True,profile='physical_v5',report_kind='pass',delayed_report=False,shutdown_silence=0.):
    events=[];clock=Clock();targets=[];spawned=[]
    binary=tmp_path/'native-test';binary.write_text('# CPU fixture only');binary.chmod(0o700)
    native_duration=.06
    report={'passed':report_kind!='false','mode':'native_suspended_protected_policy_actuation' if actuate else 'native_mixed_rate_zero_gain_can_shadow'}
    if actuate:report['motors']={name:{'final_disabled':True} for name in JOINT_NAMES}
    else:report.update(nonzero_gain_or_torque_tx_attempts=0,automatic_enable_attempts=0,
                       automatic_mode_switch_attempts=0,policy_ipc_passed=True)
    class Bundle:
        def __init__(self,path):
            events.append('bundle_load');self.contract=dict(profile_id=profile,hardware_qualified=qualified,
                imu_link='pelvis_link',frames=100,hist=10,obs_dim=99)
            self.reference={'goals':np.zeros((100,770),np.float32)};self.session=self
        def run(self,outputs,inputs):events.append('warmup');return [np.zeros((1,31),np.float32)]
        def reset(self):events.append('bundle_reset')
        def command(self,state,frame,now_ns):
            command=LowCmd_(profile_id=profile,seq=frame+1,stamp_ns=state.stamp_ns,
                session_id='policy-controller',source_state_seq=state.seq,source_state_session_id=state.session_id)
            for target in command.joint_cmd:target.mode=1;target.q=.1;target.kp=2;target.kd=.2;target.tau=.3
            return command
    class Config:
        motor_names=JOINT_NAMES
        def __init__(self):
            self.data={'profile_id':'physical_v5','native':{'binary':str(binary),'duration_s':native_duration},
                       'imu':{'attached_link':'pelvis_link'}}
        def launch_command(self,output,**kwargs):
            Path(output).mkdir(parents=True);return [str(binary),'--output',str(Path(output)/'native-report.json')]
    class Backend:
        def __init__(self,*args,**kwargs):self.reads=0
        def start(self):events.append('backend_start');return self
        def read(self):
            self.reads+=1
            if self.reads>3:
                clock.sleep(shutdown_silence)
                events.append('native_eof');raise EOFError('own native finite completion')
            stamp=int(clock()*1e9)
            return LowState_(profile_id='physical_v5',seq=self.reads,stamp_ns=stamp,session_id='robot-session',clock_domain='monotonic',
                joint_state=[JointState(valid=True,sample_ns=stamp,sample_time_kind='host_receive') for _ in range(31)],
                imu_state=IMUState(valid=True,sample_ns=stamp,sample_time_kind='host_read'),status='ready' if actuate else 'shadow')
        def enable(self):events.append('enable')
        def write(self,command):events.append('write');targets.append(command)
        def preview(self,command):events.append('preview');targets.append(command)
        def close(self):events.append('backend_close')
    class Process:
        def __init__(self,argv,**kwargs):
            events.append('spawn');spawned.append(self);self.code=None;self.report_path=Path(argv[argv.index('--output')+1])
        def poll(self):return self.code
        def wait(self,timeout):
            events.append('native_wait')
            self.code=0
            if report_kind!='missing':
                if delayed_report:
                    when=clock()+.05
                    def publish(now):
                        if now>=when and not self.report_path.exists():
                            events.append('report_write');self.report_path.write_text(json.dumps(report))
                    clock.hooks.append(publish)
                else:events.append('report_write');self.report_path.write_text(json.dumps(report))
            return self.code
        def terminate(self):events.append('terminate');self.code=2
        def kill(self):events.append('kill');self.code=-9
    def load_config(_):events.append('config_check');return Config()
    def imu_probe(config,**kwargs):events.append('imu_probe');return {'ready':True}
    args=SimpleNamespace(output=tmp_path/'out',bundle=tmp_path/'bundle',hardware=tmp_path/'hardware.json',
                         actuate=actuate,acknowledge_actuation=actuate)
    dependencies=dict(bundle_factory=Bundle,config_loader=load_config,backend_factory=Backend,
                      process_factory=Process,imu_probe=imu_probe,clock=clock,sleep=clock.sleep)
    return args,dependencies,events,targets,spawned,report


def test_shadow_waits_for_native_completion_and_report_before_close(tmp_path):
    args,deps,events,targets,spawned,_=scenario(tmp_path)
    result=deploy.run(args,**deps)
    assert result['passed'] and result['terminal_eof']
    assert len(targets)==3 and all(command.joint_cmd[0].kp==2 and command.joint_cmd[0].tau==.3 for command in targets)
    assert 'enable' not in events and 'write' not in events
    assert events.index('warmup')<events.index('imu_probe')<events.index('spawn')
    assert events.index('native_wait')<events.index('report_write')<events.index('backend_close')
    assert json.loads((args.output/'summary.json').read_text())['passed'] is True


def test_eof_waits_for_report_written_after_process_exit(tmp_path):
    args,deps,events,*_=scenario(tmp_path,delayed_report=True)
    assert deploy.run(args,**deps)['passed']
    assert events.index('native_wait')<events.index('report_write')<events.index('backend_close')


@pytest.mark.parametrize('kwargs,reason',[
    ({'actuate':True,'qualified':False},'hardware_qualified'),
    ({'profile':'legacy_v4'},'profile mismatch'),
])
def test_qualification_or_profile_failure_never_spawns(tmp_path,kwargs,reason):
    args,deps,events,targets,spawned,_=scenario(tmp_path,**kwargs)
    with pytest.raises(ValueError,match=reason):deploy.run(args,**deps)
    assert not spawned and not targets and 'imu_probe' not in events
    saved=json.loads((args.output/'summary.json').read_text())
    assert not saved['passed'] and not saved['native_spawned'] and saved['app_error']


def test_invalid_configuration_and_missing_ack_never_spawn(tmp_path):
    args,deps,events,targets,spawned,_=scenario(tmp_path,actuate=True)
    args.acknowledge_actuation=False
    with pytest.raises(ValueError,match='both'):deploy.run(args,**deps)
    assert not spawned
    args.output=tmp_path/'bad-config';args.acknowledge_actuation=True
    def missing(_):raise ValueError('missing measured motor zero')
    deps['config_loader']=missing
    with pytest.raises(ValueError,match='measured motor'):deploy.run(args,**deps)
    assert not spawned


@pytest.mark.parametrize('report_kind,reason',[('missing','report missing'),('false','did not pass')])
def test_native_eof_without_positive_report_is_failure(tmp_path,report_kind,reason):
    args,deps,events,targets,spawned,_=scenario(tmp_path,report_kind=report_kind)
    with pytest.raises(RuntimeError,match=reason):deploy.run(args,**deps)
    saved=json.loads((args.output/'summary.json').read_text())
    assert not saved['passed'] and saved['terminal_eof'] and saved['app_error']


def test_actuation_requires_every_motor_final_disabled(tmp_path):
    args,deps,events,targets,spawned,report=scenario(tmp_path,actuate=True)
    report['motors'][JOINT_NAMES[-1]]['final_disabled']=False
    with pytest.raises(RuntimeError,match='every motor'):deploy.run(args,**deps)
    assert events.count('enable')==1 and 'write' in events
    assert events.index('native_wait')<events.index('backend_close')


def test_native_finite_schedule_not_short_application_loop(tmp_path):
    args,deps,events,targets,spawned,_=scenario(tmp_path,actuate=True)
    result=deploy.run(args,**deps)
    assert result['passed'] and events.count('enable')==1 and 'preview' not in events
    assert events.index('native_eof')<events.index('native_wait')<events.index('backend_close')


def test_native_250ms_final_disable_silence_completes_without_early_close(tmp_path):
    args,deps,events,targets,spawned,_=scenario(tmp_path,actuate=True,shutdown_silence=.25)
    result=deploy.run(args,**deps)
    assert result['passed'] and result['wall_seconds']>=.25
    assert events.index('native_eof')<events.index('native_wait')<events.index('report_write')<events.index('backend_close')
