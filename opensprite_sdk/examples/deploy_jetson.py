#!/usr/bin/env python3
"""Supervise a finite Jetson policy-shadow or explicitly qualified hardware run.

Importing this module never starts a process or accesses hardware. The IMU
service must already be calibrated and running; no SPI/USB transport is selected.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

import numpy as np

from opensprite_sdk.backends.hardware import HardwareBackend
from opensprite_sdk.backends.hardware_config import HardwareConfig
from opensprite_sdk.backends.native_ipc import decode_icm
from opensprite_sdk.policy import PolicyBundle

SDK_ROOT=Path(__file__).resolve().parents[1]


def probe_imu(config, *, now_ns=time.monotonic_ns):
    """Check the explicitly configured existing IMU service without CAN access."""
    imu=config.data['imu']
    with tempfile.TemporaryDirectory(prefix='opensprite-sdk-deploy-probe-') as directory:
        with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as peer:
            peer.bind(str(Path(directory)/'rx.sock'));peer.settimeout(.1)
            peer.connect(imu['socket_path'])
            if peer.send(b'GETIMU01')!=8:raise RuntimeError('short IMU probe request')
            sample=decode_icm(peer.recv(193),now=now_ns(),rotation=imu['sensor_to_body_rotation'],
                              max_age_ns=int(imu['max_age_ms']*1e6))
    return {'sequence':sample.sequence,'sample_ns':sample.sample_ns,
            'attached_link':imu['attached_link'],'ready':True}


def _prewarm(bundle):
    c=bundle.contract
    result=bundle.session.run(['action'],{
        'observation_history':np.zeros((1,c['hist'],c['obs_dim']),np.float32),
        'goal':bundle.reference['goals'][:1].astype(np.float32)})[0]
    if result.shape!=(1,31) or not np.isfinite(result).all():
        raise ValueError('policy warmup returned invalid action')
    bundle.reset()


def _check_contract(bundle,config,actuate):
    c=bundle.contract
    profile=config.data.get('profile_id','physical_v5')
    if c['profile_id']!=profile:
        raise ValueError('policy/hardware profile mismatch; simulation profiles cannot be relabeled for deployment')
    if actuate and c.get('hardware_qualified') is not True:
        raise ValueError('actuation requires hardware_qualified is True in the exact bundle contract')
    imu=config.data['imu']
    if c['imu_link']!=imu['attached_link']:
        remap=c.get('imu_remap')
        if not remap or remap.get('source_link')!=imu['attached_link'] or remap.get('target_link')!=c['imu_link'] or not remap.get('chain'):
            raise ValueError('policy/IMU attachment mismatch without explicit kinematic remap')
    duration=float(config.data['native']['duration_s'])
    if not math.isfinite(duration) or not 0<duration<=10:
        raise ValueError('initial native deployment duration must be in (0,10] seconds')
    if c['frames']-1<math.ceil(duration*50)+2:
        raise ValueError('reference is too short for the complete finite native run; no looping/holding allowed')
    return profile,duration


def _load_report(path,*,deadline,clock,sleep):
    last=None
    while True:
        try:
            with Path(path).open() as stream:
                report=json.load(stream)
            if not isinstance(report,dict):raise ValueError('native report must be an object')
            return report
        except (FileNotFoundError,json.JSONDecodeError) as error:
            last=error
            if clock()>=deadline:raise RuntimeError(f'native completion report missing/incomplete: {path}') from last
            sleep(min(.01,max(0.,deadline-clock())))


def _check_native_report(report,*,actuate,motor_names,exit_code):
    if exit_code!=0:
        raise RuntimeError(f'native process exited with status {exit_code}')
    if report.get('passed') is not True:
        raise RuntimeError('native completion report did not pass')
    mode=report.get('mode','')
    if actuate:
        if mode!='native_suspended_protected_policy_actuation':
            raise RuntimeError('native report mode does not match actuating deployment')
        motors=report.get('motors',{})
        if set(motors)!=set(motor_names) or any(row.get('final_disabled') is not True for row in motors.values()):
            raise RuntimeError('native report lacks final-disabled confirmation for every motor')
    else:
        if mode!='native_mixed_rate_zero_gain_can_shadow':
            raise RuntimeError('native report mode does not match zero-gain shadow')
        for field in ('nonzero_gain_or_torque_tx_attempts','automatic_enable_attempts','automatic_mode_switch_attempts'):
            if type(report.get(field)) is not int or report[field]!=0:
                raise RuntimeError(f'shadow report has nonzero/missing {field}')
        if report.get('policy_ipc_passed') is not True:
            raise RuntimeError('native policy IPC shadow did not pass')


def run(args,*,bundle_factory=PolicyBundle,config_loader=HardwareConfig.load,
        backend_factory=HardwareBackend,process_factory=subprocess.Popen,
        imu_probe=probe_imu,clock=time.monotonic,sleep=time.sleep):
    """Inject factories/clocks for offline tests; production defaults use real IPC.

    Validation/warmup/IMU readiness finish before the first native process starts.
    The policy ends only when its native supervisor ends, never at a shorter
    application duration. An EOF is merely a terminal candidate until the exit
    status and on-disk report have both been checked.
    """
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('choose a new deployment output directory')
    output.mkdir(parents=True)
    summary={'passed':False,'actuation_requested':bool(args.actuate),
             'native_spawned':False,'native_exit_code':None,'native_report':None,
             'app_error':None,'commands':0,'terminal_eof':False,'forced_termination':False,
             'hardware_validated_by_this_development_session':False}
    backend=None;process=None;error=None;logs=[];report_path=output/'native/native-report.json'
    began=clock()
    try:
        if args.actuate and args.acknowledge_actuation is not True:
            raise ValueError('actuation requires both --actuate and --acknowledge-actuation')
        bundle=bundle_factory(args.bundle)
        config=config_loader(args.hardware)
        profile,duration=_check_contract(bundle,config,args.actuate)
        summary.update(profile_id=profile,native_duration_s=duration)
        _prewarm(bundle)
        # Match the native binary path convention to this independent project.
        native=config.data['native']
        binary=Path(native['binary'])
        if not binary.is_absolute():binary=SDK_ROOT/binary
        if not binary.is_file() or not os.access(binary,os.X_OK):
            raise ValueError('native CAN executable is missing; build tools/build_native.sh first')
        summary['imu_preflight']=imu_probe(config,now_ns=lambda:int(clock()*1e9))
        if summary['imu_preflight'].get('ready') is not True:
            raise RuntimeError('configured IMU service is not ready')
        argv=config.launch_command(output/'native',actuate=args.actuate,
                                   acknowledge=args.acknowledge_actuation)
        argv[0]=str(binary.resolve())
        summary['native_argv']=argv
        (output/'launch.json').write_text(json.dumps({'native_argv':argv},indent=2)+'\n')
        for filename in ('native-stdout.log','native-stderr.log'):
            logs.append((output/filename).open('w'))
        process=process_factory(argv,cwd=str(SDK_ROOT),stdout=logs[0],stderr=logs[1])
        summary['native_spawned']=True
        backend=backend_factory(config,profile_id=profile,allow_actuation=args.actuate,
                                clock=lambda:int(clock()*1e9))
        # Only retry a connection that has not succeeded. Never reconnect a live
        # policy session, and never retry an invalid state/IMU packet.
        connection_deadline=clock()+5.
        while True:
            try:
                backend.start();break
            except (FileNotFoundError,ConnectionRefusedError):
                backend.close()
                if process.poll() is not None or clock()>=connection_deadline:raise
                sleep(.01)
                backend=backend_factory(config,profile_id=profile,allow_actuation=args.actuate,
                                        clock=lambda:int(clock()*1e9))
        state=backend.read()
        if args.actuate:
            backend.enable()
            # Native startup collection now pins its first zero-gain packet;
            # this additional state read also keeps the first policy input fresh.
            state=backend.read()
        frame=0;next_tick=clock();hard_deadline=clock()+duration+5.
        while True:
            if process.poll() is not None:break
            if clock()>hard_deadline:raise TimeoutError('native controller exceeded its finite schedule')
            if frame>=bundle.contract['frames']-1:
                raise RuntimeError('reference ended before native completion; refusing wrap/hold')
            command=bundle.command(state,frame,now_ns=int(clock()*1e9))
            if args.actuate:backend.write(command)
            else:backend.preview(command)
            frame+=1;summary['commands']=frame
            next_tick+=.02
            remaining=next_tick-clock()
            if remaining<-.04:raise RuntimeError('policy missed two 20 ms deadlines')
            if remaining>0:sleep(remaining)
            if process.poll() is not None:break
            try:
                state=backend.read()
            except EOFError:
                summary['terminal_eof']=True;break
        # Do not proactively disconnect while native verifies disabled feedback.
        # EOF means the native peer has already closed; it is still not success
        # until both the process exit and its complete report have been checked.
        summary['native_exit_code']=process.wait(timeout=2.)
        report=_load_report(report_path,deadline=clock()+1.,clock=clock,sleep=sleep)
        summary['native_report']=report
        _check_native_report(report,actuate=args.actuate,motor_names=config.motor_names,
                             exit_code=summary['native_exit_code'])
        if frame<1:raise RuntimeError('native exited before any policy command was sent')
        summary['passed']=True
    except BaseException as caught:
        error=caught;summary['app_error']=f'{type(caught).__name__}: {caught}'
    finally:
        # On application failure, disconnect first so the native watchdog can
        # disable. On success, this happens only AFTER its report was verified.
        try:
            if backend is not None:backend.close()
            if process is not None and summary['native_exit_code'] is None:
                try:summary['native_exit_code']=process.wait(timeout=2.)
                except subprocess.TimeoutExpired:
                    summary['forced_termination']=True;process.terminate()
                    try:summary['native_exit_code']=process.wait(timeout=2.)
                    except subprocess.TimeoutExpired:
                        process.kill();summary['native_exit_code']=process.wait(timeout=2.)
        except BaseException as cleanup_error:
            summary['cleanup_error']=f'{type(cleanup_error).__name__}: {cleanup_error}'
            summary['passed']=False
            if error is None:
                error=cleanup_error;summary['app_error']=summary['cleanup_error']
        if summary['native_report'] is None and report_path.is_file():
            try:summary['native_report']=json.loads(report_path.read_text())
            except (OSError,json.JSONDecodeError):pass
        for stream in logs:stream.close()
        summary['wall_seconds']=clock()-began
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if error is not None:raise error
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--hardware',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--actuate',action='store_true')
    parser.add_argument('--acknowledge-actuation',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args),indent=2))


if __name__=='__main__':main()
