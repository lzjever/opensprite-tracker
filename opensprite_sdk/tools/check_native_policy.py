#!/usr/bin/env python3
"""Run the actor against a Python-free robot endpoint and externally score it."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from evaluate_native_trace import evaluate

ROOT=Path(__file__).resolve().parents[1]


def check(args):
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    work=output.parent/(output.stem+'-work-'+uuid.uuid4().hex[:8]);work.mkdir()
    bundle=args.bundle.resolve();scene=args.scene.resolve()
    contract=json.loads((bundle/'contract.json').read_text());initial=json.loads(scene.read_text())
    if initial['profile']['profile_id']!=contract['profile_id']:
        raise ValueError('scene/policy profile mismatch')
    frames=contract['frames']-1
    if args.duration is not None:
        if not 0<args.duration<float('inf'):raise ValueError('invalid duration')
        frames=min(frames,int(args.duration*50))
    if frames<1:raise ValueError('at least one policy transition required')
    robot=args.robot.resolve();report=dict(schema='opensprite-native-policy-acceptance-v1',passed=False,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),work_directory=str(work),
        video_recorded=False,hardware_accessed=False,native_robot_sha256=hashlib.sha256(robot.read_bytes()).hexdigest())
    server=client=None
    try:
        dependencies=subprocess.check_output(['ldd',str(robot)],text=True)
        if 'not found' in dependencies or 'libpython' in dependencies or 'onnxruntime' in dependencies:
            raise ValueError('native robot dependencies invalid')
        report['native_has_python_or_onnx_dependency']=False
        command=[str(robot),'--backend','mujoco','--config',str(scene),'--domain',str(args.domain),
            '--allow-enable','--duration',str(frames/50+30),'--report',str(work/'robot.json'),
            '--truth-output',str(work/'truth.jsonl')]
        policy=[sys.executable,str(ROOT/'examples/remote_policy.py'),'--bundle',str(bundle),
            '--domain',str(args.domain),'--output',str(work/'policy.json')]
        if args.duration is not None:policy+=['--duration',str(args.duration)]
        with (work/'robot.log').open('w') as rlog,(work/'policy.log').open('w') as plog:
            server=subprocess.Popen(command,stdout=rlog,stderr=subprocess.STDOUT)
            client=subprocess.Popen(policy,stdout=plog,stderr=subprocess.STDOUT)
            report['policy_exit_code']=client.wait(timeout=frames/50+25)
            if report['policy_exit_code']:raise RuntimeError(f'policy failed; see {work/"policy.log"}')
            report['robot_exit_code']=server.wait(timeout=5)
            if report['robot_exit_code']:raise RuntimeError(f'robot failed; see {work/"robot.log"}')
        runtime=json.loads((work/'robot.json').read_text());actor=json.loads((work/'policy.json').read_text())
        scoring=evaluate(work/'truth.jsonl',bundle/'reference.npz')
        report.update(runtime=runtime,policy=actor,evaluation=scoring)
        if (runtime['fault'] is not None or runtime['accepted_commands']!=frames
            or runtime['accepted_command_seq']!=frames or runtime['physics_hz']!=200
            or runtime['execution_hz']!=500 or runtime['truth_output_drops'] or runtime['truth_output_errors']
            or runtime['physics_steps']!=frames*4 or actor['transitions_sent']!=frames
            or not actor['disable_execution_verified'] or scoring['transitions_evaluated']!=frames
            or not frames*10<=runtime['execution_ticks']<=frames*10+2):
            raise AssertionError('native rates, finalcommand, phase or shutdown mismatch')
        report['passed']=True
        return report
    except BaseException as error:
        report['error']=str(error);raise
    finally:
        for process in (client,server):
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    try:process.wait(timeout=3)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
                except ProcessLookupError:pass
        report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',required=True,type=Path)
    parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--robot',type=Path,default=ROOT/'native-runtime/build/opensprite_robot')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--duration',type=float)
    parser.add_argument('--domain',type=int,default=44)
    print(json.dumps(check(parser.parse_args()),indent=2))
