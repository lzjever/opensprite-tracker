"""Review counterexamples: stop ordering and fixed nonzero actuator delay."""
import json

import pytest

from opensprite_sdk.backends.mujoco import MujocoBackend, load_profile
from opensprite_sdk.client import joint_command


def command(backend, sequence, *, mode=1, controller="review-controller", torque=0.):
    state=backend.read()
    result=joint_command(state, backend.profile["default_q"], [0.]*31, [0.]*31)
    result.seq=sequence
    result.session_id=controller
    for joint in result.joint_cmd:
        joint.mode=mode
        joint.tau=torque
    return result


def test_accepted_stop_advances_sequence_high_water_across_enable():
    with MujocoBackend() as robot:
        robot.enable()
        robot.write(command(robot, 1))
        robot.write(command(robot, 100, mode=0))
        assert not robot.enabled
        robot.enable()
        with pytest.raises(ValueError, match="replayed"):
            robot.write(command(robot, 2))
        with pytest.raises(ValueError, match="replayed"):
            robot.write(command(robot, 100))
        robot.write(command(robot, 101))
        assert robot.enabled


def test_first_command_can_be_stop_without_surrendering_control_identity():
    with MujocoBackend() as robot:
        robot.write(command(robot, 100, mode=0))
        robot.enable()
        with pytest.raises(ValueError, match="owns"):
            robot.write(command(robot, 101, controller="another-controller"))
        with pytest.raises(ValueError, match="replayed"):
            robot.write(command(robot, 99))
        robot.reset()
        robot.enable()
        robot.write(command(robot, 1, controller="another-controller"))


@pytest.mark.parametrize("delay_ticks", [1, 3])
def test_fixed_positive_delay_is_observed_before_and_after_reset(tmp_path, delay_ticks):
    profile=load_profile()
    profile["command_delay"].update(min_ticks=delay_ticks, max_ticks=delay_ticks,
                                    hold_probability=1.)
    path=tmp_path/"delay.json"
    path.write_text(json.dumps(profile))
    with MujocoBackend(profile=path) as robot:
        for _ in range(2):
            robot.reset()
            robot.enable()
            for step in range(8):
                robot.write(command(robot, step+1, torque=.1*(step+1)))
                robot.step()
                actual=robot.debug_truth()
                assert actual["command_lag_ticks"]==delay_ticks
                # At startup unavailable history is backfilled by the first
                # command, then distinct commands reveal the real tick lag.
                expected=.1*(max(0, step-delay_ticks)+1)
                assert robot._last_applied_command[0,4]==pytest.approx(expected)


def test_default_minimum_zero_profile_retains_its_seeded_delay_trace():
    traces=[]
    for _ in range(2):
        with MujocoBackend(seed=42) as robot:
            robot.enable()
            trace=[]
            for step in range(30):
                robot.write(command(robot, step+1, torque=.01*step))
                robot.step()
                trace.append((robot._lag, robot._last_applied_command[0,4]))
            traces.append(trace)
    assert traces[0]==traces[1]
    assert all(lag in (0,1) for lag,_ in traces[0])
