"""Small joint-space client using the SDK2 publisher/subscriber idiom."""
from .core.channel import ChannelPublisher, ChannelSubscriber
from .messages import LowCmd_, LowState_, JointCmd

LOWCMD = "rt/lowcmd"
LOWSTATE = "rt/lowstate"


def joint_command(state, positions, kp, kd, *, velocities=None, torques=None):
    positions, kp, kd = list(positions), list(kp), list(kd)
    velocities = [0.]*31 if velocities is None else list(velocities)
    torques = [0.]*31 if torques is None else list(torques)
    if any(len(v) != 31 for v in (positions, kp, kd, velocities, torques)):
        raise ValueError("commands require exactly 31 joints")
    result = LowCmd_(profile_id=state.profile_id, clock_domain=state.clock_domain,
                     stamp_ns=state.stamp_ns, source_state_seq=state.seq,
                     source_state_session_id=state.session_id)
    result.joint_cmd = [JointCmd(1, float(q), float(v), float(p), float(d), float(t))
                        for q, v, p, d, t in zip(positions, velocities, kp, kd, torques)]
    result.validate()
    return result


class LowLevelClient:
    def __init__(self, *, state_queue_len=1):
        self.publisher = ChannelPublisher(LOWCMD, LowCmd_)
        self.subscriber = ChannelSubscriber(LOWSTATE, LowState_)
        self.state_queue_len = state_queue_len

    def Init(self):
        self.subscriber.Init(queueLen=self.state_queue_len)
        try:
            self.publisher.Init()
        except BaseException:
            self.subscriber.Close()
            raise
        return self

    def Read(self, timeout=1.):
        message = self.subscriber.Read(timeout)
        if message is None:
            error = self.subscriber.last_error
            raise TimeoutError(f"no robot state: {error or 'timeout'}")
        return message

    def Write(self, command):
        return self.publisher.Write(command, timeout=.1)

    def Close(self):
        self.publisher.Close()
        self.subscriber.Close()

    def __enter__(self):
        return self.Init()

    def __exit__(self, *_):
        self.Close()
