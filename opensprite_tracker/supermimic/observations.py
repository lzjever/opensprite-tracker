"""Actor inputs constructed from calibrated onboard sensor measurements."""

import torch


SENSOR_OBS_DIM = 99


def sensor_observation(q_sensor, qd, gyro_site, gravity_site, a_prev, default_q):
    """Pack (..., 99) observations in the fixed 31-joint policy order.

    q_sensor/default_q use radians, qd/gyro_site use radians per second.
    gyro_site and gravity_site share the configured IMU mounting frame.
    gravity_site is an estimated unit downward gravity vector; it is NOT a
    normalized instantaneous accelerometer sample during dynamic motion.
    a_prev is the previous sent policy action, before hidden actuator delay.
    Encoder calibration is already represented in q_sensor; no simulated
    context, global pose, reference pose, or extra bias is accepted here.
    Leading batch/history dimensions are preserved. Noise is applied outside.
    """
    shape = q_sensor.shape[:-1]
    for name, value, width in (
        ("q_sensor", q_sensor, 31), ("qd", qd, 31),
        ("gyro_site", gyro_site, 3), ("gravity_site", gravity_site, 3),
        ("a_prev", a_prev, 31),
    ):
        if value.shape != shape + (width,):
            raise ValueError(f"{name} must have shape {shape + (width,)}")
    if default_q.shape != (31,):
        raise ValueError("default_q must have shape (31,)")
    return torch.cat((q_sensor - default_q, qd, gyro_site, gravity_site, a_prev), -1)
