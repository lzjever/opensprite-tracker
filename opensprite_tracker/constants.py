"""Sprite task termination and user-requested temporal actuator parameters."""

PELVIS_HEIGHT_THRESHOLD_M = 0.17
PELVIS_GRAVITY_SQUARED_THRESHOLD = 1.6

# Explicit user request: use the reference project's temporal actuator model.
# Command lag counts 5ms physics writes; refresh cadence is four writes (20ms).
COMMAND_DELAY_MIN_TICKS = 0
COMMAND_DELAY_MAX_TICKS = 1
COMMAND_DELAY_HOLD_PROBABILITY = 0.9
COMMAND_DELAY_REFRESH_TICKS = 4
TORQUE_TIME_CONSTANT_RANGE_S = (0.005, 0.010)
ACTUATOR_TEMPORAL_REVISION = (
    "user_requested_reference_delay_lag_sprite31_independent_v1"
)
