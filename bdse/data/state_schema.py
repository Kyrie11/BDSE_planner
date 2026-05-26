from __future__ import annotations

"""Canonical numeric state layouts used by BDSE.

The project stores ego/candidate states as [x, y, yaw, speed, time] and
tracked-object states as [x, y, yaw, speed, heading_rate, vx, vy, length,
width, type_id] when those fields are available.  These constants remove the
magic indices that otherwise make teacher and runtime safety code fragile.
"""

EGO_X = 0
EGO_Y = 1
EGO_YAW = 2
EGO_SPEED = 3
EGO_TIME = 4
EGO_DIM = 5

AGENT_X = 0
AGENT_Y = 1
AGENT_YAW = 2
AGENT_SPEED = 3
AGENT_HEADING_RATE = 4
AGENT_VX = 5
AGENT_VY = 6
AGENT_LENGTH = 7
AGENT_WIDTH = 8
AGENT_TYPE_ID = 9
AGENT_DIM = 10

DEFAULT_VEHICLE_LENGTH_M = 4.8
DEFAULT_VEHICLE_WIDTH_M = 2.0


def agent_length(state) -> float:
    return float(state[AGENT_LENGTH]) if len(state) > AGENT_LENGTH and float(state[AGENT_LENGTH]) > 0 else DEFAULT_VEHICLE_LENGTH_M


def agent_width(state) -> float:
    return float(state[AGENT_WIDTH]) if len(state) > AGENT_WIDTH and float(state[AGENT_WIDTH]) > 0 else DEFAULT_VEHICLE_WIDTH_M


def agent_vx(state) -> float:
    import numpy as np
    return float(state[AGENT_VX]) if len(state) > AGENT_VX else float(state[AGENT_SPEED]) * float(np.cos(float(state[AGENT_YAW])))


def agent_vy(state) -> float:
    import numpy as np
    return float(state[AGENT_VY]) if len(state) > AGENT_VY else float(state[AGENT_SPEED]) * float(np.sin(float(state[AGENT_YAW])))
