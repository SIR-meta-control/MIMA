"""Shared constants for robot structure generation."""

from __future__ import annotations


VREQ_FORMAT = ["x", "y", "z", "load", "inspect", "pack"]

BAR_TO_V = {
    "4-bar": 0,
    "8-bar": 1,
    "6-bar": 2,
}
V_TO_BAR = {value: key for key, value in BAR_TO_V.items()}

BAR_ORDER = ["4-bar", "8-bar", "6-bar"]

LEG_MOUNT_NODE_INDICES = (2, 3, 6, 7)

# Structural message-passing topology recovered from the original CrimsonGNN
# implementation. Node indices are zero-based.
TYPE_GRAPH_EDGES = {
    "4-bar": ((1, 2), (2, 3), (3, 4)),
    "8-bar": (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 0),
    ),
    "6-bar": ((1, 2), (2, 3), (3, 4), (4, 6), (6, 7), (7, 1)),
}

LOAD_Q2_RANGE = (1.39, 1.75)
PACK_TARGET_SCALE = (0.32, 0.38, 0.40)
PACK_DEFAULT_SLACK = 0.05
PACK_LIMIT_SCALE = tuple(value + PACK_DEFAULT_SLACK for value in PACK_TARGET_SCALE)

EDGE_ANGLE_EQUALITY_INDICES = (1, 3, 5)

TYPE_COLLINEAR_EDGE_PAIRS = {
    "4-bar": ((1, 7), (2, 6), (3, 5)),
    "8-bar": ((0, 4),),
}

TYPE_PARALLEL_EDGE_GROUPS = {
    "6-bar": ((1, 2, 3, 5, 6, 7),),
}
