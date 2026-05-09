from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "0.1"
CAMERA_ID = "wrist"
INTRINSICS_ID = "wrist_1920x1080_v1"
CANONICAL_WIDTH = 1920
CANONICAL_HEIGHT = 1080

CAMERA_MATRIX = np.array(
    [
        [693.35550704, 0.0, 981.92182023],
        [0.0, 692.62105461, 496.33606080],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)
DISTORTION_COEFFICIENTS = np.array(
    [0.06046540, -0.07201475, -0.00076385, 0.00059471, -0.00305710],
    dtype=float,
)

