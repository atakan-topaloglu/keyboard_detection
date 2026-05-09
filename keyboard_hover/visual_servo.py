from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ServoConfig:
    pixel_threshold: float = 15.0
    stable_frames_required: int = 3
    gain: float = 0.65
    max_iterations: int = 40
    noise_px: float = 0.0
    random_seed: int = 7


def simulate_visual_servo(
    target_px: tuple[float, float],
    initial_tip_px: tuple[float, float],
    config: ServoConfig | None = None,
) -> dict:
    cfg = config or ServoConfig()
    rng = np.random.default_rng(cfg.random_seed)
    target = np.asarray(target_px, dtype=float)
    tip = np.asarray(initial_tip_px, dtype=float)
    stable = 0
    trace = []
    for step in range(cfg.max_iterations):
        error = target - tip
        error_norm = float(np.linalg.norm(error))
        trace.append(
            {
                "step": step,
                "tip_px": [float(tip[0]), float(tip[1])],
                "error_px": [float(error[0]), float(error[1])],
                "error_norm_px": error_norm,
            }
        )
        if error_norm <= cfg.pixel_threshold:
            stable += 1
            if stable >= cfg.stable_frames_required:
                return {"accepted": True, "reason": "converged", "trace": trace}
        else:
            stable = 0
        motion = cfg.gain * error
        if cfg.noise_px > 0:
            motion += rng.normal(0.0, cfg.noise_px, size=2)
        tip = tip + motion
    return {"accepted": False, "reason": "max iterations reached", "trace": trace}

