#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keyboard_hover.visual_servo import ServoConfig, simulate_visual_servo


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate image-space visual servo convergence.")
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--tip-x", type=float, required=True)
    parser.add_argument("--tip-y", type=float, required=True)
    parser.add_argument("--threshold-px", type=float, default=15.0)
    parser.add_argument("--noise-px", type=float, default=0.0)
    parser.add_argument("--output", default="outputs/visual_servo_sim.json")
    args = parser.parse_args()

    result = simulate_visual_servo(
        target_px=(args.target_x, args.target_y),
        initial_tip_px=(args.tip_x, args.tip_y),
        config=ServoConfig(pixel_threshold=args.threshold_px, noise_px=args.noise_px),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    print(f"Accepted: {result['accepted']} ({result['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
