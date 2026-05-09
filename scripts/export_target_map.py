#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keyboard_hover.calibration import CalibrationConfig, build_target_map
from keyboard_hover.detector import load_raw_predictions
from keyboard_hover.overlay import draw_target_overlay


def image_size(path: str | None) -> tuple[int, int] | None:
    if path is None:
        return None
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw)
        return image.size


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a robot-facing target map from detector JSONs.")
    parser.add_argument("--raw-json", nargs="+", required=True, help="Roboflow-style raw detector JSON files.")
    parser.add_argument("--target-text", required=True, help="Requested text/key sequence.")
    parser.add_argument("--image", default=None, help="Reference image used for size/tip/overlay.")
    parser.add_argument("--tip-image", default=None, help="Optional image for end-effector tip detection.")
    parser.add_argument("--tip-x", type=float, default=None, help="Manual tip x override for offline debugging.")
    parser.add_argument("--tip-y", type=float, default=None, help="Manual tip y override for offline debugging.")
    parser.add_argument("--output", default="outputs/target_map.json")
    parser.add_argument("--overlay", default=None)
    parser.add_argument("--samples-required", type=int, default=3)
    parser.add_argument("--repeatability-px", type=float, default=10.0)
    args = parser.parse_args()

    samples = [load_raw_predictions(path) for path in args.raw_json]
    tip_override = None
    if args.tip_x is not None or args.tip_y is not None:
        if args.tip_x is None or args.tip_y is None:
            raise ValueError("--tip-x and --tip-y must be provided together")
        tip_override = (args.tip_x, args.tip_y)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    target_map = build_target_map(
        samples=samples,
        target_text=args.target_text.lower(),
        image_path=args.image,
        image_size=image_size(args.image),
        tip_image=args.tip_image or args.image,
        tip_override=tip_override,
        config=CalibrationConfig(
            samples_required=args.samples_required,
            repeatability_threshold_px=args.repeatability_px,
        ),
    )
    output.write_text(json.dumps(target_map, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    print(f"Calibration accepted: {target_map['calibration']['accepted']} ({target_map['calibration']['reason']})")

    if args.overlay:
        if not args.image:
            raise ValueError("--image is required when --overlay is set")
        draw_target_overlay(args.image, target_map, args.overlay)
        print(f"Saved {args.overlay}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
