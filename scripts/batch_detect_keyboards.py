#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keyboard_hover.detector import DetectorClient


def image_paths(input_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(input_dir.glob(suffix))
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run local keyboard detection on images.")
    parser.add_argument("--input-dir", default="keyboard_pictures")
    parser.add_argument("--output-dir", default="outputs/batch_detection")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--env-file", default=".env.inference")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    normalized_dir = output_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    client = DetectorClient(env_file=args.env_file)
    summary = []
    for path in image_paths(input_dir):
        result = client.detect_image(path, confidence=args.threshold)
        raw_path = raw_dir / f"{path.stem}.json"
        normalized_path = normalized_dir / f"{path.stem}.json"
        raw_path.write_text(json.dumps(result.raw, indent=2), encoding="utf-8")
        normalized_path.write_text(
            json.dumps(
                {
                    "image_path": str(path),
                    "image_size": list(result.image_size),
                    "threshold": args.threshold,
                    "end_to_end_seconds": result.end_to_end_seconds,
                    "predictions": [prediction.to_json() for prediction in result.predictions],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary.append(
            {
                "image_path": str(path),
                "raw_path": str(raw_path),
                "normalized_path": str(normalized_path),
                "detections": len(result.predictions),
                "end_to_end_seconds": result.end_to_end_seconds,
            }
        )
        print(f"{path.name}: {len(result.predictions)} detections")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
