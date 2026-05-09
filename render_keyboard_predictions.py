#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_env_file(path: Path) -> dict[str, str]:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key] = value
    return env


def normalize_image(source_path: Path) -> Image.Image:
    with Image.open(source_path) as image:
        normalized = ImageOps.exif_transpose(image)
        if normalized.mode != "RGB":
            normalized = normalized.convert("RGB")
        return normalized.copy()


def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def annotate_image(
    image: Image.Image,
    predictions: list[dict],
    threshold: float,
    inference_time_s: float | None,
    end_to_end_s: float,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    stroke_width = max(2, round(min(annotated.size) / 220))
    padding = 4

    header = (
        f"threshold={threshold:.2f} | "
        f"inference={inference_time_s * 1000:.1f} ms | "
        f"end-to-end={end_to_end_s * 1000:.1f} ms | "
        f"detections={len(predictions)}"
        if inference_time_s is not None
        else f"threshold={threshold:.2f} | end-to-end={end_to_end_s * 1000:.1f} ms | detections={len(predictions)}"
    )
    header_w, header_h = measure_text(draw, header, font)
    draw.rectangle((0, 0, header_w + padding * 2, header_h + padding * 2), fill=(0, 0, 0))
    draw.text((padding, padding), header, fill=(255, 255, 255), font=font)

    for prediction in predictions:
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
        label = str(prediction["class"])
        confidence = float(prediction["confidence"])

        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2

        if label.lower() == "keyboard":
            color = (255, 165, 0)
        else:
            color = (0, 255, 0)

        draw.rectangle((left, top, right, bottom), outline=color, width=stroke_width)

        text = label
        text_w, text_h = measure_text(draw, text, font)
        text_left = clamp(left, 0, annotated.width - text_w - padding * 2)
        candidate_top = top - text_h - padding * 2 - 1
        if candidate_top < header_h + padding * 3:
            text_top = min(bottom + 1, annotated.height - text_h - padding * 2)
        else:
            text_top = candidate_top
        draw.rectangle(
            (text_left, text_top, text_left + text_w + padding * 2, text_top + text_h + padding * 2),
            fill=color,
        )
        draw.text((text_left + padding, text_top + padding), text, fill=(0, 0, 0), font=font)

        center_radius = max(1, stroke_width)
        draw.ellipse(
            (x - center_radius, y - center_radius, x + center_radius, y + center_radius),
            fill=color,
        )

        # Keep the confidence in metadata, not the label text, so the image
        # stays readable even after lowering the threshold.
        prediction["confidence"] = confidence

    return annotated


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local keyboard detection with label overlay.")
    parser.add_argument("--image", default="key_detection.jpeg", help="Path to source image.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for outputs.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold.")
    parser.add_argument("--env-file", default=".env.inference", help="Local env file.")
    args = parser.parse_args()

    root = Path.cwd()
    image_path = (root / args.image).resolve() if not Path(args.image).is_absolute() else Path(args.image)
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    env_file = (root / args.env_file).resolve() if not Path(args.env_file).is_absolute() else Path(args.env_file)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    env_values = load_env_file(env_file)
    api_key = os.environ.get("ROBOFLOW_API_KEY") or env_values.get("ROBOFLOW_API_KEY")
    model_id = os.environ.get("ROBOFLOW_MODEL_ID") or env_values.get("ROBOFLOW_MODEL_ID", "keyboard-key-recognition-kw7nc/14")
    host = os.environ.get("INFERENCE_HOST") or env_values.get("INFERENCE_HOST", "http://localhost:9001")

    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required.")

    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalize_image(image_path)
    request_payload = {
        "id": str(uuid.uuid4()),
        "model_id": model_id,
        "api_key": api_key,
        "image": {"type": "base64", "value": encode_image(normalized)},
        "confidence": args.threshold,
    }

    start = time.perf_counter()
    response = requests.post(f"{host.rstrip('/')}/infer/object_detection", json=request_payload, timeout=120)
    end = time.perf_counter()
    response.raise_for_status()
    result = response.json()

    predictions = result.get("predictions", [])
    annotated = annotate_image(
        image=normalized,
        predictions=predictions,
        threshold=args.threshold,
        inference_time_s=result.get("time"),
        end_to_end_s=end - start,
    )

    threshold_suffix = f"{args.threshold:.2f}".replace(".", "p")
    stem = image_path.stem
    json_path = output_dir / f"{stem}_prediction_threshold_{threshold_suffix}.json"
    image_path_out = output_dir / f"{stem}_prediction_labels_threshold_{threshold_suffix}.jpg"

    result["timing"] = {
        "threshold": args.threshold,
        "inference_seconds": result.get("time"),
        "end_to_end_seconds": end - start,
    }

    json_path.write_text(json.dumps(result, indent=2))
    annotated.save(image_path_out, quality=95)

    print(f"Saved {json_path}")
    print(f"Saved {image_path_out}")
    if result.get("time") is not None:
        print(f"Inference time: {result['time'] * 1000:.1f} ms")
    print(f"End-to-end time: {(end - start) * 1000:.1f} ms")
    print(f"Detections: {len(predictions)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
