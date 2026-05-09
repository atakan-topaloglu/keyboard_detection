from __future__ import annotations

import base64
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageOps

from keyboard_hover.layout import normalize_label


@dataclass(frozen=True)
class NormalizedPrediction:
    label: str
    confidence: float
    bbox_xywh: tuple[float, float, float, float]
    center_px: tuple[float, float]
    raw_label: str

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox_xywh"] = list(self.bbox_xywh)
        data["center_px"] = list(self.center_px)
        return data


@dataclass(frozen=True)
class DetectionResult:
    image_path: str
    image_size: tuple[int, int]
    raw: dict[str, Any]
    predictions: list[NormalizedPrediction]
    end_to_end_seconds: float


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
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


def normalize_predictions(raw_predictions: list[dict[str, Any]]) -> list[NormalizedPrediction]:
    normalized: list[NormalizedPrediction] = []
    for prediction in raw_predictions:
        raw_label = str(prediction.get("class", prediction.get("label", "")))
        label = normalize_label(raw_label)
        if label is None:
            continue
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
        confidence = float(prediction.get("confidence", 0.0))
        normalized.append(
            NormalizedPrediction(
                label=label,
                confidence=confidence,
                bbox_xywh=(x, y, width, height),
                center_px=(x, y),
                raw_label=raw_label,
            )
        )
    return normalized


class DetectorClient:
    def __init__(
        self,
        env_file: str | Path = ".env.inference",
        host: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        env_path = Path(env_file)
        env_values = load_env_file(env_path)
        self.host = host or os.environ.get("INFERENCE_HOST") or env_values.get("INFERENCE_HOST", "http://localhost:9001")
        self.model_id = (
            model_id
            or os.environ.get("ROBOFLOW_MODEL_ID")
            or env_values.get("ROBOFLOW_MODEL_ID", "keyboard-key-recognition-kw7nc/14")
        )
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY") or env_values.get("ROBOFLOW_API_KEY")
        self.timeout_s = timeout_s
        if not self.api_key:
            raise RuntimeError("ROBOFLOW_API_KEY is required for live detection.")

    def detect_image(self, image_path: str | Path, confidence: float = 0.50) -> DetectionResult:
        source = Path(image_path)
        if not source.exists():
            raise FileNotFoundError(f"Image not found: {source}")
        image = normalize_image(source)
        payload = {
            "id": str(uuid.uuid4()),
            "model_id": self.model_id,
            "api_key": self.api_key,
            "image": {"type": "base64", "value": encode_image(image)},
            "confidence": confidence,
        }
        start = time.perf_counter()
        response = requests.post(
            f"{self.host.rstrip('/')}/infer/object_detection",
            json=payload,
            timeout=self.timeout_s,
        )
        end = time.perf_counter()
        response.raise_for_status()
        raw = response.json()
        return DetectionResult(
            image_path=str(source),
            image_size=image.size,
            raw=raw,
            predictions=normalize_predictions(raw.get("predictions", [])),
            end_to_end_seconds=end - start,
        )


def load_raw_predictions(path: str | Path) -> list[NormalizedPrediction]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return normalize_predictions(raw.get("predictions", []))

