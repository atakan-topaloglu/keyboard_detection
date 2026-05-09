from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from keyboard_hover.config import CAMERA_ID, INTRINSICS_ID, SCHEMA_VERSION
from keyboard_hover.detector import NormalizedPrediction
from keyboard_hover.grid_fit import FitConfig, FitResult, fit_key_grid
from keyboard_hover.layout import QWERTZ_LAYOUT, SUPPORTED_CHARS, KeyboardLayout
from keyboard_hover.tip import TipDetection, detect_tip


@dataclass(frozen=True)
class CalibrationConfig:
    samples_required: int = 3
    repeatability_threshold_px: float = 10.0
    fit_config: FitConfig = FitConfig()


@dataclass(frozen=True)
class CalibrationResult:
    accepted: bool
    reason: str
    residual_px: float
    repeatability_px: float
    key_targets: dict[str, dict[str, Any]]
    transform_layout_to_image: list[list[float]] | None
    sample_results: list[FitResult]


def _target_labels(target_text: str) -> list[str]:
    labels = []
    for char in target_text.lower():
        if char in SUPPORTED_CHARS and char not in labels:
            labels.append(char)
    return labels


def aggregate_fits(
    sample_results: list[FitResult],
    target_text: str,
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    cfg = config or CalibrationConfig()
    if len(sample_results) < cfg.samples_required:
        return CalibrationResult(
            accepted=False,
            reason=f"need {cfg.samples_required} samples, got {len(sample_results)}",
            residual_px=float("inf"),
            repeatability_px=float("inf"),
            key_targets={},
            transform_layout_to_image=None,
            sample_results=sample_results,
        )

    if any(not result.accepted for result in sample_results):
        reasons = "; ".join(result.reason for result in sample_results if not result.accepted)
        return CalibrationResult(
            accepted=False,
            reason=f"sample fit rejected: {reasons}",
            residual_px=float("inf"),
            repeatability_px=float("inf"),
            key_targets={},
            transform_layout_to_image=None,
            sample_results=sample_results,
        )

    labels = _target_labels(target_text)
    key_targets: dict[str, dict[str, Any]] = {}
    repeatabilities = []
    for label in labels:
        if not all(label in result.key_targets for result in sample_results):
            continue
        centers = np.asarray([result.key_targets[label].center_px for result in sample_results], dtype=float)
        median_center = np.median(centers, axis=0)
        distances = np.linalg.norm(centers - median_center, axis=1)
        repeatabilities.append(float(distances.max()))
        confidences = [result.key_targets[label].confidence for result in sample_results]
        sources = [result.key_targets[label].source for result in sample_results]
        key_targets[label] = {
            "center_px": [float(median_center[0]), float(median_center[1])],
            "confidence": float(np.median(confidences)),
            "source": "detected" if all(source == "detected" for source in sources) else "inferred",
        }

    missing = [label for label in labels if label not in key_targets]
    if missing:
        return CalibrationResult(
            accepted=False,
            reason=f"missing target labels after fitting: {missing}",
            residual_px=float("inf"),
            repeatability_px=float("inf"),
            key_targets={},
            transform_layout_to_image=None,
            sample_results=sample_results,
        )

    repeatability = max(repeatabilities) if repeatabilities else 0.0
    residual = float(np.median([result.residual_px for result in sample_results]))
    accepted = repeatability <= cfg.repeatability_threshold_px
    reason = "accepted" if accepted else f"repeatability {repeatability:.2f}px exceeds threshold"
    transform = sample_results[len(sample_results) // 2].transform_layout_to_image
    return CalibrationResult(
        accepted=accepted,
        reason=reason,
        residual_px=residual,
        repeatability_px=repeatability,
        key_targets=key_targets if accepted else {},
        transform_layout_to_image=transform if accepted else None,
        sample_results=sample_results,
    )


def build_target_map(
    samples: list[list[NormalizedPrediction]],
    target_text: str,
    image_path: str | Path | None = None,
    image_size: tuple[int, int] | None = None,
    tip_image: str | Path | Image.Image | None = None,
    tip_override: tuple[float, float] | None = None,
    layout: KeyboardLayout = QWERTZ_LAYOUT,
    config: CalibrationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CalibrationConfig()
    sample_results = [fit_key_grid(sample, layout=layout, config=cfg.fit_config) for sample in samples]
    calibration = aggregate_fits(sample_results, target_text=target_text, config=cfg)
    tip: TipDetection | None = None
    if tip_override is not None:
        tip = TipDetection(tip_override, 1.0, "manual override")
    elif tip_image is not None:
        tip = detect_tip(tip_image)

    width, height = image_size or (0, 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "image": {
            "path": None if image_path is None else str(image_path),
            "width": int(width),
            "height": int(height),
            "camera_id": CAMERA_ID,
            "intrinsics_id": INTRINSICS_ID,
        },
        "request": {
            "target_text": target_text,
            "layout": layout.name,
            "allowed_chars": SUPPORTED_CHARS,
        },
        "calibration": {
            "method": "markerless_key_grid",
            "num_samples": len(samples),
            "accepted": calibration.accepted,
            "reason": calibration.reason,
            "residual_px": calibration.residual_px,
            "repeatability_px": calibration.repeatability_px,
            "transform_layout_to_image": calibration.transform_layout_to_image,
        },
        "key_targets": calibration.key_targets,
        "tip": None if tip is None else tip.to_json(),
    }

