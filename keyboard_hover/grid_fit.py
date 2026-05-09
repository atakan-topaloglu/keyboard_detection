from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from keyboard_hover.detector import NormalizedPrediction
from keyboard_hover.geometry import project_points, ransac_homography
from keyboard_hover.layout import QWERTZ_LAYOUT, SUPPORTED_CHARS, KeyboardLayout


@dataclass(frozen=True)
class FitConfig:
    confidence_threshold: float = 0.50
    min_anchor_keys: int = 12
    ransac_threshold_px: float = 25.0
    median_residual_threshold_px: float = 15.0


@dataclass(frozen=True)
class KeyTarget:
    center_px: tuple[float, float]
    confidence: float
    source: str

    def to_json(self) -> dict[str, Any]:
        return {
            "center_px": [float(self.center_px[0]), float(self.center_px[1])],
            "confidence": float(self.confidence),
            "source": self.source,
        }


@dataclass(frozen=True)
class FitResult:
    accepted: bool
    reason: str
    residual_px: float
    transform_layout_to_image: list[list[float]] | None
    key_targets: dict[str, KeyTarget]
    used_anchor_labels: list[str]
    duplicate_labels: list[str]
    rejected_labels: list[str]


def _best_predictions(
    predictions: list[NormalizedPrediction],
    config: FitConfig,
) -> tuple[dict[str, NormalizedPrediction], list[str]]:
    best: dict[str, NormalizedPrediction] = {}
    duplicates: set[str] = set()
    for prediction in predictions:
        if prediction.confidence < config.confidence_threshold:
            continue
        if prediction.label not in SUPPORTED_CHARS:
            continue
        previous = best.get(prediction.label)
        if previous is not None:
            duplicates.add(prediction.label)
            if prediction.confidence <= previous.confidence:
                continue
        best[prediction.label] = prediction
    return best, sorted(duplicates)


def fit_key_grid(
    predictions: list[NormalizedPrediction],
    layout: KeyboardLayout = QWERTZ_LAYOUT,
    config: FitConfig | None = None,
) -> FitResult:
    cfg = config or FitConfig()
    best, duplicates = _best_predictions(predictions, cfg)
    template = layout.anchor_centers()
    anchor_labels = [label for label in layout.labels if label in best and label in template]
    if len(anchor_labels) < cfg.min_anchor_keys:
        return FitResult(
            accepted=False,
            reason=f"not enough anchors: {len(anchor_labels)} < {cfg.min_anchor_keys}",
            residual_px=float("inf"),
            transform_layout_to_image=None,
            key_targets={},
            used_anchor_labels=anchor_labels,
            duplicate_labels=duplicates,
            rejected_labels=[],
        )

    src = np.asarray([template[label] for label in anchor_labels], dtype=float)
    dst = np.asarray([best[label].center_px for label in anchor_labels], dtype=float)
    homography, inliers, errors = ransac_homography(src, dst, threshold_px=cfg.ransac_threshold_px)
    used_labels = [label for label, keep in zip(anchor_labels, inliers, strict=True) if keep]
    rejected_labels = [label for label, keep in zip(anchor_labels, inliers, strict=True) if not keep]
    median_residual = float(np.median(errors[inliers])) if inliers.any() else float("inf")

    accepted = len(used_labels) >= cfg.min_anchor_keys and median_residual <= cfg.median_residual_threshold_px
    reason = "accepted" if accepted else (
        f"fit rejected: anchors={len(used_labels)}, median_residual={median_residual:.2f}px"
    )

    all_template = layout.centers()
    labels = [label for label in SUPPORTED_CHARS if label in all_template]
    projected = project_points(np.asarray([all_template[label] for label in labels], dtype=float), homography)
    key_targets: dict[str, KeyTarget] = {}
    for label, point in zip(labels, projected, strict=True):
        source = "inferred"
        confidence = 0.50 if accepted else 0.0
        if label in best:
            source = "detected"
            confidence = best[label].confidence
            point = np.asarray(best[label].center_px, dtype=float)
        if label == " " and source == "inferred":
            has_lower_context = any(label in used_labels for label in "yxcvbnm") and any(
                label in used_labels for label in "asdfghjkl"
            )
            if not (accepted and has_lower_context):
                continue
        key_targets[label] = KeyTarget(
            center_px=(float(point[0]), float(point[1])),
            confidence=float(confidence),
            source=source,
        )

    return FitResult(
        accepted=accepted,
        reason=reason,
        residual_px=median_residual,
        transform_layout_to_image=homography.tolist(),
        key_targets=key_targets if accepted else {},
        used_anchor_labels=used_labels,
        duplicate_labels=duplicates,
        rejected_labels=rejected_labels,
    )

