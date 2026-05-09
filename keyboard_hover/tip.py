from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class TipDetection:
    center_px: tuple[float, float] | None
    confidence: float
    reason: str

    @property
    def accepted(self) -> bool:
        return self.center_px is not None and self.confidence >= 0.50

    def to_json(self) -> dict:
        return {
            "center_px": None if self.center_px is None else [float(self.center_px[0]), float(self.center_px[1])],
            "confidence": float(self.confidence),
            "reason": self.reason,
        }


def _load_rgb(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        rgb = ImageOps.exif_transpose(image)
        return rgb.convert("RGB") if rgb.mode != "RGB" else rgb.copy()
    with Image.open(image) as raw:
        rgb = ImageOps.exif_transpose(raw)
        return rgb.convert("RGB") if rgb.mode != "RGB" else rgb.copy()


def detect_tip(image: str | Path | Image.Image) -> TipDetection:
    rgb = _load_rgb(image)
    arr = np.asarray(rgb, dtype=np.uint8)
    h, w = arr.shape[:2]
    gray = arr.astype(float).mean(axis=2)

    y0 = int(h * 0.25)
    y1 = int(h * 0.98)
    x0 = int(w * 0.12)
    x1 = int(w * 0.88)
    roi = gray[y0:y1, x0:x1]
    mask = roi < 55.0
    if int(mask.sum()) < 80:
        return TipDetection(None, 0.0, "not enough dark pixels in end-effector ROI")

    try:
        import cv2  # type: ignore[import-not-found]

        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        best_label = None
        best_score = -1.0
        for label_idx in range(1, labels_count):
            x, y, width, height, area = stats[label_idx]
            if area < 80:
                continue
            touches_bottom = y + height >= mask.shape[0] * 0.78
            if not touches_bottom:
                continue
            cx = x + width / 2
            center_penalty = abs(cx - mask.shape[1] / 2) / max(mask.shape[1], 1)
            score = float(area) + float(height) * 10.0 - center_penalty * 5000.0
            if score > best_score:
                best_score = score
                best_label = label_idx
        if best_label is not None:
            ys, xs = np.where(labels == best_label)
        else:
            ys, xs = np.where(mask)
    except Exception:
        ys, xs = np.where(mask)
        lower = ys > mask.shape[0] * 0.45
        if lower.any():
            ys = ys[lower]
            xs = xs[lower]

    if len(xs) < 80:
        return TipDetection(None, 0.0, "no usable end-effector component")

    min_y = int(ys.min())
    tip_band = ys <= min_y + max(4, int(h * 0.006))
    tip_x = float(np.median(xs[tip_band]) + x0)
    tip_y = float(np.median(ys[tip_band]) + y0)
    confidence = float(min(1.0, len(xs) / (0.012 * w * h)))
    return TipDetection((tip_x, tip_y), confidence, "dark connected end-effector tip")

