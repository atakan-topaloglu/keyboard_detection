from __future__ import annotations

import math
import unittest

import numpy as np
from PIL import Image, ImageDraw

from keyboard_hover.calibration import CalibrationConfig, aggregate_fits
from keyboard_hover.detector import NormalizedPrediction
from keyboard_hover.geometry import project_points
from keyboard_hover.grid_fit import FitConfig, fit_key_grid
from keyboard_hover.layout import QWERTZ_LAYOUT
from keyboard_hover.tip import detect_tip
from keyboard_hover.visual_servo import ServoConfig, simulate_visual_servo


def synthetic_predictions(dx: float = 0.0, dy: float = 0.0, noise: float = 0.0) -> list[NormalizedPrediction]:
    homography = np.array(
        [
            [85.0, 8.0, 180.0 + dx],
            [4.0, 76.0, 140.0 + dy],
            [0.0008, 0.0006, 1.0],
        ],
        dtype=float,
    )
    centers = QWERTZ_LAYOUT.centers()
    labels = [label for label in QWERTZ_LAYOUT.labels]
    src = np.asarray([centers[label] for label in labels], dtype=float)
    dst = project_points(src, homography)
    predictions = []
    for index, (label, point) in enumerate(zip(labels, dst, strict=True)):
        jitter_x = math.sin(index) * noise
        jitter_y = math.cos(index) * noise
        predictions.append(
            NormalizedPrediction(
                label=label,
                confidence=0.90,
                bbox_xywh=(float(point[0] + jitter_x), float(point[1] + jitter_y), 28.0, 24.0),
                center_px=(float(point[0] + jitter_x), float(point[1] + jitter_y)),
                raw_label=label,
            )
        )
    predictions.append(
        NormalizedPrediction(
            label="h",
            confidence=0.20,
            bbox_xywh=(1.0, 1.0, 20.0, 20.0),
            center_px=(1.0, 1.0),
            raw_label="h",
        )
    )
    return predictions


class KeyboardHoverTests(unittest.TestCase):
    def test_grid_fit_accepts_synthetic_qwertz_predictions(self) -> None:
        result = fit_key_grid(synthetic_predictions(noise=0.2), config=FitConfig())
        self.assertTrue(result.accepted, result.reason)
        self.assertIn("h", result.key_targets)
        self.assertIn(" ", result.key_targets)
        self.assertLess(result.residual_px, 2.0)
        self.assertNotIn("h", result.duplicate_labels)

    def test_calibration_accepts_repeatable_samples(self) -> None:
        fits = [
            fit_key_grid(synthetic_predictions(dx=0.0, dy=0.0, noise=0.2)),
            fit_key_grid(synthetic_predictions(dx=1.0, dy=-1.0, noise=0.2)),
            fit_key_grid(synthetic_predictions(dx=-1.0, dy=1.0, noise=0.2)),
        ]
        result = aggregate_fits(fits, target_text="h2 z", config=CalibrationConfig())
        self.assertTrue(result.accepted, result.reason)
        self.assertLessEqual(result.repeatability_px, 10.0)
        self.assertEqual(set(result.key_targets), {"h", "2", " ", "z"})

    def test_calibration_rejects_non_repeatable_samples(self) -> None:
        fits = [
            fit_key_grid(synthetic_predictions(dx=0.0, dy=0.0)),
            fit_key_grid(synthetic_predictions(dx=30.0, dy=0.0)),
            fit_key_grid(synthetic_predictions(dx=-30.0, dy=0.0)),
        ]
        result = aggregate_fits(fits, target_text="h", config=CalibrationConfig())
        self.assertFalse(result.accepted)
        self.assertGreater(result.repeatability_px, 10.0)

    def test_tip_detection_finds_synthetic_tip(self) -> None:
        image = Image.new("RGB", (640, 480), (240, 240, 240))
        draw = ImageDraw.Draw(image)
        draw.polygon([(310, 210), (240, 470), (390, 470)], fill=(0, 0, 0))
        tip = detect_tip(image)
        self.assertTrue(tip.accepted, tip.reason)
        assert tip.center_px is not None
        self.assertAlmostEqual(tip.center_px[0], 310.0, delta=12.0)
        self.assertAlmostEqual(tip.center_px[1], 210.0, delta=12.0)

    def test_visual_servo_simulation_converges(self) -> None:
        result = simulate_visual_servo(
            target_px=(500.0, 400.0),
            initial_tip_px=(100.0, 100.0),
            config=ServoConfig(pixel_threshold=15.0, stable_frames_required=3, noise_px=0.0),
        )
        self.assertTrue(result["accepted"], result["reason"])
        self.assertLess(result["trace"][-1]["error_norm_px"], 15.0)


if __name__ == "__main__":
    unittest.main()
