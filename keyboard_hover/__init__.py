"""Keyboard vision-to-hover helpers for Project 4."""

from keyboard_hover.calibration import CalibrationConfig, CalibrationResult, build_target_map
from keyboard_hover.detector import DetectorClient, NormalizedPrediction, normalize_predictions
from keyboard_hover.grid_fit import FitConfig, FitResult, fit_key_grid
from keyboard_hover.layout import QWERTZ_LAYOUT, SUPPORTED_CHARS, KeyboardLayout
from keyboard_hover.tip import TipDetection, detect_tip
from keyboard_hover.visual_servo import ServoConfig, simulate_visual_servo

__all__ = [
    "CalibrationConfig",
    "CalibrationResult",
    "DetectorClient",
    "FitConfig",
    "FitResult",
    "KeyboardLayout",
    "NormalizedPrediction",
    "QWERTZ_LAYOUT",
    "SUPPORTED_CHARS",
    "ServoConfig",
    "TipDetection",
    "build_target_map",
    "detect_tip",
    "fit_key_grid",
    "normalize_predictions",
    "simulate_visual_servo",
]
