from __future__ import annotations

import itertools

import numpy as np

Point = tuple[float, float]


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src and dst must both have shape (N, 2)")
    if len(src) < 4:
        raise ValueError("At least four point correspondences are required")

    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=float))
    h = vh[-1].reshape(3, 3)
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    ones = np.ones((len(points), 1), dtype=float)
    homogeneous = np.concatenate([points, ones], axis=1) @ homography.T
    z = homogeneous[:, 2:3]
    z[np.abs(z) < 1e-12] = 1e-12
    return homogeneous[:, :2] / z


def reprojection_errors(src: np.ndarray, dst: np.ndarray, homography: np.ndarray) -> np.ndarray:
    projected = project_points(src, homography)
    return np.linalg.norm(projected - np.asarray(dst, dtype=float), axis=1)


def ransac_homography(
    src: np.ndarray,
    dst: np.ndarray,
    threshold_px: float,
    max_iterations: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if len(src) < 4:
        raise ValueError("At least four points are required")

    rng = np.random.default_rng(7)
    candidate_indices: list[tuple[int, ...]]
    all_combos = list(itertools.combinations(range(len(src)), 4))
    if len(all_combos) <= max_iterations:
        candidate_indices = all_combos
    else:
        candidate_indices = [tuple(rng.choice(len(src), size=4, replace=False).tolist()) for _ in range(max_iterations)]

    best_h = fit_homography(src, dst)
    best_errors = reprojection_errors(src, dst, best_h)
    best_inliers = best_errors <= threshold_px

    for indices in candidate_indices:
        try:
            h = fit_homography(src[list(indices)], dst[list(indices)])
        except np.linalg.LinAlgError:
            continue
        errors = reprojection_errors(src, dst, h)
        inliers = errors <= threshold_px
        if inliers.sum() > best_inliers.sum() or (
            inliers.sum() == best_inliers.sum() and np.median(errors[inliers]) < np.median(best_errors[best_inliers])
        ):
            best_h = h
            best_errors = errors
            best_inliers = inliers

    if best_inliers.sum() >= 4:
        best_h = fit_homography(src[best_inliers], dst[best_inliers])
        best_errors = reprojection_errors(src, dst, best_h)
        best_inliers = best_errors <= threshold_px

    return best_h, best_inliers, best_errors

