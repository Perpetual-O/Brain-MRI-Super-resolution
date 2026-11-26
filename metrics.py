"""Image-quality metrics used to compare reconstructions with the original slice."""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def rmse(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.sqrt(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)))


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, test, data_range=1.0))


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    return float(structural_similarity(ref, test, data_range=1.0))


def all_metrics(ref: np.ndarray, test: np.ndarray) -> dict[str, float]:
    return {"psnr": psnr(ref, test), "ssim": ssim(ref, test), "rmse": rmse(ref, test)}


def summarize(rows: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of each metric across slices."""
    out = {}
    for key in rows[0]:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[key] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    return out
