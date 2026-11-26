"""Data loading, slice extraction, degradation and subject-level splitting.

Works with three input kinds so the dataset choice can stay open:
  * NIfTI volumes (.nii / .nii.gz)  -> Medical Segmentation Decathlon, IXI   (needs nibabel)
  * HDF5 files (.h5)                -> fastMRI (see load_fastmri_h5 for the k-space step) (needs h5py)
  * NumPy arrays (.npy)             -> anything you pre-converted
"""
from __future__ import annotations

import glob
import os
from typing import Iterable

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Loading volumes
# ----------------------------------------------------------------------------
def load_nifti(path: str, channel: int = 0) -> np.ndarray:
    """Load a NIfTI file as a 3D array (X, Y, Z).

    MSD Task01 files are 4D (X, Y, Z, modality); `channel` picks one modality
    (0=FLAIR, 1=T1w, 2=T1gd, 3=T2w in that collection - check dataset.json).
    """
    try:
        import nibabel as nib
    except ImportError as e:  # pragma: no cover
        raise ImportError("nibabel is required to read NIfTI files: pip install nibabel") from e
    vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., channel]
    if vol.ndim != 3:
        raise ValueError(f"{path}: expected 3D/4D volume, got shape {vol.shape}")
    return vol


def load_fastmri_h5(path: str) -> np.ndarray:
    """Reconstruct magnitude images from a fastMRI brain multicoil .h5 file.

    fastMRI stores raw k-space, shape (slices, coils, H, W), complex. We do the
    standard root-sum-of-squares reconstruction: centered inverse FFT per coil,
    then combine coils. Returns (slices, H, W) float32.
    """
    try:
        import h5py
    except ImportError as e:  # pragma: no cover
        raise ImportError("h5py is required to read fastMRI files: pip install h5py") from e
    with h5py.File(path, "r") as f:
        kspace = f["kspace"][()]
    if kspace.ndim == 3:  # single-coil: (slices, H, W)
        kspace = kspace[:, None]
    img = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )
    return np.sqrt((np.abs(img) ** 2).sum(axis=1)).astype(np.float32)


def load_volume(path: str, channel: int = 0) -> np.ndarray:
    """Return a stack of 2D slices shaped (n_slices, H, W)."""
    if path.endswith((".nii", ".nii.gz")):
        vol = load_nifti(path, channel)
        return np.moveaxis(vol, 2, 0)  # axial slices first
    if path.endswith(".h5"):
        return load_fastmri_h5(path)
    if path.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        return arr[None] if arr.ndim == 2 else arr
    raise ValueError(f"Unsupported file type: {path}")


# ----------------------------------------------------------------------------
# Slice selection + normalization
# ----------------------------------------------------------------------------
def select_slices(
    stack: np.ndarray,
    min_foreground: float = 0.10,
    center_fraction: float = 0.6,
) -> np.ndarray:
    """Drop near-empty slices (top/bottom of the head, pure background).

    Keeps the central `center_fraction` of slices, and of those only slices
    where at least `min_foreground` of pixels are above 5% of the max intensity.
    """
    n = stack.shape[0]
    lo = int(n * (1 - center_fraction) / 2)
    hi = n - lo
    keep = []
    for i in range(lo, hi):
        s = stack[i]
        mx = float(s.max())
        if mx <= 0:
            continue
        if (s > 0.05 * mx).mean() >= min_foreground:
            keep.append(i)
    return stack[keep]


def normalize(img: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Per-slice robust min-max normalization to [0, 1] (clips outlier intensities)."""
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def center_crop_to_multiple(img: np.ndarray, multiple: int) -> np.ndarray:
    """Crop so H and W are divisible by the scale factor (keeps down/up-sampling exact)."""
    h, w = img.shape[-2:]
    h2, w2 = h - h % multiple, w - w % multiple
    top, left = (h - h2) // 2, (w - w2) // 2
    return img[..., top : top + h2, left : left + w2]


# ----------------------------------------------------------------------------
# Degradation: build the low-resolution input from a high-resolution slice
# ----------------------------------------------------------------------------
def degrade(hr: np.ndarray, scale: int, mode: str = "area") -> np.ndarray:
    """Simulate a low-resolution acquisition by downsampling `hr` by `scale`.

    mode="area"   : average pooling-style downsampling (anti-aliased; default)
    mode="kspace" : crop the centre of k-space, which is closer to how a faster
                    MRI acquisition loses high-frequency detail.
    NOTE: both are simulations; real fast or low-field scans also differ in noise
    and contrast, so say "simulated low-resolution" in the report.
    """
    hr = center_crop_to_multiple(hr, scale)
    h, w = hr.shape
    if mode == "area":
        return cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    if mode == "kspace":
        k = np.fft.fftshift(np.fft.fft2(hr))
        ch, cw = h // scale, w // scale
        t, l = (h - ch) // 2, (w - cw) // 2
        k_small = k[t : t + ch, l : l + cw]
        lr = np.fft.ifft2(np.fft.ifftshift(k_small))
        # rescale: fft2/ifft2 normalisation changes with array size
        return np.abs(lr).astype(np.float32) * (ch * cw) / (h * w)
    raise ValueError(f"unknown degradation mode: {mode}")


def upsample_bicubic(lr: np.ndarray, scale: int) -> np.ndarray:
    """Conventional baseline: bicubic interpolation back to the HR grid."""
    h, w = lr.shape
    out = cv2.resize(lr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Subject-level split (no subject appears in more than one set)
# ----------------------------------------------------------------------------
def split_subjects(
    files: Iterable[str], seed: int = 42, frac=(0.7, 0.15, 0.15)
) -> dict[str, list[str]]:
    """Split FILES (one file == one subject) into train/val/test.

    Splitting by file, not by slice, avoids leakage: neighbouring slices of the
    same brain are nearly identical, so a slice-level split inflates results.
    """
    files = sorted(files)
    if len(files) < 3:
        raise ValueError("need at least 3 subjects/files for a train/val/test split")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    n_train = max(1, int(round(frac[0] * len(files))))
    n_val = max(1, int(round(frac[1] * len(files))))
    n_train = min(n_train, len(files) - 2)
    n_val = min(n_val, len(files) - n_train - 1)
    pick = lambda ids: [files[i] for i in ids]  # noqa: E731
    return {
        "train": pick(idx[:n_train]),
        "val": pick(idx[n_train : n_train + n_val]),
        "test": pick(idx[n_train + n_val :]),
    }


def list_volumes(root: str) -> list[str]:
    pats = ("*.nii", "*.nii.gz", "*.h5", "*.npy")
    out: list[str] = []
    for p in pats:
        out += glob.glob(os.path.join(root, "**", p), recursive=True)
    # skip macOS resource forks that ship inside some MSD archives
    return sorted(f for f in out if not os.path.basename(f).startswith("._"))


def make_pairs(
    files: Iterable[str],
    scale: int,
    channel: int = 0,
    mode: str = "area",
    max_slices_per_subject: int | None = None,
):
    """Yield (subject, hr, lr) float32 pairs, each normalized to [0, 1]."""
    for f in files:
        stack = select_slices(load_volume(f, channel))
        if max_slices_per_subject and len(stack) > max_slices_per_subject:
            keep = np.linspace(0, len(stack) - 1, max_slices_per_subject).astype(int)
            stack = stack[keep]
        for s in stack:
            hr = center_crop_to_multiple(normalize(s), scale)
            yield os.path.basename(f), hr, degrade(hr, scale, mode)
