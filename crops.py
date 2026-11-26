"""Zoomed-crop figure: makes small differences between methods visible.

Picks a detailed region (cortical folds / edges) of a few TEST slices and shows it
enlarged for the original, bicubic and each CNN, with the local RMSE under each crop.

Usage (same arguments style as evaluate.py):
    python crops.py --data /content/ixi_t1 --scale 2 --model srcnn --checkpoint srcnn_x2.pt \
                    --extra vdsr vdsr_x2.pt
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter

from data import list_volumes, make_pairs, split_subjects, upsample_bicubic


def find_detailed_region(hr: np.ndarray, size: int, center_frac: float = 0.55) -> tuple[int, int]:
    """Top-left (row, col) of the size x size window with the most texture that is
    (almost) entirely tissue, so the crop shows anatomy, not background.

    Only windows whose centre lies in the middle `center_frac` of the image are
    considered. Unstripped head scans have very sharp edges at the face and neck, and
    without this limit the search tends to land there instead of on the brain.
    """
    size = min(size, *hr.shape)
    mean = uniform_filter(hr, size)
    var = uniform_filter(hr**2, size) - mean**2
    tissue = uniform_filter((hr > 0.1).astype(np.float32), size)
    base = np.where(tissue > 0.9, var, -1.0)
    h, w = hr.shape
    frac = center_frac
    while True:
        margin_r, margin_c = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
        allowed = np.zeros_like(base, dtype=bool)
        allowed[margin_r : h - margin_r, margin_c : w - margin_c] = True
        score = np.where(allowed, base, -1.0)
        if score.max() >= 0 or frac >= 1.0:  # no valid window yet -> widen the search
            break
        frac = min(1.0, frac + 0.1)
    r, c = np.unravel_index(np.argmax(score), score.shape)
    top = int(np.clip(r - size // 2, 0, hr.shape[0] - size))
    left = int(np.clip(c - size // 2, 0, hr.shape[1] - size))
    return top, left


def crop_figure(items, methods: dict, scale: int, out_png: str, size: int = 64,
                box: tuple[int, int] | None = None, center_frac: float = 0.55):
    """items: list of (subject, hr, lr). methods: name -> f(lr) -> sr (full-size array).

    box=(top, left) forces the same crop position on every slice (overrides the search).
    """
    names = list(methods)
    cols = 2 + len(names)  # full slice with box, original crop, then one per method
    fig, axes = plt.subplots(len(items), cols, figsize=(2.6 * cols, 2.7 * len(items)))
    axes = np.atleast_2d(axes)
    for r, (subject, hr, lr) in enumerate(items):
        if box is not None:
            top = int(np.clip(box[0], 0, hr.shape[0] - size))
            left = int(np.clip(box[1], 0, hr.shape[1] - size))
        else:
            top, left = find_detailed_region(hr, size, center_frac)
        sl = (slice(top, top + size), slice(left, left + size))
        axes[r, 0].imshow(hr, cmap="gray", vmin=0, vmax=1)
        axes[r, 0].add_patch(Rectangle((left, top), size, size, fill=False, ec="red", lw=1.5))
        axes[r, 1].imshow(hr[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r, 1].set_xlabel("Original", fontsize=8)
        for c, n in enumerate(names):
            sr = methods[n](lr)
            err = float(np.sqrt(np.mean((hr[sl] - sr[sl]) ** 2)))
            axes[r, 2 + c].imshow(sr[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[r, 2 + c].set_xlabel(f"{n}\ncrop RMSE {err:.3f}", fontsize=8)
    axes[0, 0].set_title("Slice (red = crop)", fontsize=9)
    axes[0, 1].set_title(f"Zoom ({size}x{size} px)", fontsize=9)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main():
    from evaluate import build_cnn_method  # lazy: only needed when CNN checkpoints are given

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--mode", default="area", choices=["area", "kspace"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=3, help="number of slices to show")
    ap.add_argument("--size", type=int, default=64, help="crop size in pixels")
    ap.add_argument("--center-frac", type=float, default=0.55,
                    help="search for the crop only in the middle fraction of the image (smaller = more central)")
    ap.add_argument("--crop", type=int, nargs=2, metavar=("TOP", "LEFT"),
                    help="force the crop position (pixels from top-left) on every slice")
    ap.add_argument("--model", choices=["srcnn", "vdsr"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--extra", nargs=2, action="append", metavar=("MODEL", "CKPT"))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    test_files = split_subjects(list_volumes(a.data), seed=a.seed)["test"]  # test subjects only
    pairs = list(make_pairs(test_files[: a.n * 2], a.scale, a.channel, a.mode, max_slices_per_subject=5))
    pick = np.linspace(0, len(pairs) - 1, a.n).astype(int)
    items = [pairs[i] for i in pick]

    methods = {"Bicubic": lambda lr: upsample_bicubic(lr, a.scale)}
    if a.model and a.checkpoint:
        methods[a.model.upper()] = build_cnn_method(a.model, a.checkpoint, a.scale)
    for m, ck in a.extra or []:
        methods[m.upper()] = build_cnn_method(m, ck, a.scale)

    out = f"{a.out}/crops_x{a.scale}.png"
    crop_figure(items, methods, a.scale, out, a.size,
                tuple(a.crop) if a.crop else None, a.center_frac)
    print("wrote", out)


if __name__ == "__main__":
    main()
