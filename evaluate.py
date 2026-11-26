"""Evaluate reconstruction methods on the TEST subjects only.

Usage:
    python evaluate.py --data /path/to/mri --scale 2 --channel 0
    python evaluate.py --data /path/to/mri --scale 2 --checkpoint srcnn_x2.pt --model srcnn

The bicubic baseline always runs. CNN models are added when --checkpoint is given
(requires PyTorch; see train.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import list_volumes, make_pairs, split_subjects, upsample_bicubic
from metrics import all_metrics, summarize


def evaluate_methods(pairs, methods: dict):
    """pairs: iterable of (subject, hr, lr). methods: name -> f(lr) -> sr."""
    rows, examples = [], []
    for subject, hr, lr in pairs:
        row_base = {"subject": subject}
        for name, fn in methods.items():
            sr = fn(lr)
            m = all_metrics(hr, sr)
            rows.append({**row_base, "method": name, **m})
            if len(examples) < 64:  # keep a few for figures
                examples.append((name, subject, hr, lr, sr))
    return rows, examples


def save_table(rows, path_csv, path_json):
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "method", "psnr", "ssim", "rmse"])
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for name in sorted({r["method"] for r in rows}):
        sub = [{k: r[k] for k in ("psnr", "ssim", "rmse")} for r in rows if r["method"] == name]
        summary[name] = {k: {"mean": m, "std": s} for k, (m, s) in summarize(sub).items()} | {
            "n_slices": len(sub)
        }
    with open(path_json, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def comparison_figure(examples, scale, out_png, n_show=3):
    """Original | low-res (nearest, so pixels are visible) | each method | error map."""
    names = list(dict.fromkeys(e[0] for e in examples))
    by_method = {n: [e for e in examples if e[0] == n] for n in names}
    n_show = min(n_show, min(len(v) for v in by_method.values()))
    idx = np.linspace(0, min(len(v) for v in by_method.values()) - 1, n_show).astype(int)
    cols = 2 + len(names)
    fig, axes = plt.subplots(n_show, cols, figsize=(2.6 * cols, 2.6 * n_show))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idx):
        _, subj, hr, lr, _ = by_method[names[0]][i]
        lr_vis = np.kron(lr, np.ones((scale, scale), dtype=np.float32))  # nearest-neighbour view
        axes[r, 0].imshow(hr, cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(lr_vis, cmap="gray", vmin=0, vmax=1)
        if r == 0:
            axes[r, 0].set_title("Original (HR)", fontsize=9)
            axes[r, 1].set_title(f"Simulated LR (x{scale})", fontsize=9)
        for c, n in enumerate(names):
            _, _, hr_i, _, sr = by_method[n][i]
            axes[r, 2 + c].imshow(sr, cmap="gray", vmin=0, vmax=1)
            if r == 0:
                axes[r, 2 + c].set_title(n, fontsize=9)
            axes[r, 2 + c].set_xlabel(f"PSNR {all_metrics(hr_i, sr)['psnr']:.1f} dB", fontsize=8)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def error_map_figure(examples, out_png):
    names = list(dict.fromkeys(e[0] for e in examples))
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        _, _, hr, _, sr = next(e for e in examples if e[0] == n)
        im = ax.imshow(np.abs(hr - sr), cmap="magma", vmin=0, vmax=0.3)
        ax.set_title(f"|error|  {n}", fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="absolute error (normalized intensity)")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_cnn_method(model_name, checkpoint, scale):
    import torch  # imported lazily so the baseline runs without PyTorch

    from models import build_model

    model = build_model(model_name)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    def run(lr):
        x = torch.from_numpy(upsample_bicubic(lr, scale))[None, None]
        with torch.no_grad():
            y = model(x)[0, 0].numpy()
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--mode", default="area", choices=["area", "kspace"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-slices", type=int, default=20, help="per subject")
    ap.add_argument("--model", choices=["srcnn", "vdsr"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--extra", nargs=2, action="append", metavar=("MODEL", "CKPT"),
                    help="additional CNN, e.g. --extra vdsr vdsr_x2.pt")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = list_volumes(a.data)
    split = split_subjects(files, seed=a.seed)
    print({k: len(v) for k, v in split.items()}, "subjects per split")

    methods = {"Bicubic": lambda lr: upsample_bicubic(lr, a.scale)}
    if a.model and a.checkpoint:
        methods[a.model.upper()] = build_cnn_method(a.model, a.checkpoint, a.scale)
    for m, ck in a.extra or []:
        methods[m.upper()] = build_cnn_method(m, ck, a.scale)

    pairs = make_pairs(split["test"], a.scale, a.channel, a.mode, a.max_slices)
    rows, examples = evaluate_methods(pairs, methods)
    summary = save_table(rows, f"{a.out}/metrics_x{a.scale}.csv", f"{a.out}/summary_x{a.scale}.json")
    comparison_figure(examples, a.scale, f"{a.out}/comparison_x{a.scale}.png")
    error_map_figure(examples, f"{a.out}/errors_x{a.scale}.png")
    for name, s in summary.items():
        print(f"{name:8s} PSNR {s['psnr']['mean']:.2f}±{s['psnr']['std']:.2f} dB  "
              f"SSIM {s['ssim']['mean']:.4f}±{s['ssim']['std']:.4f}  "
              f"RMSE {s['rmse']['mean']:.4f}  (n={s['n_slices']})")


if __name__ == "__main__":
    main()
