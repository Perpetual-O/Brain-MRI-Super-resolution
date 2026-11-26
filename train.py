"""Train SRCNN or VDSR on TRAIN subjects, pick the best epoch on VAL subjects.

Usage:
    python train.py --data /path/to/mri --model srcnn --scale 2 --epochs 30
    python train.py --data /path/to/mri --model vdsr  --scale 2 --epochs 30

Writes <model>_x<scale>.pt (best validation PSNR) and <model>_x<scale>_history.csv.
The test subjects are never touched here; evaluate.py uses them.
"""
from __future__ import annotations

import argparse
import csv
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data import list_volumes, make_pairs, split_subjects, upsample_bicubic
from metrics import psnr
from models import build_model


class SlicePairs(Dataset):
    """Pre-computed (bicubic-upsampled LR, HR) patches/slices held in memory."""

    def __init__(self, pairs, scale, patch: int | None, augment: bool):
        self.items = [(upsample_bicubic(lr, scale), hr) for _, hr, lr in pairs]
        self.patch = patch
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y = self.items[i]
        if self.patch:
            h, w = y.shape
            p = min(self.patch, h, w)
            t, l = random.randint(0, h - p), random.randint(0, w - p)
            x, y = x[t : t + p, l : l + p], y[t : t + p, l : l + p]
        if self.augment:
            if random.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            if random.random() < 0.5:
                x, y = x[::-1], y[::-1]
        return (torch.from_numpy(np.ascontiguousarray(x))[None],
                torch.from_numpy(np.ascontiguousarray(y))[None])


@torch.no_grad()
def val_psnr(model, ds, device):
    model.eval()
    scores = []
    for x, y in DataLoader(ds, batch_size=1):
        out = model(x.to(device)).clamp(0, 1)[0, 0].cpu().numpy()
        scores.append(psnr(y[0, 0].numpy(), out))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["srcnn", "vdsr"], required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--mode", default="area", choices=["area", "kspace"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--max-slices", type=int, default=20, help="per subject")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split = split_subjects(list_volumes(a.data), seed=a.seed)
    print({k: len(v) for k, v in split.items()}, "subjects per split | device:", device)
    mk = lambda files: list(make_pairs(files, a.scale, a.channel, a.mode, a.max_slices))  # noqa: E731
    train_ds = SlicePairs(mk(split["train"]), a.scale, a.patch, augment=True)
    val_ds = SlicePairs(mk(split["val"]), a.scale, patch=None, augment=False)
    print(f"{len(train_ds)} train slices, {len(val_ds)} val slices")

    model = build_model(a.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, a.epochs // 3), gamma=0.3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True)

    # baseline for reference: PSNR of plain bicubic on the validation set
    bic = float(np.mean([psnr(y, x) for x, y in val_ds.items]))
    print(f"validation PSNR of bicubic alone: {bic:.2f} dB")

    best, tag = -1.0, f"{a.model}_x{a.scale}"
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.4)  # VDSR-style stabiliser
            opt.step()
            total += loss.item() * len(x)
        sched.step()
        v = val_psnr(model, val_ds, device)
        hist.append((ep, total / len(train_ds), v))
        flag = ""
        if v > best:
            best = v
            torch.save(model.state_dict(), f"{tag}.pt")
            flag = "  *saved"
        print(f"epoch {ep:3d}  train MSE {total / len(train_ds):.5f}  val PSNR {v:.2f} dB{flag}")

    with open(f"{tag}_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "val_psnr"])
        w.writerows(hist)
    print(f"best val PSNR {best:.2f} dB (bicubic {bic:.2f} dB) -> {tag}.pt")


if __name__ == "__main__":
    main()
