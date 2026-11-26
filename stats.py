"""Subject-level comparison of methods from the metrics CSV written by evaluate.py.

Why: the +/- in evaluate.py is spread across SLICES, and slices of one brain are very
similar. Here every subject is averaged first, so each test subject counts once, and
methods are compared on the SAME subjects (paired).

Usage:
    python stats.py --csv results/metrics_x2.csv
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "rmse": False}


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """95% percentile bootstrap CI of the mean paired difference (resampling subjects)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(per_subject: pd.DataFrame, a: str, b: str, metric: str) -> dict:
    """Compare method `a` against method `b` on `metric` (positive diff = a is better)."""
    wide = per_subject.pivot(index="subject", columns="method", values=metric).dropna()
    diff = (wide[a] - wide[b]).to_numpy()
    if not HIGHER_IS_BETTER[metric]:
        diff = -diff  # lower RMSE is better, so flip the sign
    lo, hi = bootstrap_ci(diff)
    try:
        p = float(wilcoxon(diff).pvalue)
    except ValueError:  # all differences zero
        p = float("nan")
    return {
        "comparison": f"{a} vs {b}",
        "metric": metric,
        "n_subjects": len(diff),
        "mean_improvement": float(diff.mean()),
        "ci95_low": lo,
        "ci95_high": hi,
        "subjects_improved_%": float(100 * (diff > 0).mean()),
        "wilcoxon_p": p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="metrics_x*.csv from evaluate.py")
    ap.add_argument("--baseline", default="Bicubic")
    ap.add_argument("--out", default=None, help="where to write the table (default: next to the CSV)")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    per_subject = df.groupby(["subject", "method"], as_index=False)[["psnr", "ssim", "rmse"]].mean()
    methods = list(dict.fromkeys(df["method"]))
    others = [m for m in methods if m != a.baseline]

    pairs = [(m, a.baseline) for m in others]
    pairs += [(x, y) for x, y in itertools.permutations(others, 2) if methods.index(x) > methods.index(y)]

    rows = [compare(per_subject, x, y, m) for x, y in pairs for m in HIGHER_IS_BETTER]
    table = pd.DataFrame(rows)

    # Subject-level means (each subject counted once)
    means = per_subject.groupby("method")[["psnr", "ssim", "rmse"]].agg(["mean", "std"]).round(4)
    print(f"Subject-level means (n = {per_subject['subject'].nunique()} subjects):")
    print(means.to_string(), "\n")

    pd.set_option("display.width", 200)
    shown = table.round(4)
    shown["wilcoxon_p"] = table["wilcoxon_p"].map(lambda p: "<0.0001" if p < 1e-4 else f"{p:.4f}")
    print("Paired comparisons (positive improvement = first method is better):")
    print(shown.to_string(index=False))

    out = a.out or a.csv.replace("metrics_", "stats_")
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("Note: a small p-value says the difference is consistent across subjects. It does NOT\n"
          "say the improvement matters clinically, and this is one training run with one seed.")


if __name__ == "__main__":
    main()
