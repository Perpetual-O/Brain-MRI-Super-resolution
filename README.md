# Machine Learning-Based Super-Resolution of Brain MRI Images

ENGG 680 (University of Calgary, Fall 2026) course project. Compares a conventional baseline
(bicubic interpolation) with two CNNs (SRCNN and a 10-layer VDSR-style residual network) at
reconstructing high-resolution MRI slices from **simulated** 2x low-resolution versions.

> Research/education only. Nothing here is validated for diagnosis or patient care.

## Results (IXI T1, 2x, 87 held-out test subjects, mean over subjects)

| Method | PSNR (dB) | SSIM | RMSE |
|---|---|---|---|
| Bicubic | 29.62 | 0.9467 | 0.0338 |
| SRCNN | 31.29 | 0.9595 | 0.0280 |
| VDSR-style | 31.96 | 0.9650 | 0.0259 |

Both CNNs improved on bicubic for all 87 test subjects (paired bootstrap 95% CIs and
Wilcoxon tests are produced by `stats.py`).

## Pipeline

    high-res slice --degrade (area downsample x2)--> low-res --bicubic up--> CNN --> reconstruction
                      compare reconstruction with the original: PSNR, SSIM, RMSE

| File | Purpose |
|---|---|
| `data.py` | Load NIfTI / fastMRI HDF5 / .npy, pick slices, normalize, degrade, **subject-level** train/val/test split |
| `metrics.py` | PSNR, SSIM, RMSE (mean ± std across slices) |
| `models.py` | `SRCNN`, `VDSR` (PyTorch) |
| `train.py` | Train on TRAIN subjects, keep best epoch on VAL subjects |
| `evaluate.py` | Evaluate on TEST subjects only; writes CSV/JSON tables and comparison figures |
| `stats.py` | Subject-level comparison: mean improvement, 95% bootstrap CI, % of subjects improved, Wilcoxon p |
| `crops.py` | Zoomed-crop figure of a detailed region, original vs bicubic vs CNNs |
| `make_smoke_data.py` | Synthetic phantom data to check the code runs (not MRI, not a result) |

## Data

We used the IXI T1 dataset (581 volumes, healthy adults, CC BY-SA 3.0):
https://brain-development.org/ixi-dataset/

The data are **not** included in this repository. Download the T1 archive, extract it, and
pass the folder of `.nii.gz` files as `--data`. The scans are not skull-stripped.
Subjects are split 70/15/15 (407/87/87) with seed 42, and all slices of a subject stay in one split.

## Setup

    pip install -r requirements.txt

## Run

    # DATA = folder of IXI T1 .nii.gz files
    python train.py --data DATA --model srcnn --scale 2 --epochs 10 --max-slices 10
    python train.py --data DATA --model vdsr  --scale 2 --epochs 10 --max-slices 10

    python evaluate.py --data DATA --scale 2 --max-slices 10 --model srcnn \
                       --checkpoint srcnn_x2.pt --extra vdsr vdsr_x2.pt

    python stats.py --csv results/metrics_x2.csv
    python crops.py --data DATA --scale 2 --model srcnn --checkpoint srcnn_x2.pt \
                    --extra vdsr vdsr_x2.pt --crop 100 120

A GPU (for example Google Colab's free T4) is recommended for training.
The trained model files are not included; retrain with the commands above.

## Limitations

* Low-resolution inputs are simulated by downsampling, not real fast or low-field acquisitions.
* PSNR, SSIM and RMSE measure pixel similarity, not diagnostic value or anatomical correctness.
* IXI contains healthy adults only; the scans include skull and background.
* Each model was trained once (one seed, one split), at 2x only.

## Team

Perpetual Ogedegbe, Ahenkan Yeboah, Styve Yvan Chieudjui Wongoue

## AI assistance

Parts of the code were written with the help of an AI system (Claude, Anthropic). The team ran all experiments and checked the results.
