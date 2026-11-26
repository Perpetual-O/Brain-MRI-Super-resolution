# Machine Learning-Based Super-Resolution of Brain MRI Images

ENGG 680 course project. Compares a conventional baseline (bicubic interpolation) with two CNNs
(SRCNN and a VDSR-style residual network) at reconstructing high-resolution MRI slices from
**simulated** low-resolution versions.

> Research/education only. Nothing here is validated for diagnosis or patient care.

## Pipeline

    high-res slice --degrade (downsample x2/x4)--> low-res --bicubic up--> CNN --> reconstruction
                          compare reconstruction with the original: PSNR, SSIM, RMSE

| File | Purpose |
|---|---|
| `data.py` | Load NIfTI / fastMRI HDF5 / .npy, pick slices, normalize, degrade, **subject-level** train/val/test split |
| `metrics.py` | PSNR, SSIM, RMSE (mean ± std across slices) |
| `models.py` | `SRCNN`, `VDSR` (PyTorch) |
| `train.py` | Train on TRAIN subjects, keep best epoch on VAL subjects |
| `evaluate.py` | Evaluate on TEST subjects only; writes CSV/JSON tables and comparison figures |
| `stats.py` | Subject-level comparison from the metrics CSV: mean improvement, 95% bootstrap CI, % of subjects improved, Wilcoxon p |
| `crops.py` | Zoomed-crop figure of a detailed region, original vs bicubic vs CNNs |
| `make_smoke_data.py` | Synthetic phantom data to check the code runs (not MRI, not a result) |

## Setup

    pip install -r requirements.txt

## Run

    # 1. baseline only (no PyTorch needed)
    python evaluate.py --data /path/to/mri --scale 2 --channel 0

    # 2. train the CNNs (PyTorch; a GPU such as Colab's free T4 helps a lot)
    python train.py --data /path/to/mri --model srcnn --scale 2 --epochs 30
    python train.py --data /path/to/mri --model vdsr  --scale 2 --epochs 30

    # 3. compare everything on the held-out test subjects
    python evaluate.py --data /path/to/mri --scale 2 --model srcnn --checkpoint srcnn_x2.pt \
                       --extra vdsr vdsr_x2.pt

    # 4. subject-level statistics and a zoomed-crop figure (run after step 3)
    python stats.py --csv results/metrics_x2.csv
    python crops.py --data /path/to/mri --scale 2 --model srcnn --checkpoint srcnn_x2.pt \
                    --extra vdsr vdsr_x2.pt

Dataset notes
* **MSD Task01 (NIfTI)**: files are 4D; `--channel` picks the modality (check `dataset.json`; commonly 0=FLAIR, 1=T1w, 2=T1gd, 3=T2w). Use the `imagesTr` files only, not the `labelsTr` masks.
* **fastMRI (HDF5)**: raw k-space; `data.py` reconstructs with a root-sum-of-squares step. Keep the data out of the Git repo (the data-sharing agreement forbids redistribution) and cite the fastMRI papers.
* Report the scale factor, modality, number of subjects per split, and `--mode` (`area` or `kspace`) used.

## Honest limitations to state in the report
* Low-resolution inputs are simulated, not real fast or low-field acquisitions.
* PSNR/SSIM measure pixel similarity, not diagnostic value or anatomical correctness.
* MSD scans are glioma patients and skull-stripped; IXI would be healthy subjects.
* Subject-level splitting is used to avoid leakage; keep it if you change the code.
