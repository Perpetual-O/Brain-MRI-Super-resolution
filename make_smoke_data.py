"""Create a small SYNTHETIC dataset (perturbed Shepp-Logan phantoms) to test the code.

This is NOT MRI and results on it mean nothing scientifically. It only checks that
loading, splitting, degradation, metrics and figures run end to end.
"""
import os
import sys

import cv2
import numpy as np
from skimage.data import shepp_logan_phantom

out = sys.argv[1] if len(sys.argv) > 1 else "smoke_data"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
base = cv2.resize(shepp_logan_phantom().astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)

for s in range(8):  # 8 fake subjects
    slices = []
    for z in range(12):
        zoom = rng.uniform(0.9, 1.05)
        M = cv2.getRotationMatrix2D((128, 128), rng.uniform(-8, 8), zoom)
        img = cv2.warpAffine(base, M, (256, 256))
        tex = cv2.GaussianBlur(rng.normal(0, 1, (256, 256)).astype(np.float32), (0, 0), 3)
        img = np.clip(img * (1 + 0.15 * tex) + rng.normal(0, 0.01, img.shape), 0, None)
        slices.append(img.astype(np.float32))
    np.save(os.path.join(out, f"subject_{s:02d}.npy"), np.stack(slices))
print("wrote", len(os.listdir(out)), "files to", out)
