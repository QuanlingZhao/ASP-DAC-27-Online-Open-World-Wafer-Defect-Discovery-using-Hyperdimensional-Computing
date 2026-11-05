"""
mri_features.py - MRI replacements for Eric's 202 wafer descriptors.

Every function takes images of shape (N, C, S, S) and returns (N, D) float32.
The HDC encoder downstream (BinaryThermometerHDC) does not care what D is,
so you can swap feature sets freely and compare them.

  pixels  : each channel average-pooled to 16 x 16 and flattened (C*256 numbers)
  stats   : intensity, edge, frequency and texture statistics per channel (44 per channel, 132 for 3 channels)
  resnet  : frozen ImageNet ResNet-18 embedding (512 numbers); T2W/ADC/HBV used as RGB
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def pixels(images: np.ndarray, pool: int = 16) -> np.ndarray:
    n, c, s, _ = images.shape
    k = s // pool
    x = images[:, :, : k * pool, : k * pool].astype(np.float32)
    x = x.reshape(n, c, pool, k, pool, k).mean(axis=(3, 5))
    return x.reshape(n, -1)


def _channel_stats(x: np.ndarray) -> list:
    f = []
    f += list(np.percentile(x, [5, 25, 50, 75, 95]))
    f += [x.mean(), x.std(), float(stats.skew(x.ravel())), float(stats.kurtosis(x.ravel()))]
    # edges and sharpness: blur lowers these, noise raises them
    gy, gx = np.gradient(x)
    g = np.hypot(gx, gy)
    f += [g.mean(), g.std(), float(ndimage.laplace(x).var())]
    # share of energy at high spatial frequency: noise and ghosting raise it, blur lowers it
    F = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    s = x.shape[0]
    yy, xx = np.indices(F.shape)
    r = np.hypot(yy - s / 2, xx - s / 2) / (s / 2)
    f += [F[r > 0.5].sum() / (F.sum() + 1e-8), F[(r > 0.25) & (r <= 0.5)].sum() / (F.sum() + 1e-8)]
    # texture: grey-level co-occurrence on 32 levels
    q = np.clip((x - x.min()) / (np.ptp(x) + 1e-6) * 31, 0, 31).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 2], angles=[0, np.pi / 2], levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        f.append(float(np.nan_to_num(graycoprops(glcm, prop).mean())))
    # texture: uniform local binary patterns (10-bin histogram)
    lbp = local_binary_pattern(q, P=8, R=1, method="uniform")
    f += list(np.histogram(lbp, bins=10, range=(0, 10), density=True)[0])
    # coarse layout: 4 x 4 grid of means, so a dark blob in one corner shows up
    k = s // 4
    f += list(x[: 4 * k, : 4 * k].reshape(4, k, 4, k).mean(axis=(1, 3)).ravel())
    return f


def handcrafted_stats(images: np.ndarray) -> np.ndarray:
    out = []
    for img in images.astype(np.float32):
        feats = []
        for ch in img:
            feats += _channel_stats(ch)
        out.append(feats)
    return np.nan_to_num(np.asarray(out, dtype=np.float32))


def resnet(images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    x_all = images.astype(np.float32)
    if x_all.shape[1] == 1:
        x_all = np.repeat(x_all, 3, axis=1)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x_all), batch_size):
            x = torch.from_numpy(x_all[i : i + batch_size, :3])
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            outs.append(model(x).numpy())
    return np.concatenate(outs).astype(np.float32)


FEATURE_SETS = {
    "pixels": pixels,
    "stats": handcrafted_stats,
    "resnet": resnet,
    "stats+pixels": lambda im: np.concatenate([handcrafted_stats(im), pixels(im)], axis=1),
}


def extract(images: np.ndarray, feature_set: str) -> np.ndarray:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}")
    return FEATURE_SETS[feature_set](images)
