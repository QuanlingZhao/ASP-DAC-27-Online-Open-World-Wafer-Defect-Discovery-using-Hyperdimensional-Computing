"""
mri_artifacts.py - synthetic low-quality versions of a slice, for the image-quality test.

Each function takes one slice (C, S, S) float32 (already z-scored) and returns a corrupted copy.
The artifact types become the "unknown" classes, so Stage 2 should discover one cluster per type.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def noise(x, rng):  # thermal noise / low SNR
    return x + rng.normal(0, rng.uniform(0.4, 0.8), x.shape)


def blur(x, rng):  # low resolution or through-plane partial volume
    s = rng.uniform(1.2, 2.0)
    return np.stack([ndimage.gaussian_filter(ch, s) for ch in x])


def ghosting(x, rng):  # motion: phase errors on k-space lines give shifted copies along one axis
    out = []
    for ch in x:
        k = np.fft.fft2(ch)
        rows = np.arange(k.shape[0])
        phase = np.where(rows % rng.integers(2, 5) == 0, rng.uniform(0.8, 1.6), 0.0)
        out.append(np.real(np.fft.ifft2(k * np.exp(1j * phase)[:, None])))
    return np.stack(out)


def bias_field(x, rng):  # coil inhomogeneity: smooth multiplicative shading
    s = x.shape[-1]
    yy, xx = np.mgrid[0:s, 0:s] / s
    a, b = rng.uniform(-1, 1, 2)
    field = np.exp(0.8 * (a * (yy - 0.5) + b * (xx - 0.5)) + 0.6 * ((yy - 0.5) ** 2 + (xx - 0.5) ** 2))
    return (x - x.min()) * field + x.min()


ARTIFACTS = {"noise": noise, "blur": blur, "ghosting": ghosting, "bias_field": bias_field}
