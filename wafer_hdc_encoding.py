#!/usr/bin/env python3
"""
wafer_real_encoding_single.py

Single self-contained encoding module for the real-valued Stage-1 wafer pipeline.

This file contains, in one place:
  - enhanced auxiliary feature extraction
  - dataset_to_arrays and auxiliary preprocessing utilities
  - original bipolar HDC helpers/classes for backward compatibility
  - real-valued auxiliary random projection encoder
  - real-valued vectorized multiscale spatial encoder
  - real-valued spatial+auxiliary fusion utilities

Use this file instead of combining wafer_hdc_encoding.py + wafer_real_encoding_fast.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from scipy import ndimage
from skimage.transform import radon
from skimage.morphology import skeletonize
from skimage.measure import perimeter, regionprops

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:  # older scikit-image spelling
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops


def safe_div(a, b, eps: float = 1e-8) -> float:
    return float(a) / float(b + eps)


def bipolar_sign(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0, -1.0).astype(np.float32)


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A @ B.T / A.shape[1]).astype(np.float32)


def parse_csv_ints(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def unpack_sample(sample: Any) -> Tuple[np.ndarray, int]:
    if isinstance(sample, dict):
        x = sample["image"]
        y = sample["label"]
    elif isinstance(sample, (tuple, list)):
        x = sample[0]
        y = sample[1]
    else:
        raise TypeError(f"Unsupported sample type: {type(sample)}")

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()

    x = np.asarray(x)
    y = int(y)

    if x.ndim == 3:
        x = x[0]
    elif x.ndim != 2:
        raise ValueError(f"Expected 2D or 3D wafer map, got shape {x.shape}")

    return x.astype(np.float32), y


def _compute_centroid(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    return coords.mean(axis=0).astype(np.float32)


def _second_moments(coords: np.ndarray, center=None) -> np.ndarray:
    if len(coords) == 0:
        return np.zeros((2, 2), dtype=np.float32)
    if center is None:
        center = _compute_centroid(coords)
    x = coords - center[None, :]
    cov = (x.T @ x) / max(len(coords), 1)
    return cov.astype(np.float32)


def _radial_bins(rr, cc, center, num_bins, valid_mask):
    cy, cx = center
    dist = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2)
    valid_coords = np.argwhere(valid_mask)
    if len(valid_coords) == 0:
        return np.zeros_like(dist, dtype=np.int32), 1.0
    valid_dist = np.sqrt((valid_coords[:, 0] - cy) ** 2 + (valid_coords[:, 1] - cx) ** 2)
    max_r = valid_dist.max() + 1e-8
    dist_norm = dist / max_r
    bins = np.clip((dist_norm * num_bins).astype(np.int32), 0, num_bins - 1)
    return bins, max_r


def _get_valid_and_defect_masks(
    wafer_map: np.ndarray,
    defect_threshold: float = 0.5,
    invalid_value: float = -1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    wafer_map = np.asarray(wafer_map, dtype=np.float32)
    valid_mask = wafer_map != invalid_value
    defect_mask = (wafer_map > defect_threshold) & valid_mask
    return valid_mask, defect_mask


def _safe_region_density(defect_mask: np.ndarray, region: np.ndarray) -> float:
    return safe_div((defect_mask & region).sum(), region.sum())


def _projection_stats(values: np.ndarray, prefix: str, feat: Dict[str, float]):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        for suffix in ["max", "mean", "std", "peak_to_mean", "nonzero_frac"]:
            feat[f"{prefix}_{suffix}"] = 0.0
        return
    mean = float(values.mean())
    mx = float(values.max())
    feat[f"{prefix}_max"] = mx
    feat[f"{prefix}_mean"] = mean
    feat[f"{prefix}_std"] = float(values.std())
    feat[f"{prefix}_peak_to_mean"] = safe_div(mx, mean)
    feat[f"{prefix}_nonzero_frac"] = float(np.mean(values > 0))


def _run_lengths_binary(arr: np.ndarray) -> List[int]:
    arr = np.asarray(arr, dtype=bool)
    runs = []
    cur = 0
    for v in arr:
        if v:
            cur += 1
        elif cur > 0:
            runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


def _circular_longest_run(arr: np.ndarray) -> int:
    arr = np.asarray(arr, dtype=bool)
    n = len(arr)
    if n == 0 or not arr.any():
        return 0
    if arr.all():
        return n
    doubled = np.concatenate([arr, arr])
    best = 0
    cur = 0
    for v in doubled:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(min(best, n))


def _entropy_from_values(values: np.ndarray, bins: int = 16) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin + 1e-12:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(vmin, vmax), density=False)
    p = hist.astype(np.float64)
    p = p / max(p.sum(), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def _count_1d_local_peaks(values: np.ndarray) -> int:
    v = np.asarray(values, dtype=np.float32)
    if v.size < 3:
        return 0
    return int(np.sum((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:])))


def add_radon_profile_features(feat: Dict[str, float], defect_mask: np.ndarray, n_angles: int = 180, n_profile: int = 20):
    for i in range(n_profile):
        feat[f"radon_profile_{i}"] = 0.0
    if n_profile <= 0 or defect_mask.size == 0 or not np.any(defect_mask):
        return
    try:
        theta = np.linspace(0.0, 180.0, int(n_angles), endpoint=False)
        sinogram = radon(defect_mask.astype(np.float32), theta=theta, circle=False)
        if sinogram.size == 0:
            return
        g_mu = sinogram.mean(axis=1).astype(np.float32)
        x_old = np.linspace(0.0, 1.0, len(g_mu))
        x_new = np.linspace(0.0, 1.0, n_profile)
        profile = np.interp(x_new, x_old, g_mu).astype(np.float32)
        profile = profile / (float(np.max(np.abs(profile))) + 1e-8)
        for i, v in enumerate(profile):
            feat[f"radon_profile_{i}"] = float(v)
    except Exception:
        return


def add_glcm_texture_features(feat: Dict[str, float], defect_mask: np.ndarray):
    props = ["energy", "contrast", "correlation", "homogeneity"]
    for prop in props:
        feat[f"glcm_{prop}_mean"] = 0.0
        feat[f"glcm_{prop}_std"] = 0.0
    feat["glcm_entropy_mean"] = 0.0
    feat["glcm_entropy_std"] = 0.0

    binary = defect_mask.astype(np.uint8)
    if binary.size == 0 or binary.max() == 0:
        return
    try:
        distances = [1, 2, 4]
        angles = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
        glcm = graycomatrix(binary, distances=distances, angles=angles, levels=2, symmetric=True, normed=True)
        for prop in props:
            vals = graycoprops(glcm, prop).astype(np.float64).ravel()
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            feat[f"glcm_{prop}_mean"] = float(vals.mean()) if vals.size else 0.0
            feat[f"glcm_{prop}_std"] = float(vals.std()) if vals.size else 0.0

        entropies = []
        for di in range(glcm.shape[2]):
            for ai in range(glcm.shape[3]):
                p = glcm[:, :, di, ai].astype(np.float64).ravel()
                p = p[p > 0]
                entropies.append(float(-np.sum(p * np.log(p + 1e-12))))
        entropies = np.asarray(entropies, dtype=np.float32)
        feat["glcm_entropy_mean"] = float(entropies.mean()) if entropies.size else 0.0
        feat["glcm_entropy_std"] = float(entropies.std()) if entropies.size else 0.0
    except Exception:
        return


def add_local_density_features(feat: Dict[str, float], defect_mask: np.ndarray, valid_mask: np.ndarray, window_sizes=(3, 5, 7)):
    defect_float = defect_mask.astype(np.float32)
    valid_float = valid_mask.astype(np.float32)
    for ws in window_sizes:
        prefix = f"local_density{int(ws)}"
        for suffix in ["mean", "std", "max", "p95", "entropy", "active_frac", "peak_to_mean"]:
            feat[f"{prefix}_{suffix}"] = 0.0
        if defect_mask.size == 0 or not np.any(valid_mask):
            continue
        kernel = np.ones((int(ws), int(ws)), dtype=np.float32)
        local_count = ndimage.convolve(defect_float, kernel, mode="constant", cval=0.0)
        local_valid = ndimage.convolve(valid_float, kernel, mode="constant", cval=0.0)
        density = local_count / np.maximum(local_valid, 1.0)
        vals = density[valid_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        mx = float(vals.max())
        feat[f"{prefix}_mean"] = mean
        feat[f"{prefix}_std"] = float(vals.std())
        feat[f"{prefix}_max"] = mx
        feat[f"{prefix}_p95"] = float(np.percentile(vals, 95))
        feat[f"{prefix}_entropy"] = _entropy_from_values(vals, bins=16)
        feat[f"{prefix}_active_frac"] = float(np.mean(vals > 0))
        feat[f"{prefix}_peak_to_mean"] = safe_div(mx, mean)


def _rp_value(region, new_name: str, old_name: str, default: float = 0.0) -> float:
    if hasattr(region, new_name):
        return float(getattr(region, new_name))
    return float(getattr(region, old_name, default))


def add_regionprops_extension_features(feat: Dict[str, float], labeled: np.ndarray, defect_count: int, valid_count: int, wafer_shape):
    keys = [
        "largest_component_solidity", "largest_component_extent", "largest_component_equiv_diameter_norm",
        "largest_component_convex_area_frac", "largest_component_filled_area_frac",
        "largest_to_second_component_area_ratio", "component_area_entropy", "component_area_cv",
        "component_centroid_dispersion", "component_solidity_area_weighted", "component_extent_area_weighted",
    ]
    for k in keys:
        feat[k] = 0.0
    for i in range(3):
        feat[f"top{i+1}_component_solidity"] = 0.0
        feat[f"top{i+1}_component_extent"] = 0.0
        feat[f"top{i+1}_component_convex_area_frac"] = 0.0

    if defect_count <= 0:
        return
    try:
        regions = regionprops(labeled)
    except Exception:
        return
    if len(regions) == 0:
        return

    h, w = wafer_shape
    diag = float(np.sqrt(h * h + w * w)) + 1e-8
    regs = sorted(regions, key=lambda r: float(r.area), reverse=True)
    areas = np.asarray([float(r.area) for r in regs], dtype=np.float32)
    weights = areas / max(float(areas.sum()), 1.0)

    largest = regs[0]
    feat["largest_component_solidity"] = float(getattr(largest, "solidity", 0.0))
    feat["largest_component_extent"] = float(getattr(largest, "extent", 0.0))
    feat["largest_component_equiv_diameter_norm"] = safe_div(_rp_value(largest, "equivalent_diameter_area", "equivalent_diameter"), diag)
    feat["largest_component_convex_area_frac"] = safe_div(_rp_value(largest, "area_convex", "convex_area"), valid_count)
    feat["largest_component_filled_area_frac"] = safe_div(_rp_value(largest, "area_filled", "filled_area"), valid_count)
    feat["largest_to_second_component_area_ratio"] = safe_div(areas[0], areas[1] if len(areas) > 1 else 0.0)

    area_p = weights[weights > 0]
    feat["component_area_entropy"] = float(-np.sum(area_p * np.log(area_p + 1e-12))) if area_p.size else 0.0
    feat["component_area_cv"] = safe_div(float(areas.std()), float(areas.mean()) if areas.size else 0.0)

    centroids = np.asarray([r.centroid for r in regs], dtype=np.float32)
    if len(centroids) > 1:
        center = np.sum(centroids * weights[:, None], axis=0)
        d = np.sqrt(np.sum((centroids - center[None, :]) ** 2, axis=1)) / diag
        feat["component_centroid_dispersion"] = float(np.sum(d * weights))

    solidities = np.asarray([float(getattr(r, "solidity", 0.0)) for r in regs], dtype=np.float32)
    extents = np.asarray([float(getattr(r, "extent", 0.0)) for r in regs], dtype=np.float32)
    feat["component_solidity_area_weighted"] = float(np.sum(solidities * weights))
    feat["component_extent_area_weighted"] = float(np.sum(extents * weights))

    for i in range(3):
        if i < len(regs):
            r = regs[i]
            feat[f"top{i+1}_component_solidity"] = float(getattr(r, "solidity", 0.0))
            feat[f"top{i+1}_component_extent"] = float(getattr(r, "extent", 0.0))
            feat[f"top{i+1}_component_convex_area_frac"] = safe_div(_rp_value(r, "area_convex", "convex_area"), valid_count)


def add_donut_hollow_features(feat: Dict[str, float], defect_mask: np.ndarray, valid_mask: np.ndarray, dist_norm: np.ndarray):
    radial_edges = np.asarray([0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.01], dtype=np.float32)
    densities = []
    for i in range(len(radial_edges) - 1):
        region = (dist_norm >= radial_edges[i]) & (dist_norm < radial_edges[i + 1]) & valid_mask
        d = _safe_region_density(defect_mask, region)
        densities.append(d)
        feat[f"hollow_radial_density_{i}"] = d
    densities = np.asarray(densities, dtype=np.float32)
    center_density = float(np.mean(densities[:2])) if densities.size else 0.0
    mid_ring_density = float(np.max(densities[2:5])) if len(densities) >= 5 else (float(np.max(densities)) if densities.size else 0.0)
    edge_density = float(np.mean(densities[-2:])) if len(densities) >= 2 else 0.0
    feat["donut_hole_score"] = mid_ring_density * safe_div(mid_ring_density, center_density) * (1.0 - min(center_density, 1.0))
    feat["donut_mid_to_center_ratio"] = safe_div(mid_ring_density, center_density)
    feat["donut_mid_to_edge_ratio"] = safe_div(mid_ring_density, edge_density)
    p = densities / (float(densities.sum()) + 1e-8)
    p = p[p > 0]
    feat["radial_profile_entropy"] = float(-np.sum(p * np.log(p + 1e-12))) if p.size else 0.0
    feat["radial_profile_num_peaks"] = float(_count_1d_local_peaks(densities))
    feat["radial_profile_peak_bin"] = float(int(np.argmax(densities))) if densities.size else 0.0


def add_defect_sensitive_features(feat, wafer_map, valid_mask, defect_mask, rr, cc, wafer_center, dist_norm, theta, labeled, num_components, defect_count, valid_count, radon_thetas):
    eps = 1e-8

    radial_edges = [0.0, 0.20, 0.40, 0.60, 0.75, 0.85, 0.92, 1.01]
    radial_densities = []
    for i in range(len(radial_edges) - 1):
        lo, hi = radial_edges[i], radial_edges[i + 1]
        region = (dist_norm >= lo) & (dist_norm < hi) & valid_mask
        d = _safe_region_density(defect_mask, region)
        radial_densities.append(d)
        feat[f"fine_radial_density_{i}"] = d

    inner_density = float(np.mean(radial_densities[:3])) if len(radial_densities) >= 3 else 0.0
    outer_density = float(np.mean(radial_densities[-3:])) if len(radial_densities) >= 3 else 0.0
    feat["outer_minus_inner_density"] = outer_density - inner_density
    feat["outer_to_inner_density_ratio"] = safe_div(outer_density, inner_density)
    feat["outer_edge_density_max"] = float(np.max(radial_densities[-3:])) if len(radial_densities) >= 3 else 0.0
    feat["radial_density_std_fine"] = float(np.std(radial_densities)) if radial_densities else 0.0
    feat["radial_density_peak_to_mean_fine"] = safe_div(np.max(radial_densities) if radial_densities else 0.0,
                                                       np.mean(radial_densities) if radial_densities else 0.0)

    edge90 = (dist_norm >= 0.90) & valid_mask
    edge85 = (dist_norm >= 0.85) & valid_mask
    edge75_90 = (dist_norm >= 0.75) & (dist_norm < 0.90) & valid_mask
    mid40_75 = (dist_norm >= 0.40) & (dist_norm < 0.75) & valid_mask
    feat["edge90_density"] = _safe_region_density(defect_mask, edge90)
    feat["edge85_density"] = _safe_region_density(defect_mask, edge85)
    feat["edge75_90_density"] = _safe_region_density(defect_mask, edge75_90)
    feat["edge90_minus_mid_density"] = feat["edge90_density"] - _safe_region_density(defect_mask, mid40_75)
    feat["edge85_to_mid_density_ratio"] = safe_div(feat["edge85_density"], _safe_region_density(defect_mask, mid40_75))

    num_edge_sectors = 32
    sector_ids_32 = np.floor(theta / (2 * np.pi / num_edge_sectors)).astype(int)
    edge_sector_density = []
    edge_sector_hit = []
    for s in range(num_edge_sectors):
        region = (sector_ids_32 == s) & edge85
        d = _safe_region_density(defect_mask, region)
        edge_sector_density.append(d)
        edge_sector_hit.append((defect_mask & region).sum() > 0)

    edge_sector_density = np.asarray(edge_sector_density, dtype=np.float32)
    edge_sector_hit = np.asarray(edge_sector_hit, dtype=bool)
    feat["edge_angular_coverage_frac"] = float(edge_sector_hit.mean()) if edge_sector_hit.size else 0.0
    feat["edge_angular_longest_run_frac"] = safe_div(_circular_longest_run(edge_sector_hit), num_edge_sectors)
    feat["edge_sector_density_mean"] = float(edge_sector_density.mean()) if edge_sector_density.size else 0.0
    feat["edge_sector_density_std"] = float(edge_sector_density.std()) if edge_sector_density.size else 0.0
    feat["edge_sector_density_max"] = float(edge_sector_density.max()) if edge_sector_density.size else 0.0
    feat["edge_sector_peak_to_mean"] = safe_div(feat["edge_sector_density_max"], feat["edge_sector_density_mean"])
    feat["edge_ring_continuity_score"] = feat["edge85_density"] * feat["edge_angular_coverage_frac"] * feat["edge_angular_longest_run_frac"]

    grid_radial_edges = [0.0, 0.40, 0.70, 0.85, 1.01]
    num_grid_sectors = 16
    sector_ids_16 = np.floor(theta / (2 * np.pi / num_grid_sectors)).astype(int)
    grid_densities = []
    for rb in range(len(grid_radial_edges) - 1):
        rlo, rhi = grid_radial_edges[rb], grid_radial_edges[rb + 1]
        for s in range(num_grid_sectors):
            region = (dist_norm >= rlo) & (dist_norm < rhi) & (sector_ids_16 == s) & valid_mask
            grid_densities.append(_safe_region_density(defect_mask, region))
    grid_densities = np.asarray(grid_densities, dtype=np.float32)
    feat["radial_sector_grid_max_density"] = float(grid_densities.max()) if grid_densities.size else 0.0
    feat["radial_sector_grid_mean_density"] = float(grid_densities.mean()) if grid_densities.size else 0.0
    feat["radial_sector_grid_std_density"] = float(grid_densities.std()) if grid_densities.size else 0.0
    feat["radial_sector_grid_peak_to_mean"] = safe_div(feat["radial_sector_grid_max_density"], feat["radial_sector_grid_mean_density"])
    feat["radial_sector_grid_active_frac"] = float(np.mean(grid_densities > 0)) if grid_densities.size else 0.0

    row_counts = defect_mask.sum(axis=1).astype(np.float32)
    col_counts = defect_mask.sum(axis=0).astype(np.float32)
    valid_row_counts = valid_mask.sum(axis=1).astype(np.float32)
    valid_col_counts = valid_mask.sum(axis=0).astype(np.float32)
    row_density = row_counts / np.maximum(valid_row_counts, 1.0)
    col_density = col_counts / np.maximum(valid_col_counts, 1.0)
    _projection_stats(row_density, "row_density_profile", feat)
    _projection_stats(col_density, "col_density_profile", feat)

    row_hit = row_counts > 0
    col_hit = col_counts > 0
    row_runs = _run_lengths_binary(row_hit)
    col_runs = _run_lengths_binary(col_hit)
    feat["row_hit_longest_run_frac"] = safe_div(max(row_runs) if row_runs else 0, len(row_hit))
    feat["col_hit_longest_run_frac"] = safe_div(max(col_runs) if col_runs else 0, len(col_hit))
    feat["row_hit_coverage_frac"] = float(np.mean(row_hit)) if row_hit.size else 0.0
    feat["col_hit_coverage_frac"] = float(np.mean(col_hit)) if col_hit.size else 0.0

    if defect_count > 0:
        try:
            skel = skeletonize(defect_mask.astype(bool))
            skel_count = int(skel.sum())
        except Exception:
            skel = defect_mask.astype(bool)
            skel_count = int(defect_count)
    else:
        skel = defect_mask.astype(bool)
        skel_count = 0

    feat["skeleton_pixels"] = float(skel_count)
    feat["skeleton_fraction_of_defects"] = safe_div(skel_count, defect_count)
    feat["skeleton_fraction_of_valid"] = safe_div(skel_count, valid_count)

    if skel_count > 0:
        neighbor_kernel = np.ones((3, 3), dtype=np.int32)
        neigh = ndimage.convolve(skel.astype(np.int32), neighbor_kernel, mode="constant", cval=0)
        endpoints = skel & (neigh == 2)
        branchpoints = skel & (neigh >= 4)
        feat["skeleton_endpoint_count"] = float(endpoints.sum())
        feat["skeleton_branchpoint_count"] = float(branchpoints.sum())
        feat["skeleton_endpoint_fraction"] = safe_div(endpoints.sum(), skel_count)
        feat["skeleton_branchpoint_fraction"] = safe_div(branchpoints.sum(), skel_count)
    else:
        feat["skeleton_endpoint_count"] = 0.0
        feat["skeleton_branchpoint_count"] = 0.0
        feat["skeleton_endpoint_fraction"] = 0.0
        feat["skeleton_branchpoint_fraction"] = 0.0

    if defect_count > 0:
        try:
            perim = float(perimeter(defect_mask.astype(bool), neighborhood=8))
        except TypeError:
            perim = float(perimeter(defect_mask.astype(bool)))
    else:
        perim = 0.0
    feat["defect_perimeter"] = perim
    feat["perimeter_to_area"] = safe_div(perim, defect_count)
    feat["compactness_area_over_perim2"] = safe_div(float(defect_count), perim * perim)

    if num_components > 0 and defect_count > 0:
        comp_slices = ndimage.find_objects(labeled)
        comp_records = []
        for comp_id, sl in enumerate(comp_slices, start=1):
            if sl is None:
                continue
            mask = labeled[sl] == comp_id
            area = int(mask.sum())
            if area <= 0:
                continue
            coords_local = np.argwhere(mask)
            coords = coords_local + np.array([sl[0].start, sl[1].start])
            cycx = coords.mean(axis=0)
            cov = _second_moments(coords, cycx)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
            major = float(np.sqrt(eigvals[0] + eps))
            minor = float(np.sqrt(eigvals[1] + eps))
            ys, xs = coords[:, 0], coords[:, 1]
            bh = int(ys.max() - ys.min() + 1)
            bw = int(xs.max() - xs.min() + 1)
            bbox_area = float(bh * bw)
            comp_records.append({
                "area": area,
                "major": major,
                "minor": minor,
                "ecc": safe_div(major, minor),
                "bbox_fill": safe_div(area, bbox_area),
                "bbox_aspect": safe_div(max(bh, bw), min(bh, bw)),
                "radius_mean": float(np.mean(np.sqrt((coords[:, 0] - wafer_center[0]) ** 2 + (coords[:, 1] - wafer_center[1]) ** 2))),
            })

        comp_records = sorted(comp_records, key=lambda r: r["area"], reverse=True)
        for k in range(3):
            if k < len(comp_records):
                rec = comp_records[k]
                feat[f"top{k+1}_component_area_frac"] = safe_div(rec["area"], defect_count)
                feat[f"top{k+1}_component_eccentricity"] = rec["ecc"]
                feat[f"top{k+1}_component_bbox_fill"] = rec["bbox_fill"]
                feat[f"top{k+1}_component_bbox_aspect"] = rec["bbox_aspect"]
                feat[f"top{k+1}_component_mean_radius_norm"] = safe_div(rec["radius_mean"], max(wafer_map.shape))
            else:
                feat[f"top{k+1}_component_area_frac"] = 0.0
                feat[f"top{k+1}_component_eccentricity"] = 0.0
                feat[f"top{k+1}_component_bbox_fill"] = 0.0
                feat[f"top{k+1}_component_bbox_aspect"] = 0.0
                feat[f"top{k+1}_component_mean_radius_norm"] = 0.0

        comp_eccs = np.asarray([r["ecc"] for r in comp_records], dtype=np.float32)
        comp_areas = np.asarray([r["area"] for r in comp_records], dtype=np.float32)
        feat["component_eccentricity_max"] = float(comp_eccs.max()) if comp_eccs.size else 0.0
        feat["component_eccentricity_area_weighted"] = float(np.sum(comp_eccs * comp_areas) / np.maximum(comp_areas.sum(), 1.0)) if comp_eccs.size else 0.0
    else:
        for k in range(3):
            feat[f"top{k+1}_component_area_frac"] = 0.0
            feat[f"top{k+1}_component_eccentricity"] = 0.0
            feat[f"top{k+1}_component_bbox_fill"] = 0.0
            feat[f"top{k+1}_component_bbox_aspect"] = 0.0
            feat[f"top{k+1}_component_mean_radius_norm"] = 0.0
        feat["component_eccentricity_max"] = 0.0
        feat["component_eccentricity_area_weighted"] = 0.0

    radon_input = defect_mask.astype(np.float32)
    if radon_thetas is None:
        radon_thetas = np.arange(0, 180, 15)
    sinogram = radon(radon_input, theta=radon_thetas, circle=False)
    if sinogram.size > 0:
        per_angle_energy = np.mean(sinogram ** 2, axis=0)
        per_angle_max = sinogram.max(axis=0)
        total_energy = float(per_angle_energy.sum())
        best_energy = float(per_angle_energy.max()) if per_angle_energy.size else 0.0
        sorted_energy = np.sort(per_angle_energy)[::-1]
        feat["radon_energy_peak_to_mean_enh"] = safe_div(best_energy, float(per_angle_energy.mean()))
        feat["radon_energy_top1_frac"] = safe_div(sorted_energy[0] if len(sorted_energy) > 0 else 0.0, total_energy)
        feat["radon_energy_top2_frac"] = safe_div(sorted_energy[:2].sum() if len(sorted_energy) > 1 else sorted_energy.sum(), total_energy)
        feat["radon_energy_entropy"] = float(-np.sum((per_angle_energy / (total_energy + eps)) * np.log(per_angle_energy / (total_energy + eps) + eps)))
        feat["radon_max_peak_to_mean_enh"] = safe_div(float(per_angle_max.max()), float(per_angle_max.mean()))
        best_idx = int(np.argmax(per_angle_energy))
        best_theta = float(radon_thetas[best_idx])
        feat["radon_best_energy_theta_sin"] = float(np.sin(np.deg2rad(best_theta)))
        feat["radon_best_energy_theta_cos"] = float(np.cos(np.deg2rad(best_theta)))
    else:
        feat["radon_energy_peak_to_mean_enh"] = 0.0
        feat["radon_energy_top1_frac"] = 0.0
        feat["radon_energy_top2_frac"] = 0.0
        feat["radon_energy_entropy"] = 0.0
        feat["radon_max_peak_to_mean_enh"] = 0.0
        feat["radon_best_energy_theta_sin"] = 0.0
        feat["radon_best_energy_theta_cos"] = 1.0

    feat["scratch_like_score"] = feat["skeleton_fraction_of_defects"] * feat["component_eccentricity_max"] * feat["radon_energy_peak_to_mean_enh"] * (1.0 + feat["perimeter_to_area"])
    feat["edge_ring_like_score"] = feat["edge_ring_continuity_score"] * (1.0 + feat["outer_to_inner_density_ratio"]) * (1.0 + feat["edge_angular_coverage_frac"])
    feat["local_blob_like_score"] = feat["radial_sector_grid_peak_to_mean"] * feat["largest_component_fraction_of_defects"] * (1.0 + feat["component_eccentricity_area_weighted"])


def extract_wafer_aux_features(
    wafer_map: np.ndarray,
    defect_threshold: float = 0.5,
    invalid_value: float = -1.0,
    num_radial_bins: int = 5,
    radon_thetas: np.ndarray | None = None,
) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
    if radon_thetas is None:
        radon_thetas = np.arange(0, 180, 30)

    wafer_map = np.asarray(wafer_map, dtype=np.float32)
    h, w = wafer_map.shape
    valid_mask, defect_mask = _get_valid_and_defect_masks(wafer_map, defect_threshold, invalid_value)

    valid_count = int(valid_mask.sum())
    defect_count = int(defect_mask.sum())

    rr, cc = np.indices((h, w))
    wafer_center = np.array([h / 2.0, w / 2.0], dtype=np.float32)
    radial_bin_map, max_r = _radial_bins(rr, cc, wafer_center, num_radial_bins, valid_mask)
    dist = np.sqrt((rr - wafer_center[0]) ** 2 + (cc - wafer_center[1]) ** 2)
    dist_norm = dist / max_r

    center_region = (dist_norm <= 0.4) & valid_mask
    mid_region = (dist_norm > 0.4) & (dist_norm < 0.7) & valid_mask
    edge_region = (dist_norm >= 0.7) & valid_mask

    feat: Dict[str, float] = {}
    feat["defect_ratio"] = safe_div(defect_count, valid_count)
    feat["center_density"] = safe_div((defect_mask & center_region).sum(), center_region.sum())
    feat["mid_density"] = safe_div((defect_mask & mid_region).sum(), mid_region.sum())
    feat["edge_density"] = safe_div((defect_mask & edge_region).sum(), edge_region.sum())

    for b in range(num_radial_bins):
        region = (radial_bin_map == b) & valid_mask
        feat[f"radial_density_bin_{b}"] = safe_div((defect_mask & region).sum(), region.sum())

    feat["edge_minus_center_density"] = feat["edge_density"] - feat["center_density"]
    feat["center_minus_mid_density"] = feat["center_density"] - feat["mid_density"]

    labeled, num_components = ndimage.label(defect_mask)
    feat["num_components"] = float(num_components)

    if defect_count > 0:
        defect_coords = np.argwhere(defect_mask)
        defect_centroid = _compute_centroid(defect_coords)
        offset = defect_centroid - wafer_center
        feat["centroid_offset_norm"] = float(np.sqrt(offset[0] ** 2 + offset[1] ** 2) / (max_r + 1e-8))
        feat["mean_defect_radius"] = float(np.mean(np.sqrt((defect_coords[:, 0] - wafer_center[0]) ** 2 + (defect_coords[:, 1] - wafer_center[1]) ** 2)) / (max_r + 1e-8))

        cov = _second_moments(defect_coords, defect_centroid)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
        major = float(np.sqrt(eigvals[0] + 1e-8))
        minor = float(np.sqrt(eigvals[1] + 1e-8))
        feat["major_spread"] = major
        feat["minor_spread"] = minor
        feat["eccentricity"] = safe_div(major, minor)

        if eigvals[0] > 0:
            principal_vec = eigvecs[:, np.argmax(eigvals)]
            angle = np.arctan2(principal_vec[0], principal_vec[1])
            feat["orientation_sin"] = float(np.sin(angle))
            feat["orientation_cos"] = float(np.cos(angle))
        else:
            feat["orientation_sin"] = 0.0
            feat["orientation_cos"] = 1.0

        ys, xs = np.where(defect_mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        box_area = float((y1 - y0 + 1) * (x1 - x0 + 1))
        feat["bbox_fill_ratio"] = safe_div(defect_count, box_area)
        feat["bbox_aspect_ratio"] = safe_div((y1 - y0 + 1), (x1 - x0 + 1))
    else:
        for k, v in {
            "centroid_offset_norm": 0.0, "mean_defect_radius": 0.0, "major_spread": 0.0, "minor_spread": 0.0,
            "eccentricity": 0.0, "orientation_sin": 0.0, "orientation_cos": 1.0, "bbox_fill_ratio": 0.0, "bbox_aspect_ratio": 0.0,
        }.items():
            feat[k] = v

    if num_components > 0:
        comp_sizes = ndimage.sum(defect_mask, labeled, index=np.arange(1, num_components + 1))
        comp_sizes = np.asarray(comp_sizes, dtype=np.float32)
        feat["largest_component_fraction_of_defects"] = safe_div(comp_sizes.max(), defect_count)
        feat["fragmentation_ratio"] = safe_div(num_components, defect_count + 1.0)
    else:
        feat["largest_component_fraction_of_defects"] = 0.0
        feat["fragmentation_ratio"] = 0.0

    num_sectors = 8
    dy = rr - wafer_center[0]
    dx = cc - wafer_center[1]
    theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
    sector_ids = np.floor(theta / (2 * np.pi / num_sectors)).astype(int)
    for s in range(num_sectors):
        region = (sector_ids == s) & valid_mask
        feat[f"sector_density_{s}"] = safe_div((defect_mask & region).sum(), region.sum())

    sinogram = radon(defect_mask.astype(np.float32), theta=radon_thetas, circle=False)
    feat["radon_mean"] = float(sinogram.mean()) if sinogram.size > 0 else 0.0
    feat["radon_std"] = float(sinogram.std()) if sinogram.size > 0 else 0.0
    feat["radon_energy"] = float(np.mean(sinogram ** 2)) if sinogram.size > 0 else 0.0

    if sinogram.size > 0:
        per_angle_max = sinogram.max(axis=0)
        sorted_peaks = np.sort(per_angle_max)[::-1]
        feat["radon_peak_0"] = float(sorted_peaks[0]) if len(sorted_peaks) > 0 else 0.0
        feat["radon_peak_1"] = float(sorted_peaks[1]) if len(sorted_peaks) > 1 else 0.0
        best_theta = float(radon_thetas[int(np.argmax(per_angle_max))])
        feat["radon_best_theta_sin"] = float(np.sin(np.deg2rad(best_theta)))
        feat["radon_best_theta_cos"] = float(np.cos(np.deg2rad(best_theta)))
    else:
        feat["radon_peak_0"] = 0.0
        feat["radon_peak_1"] = 0.0
        feat["radon_best_theta_sin"] = 0.0
        feat["radon_best_theta_cos"] = 1.0

    add_defect_sensitive_features(
        feat=feat,
        wafer_map=wafer_map,
        valid_mask=valid_mask,
        defect_mask=defect_mask,
        rr=rr,
        cc=cc,
        wafer_center=wafer_center,
        dist_norm=dist_norm,
        theta=theta,
        labeled=labeled,
        num_components=num_components,
        defect_count=defect_count,
        valid_count=valid_count,
        radon_thetas=np.arange(0, 180, 15),
    )

    add_radon_profile_features(feat, defect_mask, n_angles=180, n_profile=20)
    add_glcm_texture_features(feat, defect_mask)
    add_local_density_features(feat, defect_mask, valid_mask, window_sizes=(3, 5, 7))
    add_regionprops_extension_features(feat, labeled, defect_count, valid_count, wafer_map.shape)
    add_donut_hollow_features(feat, defect_mask, valid_mask, dist_norm)

    feature_names = list(feat.keys())
    feat_vec = np.array([feat[k] for k in feature_names], dtype=np.float32)
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return feat, feat_vec, feature_names


def dataset_to_arrays(ds, defect_threshold: float = 0.5, invalid_value: float = -1.0):
    wafer_maps = []
    aux_list = []
    y_multi = []
    feature_names = None
    for i in range(len(ds)):
        wafer, y = unpack_sample(ds[i])
        _, feat_vec, feature_names = extract_wafer_aux_features(wafer, defect_threshold=defect_threshold, invalid_value=invalid_value)
        wafer_maps.append(wafer)
        aux_list.append(feat_vec)
        y_multi.append(y)
    return (
        np.stack(wafer_maps, axis=0).astype(np.float32),
        np.stack(aux_list, axis=0).astype(np.float32),
        np.asarray(y_multi, dtype=np.int64),
        feature_names,
    )


def fit_clip_stats(X: np.ndarray, low_q: float = 1.0, high_q: float = 99.0):
    lo = np.percentile(X, low_q, axis=0)
    hi = np.percentile(X, high_q, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


def apply_clip(X: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    return np.clip(X, lo, hi)


def fit_standardize(X: np.ndarray, eps: float = 1e-6):
    mu = X.mean(axis=0).astype(np.float32)
    sigma = X.std(axis=0).astype(np.float32)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return mu, sigma


def apply_standardize(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return ((X - mu) / sigma).astype(np.float32)


class MultiScaleSpatialHDC:
    def __init__(
        self,
        image_size: int,
        hd_dim: int,
        patch_sizes: List[int],
        strides: List[int],
        seed: int = 0,
        defect_threshold: float = 0.5,
        invalid_value: float = -1.0,
    ):
        assert len(patch_sizes) == len(strides)
        self.image_size = image_size
        self.hd_dim = hd_dim
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.seed = seed
        self.defect_threshold = defect_threshold
        self.invalid_value = invalid_value

        rng = np.random.default_rng(seed)
        self.scale_hv = []
        self.pos_hv = []
        self.val_hv = {}
        self.num_occ_bins = 8

        for b in range(self.num_occ_bins):
            self.val_hv[b] = bipolar_sign(rng.standard_normal(hd_dim))

        for ps, st in zip(patch_sizes, strides):
            n_rows = max(1, 1 + (image_size - ps) // st)
            n_cols = max(1, 1 + (image_size - ps) // st)
            self.scale_hv.append(bipolar_sign(rng.standard_normal(hd_dim)))
            self.pos_hv.append(bipolar_sign(rng.standard_normal((n_rows, n_cols, hd_dim))).astype(np.float32))

    def _quantize_patch(self, patch: np.ndarray) -> int:
        valid_mask = patch != self.invalid_value
        if valid_mask.sum() == 0:
            return 0
        defect_mask = (patch > self.defect_threshold) & valid_mask
        occ = defect_mask.sum() / max(valid_mask.sum(), 1)
        b = int(np.floor(occ * self.num_occ_bins))
        return int(np.clip(b, 0, self.num_occ_bins - 1))

    def encode_single(self, wafer_map: np.ndarray) -> np.ndarray:
        wafer_map = np.asarray(wafer_map, dtype=np.float32)
        h, w = wafer_map.shape
        assert h == self.image_size and w == self.image_size, f"Expected {self.image_size}x{self.image_size}, got {wafer_map.shape}"
        accum = np.zeros(self.hd_dim, dtype=np.float32)
        for scale_idx, (ps, st) in enumerate(zip(self.patch_sizes, self.strides)):
            n_rows = self.pos_hv[scale_idx].shape[0]
            n_cols = self.pos_hv[scale_idx].shape[1]
            scale_accum = np.zeros(self.hd_dim, dtype=np.float32)
            for i in range(n_rows):
                r = i * st
                for j in range(n_cols):
                    c = j * st
                    patch = wafer_map[r:r + ps, c:c + ps]
                    b = self._quantize_patch(patch)
                    patch_hv = self.val_hv[b] * self.pos_hv[scale_idx][i, j] * self.scale_hv[scale_idx]
                    scale_accum += patch_hv
            accum += bipolar_sign(scale_accum)
        return bipolar_sign(accum)

    def encode_batch(self, wafer_maps: np.ndarray) -> np.ndarray:
        return np.stack([self.encode_single(w) for w in wafer_maps], axis=0).astype(np.float32)


class AuxRandomProjectionHDC:
    def __init__(self, in_dim: int, hd_dim: int, seed: int = 0, scale: float = 1.0):
        rng = np.random.default_rng(seed + 999)
        self.W = (rng.standard_normal((in_dim, hd_dim)).astype(np.float32) * scale)

    def encode_batch(self, X: np.ndarray) -> np.ndarray:
        return bipolar_sign(np.asarray(X, dtype=np.float32) @ self.W)


def fuse_hvs(
    h_spatial: np.ndarray,
    h_aux: np.ndarray,
    spatial_weight: float = 1.0,
    aux_weight: float = 0.5,
    binarize: bool = True,
) -> np.ndarray:
    z = spatial_weight * h_spatial + aux_weight * h_aux
    return bipolar_sign(z) if binarize else z.astype(np.float32)


# ============================================================
# Real-valued random-vector encoders and utilities
# ============================================================

def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        n = float(np.linalg.norm(X))
        return (X / max(n, eps)).astype(np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(n, eps)).astype(np.float32)


def squared_euclidean_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    D2 = A2 + B2 - 2.0 * (A @ B.T)
    return np.maximum(D2, 0.0).astype(np.float32)


class RealAuxRandomProjection:
    """Distance-preserving real-valued random projection for auxiliary features."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed + 999)
        # JL-style scaling: E[||XW/sqrt(D)||^2] ≈ ||X||^2.
        self.W = rng.standard_normal((int(in_dim), int(out_dim))).astype(np.float32) / np.sqrt(float(out_dim))

    def encode_batch(self, X: np.ndarray, normalize: bool = False) -> np.ndarray:
        Z = np.asarray(X, dtype=np.float32) @ self.W
        Z = Z.astype(np.float32)
        return l2_normalize(Z) if normalize else Z


class RealMultiScaleSpatialEncoder:
    """
    Vectorized real-valued multiscale spatial random-vector encoder.

    It follows the same multiscale patch-occupancy idea as the HDC spatial encoder,
    but it does not binarize the bundled vectors. Random value/position/scale vectors
    are real Gaussian vectors; selected patch codes are summed and normalized.
    """

    def __init__(
        self,
        image_size: int,
        out_dim: int,
        patch_sizes: List[int],
        strides: List[int],
        seed: int = 0,
        defect_threshold: float = 0.5,
        invalid_value: float = -1.0,
        batch_size: int = 32,
        position_chunk_size: int = 64,
        num_occ_bins: int = 8,
        normalize_output: bool = True,
    ):
        assert len(patch_sizes) == len(strides)
        self.image_size = int(image_size)
        self.out_dim = int(out_dim)
        self.patch_sizes = [int(x) for x in patch_sizes]
        self.strides = [int(x) for x in strides]
        self.seed = int(seed)
        self.defect_threshold = float(defect_threshold)
        self.invalid_value = float(invalid_value)
        self.batch_size = int(max(1, batch_size))
        self.position_chunk_size = int(max(1, position_chunk_size))
        self.num_occ_bins = int(num_occ_bins)
        self.normalize_output = bool(normalize_output)

        rng = np.random.default_rng(seed)
        # Real-valued base vectors. Unit variance keeps binding products unit variance.
        self.val_vec_mat = rng.standard_normal((self.num_occ_bins, self.out_dim)).astype(np.float32)

        self.pos_scale_flat: List[np.ndarray] = []
        self.scale_shapes: List[Tuple[int, int]] = []
        for ps, st in zip(self.patch_sizes, self.strides):
            n_rows = max(1, 1 + (self.image_size - ps) // st)
            n_cols = max(1, 1 + (self.image_size - ps) // st)
            scale = rng.standard_normal(self.out_dim).astype(np.float32)
            pos = rng.standard_normal((n_rows, n_cols, self.out_dim)).astype(np.float32)
            self.pos_scale_flat.append((pos.reshape(-1, self.out_dim) * scale[None, :]).astype(np.float32))
            self.scale_shapes.append((n_rows, n_cols))

    @staticmethod
    def _window_sum_batch(mask: np.ndarray, ps: int, st: int, n_rows: int, n_cols: int) -> np.ndarray:
        x = np.asarray(mask, dtype=np.float32)
        sat = np.pad(x.cumsum(axis=1).cumsum(axis=2), ((0, 0), (1, 0), (1, 0)), mode="constant")
        rows = np.arange(n_rows, dtype=np.int64) * int(st)
        cols = np.arange(n_cols, dtype=np.int64) * int(st)
        r0 = rows[:, None]
        c0 = cols[None, :]
        r1 = r0 + int(ps)
        c1 = c0 + int(ps)
        return sat[:, r1, c1] - sat[:, r0, c1] - sat[:, r1, c0] + sat[:, r0, c0]

    def _bin_maps_for_scale(self, wafer_batch: np.ndarray, scale_idx: int) -> np.ndarray:
        ps = self.patch_sizes[scale_idx]
        st = self.strides[scale_idx]
        n_rows, n_cols = self.scale_shapes[scale_idx]
        valid = wafer_batch != self.invalid_value
        defect = (wafer_batch > self.defect_threshold) & valid
        valid_counts = self._window_sum_batch(valid, ps, st, n_rows, n_cols)
        defect_counts = self._window_sum_batch(defect, ps, st, n_rows, n_cols)
        occ = defect_counts / np.maximum(valid_counts, 1.0)
        bins = np.floor(occ * self.num_occ_bins).astype(np.int16)
        bins = np.clip(bins, 0, self.num_occ_bins - 1)
        bins[valid_counts <= 0] = 0
        return bins.reshape(wafer_batch.shape[0], -1)

    def _encode_batch_chunk(self, wafer_batch: np.ndarray) -> np.ndarray:
        wafer_batch = np.asarray(wafer_batch, dtype=np.float32)
        B = wafer_batch.shape[0]
        accum = np.zeros((B, self.out_dim), dtype=np.float32)

        for scale_idx in range(len(self.patch_sizes)):
            bin_flat = self._bin_maps_for_scale(wafer_batch, scale_idx)
            pos_scale = self.pos_scale_flat[scale_idx]
            P = int(pos_scale.shape[0])
            scale_accum = np.zeros((B, self.out_dim), dtype=np.float32)
            pc = self.position_chunk_size
            for start in range(0, P, pc):
                end = min(start + pc, P)
                val_chunk = self.val_vec_mat[bin_flat[:, start:end]]
                bound = val_chunk * pos_scale[start:end][None, :, :]
                scale_accum += bound.sum(axis=1).astype(np.float32)
            # Normalize each scale so denser patch grids do not dominate solely by patch count.
            accum += scale_accum / np.sqrt(max(P, 1.0))

        accum = accum / np.sqrt(max(len(self.patch_sizes), 1.0))
        return l2_normalize(accum) if self.normalize_output else accum.astype(np.float32)

    def encode_single(self, wafer_map: np.ndarray) -> np.ndarray:
        wafer_map = np.asarray(wafer_map, dtype=np.float32)
        if wafer_map.shape != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size}, got {wafer_map.shape}")
        return self._encode_batch_chunk(wafer_map[None, :, :])[0]

    def encode_batch(self, wafer_maps: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        wafer_maps = np.asarray(wafer_maps, dtype=np.float32)
        if wafer_maps.ndim != 3:
            raise ValueError(f"Expected wafer_maps shape (N,H,W), got {wafer_maps.shape}")
        if wafer_maps.shape[1:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected maps {self.image_size}x{self.image_size}, got {wafer_maps.shape[1:]}")
        bs = self.batch_size if batch_size is None else int(max(1, batch_size))
        out = np.empty((wafer_maps.shape[0], self.out_dim), dtype=np.float32)
        for start in range(0, wafer_maps.shape[0], bs):
            end = min(start + bs, wafer_maps.shape[0])
            out[start:end] = self._encode_batch_chunk(wafer_maps[start:end])
        return out


class FastPatchSketchSpatialEncoder:
    """
    Fast sparse patch-sketch HDC wafer-map encoder.

    This is a faster alternative to RandomPatchSketchSpatialEncoder. It keeps the
    HDC idea—bind local raw patch pattern with scale and coarse position, then
    bundle tokens—but avoids dense 5000-D random projections per patch. Each
    composite token is written into a high-dimensional CountSketch vector using
    only a few signed hash locations.

        token = (scale, coarse position, pooled local patch code)
        wafer_hv = CountSketch-sum(token)

    No scalar morphology descriptors are computed: no centroid, radial density,
    connected components, skeleton, Radon, or edge-density summaries.
    """

    def __init__(
        self,
        image_size: int,
        out_dim: int,
        seed: int = 0,
        defect_threshold: float = 1.5,
        invalid_value: float = 0.0,
        position_grid: int = 12,
        patch_sizes: List[int] | Tuple[int, ...] = (9, 21),
        code_size: int = 4,
        max_tokens_per_map: int = 64,
        min_defect_pixels: int = 1,
        num_hashes: int = 4,
        relation_weight: float = 0.0,
        batch_size: int = 128,
        normalize_output: bool = True,
    ):
        self.image_size = int(image_size)
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        self.defect_threshold = float(defect_threshold)
        self.invalid_value = float(invalid_value)
        self.position_grid = int(max(1, position_grid))
        self.patch_sizes = [int(max(1, s)) for s in patch_sizes]
        self.code_size = int(max(2, min(8, code_size)))
        self.max_tokens_per_map = int(max(1, max_tokens_per_map))
        self.min_defect_pixels = int(max(1, min_defect_pixels))
        self.num_hashes = int(max(1, num_hashes))
        self.relation_weight = float(max(0.0, relation_weight))
        self.batch_size = int(max(1, batch_size))
        self.normalize_output = bool(normalize_output)
        self.pattern_bits = self.code_size * self.code_size
        if self.pattern_bits > 60:
            raise ValueError("code_size is too large for compact integer patch codes.")

        self.centers = np.linspace(0, self.image_size - 1, self.position_grid).round().astype(np.int32)
        # Fixed salts for sparse HDC/CountSketch token vectors.
        rng = np.random.default_rng(self.seed + 88123)
        self._salt_a = rng.integers(1, np.iinfo(np.uint64).max, size=self.num_hashes, dtype=np.uint64)
        self._salt_b = rng.integers(0, np.iinfo(np.uint64).max, size=self.num_hashes, dtype=np.uint64)
        self._weight = np.float32(1.0 / np.sqrt(float(self.num_hashes)))
        self.rel_dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]

    def _mix64(self, x: int, h: int) -> int:
        # SplitMix64-style finalizer after salted affine mixing.
        z = (np.uint64(x) * self._salt_a[h] + self._salt_b[h] + np.uint64(0x9E3779B97F4A7C15))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return int(z)

    def _add_token(self, out: np.ndarray, token_id: int, weight: float = 1.0):
        w = np.float32(weight) * self._weight
        for h in range(self.num_hashes):
            z = self._mix64(token_id + 104729 * h, h)
            j = z % self.out_dim
            sign = 1.0 if ((z >> 63) & 1) == 0 else -1.0
            out[j] += np.float32(sign) * w

    @staticmethod
    def _rect_sum(integral: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> int:
        H = integral.shape[0] - 1
        W = integral.shape[1] - 1
        y0 = max(0, min(H, int(y0)))
        y1 = max(0, min(H, int(y1)))
        x0 = max(0, min(W, int(x0)))
        x1 = max(0, min(W, int(x1)))
        if y1 <= y0 or x1 <= x0:
            return 0
        return int(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])

    def _patch_code(self, integral: np.ndarray, cy: int, cx: int, size: int) -> int:
        # Adaptive binary pooling over the nominal local patch. The resulting
        # code captures local layout but is not exposed as scalar features.
        r = int(size) // 2
        y0 = int(cy) - r
        x0 = int(cx) - r
        patch_side = 2 * r + 1
        P = self.code_size
        code = 0
        bit = 0
        ys = np.linspace(0, patch_side, P + 1).astype(np.int32)
        xs = np.linspace(0, patch_side, P + 1).astype(np.int32)
        for iy in range(P):
            yy0 = y0 + int(ys[iy])
            yy1 = y0 + int(ys[iy + 1])
            if yy1 <= yy0:
                yy1 = yy0 + 1
            for ix in range(P):
                xx0 = x0 + int(xs[ix])
                xx1 = x0 + int(xs[ix + 1])
                if xx1 <= xx0:
                    xx1 = xx0 + 1
                if self._rect_sum(integral, yy0, yy1, xx0, xx1) > 0:
                    code |= (1 << bit)
                bit += 1
        return int(code)

    def _candidate_tokens(self, defect: np.ndarray):
        integral = np.pad(defect.astype(np.int32), ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        H, W = defect.shape
        candidates = []
        for sidx, size in enumerate(self.patch_sizes):
            r = int(size) // 2
            for pr, cy in enumerate(self.centers):
                y0, y1 = int(cy) - r, int(cy) + r + 1
                for pc, cx in enumerate(self.centers):
                    x0, x1 = int(cx) - r, int(cx) + r + 1
                    cnt = self._rect_sum(integral, y0, y1, x0, x1)
                    if cnt >= self.min_defect_pixels:
                        pos_id = pr * self.position_grid + pc
                        code = self._patch_code(integral, int(cy), int(cx), int(size))
                        if code != 0:
                            candidates.append((cnt, sidx, pos_id, pr, pc, code))
        if len(candidates) > self.max_tokens_per_map:
            # Deterministic diversity-friendly thinning: sort by local defect mass
            # and take evenly across the high-mass candidate list. This is faster
            # than per-map random sampling and avoids near-full maps dominating by
            # all candidate positions.
            candidates.sort(key=lambda x: x[0], reverse=True)
            idx = np.linspace(0, len(candidates) - 1, self.max_tokens_per_map).round().astype(np.int64)
            candidates = [candidates[int(i)] for i in idx]
        return candidates

    def _encode_one(self, wafer_map: np.ndarray) -> np.ndarray:
        valid = wafer_map != self.invalid_value
        defect = (wafer_map > self.defect_threshold) & valid
        out = np.zeros((self.out_dim,), dtype=np.float32)
        if not np.any(defect):
            return out
        candidates = self._candidate_tokens(defect)
        if len(candidates) == 0:
            return out

        active = {}
        num_pos = self.position_grid * self.position_grid
        num_codes = 1 << self.pattern_bits
        for cnt, sidx, pos_id, pr, pc, code in candidates:
            # Composite HDC token: scale x position x local raw-patch code.
            token_id = int((sidx * num_pos + pos_id) * num_codes + code + 1)
            self._add_token(out, token_id, weight=1.0)
            active[(sidx, pr, pc)] = token_id

        if self.relation_weight > 0.0 and len(active) > 1:
            rel_base = (len(self.patch_sizes) * num_pos * num_codes) + 100003
            for (sidx, pr, pc), tid in list(active.items()):
                for ridx, (dr, dc) in enumerate(self.rel_dirs):
                    nb = (sidx, pr + dr, pc + dc)
                    if nb in active:
                        # Relation token binds two neighboring local-pattern tokens
                        # with a direction. This is still sparse and cheap.
                        rel_id = rel_base + (((tid * 1315423911) ^ (active[nb] * 2654435761) ^ (ridx * 97531)) & 0x7FFFFFFFFFFFFFFF)
                        self._add_token(out, int(rel_id), weight=self.relation_weight)

        if self.normalize_output:
            n = float(np.linalg.norm(out))
            if n > 1e-8:
                out = out / n
        return out.astype(np.float32)

    def encode_single(self, wafer_map: np.ndarray) -> np.ndarray:
        wafer_map = np.asarray(wafer_map, dtype=np.float32)
        if wafer_map.shape != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size}, got {wafer_map.shape}")
        return self._encode_one(wafer_map)

    def encode_batch(self, wafer_maps: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        wafer_maps = np.asarray(wafer_maps, dtype=np.float32)
        if wafer_maps.ndim != 3:
            raise ValueError(f"Expected wafer_maps shape (N,H,W), got {wafer_maps.shape}")
        if wafer_maps.shape[1:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected maps {self.image_size}x{self.image_size}, got {wafer_maps.shape[1:]}")
        out = np.empty((wafer_maps.shape[0], self.out_dim), dtype=np.float32)
        bs = self.batch_size if batch_size is None else int(max(1, batch_size))
        for start in range(0, wafer_maps.shape[0], bs):
            end = min(start + bs, wafer_maps.shape[0])
            for i in range(start, end):
                out[i] = self._encode_one(wafer_maps[i])
        return out.astype(np.float32)


class RandomPatchSketchSpatialEncoder:
    """
    Local patch-sketch HDC wafer-map encoder.

    This encoder is intentionally not a scalar morphology descriptor extractor.
    It encodes the raw wafer defect mask as a set of local patch memories:

        token = ScaleHV(s) * PositionHV(i,j) * PatternHV(patch)

    where PatternHV(patch) is a bipolar random projection of the local binary
    patch after adaptive pooling.  Optional relation tokens bind adjacent active
    patch memories, giving the hypervector some sensitivity to local continuity
    and layout (e.g., arcs, scratches, and chains) without computing hand-crafted
    region statistics such as centroid, radial density, connected components, or
    Radon features.
    """

    def __init__(
        self,
        image_size: int,
        out_dim: int,
        seed: int = 0,
        defect_threshold: float = 1.5,
        invalid_value: float = 0.0,
        position_grid: int = 16,
        patch_sizes: List[int] | Tuple[int, ...] = (7, 15, 31),
        code_size: int = 7,
        max_tokens_per_map: int = 128,
        min_defect_pixels: int = 1,
        relation_weight: float = 0.35,
        batch_size: int = 64,
        normalize_output: bool = True,
    ):
        self.image_size = int(image_size)
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        self.defect_threshold = float(defect_threshold)
        self.invalid_value = float(invalid_value)
        self.position_grid = int(max(1, position_grid))
        self.patch_sizes = [int(max(1, s)) for s in patch_sizes]
        self.code_size = int(max(2, code_size))
        self.max_tokens_per_map = int(max(1, max_tokens_per_map))
        self.min_defect_pixels = int(max(1, min_defect_pixels))
        self.relation_weight = float(max(0.0, relation_weight))
        self.batch_size = int(max(1, batch_size))
        self.normalize_output = bool(normalize_output)

        rng = np.random.default_rng(self.seed + 44381)
        self.pattern_dim = self.code_size * self.code_size
        # One random projection per scale.  The sign of the projection is the
        # local pattern hypervector.
        self.W = []
        for _ in self.patch_sizes:
            W = rng.standard_normal((self.pattern_dim, self.out_dim)).astype(np.float32)
            W /= np.sqrt(float(self.pattern_dim))
            self.W.append(W)
        self.scale_vecs = bipolar_sign(rng.standard_normal((len(self.patch_sizes), self.out_dim)).astype(np.float32))
        self.pos_vecs = bipolar_sign(rng.standard_normal((self.position_grid * self.position_grid, self.out_dim)).astype(np.float32))
        # Four local relation directions: right, down, down-right, down-left.
        self.rel_dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
        self.rel_vecs = bipolar_sign(rng.standard_normal((len(self.rel_dirs), self.out_dim)).astype(np.float32))

    @staticmethod
    def _parse_sizes(sizes: str | List[int] | Tuple[int, ...]) -> List[int]:
        if isinstance(sizes, str):
            return [int(x.strip()) for x in sizes.split(',') if x.strip()]
        return [int(x) for x in sizes]

    def _extract_patch(self, defect: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
        r = int(size) // 2
        H, W = defect.shape
        y0, y1 = cy - r, cy + r + 1
        x0, x1 = cx - r, cx + r + 1
        patch = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)
        yy0, yy1 = max(0, y0), min(H, y1)
        xx0, xx1 = max(0, x0), min(W, x1)
        py0, px0 = yy0 - y0, xx0 - x0
        if yy1 > yy0 and xx1 > xx0:
            patch[py0:py0 + (yy1 - yy0), px0:px0 + (xx1 - xx0)] = defect[yy0:yy1, xx0:xx1].astype(np.float32)
        return patch

    def _adaptive_avg_pool(self, patch: np.ndarray) -> np.ndarray:
        # Adaptive average pooling to code_size x code_size.  This is a raw local
        # patch sketch, not a morphology statistic exposed to the downstream model.
        H, W = patch.shape
        P = self.code_size
        ys = np.linspace(0, H, P + 1).astype(np.int32)
        xs = np.linspace(0, W, P + 1).astype(np.int32)
        out = np.zeros((P, P), dtype=np.float32)
        for i in range(P):
            y0, y1 = int(ys[i]), int(ys[i + 1])
            if y1 <= y0:
                y1 = min(H, y0 + 1)
            for j in range(P):
                x0, x1 = int(xs[j]), int(xs[j + 1])
                if x1 <= x0:
                    x1 = min(W, x0 + 1)
                out[i, j] = float(patch[y0:y1, x0:x1].mean())
        return out

    def _pattern_hv(self, pooled_patch: np.ndarray, scale_idx: int) -> np.ndarray:
        v = pooled_patch.reshape(-1).astype(np.float32)
        # Remove the mean to emphasize arrangement inside the patch rather than
        # raw local density.  Fall back to the raw vector if the patch is constant.
        v_centered = v - float(v.mean())
        norm = float(np.linalg.norm(v_centered))
        if norm > 1e-6:
            v = v_centered / norm
        else:
            norm = float(np.linalg.norm(v))
            if norm > 1e-6:
                v = v / norm
        return bipolar_sign(v @ self.W[scale_idx])

    def _candidate_cells(self, defect: np.ndarray, global_index: int):
        H, W = defect.shape
        centers = np.linspace(0, self.image_size - 1, self.position_grid).round().astype(np.int32)
        candidates = []
        for sidx, size in enumerate(self.patch_sizes):
            r = int(size) // 2
            # Integral image for quick defect counts in candidate windows.
            integral = np.pad(defect.astype(np.int32), ((1, 0), (1, 0)), mode='constant').cumsum(axis=0).cumsum(axis=1)
            for pr, cy in enumerate(centers):
                y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)
                for pc, cx in enumerate(centers):
                    x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
                    cnt = int(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])
                    if cnt >= self.min_defect_pixels:
                        pos_id = pr * self.position_grid + pc
                        candidates.append((sidx, pos_id, pr, pc, int(cy), int(cx), int(size), cnt))
        if len(candidates) > self.max_tokens_per_map:
            # Keep a deterministic random subset.  This avoids near-full wafers
            # dominating solely by token count while preserving layout diversity.
            rng = np.random.default_rng(self.seed + 13007 * (int(global_index) + 1))
            idx = rng.choice(len(candidates), size=self.max_tokens_per_map, replace=False)
            candidates = [candidates[int(i)] for i in idx]
        return candidates

    def _encode_one(self, wafer_map: np.ndarray, global_index: int = 0) -> np.ndarray:
        valid = wafer_map != self.invalid_value
        defect = (wafer_map > self.defect_threshold) & valid
        out = np.zeros((self.out_dim,), dtype=np.float32)
        if not np.any(defect):
            return out
        candidates = self._candidate_cells(defect, global_index=global_index)
        if len(candidates) == 0:
            return out

        active: Dict[Tuple[int, int, int], np.ndarray] = {}
        for sidx, pos_id, pr, pc, cy, cx, size, cnt in candidates:
            patch = self._extract_patch(defect, cy, cx, size)
            pooled = self._adaptive_avg_pool(patch)
            phv = self._pattern_hv(pooled, sidx)
            token_hv = phv * self.scale_vecs[sidx] * self.pos_vecs[pos_id]
            out += token_hv
            active[(sidx, pr, pc)] = token_hv

        if self.relation_weight > 0.0 and len(active) > 1:
            for (sidx, pr, pc), hv in list(active.items()):
                for ridx, (dr, dc) in enumerate(self.rel_dirs):
                    nb = (sidx, pr + dr, pc + dc)
                    if nb in active:
                        out += self.relation_weight * hv * active[nb] * self.rel_vecs[ridx] * self.scale_vecs[sidx]

        out /= np.sqrt(float(max(len(candidates), 1)))
        if self.normalize_output:
            n = float(np.linalg.norm(out))
            if n > 1e-8:
                out = out / n
        return out.astype(np.float32)

    def encode_single(self, wafer_map: np.ndarray) -> np.ndarray:
        wafer_map = np.asarray(wafer_map, dtype=np.float32)
        if wafer_map.shape != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size}, got {wafer_map.shape}")
        return self._encode_one(wafer_map, global_index=0)

    def encode_batch(self, wafer_maps: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        wafer_maps = np.asarray(wafer_maps, dtype=np.float32)
        if wafer_maps.ndim != 3:
            raise ValueError(f"Expected wafer_maps shape (N,H,W), got {wafer_maps.shape}")
        if wafer_maps.shape[1:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected maps {self.image_size}x{self.image_size}, got {wafer_maps.shape[1:]}")
        out = np.empty((wafer_maps.shape[0], self.out_dim), dtype=np.float32)
        bs = self.batch_size if batch_size is None else int(max(1, batch_size))
        for start in range(0, wafer_maps.shape[0], bs):
            end = min(start + bs, wafer_maps.shape[0])
            for i in range(start, end):
                out[i] = self._encode_one(wafer_maps[i], global_index=i)
        return out.astype(np.float32)


class SparseLocalPatternTokenSpatialEncoder:
    """
    Sparse local-token HDC wafer-map encoder.

    This encoder is intentionally different from the scalar auxiliary/morphology
    descriptors. It does not compute global density, centroid, radial profiles,
    connected components, Radon features, etc. Instead, it encodes the wafer map
    as a multiset of local raw binary-pattern tokens:

        token = (scale, coarse position cell, local 3x3 pattern code)

    Each composite token is mapped to a sparse high-dimensional vector using a
    CountSketch-style deterministic hash with several signed nonzero dimensions.
    The wafer hypervector is the bundled sum of these token hypervectors.
    """

    def __init__(
        self,
        image_size: int,
        out_dim: int,
        seed: int = 0,
        defect_threshold: float = 1.5,
        invalid_value: float = 0.0,
        position_grid: int = 16,
        pattern_scales: List[int] | Tuple[int, ...] = (1, 2, 4),
        max_tokens_per_map: int = 96,
        num_hashes: int = 8,
        batch_size: int = 128,
        normalize_output: bool = True,
    ):
        self.image_size = int(image_size)
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        self.defect_threshold = float(defect_threshold)
        self.invalid_value = float(invalid_value)
        self.position_grid = int(max(1, position_grid))
        self.pattern_scales = [int(max(1, s)) for s in pattern_scales]
        self.max_tokens_per_map = int(max(1, max_tokens_per_map))
        self.num_hashes = int(max(1, num_hashes))
        self.batch_size = int(max(1, batch_size))
        self.normalize_output = bool(normalize_output)
        self.num_pattern_codes = 512  # 3x3 binary pattern.
        self.num_position_codes = self.position_grid * self.position_grid
        self.vocab_size = len(self.pattern_scales) * self.num_position_codes * self.num_pattern_codes
        # Fixed salts for CountSketch-style sparse token HVs.
        rng = np.random.default_rng(self.seed + 77731)
        self._salt_a = rng.integers(1, np.iinfo(np.uint64).max, size=self.num_hashes, dtype=np.uint64)
        self._salt_b = rng.integers(0, np.iinfo(np.uint64).max, size=self.num_hashes, dtype=np.uint64)
        self._weight = np.float32(1.0 / np.sqrt(float(self.num_hashes)))

    @staticmethod
    def _parse_scales(scales: str | List[int] | Tuple[int, ...]) -> List[int]:
        if isinstance(scales, str):
            return [int(x.strip()) for x in scales.split(",") if x.strip()]
        return [int(x) for x in scales]

    def _local_pattern_codes(self, defect: np.ndarray, radius: int) -> np.ndarray:
        """Return 3x3 binary pattern codes at the given radius for a batch mask."""
        r = int(max(1, radius))
        B, H, W = defect.shape
        padded = np.pad(defect.astype(np.uint16), ((0, 0), (r, r), (r, r)), mode="constant")
        code = np.zeros((B, H, W), dtype=np.uint16)
        bit = 0
        # Include center and eight neighbors at this radius. This is a local raw
        # pattern code, not a hand-crafted morphology statistic.
        for dr in (-r, 0, r):
            for dc in (-r, 0, r):
                sl = padded[:, r + dr:r + dr + H, r + dc:r + dc + W]
                code |= (sl.astype(np.uint16) << np.uint16(bit))
                bit += 1
        return code

    def _token_ids_for_map(self, defect_b: np.ndarray, code_maps_b: List[np.ndarray], global_index: int) -> np.ndarray:
        coords = np.argwhere(defect_b)
        if coords.shape[0] == 0:
            return np.empty((0,), dtype=np.uint64)
        if coords.shape[0] > self.max_tokens_per_map:
            rng = np.random.default_rng(self.seed + 104729 * (int(global_index) + 1))
            sel = rng.choice(coords.shape[0], size=self.max_tokens_per_map, replace=False)
            coords = coords[sel]
        rows = coords[:, 0].astype(np.int64)
        cols = coords[:, 1].astype(np.int64)
        # Coarse absolute position. This preserves layout without computing radial
        # or global shape descriptors.
        pr = np.minimum((rows * self.position_grid) // self.image_size, self.position_grid - 1)
        pc = np.minimum((cols * self.position_grid) // self.image_size, self.position_grid - 1)
        pos_id = (pr * self.position_grid + pc).astype(np.uint64)
        all_ids = []
        base_pos = self.num_pattern_codes
        base_scale = self.num_position_codes * self.num_pattern_codes
        for sidx, cm in enumerate(code_maps_b):
            code = cm[rows, cols].astype(np.uint64)
            token_id = np.uint64(sidx * base_scale) + pos_id * np.uint64(base_pos) + code
            all_ids.append(token_id)
        if len(all_ids) == 0:
            return np.empty((0,), dtype=np.uint64)
        return np.concatenate(all_ids).astype(np.uint64, copy=False)

    def _add_token_hvs(self, out_vec: np.ndarray, token_ids: np.ndarray) -> None:
        if token_ids.size == 0:
            return
        ids = token_ids.astype(np.uint64, copy=False)[:, None]
        # SplitMix/LCG-like independent hashes. We use uint64 overflow on purpose.
        h = ids * self._salt_a[None, :] + self._salt_b[None, :]
        h ^= (h >> np.uint64(33))
        h *= np.uint64(0xff51afd7ed558ccd)
        h ^= (h >> np.uint64(33))
        dims = (h % np.uint64(self.out_dim)).astype(np.int64).ravel()
        signs = np.where(((h >> np.uint64(63)) & np.uint64(1)).astype(np.int8).ravel() > 0, 1.0, -1.0).astype(np.float32)
        np.add.at(out_vec, dims, signs * self._weight)

    def _encode_batch_chunk(self, wafer_batch: np.ndarray, start_index: int = 0) -> np.ndarray:
        wafer_batch = np.asarray(wafer_batch, dtype=np.float32)
        if wafer_batch.ndim != 3:
            raise ValueError(f"Expected wafer_batch shape (B,H,W), got {wafer_batch.shape}")
        B, H, W = wafer_batch.shape
        if (H, W) != (self.image_size, self.image_size):
            raise ValueError(f"Expected maps {self.image_size}x{self.image_size}, got {(H, W)}")
        valid = wafer_batch != self.invalid_value
        defect = (wafer_batch > self.defect_threshold) & valid
        code_maps = [self._local_pattern_codes(defect, r) for r in self.pattern_scales]
        out = np.zeros((B, self.out_dim), dtype=np.float32)
        for b in range(B):
            cm_b = [cm[b] for cm in code_maps]
            token_ids = self._token_ids_for_map(defect[b], cm_b, global_index=start_index + b)
            self._add_token_hvs(out[b], token_ids)
        if self.normalize_output:
            out = l2_normalize(out)
        return out.astype(np.float32)

    def encode_single(self, wafer_map: np.ndarray) -> np.ndarray:
        wafer_map = np.asarray(wafer_map, dtype=np.float32)
        return self._encode_batch_chunk(wafer_map[None, :, :], start_index=0)[0]

    def encode_batch(self, wafer_maps: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        wafer_maps = np.asarray(wafer_maps, dtype=np.float32)
        if wafer_maps.ndim != 3:
            raise ValueError(f"Expected wafer_maps shape (N,H,W), got {wafer_maps.shape}")
        if wafer_maps.shape[1:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected maps {self.image_size}x{self.image_size}, got {wafer_maps.shape[1:]}")
        bs = self.batch_size if batch_size is None else int(max(1, batch_size))
        out = np.empty((wafer_maps.shape[0], self.out_dim), dtype=np.float32)
        for start in range(0, wafer_maps.shape[0], bs):
            end = min(start + bs, wafer_maps.shape[0])
            out[start:end] = self._encode_batch_chunk(wafer_maps[start:end], start_index=start)
        return out.astype(np.float32)

class RealMorphologySpatialEncoder:
    """
    Morphology-aware real-valued spatial HDC encoder.

    Unlike RealMultiScaleSpatialEncoder, which only bundles local patch occupancy,
    this encoder first extracts a structured wafer-map morphology profile and then
    maps it into a high-dimensional memory vector through a JL-style random
    projection.  The profile is still purely spatial: it is computed from the raw
    wafer map and contains radial, angular, edge-distance, projection, component,
    and coarse orientation descriptors.  The projection gives the resulting vector
    the same high-dimensional memory interface used by the Stage-2 online
    prototype clusterer.
    """

    def __init__(
        self,
        image_size: int,
        out_dim: int,
        seed: int = 0,
        defect_threshold: float = 0.5,
        invalid_value: float = -1.0,
        radial_bins: int = 10,
        angular_bins: int = 16,
        polar_radial_bins: int = 5,
        polar_angular_bins: int = 16,
        edge_bins: int = 8,
        projection_bins: int = 16,
        top_components: int = 5,
        batch_size: int = 256,
        normalize_output: bool = False,
        standardize: bool = True,
        clip_value: float = 6.0,
    ):
        self.image_size = int(image_size)
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        self.defect_threshold = float(defect_threshold)
        self.invalid_value = float(invalid_value)
        self.radial_bins = int(radial_bins)
        self.angular_bins = int(angular_bins)
        self.polar_radial_bins = int(polar_radial_bins)
        self.polar_angular_bins = int(polar_angular_bins)
        self.edge_bins = int(edge_bins)
        self.projection_bins = int(projection_bins)
        self.top_components = int(top_components)
        self.batch_size = int(max(1, batch_size))
        self.normalize_output = bool(normalize_output)
        self.standardize = bool(standardize)
        self.clip_value = float(clip_value)

        H = W = self.image_size
        rr, cc = np.indices((H, W), dtype=np.float32)
        cy = (H - 1.0) / 2.0
        cx = (W - 1.0) / 2.0
        dy = rr - cy
        dx = cc - cx
        rad = np.sqrt(dx * dx + dy * dy)
        max_r = float(np.max(rad) + 1e-8)
        theta = (np.arctan2(dy, dx) + 2.0 * np.pi) % (2.0 * np.pi)

        self.rr = rr
        self.cc = cc
        self.cy = cy
        self.cx = cx
        self.max_r = max_r
        self.rad_norm = (rad / max_r).astype(np.float32)
        self.radial_bin_map = np.clip((self.rad_norm * self.radial_bins).astype(np.int32), 0, self.radial_bins - 1)
        self.angular_bin_map = np.clip((theta / (2.0 * np.pi) * self.angular_bins).astype(np.int32), 0, self.angular_bins - 1)
        self.polar_r_bin_map = np.clip((self.rad_norm * self.polar_radial_bins).astype(np.int32), 0, self.polar_radial_bins - 1)
        self.polar_a_bin_map = np.clip((theta / (2.0 * np.pi) * self.polar_angular_bins).astype(np.int32), 0, self.polar_angular_bins - 1)
        self.polar_bin_map = self.polar_r_bin_map * self.polar_angular_bins + self.polar_a_bin_map

        edge_dist = np.minimum.reduce([rr, cc, (H - 1.0) - rr, (W - 1.0) - cc])
        max_edge = float(np.max(edge_dist) + 1e-8)
        # bin 0 = edge-near, larger bins = more central.
        self.edge_bin_map = np.clip((edge_dist / max_edge * self.edge_bins).astype(np.int32), 0, self.edge_bins - 1)

        self.row_bin_map = np.clip((rr / max(H - 1.0, 1.0) * self.projection_bins).astype(np.int32), 0, self.projection_bins - 1)
        self.col_bin_map = np.clip((cc / max(W - 1.0, 1.0) * self.projection_bins).astype(np.int32), 0, self.projection_bins - 1)
        diag1 = (rr + cc) / max((H - 1.0) + (W - 1.0), 1.0)
        diag2 = (rr - cc + (W - 1.0)) / max((H - 1.0) + (W - 1.0), 1.0)
        self.diag1_bin_map = np.clip((diag1 * self.projection_bins).astype(np.int32), 0, self.projection_bins - 1)
        self.diag2_bin_map = np.clip((diag2 * self.projection_bins).astype(np.int32), 0, self.projection_bins - 1)

        self.feature_names: List[str] = []
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.W_: np.ndarray | None = None

    @staticmethod
    def _densities_and_masses(defect_flat: np.ndarray, valid_flat: np.ndarray, bin_flat: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        n_bins = int(n_bins)
        v = valid_flat.astype(bool)
        d = defect_flat.astype(bool)
        if v.sum() == 0:
            return np.zeros(n_bins, dtype=np.float32), np.zeros(n_bins, dtype=np.float32)
        valid_counts = np.bincount(bin_flat[v], minlength=n_bins).astype(np.float32)
        defect_counts = np.bincount(bin_flat[d], minlength=n_bins).astype(np.float32)
        density = defect_counts / np.maximum(valid_counts, 1.0)
        mass = defect_counts / max(float(defect_counts.sum()), 1.0)
        return density.astype(np.float32), mass.astype(np.float32)

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        p = np.asarray(p, dtype=np.float64)
        s = float(p.sum())
        if s <= 0:
            return 0.0
        p = p / s
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-12)))

    @staticmethod
    def _longest_circular_run(mask_1d: np.ndarray) -> float:
        arr = np.asarray(mask_1d, dtype=bool)
        n = len(arr)
        if n == 0 or not arr.any():
            return 0.0
        if arr.all():
            return 1.0
        doubled = np.concatenate([arr, arr])
        best = 0
        cur = 0
        for v in doubled:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return float(min(best, n)) / float(max(n, 1))

    def _append_named(self, vals: List[float], names: List[str], name: str, value: float):
        vals.append(float(value))
        names.append(name)

    def _append_vector(self, vals: List[float], names: List[str], prefix: str, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).ravel()
        for i, v in enumerate(x):
            vals.append(float(v))
            names.append(f"{prefix}_{i}")

    def _profile_single(self, wafer_map: np.ndarray, collect_names: bool = False) -> Tuple[np.ndarray, List[str]]:
        wm = np.asarray(wafer_map, dtype=np.float32)
        valid = wm != self.invalid_value
        defect = (wm > self.defect_threshold) & valid
        valid_flat = valid.ravel()
        defect_flat = defect.ravel()
        vals: List[float] = []
        names: List[str] = []

        n_pix = float(wm.size)
        n_valid = float(valid.sum())
        n_def = float(defect.sum())
        defect_frac_valid = n_def / max(n_valid, 1.0)
        self._append_named(vals, names, "valid_fraction", n_valid / max(n_pix, 1.0))
        self._append_named(vals, names, "defect_fraction_valid", defect_frac_valid)
        self._append_named(vals, names, "defect_fraction_image", n_def / max(n_pix, 1.0))

        if n_def <= 0:
            # keep the feature length identical for empty maps
            coords_y = np.asarray([], dtype=np.float32)
            coords_x = np.asarray([], dtype=np.float32)
        else:
            coords = np.argwhere(defect).astype(np.float32)
            coords_y = coords[:, 0]
            coords_x = coords[:, 1]

        if n_def > 0:
            y_mean = float(coords_y.mean())
            x_mean = float(coords_x.mean())
            y_norm = (y_mean - self.cy) / max(self.image_size / 2.0, 1.0)
            x_norm = (x_mean - self.cx) / max(self.image_size / 2.0, 1.0)
            dy = coords_y - y_mean
            dx = coords_x - x_mean
            var_y = float(np.mean(dy * dy)) / (self.image_size * self.image_size)
            var_x = float(np.mean(dx * dx)) / (self.image_size * self.image_size)
            cov_xy = float(np.mean(dx * dy)) / (self.image_size * self.image_size)
            cov = np.array([[var_y, cov_xy], [cov_xy, var_x]], dtype=np.float64)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 0.0)
                major = float(eigvals[-1])
                minor = float(eigvals[0])
                v = eigvecs[:, -1]
                orient = float(np.arctan2(v[0], v[1]))
            except Exception:
                major = minor = orient = 0.0
            elong = major / (minor + 1e-8)
            y0, y1 = float(coords_y.min()), float(coords_y.max())
            x0, x1 = float(coords_x.min()), float(coords_x.max())
            bbox_h = (y1 - y0 + 1.0) / max(self.image_size, 1.0)
            bbox_w = (x1 - x0 + 1.0) / max(self.image_size, 1.0)
            bbox_area = bbox_h * bbox_w
            centroid_r = np.sqrt((x_mean - self.cx) ** 2 + (y_mean - self.cy) ** 2) / self.max_r
            centroid_ang = np.arctan2(y_mean - self.cy, x_mean - self.cx)
        else:
            y_norm = x_norm = var_y = var_x = cov_xy = 0.0
            major = minor = elong = orient = 0.0
            bbox_h = bbox_w = bbox_area = 0.0
            centroid_r = centroid_ang = 0.0

        for name, value in [
            ("centroid_y_norm", y_norm), ("centroid_x_norm", x_norm),
            ("centroid_r_norm", centroid_r), ("centroid_angle_sin", np.sin(centroid_ang)),
            ("centroid_angle_cos", np.cos(centroid_ang)), ("var_y", var_y),
            ("var_x", var_x), ("cov_xy", cov_xy), ("major_var", major),
            ("minor_var", minor), ("elongation", np.log1p(elong)),
            ("orientation_sin2", np.sin(2.0 * orient)), ("orientation_cos2", np.cos(2.0 * orient)),
            ("bbox_h", bbox_h), ("bbox_w", bbox_w), ("bbox_area", bbox_area),
            ("bbox_fill", n_def / max(bbox_area * self.image_size * self.image_size, 1.0)),
        ]:
            self._append_named(vals, names, name, value)

        # Region profiles: density within valid die and mass distribution over defects.
        r_density, r_mass = self._densities_and_masses(defect_flat, valid_flat, self.radial_bin_map.ravel(), self.radial_bins)
        a_density, a_mass = self._densities_and_masses(defect_flat, valid_flat, self.angular_bin_map.ravel(), self.angular_bins)
        p_density, p_mass = self._densities_and_masses(defect_flat, valid_flat, self.polar_bin_map.ravel(), self.polar_radial_bins * self.polar_angular_bins)
        e_density, e_mass = self._densities_and_masses(defect_flat, valid_flat, self.edge_bin_map.ravel(), self.edge_bins)
        row_density, row_mass = self._densities_and_masses(defect_flat, valid_flat, self.row_bin_map.ravel(), self.projection_bins)
        col_density, col_mass = self._densities_and_masses(defect_flat, valid_flat, self.col_bin_map.ravel(), self.projection_bins)
        d1_density, d1_mass = self._densities_and_masses(defect_flat, valid_flat, self.diag1_bin_map.ravel(), self.projection_bins)
        d2_density, d2_mass = self._densities_and_masses(defect_flat, valid_flat, self.diag2_bin_map.ravel(), self.projection_bins)

        self._append_vector(vals, names, "radial_density", r_density)
        self._append_vector(vals, names, "radial_mass", r_mass)
        self._append_vector(vals, names, "angular_density", a_density)
        self._append_vector(vals, names, "angular_mass", a_mass)
        self._append_vector(vals, names, "polar_density", p_density)
        self._append_vector(vals, names, "edge_density", e_density)
        self._append_vector(vals, names, "edge_mass", e_mass)
        self._append_vector(vals, names, "row_density", row_density)
        self._append_vector(vals, names, "col_density", col_density)
        self._append_vector(vals, names, "diag1_density", d1_density)
        self._append_vector(vals, names, "diag2_density", d2_density)

        # Summary statistics for profiles.  These are useful for ring vs localized defects.
        for prefix, dens, mass in [
            ("radial", r_density, r_mass), ("angular", a_density, a_mass),
            ("edge", e_density, e_mass), ("row", row_density, row_mass),
            ("col", col_density, col_mass), ("diag1", d1_density, d1_mass),
            ("diag2", d2_density, d2_mass),
        ]:
            self._append_named(vals, names, f"{prefix}_density_max", float(np.max(dens)) if dens.size else 0.0)
            self._append_named(vals, names, f"{prefix}_density_std", float(np.std(dens)) if dens.size else 0.0)
            self._append_named(vals, names, f"{prefix}_mass_entropy", self._entropy(mass))
            thr = float(np.mean(dens) + np.std(dens)) if dens.size else 0.0
            self._append_named(vals, names, f"{prefix}_active_run", self._longest_circular_run(dens > thr) if prefix in ["angular"] else float(np.mean(dens > thr)) if dens.size else 0.0)

        # Connected components on the defect map; this separates loc/random/scratch-like morphologies.
        if n_def > 0:
            lab, ncomp = ndimage.label(defect)
            if ncomp > 0:
                comp_sizes = np.bincount(lab.ravel(), minlength=ncomp + 1).astype(np.float32)[1:]
                comp_sizes_sorted = np.sort(comp_sizes)[::-1]
                comp_mass = comp_sizes / max(float(comp_sizes.sum()), 1.0)
            else:
                comp_sizes_sorted = np.asarray([], dtype=np.float32)
                comp_mass = np.asarray([], dtype=np.float32)
        else:
            ncomp = 0
            comp_sizes_sorted = np.asarray([], dtype=np.float32)
            comp_mass = np.asarray([], dtype=np.float32)
        self._append_named(vals, names, "component_count_log", np.log1p(float(ncomp)))
        self._append_named(vals, names, "component_density", float(ncomp) / max(n_def, 1.0))
        self._append_named(vals, names, "component_entropy", self._entropy(comp_mass))
        self._append_named(vals, names, "largest_component_frac", float(comp_sizes_sorted[0]) / max(n_def, 1.0) if comp_sizes_sorted.size else 0.0)
        for i in range(self.top_components):
            v = float(comp_sizes_sorted[i]) / max(n_def, 1.0) if i < comp_sizes_sorted.size else 0.0
            self._append_named(vals, names, f"top_component_frac_{i}", v)

        return np.asarray(vals, dtype=np.float32), names

    def extract_features_batch(self, wafer_maps: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        wafer_maps = np.asarray(wafer_maps, dtype=np.float32)
        if wafer_maps.ndim != 3:
            raise ValueError(f"Expected wafer_maps shape (N,H,W), got {wafer_maps.shape}")
        if wafer_maps.shape[1:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected maps {self.image_size}x{self.image_size}, got {wafer_maps.shape[1:]}")
        feats = []
        feature_names = None
        for i in range(wafer_maps.shape[0]):
            f, names = self._profile_single(wafer_maps[i], collect_names=(i == 0))
            feats.append(f)
            if i == 0:
                feature_names = names
        self.feature_names = feature_names or []
        return np.stack(feats, axis=0).astype(np.float32)

    def _fit_from_features(self, X: np.ndarray):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.standardize:
            med = np.median(X, axis=0).astype(np.float32)
            q25 = np.percentile(X, 25, axis=0).astype(np.float32)
            q75 = np.percentile(X, 75, axis=0).astype(np.float32)
            iqr = q75 - q25
            std = X.std(axis=0).astype(np.float32)
            scale = np.where(iqr > 1e-6, iqr / 1.349, std)
            scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
        else:
            med = np.zeros(X.shape[1], dtype=np.float32)
            scale = np.ones(X.shape[1], dtype=np.float32)
        self.center_ = med
        self.scale_ = scale
        rng = np.random.default_rng(self.seed + 424242)
        self.W_ = (rng.standard_normal((X.shape[1], self.out_dim)).astype(np.float32) / np.sqrt(float(self.out_dim)))
        return self

    def fit(self, wafer_maps: np.ndarray):
        X = self.extract_features_batch(wafer_maps)
        return self._fit_from_features(X)

    def _project_features(self, X: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None or self.W_ is None:
            raise RuntimeError("RealMorphologySpatialEncoder must be fit before transform/encode_batch.")
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        Xz = (X - self.center_[None, :]) / self.scale_[None, :]
        if self.clip_value > 0:
            Xz = np.clip(Xz, -self.clip_value, self.clip_value)
        Z = (Xz @ self.W_).astype(np.float32)
        return l2_normalize(Z) if self.normalize_output else Z

    def transform(self, wafer_maps: np.ndarray) -> np.ndarray:
        X = self.extract_features_batch(wafer_maps)
        return self._project_features(X)

    def fit_transform(self, wafer_maps: np.ndarray) -> np.ndarray:
        X = self.extract_features_batch(wafer_maps)
        self._fit_from_features(X)
        return self._project_features(X)

    def encode_batch(self, wafer_maps: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        return self.transform(wafer_maps)



def fuse_real_vectors(
    Z_spatial: np.ndarray,
    Z_aux: np.ndarray,
    spatial_weight: float = 1.0,
    aux_weight: float = 0.5,
    normalize_inputs: bool = True,
    normalize_output: bool = True,
) -> np.ndarray:
    Zs = l2_normalize(Z_spatial) if normalize_inputs else np.asarray(Z_spatial, dtype=np.float32)
    Za = l2_normalize(Z_aux) if normalize_inputs else np.asarray(Z_aux, dtype=np.float32)
    Z = float(spatial_weight) * Zs + float(aux_weight) * Za
    return l2_normalize(Z) if normalize_output else Z.astype(np.float32)

# ---- Binary thermometer HDC encoder and Hamming utilities (v10) ----

def _popcount_uint64(x: np.ndarray) -> np.ndarray:
    """Vectorized popcount for uint64 arrays, with NumPy-version fallback."""
    x = np.asarray(x, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    # Fallback for older NumPy: byte lookup table.
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return lut[x.view(np.uint8)].reshape(x.shape + (8,)).sum(axis=-1)


def pack_bits_little(bits: np.ndarray) -> np.ndarray:
    """Pack bool/int bits into uint64 words using little-endian bit order."""
    bits = np.asarray(bits, dtype=np.bool_)
    n, d = bits.shape
    pad = (-d) % 64
    if pad:
        bits = np.pad(bits, ((0, 0), (0, pad)), constant_values=False)
    packed8 = np.packbits(bits, axis=1, bitorder="little")
    # Ensure byte dimension is divisible by 8 before view.
    byte_pad = (-packed8.shape[1]) % 8
    if byte_pad:
        packed8 = np.pad(packed8, ((0, 0), (0, byte_pad)), constant_values=0)
    return np.ascontiguousarray(packed8).view(np.uint64)


def unpack_bits_little(packed: np.ndarray, bit_dim: int) -> np.ndarray:
    """Unpack uint64 packed hypervectors to bool matrix with exactly bit_dim bits."""
    packed = np.ascontiguousarray(np.asarray(packed, dtype=np.uint64))
    if packed.ndim == 1:
        packed = packed[None, :]
    bits = np.unpackbits(packed.view(np.uint8), axis=1, bitorder="little")
    return bits[:, :int(bit_dim)].astype(np.bool_, copy=False)


def hamming_to_one(A_packed: np.ndarray, b_packed: np.ndarray) -> np.ndarray:
    """Hamming distance from each row of A_packed to one packed vector b_packed."""
    A = np.asarray(A_packed, dtype=np.uint64)
    b = np.asarray(b_packed, dtype=np.uint64).reshape(1, -1)
    return _popcount_uint64(np.bitwise_xor(A, b)).sum(axis=1).astype(np.float32)


def hamming_matrix(A_packed: np.ndarray, B_packed: np.ndarray, b_chunk: int = 512) -> np.ndarray:
    """Dense Hamming distance matrix, chunked over B to limit peak memory."""
    A = np.asarray(A_packed, dtype=np.uint64)
    B = np.asarray(B_packed, dtype=np.uint64)
    out = np.empty((A.shape[0], B.shape[0]), dtype=np.float32)
    for j0 in range(0, B.shape[0], int(b_chunk)):
        j1 = min(j0 + int(b_chunk), B.shape[0])
        xor = np.bitwise_xor(A[:, None, :], B[None, j0:j1, :])
        out[:, j0:j1] = _popcount_uint64(xor).sum(axis=2).astype(np.float32)
    return out


def nearest_hamming_to_set(A_packed: np.ndarray, B_packed: np.ndarray, batch_size: int = 128, b_chunk: int = 512) -> np.ndarray:
    """Nearest Hamming distance from each row in A to a reference set B."""
    A = np.asarray(A_packed, dtype=np.uint64)
    out = np.empty(A.shape[0], dtype=np.float32)
    for i0 in range(0, A.shape[0], int(batch_size)):
        i1 = min(i0 + int(batch_size), A.shape[0])
        D = hamming_matrix(A[i0:i1], B_packed, b_chunk=b_chunk)
        out[i0:i1] = D.min(axis=1)
    return out


def topk_mean_hamming_to_set(A_packed: np.ndarray, B_packed: np.ndarray, k: int, batch_size: int = 64, b_chunk: int = 512) -> np.ndarray:
    """Mean top-k nearest Hamming distances from each row in A to B."""
    A = np.asarray(A_packed, dtype=np.uint64)
    B = np.asarray(B_packed, dtype=np.uint64)
    k = int(max(1, min(k, B.shape[0])))
    out = np.empty(A.shape[0], dtype=np.float32)
    for i0 in range(0, A.shape[0], int(batch_size)):
        i1 = min(i0 + int(batch_size), A.shape[0])
        D = hamming_matrix(A[i0:i1], B, b_chunk=b_chunk)
        topk = np.partition(D, k - 1, axis=1)[:, :k]
        out[i0:i1] = topk.mean(axis=1).astype(np.float32)
    return out


def leave_one_out_nearest_hamming_sample(H: np.ndarray, max_points: int = 3000, seed: int = 0,
                                         batch_size: int = 128, b_chunk: int = 512) -> np.ndarray:
    """Approximate local scale by nearest-neighbor Hamming distances in a sample."""
    H = np.asarray(H, dtype=np.uint64)
    n = H.shape[0]
    if n <= 1:
        return np.asarray([1.0], dtype=np.float32)
    rng = np.random.default_rng(seed)
    if n > max_points:
        idx = rng.choice(n, size=int(max_points), replace=False)
        S = H[idx]
    else:
        S = H
    out = np.empty(S.shape[0], dtype=np.float32)
    for i0 in range(0, S.shape[0], int(batch_size)):
        i1 = min(i0 + int(batch_size), S.shape[0])
        D = hamming_matrix(S[i0:i1], S, b_chunk=b_chunk)
        # Exclude self only when this is the full same array. For sampled rows,
        # row i0+r corresponds to column i0+r inside S.
        for r, ii in enumerate(range(i0, i1)):
            D[r, ii] = np.inf
        out[i0:i1] = D.min(axis=1)
    out = out[np.isfinite(out)]
    return out.astype(np.float32) if len(out) else np.asarray([1.0], dtype=np.float32)


class BinaryThermometerHDC:
    """Distance-preserving binary HDC encoder for real-valued wafer descriptors.

    Let y = xR be random projections.  Each projected coordinate is robustly
    standardized with median/IQR estimated on training data, clipped to
    [-clip_value, clip_value], and converted to a thermometer code with
    num_bins bits.  Hamming distance between two coordinates equals the absolute
    quantized projected difference, and averaging over random projections gives
    a binary distance sketch of Euclidean distance.
    """

    def __init__(self, in_dim: int, num_proj: int = 1024, num_bins: int = 8,
                 seed: int = 0, clip_value: float = 4.0, projection_scale: str = "sqrt_dim"):
        self.in_dim = int(in_dim)
        self.num_proj = int(num_proj)
        self.num_bins = int(num_bins)
        self.seed = int(seed)
        self.clip_value = float(clip_value)
        self.projection_scale = str(projection_scale)
        if self.num_proj <= 0 or self.num_bins <= 1:
            raise ValueError("num_proj must be positive and num_bins must be > 1")
        rng = np.random.default_rng(self.seed + 12345)
        W = rng.standard_normal((self.in_dim, self.num_proj)).astype(np.float32)
        if projection_scale == "sqrt_dim":
            W /= np.sqrt(max(self.in_dim, 1))
        elif projection_scale == "sqrt_proj":
            W /= np.sqrt(max(self.num_proj, 1))
        self.W = W
        self.median_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.bit_dim = self.num_proj * self.num_bins
        self.num_words = int((self.bit_dim + 63) // 64)

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        Y = X @ self.W
        med = np.median(Y, axis=0).astype(np.float32)
        q75 = np.percentile(Y, 75, axis=0).astype(np.float32)
        q25 = np.percentile(Y, 25, axis=0).astype(np.float32)
        scale = (q75 - q25) / 1.349
        std = np.std(Y, axis=0).astype(np.float32)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, std + 1e-6).astype(np.float32)
        self.median_ = med
        self.scale_ = scale
        return self

    def _levels(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.scale_ is None:
            raise RuntimeError("BinaryThermometerHDC must be fit before transform.")
        X = np.asarray(X, dtype=np.float32)
        Y = (X @ self.W - self.median_[None, :]) / (self.scale_[None, :] + 1e-6)
        Y = np.clip(Y, -self.clip_value, self.clip_value)
        # Map [-clip, clip] to integer levels 0..num_bins inclusive.
        levels = np.floor((Y + self.clip_value) / (2.0 * self.clip_value) * self.num_bins + 1e-6)
        return np.clip(levels, 0, self.num_bins).astype(np.int16)

    def transform(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        outs = []
        thresholds = np.arange(self.num_bins, dtype=np.int16)  # bit b is 1 iff level > b
        for i0 in range(0, X.shape[0], int(batch_size)):
            i1 = min(i0 + int(batch_size), X.shape[0])
            L = self._levels(X[i0:i1])
            bits = (thresholds[None, None, :] < L[:, :, None]).reshape(L.shape[0], self.bit_dim)
            outs.append(pack_bits_little(bits))
        return np.vstack(outs).astype(np.uint64)

    def fit_transform(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.fit(X)
        return self.transform(X, batch_size=batch_size)

    def meta(self) -> dict:
        return {
            "encoder": "BinaryThermometerHDC",
            "in_dim": self.in_dim,
            "num_proj": self.num_proj,
            "num_bins": self.num_bins,
            "bit_dim": self.bit_dim,
            "num_words_uint64": self.num_words,
            "clip_value": self.clip_value,
            "seed": self.seed,
        }
