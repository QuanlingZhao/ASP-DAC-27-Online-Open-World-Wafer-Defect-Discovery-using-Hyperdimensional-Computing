#!/usr/bin/env python3
"""
wafer_data_prep.py

Fixed data preparation for the normal-only phased-stream wafer setting.

This file intentionally removes experiment-setting hyperparameters:
  - setup is always normal_only
  - known class is always the inferred normal/none label
  - all non-normal labels are unknown
  - online stream is always phased Protocol A
  - train/val/test defect samples are included in the stream as unlabeled
    unknown samples, while train/val/test normals keep their standard roles

The only remaining arguments are data/loading mechanics such as path, split,
seed, image size, batch size, and worker count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import hashlib
import json
import os
import numpy as np
from PIL import Image

from get_data import get_data
from wafer_hdc_encoding import (
    dataset_to_arrays,
    fit_clip_stats,
    apply_clip,
    fit_standardize,
    apply_standardize,
    RealMorphologySpatialEncoder,
)

NORMAL_NAMES = {"none", "normal", "no defect", "non-patterned", "random-none"}
DEFAULT_DEFECT_ORDER = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"
]


@dataclass
class WaferDataBundle:
    label_names: List[str]
    normal_label: int
    known_ids: List[int]
    unknown_ids: List[int]
    known_names: List[str]
    unknown_names: List[str]

    W_train: np.ndarray
    W_val: np.ndarray
    W_test: np.ndarray
    X_aux_train: np.ndarray
    X_aux_val: np.ndarray
    X_aux_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]

    train_known_mask: np.ndarray
    val_known_mask: np.ndarray
    test_static_mask: np.ndarray

    stream_sources: np.ndarray
    stream_indices: np.ndarray
    y_stream_true: np.ndarray


def get_label_names(le) -> List[str]:
    if le is not None and hasattr(le, "classes_"):
        return [str(x) for x in le.classes_]
    return []


def infer_normal_label_id(le) -> int:
    names = get_label_names(le)
    if not names:
        return 0
    for idx, name in enumerate(names):
        if str(name).strip().lower() in NORMAL_NAMES:
            return idx
    for idx, name in enumerate(names):
        if "none" in str(name).strip().lower():
            return idx
    print("[WARN] Could not confidently infer normal label; falling back to 0.")
    return 0


def _subsample_indices(indices: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if frac >= 1.0:
        return indices
    if frac <= 0.0 or len(indices) == 0:
        return np.asarray([], dtype=np.int64)
    return indices[rng.random(len(indices)) <= frac]


def _parse_defect_order(label_names: List[str], unknown_ids: List[int]) -> List[int]:
    name_to_id = {str(n).strip().lower(): i for i, n in enumerate(label_names)}
    out = []
    for name in DEFAULT_DEFECT_ORDER:
        cid = name_to_id.get(name.lower(), None)
        if cid is not None and cid in set(unknown_ids):
            out.append(int(cid))
    for cid in unknown_ids:
        if int(cid) not in out:
            out.append(int(cid))
    return out


def build_fixed_normal_only_phased_stream(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    known_ids: List[int],
    unknown_ids: List[int],
    label_names: List[str],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fixed Protocol A stream:
      - test normals only for normal stream
      - defects from test/train/val included as unlabeled unknown stream samples
      - phased defect order
      - normals split approximately evenly across defect phases
    """
    known_set = set(int(x) for x in known_ids)
    normal_test_idx = np.where(np.isin(y_test, list(known_set)))[0]

    defect_pools: Dict[int, List[Tuple[str, int]]] = {int(c): [] for c in unknown_ids}

    def add_defects(split_name: str, y: np.ndarray):
        for c in unknown_ids:
            idxs = np.where(y == int(c))[0]
            for idx in idxs:
                defect_pools[int(c)].append((split_name, int(idx)))

    # Fixed full stress stream: all defect samples are treated as unlabeled stream inputs.
    add_defects("test", y_test)
    add_defects("train", y_train)
    add_defects("val", y_val)

    for c in defect_pools:
        rng.shuffle(defect_pools[c])

    order = _parse_defect_order(label_names, unknown_ids)
    normal_test_idx = np.asarray(normal_test_idx, dtype=np.int64)
    rng.shuffle(normal_test_idx)
    normal_chunks = np.array_split(normal_test_idx, max(len(order), 1))

    items: List[Tuple[str, int]] = []
    for phase_i, c in enumerate(order):
        phase_items = [("test", int(i)) for i in normal_chunks[phase_i]]
        phase_items.extend(defect_pools[int(c)])
        rng.shuffle(phase_items)
        items.extend(phase_items)

    stream_sources = np.asarray([s for s, _ in items], dtype=object)
    stream_indices = np.asarray([i for _, i in items], dtype=np.int64)

    def label_of(src: str, idx: int) -> int:
        if src == "train":
            return int(y_train[idx])
        if src == "val":
            return int(y_val[idx])
        return int(y_test[idx])

    y_stream_true = np.asarray(
        [label_of(s, int(i)) for s, i in zip(stream_sources, stream_indices)],
        dtype=np.int64,
    )
    return stream_sources, stream_indices, y_stream_true



def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _short_hash(obj: Dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]


def _resize_raw_wafer_map(wafer_map: np.ndarray, image_size: int) -> np.ndarray:
    """Resize an original 2D WM-811K wafer map to image_size x image_size.

    This keeps the discrete wafer codes with nearest-neighbor interpolation. It is
    separate from dataset_to_arrays(), which extracts auxiliary features from the
    dataset tensor representation and is kept unchanged for backward-compatible
    aux-feature results.
    """
    a = np.asarray(wafer_map)
    if a.ndim != 2:
        raise ValueError(f"Expected raw 2D wafer map, got shape={a.shape}")
    if a.shape == (int(image_size), int(image_size)):
        return a.astype(np.float32)
    try:
        resample = Image.Resampling.NEAREST
    except AttributeError:  # Pillow < 9
        resample = Image.NEAREST
    img = Image.fromarray(a.astype(np.float32))
    img = img.resize((int(image_size), int(image_size)), resample=resample)
    return np.asarray(img, dtype=np.float32)


def dataset_to_raw_wafer_maps(ds, image_size: int) -> np.ndarray:
    """Return resized raw wafer maps from WaferDataset.

    WaferDataset keeps the original wafer maps in ds.wafer_maps. Using this field
    avoids reconstructing maps from normalized RGB tensors and gives the spatial
    HDC encoder the same discrete wafer-map codes used by the data loader.
    """
    if not hasattr(ds, "wafer_maps"):
        raise ValueError("Dataset does not expose raw wafer_maps; cannot build spatial HDC inputs.")
    return np.stack([_resize_raw_wafer_map(w, image_size) for w in ds.wafer_maps], axis=0).astype(np.float32)


def _feature_cache_prefix(
    wm811k_pkl: str,
    test_size: float,
    val_size: float,
    seed: int,
    none_cap: int,
    image_size: int,
    defect_threshold: float,
    invalid_value: float,
    aux_feature_set: str = "original",
    morph_defect_threshold: float = 1.5,
    morph_invalid_value: float = 0.0,
    morph_radial_bins: int = 10,
    morph_angular_bins: int = 16,
    morph_polar_radial_bins: int = 5,
    morph_polar_angular_bins: int = 16,
    morph_edge_bins: int = 8,
    morph_projection_bins: int = 16,
    morph_top_components: int = 5,
) -> str:
    """Cache key for raw extracted auxiliary features.

    This intentionally does NOT include clipping/standardization parameters or
    RP/encoding parameters, because the cache stores raw extracted features
    before preprocessing and before encoding.  It *does* include the feature-set
    choice because the original 202-D aux descriptors and the morphology-only
    descriptors are different raw feature spaces.
    """
    aux_feature_set = str(aux_feature_set).strip().lower()
    sig = {
        "version": "normal_only_aux_feature_cache_v2_selectable_aux_features",
        "wm811k_pkl": os.path.abspath(wm811k_pkl),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "seed": int(seed),
        "none_cap": int(none_cap),
        "image_size": int(image_size),
        "defect_threshold": float(defect_threshold),
        "invalid_value": float(invalid_value),
        "aux_feature_set": aux_feature_set,
        "morph_defect_threshold": float(morph_defect_threshold),
        "morph_invalid_value": float(morph_invalid_value),
        "morph_radial_bins": int(morph_radial_bins),
        "morph_angular_bins": int(morph_angular_bins),
        "morph_polar_radial_bins": int(morph_polar_radial_bins),
        "morph_polar_angular_bins": int(morph_polar_angular_bins),
        "morph_edge_bins": int(morph_edge_bins),
        "morph_projection_bins": int(morph_projection_bins),
        "morph_top_components": int(morph_top_components),
    }
    return f"wm811k_{aux_feature_set}_aux_features_{_short_hash(sig)}"


def _extract_morphology_aux_features_from_raw_maps(
    W_train: np.ndarray,
    W_val: np.ndarray,
    W_test: np.ndarray,
    image_size: int,
    defect_threshold: float,
    invalid_value: float,
    radial_bins: int = 10,
    angular_bins: int = 16,
    polar_radial_bins: int = 5,
    polar_angular_bins: int = 16,
    edge_bins: int = 8,
    projection_bins: int = 16,
    top_components: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extract the morphology descriptor set as the new auxiliary features.

    This deliberately returns *raw scalar descriptors*, not a spatial-HDC vector.
    Normal-only clipping/standardization and the real-valued random projection are
    applied later in the standard aux pipeline.
    """
    enc = RealMorphologySpatialEncoder(
        image_size=image_size,
        out_dim=1,  # unused when calling extract_features_batch()
        seed=0,
        defect_threshold=defect_threshold,
        invalid_value=invalid_value,
        radial_bins=radial_bins,
        angular_bins=angular_bins,
        polar_radial_bins=polar_radial_bins,
        polar_angular_bins=polar_angular_bins,
        edge_bins=edge_bins,
        projection_bins=projection_bins,
        top_components=top_components,
        batch_size=256,
        normalize_output=False,
        standardize=False,
    )
    print("Extracting morphology descriptor auxiliary features...")
    X_train = enc.extract_features_batch(W_train)
    X_val = enc.extract_features_batch(W_val)
    X_test = enc.extract_features_batch(W_test)
    names = [f"morph_{n}" for n in enc.feature_names]
    print(f"  morphology aux feature dim: {len(names)}")
    return X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32), names


def _load_or_extract_aux_feature_arrays(
    train_ds,
    val_ds,
    test_ds,
    cache_dir: str,
    prefix: str,
    defect_threshold: float,
    invalid_value: float,
    overwrite_cache: bool = False,
    return_wafer_maps: bool = False,
    image_size: int = 64,
    aux_feature_set: str = "original",
    morph_defect_threshold: float = 1.5,
    morph_invalid_value: float = 0.0,
    morph_radial_bins: int = 10,
    morph_angular_bins: int = 16,
    morph_polar_radial_bins: int = 5,
    morph_polar_angular_bins: int = 16,
    morph_edge_bins: int = 8,
    morph_projection_bins: int = 16,
    morph_top_components: int = 5,
):
    """Load/save raw extracted auxiliary features and labels.

    We do not cache RP encodings here. The caller still performs clipping,
    standardization, and random-projection encoding every run.
    """
    aux_feature_set = str(aux_feature_set).strip().lower()
    if aux_feature_set not in {"original", "morph", "original_plus_morph"}:
        raise ValueError(f"Unknown aux_feature_set={aux_feature_set!r}. Use original, morph, or original_plus_morph.")

    def build_selected_features(X_orig_train, X_orig_val, X_orig_test, orig_names, Wraw_train, Wraw_val, Wraw_test):
        if aux_feature_set == "original":
            return X_orig_train, X_orig_val, X_orig_test, list(orig_names), None
        Xm_train, Xm_val, Xm_test, morph_names = _extract_morphology_aux_features_from_raw_maps(
            Wraw_train, Wraw_val, Wraw_test,
            image_size=image_size,
            defect_threshold=morph_defect_threshold,
            invalid_value=morph_invalid_value,
            radial_bins=morph_radial_bins,
            angular_bins=morph_angular_bins,
            polar_radial_bins=morph_polar_radial_bins,
            polar_angular_bins=morph_polar_angular_bins,
            edge_bins=morph_edge_bins,
            projection_bins=morph_projection_bins,
            top_components=morph_top_components,
        )
        if aux_feature_set == "morph":
            return Xm_train, Xm_val, Xm_test, morph_names, {"morph_feature_dim": len(morph_names)}
        names = [f"orig_{n}" for n in orig_names] + morph_names
        return (
            np.concatenate([X_orig_train, Xm_train], axis=1).astype(np.float32),
            np.concatenate([X_orig_val, Xm_val], axis=1).astype(np.float32),
            np.concatenate([X_orig_test, Xm_test], axis=1).astype(np.float32),
            names,
            {"morph_feature_dim": len(morph_names), "original_feature_dim": len(orig_names)},
        )

    if cache_dir is None or str(cache_dir).strip() == "":
        print("Extracting original train arrays...")
        W_train0, X_orig_train, y_train, orig_names = dataset_to_arrays(train_ds, defect_threshold, invalid_value)
        print("Extracting original val arrays...")
        W_val0, X_orig_val, y_val, _ = dataset_to_arrays(val_ds, defect_threshold, invalid_value)
        print("Extracting original test arrays...")
        W_test0, X_orig_test, y_test, _ = dataset_to_arrays(test_ds, defect_threshold, invalid_value)

        need_raw_maps = return_wafer_maps or aux_feature_set in {"morph", "original_plus_morph"}
        if need_raw_maps:
            msg = "Extracting raw wafer maps for morphology aux features" if aux_feature_set != "original" else "Extracting raw wafer maps"
            print(msg + "...")
            Wraw_train = dataset_to_raw_wafer_maps(train_ds, image_size)
            Wraw_val = dataset_to_raw_wafer_maps(val_ds, image_size)
            Wraw_test = dataset_to_raw_wafer_maps(test_ds, image_size)
        else:
            Wraw_train, Wraw_val, Wraw_test = W_train0, W_val0, W_test0

        X_aux_train, X_aux_val, X_aux_test, feature_names, _ = build_selected_features(
            X_orig_train, X_orig_val, X_orig_test, orig_names, Wraw_train, Wraw_val, Wraw_test
        )
        W_train = Wraw_train if return_wafer_maps else np.empty((len(y_train), 0, 0), dtype=np.float32)
        W_val = Wraw_val if return_wafer_maps else np.empty((len(y_val), 0, 0), dtype=np.float32)
        W_test = Wraw_test if return_wafer_maps else np.empty((len(y_test), 0, 0), dtype=np.float32)
        return W_train, W_val, W_test, X_aux_train, X_aux_val, X_aux_test, y_train, y_val, y_test, feature_names, False

    os.makedirs(cache_dir, exist_ok=True)
    paths = {
        "X_aux_train": os.path.join(cache_dir, f"{prefix}_X_aux_train_raw.npy"),
        "X_aux_val": os.path.join(cache_dir, f"{prefix}_X_aux_val_raw.npy"),
        "X_aux_test": os.path.join(cache_dir, f"{prefix}_X_aux_test_raw.npy"),
        "y_train": os.path.join(cache_dir, f"{prefix}_y_train.npy"),
        "y_val": os.path.join(cache_dir, f"{prefix}_y_val.npy"),
        "y_test": os.path.join(cache_dir, f"{prefix}_y_test.npy"),
        "feature_names": os.path.join(cache_dir, f"{prefix}_feature_names.json"),
    }
    wafer_paths = {
        "W_train": os.path.join(cache_dir, f"{prefix}_W_train_raw{int(image_size)}.npy"),
        "W_val": os.path.join(cache_dir, f"{prefix}_W_val_raw{int(image_size)}.npy"),
        "W_test": os.path.join(cache_dir, f"{prefix}_W_test_raw{int(image_size)}.npy"),
    }
    have_cache = all(os.path.exists(v) for v in paths.values())
    if have_cache and not overwrite_cache:
        print("Loading cached extracted selected auxiliary features...")
        print(f"  feature cache prefix: {prefix}")
        X_aux_train = np.load(paths["X_aux_train"])
        X_aux_val = np.load(paths["X_aux_val"])
        X_aux_test = np.load(paths["X_aux_test"])
        y_train = np.load(paths["y_train"])
        y_val = np.load(paths["y_val"])
        y_test = np.load(paths["y_test"])
        with open(paths["feature_names"], "r") as f:
            feature_names = json.load(f)
        if return_wafer_maps:
            have_w_cache = all(os.path.exists(v) for v in wafer_paths.values())
            if have_w_cache:
                print("Loading cached raw wafer maps for spatial HDC...")
                W_train = np.load(wafer_paths["W_train"])
                W_val = np.load(wafer_paths["W_val"])
                W_test = np.load(wafer_paths["W_test"])
            else:
                print("Aux feature cache found, but raw wafer-map cache is missing.")
                print("Extracting raw wafer maps for spatial HDC...")
                W_train = dataset_to_raw_wafer_maps(train_ds, image_size)
                W_val = dataset_to_raw_wafer_maps(val_ds, image_size)
                W_test = dataset_to_raw_wafer_maps(test_ds, image_size)
                np.save(wafer_paths["W_train"], W_train.astype(np.float32))
                np.save(wafer_paths["W_val"], W_val.astype(np.float32))
                np.save(wafer_paths["W_test"], W_test.astype(np.float32))
        else:
            # Wafer images are not needed by aux-only Stage 1. Return empty placeholders
            # with correct leading dimensions to keep the existing data bundle interface.
            W_train = np.empty((len(y_train), 0, 0), dtype=np.float32)
            W_val = np.empty((len(y_val), 0, 0), dtype=np.float32)
            W_test = np.empty((len(y_test), 0, 0), dtype=np.float32)
        return W_train, W_val, W_test, X_aux_train, X_aux_val, X_aux_test, y_train, y_val, y_test, feature_names, True

    print("Extracting original train arrays...")
    W_train0, X_orig_train, y_train, orig_names = dataset_to_arrays(train_ds, defect_threshold, invalid_value)
    print("Extracting original val arrays...")
    W_val0, X_orig_val, y_val, _ = dataset_to_arrays(val_ds, defect_threshold, invalid_value)
    print("Extracting original test arrays...")
    W_test0, X_orig_test, y_test, _ = dataset_to_arrays(test_ds, defect_threshold, invalid_value)

    need_raw_maps = return_wafer_maps or aux_feature_set in {"morph", "original_plus_morph"}
    if need_raw_maps:
        msg = "Extracting raw wafer maps for morphology aux features" if aux_feature_set != "original" else "Extracting raw wafer maps"
        print(msg + "...")
        Wraw_train = dataset_to_raw_wafer_maps(train_ds, image_size)
        Wraw_val = dataset_to_raw_wafer_maps(val_ds, image_size)
        Wraw_test = dataset_to_raw_wafer_maps(test_ds, image_size)
    else:
        Wraw_train, Wraw_val, Wraw_test = W_train0, W_val0, W_test0

    X_aux_train, X_aux_val, X_aux_test, feature_names, _ = build_selected_features(
        X_orig_train, X_orig_val, X_orig_test, orig_names, Wraw_train, Wraw_val, Wraw_test
    )
    W_train = Wraw_train if return_wafer_maps else np.empty((len(y_train), 0, 0), dtype=np.float32)
    W_val = Wraw_val if return_wafer_maps else np.empty((len(y_val), 0, 0), dtype=np.float32)
    W_test = Wraw_test if return_wafer_maps else np.empty((len(y_test), 0, 0), dtype=np.float32)

    print("Saving extracted auxiliary feature cache...")
    print(f"  feature cache prefix: {prefix}")
    np.save(paths["X_aux_train"], X_aux_train.astype(np.float32))
    np.save(paths["X_aux_val"], X_aux_val.astype(np.float32))
    np.save(paths["X_aux_test"], X_aux_test.astype(np.float32))
    np.save(paths["y_train"], y_train.astype(np.int64))
    np.save(paths["y_val"], y_val.astype(np.int64))
    np.save(paths["y_test"], y_test.astype(np.int64))
    with open(paths["feature_names"], "w") as f:
        json.dump(feature_names, f, indent=2)
    if return_wafer_maps:
        print("Saving raw wafer-map cache for spatial HDC...")
        np.save(wafer_paths["W_train"], W_train.astype(np.float32))
        np.save(wafer_paths["W_val"], W_val.astype(np.float32))
        np.save(wafer_paths["W_test"], W_test.astype(np.float32))
    return W_train, W_val, W_test, X_aux_train, X_aux_val, X_aux_test, y_train, y_val, y_test, feature_names, False

def prepare_normal_only_phased_data(
    wm811k_pkl: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 0,
    none_cap: int = 5000,
    image_size: int = 64,
    batch_size: int = 256,
    num_workers: int = 4,
    defect_threshold: float = 0.5,
    invalid_value: float = -1.0,
    clip_low_q: float = 1.0,
    clip_high_q: float = 99.0,
    feature_cache_dir: str = "",
    overwrite_feature_cache: bool = False,
    return_wafer_maps: bool = False,
    aux_feature_set: str = "original",
    morph_defect_threshold: float = 1.5,
    morph_invalid_value: float = 0.0,
    morph_radial_bins: int = 10,
    morph_angular_bins: int = 16,
    morph_polar_radial_bins: int = 5,
    morph_polar_angular_bins: int = 16,
    morph_edge_bins: int = 8,
    morph_projection_bins: int = 16,
    morph_top_components: int = 5,
) -> WaferDataBundle:
    rng = np.random.default_rng(seed)

    train_ds, val_ds, test_ds, *_loaders, le = get_data(
        wm811k_pkl,
        test_size,
        val_size,
        seed,
        none_cap,
        image_size,
        batch_size,
        num_workers,
    )

    label_names = get_label_names(le)
    normal_label = infer_normal_label_id(le)
    known_ids = [int(normal_label)]
    unknown_ids = [i for i in range(len(label_names)) if i != int(normal_label)]
    known_names = [label_names[i] for i in known_ids]
    unknown_names = [label_names[i] for i in unknown_ids]

    print("Fixed setup: normal_only")
    print("Known class IDs:", known_ids)
    print("Known class names:", known_names)
    print("Unknown class IDs:", unknown_ids)
    print("Unknown class names:", unknown_names)

    aux_feature_set = str(aux_feature_set).strip().lower()
    print("Aux feature set:", aux_feature_set)
    if aux_feature_set in {"morph", "original_plus_morph"}:
        print("Morph aux raw-map threshold/invalid:", morph_defect_threshold, morph_invalid_value)

    feature_cache_prefix = _feature_cache_prefix(
        wm811k_pkl=wm811k_pkl,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        none_cap=none_cap,
        image_size=image_size,
        defect_threshold=defect_threshold,
        invalid_value=invalid_value,
        aux_feature_set=aux_feature_set,
        morph_defect_threshold=morph_defect_threshold,
        morph_invalid_value=morph_invalid_value,
        morph_radial_bins=morph_radial_bins,
        morph_angular_bins=morph_angular_bins,
        morph_polar_radial_bins=morph_polar_radial_bins,
        morph_polar_angular_bins=morph_polar_angular_bins,
        morph_edge_bins=morph_edge_bins,
        morph_projection_bins=morph_projection_bins,
        morph_top_components=morph_top_components,
    )
    if feature_cache_dir:
        print("Feature cache enabled:", feature_cache_dir)
        print("Feature cache prefix:", feature_cache_prefix)

    (
        W_train, W_val, W_test,
        X_aux_train, X_aux_val, X_aux_test,
        y_train, y_val, y_test,
        feature_names, loaded_feature_cache,
    ) = _load_or_extract_aux_feature_arrays(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        cache_dir=feature_cache_dir,
        prefix=feature_cache_prefix,
        defect_threshold=defect_threshold,
        invalid_value=invalid_value,
        overwrite_cache=overwrite_feature_cache,
        return_wafer_maps=return_wafer_maps,
        image_size=image_size,
        aux_feature_set=aux_feature_set,
        morph_defect_threshold=morph_defect_threshold,
        morph_invalid_value=morph_invalid_value,
        morph_radial_bins=morph_radial_bins,
        morph_angular_bins=morph_angular_bins,
        morph_polar_radial_bins=morph_polar_radial_bins,
        morph_polar_angular_bins=morph_polar_angular_bins,
        morph_edge_bins=morph_edge_bins,
        morph_projection_bins=morph_projection_bins,
        morph_top_components=morph_top_components,
    )

    print("Train shapes:", W_train.shape, X_aux_train.shape)
    print("Val shapes  :", W_val.shape, X_aux_val.shape)
    print("Test shapes :", W_test.shape, X_aux_test.shape)
    print("Selected aux feature dim:", int(X_aux_train.shape[1]))

    train_known_mask = np.isin(y_train, known_ids)
    val_known_mask = np.isin(y_val, known_ids)
    if not np.any(train_known_mask):
        raise RuntimeError("No normal/known samples found in training set.")

    # Normal-only preprocessing: fit clip/standardization only on train normals.
    clip_lo, clip_hi = fit_clip_stats(X_aux_train[train_known_mask], clip_low_q, clip_high_q)
    X_aux_train = apply_clip(X_aux_train, clip_lo, clip_hi)
    X_aux_val = apply_clip(X_aux_val, clip_lo, clip_hi)
    X_aux_test = apply_clip(X_aux_test, clip_lo, clip_hi)

    mu, sigma = fit_standardize(X_aux_train[train_known_mask])
    X_aux_train = apply_standardize(X_aux_train, mu, sigma)
    X_aux_val = apply_standardize(X_aux_val, mu, sigma)
    X_aux_test = apply_standardize(X_aux_test, mu, sigma)

    stream_sources, stream_indices, y_stream_true = build_fixed_normal_only_phased_stream(
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        known_ids=known_ids,
        unknown_ids=unknown_ids,
        label_names=label_names,
        rng=rng,
    )

    print("Fixed online stream construction:")
    print("  setup: normal_only")
    print("  stream_mode: phased")
    print("  include_train_val_test_defects_as_unlabeled_stream: True")
    print("  stream size:", int(len(y_stream_true)))
    for cid, cnt in zip(*np.unique(y_stream_true, return_counts=True)):
        cname = label_names[int(cid)] if int(cid) < len(label_names) else str(cid)
        print(f"  stream {cname:>12s}: {int(cnt)}")

    return WaferDataBundle(
        label_names=label_names,
        normal_label=int(normal_label),
        known_ids=known_ids,
        unknown_ids=unknown_ids,
        known_names=known_names,
        unknown_names=unknown_names,
        W_train=W_train,
        W_val=W_val,
        W_test=W_test,
        X_aux_train=X_aux_train,
        X_aux_val=X_aux_val,
        X_aux_test=X_aux_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        train_known_mask=train_known_mask,
        val_known_mask=val_known_mask,
        test_static_mask=np.isin(y_test, known_ids + unknown_ids),
        stream_sources=stream_sources,
        stream_indices=stream_indices,
        y_stream_true=y_stream_true,
    )


def gather_stream_arrays(
    stream_sources: np.ndarray,
    stream_indices: np.ndarray,
    H_train: np.ndarray,
    H_val: np.ndarray,
    H_test: np.ndarray,
    H_aux_train: np.ndarray,
    H_aux_val: np.ndarray,
    H_aux_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    H_list = []
    H_aux_list = []
    for src, idx in zip(stream_sources, stream_indices):
        idx = int(idx)
        if src == "train":
            H_list.append(H_train[idx])
            H_aux_list.append(H_aux_train[idx])
        elif src == "val":
            H_list.append(H_val[idx])
            H_aux_list.append(H_aux_val[idx])
        else:
            H_list.append(H_test[idx])
            H_aux_list.append(H_aux_test[idx])
    return np.stack(H_list, axis=0).astype(np.float32), np.stack(H_aux_list, axis=0).astype(np.float32)
