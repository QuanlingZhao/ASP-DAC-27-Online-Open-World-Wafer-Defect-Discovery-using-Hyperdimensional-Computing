#!/usr/bin/env python3
"""
Stage-1 evaluation for the binary-HDC normal-only wafer gate.

This script is intentionally hard-coded for the paper evaluation setting. It
reuses the implementation in run_full_hdc_stage2.py and evaluates only the
Stage-1 normal/non-defect screening gate on validation, test, and online stream
splits.

Expected directory layout:
  project_root/
    run_full_hdc_stage2.py
    wafer_data_prep_hdc_stage2.py
    wafer_hdc_encoding.py
    eval_stage1_binary_hdc_hardcoded.py
    data/LSWMD.pkl
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

from wafer_data_prep_hdc_stage2 import prepare_normal_only_phased_data
from wafer_hdc_encoding import BinaryThermometerHDC
from run_full_hdc import (
    BinaryHDCGate,
    evaluate_stage1,
    json_default,
    print_metrics,
    set_seed,
    gather_stream_arrays,
)


@dataclass(frozen=True)
class Stage1EvalConfig:
    # Paths
    wm811k_pkl: str = "./data/LSWMD.pkl"
    save_dir: str = "./results_stage1_binary_hdc_clip3"
    cache_dir: str = "./aux_feature_cache_stage1"
    overwrite_cache: bool = False

    # Data split / preprocessing
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 0
    none_cap: int = 5000
    image_size: int = 64
    batch_size: int = 256
    num_workers: int = 4
    clip_low_q: float = 1.0
    clip_high_q: float = 99.0
    defect_threshold: float = 0.5
    invalid_value: float = -1.0

    # Binary HDC encoder
    binary_num_proj: int = 1024
    binary_num_bins: int = 8
    binary_clip_value: float = 3.0
    binary_encode_batch_size: int = 1024

    # Stage-1 normal gate
    known_reject_q: float = 10.0
    num_subprototypes: int = 10
    subproto_max_iter: int = 8
    local_k: int = 10
    max_ref_normal: int = 2000
    local_score_batch_size: int = 64


CFG = Stage1EvalConfig()


def class_counts(y: np.ndarray, label_names: list[str]) -> Dict[str, int]:
    vals, cnts = np.unique(np.asarray(y, dtype=np.int64), return_counts=True)
    out: Dict[str, int] = {}
    for v, c in zip(vals, cnts):
        name = label_names[int(v)] if int(v) < len(label_names) else str(int(v))
        out[name] = int(c)
    return out


def run_stage1_eval(cfg: Stage1EvalConfig = CFG) -> Dict[str, Any]:
    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)

    print("Loading fixed normal-only phased-stream data...")
    data = prepare_normal_only_phased_data(
        wm811k_pkl=cfg.wm811k_pkl,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        seed=cfg.seed,
        none_cap=cfg.none_cap,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        defect_threshold=cfg.defect_threshold,
        invalid_value=cfg.invalid_value,
        clip_low_q=cfg.clip_low_q,
        clip_high_q=cfg.clip_high_q,
        feature_cache_dir=cfg.cache_dir,
        overwrite_feature_cache=cfg.overwrite_cache,
        return_wafer_maps=False,
        aux_feature_set="original",
    )

    print("\nEncoding 202-D auxiliary features into binary thermometer HDC vectors...")
    enc = BinaryThermometerHDC(
        in_dim=data.X_aux_train.shape[1],
        num_proj=cfg.binary_num_proj,
        num_bins=cfg.binary_num_bins,
        seed=cfg.seed,
        clip_value=cfg.binary_clip_value,
    )

    t_encode = time.time()
    H_train = enc.fit_transform(data.X_aux_train, batch_size=cfg.binary_encode_batch_size)
    H_val = enc.transform(data.X_aux_val, batch_size=cfg.binary_encode_batch_size)
    H_test = enc.transform(data.X_aux_test, batch_size=cfg.binary_encode_batch_size)
    encode_time_sec = time.time() - t_encode
    print(f"Binary HDC encoding time: {encode_time_sec:.2f}s")
    print(f"Binary HDC: {enc.bit_dim} bits = {enc.num_words} uint64 words per wafer")

    print("\nFitting Stage-1 binary-HDC normal gate...")
    stage1 = BinaryHDCGate(
        bit_dim=enc.bit_dim,
        num_subprototypes=cfg.num_subprototypes,
        subproto_max_iter=cfg.subproto_max_iter,
        local_k=cfg.local_k,
        max_ref_normal=cfg.max_ref_normal,
        score_batch_size=cfg.local_score_batch_size,
        seed=cfg.seed,
    )

    t_fit = time.time()
    stage1.fit(H_train[data.train_known_mask])
    fit_time_sec = time.time() - t_fit

    t_cal = time.time()
    calibration = stage1.calibrate(H_val[data.val_known_mask], known_reject_q=cfg.known_reject_q)
    calibration_time_sec = time.time() - t_cal
    print("Stage-1 calibration:", calibration)

    print("\nEvaluating Stage-1 on validation/test/stream...")
    t_eval = time.time()
    val_metrics = evaluate_stage1(stage1, H_val, data.y_val, data.known_ids, data.unknown_ids)
    test_metrics = evaluate_stage1(stage1, H_test, data.y_test, data.known_ids, data.unknown_ids)

    H_stream = gather_stream_arrays(data.stream_sources, data.stream_indices, H_train, H_val, H_test)
    stream_metrics = evaluate_stage1(stage1, H_stream, data.y_stream_true, data.known_ids, data.unknown_ids)
    stream_scores = stage1.predict_scores(H_stream)
    eval_time_sec = time.time() - t_eval

    print_metrics("STAGE-1 STATIC VALIDATION", val_metrics)
    print_metrics("STAGE-1 STATIC TEST", test_metrics)
    print_metrics("STAGE-1 ONLINE STREAM", stream_metrics)

    result: Dict[str, Any] = {
        "method": "stage1_binary_thermometer_hdc_normal_gate",
        "description": (
            "Stage-1 normal-only wafer screening using original 202-D auxiliary "
            "features encoded into distance-preserving binary thermometer HDC. "
            "The gate combines nearest normal subprototype Hamming distance and "
            "local top-k normal-reference Hamming distance."
        ),
        "config": asdict(cfg),
        "binary_hdc_encoder": enc.meta(),
        "label_names": data.label_names,
        "normal_label_id": int(data.normal_label),
        "known_ids": data.known_ids,
        "known_names": data.known_names,
        "unknown_ids": data.unknown_ids,
        "unknown_names": data.unknown_names,
        "feature_names": data.feature_names,
        "split_class_counts": {
            "train": class_counts(data.y_train, data.label_names),
            "val": class_counts(data.y_val, data.label_names),
            "test": class_counts(data.y_test, data.label_names),
            "stream": class_counts(data.y_stream_true, data.label_names),
        },
        "stage1_calibration": calibration,
        "static_validation": val_metrics,
        "static_test": test_metrics,
        "online_stream": stream_metrics,
        "timing": {
            "binary_encoding_time_sec": float(encode_time_sec),
            "stage1_fit_time_sec": float(fit_time_sec),
            "stage1_calibration_time_sec": float(calibration_time_sec),
            "stage1_eval_time_sec": float(eval_time_sec),
        },
    }

    out_json = os.path.join(cfg.save_dir, "stage1_eval_results.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=json_default)

    # Compact numerical dump for later plotting/debugging.
    np.savez_compressed(
        os.path.join(cfg.save_dir, "stage1_stream_scores.npz"),
        y_stream_true=np.asarray(data.y_stream_true, dtype=np.int64),
        stage1_accepted=np.asarray(stream_scores["accepted"], dtype=bool),
        binary_gate_score=np.asarray(stream_scores["binary_gate_score"], dtype=np.float32),
        binary_global_hamming=np.asarray(stream_scores["binary_global_hamming"], dtype=np.float32),
        binary_local_topk_hamming=np.asarray(stream_scores["binary_local_topk_hamming"], dtype=np.float32),
    )

    print(f"\nSaved Stage-1 outputs to: {cfg.save_dir}")
    return result


if __name__ == "__main__":
    run_stage1_eval(CFG)
