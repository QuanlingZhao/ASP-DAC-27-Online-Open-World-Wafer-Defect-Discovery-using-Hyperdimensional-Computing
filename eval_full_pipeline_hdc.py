#!/usr/bin/env python3
"""
Stage-2 / full-pipeline evaluation for binary-HDC online wafer discovery.

This script is intentionally hard-coded for the paper evaluation setting. It
runs the full pipeline:

  202D aux features -> binary HDC -> Stage-1 normal gate -> Stage-2 online
  temporary/novel HDC memory -> end-to-end open-world metrics.

Expected directory layout:
  project_root/
    run_full_hdc_stage2.py
    wafer_data_prep_hdc_stage2.py
    wafer_hdc_encoding.py
    eval_full_pipeline_binary_hdc_hardcoded.py
    data/LSWMD.pkl
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np

from wafer_data_prep_hdc_stage2 import prepare_normal_only_phased_data
from wafer_hdc_encoding import BinaryThermometerHDC
from run_full_hdc import (
    BinaryHDCGate,
    BinaryHDCOnlineClusterer,
    classwise_stage2_behavior,
    evaluate_stage1,
    evaluate_stage2,
    gather_stream_arrays,
    json_default,
    print_metrics,
    set_seed,
    stage2_auto_thresholds_binary,
)


@dataclass(frozen=True)
class Stage2FullPipelineConfig:
    # Paths
    wm811k_pkl: str = "./data/LSWMD.pkl"
    save_dir: str = "./results_full_pipeline_binary_hdc_clip3_s2_aggressive"
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

    # Binary HDC encoder — best Stage-1 setting
    binary_num_proj: int = 1024
    binary_num_bins: int = 8
    binary_clip_value: float = 3.0
    binary_encode_batch_size: int = 1024

    # Stage-1 normal gate — fixed best Stage-1 setting
    known_reject_q: float = 10.0
    num_subprototypes: int = 10
    subproto_max_iter: int = 8
    local_k: int = 10
    max_ref_normal: int = 2000
    local_score_batch_size: int = 64

    # Stage-2 online discovery — current best aggressive setting
    stage2_min_cluster_size: int = 30
    stage2_min_cluster_age: int = 0
    stage2_tau_cluster: float | None = None
    stage2_max_cluster_radius: float | None = None
    stage2_novel_accept_tau: float | None = None
    stage2_auto_q: float = 90.0
    stage2_radius_scale: float = 1.6
    stage2_novel_scale: float = 1.7
    stage2_max_auto_points: int = 3000
    stage2_max_temp_clusters: int = 2000
    stage2_min_mean_score: float | None = None
    stage2_large_temp_min_size: int = 20
    stage2_merge_temp_every: int = 0
    stage2_merge_temp_scale: float = 0.5
    stage2_proto_distance: str = "count"  # "count" or "majority"
    stage2_countproto_topk: int = 128



CFG = Stage2FullPipelineConfig()


def class_counts(y: np.ndarray, label_names: list[str]) -> Dict[str, int]:
    vals, cnts = np.unique(np.asarray(y, dtype=np.int64), return_counts=True)
    out: Dict[str, int] = {}
    for v, c in zip(vals, cnts):
        name = label_names[int(v)] if int(v) < len(label_names) else str(int(v))
        out[name] = int(c)
    return out


def run_full_pipeline_eval(cfg: Stage2FullPipelineConfig = CFG) -> Dict[str, Any]:
    if cfg.stage2_proto_distance not in {"count", "majority"}:
        raise ValueError("stage2_proto_distance must be 'count' or 'majority'.")

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

    print("\nFitting/calibrating Stage-1 binary-HDC normal gate...")
    stage1 = BinaryHDCGate(
        bit_dim=enc.bit_dim,
        num_subprototypes=cfg.num_subprototypes,
        subproto_max_iter=cfg.subproto_max_iter,
        local_k=cfg.local_k,
        max_ref_normal=cfg.max_ref_normal,
        score_batch_size=cfg.local_score_batch_size,
        seed=cfg.seed,
    )
    t_stage1_fit = time.time()
    stage1.fit(H_train[data.train_known_mask])
    calibration = stage1.calibrate(H_val[data.val_known_mask], known_reject_q=cfg.known_reject_q)
    stage1_fit_cal_time_sec = time.time() - t_stage1_fit
    print("Stage-1 calibration:", calibration)

    print("\nEvaluating static Stage-1 gate...")
    val_metrics = evaluate_stage1(stage1, H_val, data.y_val, data.known_ids, data.unknown_ids)
    static_test = evaluate_stage1(stage1, H_test, data.y_test, data.known_ids, data.unknown_ids)
    print_metrics("STATIC OPEN-SET VALIDATION", val_metrics)
    print_metrics("STATIC OPEN-SET TEST", static_test)

    print("\nGathering fixed phased-stream binary HDC arrays...")
    H_stream = gather_stream_arrays(data.stream_sources, data.stream_indices, H_train, H_val, H_test)
    stream_scores = stage1.predict_scores(H_stream)
    stage1_accepted = stream_scores["accepted"].astype(bool)
    stage1_stream_metrics = evaluate_stage1(stage1, H_stream, data.y_stream_true, data.known_ids, data.unknown_ids)
    print_metrics("ONLINE STREAM STAGE-1 BINARY-HDC GATE ONLY", stage1_stream_metrics)

    print("\nCalibrating Stage-2 binary-HDC clustering thresholds...")
    val_scores = stage1.predict_scores(H_val)
    auto = stage2_auto_thresholds_binary(
        H_val=H_val,
        rejected_val=~val_scores["accepted"],
        scores_val=val_scores["unknown_score_binary_gate"],
        stage1_tau=stage1.tau_score,
        q=cfg.stage2_auto_q,
        radius_scale=cfg.stage2_radius_scale,
        novel_scale=cfg.stage2_novel_scale,
        max_points=cfg.stage2_max_auto_points,
        seed=cfg.seed,
    )
    tau_cluster = auto["tau_cluster"] if cfg.stage2_tau_cluster is None else float(cfg.stage2_tau_cluster)
    max_radius = auto["max_cluster_radius"] if cfg.stage2_max_cluster_radius is None else float(cfg.stage2_max_cluster_radius)
    novel_tau = auto["novel_accept_tau"] if cfg.stage2_novel_accept_tau is None else float(cfg.stage2_novel_accept_tau)
    min_mean_score = auto["min_mean_stage1_score"] if cfg.stage2_min_mean_score is None else float(cfg.stage2_min_mean_score)

    stage2_calib = dict(auto)
    stage2_calib.update({
        "tau_cluster_used": float(tau_cluster),
        "max_cluster_radius_used": float(max_radius),
        "novel_accept_tau_used": float(novel_tau),
        "min_mean_stage1_score_used": float(min_mean_score),
        "stage2_proto_distance": str(cfg.stage2_proto_distance),
        "stage2_countproto_topk": int(cfg.stage2_countproto_topk),
    })
    print("Stage-2 calibration:", stage2_calib)

    print("\n================ ONLINE STREAM STAGE-2 BINARY-HDC CLUSTERING ================")
    clusterer = BinaryHDCOnlineClusterer(
        bit_dim=enc.bit_dim,
        tau_cluster=tau_cluster,
        max_cluster_radius=max_radius,
        novel_accept_tau=novel_tau,
        min_cluster_size=cfg.stage2_min_cluster_size,
        min_cluster_age=cfg.stage2_min_cluster_age,
        min_mean_stage1_score=min_mean_score,
        max_temp_clusters=cfg.stage2_max_temp_clusters,
        novel_id_start=100000,
        merge_temp_every=cfg.stage2_merge_temp_every,
        merge_temp_scale=cfg.stage2_merge_temp_scale,
        proto_distance=cfg.stage2_proto_distance,
        countproto_topk=cfg.stage2_countproto_topk,
    )

    stage2_pred = np.full(len(data.y_stream_true), -1, dtype=np.int64)
    stage2_status: List[str] = []
    per_sample_records: List[Dict[str, Any]] = []

    t_stage2 = time.time()
    for t, (h, y_true) in enumerate(zip(H_stream, data.y_stream_true)):
        accepted = bool(stage1_accepted[t])
        rec: Dict[str, Any] = {
            "t": int(t),
            "source_split": str(data.stream_sources[t]),
            "source_index": int(data.stream_indices[t]),
            "true_label": int(y_true),
            "true_name": data.label_names[int(y_true)] if int(y_true) < len(data.label_names) else str(y_true),
            "stage1_accepted_as_normal": accepted,
            "stage1_rejected_as_suspicious": bool(not accepted),
            "binary_global_hamming": float(stream_scores["binary_global_hamming"][t]),
            "binary_local_topk_hamming": float(stream_scores["binary_local_topk_hamming"][t]),
            "binary_gate_score": float(stream_scores["binary_gate_score"][t]),
            "unknown_score_binary_gate": float(stream_scores["unknown_score_binary_gate"][t]),
            "stage2_representation": "binary_thermometer_hdc",
        }

        if accepted:
            stage2_status.append("known_normal")
            stage2_pred[t] = int(data.normal_label)
            rec.update({
                "stage2_status": "known_normal",
                "stage2_pred": int(data.normal_label),
                "novel_id": None,
                "temp_uid": None,
                "temp_distance": None,
                "novel_distance": None,
                "new_class_id": None,
            })
        else:
            out = clusterer.process(
                h,
                t=t,
                score=float(stream_scores["unknown_score_binary_gate"][t]),
                true_label_for_log=int(y_true),
            )
            stage2_status.append(str(out["stage2_status"]))
            stage2_pred[t] = int(out["stage2_pred"])
            rec.update(out)

        per_sample_records.append(rec)

    stage2_time_sec = time.time() - t_stage2

    print("\nEvaluating Stage-2 / full-pipeline metrics...")
    stage2_metrics = evaluate_stage2(
        y_true=data.y_stream_true,
        stage1_accepted=stage1_accepted,
        stage2_pred=stage2_pred,
        stage2_status=stage2_status,
        known_ids=data.known_ids,
        unknown_ids=data.unknown_ids,
    )
    cluster_summary = clusterer.summary()

    # Extra practical metric: suspicious samples grouped into large temporary
    # clusters are not yet promoted, but may already be engineer-inspectable.
    temp_size_by_uid = {int(c["uid"]): int(c["count"]) for c in clusterer.temp}
    large_temp_uids = {uid for uid, cnt in temp_size_by_uid.items() if cnt >= int(cfg.stage2_large_temp_min_size)}
    large_temp_mask = np.zeros(len(per_sample_records), dtype=bool)

    for i, rec in enumerate(per_sample_records):
        uid = rec.get("temp_uid", None)
        if uid is None:
            rec["large_temp_suspicious_mode"] = False
            rec["resolved_temp_uid"] = None
            continue
        resolved = clusterer.resolve_uid(uid)
        if resolved in large_temp_uids:
            large_temp_mask[i] = True
            rec["large_temp_suspicious_mode"] = True
            rec["resolved_temp_uid"] = int(resolved)
        else:
            rec["large_temp_suspicious_mode"] = False
            rec["resolved_temp_uid"] = None if resolved is None else int(resolved)

    y_stream_arr = np.asarray(data.y_stream_true, dtype=np.int64)
    unknown_mask = np.isin(y_stream_arr, data.unknown_ids)
    known_mask = np.isin(y_stream_arr, data.known_ids)
    novel_mask = stage2_pred >= 100000
    detected_unknown = unknown_mask & (~stage1_accepted)
    large_temp_unknown = unknown_mask & (~novel_mask) & large_temp_mask
    large_temp_known = known_mask & (~novel_mask) & large_temp_mask

    stage2_metrics.update({
        "stage2_large_temp_min_size": int(cfg.stage2_large_temp_min_size),
        "stage2_num_large_temporary_clusters": int(len(large_temp_uids)),
        "stage2_large_temp_fraction_of_unknown": float(np.mean(large_temp_unknown[unknown_mask])) if np.any(unknown_mask) else float("nan"),
        "stage2_large_temp_fraction_of_detected_unknown": float(np.mean(large_temp_unknown[detected_unknown])) if np.any(detected_unknown) else float("nan"),
        "stage2_meaningful_grouped_fraction_of_unknown": float(np.mean((novel_mask | large_temp_mask)[unknown_mask])) if np.any(unknown_mask) else float("nan"),
        "stage2_large_temp_fraction_of_known": float(np.mean(large_temp_known[known_mask])) if np.any(known_mask) else float("nan"),
    })

    print_metrics("ONLINE STREAM STAGE-2 / FULL PIPELINE METRICS", stage2_metrics)
    print(f"stage2_time_sec: {stage2_time_sec:.4f}")
    print(f"stage2_time_per_sample_sec: {stage2_time_sec / max(len(data.y_stream_true), 1):.8f}")

    cluster_summary["large_temp_min_size"] = int(cfg.stage2_large_temp_min_size)
    cluster_summary["num_large_temporary_clusters"] = int(len(large_temp_uids))
    cluster_summary["large_temporary_cluster_sizes"] = sorted([int(temp_size_by_uid[u]) for u in large_temp_uids], reverse=True)

    print("\nStage-2 cluster summary:")
    print(json.dumps(cluster_summary, indent=2, default=json_default)[:4000])

    print("\n===== Stream class-wise Stage-2 behavior =====")
    class_rows = classwise_stage2_behavior(
        data.y_stream_true,
        data.label_names,
        data.known_ids,
        data.unknown_ids,
        stage1_accepted,
        stage2_pred,
        stage2_status,
    )
    for r in class_rows:
        print(
            f"{r['class_name']:>12s} | count={r['count']:6d} | "
            f"normal={r['accepted_as_normal_rate']:.4f} | "
            f"suspicious={r['rejected_suspicious_rate']:.4f} | "
            f"novel_assigned={r['novel_assigned_rate']:.4f} | "
            f"unassigned={r['suspicious_unassigned_rate']:.4f} | "
            f"clustered={r['unknown_clustered_rate']:.4f} | "
            f"promoted={r['novel_promoted_rate']:.4f}"
        )

    result: Dict[str, Any] = {
        "method": "normal_only_binary_thermometer_hdc_countproto_memory",
        "description": (
            "Full Stage-1 + Stage-2 binary-HDC open-world wafer evaluation. "
            "Original 202-D auxiliary descriptors are encoded into binary "
            "thermometer HDC. Stage 1 uses Hamming normal memory. Stage 2 uses "
            "temporary and promoted HDC bit-count bundle memories for online "
            "novel defect-mode discovery."
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
        "stage1_known_reject_q": float(cfg.known_reject_q),
        "stage1_calibration": calibration,
        "static_validation": val_metrics,
        "static_test": static_test,
        "stage1_online_stream_metrics": stage1_stream_metrics,
        "stage2_calibration": stage2_calib,
        "stage2_online_metrics": stage2_metrics,
        "stage2_cluster_summary": cluster_summary,
        "classwise_stream_stage2": class_rows,
        "timing": {
            "binary_encoding_time_sec": float(encode_time_sec),
            "stage1_fit_calibration_time_sec": float(stage1_fit_cal_time_sec),
            "stage2_time_sec": float(stage2_time_sec),
            "stage2_time_per_sample_sec": float(stage2_time_sec / max(len(data.y_stream_true), 1)),
        },
    }

    with open(os.path.join(cfg.save_dir, "full_pipeline_results.json"), "w") as f:
        json.dump(result, f, indent=2, default=json_default)

    with open(os.path.join(cfg.save_dir, "full_pipeline_stream_records.json"), "w") as f:
        json.dump(per_sample_records, f, indent=2, default=json_default)

    np.savez_compressed(
        os.path.join(cfg.save_dir, "full_pipeline_arrays.npz"),
        y_stream_true=np.asarray(data.y_stream_true, dtype=np.int64),
        stage1_accepted=np.asarray(stage1_accepted, dtype=bool),
        stage2_pred=np.asarray(stage2_pred, dtype=np.int64),
        binary_gate_score=np.asarray(stream_scores["binary_gate_score"], dtype=np.float32),
        large_temp_mask=np.asarray(large_temp_mask, dtype=bool),
    )

    print(f"\nSaved Stage-2/full-pipeline outputs to: {cfg.save_dir}")
    return result


if __name__ == "__main__":
    run_full_pipeline_eval(CFG)
