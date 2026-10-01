#!/usr/bin/env python3
"""
Oracle Stage-2-only evaluation for binary-HDC online wafer defect-mode discovery.

This script intentionally removes Stage 1 from the Stage-2-only experiment.
Instead of using the Stage-1 rejected stream, it uses only defect/unknown wafers
as the online stream. This isolates the Stage-2 online HDC memory itself:

  202-D aux features -> binary thermometer HDC -> defect-only online stream
  -> temporary/novel HDC memory -> defect-mode discovery metrics.

Important: because labels are used to remove normal wafers from the stream, this
is an oracle/diagnostic Stage-2 experiment. It should not be reported as an
end-to-end open-world result. Use the full-pipeline script for end-to-end claims.

Expected directory layout:
  project_root/
    run_full_hdc.py
    wafer_data_prep_hdc_stage2.py
    wafer_hdc_encoding.py
    eval_stage2_only_defect_stream_binary_hdc.py
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
    BinaryHDCOnlineClusterer,
    classwise_stage2_behavior,
    evaluate_stage2,
    gather_stream_arrays,
    json_default,
    print_metrics,
    set_seed,
    stage2_auto_thresholds_binary,
)


@dataclass(frozen=True)
class Stage2DefectStreamEvalConfig:
    # Paths
    wm811k_pkl: str = "./data/LSWMD.pkl"
    save_dir: str = "./results_stage2_only_defect_stream_binary_hdc_clip3_s2_balanced"
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

    # Stage-2 online discovery — less aggressive for defect-only stream
    stage2_min_cluster_size: int = 40
    stage2_min_cluster_age: int = 0
    stage2_tau_cluster: float | None = None
    stage2_max_cluster_radius: float | None = None
    stage2_novel_accept_tau: float | None = None
    stage2_auto_q: float = 85.0
    stage2_radius_scale: float = 1.5
    stage2_novel_scale: float = 1.55
    stage2_max_auto_points: int = 3000
    stage2_max_temp_clusters: int = 2000
    stage2_min_mean_score: float | None = None
    stage2_large_temp_min_size: int = 20
    stage2_merge_temp_every: int = 0
    stage2_merge_temp_scale: float = 0.5
    stage2_proto_distance: str = "count"
    stage2_countproto_topk: int = 128
'''
class Stage2DefectStreamEvalConfig:
    # Paths
    wm811k_pkl: str = "./data/LSWMD.pkl"
    save_dir: str = "./results_stage2_only_defect_stream_binary_hdc_clip3_s2_aggressive"
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

    # Binary HDC encoder: keep best Stage-1 representation setting, but no Stage 1 is run.
    binary_num_proj: int = 1024
    binary_num_bins: int = 8
    binary_clip_value: float = 3.0
    binary_encode_batch_size: int = 1024

    # Stage-2 online discovery: current best aggressive setting.
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
'''

'''
class Stage2DefectStreamEvalConfig:
    # Paths
    wm811k_pkl: str = "./data/LSWMD.pkl"
    save_dir: str = "./results_stage2_only_defect_stream_binary_hdc_clip3_s2_purity"
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

    # Stage-2 online discovery — purity-oriented
    stage2_min_cluster_size: int = 50
    stage2_min_cluster_age: int = 0
    stage2_tau_cluster: float | None = None
    stage2_max_cluster_radius: float | None = None
    stage2_novel_accept_tau: float | None = None
    stage2_auto_q: float = 80.0
    stage2_radius_scale: float = 1.4
    stage2_novel_scale: float = 1.4
    stage2_max_auto_points: int = 3000
    stage2_max_temp_clusters: int = 2000
    stage2_min_mean_score: float | None = None
    stage2_large_temp_min_size: int = 20
    stage2_merge_temp_every: int = 0
    stage2_merge_temp_scale: float = 0.5
    stage2_proto_distance: str = "count"
    stage2_countproto_topk: int = 128
'''


CFG = Stage2DefectStreamEvalConfig()


def class_counts(y: np.ndarray, label_names: list[str]) -> Dict[str, int]:
    vals, cnts = np.unique(np.asarray(y, dtype=np.int64), return_counts=True)
    out: Dict[str, int] = {}
    for v, c in zip(vals, cnts):
        name = label_names[int(v)] if int(v) < len(label_names) else str(int(v))
        out[name] = int(c)
    return out


def run_stage2_defect_stream_eval(cfg: Stage2DefectStreamEvalConfig = CFG) -> Dict[str, Any]:
    if cfg.stage2_proto_distance not in {"count", "majority"}:
        raise ValueError("stage2_proto_distance must be 'count' or 'majority'.")

    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)

    print("Loading fixed phased-stream data...")
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

    print("\nBuilding oracle defect-only stream for Stage 2...")
    H_stream_all = gather_stream_arrays(data.stream_sources, data.stream_indices, H_train, H_val, H_test)
    y_stream_all = np.asarray(data.y_stream_true, dtype=np.int64)
    defect_stream_mask = np.isin(y_stream_all, data.unknown_ids)

    H_stream = H_stream_all[defect_stream_mask]
    y_stream = y_stream_all[defect_stream_mask]
    stream_sources = np.asarray(data.stream_sources, dtype=object)[defect_stream_mask]
    stream_indices = np.asarray(data.stream_indices, dtype=np.int64)[defect_stream_mask]

    # Stage 1 is intentionally removed: every defect-stream sample is treated
    # as suspicious and passed to Stage 2.
    stage1_accepted = np.zeros(len(y_stream), dtype=bool)
    stream_scores = np.ones(len(y_stream), dtype=np.float32)

    print(f"Defect-only stream size: {len(y_stream)}")
    print("Defect-only stream class counts:")
    print(json.dumps(class_counts(y_stream, data.label_names), indent=2))

    print("\nCalibrating Stage-2 binary-HDC clustering thresholds without Stage 1...")
    val_unknown_mask = np.isin(np.asarray(data.y_val, dtype=np.int64), data.unknown_ids)
    H_calib = H_val[val_unknown_mask]
    if len(H_calib) < 2:
        # Fallback should rarely trigger, but keeps the script robust.
        H_calib = H_stream

    no_stage1_min_score = -1.0e9
    auto = stage2_auto_thresholds_binary(
        H_val=H_calib,
        rejected_val=np.ones(len(H_calib), dtype=bool),
        scores_val=np.ones(len(H_calib), dtype=np.float32),
        stage1_tau=no_stage1_min_score,
        q=cfg.stage2_auto_q,
        radius_scale=cfg.stage2_radius_scale,
        novel_scale=cfg.stage2_novel_scale,
        max_points=cfg.stage2_max_auto_points,
        seed=cfg.seed,
    )
    auto["calibration_stream"] = "validation_unknown_defects_only"
    auto["no_stage1"] = True

    tau_cluster = auto["tau_cluster"] if cfg.stage2_tau_cluster is None else float(cfg.stage2_tau_cluster)
    max_radius = auto["max_cluster_radius"] if cfg.stage2_max_cluster_radius is None else float(cfg.stage2_max_cluster_radius)
    novel_tau = auto["novel_accept_tau"] if cfg.stage2_novel_accept_tau is None else float(cfg.stage2_novel_accept_tau)
    min_mean_score = auto["min_mean_stage1_score"] if cfg.stage2_min_mean_score is None else float(cfg.stage2_min_mean_score)

    stage2_calib = dict(auto)
    stage2_calib.update({
        "tau_cluster_used": float(tau_cluster),
        "max_cluster_radius_used": float(max_radius),
        "novel_accept_tau_used": float(novel_tau),
        "min_mean_score_used": float(min_mean_score),
        "stage2_proto_distance": str(cfg.stage2_proto_distance),
        "stage2_countproto_topk": int(cfg.stage2_countproto_topk),
    })
    print("Stage-2 calibration:", stage2_calib)

    print("\n================ ORACLE DEFECT-STREAM STAGE-2 CLUSTERING ================")
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

    stage2_pred = np.full(len(y_stream), -1, dtype=np.int64)
    stage2_status: List[str] = []
    per_sample_records: List[Dict[str, Any]] = []

    t_stage2 = time.time()
    for t, (h, y_true) in enumerate(zip(H_stream, y_stream)):
        rec: Dict[str, Any] = {
            "t": int(t),
            "source_split": str(stream_sources[t]),
            "source_index": int(stream_indices[t]),
            "true_label": int(y_true),
            "true_name": data.label_names[int(y_true)] if int(y_true) < len(data.label_names) else str(y_true),
            "oracle_defect_stream": True,
            "stage1_used": False,
            "stage1_accepted_as_normal": False,
            "stage1_rejected_as_suspicious": True,
            "unknown_score_binary_gate": float(stream_scores[t]),
            "stage2_representation": "binary_thermometer_hdc",
        }

        out = clusterer.process(
            h,
            t=t,
            score=float(stream_scores[t]),
            true_label_for_log=int(y_true),
        )
        stage2_status.append(str(out["stage2_status"]))
        stage2_pred[t] = int(out["stage2_pred"])
        rec.update(out)
        per_sample_records.append(rec)

    stage2_time_sec = time.time() - t_stage2

    print("\nEvaluating oracle defect-stream Stage-2 metrics...")
    stage2_metrics = evaluate_stage2(
        y_true=y_stream,
        stage1_accepted=stage1_accepted,
        stage2_pred=stage2_pred,
        stage2_status=stage2_status,
        known_ids=data.known_ids,
        unknown_ids=data.unknown_ids,
    )
    cluster_summary = clusterer.summary()

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

    unknown_mask = np.isin(y_stream, data.unknown_ids)
    known_mask = np.isin(y_stream, data.known_ids)
    novel_mask = stage2_pred >= 100000
    detected_unknown = unknown_mask & (~stage1_accepted)
    large_temp_unknown = unknown_mask & (~novel_mask) & large_temp_mask
    large_temp_known = known_mask & (~novel_mask) & large_temp_mask

    stage2_metrics.update({
        "oracle_defect_stream": True,
        "stage1_removed": True,
        "stage2_large_temp_min_size": int(cfg.stage2_large_temp_min_size),
        "stage2_num_large_temporary_clusters": int(len(large_temp_uids)),
        "stage2_large_temp_fraction_of_unknown": float(np.mean(large_temp_unknown[unknown_mask])) if np.any(unknown_mask) else float("nan"),
        "stage2_large_temp_fraction_of_detected_unknown": float(np.mean(large_temp_unknown[detected_unknown])) if np.any(detected_unknown) else float("nan"),
        "stage2_meaningful_grouped_fraction_of_unknown": float(np.mean((novel_mask | large_temp_mask)[unknown_mask])) if np.any(unknown_mask) else float("nan"),
        "stage2_large_temp_fraction_of_known": float(np.mean(large_temp_known[known_mask])) if np.any(known_mask) else float("nan"),
    })

    print_metrics("ORACLE DEFECT-STREAM STAGE-2 METRICS", stage2_metrics)
    print(f"stage2_time_sec: {stage2_time_sec:.4f}")
    print(f"stage2_time_per_sample_sec: {stage2_time_sec / max(len(y_stream), 1):.8f}")

    cluster_summary["large_temp_min_size"] = int(cfg.stage2_large_temp_min_size)
    cluster_summary["num_large_temporary_clusters"] = int(len(large_temp_uids))
    cluster_summary["large_temporary_cluster_sizes"] = sorted([int(temp_size_by_uid[u]) for u in large_temp_uids], reverse=True)

    print("\nStage-2 cluster summary:")
    print(json.dumps(cluster_summary, indent=2, default=json_default)[:4000])

    print("\n===== Defect-stream class-wise Stage-2 behavior =====")
    class_rows = classwise_stage2_behavior(
        y_stream,
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
        "method": "stage2_only_oracle_defect_stream_binary_thermometer_hdc_countproto_memory",
        "description": (
            "Oracle Stage-2-only online discovery evaluation. Stage 1 is removed; "
            "the stream contains only defect/unknown wafers selected using labels. "
            "This isolates the binary-HDC temporary/novel bit-count memory used for "
            "defect-mode discovery, and should not be interpreted as an end-to-end result."
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
            "original_stream": class_counts(data.y_stream_true, data.label_names),
            "defect_only_stream": class_counts(y_stream, data.label_names),
        },
        "stage1_used_for_stage2": False,
        "stage2_calibration": stage2_calib,
        "stage2_online_metrics": stage2_metrics,
        "stage2_cluster_summary": cluster_summary,
        "classwise_stream_stage2": class_rows,
        "timing": {
            "binary_encoding_time_sec": float(encode_time_sec),
            "stage2_time_sec": float(stage2_time_sec),
            "stage2_time_per_sample_sec": float(stage2_time_sec / max(len(y_stream), 1)),
        },
    }

    with open(os.path.join(cfg.save_dir, "stage2_defect_stream_results.json"), "w") as f:
        json.dump(result, f, indent=2, default=json_default)

    with open(os.path.join(cfg.save_dir, "stage2_defect_stream_records.json"), "w") as f:
        json.dump(per_sample_records, f, indent=2, default=json_default)

    np.savez_compressed(
        os.path.join(cfg.save_dir, "stage2_defect_stream_arrays.npz"),
        y_stream_true=np.asarray(y_stream, dtype=np.int64),
        stage1_accepted=np.asarray(stage1_accepted, dtype=bool),
        stage2_pred=np.asarray(stage2_pred, dtype=np.int64),
        large_temp_mask=np.asarray(large_temp_mask, dtype=bool),
    )

    print(f"\nSaved oracle defect-stream Stage-2 outputs to: {cfg.save_dir}")
    return result


if __name__ == "__main__":
    run_stage2_defect_stream_eval(CFG)
