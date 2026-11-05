#!/usr/bin/env python3
"""
run_mri_hdc.py - Eric's Stage 1 + Stage 2 HDC pipeline on PI-CAI MRI slices.

Same steps as run_full_hdc.py main(), with the wafer data call swapped for
prepare_picai_bundle(), plus three MRI extras:
  - a no-HDC baseline (same kNN score on the float features) to see what HDC costs
  - patient-level AUROC (max slice score per patient) for the lesion task
  - a picture of the highest- and lowest-scoring test slices

Example:
  python run_mri_hdc.py --slices_npz picai_slices.npz --task quality --feature_set stats
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from picai_bundle import prepare_picai_bundle
from run_full_hdc import (
    BinaryHDCGate, BinaryHDCOnlineClusterer, binary_auprc, binary_auroc, classwise_stage2_behavior,
    evaluate_stage1, evaluate_stage2, gather_stream_arrays, json_default, print_metrics, set_seed,
    stage2_auto_thresholds_binary,
)
from wafer_hdc_encoding import BinaryThermometerHDC


def float_knn_score(X_ref: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """No-HDC baseline: mean Euclidean distance to the k nearest training normals."""
    from sklearn.neighbors import NearestNeighbors
    dist, _ = NearestNeighbors(n_neighbors=min(k, len(X_ref))).fit(X_ref).kneighbors(X)
    return dist.mean(axis=1)


def patient_auroc(pid: np.ndarray, y: np.ndarray, score: np.ndarray) -> float:
    pats = np.unique(pid)
    p_score = np.array([score[pid == p].max() for p in pats])
    p_label = np.array([int((y[pid == p] > 0).any()) for p in pats])
    return binary_auroc(p_label, p_score)


def save_gallery(images, y, score, label_names, path, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(-score)
    picks = [("highest score", order[:n]), ("lowest score", order[-n:])]
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.6))
    for row, (title, idx) in enumerate(picks):
        for col, i in enumerate(idx):
            ax = axes[row, col]
            ax.imshow(images[i][0].astype(np.float32), cmap="gray")
            ax.set_title(f"{label_names[y[i]]}\n{score[i]:.2f}", fontsize=7)
            ax.axis("off")
        axes[row, 0].text(-0.3, 0.5, title, transform=axes[row, 0].transAxes, rotation=90, va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices_npz", required=True)
    ap.add_argument("--task", choices=["lesion", "quality"], default="quality")
    ap.add_argument("--feature_set", default="stats", help="pixels | stats | resnet | stats+pixels")
    ap.add_argument("--save_dir", default="./results_mri_hdc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--min_lesion_pixels", type=int, default=10)
    ap.add_argument("--artifact_frac", type=float, default=0.5)
    # HDC encoder (paper values)
    ap.add_argument("--num_proj", type=int, default=1024)
    ap.add_argument("--num_bins", type=int, default=8)
    ap.add_argument("--clip_value", type=float, default=3.0)
    # Stage 1 gate (paper values)
    ap.add_argument("--known_reject_q", type=float, default=10.0)
    ap.add_argument("--num_subprototypes", type=int, default=10)
    ap.add_argument("--local_k", type=int, default=10)
    ap.add_argument("--max_ref_normal", type=int, default=2000)
    # Stage 2 (min_cluster_size is 50 for wafers; MRI subsets are smaller)
    ap.add_argument("--stage2_min_cluster_size", type=int, default=20)
    ap.add_argument("--stage2_auto_q", type=float, default=85.0)
    ap.add_argument("--skip_stage2", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    data, meta = prepare_picai_bundle(
        args.slices_npz, task=args.task, feature_set=args.feature_set, seed=args.seed,
        val_frac=args.val_frac, test_frac=args.test_frac, min_lesion_pixels=args.min_lesion_pixels,
        artifact_frac=args.artifact_frac,
    )

    # ---- HDC encoding (unchanged from Eric) ----
    enc = BinaryThermometerHDC(in_dim=data.X_aux_train.shape[1], num_proj=args.num_proj,
                               num_bins=args.num_bins, seed=args.seed, clip_value=args.clip_value)
    t0 = time.time()
    enc.fit(data.X_aux_train[data.train_known_mask])  # Eric fits on all train; normals-only is stricter
    H_train, H_val, H_test = (enc.transform(x) for x in (data.X_aux_train, data.X_aux_val, data.X_aux_test))
    print(f"HDC: {enc.bit_dim} bits per slice, encoded in {time.time() - t0:.1f}s")

    # ---- Stage 1 normal gate (unchanged) ----
    gate = BinaryHDCGate(bit_dim=enc.bit_dim, num_subprototypes=args.num_subprototypes,
                         local_k=args.local_k, max_ref_normal=args.max_ref_normal, seed=args.seed)
    gate.fit(H_train[data.train_known_mask])
    calib = gate.calibrate(H_val[data.val_known_mask], known_reject_q=args.known_reject_q)
    val_m = evaluate_stage1(gate, H_val, data.y_val, data.known_ids, data.unknown_ids)
    test_m = evaluate_stage1(gate, H_test, data.y_test, data.known_ids, data.unknown_ids)
    print_metrics("STAGE 1 VALIDATION", val_m)
    print_metrics("STAGE 1 TEST", test_m)

    test_score = gate.predict_scores(H_test)["unknown_score_binary_gate"]
    y_bin = (data.y_test > 0).astype(np.int64)
    extras = {}

    # ---- no-HDC baseline on the same features ----
    ref = data.X_aux_train[data.train_known_mask][: args.max_ref_normal]
    base = float_knn_score(ref, data.X_aux_test, args.local_k)
    extras["baseline_float_knn_auroc"] = binary_auroc(y_bin, base)
    extras["baseline_float_knn_auprc"] = binary_auprc(y_bin, base)
    extras["hdc_gate_auroc"] = test_m["unknown_auroc_by_binary_hdc_gate"]
    # per unknown class: which kinds of anomaly the gate finds
    for c in data.unknown_ids:
        m = (data.y_test == 0) | (data.y_test == c)
        if (data.y_test == c).any():
            extras[f"hdc_auroc_{data.label_names[c]}"] = binary_auroc((data.y_test[m] == c).astype(int), test_score[m])
    if args.task == "lesion":
        extras["hdc_patient_auroc"] = patient_auroc(meta["test"]["pid"], data.y_test, test_score)
        extras["baseline_patient_auroc"] = patient_auroc(meta["test"]["pid"], data.y_test, base)
    print_metrics("HDC VS NO-HDC BASELINE (TEST)", extras)
    save_gallery(data.W_test, data.y_test, test_score, data.label_names,
                 os.path.join(args.save_dir, "test_extremes.png"))

    # ---- Stage 2 online discovery (same loop as run_full_hdc.py, without the logging extras) ----
    stage2 = {}
    if not args.skip_stage2:
        H_stream = gather_stream_arrays(data.stream_sources, data.stream_indices, H_train, H_val, H_test)
        s = gate.predict_scores(H_stream)
        val_s = gate.predict_scores(H_val)
        auto = stage2_auto_thresholds_binary(H_val, ~val_s["accepted"], val_s["unknown_score_binary_gate"],
                                             gate.tau_score, q=args.stage2_auto_q, seed=args.seed)
        clus = BinaryHDCOnlineClusterer(
            bit_dim=enc.bit_dim, tau_cluster=auto["tau_cluster"], max_cluster_radius=auto["max_cluster_radius"],
            novel_accept_tau=auto["novel_accept_tau"], min_cluster_size=args.stage2_min_cluster_size,
            min_mean_stage1_score=auto["min_mean_stage1_score"],
        )
        pred = np.full(len(H_stream), -1, dtype=np.int64)
        status = []
        for t, h in enumerate(H_stream):
            if s["accepted"][t]:
                pred[t] = 0
                status.append("known_normal")
            else:
                out = clus.process(h, t=t, score=float(s["unknown_score_binary_gate"][t]),
                                   true_label_for_log=int(data.y_stream_true[t]))
                pred[t] = int(out["stage2_pred"])
                status.append(str(out["stage2_status"]))
        stage2 = evaluate_stage2(data.y_stream_true, s["accepted"], pred, status, data.known_ids, data.unknown_ids)
        stage2["classwise"] = classwise_stage2_behavior(data.y_stream_true, data.label_names, data.known_ids,
                                                        data.unknown_ids, s["accepted"], pred, status)
        stage2["cluster_summary"] = clus.summary()
        print_metrics("STAGE 2 STREAM", {k: v for k, v in stage2.items() if not isinstance(v, (list, dict))})

    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "label_names": data.label_names, "encoder": enc.meta(),
                   "stage1_calibration": calib, "stage1_val": val_m, "stage1_test": test_m,
                   "extras": extras, "stage2": stage2}, f, indent=2, default=json_default)
    print("saved to", args.save_dir)


if __name__ == "__main__":
    main()
