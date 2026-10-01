#!/usr/bin/env python3
"""
Binary HDC normal-only online wafer screening + novel-mode discovery.

This streamlined version removes the previous spatial encoders and real-valued
RP Stage-2 variants.  It uses only the original 202 auxiliary descriptors, but
encodes them into a binary distance-preserving HDC representation:

    202-D aux features -> random projections -> thermometer bits -> packed uint64

All Stage-1 and Stage-2 memory comparisons use Hamming distance / popcount.
Cluster memories are true HDC bundles: each temporary or promoted novel memory
stores per-bit counts.  Stage 2 can compare a sample either to the hard
majority-bit prototype or to the soft bit-count centroid, which is the average
Hamming distance to the cluster members implied by the per-bit counts.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
except Exception:
    adjusted_rand_score = None
    normalized_mutual_info_score = None

from wafer_data_prep_hdc_stage2 import prepare_normal_only_phased_data
from wafer_hdc_encoding import (
    BinaryThermometerHDC,
    hamming_matrix,
    hamming_to_one,
    nearest_hamming_to_set,
    topk_mean_hamming_to_set,
    leave_one_out_nearest_hamming_sample,
    unpack_bits_little,
    pack_bits_little,
)


def set_seed(seed: int):
    np.random.seed(int(seed))


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def robust_scale(v: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(v, dtype=np.float32)
    med = float(np.median(v))
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
    if not np.isfinite(iqr) or iqr < 1e-6:
        iqr = float(np.std(v) + 1e-6)
    return med, iqr


def z_score(v: np.ndarray, stat: Tuple[float, float]) -> np.ndarray:
    med, scale = stat
    return ((np.asarray(v, dtype=np.float32) - med) / (scale + 1e-6)).astype(np.float32)


def binary_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float32)
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(score), dtype=np.float64)
    # average ranks for ties
    unique_vals, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for k, c in enumerate(counts):
            if c > 1:
                idx = np.where(inv == k)[0]
                ranks[idx] = ranks[idx].mean()
    sum_pos = ranks[y_true == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((sum_pos - n_pos * (n_pos - 1) / 2.0) / max(n_pos * n_neg, 1))


def binary_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float32)
    if np.sum(y_true == 1) == 0:
        return float("nan")
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(np.sum(y == 1), 1)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def open_set_tradeoff_at_fixed_known_accept(known_scores: np.ndarray, unknown_scores: np.ndarray, prefix: str) -> Dict[str, float]:
    out = {}
    known_scores = np.asarray(known_scores, dtype=np.float32)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float32)
    for accept in [0.90, 0.80, 0.70]:
        if len(known_scores) == 0 or len(unknown_scores) == 0:
            continue
        reject_q = 100.0 * accept
        tau = float(np.percentile(known_scores, reject_q))
        out[f"unknown_reject_at_{int(accept*100)}known_{prefix}"] = float(np.mean(unknown_scores > tau))
        out[f"known_accept_at_{int(accept*100)}known_{prefix}"] = float(np.mean(known_scores <= tau))
        out[f"threshold_at_{int(accept*100)}known_{prefix}"] = tau
    return out


def cluster_purity(y_true: np.ndarray, cluster_ids: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    total = 0
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        _, counts = np.unique(y_true[mask], return_counts=True)
        total += int(np.max(counts))
    return float(total / len(y_true))


def best_map_cluster_to_true(y_true: np.ndarray, cluster_ids: np.ndarray) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        vals, counts = np.unique(y_true[mask], return_counts=True)
        out[int(cid)] = int(vals[np.argmax(counts)])
    return out


def safe_cluster_metrics(y_true: np.ndarray, cluster_ids: np.ndarray) -> Dict[str, float]:
    out = {"ari": float("nan"), "nmi": float("nan")}
    if len(y_true) == 0 or len(np.unique(y_true)) <= 1 or len(np.unique(cluster_ids)) <= 1:
        return out
    if adjusted_rand_score is not None:
        out["ari"] = float(adjusted_rand_score(y_true, cluster_ids))
    if normalized_mutual_info_score is not None:
        out["nmi"] = float(normalized_mutual_info_score(y_true, cluster_ids))
    return out


def print_metrics(title: str, metrics: Dict[str, Any]):
    print(f"\n================ {title} ================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


def gather_stream_arrays(stream_sources: np.ndarray, stream_indices: np.ndarray,
                         H_train: np.ndarray, H_val: np.ndarray, H_test: np.ndarray) -> np.ndarray:
    out = []
    for src, idx in zip(stream_sources, stream_indices):
        idx = int(idx)
        if src == "train":
            out.append(H_train[idx])
        elif src == "val":
            out.append(H_val[idx])
        else:
            out.append(H_test[idx])
    return np.stack(out, axis=0).astype(np.uint64)


def majority_proto_from_bits(bit_counts: np.ndarray, count: int) -> np.ndarray:
    bits = (bit_counts * 2 >= int(count)).reshape(1, -1)
    return pack_bits_little(bits)[0]


def mean_hamming_radius_from_counts(bit_counts: np.ndarray, count: int) -> float:
    if count <= 0:
        return 0.0
    bc = np.asarray(bit_counts, dtype=np.int64)
    return float(np.minimum(bc, int(count) - bc).sum() / float(count))


def soft_count_radius_from_counts(bit_counts: np.ndarray, count: int) -> float:
    """Mean soft-centroid Hamming distance of members to their bit-count memory.

    For a bit with c ones among n samples, the expected contribution of a
    randomly selected member to the bit-count centroid distance is
        2 c (n-c) / n^2.
    This is a softer radius than distance to the hard majority prototype and
    keeps within-cluster uncertainty instead of thresholding it away.
    """
    n = int(count)
    if n <= 0:
        return 0.0
    bc = np.asarray(bit_counts, dtype=np.float32)
    return float((2.0 * bc * (float(n) - bc)).sum() / float(n * n))


def soft_count_distance_to_candidates(h_bits: np.ndarray, candidates: List[Dict[str, Any]]) -> np.ndarray:
    """Distance from one binary vector to several bit-count memories.

    For cluster C with n samples and per-bit counts c_j, this computes the
    average Hamming distance from x to the cluster members:
        d(x,C) = sum_j [x_j (n-c_j) + (1-x_j)c_j] / n
               = (sum_j c_j + x^T (n - 2c)) / n.
    """
    if not candidates:
        return np.empty((0,), dtype=np.float32)
    x = np.asarray(h_bits, dtype=np.float32)
    counts = np.asarray([max(int(c["count"]), 1) for c in candidates], dtype=np.float32)
    bc = np.stack([np.asarray(c["bit_counts"], dtype=np.float32) for c in candidates], axis=0)
    d = (bc.sum(axis=1) + (counts[:, None] - 2.0 * bc).dot(x)) / counts
    return d.astype(np.float32)


def soft_count_distance_between_clusters(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Average pairwise Hamming distance between two bit-count memories."""
    na = max(int(a["count"]), 1)
    nb = max(int(b["count"]), 1)
    ca = np.asarray(a["bit_counts"], dtype=np.float32)
    cb = np.asarray(b["bit_counts"], dtype=np.float32)
    d = (ca * (float(nb) - cb) + (float(na) - ca) * cb).sum() / float(na * nb)
    return float(d)


def make_binary_subprototypes(H: np.ndarray, bit_dim: int, k: int, seed: int = 0, max_iter: int = 8) -> np.ndarray:
    """Small Hamming k-means with majority-bit prototypes."""
    H = np.asarray(H, dtype=np.uint64)
    n = H.shape[0]
    k = int(max(1, min(k, n)))
    rng = np.random.default_rng(seed)
    if k == 1:
        bits = unpack_bits_little(H, bit_dim).astype(np.int32)
        return majority_proto_from_bits(bits.sum(axis=0), n).reshape(1, -1)
    protos = H[rng.choice(n, size=k, replace=False)].copy()
    labels = np.full(n, -1, dtype=np.int64)
    bits_all = None
    for _ in range(int(max_iter)):
        D = hamming_matrix(H, protos, b_chunk=max(64, k))
        new_labels = np.argmin(D, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        if bits_all is None:
            bits_all = unpack_bits_little(H, bit_dim).astype(np.int32)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                protos[j] = majority_proto_from_bits(bits_all[mask].sum(axis=0), int(mask.sum()))
            else:
                protos[j] = H[int(rng.integers(0, n))]
    return protos.astype(np.uint64)


class BinaryHDCGate:
    """Stage-1 normal-memory gate using Hamming distance over binary HDC vectors."""

    def __init__(self, bit_dim: int, num_subprototypes: int = 10, subproto_max_iter: int = 8,
                 local_k: int = 10, max_ref_normal: int = 2000, score_batch_size: int = 64, seed: int = 0):
        self.bit_dim = int(bit_dim)
        self.num_subprototypes = int(num_subprototypes)
        self.subproto_max_iter = int(subproto_max_iter)
        self.local_k = int(local_k)
        self.max_ref_normal = int(max_ref_normal)
        self.score_batch_size = int(max(1, score_batch_size))
        self.seed = int(seed)
        self.protos: Optional[np.ndarray] = None
        self.ref: Optional[np.ndarray] = None
        self.global_stat = (0.0, 1.0)
        self.local_stat = (0.0, 1.0)
        self.tau_score = 0.0

    def fit(self, H_normal: np.ndarray):
        H_normal = np.asarray(H_normal, dtype=np.uint64)
        self.protos = make_binary_subprototypes(
            H_normal, bit_dim=self.bit_dim, k=self.num_subprototypes,
            seed=self.seed + 202, max_iter=self.subproto_max_iter,
        )
        rng = np.random.default_rng(self.seed + 777)
        if len(H_normal) > self.max_ref_normal:
            idx = rng.choice(len(H_normal), size=self.max_ref_normal, replace=False)
            self.ref = H_normal[idx].copy()
        else:
            self.ref = H_normal.copy()

    def global_distance(self, H: np.ndarray) -> np.ndarray:
        if self.protos is None:
            raise RuntimeError("Gate is not fit.")
        return nearest_hamming_to_set(H, self.protos, batch_size=256, b_chunk=64)

    def local_distance(self, H: np.ndarray) -> np.ndarray:
        if self.ref is None:
            raise RuntimeError("Gate is not fit.")
        return topk_mean_hamming_to_set(H, self.ref, k=self.local_k, batch_size=self.score_batch_size, b_chunk=512)

    def combined_score_from_distances(self, d_global: np.ndarray, d_local: np.ndarray) -> np.ndarray:
        return np.maximum(z_score(d_global, self.global_stat), z_score(d_local, self.local_stat)).astype(np.float32)

    def calibrate(self, H_val_normal: np.ndarray, known_reject_q: float = 10.0) -> Dict[str, Any]:
        d_global = self.global_distance(H_val_normal)
        d_local = self.local_distance(H_val_normal)
        self.global_stat = robust_scale(d_global)
        self.local_stat = robust_scale(d_local)
        score = self.combined_score_from_distances(d_global, d_local)
        self.tau_score = float(np.percentile(score, 100.0 - float(known_reject_q)))
        return {
            "tau_binary_hdc_gate_score": float(self.tau_score),
            "known_reject_q": float(known_reject_q),
            "val_normal_global_hamming_mean": float(np.mean(d_global)),
            "val_normal_local_hamming_mean": float(np.mean(d_local)),
            "global_distance_stat": list(self.global_stat),
            "local_distance_stat": list(self.local_stat),
            "local_k": int(self.local_k),
            "local_ref_size": int(0 if self.ref is None else len(self.ref)),
        }

    def predict_scores(self, H: np.ndarray) -> Dict[str, np.ndarray]:
        d_global = self.global_distance(H)
        d_local = self.local_distance(H)
        score = self.combined_score_from_distances(d_global, d_local)
        accepted = score <= self.tau_score
        return {
            "binary_global_hamming": d_global,
            "binary_local_topk_hamming": d_local,
            "binary_gate_score": score,
            "unknown_score_binary_gate": score,
            "accepted": accepted,
            "pred": np.where(accepted, 1, -1).astype(np.int64),
        }


class BinaryHDCOnlineClusterer:
    """Stage-2 online cluster memory with binary HDC bit-count prototypes."""

    def __init__(self, bit_dim: int, tau_cluster: float, max_cluster_radius: float, novel_accept_tau: float,
                 min_cluster_size: int = 50, min_cluster_age: int = 0, min_mean_stage1_score: float = -np.inf,
                 max_temp_clusters: int = 1000, novel_id_start: int = 100000,
                 merge_temp_every: int = 0, merge_temp_scale: float = 0.50,
                 proto_distance: str = "count", countproto_topk: int = 64):
        self.bit_dim = int(bit_dim)
        self.tau_cluster = float(tau_cluster)
        self.max_cluster_radius = float(max_cluster_radius)
        self.novel_accept_tau = float(novel_accept_tau)
        self.min_cluster_size = int(min_cluster_size)
        self.min_cluster_age = int(min_cluster_age)
        self.min_mean_stage1_score = float(min_mean_stage1_score)
        self.max_temp_clusters = int(max_temp_clusters)
        self.next_uid = 0
        self.next_novel_id = int(novel_id_start)
        self.temp: List[Dict[str, Any]] = []
        self.novel: Dict[int, Dict[str, Any]] = {}
        self.promotion_events: List[Dict[str, Any]] = []
        self.processed_rejected = 0
        self.merge_temp_every = int(max(0, merge_temp_every))
        self.merge_temp_tau = float(self.tau_cluster * float(merge_temp_scale))
        self.merge_temp_scale = float(merge_temp_scale)
        self.proto_distance = str(proto_distance).lower()
        if self.proto_distance not in {"majority", "count"}:
            raise ValueError("proto_distance must be 'majority' or 'count'")
        self.countproto_topk = int(max(1, countproto_topk))
        self.uid_parent: Dict[int, int] = {}
        self.num_temp_merges = 0

    def _unpack_one(self, h: np.ndarray) -> np.ndarray:
        return unpack_bits_little(np.asarray(h, dtype=np.uint64).reshape(1, -1), self.bit_dim)[0].astype(np.int32)

    def _make_proto(self, bit_counts: np.ndarray, count: int) -> np.ndarray:
        return majority_proto_from_bits(bit_counts, count).astype(np.uint64)

    def _radius(self, c: Dict[str, Any]) -> float:
        if self.proto_distance == "count":
            return soft_count_radius_from_counts(c["bit_counts"], int(c["count"]))
        return mean_hamming_radius_from_counts(c["bit_counts"], int(c["count"]))

    def _candidate_indices_from_majority(self, centers: np.ndarray, h: np.ndarray, topk: Optional[int] = None) -> np.ndarray:
        d0 = hamming_to_one(centers, h)
        if topk is None or len(d0) <= int(topk):
            return np.arange(len(d0), dtype=np.int64)
        k = int(max(1, topk))
        return np.argpartition(d0, k - 1)[:k].astype(np.int64)

    def resolve_uid(self, uid: Optional[int]) -> Optional[int]:
        if uid is None:
            return None
        uid = int(uid)
        seen = set()
        while uid in self.uid_parent and uid not in seen:
            seen.add(uid)
            uid = int(self.uid_parent[uid])
        return int(uid)

    def _new_cluster(self, h: np.ndarray, t: int, score: float) -> int:
        bits = self._unpack_one(h)
        c = {
            "uid": int(self.next_uid),
            "bit_counts": bits.astype(np.int32),
            "proto": np.asarray(h, dtype=np.uint64).copy(),
            "count": 1,
            "score_sum": float(score),
            "created_t": int(t),
            "last_t": int(t),
        }
        self.next_uid += 1
        self.temp.append(c)
        if len(self.temp) > self.max_temp_clusters:
            idx = int(np.argmin(np.asarray([cc["count"] for cc in self.temp], dtype=np.float32)))
            dropped_uid = int(self.temp[idx]["uid"])
            self.uid_parent[dropped_uid] = -1
            del self.temp[idx]
        return int(c["uid"])

    def _update_cluster(self, idx: int, h: np.ndarray, t: int, score: float):
        c = self.temp[idx]
        c["bit_counts"] += self._unpack_one(h)
        c["count"] = int(c["count"]) + 1
        c["proto"] = self._make_proto(c["bit_counts"], int(c["count"]))
        c["score_sum"] = float(c["score_sum"] + float(score))
        c["last_t"] = int(t)

    def _nearest_temp(self, h: np.ndarray) -> Tuple[Optional[int], float]:
        if not self.temp:
            return None, float("inf")
        centers = np.stack([c["proto"] for c in self.temp], axis=0).astype(np.uint64)
        if self.proto_distance == "majority":
            d = hamming_to_one(centers, h)
            idx = int(np.argmin(d))
            return idx, float(d[idx])

        # Fast two-stage search: majority prototype finds candidate memories,
        # then bit-count centroid distance reranks only the closest candidates.
        cand_idx = self._candidate_indices_from_majority(centers, h, self.countproto_topk)
        h_bits = self._unpack_one(h)
        cands = [self.temp[int(i)] for i in cand_idx]
        d = soft_count_distance_to_candidates(h_bits, cands)
        j = int(np.argmin(d))
        return int(cand_idx[j]), float(d[j])

    def _nearest_novel_with_second(self, h: np.ndarray) -> Tuple[Optional[int], float, float]:
        if not self.novel:
            return None, float("inf"), float("inf")
        ids = list(self.novel.keys())
        centers = np.stack([self.novel[i]["proto"] for i in ids], axis=0).astype(np.uint64)
        if self.proto_distance == "majority":
            d = hamming_to_one(centers, h)
            order = np.argsort(d)
            idx0 = int(order[0])
            second = float(d[int(order[1])]) if len(order) > 1 else float("inf")
            return int(ids[idx0]), float(d[idx0]), second

        cand_idx = self._candidate_indices_from_majority(centers, h, self.countproto_topk)
        h_bits = self._unpack_one(h)
        cands = [self.novel[ids[int(i)]] for i in cand_idx]
        d = soft_count_distance_to_candidates(h_bits, cands)
        order = np.argsort(d)
        j0 = int(order[0])
        second = float(d[int(order[1])]) if len(order) > 1 else float("inf")
        return int(ids[int(cand_idx[j0])]), float(d[j0]), second

    def _should_promote(self, idx: int, t: int) -> Tuple[bool, str]:
        c = self.temp[idx]
        if c["count"] < self.min_cluster_size:
            return False, "too_small"
        if int(t) - int(c["created_t"]) < self.min_cluster_age:
            return False, "too_young"
        if self._radius(c) > self.max_cluster_radius:
            return False, "too_broad"
        if float(c["score_sum"] / max(c["count"], 1)) < self.min_mean_stage1_score:
            return False, "not_suspicious_enough"
        return True, "promote"

    def _promote(self, idx: int, t: int, true_label_for_log: Optional[int] = None) -> int:
        c = self.temp[idx]
        nid = int(self.next_novel_id)
        self.next_novel_id += 1
        self.novel[nid] = {
            "bit_counts": c["bit_counts"].copy(),
            "proto": c["proto"].copy(),
            "count": int(c["count"]),
            "score_sum": float(c["score_sum"]),
            "created_t": int(c["created_t"]),
            "last_t": int(t),
            "source_uid": int(c["uid"]),
        }
        ev = {
            "t": int(t),
            "new_class_id": nid,
            "source_uid": int(c["uid"]),
            "cluster_size": int(c["count"]),
            "cluster_radius": self._radius(c),
            "mean_stage1_score": float(c["score_sum"] / max(c["count"], 1)),
            "true_label_at_promotion_step": None if true_label_for_log is None else int(true_label_for_log),
        }
        self.promotion_events.append(ev)
        del self.temp[idx]
        return nid

    def update_novel(self, novel_id: int, h: np.ndarray, score: float):
        c = self.novel[int(novel_id)]
        c["bit_counts"] += self._unpack_one(h)
        c["count"] = int(c["count"]) + 1
        c["proto"] = self._make_proto(c["bit_counts"], int(c["count"]))
        c["score_sum"] = float(c["score_sum"] + float(score))

    def _merge_two_cluster_dicts(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        if (b["count"] > a["count"]) or (b["count"] == a["count"] and b["created_t"] < a["created_t"]):
            a, b = b, a
        bit_counts = a["bit_counts"] + b["bit_counts"]
        count = int(a["count"] + b["count"])
        return {
            "uid": int(a["uid"]),
            "bit_counts": bit_counts.astype(np.int32),
            "proto": self._make_proto(bit_counts, count),
            "count": count,
            "score_sum": float(a["score_sum"] + b["score_sum"]),
            "created_t": int(min(a["created_t"], b["created_t"])),
            "last_t": int(max(a["last_t"], b["last_t"])),
        }

    def _merge_temp_once(self) -> int:
        if len(self.temp) < 2 or self.merge_temp_tau <= 0:
            return 0
        merges = 0
        changed = True
        while changed and len(self.temp) >= 2:
            changed = False
            # Merging remains based on packed majority prototypes for speed.
            # Assignment/radius/promotion can use the softer bit-count distance.
            centers = np.stack([c["proto"] for c in self.temp], axis=0).astype(np.uint64)
            D = hamming_matrix(centers, centers, b_chunk=256)
            np.fill_diagonal(D, np.inf)
            i, j = np.unravel_index(int(np.argmin(D)), D.shape)
            best = float(D[i, j])
            if best <= self.merge_temp_tau:
                if j < i:
                    i, j = j, i
                ci, cj = self.temp[i], self.temp[j]
                merged = self._merge_two_cluster_dicts(ci, cj)
                self.uid_parent[int(cj["uid"])] = int(merged["uid"])
                self.uid_parent[int(ci["uid"])] = int(merged["uid"])
                self.temp[i] = merged
                del self.temp[j]
                merges += 1
                changed = True
        return merges

    def maybe_merge_temp(self):
        if self.merge_temp_every <= 0:
            return
        if self.processed_rejected > 0 and self.processed_rejected % self.merge_temp_every == 0:
            self.num_temp_merges += self._merge_temp_once()

    def process(self, h: np.ndarray, t: int, score: float, true_label_for_log: Optional[int] = None) -> Dict[str, Any]:
        h = np.asarray(h, dtype=np.uint64)
        self.processed_rejected += 1
        nid, nd, sd = self._nearest_novel_with_second(h)
        if nid is not None and nd <= self.novel_accept_tau:
            self.update_novel(nid, h, score)
            self.maybe_merge_temp()
            return {
                "stage2_status": "novel_assigned",
                "stage2_pred": int(nid),
                "novel_id": int(nid),
                "novel_distance": float(nd),
                "novel_second_distance": None if not np.isfinite(sd) else float(sd),
                "temp_uid": None,
                "temp_distance": None,
                "promoted": False,
                "new_class_id": None,
            }

        idx, d = self._nearest_temp(h)
        if idx is None or d > self.tau_cluster:
            uid = self._new_cluster(h, t, score)
            self.maybe_merge_temp()
            return {
                "stage2_status": "suspicious_unassigned",
                "stage2_pred": -1,
                "novel_id": None,
                "novel_distance": None if nid is None else float(nd),
                "novel_second_distance": None if not np.isfinite(sd) else float(sd),
                "temp_uid": int(uid),
                "temp_distance": None if idx is None else float(d),
                "promoted": False,
                "new_class_id": None,
            }

        uid_before = int(self.temp[idx]["uid"])
        self._update_cluster(idx, h, t, score)
        should, reason = self._should_promote(idx, t)
        if should:
            new_id = self._promote(idx, t, true_label_for_log=true_label_for_log)
            self.maybe_merge_temp()
            return {
                "stage2_status": "novel_promoted",
                "stage2_pred": int(new_id),
                "novel_id": int(new_id),
                "novel_distance": None if nid is None else float(nd),
                "novel_second_distance": None if not np.isfinite(sd) else float(sd),
                "temp_uid": int(uid_before),
                "temp_distance": float(d),
                "promoted": True,
                "new_class_id": int(new_id),
                "promote_reason": reason,
            }
        self.maybe_merge_temp()
        resolved_uid = self.resolve_uid(uid_before)
        return {
            "stage2_status": "unknown_clustered",
            "stage2_pred": -1,
            "novel_id": None,
            "novel_distance": None if nid is None else float(nd),
            "novel_second_distance": None if not np.isfinite(sd) else float(sd),
            "temp_uid": int(uid_before),
            "resolved_temp_uid": None if resolved_uid is None else int(resolved_uid),
            "temp_distance": float(d),
            "promoted": False,
            "new_class_id": None,
            "cluster_radius": self._radius(self.temp[idx]) if idx < len(self.temp) else None,
            "promote_reason": reason,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "num_temporary_clusters": int(len(self.temp)),
            "num_promoted_novel_clusters": int(len(self.novel)),
            "temporary_cluster_sizes": [int(c["count"]) for c in self.temp],
            "novel_cluster_sizes": {int(k): int(v["count"]) for k, v in self.novel.items()},
            "promotion_events": self.promotion_events,
            "tau_cluster": float(self.tau_cluster),
            "max_cluster_radius": float(self.max_cluster_radius),
            "novel_accept_tau": float(self.novel_accept_tau),
            "min_cluster_size": int(self.min_cluster_size),
            "min_mean_stage1_score": float(self.min_mean_stage1_score),
            "merge_temp_every": int(self.merge_temp_every),
            "merge_temp_scale": float(self.merge_temp_scale),
            "merge_temp_tau": float(self.merge_temp_tau),
            "num_temp_merges": int(self.num_temp_merges),
        }


def evaluate_stage1(model: BinaryHDCGate, H: np.ndarray, y: np.ndarray, known_ids: List[int], unknown_ids: List[int]) -> Dict[str, Any]:
    scores = model.predict_scores(H)
    accepted = scores["accepted"]
    known_mask = np.isin(y, known_ids)
    unknown_mask = np.isin(y, unknown_ids)
    y_bin = unknown_mask.astype(np.int64)
    out: Dict[str, Any] = {
        "known_accept_rate": float(np.mean(accepted[known_mask])) if np.any(known_mask) else float("nan"),
        "unknown_reject_rate": float(np.mean(~accepted[unknown_mask])) if np.any(unknown_mask) else float("nan"),
        "false_unknown_rate_on_known": float(np.mean(~accepted[known_mask])) if np.any(known_mask) else float("nan"),
        "false_known_rate_on_unknown": float(np.mean(accepted[unknown_mask])) if np.any(unknown_mask) else float("nan"),
        "binary_hdc_gate_reject_rate_on_known": float(np.mean(~accepted[known_mask])) if np.any(known_mask) else float("nan"),
        "binary_hdc_gate_reject_rate_on_unknown": float(np.mean(~accepted[unknown_mask])) if np.any(unknown_mask) else float("nan"),
        "unknown_auroc_by_binary_hdc_gate": binary_auroc(y_bin, scores["unknown_score_binary_gate"]),
        "unknown_auprc_by_binary_hdc_gate": binary_auprc(y_bin, scores["unknown_score_binary_gate"]),
        "num_eval": int(len(y)),
        "num_known_eval": int(np.sum(known_mask)),
        "num_unknown_eval": int(np.sum(unknown_mask)),
    }
    out.update(open_set_tradeoff_at_fixed_known_accept(
        scores["unknown_score_binary_gate"][known_mask],
        scores["unknown_score_binary_gate"][unknown_mask],
        prefix="binary_hdc_gate",
    ))
    return out


def evaluate_stage2(y_true: np.ndarray, stage1_accepted: np.ndarray, stage2_pred: np.ndarray,
                    stage2_status: List[str], known_ids: List[int], unknown_ids: List[int]) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    stage1_accepted = np.asarray(stage1_accepted, dtype=bool)
    stage2_pred = np.asarray(stage2_pred, dtype=np.int64)
    known_mask = np.isin(y_true, known_ids)
    unknown_mask = np.isin(y_true, unknown_ids)
    detected_unknown = unknown_mask & (~stage1_accepted)
    assigned_novel = unknown_mask & (stage2_pred >= 100000)
    suspicious_unassigned = detected_unknown & (stage2_pred < 100000)
    out: Dict[str, Any] = {
        "stage1_known_accept_rate": float(np.mean(stage1_accepted[known_mask])) if np.any(known_mask) else float("nan"),
        "stage1_unknown_reject_rate": float(np.mean(~stage1_accepted[unknown_mask])) if np.any(unknown_mask) else float("nan"),
        "stage2_novel_assigned_fraction_of_unknown": float(np.mean(assigned_novel[unknown_mask])) if np.any(unknown_mask) else float("nan"),
        "stage2_suspicious_unassigned_fraction_of_unknown": float(np.mean(suspicious_unassigned[unknown_mask])) if np.any(unknown_mask) else float("nan"),
        "stage2_suspicious_unassigned_fraction_of_detected_unknown": float(np.mean(suspicious_unassigned[detected_unknown])) if np.any(detected_unknown) else float("nan"),
        "stage2_num_novel_assigned_unknown": int(np.sum(assigned_novel)),
        "stage2_num_detected_unknown": int(np.sum(detected_unknown)),
    }
    if np.any(assigned_novel):
        y_a = y_true[assigned_novel]
        p_a = stage2_pred[assigned_novel]
        out["stage2_novel_assignment_purity"] = cluster_purity(y_a, p_a)
        cm = safe_cluster_metrics(y_a, p_a)
        out["stage2_novel_assignment_ari"] = cm["ari"]
        out["stage2_novel_assignment_nmi"] = cm["nmi"]
        out["stage2_novel_to_true_majority_map"] = best_map_cluster_to_true(y_a, p_a)
    else:
        out["stage2_novel_assignment_purity"] = float("nan")
        out["stage2_novel_assignment_ari"] = float("nan")
        out["stage2_novel_assignment_nmi"] = float("nan")
        out["stage2_novel_to_true_majority_map"] = {}
    discovered, delays = {}, {}
    for y in unknown_ids:
        idx = np.where(y_true == int(y))[0]
        if len(idx) == 0:
            continue
        hit = idx[stage2_pred[idx] >= 100000]
        discovered[int(y)] = bool(len(hit) > 0)
        delays[int(y)] = None if len(hit) == 0 else int(hit[0] - idx[0])
    vals = [d for d in delays.values() if d is not None]
    out["stage2_novel_discovery_recall"] = float(np.mean(list(discovered.values()))) if discovered else float("nan")
    out["stage2_avg_discovery_delay"] = float(np.mean(vals)) if vals else float("nan")
    out["stage2_per_unknown_class_discovered"] = discovered
    out["stage2_per_unknown_class_delay"] = delays
    return out


def classwise_stage2_behavior(y_true: np.ndarray, label_names: List[str], known_ids: List[int], unknown_ids: List[int],
                              stage1_accepted: np.ndarray, stage2_pred: np.ndarray, status: List[str]) -> List[Dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=np.int64)
    status = np.asarray(status)
    rows = []
    for cid in known_ids + unknown_ids:
        mask = y_true == int(cid)
        if not np.any(mask):
            continue
        rows.append({
            "class_id": int(cid),
            "class_name": label_names[int(cid)] if int(cid) < len(label_names) else str(cid),
            "count": int(np.sum(mask)),
            "accepted_as_normal_rate": float(np.mean(stage1_accepted[mask])),
            "rejected_suspicious_rate": float(np.mean(~stage1_accepted[mask])),
            "novel_assigned_rate": float(np.mean(stage2_pred[mask] >= 100000)),
            "suspicious_unassigned_rate": float(np.mean((~stage1_accepted[mask]) & (stage2_pred[mask] < 100000))),
            "unknown_clustered_rate": float(np.mean(status[mask] == "unknown_clustered")),
            "novel_promoted_rate": float(np.mean(status[mask] == "novel_promoted")),
        })
    return rows


def stage2_auto_thresholds_binary(H_val: np.ndarray, rejected_val: np.ndarray, scores_val: np.ndarray,
                                  stage1_tau: float, q: float = 85.0, radius_scale: float = 1.5,
                                  novel_scale: float = 1.5, max_points: int = 3000, seed: int = 0) -> Dict[str, Any]:
    H_val = np.asarray(H_val, dtype=np.uint64)
    idx = np.where(np.asarray(rejected_val, dtype=bool))[0]
    if len(idx) < 2:
        idx = np.arange(min(len(H_val), max_points))
    S = H_val[idx]
    nn = leave_one_out_nearest_hamming_sample(S, max_points=max_points, seed=seed)
    tau = float(np.percentile(nn, float(q)))
    tau = max(tau, 1.0)
    return {
        "tau_cluster": float(tau),
        "max_cluster_radius": float(tau * float(radius_scale)),
        "novel_accept_tau": float(tau * float(novel_scale)),
        "min_mean_stage1_score": float(stage1_tau),
        "stage2_auto_q": float(q),
        "stage2_auto_num_points": int(min(len(S), max_points)),
        "nn_hamming_mean": float(np.mean(nn)),
        "nn_hamming_median": float(np.median(nn)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm811k_pkl", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./stage1_stage2_binary_hdc")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--none_cap", type=int, default=5000)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clip_low_q", type=float, default=1.0)
    parser.add_argument("--clip_high_q", type=float, default=99.0)
    parser.add_argument("--defect_threshold", type=float, default=0.5)
    parser.add_argument("--invalid_value", type=float, default=-1.0)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--overwrite_cache", action="store_true")

    # Binary HDC encoder.  --hd_dim is accepted for compatibility and maps to
    # the number of random projections, not the final bit dimension.
    parser.add_argument("--binary_num_proj", "--hd_dim", dest="binary_num_proj", type=int, default=1024,
                        help="Number of random projection coordinates before thermometer encoding. Final bit dimension is binary_num_proj * binary_num_bins.")
    parser.add_argument("--binary_num_bins", type=int, default=8, help="Thermometer bits per random projection coordinate.")
    parser.add_argument("--binary_clip_value", type=float, default=4.0)
    parser.add_argument("--binary_encode_batch_size", type=int, default=1024)

    # Stage 1
    parser.add_argument("--local_gate_known_reject_q", "--threshold_quantile", dest="known_reject_q", type=float, default=10.0)
    parser.add_argument("--num_subprototypes", type=int, default=10)
    parser.add_argument("--subproto_max_iter", type=int, default=8)
    parser.add_argument("--local_k", type=int, default=10)
    parser.add_argument("--max_ref_normal", type=int, default=2000)
    parser.add_argument("--local_score_batch_size", type=int, default=64)

    # Stage 2
    parser.add_argument("--stage2_min_cluster_size", type=int, default=50)
    parser.add_argument("--stage2_min_cluster_age", type=int, default=0)
    parser.add_argument("--stage2_tau_cluster", type=float, default=None)
    parser.add_argument("--stage2_max_cluster_radius", type=float, default=None)
    parser.add_argument("--stage2_novel_accept_tau", type=float, default=None)
    parser.add_argument("--stage2_auto_q", type=float, default=85.0)
    parser.add_argument("--stage2_radius_scale", type=float, default=1.5)
    parser.add_argument("--stage2_novel_scale", type=float, default=1.5)
    parser.add_argument("--stage2_max_auto_points", type=int, default=3000)
    parser.add_argument("--stage2_max_temp_clusters", type=int, default=1000)
    parser.add_argument("--stage2_min_mean_score", type=float, default=None)
    parser.add_argument("--stage2_large_temp_min_size", type=int, default=20)
    parser.add_argument("--stage2_merge_temp_every", type=int, default=0)
    parser.add_argument("--stage2_merge_temp_scale", type=float, default=0.5)
    parser.add_argument("--stage2_proto_distance", type=str, default="count", choices=["count", "majority"],
                        help="Stage-2 cluster distance. 'count' uses soft bit-count centroid distance; 'majority' reproduces the old hard-majority Hamming prototype.")
    parser.add_argument("--stage2_countproto_topk", type=int, default=64,
                        help="Number of majority-prototype candidates reranked by bit-count centroid distance.")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    print("Loading fixed normal-only phased-stream data...")
    data = prepare_normal_only_phased_data(
        wm811k_pkl=args.wm811k_pkl,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        none_cap=args.none_cap,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        defect_threshold=args.defect_threshold,
        invalid_value=args.invalid_value,
        clip_low_q=args.clip_low_q,
        clip_high_q=args.clip_high_q,
        feature_cache_dir=args.cache_dir,
        overwrite_feature_cache=args.overwrite_cache,
        return_wafer_maps=False,
        aux_feature_set="original",
    )

    print("\nEncoding 202-D auxiliary features into binary thermometer HDC vectors...")
    enc = BinaryThermometerHDC(
        in_dim=data.X_aux_train.shape[1],
        num_proj=args.binary_num_proj,
        num_bins=args.binary_num_bins,
        seed=args.seed,
        clip_value=args.binary_clip_value,
    )
    t0 = time.time()
    H_train = enc.fit_transform(data.X_aux_train, batch_size=args.binary_encode_batch_size)
    H_val = enc.transform(data.X_aux_val, batch_size=args.binary_encode_batch_size)
    H_test = enc.transform(data.X_aux_test, batch_size=args.binary_encode_batch_size)
    encode_time = time.time() - t0
    print(f"Binary HDC encoding time: {encode_time:.2f}s")
    print(f"Binary HDC: {enc.bit_dim} bits = {enc.num_words} uint64 words per wafer")

    print("\nFitting Stage-1 binary-HDC normal gate...")
    stage1 = BinaryHDCGate(
        bit_dim=enc.bit_dim,
        num_subprototypes=args.num_subprototypes,
        subproto_max_iter=args.subproto_max_iter,
        local_k=args.local_k,
        max_ref_normal=args.max_ref_normal,
        score_batch_size=args.local_score_batch_size,
        seed=args.seed,
    )
    stage1.fit(H_train[data.train_known_mask])
    calibration = stage1.calibrate(H_val[data.val_known_mask], known_reject_q=args.known_reject_q)
    print("Stage-1 calibration:", calibration)

    val_metrics = evaluate_stage1(stage1, H_val, data.y_val, data.known_ids, data.unknown_ids)
    print_metrics("STATIC OPEN-SET VALIDATION", val_metrics)
    static_test = evaluate_stage1(stage1, H_test, data.y_test, data.known_ids, data.unknown_ids)
    print_metrics("STATIC OPEN-SET TEST", static_test)

    print("\nGathering fixed phased stream binary HDC arrays...")
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
        q=args.stage2_auto_q,
        radius_scale=args.stage2_radius_scale,
        novel_scale=args.stage2_novel_scale,
        max_points=args.stage2_max_auto_points,
        seed=args.seed,
    )
    tau_cluster = auto["tau_cluster"] if args.stage2_tau_cluster is None else float(args.stage2_tau_cluster)
    max_radius = auto["max_cluster_radius"] if args.stage2_max_cluster_radius is None else float(args.stage2_max_cluster_radius)
    novel_tau = auto["novel_accept_tau"] if args.stage2_novel_accept_tau is None else float(args.stage2_novel_accept_tau)
    min_mean_score = auto["min_mean_stage1_score"] if args.stage2_min_mean_score is None else float(args.stage2_min_mean_score)
    stage2_calib = dict(auto)
    stage2_calib.update({
        "tau_cluster_used": float(tau_cluster),
        "max_cluster_radius_used": float(max_radius),
        "novel_accept_tau_used": float(novel_tau),
        "min_mean_stage1_score_used": float(min_mean_score),
        "stage2_proto_distance": str(args.stage2_proto_distance),
        "stage2_countproto_topk": int(args.stage2_countproto_topk),
    })
    print("Stage-2 calibration:", stage2_calib)

    print("\n================ ONLINE STREAM STAGE-2 BINARY-HDC CLUSTERING ================")
    clusterer = BinaryHDCOnlineClusterer(
        bit_dim=enc.bit_dim,
        tau_cluster=tau_cluster,
        max_cluster_radius=max_radius,
        novel_accept_tau=novel_tau,
        min_cluster_size=args.stage2_min_cluster_size,
        min_cluster_age=args.stage2_min_cluster_age,
        min_mean_stage1_score=min_mean_score,
        max_temp_clusters=args.stage2_max_temp_clusters,
        novel_id_start=100000,
        merge_temp_every=args.stage2_merge_temp_every,
        merge_temp_scale=args.stage2_merge_temp_scale,
        proto_distance=args.stage2_proto_distance,
        countproto_topk=args.stage2_countproto_topk,
    )

    stage2_pred = np.full(len(data.y_stream_true), -1, dtype=np.int64)
    stage2_status: List[str] = []
    per_sample_records = []
    t0 = time.time()
    for t, (h, y_true) in enumerate(zip(H_stream, data.y_stream_true)):
        accepted = bool(stage1_accepted[t])
        rec = {
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
            out = clusterer.process(h, t=t, score=float(stream_scores["unknown_score_binary_gate"][t]), true_label_for_log=int(y_true))
            stage2_status.append(str(out["stage2_status"]))
            stage2_pred[t] = int(out["stage2_pred"])
            rec.update(out)
        per_sample_records.append(rec)
    stage2_time = time.time() - t0

    stage2_metrics = evaluate_stage2(
        y_true=data.y_stream_true,
        stage1_accepted=stage1_accepted,
        stage2_pred=stage2_pred,
        stage2_status=stage2_status,
        known_ids=data.known_ids,
        unknown_ids=data.unknown_ids,
    )
    cluster_summary = clusterer.summary()

    temp_size_by_uid = {int(c["uid"]): int(c["count"]) for c in clusterer.temp}
    large_temp_uids = {uid for uid, cnt in temp_size_by_uid.items() if cnt >= int(args.stage2_large_temp_min_size)}
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
        "stage2_large_temp_min_size": int(args.stage2_large_temp_min_size),
        "stage2_num_large_temporary_clusters": int(len(large_temp_uids)),
        "stage2_large_temp_fraction_of_unknown": float(np.mean(large_temp_unknown[unknown_mask])) if np.any(unknown_mask) else float("nan"),
        "stage2_large_temp_fraction_of_detected_unknown": float(np.mean(large_temp_unknown[detected_unknown])) if np.any(detected_unknown) else float("nan"),
        "stage2_meaningful_grouped_fraction_of_unknown": float(np.mean((novel_mask | large_temp_mask)[unknown_mask])) if np.any(unknown_mask) else float("nan"),
        "stage2_large_temp_fraction_of_known": float(np.mean(large_temp_known[known_mask])) if np.any(known_mask) else float("nan"),
    })

    print_metrics("ONLINE STREAM STAGE-2 METRICS", stage2_metrics)
    print(f"stage2_time_sec: {stage2_time:.4f}")
    print(f"stage2_time_per_sample_sec: {stage2_time / max(len(data.y_stream_true), 1):.8f}")

    cluster_summary["large_temp_min_size"] = int(args.stage2_large_temp_min_size)
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

    out_json = {
        "method": "normal_only_binary_thermometer_hdc_countproto_memory",
        "description": "Original 202-D auxiliary descriptors encoded into distance-preserving binary thermometer HDC. Stage 1 uses Hamming distance. Stage 2 uses binary HDC bundle memories with optional soft bit-count centroid distance.",
        "binary_hdc_encoder": enc.meta(),
        "label_names": data.label_names,
        "normal_label_id": int(data.normal_label),
        "known_ids": data.known_ids,
        "known_names": data.known_names,
        "unknown_ids": data.unknown_ids,
        "unknown_names": data.unknown_names,
        "feature_names": data.feature_names,
        "stage1_known_reject_q": float(args.known_reject_q),
        "stage1_calibration": calibration,
        "static_validation": val_metrics,
        "static_test": static_test,
        "stage1_online_stream_metrics": stage1_stream_metrics,
        "stage2_calibration": stage2_calib,
        "stage2_online_metrics": stage2_metrics,
        "stage2_cluster_summary": cluster_summary,
        "classwise_stream_stage2": class_rows,
        "binary_encoding_time_sec": encode_time,
        "stage2_time_sec": stage2_time,
        "stage2_time_per_sample_sec": stage2_time / max(len(data.y_stream_true), 1),
    }

    with open(os.path.join(args.save_dir, "stage1_stage2_results.json"), "w") as f:
        json.dump(out_json, f, indent=2, default=json_default)
    with open(os.path.join(args.save_dir, "stage1_stage2_stream_records.json"), "w") as f:
        json.dump(per_sample_records, f, indent=2, default=json_default)
    with open(os.path.join(args.save_dir, "feature_names.json"), "w") as f:
        json.dump(data.feature_names, f, indent=2)
    print(f"\nSaved Stage-1+Stage-2 outputs to: {args.save_dir}")


if __name__ == "__main__":
    main()
