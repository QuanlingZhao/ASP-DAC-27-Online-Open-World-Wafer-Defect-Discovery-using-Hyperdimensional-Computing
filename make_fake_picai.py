#!/usr/bin/env python3
"""
make_fake_picai.py - write a tiny fake PI-CAI dataset (same folders and file names)
so the scripts can be smoke-tested before downloading the real images.

The fake prostate is a textured ellipsoid; positive cases get a lesion that is
dark on ADC and bright on HBV, as csPCa usually looks. Results on it mean nothing.
"""
import argparse
import csv
import os

import numpy as np
import SimpleITK as sitk


def save(arr, path, spacing):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(img, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./fake_picai")
    ap.add_argument("--n_neg", type=int, default=16)
    ap.add_argument("--n_pos", type=int, default=8)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    img_dir = os.path.join(args.out, "images", "picai_public_images_fold0")
    lab = os.path.join(args.out, "picai_labels")
    rows = []
    for i in range(args.n_neg + args.n_pos):
        pid, sid = 10000 + i, 1000000 + i
        pos = i >= args.n_neg
        Z, Y, X = 20, 128, 128
        zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
        c = np.array([Z / 2, Y / 2 + rng.integers(-6, 6), X / 2 + rng.integers(-6, 6)])
        r = np.array([7, 26 + rng.integers(-4, 4), 32 + rng.integers(-4, 4)])
        gland = (((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2) <= 1
        tex = rng.normal(0, 1, (Z, Y, X))
        t2w = 300 + 200 * gland + 40 * tex
        adc = 800 + 600 * gland + 60 * tex
        hbv = 20 + 15 * gland + 4 * tex
        lesion = np.zeros((Z, Y, X), np.uint8)
        if pos:
            lc = c + np.array([rng.integers(-2, 3), rng.integers(-10, 10), rng.integers(-12, 12)])
            les = (((zz - lc[0]) / 3) ** 2 + ((yy - lc[1]) / 7) ** 2 + ((xx - lc[2]) / 7) ** 2) <= 1
            les &= gland
            lesion[les] = 2 + int(rng.integers(0, 3))
            t2w[les] -= 120; adc[les] -= 700; hbv[les] += 25
        key = f"{pid}_{sid}"
        sp = (0.5, 0.5, 3.0)
        for name, v in (("t2w", t2w), ("adc", adc), ("hbv", hbv)):
            save(v.astype(np.float32), os.path.join(img_dir, str(pid), f"{key}_{name}.mha"), sp)
        save(gland.astype(np.uint8), os.path.join(lab, "anatomical_delineations/whole_gland/AI/Bosma22b", key + ".nii.gz"), sp)
        save(lesion, os.path.join(lab, "csPCa_lesion_delineations/human_expert/resampled", key + ".nii.gz"), sp)
        rows.append({"patient_id": pid, "study_id": sid, "case_ISUP": int(lesion.max()) if pos else 0,
                     "case_csPCa": "YES" if pos else "NO"})
    os.makedirs(os.path.join(lab, "clinical_information"), exist_ok=True)
    with open(os.path.join(lab, "clinical_information", "marksheet.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("fake dataset written to", args.out)


if __name__ == "__main__":
    main()
