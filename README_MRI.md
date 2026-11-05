# PI-CAI MRI add-on for Eric's WaferHDC code

Copy these files into the root of your fork of Eric's repo (next to `run_full_hdc.py`).
None of Eric's files are modified; these import his encoder, gate, clusterer and metrics.

| File | Role |
| --- | --- |
| `picai_slices.py` | PI-CAI 3D cases -> prostate-cropped 2D slices (T2W, ADC, HBV) in one .npz |
| `mri_features.py` | MRI feature sets that replace the 202 wafer features: `pixels`, `stats`, `resnet`, `stats+pixels` |
| `mri_artifacts.py` | Synthetic low-quality slices (noise, blur, ghosting, bias field) for the quality task |
| `picai_bundle.py` | MRI version of `prepare_normal_only_phased_data`: patient-level split, features, same `WaferDataBundle` |
| `run_mri_hdc.py` | Stage 1 + Stage 2 on MRI, plus a no-HDC baseline, patient-level AUROC and a picture of top/bottom slices |
| `make_fake_picai.py` | Tiny fake dataset in PI-CAI layout, for a smoke test before downloading |

Extra packages on top of `requirements.txt`: `pip install SimpleITK matplotlib`

## Smoke test (2 minutes, no download)

    python make_fake_picai.py --out fake
    python picai_slices.py --images_dir fake/images --labels_dir fake/picai_labels --out fake_slices.npz
    python run_mri_hdc.py --slices_npz fake_slices.npz --task quality --stage2_min_cluster_size 5

## Real data

    git clone https://github.com/DIAGNijmegen/picai_labels
    # images: https://zenodo.org/record/6624726 (download picai_public_images_fold0.zip first, unzip into ./picai_images)
    python picai_slices.py --images_dir picai_images --labels_dir picai_labels --out picai_fold0.npz
    python run_mri_hdc.py --slices_npz picai_fold0.npz --task quality --feature_set stats --save_dir results/quality_stats
    python run_mri_hdc.py --slices_npz picai_fold0.npz --task lesion  --feature_set stats --save_dir results/lesion_stats

Each run writes `results.json` and `test_extremes.png` into `--save_dir`.
Tested only on the fake dataset so far.
