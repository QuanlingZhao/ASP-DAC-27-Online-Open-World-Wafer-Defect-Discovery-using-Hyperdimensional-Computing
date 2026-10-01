# Online Open-World Wafer Defect Discovery using Hyperdimensional Computing

Code release for ASP-DAC 2027 paper: Online Open-World Wafer Defect Discovery using Hyperdimensional Computing

The pipeline has two stages:

1. **Stage 1 — normality screening:** encode wafer descriptors into binary HDC vectors and reject suspicious/unknown wafers using Hamming-distance-based normal memory.
2. **Stage 2 — online defect discovery:** incrementally group rejected wafers using temporary and promoted binary HDC bundle memories.

## Repository structure

```text
.
├── data/
│   └── LSWMD.pkl
├── eval_full_pipeline_hdc.py
├── eval_stage1_hdc.py
├── eval_stage2_hdc.py
├── get_data.py
├── run_full_hdc.py
├── wafer_data_prep_hdc_stage2.py
├── wafer_hdc_encoding.py
├── README.md
└── requirements.txt
```

## Requirements

Python **3.10+** is recommended.

Create and activate a virtual environment, then install the dependencies:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

This code uses the **WM-811K / LSWMD** wafer-map dataset.

Obtain the dataset separately and place the pickle file at:

```text
data/LSWMD.pkl
```

The evaluation scripts use this path by default.

## Quick start

Run the commands below from the repository root.

### 1. Stage-1 HDC normality gate

```bash
python eval_stage1_hdc.py
```

This evaluates the binary-HDC Stage-1 normal/unknown screening gate.

### 2. Stage-2 HDC defect discovery

```bash
python eval_stage2_hdc.py
```

**Note:** this is the Stage-2-only oracle/diagnostic experiment. It constructs a defect-only stream using ground-truth labels in order to isolate the online HDC clustering memory. It is not the end-to-end open-world evaluation.

### 3. Full two-stage pipeline

```bash
python eval_full_pipeline_hdc.py
```

This runs the paper-style end-to-end pipeline:

```text
Wafer map
   ↓
202-D auxiliary descriptors
   ↓
Binary thermometer HDC encoding
   ↓
Stage-1 normal gate
   ↓
Rejected / suspicious wafers
   ↓
Stage-2 online HDC memory
   ↓
Discovered defect modes
```

The evaluation scripts save their outputs into automatically created result directories.

## Configurable full-pipeline run

`run_full_hdc.py` exposes the main experimental parameters through command-line arguments.

A basic run is:

```bash
python run_full_hdc.py \
    --wm811k_pkl ./data/LSWMD.pkl \
    --save_dir ./stage1_stage2_binary_hdc
```

On Windows PowerShell:

```powershell
python run_full_hdc.py `
    --wm811k_pkl ./data/LSWMD.pkl `
    --save_dir ./stage1_stage2_binary_hdc
```

To see all available options:

```bash
python run_full_hdc.py --help
```

## Main files

- `wafer_hdc_encoding.py` — feature extraction, HDC encoding, and Hamming-distance utilities.
- `wafer_data_prep_hdc_stage2.py` — fixed normal-only data preparation and phased online-stream construction.
- `run_full_hdc.py` — main configurable Stage-1 + Stage-2 implementation.
- `eval_stage1_hdc.py` — paper-setting Stage-1 evaluation.
- `eval_stage2_hdc.py` — Stage-2-only oracle/diagnostic evaluation.
- `eval_full_pipeline_hdc.py` — paper-setting end-to-end evaluation.
- `get_data.py` — WM-811K/LSWMD loading and dataset utilities.

## Notes

- The paper evaluation scripts intentionally contain fixed/hard-coded settings for reproducibility.
- Feature caches and generated result folders are created locally and do not need to be committed to the repository.
- The core HDC pipeline is NumPy-based; a GPU is not required for the HDC computation itself.

## Citation

If you use this code, please cite the associated paper.

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  year    = {2026}
}
```

Replace the placeholder above with the final publication citation.
