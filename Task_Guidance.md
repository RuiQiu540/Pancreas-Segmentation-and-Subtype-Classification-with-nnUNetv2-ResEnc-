# Task Guidance (nnUNetv2 · 2D/3D)

> Minimal, copy‑paste friendly steps to go from raw data → nnUNetv2 format → plan & preprocess → train → predict.
> Works on Windows PowerShell; Bash is analogous.

---

## Table of Contents
- [0) Prerequisites](#0-prerequisites)
- [1) Environment (PowerShell)](#1-environment-powershell)
- [2) Convert raw data → nnUNetv2 RAW](#2-convert-raw-data--nnunetv2-raw)
- [3) Planning & Preprocessing](#3-planning--preprocessing)
  - [A) One‑liner (quick start)](#a-one-liner-quick-start)
  - [B) Manual 3‑step (inspect/modify plans)](#b-manual-3-step-inspectmodify-plans)
  - [Read the plans identifier](#read-the-plans-identifier)
- [4) Train](#4-train)
- [5) Predict](#5-predict)
- [6) Package submission (results.zip)](#6-package-submission-resultszip)
- [7) Troubleshooting & tips](#7-troubleshooting--tips)
- [8) Optional: 3D configurations](#8-optional-3d-configurations)

---

## 0) Prerequisites

- CUDA‑capable GPU (tested on NVIDIA RTX 3070 8 GB).
- Python 3.10+ and Conda.
- PyTorch built for your CUDA version.
- nnUNetv2 installed (editable install recommended).

---

## 1) Environment (PowerShell)

```powershell
# Go to your project root
cd D:\path	o\your\project

# Activate your conda env (make sure PyTorch is installed)
conda activate <ENV_NAME>

# Install nnUNetv2 in editable mode (run inside its repo if needed)
pip install -e .

# nnUNet path variables (edit to your own)
$env:nnUNet_raw="D:\...
nUNet_raw"
$env:nnUNet_preprocessed="D:\...
nUNet_preprocessed"
$env:nnUNet_results="D:\...
nUNet_results"

# Optional performance flags
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="1"
```

---

## 2) Convert raw data → nnUNetv2 RAW

Your source layout (before conversion) should follow this pattern:
```
data/
  train/subtype{0,1,2}/   quiz_XXX_0000.nii.gz  +  quiz_XXX.nii.gz
  validation/subtype{0,1,2}/  (same pattern as train)
  test/  quiz_YYY_0000.nii.gz
```

Run the converter (adjust arguments):
```powershell
python .\convert_to_nnunet.py --src D:\path	o\data --dataset-id 701 --dataset-name PancreasQuiz
```

**Outputs:**
```
$env:nnUNet_raw\Dataset701_PancreasQuiz  imagesTr\   labelsTr\   imagesTs\   dataset.json

$env:nnUNet_preprocessed\Dataset701_PancreasQuiz  case_to_subtype.csv   splits_final.json
```

> Notes:
> - `imagesTr` contains both training **and** validation images; `labelsTr` their labels.
> - `imagesTs` is for test images (no labels).
> - Labels are integerized to {0,1,2}.

---

## 3) Planning & Preprocessing

### A) One‑liner (quick start)

```powershell
nnUNetv2_plan_and_preprocess -d 701 -c 2d -pl nnUNetPlannerResEncM
```
- Creates `plans.json` and runs preprocessing for **2D ResEnc‑M** (good fit for 8 GB VRAM).
- You may try `-c 3d_lowres` or `-c 3d_fullres` if your GPU allows.

### B) Manual 3‑step (inspect/modify plans)

```powershell
# 1) Fingerprint (checks/collects dataset stats)
nnUNetv2_extract_dataset_fingerprint -d 701 --verify_dataset_integrity

# 2) Plan (choose configuration & planner tier)
nnUNetv2_plan_experiment -d 701 -c 2d -pl nnUNetPlannerResEncM

# 3) Preprocess (uses the generated plans)
nnUNetv2_preprocess -d 701 -c 2d
```

### Read the plans identifier

You must use the **same** plans identifier for train/predict:
```powershell
$pre="$env:nnUNet_preprocessed\Dataset701_PancreasQuiz"
$plans_id=(Get-Content "$pre\plans.json" | ConvertFrom-Json).plans_identifier
```

---

## 4) Train

```powershell
# Fold 0 (splits_final.json defines the fixed validation cases)
nnUNetv2_train 701 2d 0 -tr nnUNetTrainer -p $plans_id
```
Outputs go to:
```
$env:nnUNet_results\Dataset701_PancreasQuiz\...
```

**RTX 3070 8 GB tips:** keep AMP on by default; if OOM, lower batch size first, then slightly reduce patch size.

---

## 5) Predict

```powershell
nnUNetv2_predict `
  -d 701 -c 2d -f 0 -tr nnUNetTrainer -p $plans_id -chk best `
  -i "$env:nnUNet_raw\Dataset701_PancreasQuiz\imagesTs" `
  -o "$env:nnUNet_results\Dataset701_PancreasQuiz\pred_2d_f0"
```

- Optional speed/accuracy trade‑offs:
  - `--disable_tta` (faster inference, a little less accurate)
  - `--step_size 0.7` (default is 0.5; larger is faster but may reduce accuracy)
- Outputs include predicted `quiz_XXX.nii.gz` masks; your pipeline may also emit `subtype_results.csv` with columns `Names,Subtype`.

---

## 6) Package submission (results.zip)

Per typical assignment rules, submit a zip containing **only test outputs**:
- Segmentation masks: `quiz_XXX.nii.gz` (no `_0000` suffix; integer labels {0,1,2}).
- Classification CSV: `case_to_subtype.csv` with **exact** headers: `Names,Subtype`.

Checklist:
- File count matches test set size; filenames exactly match expected IDs.
- CSV encoding UTF‑8 (no BOM), comma separated, no trailing spaces.

---

## 7) Troubleshooting & tips

- **Environment variables:** make sure `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` are set in the same shell running the commands.
- **Label dtype:** labels must be integer (0/1/2). Re‑run your converter if needed.
- **Extension fix:** if your CSV `Names` column lacks `.nii.gz`, append it (e.g., via a one‑liner or a tiny Python script).
- **Plans mismatch:** always use the `plans_identifier` read from the preprocessed dataset’s `plans.json` for both train & predict.
- **OOM:** reduce batch first; disable TTA at prediction; fall back from 3D to 2D if needed.
- **Integrity check:** `nnUNetv2_extract_dataset_fingerprint -d <ID> --verify_dataset_integrity` to verify images/labels pairing and shapes.

---

## 8) Optional: 3D configurations

If you want to try 3D (requires more VRAM/time):

```powershell
# Plan & preprocess (3D fullres with ResEnc‑M; use ResEnc‑L if OOM)
nnUNetv2_plan_experiment -d 701 -c 3d_fullres -pl nnUNetPlannerResEncM
nnUNetv2_preprocess -d 701 -c 3d_fullres

# Train (fold 0)
nnUNetv2_train 701 3d_fullres 0 -tr nnUNetTrainer -p $plans_id

# Predict (TTA off recommended for speed/memory)
nnUNetv2_predict `
  -d 701 -c 3d_fullres -f 0 -tr nnUNetTrainer -p $plans_id -chk best `
  -i "$env:nnUNet_raw\Dataset701_PancreasQuiz\imagesTs" `
  -o "$env:nnUNet_results\Dataset701_PancreasQuiz\pred_3d_f0" `
  --disable_tta
```

> If you change planner/configuration, re‑run planning + preprocessing and refresh `$plans_id`.
