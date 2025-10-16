#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One-shot converter for PancreasQuiz-like data to nnUNetv2 format.

Input layout (source):
data/
  train/
    subtype0/  subtype1/  subtype2/
      quiz_XXX_0000.nii.gz   # image
      quiz_XXX.nii.gz        # label (0=bg, 1=pancreas, 2=lesion)
  validation/
    subtype0/1/2/            # same as train
  test/                      # images only (quiz_YYY_0000.nii.gz)

Outputs:
  <NNUNET_RAW>/Dataset{ID}_{NAME}/
    imagesTr/    labelsTr/    imagesTs/    dataset.json
  <NNUNET_PREPROCESSED>/Dataset{ID}_{NAME}/
    case_to_subtype.csv       splits_final.json
And it will also *integerize* all label volumes under labelsTr (round->int32).

Environment variables (case-insensitive keys supported):
  NNUNET_RAW / nnUNet_raw
  NNUNET_PREPROCESSED / nnUNet_preprocessed
  NNUNET_RESULTS / nnUNet_results  (optional; printed for info)

Usage:
  python convert_to_nnunet_merged.py --src D:\path\to\data --dataset-id 701 --dataset-name PancreasQuiz

Notes:
 - All training + validation images/labels will be placed in imagesTr/labelsTr.
 - splits_final.json fixes the train/val split according to your source folders.
 - case_to_subtype.csv is inferred from the 'subtypeX' folder names (X in {0,1,2}).
 - Images must be named *_0000.nii(.gz); labels: same basename without _0000.
 - We *only* integerize labels (not images). Integer type: int32 (rounding first).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib

# ------------------------- helpers -------------------------

def first_env(*keys: str) -> Path:
    """
    Return Path for the first environment variable that exists.
    Keys are tried case-sensitively first; then a case-insensitive fallback.
    """
    for k in keys:
        v = os.environ.get(k, None)
        if v:
            return Path(v)
    env_lower = {k.lower(): v for k, v in os.environ.items()}
    for k in keys:
        if k.lower() in env_lower:
            return Path(env_lower[k.lower()])
    raise RuntimeError(f"Environment variable not set. Tried: {keys}")

def strip_suffixes(name: str) -> str:
    """Strip .nii/.nii.gz and trailing _0000 to get case ID (e.g., quiz_037)."""
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    return name

def is_img(fname: str) -> bool:
    return fname.endswith("_0000.nii.gz") or fname.endswith("_0000.nii")

def is_lbl(fname: str) -> bool:
    return (fname.endswith(".nii.gz") or fname.endswith(".nii")) and not is_img(fname)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy2(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if not dst.exists():
        shutil.copy2(src, dst)

# ------------------------- gather -------------------------

def gather_cases(src: Path, subset: str) -> List[Tuple[str, Path, Path, int]]:
    """
    Scan subset ('train' or 'validation') and gather (case_id, img, lbl, subtype).
    """
    cases = []
    base = src / subset
    if not base.exists():
        return cases
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        m = re.search(r"subtype\s*([0-9]+)", subdir.name, re.IGNORECASE)
        if not m:
            continue
        subtype = int(m.group(1))
        imgs = sorted([p for p in subdir.iterdir() if p.is_file() and is_img(p.name)])
        for img in imgs:
            case_id = strip_suffixes(img.name)
            # expected label next to image
            lbl1 = img.with_name(img.name.replace("_0000.nii.gz", ".nii.gz"))
            lbl2 = img.with_name(img.name.replace("_0000.nii", ".nii"))
            label = lbl1 if lbl1.exists() else lbl2
            if not label.exists():
                raise FileNotFoundError(f"Missing label for {img} (expected {lbl1.name} or {lbl2.name})")
            cases.append((case_id, img, label, subtype))
    return cases

def gather_test_images(src: Path) -> List[Tuple[str, Path]]:
    """Scan 'test' and gather (case_id, img)."""
    res = []
    base = src / "test"
    if not base.exists():
        return res
    for img in sorted(base.iterdir()):
        if img.is_file() and is_img(img.name):
            case_id = strip_suffixes(img.name)
            res.append((case_id, img))
    return res

# ------------------------- writers -------------------------

def write_dataset_json(dst_raw: Path, dataset_id: int, dataset_name: str, num_training: int):
    content = {
        "name": dataset_name,
        "description": "auto-generated",
        "tensorImageSize": "3D",
        "reference": "",
        "licence": "",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "pancreas": 1, "lesion": 2},
        "modality": {"0": "CT"},
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "numTraining": int(num_training),
        "dataset_id": int(dataset_id)
    }
    with open(dst_raw / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)
    return content

def write_case_to_subtype(dst_raw: Path, dst_pre: Path, train_cases, val_cases):
    """
    写 case_to_subtype.csv to RAW and PREPROCESSED directories
    Names eliminate suffix and eliminate _0000  case_id (such as quiz_037)
    """
    ensure_dir(dst_raw)
    ensure_dir(dst_pre)
    rows = [(case_id, int(subtype)) for case_id, _, _, subtype in (train_cases + val_cases)]

    for root in (dst_raw, dst_pre):
        out_csv = root / "case_to_subtype.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["Names", "Subtype"])
            for case_id, subtype in rows:
                w.writerow([case_id, subtype])
    return (dst_raw / "case_to_subtype.csv", dst_pre / "case_to_subtype.csv")


def write_splits_final(dst_pre: Path, train_cases, val_cases):
    """Write splits_final.json with a single fold {train:[...], val:[...]}."""
    ensure_dir(dst_pre)
    fold = {"train": [c[0] for c in train_cases], "val": [c[0] for c in val_cases]}
    out_json = dst_pre / "splits_final.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([fold], f, indent=2)
    return out_json

# ------------------------- integerize labels -------------------------

def integerize_labels(labels_dir: Path) -> int:
    """
    Round and cast all label NIfTI volumes under labels_dir to int32.
    Returns the number of files converted.
    """
    if not labels_dir.exists():
        return 0
    nii_files = list(labels_dir.glob("*.nii.gz")) + list(labels_dir.glob("*.nii"))
    converted = 0
    for fp in nii_files:
        try:
            img = nib.load(str(fp))
            data = img.get_fdata()
            if np.issubdtype(data.dtype, np.integer):
                continue
            data_i = np.rint(data).astype(np.int32)
            new_img = nib.Nifti1Image(data_i, img.affine, img.header)
            new_img.set_data_dtype(np.int32)
            nib.save(new_img, str(fp))
            converted += 1
        except Exception as e:
            print(f"[warn] failed to convert {fp.name}: {e}")
    return converted

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert to nnUNetv2 format AND integerize labels (one shot).")
    ap.add_argument("--src", type=Path, required=True, help="Path to the 'data' folder (train/validation/test).")
    ap.add_argument("--dataset-id", type=int, default=701, help="Dataset ID, e.g., 701.")
    ap.add_argument("--dataset-name", type=str, default="PancreasQuiz", help="Dataset name.")
    ap.add_argument("--no-int", action="store_true", help="Do NOT integerize labels (default is integerize).")
    args = ap.parse_args()

    # envs
    nnunet_raw = first_env("NNUNET_RAW", "nnUNet_raw")
    nnunet_pre = first_env("NNUNET_PREPROCESSED", "nnUNet_preprocessed")
    nnunet_res = os.environ.get("NNUNET_RESULTS") or os.environ.get("nnUNet_results")

    ds_name = f"Dataset{args.dataset_id}_{args.dataset_name}"
    dst_raw = nnunet_raw / ds_name
    dst_pre = nnunet_pre / ds_name
    imagesTr = dst_raw / "imagesTr"
    labelsTr = dst_raw / "labelsTr"
    imagesTs = dst_raw / "imagesTs"
    ensure_dir(imagesTr); ensure_dir(labelsTr); ensure_dir(imagesTs)

    print(f"[env] NNUNET_RAW           = {nnunet_raw}")
    print(f"[env] NNUNET_PREPROCESSED  = {nnunet_pre}")
    print(f"[env] NNUNET_RESULTS       = {nnunet_res}")
    print(f"[dst] RAW dataset folder   = {dst_raw}")
    print(f"[dst] PREPROC folder       = {dst_pre}")

    # gather
    train_cases = gather_cases(args.src, "train")
    val_cases   = gather_cases(args.src, "validation")
    test_cases  = gather_test_images(args.src)
    if not train_cases:
        raise RuntimeError("No training cases found under 'train/'. Check your input layout.")
    if not val_cases:
        print("[warn] No validation cases found; proceeding with train-only split.")

    # copy train+val
    for case_id, img, lbl, subtype in train_cases + val_cases:
        copy2(img,  imagesTr / f"{case_id}_0000.nii.gz")
        copy2(lbl,  labelsTr / f"{case_id}.nii.gz")

    # copy test
    for case_id, img in test_cases:
        copy2(img, imagesTs / f"{case_id}_0000.nii.gz")

    # dataset.json
    write_dataset_json(dst_raw, args.dataset_id, args.dataset_name, num_training=len(train_cases) + len(val_cases))

    # case_to_subtype.csv + splits_final.json
    case_csv_raw, case_csv_pre = write_case_to_subtype(dst_raw, dst_pre, train_cases, val_cases)
    splits_json = write_splits_final(dst_pre, train_cases, val_cases)


    # integerize labels
    converted = 0
    if not args.no_int:
        print("\n[integerize] Converting label volumes in labelsTr to int32 ...")
        converted = integerize_labels(labelsTr)
        print(f"[integerize] Converted {converted} file(s).")

    # summary
    print("\n=== Summary ===")
    print(f"Train cases:      {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")
    print(f"Test images:      {len(test_cases)}")
    print(f"labelsTr integerized: {converted} file(s)")
    print(f"Wrote -> {dst_raw / 'dataset.json'}")
    print(f"Wrote -> {dst_raw / 'case_to_subtype.csv'}")
    print(f"Wrote -> {dst_pre / 'case_to_subtype.csv'}")
    print(f"Wrote -> {dst_pre / 'splits_final.json'}")
    print("\nNext steps (example):")
    print(f"  nnUNetv2_extract_dataset_fingerprint -d {args.dataset_id} --verify_dataset_integrity")
    print(f"  nnUNetv2_plan_experiment -d {args.dataset_id} -c 2d -pl nnUNetPlannerResEncM")
    print(f"  nnUNetv2_preprocess -d {args.dataset_id} -c 2d")
    print(f"  $plans_id = (Get-Content \"$env:nnUNet_preprocessed\\{ds_name}\\plans.json\" | ConvertFrom-Json).plans_identifier")
    print(f"  nnUNetv2_train {args.dataset_id} 2d 0 -tr nnUNetTrainer -p $plans_id")

if __name__ == "__main__":
    main()
