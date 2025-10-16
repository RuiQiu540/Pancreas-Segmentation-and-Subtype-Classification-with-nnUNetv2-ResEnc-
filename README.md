# Pancreas Segmentation & Subtype Classification (nnUNetv2 . ResEnc)

> **What this repo is for**: a compact reproduction of pancreas **segmentation** (nnUNetv2 + ResEnc) plus **subtype classification (0/1/2)**. Expected outputs per case: a NIfTI mask `quiz_XXX.nii.gz` and a CSV `subtype_results.csv` with two columns **`Names,Subtype`**.

---

## Which files should I replace (if I use the provided tweaks)?
> You can run nnUNet as-is. If you want our minimal tweaks/quality-of-life improvements, replace the following files in your local nnUNet install:
- `patches/nnUNetTrainer.py` -> overwrite **`nnunetv2/training/nnUNetTrainer.py`**
- `patches/predict_from_raw_data.py` -> overwrite **`nnunetv2/inference/predict_from_raw_data.py`**

> Tip: install nnUNet in *editable* mode (`pip install -e .`), then copy these two files over the original paths. If you prefer not to overwrite, you can keep them in your project and explicitly point to your custom Trainer/inference script in your commands.

---

## Key scripts
- `convert_to_nnunet.py` - one-shot conversion of your `data/` (train/validation/test/subtype*) into nnUNetv2 RAW layout. It **integerizes labels** and writes `dataset.json`, `case_to_subtype.csv`, and `splits_final.json`.

## Short citation / acknowledgements
- **nnU-Net (core method)**: Isensee *et al.*, *Nature Methods* 2021 - see `CITATION.cff`.  
- **Data source**: Data courtesy of **Dr. Jun Ma** (University of Toronto).  
- **Reference implementations (FLARE23 winners)**:  
  - Ziyan-Huang/**FLARE23** (nnUNetv2 inference acceleration / pseudo-labeling)  
  - youngkong/**MICCAI-FLARE23** (label fusion / Swift nnU-Net pipeline)

> For full BibTeX and detailed references, please check `CITATION.cff`.

---

## Minimal repository layout
```
README.md (this file)  <- concise description + quick nav
Reproduce_process.md   <- full pipeline, environment, evaluation
convert_to_nnunet_merged.py
patches/
  nnUNetTrainer.py
  predict_from_raw_data.py
CITATION.cff
LICENSE
NOTICE
environment.yml
```
