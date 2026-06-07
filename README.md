# Person Search on PRW - SeqNeXt/GFN

**Student ID:** 0001169541  
**Full Name:** Julian Pajo  
**Email:** julian.pajo@studio.unibo.it

---

## Project Overview

This project implements and evaluates a Person Search pipeline based on the SeqNeXt/GFN architecture (Jaffe & Zakhor, CVPR 2022) on the PRW dataset. The system jointly performs pedestrian detection and person re-identification in a two-stage Faster R-CNN style pipeline with a ConvNeXt-Base backbone.

Two ablation studies are included:
- **Ablation 1:** OIM loss replaced with ArcFace loss
- **Ablation 2:** OIM loss without Circular Queue (`use_cq=False`)

---

## How to Run

This notebook is designed to run on **Kaggle** with T4 x2 GPU.

### Step 1 — Add the required datasets on Kaggle

Add the following datasets as input to the notebook:

- **PRW dataset:** `edoardomerli/prw-person-re-identification-in-the-wild`
- **Checkpoints and cached features:** `julianpajo/checkpoints`  
  Available at: https://www.kaggle.com/datasets/julianpajo/checkpoints

The checkpoints dataset contains:
- `prw_final_convnext-base_e30.pt` — official PRW checkpoint
- `best_val_loss_seqnext_gfn_base.pth` — BASE_FT fine-tuned checkpoint
- `best_val_loss_seqnext_gfn_arcface.pth` — ArcFace ablation checkpoint
- `best_val_loss_seqnext_gfn_nocq.pth` — OIM no-CQ ablation checkpoint
- `prw_seqnext_cached_features_base.npz` — pre-extracted BASE_FT features
- `prw_seqnext_cached_features_arcface.npz` — pre-extracted ArcFace features
- `prw_seqnext_cached_features_nocq.npz` — pre-extracted OIM no-CQ features
- `prw_eval_datasets.pkl` — cached PRW evaluation datasets
- `prw_seqnext_cached_datasets.pkl` — cached PRW training datasets
- `results_prw.pkl` — cached evaluation results
- `wandb_histories.pkl` — cached W&B training curves

### Step 2 — Open the notebook on Kaggle

Upload `main.ipynb` to Kaggle and attach the datasets above as input.

### Step 3 — Run the notebook

The notebook is configured to run in **inference mode** by default:
- All `do_train` flags are set to `False`
- All `do_eval` flags are set to `False`
- Model weights and cached features are loaded automatically from the checkpoints dataset

Simply run all cells from top to bottom. No training or feature extraction will be triggered.

---

## Project Structure

```
main.ipynb                  # Main notebook
README.md                   # This file
```

All model weights, cached features, and evaluation results are hosted on the Kaggle dataset linked above.

---

## Hardware Requirements

- GPU: NVIDIA T4 x2 (Kaggle)
- RAM: 13 GB
- Disk: ~20 GB (PRW dataset + checkpoints)
