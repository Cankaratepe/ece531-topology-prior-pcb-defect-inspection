# ece531-topology-prior-pcb-defect-inspection
ECE531 Computer Vision term project: topology-prior guided two-stage transistor defect detection and segmentation.

# Topology-Prior Guided PCB Transistor Defect Inspection

This repository contains the MATLAB implementation for my ECE531 Computer Vision term project.

The project investigates a two-stage computer vision pipeline for transistor defect inspection using the MVTec AD transistor dataset.

## Method Overview

The pipeline has two stages:

1. **Stage 1: Image-Level Go/No-Go Detection**
   - ResNet-18 binary classifier
   - Classes: Healthy and Faulty
   - 22 synthetic defect images were used only for image-level classifier augmentation

2. **Stage 2: Pixel-Level Defect Segmentation**
   - RGB U-Net baseline
   - Fixed-Prior U-Net using a topology prior channel
   - Learnable Topology U-Net ablation

The best Stage 2 model was the Fixed-Prior U-Net.

## Final Results

### Stage 1: Synthetic-Augmented ResNet-18

| Metric | Value |
|---|---:|
| Accuracy | 0.9559 |
| Precision | 0.8000 |
| Recall | 1.0000 |
| F1-score | 0.8889 |
| AUROC | 0.9911 |

### Stage 2: Real MVTec Images Only

| Model | F1-score | Defect IoU | Mean IoU | Pixel AUROC |
|---|---:|---:|---:|---:|
| RGB U-Net | 0.4026 | 0.2520 | 0.5975 | 0.8869 |
| Fixed-Prior U-Net | 0.7635 | 0.6175 | 0.8000 | 0.9771 |
| Learnable Topology U-Net | 0.5169 | 0.3485 | 0.6531 | 0.8120 |

## Dataset

This project uses the transistor category of the MVTec AD dataset.

Download the dataset from the official MVTec AD website and place the transistor folder as:

```text
project_root/
└── transistor/
    ├── train/
    ├── test/
    └── ground_truth/
```
The dataset is not included in this repository.

## Synthetic Images

Synthetic images were used only for Stage 1 classification augmentation because pixel-level masks were not available.

Expected optional folder structure:

```text
transistor/
└── synthetic/
    └── classification_train/
        ├── bent_lead/
        ├── cut_lead/
        ├── damaged_case/
        └── misplaced/
```

# How to Run
1. Open MATLAB.
2. Set the project folder as the current working directory.
3. Make sure the transistor/ dataset folder is present.
4. Run:
```text
main_train_all
main_evaluate_all
generate_defect_type_figures_selected
generate_clean_heatmap_figure
```

# Hardware

Experiments were run in MATLAB using an NVIDIA Quadro P2000 GPU.

Approximate training times:
| Model                    |        Time |
| ------------------------ | ----------: |
| Stage 1 ResNet-18        |  1 min 45 s |
| RGB U-Net                | 13 min 30 s |
| Fixed-Prior U-Net        | 14 min 14 s |
| Learnable Topology U-Net | 13 min 45 s |

# Notes

This repository is provided for academic use as part of the ECE531 Computer Vision final term project.
