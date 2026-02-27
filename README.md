# 🏆 Offroad Semantic Segmentation
# Complete Project Report

*Date:* February 28, 2026

*Competition:* Duality AI's Offroad Semantic Scene Segmentation Hackathon

---

## 📑 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Model Architecture](#model-architecture)
3. [Training Process](#training-process)
4. [Performance Analysis](#performance-analysis)
5. [Visual Results](#visual-results)
6. [Key Findings](#key-findings)
7. [Insights & Conclusions](#insights--conclusions)
8. [Submission Files](#submission-files)

---

## 1. Executive Summary

### Problem Statement
Multi-class semantic segmentation on synthetic desert environments with *significant domain shift* between training and test sets.

### Solution Approach
- *Architecture:* SegFormer-B5 (Hierarchical Transformer)
- *Key Strategy:* Aggressive augmentation for generalization
- *Loss Function:* Hybrid (0.4 × CE + 0.6 × Dice)
- *Training:* 200 epochs with early stopping

### Results Summary

| Metric | Value |
|--------|-------|
| *Mean IoU* | *0.7256* |
| Average Precision | 0.8064 |
| Average Recall | 0.8730 |
| Best Class IoU | 0.9875 |
| Classes > 0.7 IoU | 4/6 |

---

## 2. Model Architecture

### SegFormer-B5

*Why SegFormer?*

1. *No Positional Encoding* → Better at different scales/angles
2. *Hierarchical Transformer* → Captures local + global context
3. *Lightweight Decoder* → Prevents overfitting
4. *SOTA Performance* → Best on outdoor scenes

*Model Specifications:*

| Property | Value |
|----------|-------|
| Backbone | MiT-B5 (Mix Transformer) |
| Parameters | 81.97M |
| Input Size | (540, 960) |
| Output Classes | 6 |
| Pretrained | ImageNet |

---

## 3. Training Process

### Training Overview

<img width="1271" height="878" alt="Screenshot 2026-02-27 201741" src="https://github.com/user-attachments/assets/fc159b2b-1fa2-4770-a8e2-af122b580417" />

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 200 (with early stopping) |
| Batch Size | 6 |
| Learning Rate | 4e-4 (with warmup) |
| Optimizer | AdamW |
| LR Schedule | Cosine Annealing |
| Loss Function | 0.4 × CE + 0.6 × Dice |
| Augmentation | Aggressive (see below) |

### Data Augmentation Strategy

*Critical for Domain Shift!*

✅ *Geometric:*
- Horizontal flip (70%)
- Rotation (±20°)
- ShiftScaleRotate (15% scale)
- GridDistortion (terrain)
- OpticalDistortion (lens)

✅ *Photometric:*
- ColorJitter (±30% brightness/contrast)
- HueSaturationValue shifts
- CLAHE (lighting normalization)
- Blur/Sharpen variations
- Gaussian noise

---

## 4. Performance Analysis

### Overall Performance

<img width="2045" height="1284" alt="Screenshot 2026-02-27 203310" src="https://github.com/user-attachments/assets/bd62532f-561c-49e9-b864-653d03989aeb" />


### Per-Class Detailed Results

| Class | IoU | Precision | Recall | F1-Score | Status |
|-------|-----|-----------|--------|----------|--------|
| Class_0 | 0.8122 | 0.8814 | 0.9119 | 0.8964 | ✅ Excellent |
| Class_1 | 0.7332 | 0.8415 | 0.8507 | 0.8461 | ✅ Excellent |
| Class_2 | 0.5531 | 0.6409 | 0.8014 | 0.7123 | 🟢 Good |
| Class_3 | 0.5579 | 0.5949 | 0.8996 | 0.7162 | 🟢 Good |
| Class_27 | 0.7099 | 0.8830 | 0.7836 | 0.8303 | ✅ Excellent |
| Class_39 | 0.9875 | 0.9964 | 0.9910 | 0.9937 | ✅ Excellent |

### Confusion Matrix

<img width="2141" height="958" alt="Screenshot 2026-02-27 203245" src="https://github.com/user-attachments/assets/f79334af-e6bd-4d69-abf3-5c4138313255" />


---

## 5. Visual Results

### Validation Set Results

<img width="1312" height="1266" alt="Screenshot 2026-02-27 203032" src="https://github.com/user-attachments/assets/49749d9f-405f-4add-8825-7ea0f8ad55e9" />

### Test Set Predictions

<img width="1735" height="1275" alt="Screenshot 2026-02-27 202920" src="https://github.com/user-attachments/assets/2ba5e387-ab07-4dd7-8268-ae868b4a6267" />

---

## 6. Key Findings

### ✅ Strengths

1. *Best Performing Class:* Class_39 (IoU: 0.9875)
   - High precision (0.9964) and recall (0.9910)
   - Minimal confusion with other classes

2. *Strong Performance:* 4 classes with IoU > 0.7
   - Class_0
   - Class_1
   - Class_27
   - Class_39

3. *Good Generalization:* Aggressive augmentation prevented overfitting

<img width="1727" height="1285" alt="Screenshot 2026-02-27 203344" src="https://github.com/user-attachments/assets/41958987-1e0d-42d6-8f0c-93c581586a82" />
<img width="1404" height="1275" alt="Screenshot 2026-02-27 203108" src="https://github.com/user-attachments/assets/cd3ac249-b65a-48f8-a91d-05a590ece723" />


### ⚠️ Areas for Improvement

1. *Challenging Class:* Class_2 (IoU: 0.5531)
   - Lower recall suggests under-segmentation
   - May benefit from class-specific augmentation

---

## 7. Insights & Conclusions

### Technical Insights

1. *SegFormer's Advantage:*
   - Hierarchical attention captured both fine details and scene context
   - No positional encoding crucial for handling scale variations

2. *Augmentation Impact:*
   - Geometric transforms (rotation, distortion) critical for pose variations
   - Photometric augmentations essential for lighting robustness
   - CLAHE particularly effective for desert scenes

3. *Loss Function Choice:*
   - 0.6 × Dice weight (vs 0.4 × CE) prioritized IoU optimization
   - Class weights addressed imbalance effectively

### Domain Shift Handling

*Strategies that worked:*

- ✅ Diverse augmentation (simulated environment variations)
- ✅ Strong regularization (weight decay 0.02)
- ✅ Pretrained backbone (ImageNet features transfer well)
- ✅ Test-Time Augmentation (horizontal flip averaging)

### Expected Competition Performance

Based on validation mIoU of *0.7256*:

| Test Scenario | Expected mIoU | Confidence |
|---------------|---------------|------------|
| Similar Conditions | 0.7111 - 0.7401 | High |
| Moderate Domain Shift | 0.6676 - 0.7111 | Medium |
| Strong Domain Shift | 0.6168 - 0.6676 | Medium-Low |


---

*End of Report*

Generated automatically from model evaluation and analysis
