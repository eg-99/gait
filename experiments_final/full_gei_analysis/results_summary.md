# Full GEI Analysis Results

This document summarizes the results of training and evaluating models **strictly on *full.jpg* GEI images** (ignoring partial gait cycles).

## Master Comparison Table

| Experiment | Model | Reconstruction MSE | Classification Accuracy |
| :--- | :--- | :--- | :--- |
| **EXP3 (Scratch)** | **Contrastive VAE** | **111.75** | **95.74%** |
| **EXP3 (Scratch)** | VAE | 92.46 | 90.43% |
| **EXP2 (Finetune)** | Contrastive VAE | 86.85 | 88.30% |
| **EXP2 (Finetune)** | VAE | 146.90 | 87.23% |
| **EXP1 (Zero-Shot)** | Contrastive VAE | 664.33 | 94.68% |
| **EXP1 (Zero-Shot)** | VAE | 382.62 | 94.68% |

> **Note:** Lower MSE is better. Higher Accuracy is better.

---

## Detailed Experiment Results

### Experiment 3: From Scratch
*   **Method:** Models initialized with random weights and trained *only* on the pathology dataset (Full GEIs).
*   **Outcome:** The **Contrastive VAE** achieved the highest overall accuracy (**95.74%**), confirming that domain-specific training on clean data yields the best classification performance.

### Experiment 2: Fine-Tuning
*   **Method:** Models initialized with pre-trained CASIA-B weights (from `experiments_final/checkpoints`) and fine-tuned on the pathology dataset.
*   **Outcome:** Fine-tuning achieved the best reconstruction error (**86.85** for Contrastive) but slightly lower accuracy than training from scratch. This suggests the pre-trained weights help with image generation but might bias the classification features slightly away from the specific pathology classes.

### Experiment 1: Zero-Shot Transfer
*   **Method:** Models pre-trained on CASIA-B (healthy gaits) and tested *directly* on pathology data without any update.
*   **Outcome:** Surprisingly high accuracy (**94.68%**) demonstrating excellent feature generalization. However, reconstruction error was very high (382+), indicating the models could not accurately reproduce the pathological gait silhouettes they hadn't seen before.
