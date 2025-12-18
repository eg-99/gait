# Results Summary: Full-GEI Analysis

**Status:** Final Verified Run (5-Class Dataset + Subject-Aware Split + Augmentation).

## 1. Executive Summary
We evaluated three approaches (Zero-Shot, Fine-Tuning, Training from Scratch) using strict **Subject-Aware Splitting** on the complete 5-class dataset (Normal, Diplegic, Neuropathic, Parkinson, Hemiplegic).

**The Conclusion:** **EXP2 (Fine-Tuning)** is the most robust model for the 5-class setup, achieving high nearest-neighbor accuracy (**74.1%**) and strong binary detection. While **EXP3 (Scratch)** performed well on the simpler 3-class subset, the complex 5-class problem benefits significantly from the pre-trained features of CASIA-B.

## 2. Comprehensive Metric Comparison (5 Classes)

| Model & Strategy | **Reconstruction (MSE)** | **SVM Accuracy** | **Binary Accuracy** | **KNN Accuracy** |
| :--- | :--- | :--- | :--- | :--- |
| **EXP2 (Finetune) - Contrastive** | 76.54 | 61.6% | **91.1%** | **74.1%** (Best) |
| **EXP1 (Zero-Shot) - Contrastive** | 664.48 | \textbf{79.5%} | 83.9% | 73.2% |
| **EXP3 (Scratch) - Contrastive** | \textbf{65.96} | 70.5% | 89.3% | 71.4% |
| **EXP3 (Scratch) - VAE** | 115.56 | 69.6% | 89.3% | 70.5% |
| **EXP2 (Finetune) - VAE** | 124.27 | 65.2% | 83.9% | 68.8% |
| **EXP1 (Zero-Shot) - VAE** | 387.30 | 60.7% | 92.0% | 67.0% |

*(Note: MSE is Mean Squared Error, lower is better. All other metrics: higher is better.)*

## 3. Detailed Findings

### 3.1. Generalization & Robustness
**Winner: EXP2 Contrastive (Fine-Tuned)**
*   It achieves **74.1% KNN Accuracy** on the challenging 5-way classification.
*   The **Zero-Shot (EXP1)** model surprisingly achieves the highest **SVM Accuracy (79.5%)**, suggesting that the generic motion features from CASIA-B are linearly separable even without fine-tuning, although they are noisy (high MSE).

### 3.2. Image Reconstruction Quality
**Winner: EXP3 Contrastive (Scratch)**
*   With an **MSE of 65.96**, this model generates the sharpest reconstructions.
*   However, this generative quality does not fully translate to classification accuracy in the 5-class setting, likely because the model overfits to the visual appearance rather than the subtle gait dynamics of the rarer classes (Diplegic/Hemiplegic).

### 3.3. Binary Anomaly Detection
**Winner: EXP1 VAE (Zero-Shot) / EXP2 Contrastive**
*   **EXP1 VAE** achieves **92.0% Binary Accuracy**, confirming that anomaly detection works best when "normality" is defined by a massive external dataset (CASIA-B).
*   **EXP2 Contrastive** follows closely with **91.1%**, offering a good balance between binary screening and specific pathology classification.

## 4. Final Recommendation
For a holistic system:
*   Use **EXP1 VAE** as a first-pass **Anomaly Detector** (Is the patient healthy?).
*   Use **EXP2 Contrastive** as the **Classifier** (Which pathology is it?) for flagged patients.
