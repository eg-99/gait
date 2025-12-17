# Methodology: End-to-End Gait Analysis Pipeline

This document details the complete pipeline for the Pathological Gait Analysis system, spanning from raw video ingestion to final model inference.

## 1. Preprocessing: From Video to GEI
The raw input consists of video recordings of patients with various gait pathologies. We process these into **Gait Energy Images (GEIs)**, which summarize spatiotemporal gait information into a single image.

### 1.1. Pipeline Steps
1.  **Frame Extraction:** Raw videos are decomposed into individual frames.
2.  **Human Detection (YOLOv8):** Each frame is passed through a YOLOv8 detector to locate the person. Bounding boxes are generated.
3.  **Silhouette Segmentation (SAM 2):** The breakdown of the human figure is refined using the **Segment Anything Model 2 (SAM 2)**. This state-of-the-art model generates precise binary silhouettes, removing background noise.
4.  **Centering & Alignment:** Silhouettes are centered based on their center of mass to align the subject across frames.
5.  **Gait Cycle Detection (Optical Flow):** We compute the optical flow between consecutive silhouettes. A full "Gait Cycle" is identified by analyzing the periodicity of leg movement.
6.  **GEI Generation:** All binary silhouettes within a single complete gait cycle are averaged together to produce the **Gait Energy Image (GEI)**.
    *   **Result:** A grayscale image where pixel intensity represents the amount of time that spatial location was occupied by the body during the walk. Higher intensity = more static body parts (torso); Lower intensity = dynamic parts (legs/arms).

### 1.2. Data Filtering
For this final analysis, we apply a strict quality filter:
*   **Selection:** We utilize **only** `*full.jpg` images. These represent complete, valid gait cycles verified by the preprocessing pipeline. Partial or incomplete cycles are excluded to ensure feature consistency.

---

## 2. Experimental Protocol

### 2.1. Dataset Preparation
*   **Total Samples:** ~400 verified `*full.jpg` GEIs.
*   **Standardization:** Images are resized to **64x128** pixels, converted to grayscale, and normalized to range `[0, 1]`.

### 2.2. Subject-Aware Splitting (Zero Leakage)
To assess true generalization, we employ a **Subject-Aware Split**:
*   **Method:** Subjects are grouped by pathology. For each pathology, **20% of unique subjects** are randomly selected and completely held out for the Test Set.
*   **Separation:** The Training Set and Test Set share **zero subjects**. This prevents the model from "memorizing" a person's identity and forces it to learn disease features.
*   **Reproducibility:** A global random seed (`SEED=42`) is enforced across Python, NumPy, and PyTorch.

### 2.3. Data Augmentation (Contrastive Learning)
To train the Contrastive model effectively on this dataset we create two distinct "views" of every image during training:
1.  **View 1:** The original GEI.
2.  **View 2:** An augmented GEI with **RandomAffine** transformations (Rotation ±10°, Translation ±5%).
This forces the model to learn features that are invariant to small changes in viewing angle or cycle alignment.

---

## 3. Model Architectures

### 3.1. GEI_VAE (Variational Autoencoder)
A generative model designed to learn a compressed latent representation of gait.
*   **Encoder:** 4-layer Convolutional Neural Network (CNN) with `BatchNorm` and `ReLU`. Compresses input (128x64) -> Latent Vector (128d).
*   **Latent Space:** Parameters `mu` (mean) and `log_var` (variance) are learned.
*   **Decoder:** Transpose Convolutional network that reconstructs the GEI from the latent vector.
*   **Loss:** `MSE` (Reconstruction Quality) + `KL Divergence` (Regularization).

### 3.2. ContrastiveVAE
Enhances the VAE by enforcing that embeddings of the *same* image (original and augmented view) are closer together than embeddings of *different* images.
*   **Backbone:** Uses the same GEI_VAE encoder.
*   **Projection Head:** A multi-layer perceptron (Linear -> ReLU -> Dropout -> Linear) that projects latent vectors into a space optimized for contrastive loss.
*   **Loss:** `VAE Loss` + `NT-Xent Loss` (Normalized Temperature-scaled Cross Entropy).

---

## 4. Experiments

We evaluate three distinct training strategies to determine the optimal approach for Pathological Gait Analysis.

| Experiment | Name | Methodology | Goal |
| :--- | :--- | :--- | :--- |
| **EXP 1** | **Zero-Shot Transfer** | **No Training.** We use a VAE pre-trained on the massive **CASIA-B** dataset (healthy walking). We extract features for Pathology images directly. | Test feature reusability from healthy to pathological domains. |
| **EXP 2** | **Fine-Tuning** | **Transfer Learning.** We optimize the CASIA-B pre-trained weights using the Pathology Training Set (low learning rate). | Adapt generic gait features to specific pathologies. |
| **EXP 3** | **From Scratch** | **Domain Training.** We initialize the model randomly and train *only* on the Pathology Training Set (30 epochs). | Learn dataset-specific features without bias from healthy datasets. |
