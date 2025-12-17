# Methodology: Full-GEI Analysis

This document explains the exact methodology used for the "Full GEI" experiments, addressing the discrepancy between training on "All GEIs" (partials) and analyzing "Full GEIs."

## 1. Data Selection
*   **Source:** We used the data located in `Pathology_dataset/`.
*   **Filtering:** The script enforced a strict filter to load **ONLY** images ending in `*full.jpg`.
*   **Impact:** This reduced the dataset size to approximately 400 clean, complete gait cycle silhouettes, discarding thousands of partial/noisy fragments used in the original `experiments_final` training.

## 2. Model Architecture
*   **VAE (Variational Autoencoder):**
    *   Encoder: 4 Convolutional layers + **BatchNorm** + ReLU.
    *   Latent Space: 128 dimensions.
    *   Decoder: 4 Transposed Convolutional layers + **BatchNorm** + ReLU.
*   **Contrastive VAE:**
    *   Uses the VAE Encoder as a backbone.
    *   Adds a projection head (Linear -> ReLU -> Dropout -> Linear) for contrastive loss optimization.

> **Crucial Detail:** We ensured the model architecture exactly matched the saved checkpoints (specifically the inclusion of `BatchNorm2d` layers) to allow for valid transfer learning and evaluation.

## 3. Experimental Setup

### Run 1: Experiment 3 (From Scratch)
*   **Initialization:** Random weights.
*   **Training:** Trained for 30 epochs on the `*full.jpg` dataset.
*   **Goal:** Establish a baseline for performance when learning *only* from the target domain without prior knowledge.

### Run 2: Experiment 2 (Fine-Tuning)
*   **Initialization:** Loaded pre-trained weights from `experiments_final/checkpoints/exp1_*_casia.pth`.
*   **Training:** Fine-tuned for 30 epochs on the `*full.jpg` dataset with a lower learning rate (`1e-5`).
*   **Goal:** Assess if starting from a "healthy gait" model helps when data is limited to only full cycles.

### Run 3: Experiment 1 (Zero-Shot)
*   **Initialization:** Loaded pre-trained weights from `experiments_final/checkpoints/exp1_*_casia.pth`.
*   **Training:** **None.** The model was switched to evaluation mode immediately.
*   **Goal:** Evaluate how well the "healthy" model features cluster pathological data without any adaptation.

## 4. Evaluation Metrics
*   **Reconstruction MSE:** Mean Squared Error between the input GEI and the VAE's reconstruction. Measures how well the model "understands" the shape.
*   **Classification Accuracy:** We extracted embeddings (latent mean $\mu$) for all images, trained a KNN classifier (k=5) on 80% of them, and tested on the remaining 20%. This measures how well the latent space separates the different pathology conditions.
