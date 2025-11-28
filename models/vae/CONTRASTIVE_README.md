# Contrastive Learning for GEI Embeddings

This directory contains a contrastive learning implementation for training robust GEI (Gait Energy Image) embeddings using a VAE backbone.

## Overview

The contrastive learning approach combines:
- **VAE Reconstruction Loss**: Learns to reconstruct GEI images from latent embeddings
- **Contrastive Loss**: Pulls embeddings of the same person together and pushes different people apart

This combination results in more robust embeddings that are better suited for gait recognition tasks.

## Architecture

The model consists of:
1. **VAE Encoder**: Extracts latent representations (μ, log_var) from GEI images
2. **VAE Decoder**: Reconstructs GEI images from latent codes
3. **Projection Head**: Maps latent embeddings to a space optimized for contrastive learning

## Files

- `contrastive_model.py`: Model architecture combining VAE with contrastive learning
- `contrastive_loss.py`: InfoNCE and supervised contrastive loss functions
- `augmentations.py`: Data augmentation strategies for creating positive pairs
- `train_contrastive.py`: Training script with combined VAE + contrastive loss
- `extract_embeddings_contrastive.py`: Script to extract embeddings from trained model

## Usage

### Training

Train a contrastive VAE model:

```bash
python train_contrastive.py \
    --data_root /path/to/preprocessed/data \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --beta 1.0 \
    --contrastive_weight 0.5 \
    --temperature 0.07 \
    --use_augmentation \
    --contrastive_loss_type info_nce
```

**Key Parameters:**
- `--contrastive_weight`: Weight for contrastive loss (0.0 to 1.0). Higher values emphasize contrastive learning.
- `--temperature`: Temperature parameter for contrastive loss (typically 0.07)
- `--use_augmentation`: Enable data augmentation for creating positive pairs
- `--contrastive_loss_type`: Choose 'info_nce' or 'supervised'

### Extracting Embeddings

Extract embeddings from a trained model:

```bash
python extract_embeddings_contrastive.py \
    --checkpoint checkpoints_contrastive/best_model.pth \
    --data_root /path/to/preprocessed/data \
    --latent_dim 128 \
    --projection_dim 128 \
    --output_dir embeddings_contrastive
```

Use `--use_projection` flag to extract projection head embeddings instead of raw latent means.

## Loss Function

The total loss is a combination of:

```
Total Loss = VAE Loss + λ * Contrastive Loss
```

Where:
- **VAE Loss** = Reconstruction Loss + β * KL Divergence
- **Contrastive Loss** = InfoNCE/Supervised Contrastive Loss

The contrastive loss encourages:
- **Positive pairs** (same person, different views/augmentations) to have similar embeddings
- **Negative pairs** (different people) to have dissimilar embeddings

## Data Augmentation

The augmentation module provides several strategies for creating positive pairs:
- Gaussian noise
- Brightness adjustment
- Contrast adjustment
- Rotation
- Translation
- Horizontal flip
- Combined augmentations

## Benefits of Contrastive Learning

1. **Better Embeddings**: Embeddings are more discriminative between different people
2. **Robustness**: More robust to variations in viewing angle, clothing, etc.
3. **Transfer Learning**: Learned embeddings can be used for downstream tasks
4. **Fewer Labels**: Contrastive learning can work with fewer labeled examples

## Comparison with Standard VAE

| Aspect | Standard VAE | Contrastive VAE |
|--------|--------------|-----------------|
| Loss | Reconstruction + KL | Reconstruction + KL + Contrastive |
| Embeddings | Learned for reconstruction | Learned for both reconstruction and discrimination |
| Discriminative Power | Lower | Higher |
| Training Time | Faster | Slightly slower (due to augmentation) |

## Tips

1. **Start with lower contrastive_weight** (0.1-0.3) and gradually increase
2. **Use augmentation** for better positive pair diversity
3. **Monitor both losses** - VAE loss should decrease, contrastive loss should also decrease
4. **Temperature tuning**: Lower temperature (0.05-0.1) for harder negatives, higher (0.1-0.2) for softer
5. **Batch size**: Larger batches provide more negative examples for contrastive learning

## Example Training Output

```
Epoch 1/50
  Train - Total: 2.3456, VAE: 1.8234, Contrastive: 1.0444, Recon: 1.5123, KL: 0.3111
  Val   - Total: 2.1234, VAE: 1.7123, Contrastive: 0.8222, Recon: 1.4123, KL: 0.3000
```

## Citation

If you use this contrastive learning implementation, please cite:
- SimCLR: A Simple Framework for Contrastive Learning of Visual Representations (Chen et al., 2020)
- Supervised Contrastive Learning (Khosla et al., 2020)

