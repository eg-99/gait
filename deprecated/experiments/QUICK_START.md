# Quick Start Guide: Pathology Detection Experiments

This guide walks you through running all three experiments to compare VAE vs Contrastive VAE for gait pathology detection.

---

## Prerequisites

1. **Trained models on CASIA-B** (for Experiments 1 & 2):
   ```bash
   # Train VAE
   python models/vae/train_vae.py \
     --images_root Casia-B-Images/output \
     --checkpoint_dir checkpoints/exp1_casia_only/vae
   
   # Train Contrastive VAE
   python models/vae/train_contrastive.py \
     --images_root Casia-B-Images/output \
     --checkpoint_dir checkpoints/exp1_casia_only/contrastive \
     --contrastive_weight 0.5
   ```

2. **Pathology dataset** organized as:
   ```
   pathology_data/
     normal/
       subject_001/*.png
     parkinson/
       subject_001/*.png
     neuropathy/
       subject_001/*.png
     ...
   ```

---

## Experiment 1: Zero-Shot Transfer

**Goal**: Test models trained on healthy gaits on pathology data (no retraining).

```bash
python experiments/exp1_zero_shot.py \
  --vae_checkpoint checkpoints/exp1_casia_only/vae/best_model.pth \
  --contrastive_checkpoint checkpoints/exp1_casia_only/contrastive/best_model.pth \
  --pathology_root path/to/pathology_data \
  --output_dir results/exp1_zero_shot
```

**Outputs**:
- `results/exp1_zero_shot/vae_binary.json` - VAE binary classification metrics
- `results/exp1_zero_shot/contrastive_binary.json` - Contrastive VAE binary metrics
- `results/exp1_zero_shot/vae_multiclass.json` - VAE multi-class metrics
- `results/exp1_zero_shot/contrastive_multiclass.json` - Contrastive VAE multi-class metrics
- ROC curves and confusion matrices

**Expected time**: 5-10 minutes

---

## Experiment 2: Fine-Tuning

**Goal**: Fine-tune pretrained models on small subset of pathology data.

### Step 1: Fine-tune VAE

```bash
python experiments/exp2_finetune.py \
  --pretrained_checkpoint checkpoints/exp1_casia_only/vae/best_model.pth \
  --model_type vae \
  --pathology_root path/to/pathology_data \
  --train_split 0.25 \
  --val_split 0.15 \
  --freeze_encoder \
  --epochs 50 \
  --lr 1e-4 \
  --output_dir checkpoints/exp2_finetune
```

### Step 2: Fine-tune Contrastive VAE

```bash
python experiments/exp2_finetune.py \
  --pretrained_checkpoint checkpoints/exp1_casia_only/contrastive/best_model.pth \
  --model_type contrastive \
  --pathology_root path/to/pathology_data \
  --train_split 0.25 \
  --val_split 0.15 \
  --epochs 50 \
  --lr 1e-4 \
  --output_dir checkpoints/exp2_finetune
```

**Options**:
- `--freeze_encoder`: Only train classifier head (faster, less overfitting)
- Without `--freeze_encoder`: Full fine-tuning (may be better with more data)

### Step 3: Evaluate Fine-Tuned Models

```bash
# Evaluate VAE
python experiments/exp2_evaluate.py \
  --checkpoint checkpoints/exp2_finetune/vae/best_model.pth \
  --model_type vae \
  --pathology_root path/to/pathology_data \
  --split_file checkpoints/exp2_finetune/vae/data_splits.json \
  --output_dir results/exp2_finetune

# Evaluate Contrastive VAE
python experiments/exp2_evaluate.py \
  --checkpoint checkpoints/exp2_finetune/contrastive/best_model.pth \
  --model_type contrastive \
  --pathology_root path/to/pathology_data \
  --split_file checkpoints/exp2_finetune/contrastive/data_splits.json \
  --output_dir results/exp2_finetune
```

**Expected time**: 
- Training: 1-2 hours per model (depends on dataset size)
- Evaluation: 5 minutes

---

## Experiment 3: Train From Scratch

**Goal**: Train entirely on pathology data (no CASIA-B pretraining).

### Step 1: Train VAE from Scratch

```bash
python models/vae/train_vae.py \
  --images_root path/to/pathology_data \
  --checkpoint_dir checkpoints/exp3_from_scratch/vae \
  --epochs 100
```

### Step 2: Train Contrastive VAE from Scratch

```bash
python models/vae/train_contrastive.py \
  --images_root path/to/pathology_data \
  --checkpoint_dir checkpoints/exp3_from_scratch/contrastive \
  --contrastive_weight 0.5 \
  --epochs 100
```

### Step 3: Evaluate

```bash
# Evaluate VAE
python experiments/exp2_evaluate.py \
  --checkpoint checkpoints/exp3_from_scratch/vae/best_model.pth \
  --model_type vae \
  --pathology_root path/to/pathology_data \
  --split_file checkpoints/exp3_from_scratch/vae/data_splits.json \
  --output_dir results/exp3_from_scratch

# Evaluate Contrastive VAE
python experiments/exp2_evaluate.py \
  --checkpoint checkpoints/exp3_from_scratch/contrastive/best_model.pth \
  --model_type contrastive \
  --pathology_root path/to/pathology_data \
  --split_file checkpoints/exp3_from_scratch/contrastive/data_splits.json \
  --output_dir results/exp3_from_scratch
```

**Note**: You'll need to create data splits manually or modify training scripts to save splits.

**Expected time**: 2-4 hours per model

---

## Compare All Results

After running all experiments, generate comprehensive comparison:

```bash
python experiments/compare_results.py \
  --results_dir results \
  --output_dir results/comparison
```

**Outputs**:
- `results/comparison/summary_report.txt` - Text summary of all experiments
- `results/comparison/binary_comparison.csv` - Binary classification metrics table
- `results/comparison/multiclass_comparison.csv` - Multi-class metrics table
- Bar charts comparing metrics across experiments
- Heatmaps showing performance patterns

**Expected time**: 1 minute

---

## Quick Example: Minimal Test

If you want to quickly test the pipeline with minimal data:

```bash
# 1. Experiment 1 only (fastest)
python experiments/exp1_zero_shot.py \
  --vae_checkpoint checkpoints/exp1_casia_only/vae/best_model.pth \
  --contrastive_checkpoint checkpoints/exp1_casia_only/contrastive/best_model.pth \
  --pathology_root path/to/pathology_data \
  --output_dir results/exp1_zero_shot

# 2. View results
cat results/exp1_zero_shot/vae_binary.json
cat results/exp1_zero_shot/contrastive_binary.json
```

---

## Understanding Results

### Binary Classification Metrics

- **Accuracy**: Overall correctness (% correct predictions)
- **Precision**: Of predicted pathological, how many are correct?
- **Recall**: Of actual pathological, how many detected?
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve (0.5 = random, 1.0 = perfect)

**Clinical interpretation**:
- High **recall** = Don't miss pathological cases (important for screening)
- High **precision** = Few false alarms (reduces unnecessary follow-ups)
- High **AUC-ROC** = Good overall discrimination ability

### Multi-class Metrics

- **Top-1 Accuracy**: Exact match (predicted class = true class)
- **Top-2 Accuracy**: True class in top 2 predictions
- **Macro F1**: Average F1 across all classes (equal weight per class)
- **Confusion Matrix**: Shows which conditions are confused

**Clinical interpretation**:
- High **Top-2 accuracy** = Useful for differential diagnosis (narrow down possibilities)
- **Confusion matrix** = Reveals which conditions are hard to distinguish

---

## Expected Findings

Based on contrastive learning theory, you should see:

1. **Contrastive VAE > Standard VAE**: 
   - Better AUC-ROC in binary classification
   - Higher Top-1 accuracy in multi-class
   - Clearer confusion matrices (less misclassification)

2. **Fine-Tuning > Zero-Shot**:
   - Significant improvement from adapting to pathology data
   - Most data-efficient approach

3. **From Scratch varies**:
   - Best IF pathology dataset is large enough (500+ per class)
   - May overfit if dataset too small
   - Shows whether healthy gait pretraining helps

---

## Troubleshooting

### "No samples found"
Check pathology dataset structure matches expected format:
```
pathology_root/
  condition_name/
    subject_id/
      *.png
```

### "Checkpoint not found"
Ensure you've trained models first:
- Experiment 1 & 2 need CASIA-B pretrained models
- Experiment 3 trains from scratch (no pretraining needed)

### CUDA out of memory
Reduce batch size:
```bash
--batch_size 16  # or even 8
```

### Poor performance
- Check image quality (GEI images should be clear silhouettes)
- Ensure enough samples per class (recommend 50+ minimum)
- Try longer training (more epochs)
- Experiment with contrastive weight (0.3 to 0.7)

---

## Next Steps

After completing experiments:

1. **Analyze confusion matrices**: Which conditions are confused? Why?
2. **t-SNE visualization**: Add code to visualize embedding spaces
3. **Cross-validation**: Run multiple random seeds for robust results
4. **Hyperparameter tuning**: Try different contrastive weights, learning rates
5. **Ensemble methods**: Combine VAE and Contrastive VAE predictions

---

## Citation

If you use this code, please cite the relevant papers:
- VAE: Kingma & Welling (2014)
- Contrastive Learning: Chen et al. (2020) - SimCLR
- Supervised Contrastive: Khosla et al. (2020)
