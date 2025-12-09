# Pathological Gait Recognition - Experimental Analysis

## What We Did

Trained and compared 3 approaches for classifying pathological gaits (5 conditions: diplegic, hemiplegic, neuropathic, normal, parkinson):

**Experiment 1 - Zero-Shot Transfer**
- Trained VAE + Contrastive VAE on CASIA-B (healthy gaits only)
- Tested directly on pathology data without fine-tuning

**Experiment 2 - Fine-Tuned**
- Started with CASIA-B models
- Fine-tuned on 250 pathology images (50 per condition)

**Experiment 3 - From Scratch**
- Trained entirely on pathology dataset
- No transfer learning

Each experiment evaluated on: reconstruction error, multi-class classification (KNN/SVM), binary anomaly detection, subject identification, t-SNE visualization.

---

## Results

| Experiment | Reconstruction MSE | Classification Accuracy | Binary Detection |
|------------|-------------------|------------------------|------------------|
| EXP1 (Zero-shot) | 320.43 | 84.5% (VAE) | **96.9%** |
| EXP2 (Fine-tuned) | 344.78 | 81.0% (VAE) | 96.1% |
| EXP3 (Scratch) | **37.53** | **89.5% (Contrastive)** | 94.2% |

**Winner: EXP3 Contrastive VAE**
- Best multi-class accuracy: 89.5%
- Best reconstruction: 37.53 MSE (10x better than others)
- Best F1-score: 0.898

**Surprise: EXP1 VAE wins binary detection (96.9%)**
- Zero-shot transfer works extremely well for normal vs abnormal classification
- CASIA-B training provides strong healthy baseline

---

## What Happened

**Why EXP3 wins everything except binary:**
Domain-specific training optimizes for pathology features. Models learn actual gait abnormalities, not just "different from healthy."

**Why EXP1 wins binary detection:**
CASIA-B models learn perfect "normal" representation. Anything different = pathological. Binary task benefits from this clean separation.

**Why reconstruction ≠ classification:**
EXP1 has terrible reconstruction (320 MSE) but decent classification (84.5%). Models extract discriminative features despite domain shift. Latent space captures classification-relevant info even when decoder fails.

**Key insight:** Transfer learning works (84.5% zero-shot) but domain-specific training dominates (89.5%).

---

## How to Run

### Prerequisites

**Required checkpoints** (must exist in `checkpoints/`):
```
checkpoints/
├── exp1_vae_casia.pth              # CASIA-B trained
├── exp1_contrastive_casia.pth
├── exp2_vae_finetuned.pth          # Fine-tuned from CASIA-B
├── exp2_contrastive_finetuned.pth
├── exp3_vae_pathology.pth          # Trained from scratch
└── exp3_contrastive_pathology.pth
```

**Required data** (must exist in `data/`):
```
data/pathology_data_for_training/
├── diplegic/
├── hemiplegic/
├── neuropathic/
├── normal/
└── parkinson/
```
Total: 1,288 images, 53 subjects

**Required files**:
- `model.py` - VAE architecture
- `contrastive_model.py` - Contrastive VAE architecture
- `pathology_dataset.py` - Dataset loader

### Installation

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm pillow
```

### Run Analysis

Open and run `notebooks/Complete_Experimental_Analysis.ipynb` in Jupyter/VSCode.

All cells execute sequentially. Takes ~5-10 minutes on CPU.

**Outputs:**
- `experiment_summary.csv` - All metrics
- 6 visualizations (embedded in notebook)
- Complete analysis for all 3 experiments

---

## Training New Models

If you want to retrain:

**EXP1 (CASIA-B):**
```bash
python train_exp1_contrastive.py
```
Requires: `preprocessing/preprocessed_data/` (13,592 CASIA-B GEI samples)

**EXP2 (Fine-tune):**
```bash
python train_exp2_finetune.py
```
Requires: EXP1 checkpoints + pathology data

---

## File Structure

```
experiments_final/
├── notebooks/
│   └── Complete_Experimental_Analysis.ipynb  # Main analysis
├── checkpoints/                               # 6 trained models
├── data/pathology_data_for_training/         # 1,288 pathology images
├── model.py                                   # VAE
├── contrastive_model.py                       # Contrastive VAE
├── pathology_dataset.py                       # Data loader
├── train_exp1_contrastive.py                  # Training scripts
├── train_exp2_finetune.py
└── experiment_summary.csv                     # Results
```

