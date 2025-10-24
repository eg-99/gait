# Gait Recognition Models

This directory contains the GEI-CNN model implementation for gait-based person recognition using the CASIA-B dataset.

---

## 📁 Directory Structure

```
models/
├── gei_cnn/
│   ├── model.py                          # GEI-CNN 
│   ├── train.py                          # Training script
│   └── gei_cnn_difference_detector.py    # Same/different 
├── utils/
│   ├── metrics.py                        # Evaluation 
│   └── training.py                       # Training 
```

---

## 🚀 Quick Start

### Prerequisites

1. **Preprocessed data** must be ready in `preprocessing/preprocessed_data/`
2. **Virtual environment** activated with all dependencies installed

```bash
cd gait
source .venv/bin/activate
```

---

## 📚 Model Components

### 1. `model.py` - GEI-CNN Architecture

**What it does:**
- Defines the CNN architecture for Gait Energy Image (GEI) classification
- Takes a single GEI image (128×64 pixels) as input
- Outputs classification probabilities for 124 subjects

**Architecture:**
```
Input: GEI (1 × 128 × 64)
  ↓
4× Conv Blocks (32 → 64 → 128 → 256 channels)
  Each block: Conv2D → BatchNorm → ReLU → MaxPool
  ↓
Flatten (256 × 8 × 4 = 8192 features)
  ↓
Fully Connected Layers:
  - FC1: 8192 → 512 (with BatchNorm + Dropout)
  - FC2: 512 → 256 (with BatchNorm + Dropout)
  - FC3: 256 → 124 (output classes)
  ↓
Output: Class probabilities for 124 subjects
```

**Key Features:**
- `forward()`: Standard classification forward pass
- `extract_features()`: Extract 256-dimensional feature embeddings (useful for similarity comparisons)
- `create_gei_cnn()`: Factory function with proper weight initialization

**Usage:**
```python
from gei_cnn.model import create_gei_cnn

# Create model
model = create_gei_cnn(num_classes=124, dropout=0.4)

# For classification
output = model(gei_image)  # Shape: (batch, 124)

# For feature extraction
features = model.extract_features(gei_image)  # Shape: (batch, 256)
```

---

### 2. `train.py` - Model Training Script

**What it does:**
- Trains the GEI-CNN model on preprocessed CASIA-B data
- Performs training, validation, and final test evaluation
- Saves checkpoints, logs, and visualizations

**Training Process:**
1. **Data Loading**: Loads GEI images from preprocessed data with sequence-based splits
   - Train: nm-01 to nm-04 (normal walking sequences)
   - Validation: nm-05, bg-01 (normal + bag carrying)
   - Test: nm-06, bg-02, cl-01, cl-02 (diverse conditions)

2. **Model Training**: 
   - Uses cross-entropy loss for classification
   - Applies data augmentation (dropout, batch normalization)
   - Implements learning rate scheduling
   - Early stopping based on validation accuracy

3. **Checkpointing**: Saves best model based on validation performance

4. **Final Evaluation**: Tests on held-out test set and generates:
   - Confusion matrix
   - Per-class metrics
   - Training history plots

**How to Run:**

```bash
cd models/gei_cnn

python train.py \
    --data_root ../../preprocessing/preprocessed_data \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --dropout 0.4 \
    --weight_decay 0.0001 \
    --scheduler step \
    --patience 20
```

**Command-Line Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | Required | Path to preprocessed data directory |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `32` | Batch size for training |
| `--lr` | `0.001` | Learning rate |
| `--dropout` | `0.5` | Dropout probability (0.4 recommended) |
| `--weight_decay` | `0.0001` | L2 regularization strength |
| `--optimizer` | `adam` | Optimizer (adam or sgd) |
| `--scheduler` | `step` | LR scheduler (step, cosine, plateau, none) |
| `--step_size` | `10` | Steps before LR decay (for step scheduler) |
| `--gamma` | `0.1` | LR decay factor (for step scheduler) |
| `--patience` | `10` | Early stopping patience (epochs without improvement) |
| `--num_workers` | `4` | DataLoader worker processes |
| `--cpu` | Flag | Force CPU usage (otherwise uses GPU/MPS if available) |

**Example Commands:**

```bash
# Basic training with default settings
python train.py --data_root ../../preprocessing/preprocessed_data

# Recommended settings (used for best results)
python train.py \
    --data_root ../../preprocessing/preprocessed_data \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --dropout 0.4 \
    --weight_decay 0.0001 \
    --scheduler step \
    --patience 20

# Fast training for testing
python train.py \
    --data_root ../../preprocessing/preprocessed_data \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001
```

**Output:**
- **Checkpoints**: Saved to `../checkpoints/gei_cnn/`
  - `gei_cnn_best.pth` - Best model based on validation accuracy
  - `gei_cnn_epoch_N.pth` - Checkpoints for each epoch
- **Logs**: Saved to `../logs/gei_cnn/`
  - `train.log` - Text log of training progress
  - `training_config.json` - Hyperparameters used
  - `training_history.png` - Loss/accuracy curves
  - `confusion_matrix.png` - Test set confusion matrix
  - `test_metrics.json` - Final test results

---

### 3. `gei_cnn_difference_detector.py` - Same/Different Person Verification

**What it does:**
- Tests the trained model's ability to identify whether two GEI images are from the same person or different people
- Uses **feature extraction** (256-dimensional embeddings) instead of direct classification
- Computes **cosine similarity** between feature vectors to determine same/different
- Evaluates performance on verification task (binary: same=1, different=0)

**Verification Process:**

1. **Feature Extraction**: 
   - Loads trained model from checkpoint
   - Extracts 256-dim features for all test GEIs using `model.extract_features()`

2. **Pair Generation**:
   - Creates pairs of GEI images:
     - **Same-person pairs**: Two different sequences from the same subject
     - **Different-person pairs**: Sequences from two different subjects

3. **Similarity Computation**:
   - Computes cosine similarity between feature vectors
   - Range: -1 to 1 (higher = more similar)

4. **Threshold Optimization**:
   - Uses ROC curve analysis to find optimal decision threshold
   - Threshold determines same vs different classification

5. **Evaluation**:
   - Computes metrics: accuracy, precision, recall, F1-score, AUC-ROC
   - Generates confusion matrix
   - Creates visualizations (ROC curve, similarity distributions)

**How to Run:**

```bash
cd models/gei_cnn

python gei_cnn_difference_detector.py
```

**No command-line arguments needed!** The script automatically:
- Uses the best checkpoint from `../checkpoints/gei_cnn/gei_cnn_best.pth`
- Loads test data from `../../preprocessing/preprocessed_data/`
- Detects GPU/MPS/CPU automatically

**What it Evaluates:**

| Metric | Meaning |
|--------|---------|
| **Overall Accuracy** | Percentage of correct same/different decisions |
| **Same Person Detection** | Recall for identifying same person (True Positive Rate) |
| **Different Person Detection** | Recall for identifying different people (True Negative Rate) |
| **AUC-ROC** | Area under ROC curve (1.0 = perfect, 0.5 = random) |
| **Optimal Threshold** | Best similarity cutoff for decision |

**Output:**
- **Console**: Detailed metrics printed to terminal
- **Plots**: `../logs/gei_cnn/difference_detection_results.png`
  - ROC curve showing true positive vs false positive rates
  - Similarity distributions for same/different pairs
- **Metrics**: `../logs/gei_cnn/difference_detection_metrics.json`
  - All evaluation metrics in JSON format

**Use Case:**
This is useful for **person verification** scenarios:
- **Access control**: "Is this the same person who registered?"
- **Re-identification**: "Have we seen this person before?"
- **Matching**: "Are these two gait samples from the same individual?"

Unlike classification (which asks "which of 124 people is this?"), verification asks "are these two samples from the same person?" - a simpler and more flexible task.

---

## 🔄 Typical Workflow

### Step 1: Train the Model
```bash
cd models/gei_cnn

python train.py \
    --data_root ../../preprocessing/preprocessed_data \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --dropout 0.4 \
    --weight_decay 0.0001 \
    --scheduler step \
    --patience 20
```

**Expected time**: 30-40 minutes on M1/M2 Mac with MPS

### Step 2: Run Difference Detection Experiment
```bash
python gei_cnn_difference_detector.py
```

**Expected time**: 2-3 minutes

### Step 3: Review Results
- Check `../logs/gei_cnn/test_metrics.json` for classification performance
- Check `../logs/gei_cnn/difference_detection_metrics.json` for verification performance
- View plots in `../logs/gei_cnn/`

---

## 📊 Understanding the Outputs

### Classification Task (train.py)
```json
{
  "test_accuracy": 0.65,      // Can the model identify which of 124 people this is?
  "test_top5_accuracy": 0.85  // Is correct person in top-5 predictions?
}
```

### Verification Task (gei_cnn_difference_detector.py)
```json
{
  "overall_accuracy": 0.74,              // Overall same/different accuracy
  "same_person_detection": 0.67,         // Correctly identifying same person
  "different_person_detection": 0.82,    // Correctly identifying different people
  "auc_roc": 0.82                        // Overall discrimination ability
}
```

---

## 🛠️ Advanced Usage

### Resume Training from Checkpoint

```bash
python train.py \
    --data_root ../../preprocessing/preprocessed_data \
    --checkpoint_dir ../checkpoints/gei_cnn \
    --epochs 100  # Continue for more epochs
```

The script will automatically load the latest checkpoint if available.

### CPU-Only Training

```bash
python train.py \
    --data_root ../../preprocessing/preprocessed_data \
    --cpu
```

### Custom Learning Rate Schedule

```bash
# Cosine annealing (smooth decay)
python train.py \
    --data_root ../../preprocessing/preprocessed_data \
    --scheduler cosine \
    --epochs 50

# Step decay (drop at intervals)
python train.py \
    --data_root ../../preprocessing/preprocessed_data \
    --scheduler step \
    --step_size 15 \
    --gamma 0.1

# Reduce on plateau (adaptive)
python train.py \
    --data_root ../../preprocessing/preprocessed_data \
    --scheduler plateau \
    --patience 10
```

---

## 📝 Notes

### Data Requirements
- **Preprocessed GEI files** must exist in `preprocessing/preprocessed_data/`
- **Sequence-based splits** (not subject-based) - all 124 subjects in train/val/test
- Split structure defined in `preprocessed_data/data_splits.json`

### Hardware Recommendations
- **GPU/MPS**: Recommended for faster training (~4-5 it/s)
- **CPU**: Works but slower (~1-2 it/s)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~2GB for preprocessed data + checkpoints

### Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'torch'`
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
```

**Issue**: `FileNotFoundError: data_splits.json not found`
```bash
# Run preprocessing first
cd preprocessing
python casia_b_loader.py --dataset_root /path/to/CASIA-B --output_root preprocessed_data --create_splits
```

**Issue**: Low accuracy (~1-5%)
- Check that you're using **sequence-based splits** (not subject-based)
- Verify `data_splits.json` has `"split_type": "sequence_based"`
- Expected accuracy: 65-75% for classification, 74-85% for verification

---

## 📖 Further Reading

- `MODELS.md` - Detailed architecture documentation and theory
- `preprocessing/QUICK_START.md` - Data preprocessing guide
- Model checkpoints and logs in `checkpoints/` and `logs/` directories
