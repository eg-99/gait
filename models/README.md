# Gait Recognition Models

## How to Run

### GEI-CNN

```bash
cd gait
source .venv/bin/activate
cd models/gei_cnn
python train.py --epochs 50 --batch_size 32 --lr 0.001
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--num_workers`: Data loading workers (default: 4)

### Autoencoder

```bash
cd gait
source .venv/bin/activate
cd models/autoencoder
python train.py --epochs 50 --batch_size 16 --lr 0.001 --num_workers 2 --save_every 5
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--num_workers`: Data loading workers (default: 4)
- `--save_every`: Save checkpoint every N epochs (default: 5)

**Extract Embeddings:**
```bash
cd models/autoencoder
python extract_embeddings.py --checkpoint checkpoints/best_model.pth --batch_size 64 --num_workers 0
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint (required)
- `--batch_size`: Batch size for extraction (default: 64)
- `--num_workers`: Data loading workers (default: 4, use 0 on macOS)

### VAE

```bash
cd gait
source .venv/bin/activate
cd models/vae
python train.py --epochs 50 --batch_size 16 --lr 0.001 --beta 1.0 --num_workers 2 --save_every 5
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--beta`: KL divergence weight (default: 1.0)
- `--num_workers`: Data loading workers (default: 4)
- `--save_every`: Save checkpoint every N epochs (default: 5)

**Extract Embeddings:**
```bash
cd models/vae
python extract_embeddings.py --checkpoint checkpoints/best_model.pth --batch_size 64 --num_workers 0
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint (required)
- `--batch_size`: Batch size for extraction (default: 64)
- `--num_workers`: Data loading workers (default: 4, use 0 on macOS)

## How to Use Saved Models

### Loading a Checkpoint

**Autoencoder:**
```python
import torch
from models.autoencoder.model import create_autoencoder

# Load checkpoint
checkpoint = torch.load('models/autoencoder/checkpoints/best_model.pth')
model = create_autoencoder(embedding_dim=128)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for encoding
with torch.no_grad():
    gei = torch.randn(1, 1, 128, 64)  # Example GEI
    embedding = model.encoder(gei)  # Shape: (1, 128)
```

**VAE:**
```python
import torch
from models.vae.model import create_vae

# Load checkpoint
checkpoint = torch.load('models/vae/checkpoints/best_model.pth')
model = create_vae(latent_dim=128, beta=1.0)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for encoding
with torch.no_grad():
    gei = torch.randn(1, 1, 128, 64)  # Example GEI
    mu, log_var = model.encode(gei)  # mu: (1, 128), log_var: (1, 128)
```

**GEI-CNN:**
```python
import torch
from models.gei_cnn.model import create_gei_cnn

# Load checkpoint
checkpoint = torch.load('models/checkpoints/gei_cnn/gei_cnn_best.pth')
model = create_gei_cnn(num_classes=124)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for classification
with torch.no_grad():
    gei = torch.randn(1, 1, 128, 64)  # Example GEI
    output = model(gei)  # Shape: (1, 124) - class probabilities
```

### Loading Embedding Files

**Load embeddings:**
```python
import pickle
import numpy as np

# Load embeddings
with open('models/autoencoder/embeddings/test_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

embeddings = data['embeddings']  # numpy array (N, 128)
labels = data['labels']          # numpy array (N,) - subject IDs
file_paths = data['file_paths']  # list of file paths
```

**Find embeddings for a specific subject:**
```python
subject_id = 5
subject_mask = (labels == subject_id)
subject_embeddings = embeddings[subject_mask]
subject_files = [file_paths[i] for i in np.where(subject_mask)[0]]

print(f"Subject {subject_id} has {len(subject_embeddings)} samples")
```

**Compute similarity between embeddings:**
```python
from sklearn.metrics.pairwise import cosine_similarity

# Get two embeddings
emb1 = embeddings[0:1]  # Shape: (1, 128)
emb2 = embeddings[1:2]  # Shape: (1, 128)

# Compute cosine similarity
similarity = cosine_similarity(emb1, emb2)[0, 0]
print(f"Similarity: {similarity:.4f}")  # Range: [-1, 1]
```

**Compare all samples of two subjects:**
```python
# Get embeddings for two subjects
subject_a_embs = embeddings[labels == 10]
subject_b_embs = embeddings[labels == 20]

# Compute all pairwise similarities
similarities = cosine_similarity(subject_a_embs, subject_b_embs)

# Average similarity between the two subjects
avg_similarity = similarities.mean()
print(f"Average similarity between subject 10 and 20: {avg_similarity:.4f}")
```

