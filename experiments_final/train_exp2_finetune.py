"""
Fine-tune models for Experiment 2 (Transfer Learning)
Takes pretrained CASIA-B models and fine-tunes on pathology data
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import random

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from model import create_vae
from contrastive_model import create_contrastive_vae

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class PathologyGEIDataset(Dataset):
    """Dataset for pathology GEI images"""
    
    def __init__(self, root_dir, target_size=(128, 64), max_samples_per_condition=50):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.samples = []
        
        # Collect all images with limit per condition
        for cond_dir in sorted(d for d in self.root_dir.iterdir() if d.is_dir()):
            condition = cond_dir.name.lower()
            condition_samples = []
            
            # Get subject directories or images directly
            subject_dirs = sorted(d for d in cond_dir.iterdir() if d.is_dir())
            if not subject_dirs:
                subject_dirs = [cond_dir]
            
            for subj_dir in subject_dirs:
                subject_id = subj_dir.name
                
                # Get all image files
                for img_path in subj_dir.rglob("*.png"):
                    condition_samples.append({
                        'path': img_path,
                        'subject': subject_id,
                        'condition': condition
                    })
                for img_path in subj_dir.rglob("*.jpg"):
                    condition_samples.append({
                        'path': img_path,
                        'subject': subject_id,
                        'condition': condition
                    })
            
            # Limit samples per condition
            if max_samples_per_condition and len(condition_samples) > max_samples_per_condition:
                random.shuffle(condition_samples)
                condition_samples = condition_samples[:max_samples_per_condition]
            
            self.samples.extend(condition_samples)
        
        print(f"Loaded {len(self.samples)} pathology images (max {max_samples_per_condition} per condition)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess image
        img = Image.open(sample['path']).convert('L')
        img = img.resize((self.target_size[1], self.target_size[0]), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        
        return torch.from_numpy(arr).unsqueeze(0), sample['subject']


def contrastive_loss_fn(z_i, z_j, temperature=0.07):
    """NT-Xent contrastive loss"""
    batch_size = z_i.size(0)
    z_i = nn.functional.normalize(z_i, dim=1)
    z_j = nn.functional.normalize(z_j, dim=1)
    
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    
    labels = torch.arange(batch_size, device=z_i.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    
    similarity_matrix = similarity_matrix / temperature
    criterion = nn.CrossEntropyLoss()
    loss = criterion(similarity_matrix, labels)
    
    return loss


def train_epoch_vae(model, dataloader, optimizer, device):
    """Train VAE for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    pbar = tqdm(dataloader, desc="Training VAE")
    for images, _ in pbar:
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon, mu, logvar = model(images)
        
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon, images, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + kl_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        pbar.set_postfix({
            'loss': loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item()
        })
    
    n = len(dataloader)
    return {
        'total_loss': total_loss / n,
        'recon_loss': total_recon_loss / n,
        'kl_loss': total_kl_loss / n
    }


def train_epoch_contrastive(model, dataloader, optimizer, device, lambda_contrastive=0.5):
    """Train Contrastive VAE for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_contrastive_loss = 0
    
    pbar = tqdm(dataloader, desc="Training Contrastive VAE")
    for images, _ in pbar:
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon, mu, logvar, z_proj = model(images, return_projection=True)
        
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon, images, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # VAE loss
        vae_loss = recon_loss + kl_loss
        
        # Contrastive loss
        batch_size = images.size(0)
        if batch_size > 1:
            idx1 = torch.randperm(batch_size, device=device)[:batch_size//2]
            idx2 = torch.randperm(batch_size, device=device)[:batch_size//2]
            z1 = z_proj[idx1]
            z2 = z_proj[idx2]
            contrastive_loss = contrastive_loss_fn(z1, z2)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        loss = vae_loss + lambda_contrastive * contrastive_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_contrastive_loss += contrastive_loss.item()
        
        pbar.set_postfix({
            'loss': loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'contr': contrastive_loss.item()
        })
    
    n = len(dataloader)
    return {
        'total_loss': total_loss / n,
        'recon_loss': total_recon_loss / n,
        'kl_loss': total_kl_loss / n,
        'contrastive_loss': total_contrastive_loss / n
    }


def validate_vae(model, dataloader, device):
    """Validate VAE"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            
            recon, mu, logvar = model(images)
            recon_loss = nn.functional.mse_loss(recon, images, reduction='mean')
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    n = len(dataloader)
    return {
        'total_loss': total_loss / n,
        'recon_loss': total_recon_loss / n,
        'kl_loss': total_kl_loss / n
    }


def validate_contrastive(model, dataloader, device, lambda_contrastive=0.5):
    """Validate Contrastive VAE"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_contrastive_loss = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            
            recon, mu, logvar, z_proj = model(images, return_projection=True)
            recon_loss = nn.functional.mse_loss(recon, images, reduction='mean')
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = recon_loss + kl_loss
            
            batch_size = images.size(0)
            if batch_size > 1:
                idx1 = torch.randperm(batch_size, device=device)[:batch_size//2]
                idx2 = torch.randperm(batch_size, device=device)[:batch_size//2]
                z1 = z_proj[idx1]
                z2 = z_proj[idx2]
                contrastive_loss = contrastive_loss_fn(z1, z2)
            else:
                contrastive_loss = torch.tensor(0.0, device=device)
            
            loss = vae_loss + lambda_contrastive * contrastive_loss
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_contrastive_loss += contrastive_loss.item()
    
    n = len(dataloader)
    return {
        'total_loss': total_loss / n,
        'recon_loss': total_recon_loss / n,
        'kl_loss': total_kl_loss / n,
        'contrastive_loss': total_contrastive_loss / n
    }


def finetune_vae():
    """Fine-tune VAE on pathology data"""
    print("\n" + "="*80)
    print("FINE-TUNING VAE (Experiment 2)")
    print("="*80)
    
    # Configuration
    DATA_DIR = Path("data/pathology_data_for_training")
    CHECKPOINT_DIR = Path("checkpoints")
    PRETRAINED_PATH = CHECKPOINT_DIR / "exp1_vae_casia.pth"  # Using standard checkpoint
    
    BATCH_SIZE = 16
    NUM_EPOCHS = 10  # Reduced for quick fine-tuning
    LEARNING_RATE = 1e-5  # Lower LR for fine-tuning
    MAX_SAMPLES_PER_CONDITION = 50  # Small subset for quick training
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset (small subset)
    print("\nLoading pathology dataset (small subset)...")
    dataset = PathologyGEIDataset(DATA_DIR, max_samples_per_condition=MAX_SAMPLES_PER_CONDITION)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load pretrained model
    print(f"\nLoading pretrained model from {PRETRAINED_PATH}...")
    model = create_vae(latent_dim=128)
    checkpoint = torch.load(PRETRAINED_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("✅ Loaded pretrained weights")
    
    # Optimizer with lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 80)
        
        train_metrics = train_epoch_vae(model, train_loader, optimizer, device)
        
        print("Validating...")
        val_metrics = validate_vae(model, val_loader, device)
        
        print(f"\nTrain - Loss: {train_metrics['total_loss']:.4f}, "
              f"Recon: {train_metrics['recon_loss']:.4f}, "
              f"KL: {train_metrics['kl_loss']:.4f}")
        print(f"Val   - Loss: {val_metrics['total_loss']:.4f}, "
              f"Recon: {val_metrics['recon_loss']:.4f}, "
              f"KL: {val_metrics['kl_loss']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_total_loss': train_metrics['total_loss'],
            'val_total_loss': val_metrics['total_loss']
        }
        
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(checkpoint, CHECKPOINT_DIR / 'exp2_vae_finetuned.pth')
            print(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
    
    print(f"\n✅ Fine-tuning complete! Best val loss: {best_val_loss:.4f}")


def finetune_contrastive():
    """Fine-tune Contrastive VAE on pathology data"""
    print("\n" + "="*80)
    print("FINE-TUNING CONTRASTIVE VAE (Experiment 2)")
    print("="*80)
    
    # Configuration
    DATA_DIR = Path("data/pathology_data_for_training")
    CHECKPOINT_DIR = Path("checkpoints")
    PRETRAINED_PATH = CHECKPOINT_DIR / "exp1_contrastive_casia_latest.pth"  # Using latest checkpoint
    
    BATCH_SIZE = 16
    NUM_EPOCHS = 10  # Reduced for quick fine-tuning
    LEARNING_RATE = 1e-5
    LAMBDA_CONTRASTIVE = 0.5
    MAX_SAMPLES_PER_CONDITION = 50  # Small subset for quick training
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Check if pretrained model exists
    if not PRETRAINED_PATH.exists():
        print(f"⚠️ Pretrained model not found at {PRETRAINED_PATH}")
        print("Please train Exp1 Contrastive VAE first!")
        return
    
    # Load dataset (small subset)
    print("\nLoading pathology dataset (small subset)...")
    dataset = PathologyGEIDataset(DATA_DIR, max_samples_per_condition=MAX_SAMPLES_PER_CONDITION)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load pretrained model
    print(f"\nLoading pretrained model from {PRETRAINED_PATH}...")
    model = create_contrastive_vae(latent_dim=128, projection_dim=128)
    checkpoint = torch.load(PRETRAINED_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("✅ Loaded pretrained weights")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 80)
        
        train_metrics = train_epoch_contrastive(model, train_loader, optimizer, device, LAMBDA_CONTRASTIVE)
        
        print("Validating...")
        val_metrics = validate_contrastive(model, val_loader, device, LAMBDA_CONTRASTIVE)
        
        print(f"\nTrain - Loss: {train_metrics['total_loss']:.4f}, "
              f"Recon: {train_metrics['recon_loss']:.4f}, "
              f"KL: {train_metrics['kl_loss']:.4f}, "
              f"Contrastive: {train_metrics['contrastive_loss']:.4f}")
        print(f"Val   - Loss: {val_metrics['total_loss']:.4f}, "
              f"Recon: {val_metrics['recon_loss']:.4f}, "
              f"KL: {val_metrics['kl_loss']:.4f}, "
              f"Contrastive: {val_metrics['contrastive_loss']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_total_loss': train_metrics['total_loss'],
            'val_total_loss': val_metrics['total_loss'],
            'val_contrastive_loss': val_metrics['contrastive_loss']
        }
        
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(checkpoint, CHECKPOINT_DIR / 'exp2_contrastive_finetuned.pth')
            print(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
    
    print(f"\n✅ Fine-tuning complete! Best val loss: {best_val_loss:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['vae', 'contrastive', 'both'], 
                        default='both', help='Which model to fine-tune')
    args = parser.parse_args()
    
    if args.model in ['vae', 'both']:
        finetune_vae()
    
    if args.model in ['contrastive', 'both']:
        finetune_contrastive()


if __name__ == "__main__":
    main()
