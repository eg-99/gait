"""
Train Contrastive VAE on CASIA-B data for Experiment 1 (Zero-shot Transfer)
Uses preprocessed GEI data from preprocessing/preprocessed_data/
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import random

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from contrastive_model import create_contrastive_vae

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class CASIABGEIDataset(Dataset):
    """Dataset for CASIA-B preprocessed GEI files"""
    
    def __init__(self, root_dir, subjects=None, target_size=(128, 64)):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.samples = []
        
        # Get all subjects
        if subjects is None:
            subjects = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        
        # Collect all GEI files
        for subject_id in subjects:
            subject_dir = self.root_dir / subject_id
            if not subject_dir.exists():
                continue
            
            gei_files = list(subject_dir.glob("*_gei.npy"))
            for gei_path in gei_files:
                # Extract metadata from filename: 001_nm-01_036_gei.npy
                parts = gei_path.stem.replace('_gei', '').split('_')
                if len(parts) >= 3:
                    subj, condition, angle = parts[0], parts[1], parts[2]
                    self.samples.append({
                        'path': gei_path,
                        'subject': subj,
                        'condition': condition,
                        'angle': angle
                    })
        
        print(f"Loaded {len(self.samples)} GEI samples from {len(subjects)} subjects")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load GEI
        gei = np.load(sample['path'])
        
        # Ensure correct shape and normalize
        if gei.ndim == 2:
            gei = gei[np.newaxis, :]  # Add channel dimension
        
        # Resize if needed
        if gei.shape[1:] != self.target_size:
            from PIL import Image
            gei_img = Image.fromarray((gei[0] * 255).astype(np.uint8))
            gei_img = gei_img.resize((self.target_size[1], self.target_size[0]), Image.LANCZOS)
            gei = np.array(gei_img, dtype=np.float32)[np.newaxis, :] / 255.0
        
        # Normalize to [0, 1]
        if gei.max() > 1.0:
            gei = gei / 255.0
        
        return torch.from_numpy(gei).float(), sample['subject']


def contrastive_loss_fn(z_i, z_j, temperature=0.07):
    """NT-Xent contrastive loss"""
    batch_size = z_i.size(0)
    z_i = nn.functional.normalize(z_i, dim=1)
    z_j = nn.functional.normalize(z_j, dim=1)
    
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    
    # Create labels: positive pairs are (i, i+batch_size)
    labels = torch.arange(batch_size, device=z_i.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    # Mask to remove self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    
    # Compute loss
    similarity_matrix = similarity_matrix / temperature
    criterion = nn.CrossEntropyLoss()
    loss = criterion(similarity_matrix, labels)
    
    return loss


def train_epoch(model, dataloader, optimizer, device, lambda_contrastive=0.5):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_contrastive_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, subjects) in enumerate(pbar):
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
        
        # Contrastive loss (use same batch augmented views)
        # For simplicity, we'll use random pairs from same subject
        batch_size = images.size(0)
        if batch_size > 1:
            # Create positive pairs by random pairing within batch
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
        
        # Track losses
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


def validate(model, dataloader, device, lambda_contrastive=0.5):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_contrastive_loss = 0
    
    with torch.no_grad():
        for images, subjects in dataloader:
            images = images.to(device)
            
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


def main():
    # Configuration
    BASE_DIR = Path(__file__).parent.parent
    PREPROCESSED_DATA = BASE_DIR / "preprocessing/preprocessed_data"
    OUTPUT_DIR = Path(__file__).parent / "checkpoints"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    LATENT_DIM = 128
    PROJECTION_DIM = 128
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    LAMBDA_CONTRASTIVE = 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data splits
    splits_file = PREPROCESSED_DATA / "data_splits.json"
    if splits_file.exists():
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        train_subjects = splits['train']
        val_subjects = splits['val']
        print(f"Loaded splits: {len(train_subjects)} train, {len(val_subjects)} val subjects")
    else:
        # Create default split
        all_subjects = sorted([d.name for d in PREPROCESSED_DATA.iterdir() if d.is_dir()])
        split_idx = int(0.8 * len(all_subjects))
        train_subjects = all_subjects[:split_idx]
        val_subjects = all_subjects[split_idx:]
        print(f"Created default split: {len(train_subjects)} train, {len(val_subjects)} val subjects")
    
    # Create datasets
    print("\nLoading training data...")
    train_dataset = CASIABGEIDataset(PREPROCESSED_DATA, subjects=train_subjects)
    print("Loading validation data...")
    val_dataset = CASIABGEIDataset(PREPROCESSED_DATA, subjects=val_subjects)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating Contrastive VAE model...")
    model = create_contrastive_vae(latent_dim=LATENT_DIM, projection_dim=PROJECTION_DIM)
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\n" + "="*80)
    print("TRAINING CONTRASTIVE VAE ON CASIA-B (Experiment 1)")
    print("="*80)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, LAMBDA_CONTRASTIVE)
        
        # Validate
        print("Validating...")
        val_metrics = validate(model, val_loader, device, LAMBDA_CONTRASTIVE)
        
        # Print metrics
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
            'train_recon_loss': train_metrics['recon_loss'],
            'train_kl_loss': train_metrics['kl_loss'],
            'train_contrastive_loss': train_metrics['contrastive_loss'],
            'val_total_loss': val_metrics['total_loss'],
            'val_recon_loss': val_metrics['recon_loss'],
            'val_kl_loss': val_metrics['kl_loss'],
            'val_contrastive_loss': val_metrics['contrastive_loss']
        }
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(checkpoint, OUTPUT_DIR / 'exp1_contrastive_casia.pth')
            print(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save latest
        torch.save(checkpoint, OUTPUT_DIR / 'exp1_contrastive_casia_latest.pth')
    
    print("\n" + "="*80)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {OUTPUT_DIR / 'exp1_contrastive_casia.pth'}")
    print("="*80)


if __name__ == "__main__":
    main()
