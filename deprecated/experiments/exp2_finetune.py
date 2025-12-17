"""
Experiment 2: Fine-Tuning
Fine-tune models pretrained on CASIA-B using small subset of pathology data.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

# Add model paths
sys.path.append(str(Path(__file__).parent.parent / 'models' / 'vae'))
sys.path.append(str(Path(__file__).parent))

from models.vae.model import create_vae
from models.vae.contrastive_model import create_contrastive_vae
from models.vae.contrastive_loss import InfoNCELoss, SupervisedContrastiveLoss
from evaluation_utils import (
    collect_pathology_samples, extract_embeddings,
    binary_classification, multiclass_classification,
    plot_confusion_matrix, plot_roc_curve, save_results, compare_models
)


class PathologyDataset(Dataset):
    """Dataset for pathology images with labels."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('L')
        img = img.resize((64, 128), Image.LANCZOS)
        
        # Convert to tensor [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, self.labels[idx]


def split_data(samples, train_ratio=0.25, val_ratio=0.15, random_state=42):
    """
    Split data into train/val/test sets.
    
    Args:
        samples: Dict with 'paths' and 'conditions'
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        random_state: Random seed
    
    Returns:
        dict: Split indices
    """
    from sklearn.model_selection import train_test_split
    
    n_samples = len(samples['paths'])
    indices = np.arange(n_samples)
    
    # First split: train+val vs test
    test_ratio = 1 - train_ratio - val_ratio
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=random_state,
        stratify=samples['conditions']
    )
    
    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    conditions_train_val = [samples['conditions'][i] for i in train_val_idx]
    
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio_adjusted, random_state=random_state,
        stratify=conditions_train_val
    )
    
    return {
        'train': train_idx.tolist(),
        'val': val_idx.tolist(),
        'test': test_idx.tolist()
    }


def train_epoch(model, dataloader, optimizer, criterion, device, model_type='vae'):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for images, labels in tqdm(dataloader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if model_type == 'vae':
            recon, mu, logvar = model(images)
            # VAE loss
            recon_loss = nn.functional.mse_loss(recon, images, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + kl_loss) / images.size(0)
        
        elif model_type == 'contrastive':
            # Contrastive VAE loss (simplified for fine-tuning)
            recon, mu, logvar = model.vae(images)
            recon_loss = nn.functional.mse_loss(recon, images, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = (recon_loss + kl_loss) / images.size(0)
            
            # Contrastive loss (supervised)
            projection = model.projection_head(mu)
            contrastive_criterion = SupervisedContrastiveLoss(temperature=0.07)
            contrastive_loss = contrastive_criterion(projection.unsqueeze(1), labels)
            
            loss = vae_loss + 0.5 * contrastive_loss  # Weight contrastive loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device, model_type='vae'):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            if model_type == 'vae':
                recon, mu, logvar = model(images)
                recon_loss = nn.functional.mse_loss(recon, images, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (recon_loss + kl_loss) / images.size(0)
            
            elif model_type == 'contrastive':
                recon, mu, logvar = model.vae(images)
                recon_loss = nn.functional.mse_loss(recon, images, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (recon_loss + kl_loss) / images.size(0)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Experiment 2: Fine-Tuning')
    parser.add_argument('--pretrained_checkpoint', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['vae', 'contrastive'], required=True,
                        help='Model type to fine-tune')
    parser.add_argument('--pathology_root', type=str, required=True,
                        help='Root directory of pathology dataset')
    parser.add_argument('--output_dir', type=str, default='checkpoints/exp2_finetune',
                        help='Output directory for checkpoints')
    parser.add_argument('--train_split', type=float, default=0.25,
                        help='Proportion of data for training')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Proportion of data for validation')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights (only train classifier)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: FINE-TUNING")
    print("="*70)
    print(f"Model type: {args.model_type}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Train split: {args.train_split:.2%}")
    print(f"Val split: {args.val_split:.2%}")
    print("="*70 + "\n")
    
    # Load pathology samples
    print("Loading pathology dataset...")
    samples = collect_pathology_samples(args.pathology_root)
    
    if len(samples['paths']) == 0:
        print("❌ No samples found. Check pathology_root path.")
        return
    
    # Split data
    print(f"\nSplitting data (train={args.train_split:.2%}, val={args.val_split:.2%})...")
    splits = split_data(samples, args.train_split, args.val_split)
    
    # Save splits
    with open(output_dir / 'data_splits.json', 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val:   {len(splits['val'])} samples")
    print(f"  Test:  {len(splits['test'])} samples")
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    all_labels = le.fit_transform(samples['conditions'])
    
    # Create datasets
    train_dataset = PathologyDataset(
        [samples['paths'][i] for i in splits['train']],
        [all_labels[i] for i in splits['train']]
    )
    val_dataset = PathologyDataset(
        [samples['paths'][i] for i in splits['val']],
        [all_labels[i] for i in splits['val']]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load pretrained model
    print(f"\nLoading pretrained model from {args.pretrained_checkpoint}...")
    if args.model_type == 'vae':
        model = create_vae(latent_dim=128)
    else:
        model = create_contrastive_vae(latent_dim=128, projection_dim=128)
    
    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✅ Model loaded (Epoch: {checkpoint.get('epoch', 'N/A')})")
    
    # Freeze encoder if requested
    if args.freeze_encoder:
        print("\n🔒 Freezing encoder weights...")
        if args.model_type == 'vae':
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.vae.encoder.parameters():
                param.requires_grad = False
    
    # Setup optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, None, device, args.model_type)
        val_loss = validate(model, val_loader, device, args.model_type)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")
    
    print(f"\n✅ Fine-tuning complete! Best model saved to {output_dir / 'best_model.pth'}")
    print(f"\n📊 To evaluate, run:")
    print(f"python experiments/exp2_evaluate.py \\")
    print(f"  --checkpoint {output_dir / 'best_model.pth'} \\")
    print(f"  --model_type {args.model_type} \\")
    print(f"  --pathology_root {args.pathology_root} \\")
    print(f"  --split_file {output_dir / 'data_splits.json'}")


if __name__ == '__main__':
    main()
