"""
Training script for GEI Autoencoder

Trains autoencoder to reconstruct GEI images, learning a 128-dim embedding space.

Usage:
    python train.py --epochs 50 --batch_size 32 --lr 0.001
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from preprocessing.data_loader import GaitDataset
from model import create_autoencoder


def parse_args():
    parser = argparse.ArgumentParser(description='Train GEI Autoencoder')
    
    # Data
    parser.add_argument('--data_root', type=str, 
                       default='/Users/AdityaNangia/Desktop/ADITYA/A College/COLUMBIA/Sem 3/CV 1/Project/gait/preprocessing/preprocessed_data',
                       help='Root directory of preprocessed GEI data')
    
    # Model
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Dimension of bottleneck embedding')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (L2 regularization)')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save training logs')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Hardware
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    return parser.parse_args()


def compute_reconstruction_loss(reconstruction, target):
    """
    Compute MSE reconstruction loss.
    
    Args:
        reconstruction: Reconstructed images (batch, 1, 128, 64)
        target: Original images (batch, 1, 128, 64)
    
    Returns:
        MSE loss (scalar)
    """
    return nn.functional.mse_loss(reconstruction, target)


def train_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (gei, _, _) in enumerate(pbar):  # Unpack (data, label, metadata)
        gei = gei.to(device)
        
        # Forward pass
        reconstruction, embedding = model(gei)
        
        # Compute loss
        loss = compute_reconstruction_loss(reconstruction, gei)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, device):
    """
    Validate the model.
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for gei, _, _ in tqdm(dataloader, desc='Validation'):  # Unpack (data, label, metadata)
            gei = gei.to(device)
            
            # Forward pass
            reconstruction, embedding = model(gei)
            
            # Compute loss
            loss = compute_reconstruction_loss(reconstruction, gei)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Autoencoder Training Progress', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close()


def save_reconstruction_samples(model, dataloader, device, save_path, num_samples=8):
    """Save sample reconstructions for visualization."""
    model.eval()
    
    # Get a batch
    gei, _, _ = next(iter(dataloader))  # Unpack (data, label, metadata)
    gei = gei[:num_samples].to(device)
    
    with torch.no_grad():
        reconstruction, _ = model(gei)
    
    # Move to CPU and convert to numpy
    gei = gei.cpu().numpy()
    reconstruction = reconstruction.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 4))
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(gei[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstruction
        axes[1, i].imshow(reconstruction[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Reconstruction samples saved to {save_path}")
    plt.close()


def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.log_dir, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading datasets...")
    train_dataset = GaitDataset(args.data_root, split='train', data_type='gei')
    val_dataset = GaitDataset(args.data_root, split='val', data_type='gei')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print(f"\nCreating autoencoder (embedding_dim={args.embedding_dim})...")
    model = create_autoencoder(embedding_dim=args.embedding_dim)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print("-" * 80)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                os.path.join(args.save_dir, 'best_model.pth')
            )
            print(f"  ✓ New best model! (val_loss: {val_loss:.6f})")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            )
        
        # Save reconstruction samples
        if epoch % 10 == 0 or epoch == 1:
            save_reconstruction_samples(
                model, val_loader, device,
                os.path.join(args.log_dir, f'reconstruction_epoch_{epoch}.png')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs, train_losses[-1], val_losses[-1],
        os.path.join(args.save_dir, 'final_model.pth')
    )
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses,
        os.path.join(args.log_dir, 'training_curves.png')
    )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    with open(os.path.join(args.log_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {args.save_dir}")
    print(f"Logs saved to: {args.log_dir}")


if __name__ == '__main__':
    main()
