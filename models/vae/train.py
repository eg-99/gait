"""
Training script for GEI Variational Autoencoder (VAE)

Trains VAE to learn probabilistic latent space for GEI images.

Usage:
    python train.py --epochs 50 --batch_size 32 --lr 0.001 --beta 1.0
"""

import torch
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
from model import create_vae, vae_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train GEI VAE')
    
    # Data
    parser.add_argument('--data_root', type=str, 
                       default='/Users/AdityaNangia/Desktop/ADITYA/A College/COLUMBIA/Sem 3/CV 1/Project/gait/preprocessing/preprocessed_data',
                       help='Root directory of preprocessed GEI data')
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Dimension of latent space')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for KL divergence term (β-VAE)')
    
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


def train_epoch(model, dataloader, optimizer, device, epoch, beta):
    """
    Train for one epoch.
    
    Returns:
        Average total loss, reconstruction loss, and KL divergence for the epoch
    """
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (gei, _, _) in enumerate(pbar):  # Unpack (data, label, metadata)
        gei = gei.to(device)
        
        # Forward pass
        reconstruction, mu, log_var = model(gei)
        
        # Compute loss
        total_loss, recon_loss, kl_loss = vae_loss(reconstruction, gei, mu, log_var, beta=beta)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'total': f'{total_loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })
    
    avg_total_loss = total_loss_sum / num_batches
    avg_recon_loss = recon_loss_sum / num_batches
    avg_kl_loss = kl_loss_sum / num_batches
    
    return avg_total_loss, avg_recon_loss, avg_kl_loss


def validate(model, dataloader, device, beta):
    """
    Validate the model.
    
    Returns:
        Average total loss, reconstruction loss, and KL divergence
    """
    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for gei, _, _ in tqdm(dataloader, desc='Validation'):  # Unpack (data, label, metadata)
            gei = gei.to(device)
            
            # Forward pass
            reconstruction, mu, log_var = model(gei)
            
            # Compute loss
            total_loss, recon_loss, kl_loss = vae_loss(reconstruction, gei, mu, log_var, beta=beta)
            
            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1
    
    avg_total_loss = total_loss_sum / num_batches
    avg_recon_loss = recon_loss_sum / num_batches
    avg_kl_loss = kl_loss_sum / num_batches
    
    return avg_total_loss, avg_recon_loss, avg_kl_loss


def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_total_loss': train_losses[0],
        'train_recon_loss': train_losses[1],
        'train_kl_loss': train_losses[2],
        'val_total_loss': val_losses[0],
        'val_recon_loss': val_losses[1],
        'val_kl_loss': val_losses[2]
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def plot_training_curves(history, save_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_total_loss']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['train_total_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_total_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('Total VAE Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(epochs, history['train_recon_loss'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_recon_loss'], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[1].set_title('Reconstruction Loss (MSE)', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # KL divergence
    axes[2].plot(epochs, history['train_kl_loss'], 'b-', label='Train', linewidth=2)
    axes[2].plot(epochs, history['val_kl_loss'], 'r-', label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('KL Divergence', fontsize=12)
    axes[2].set_title('KL Divergence', fontsize=14)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
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
        reconstruction, mu, log_var = model(gei)
    
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


def save_generated_samples(model, device, save_path, num_samples=16):
    """Generate and save samples from prior."""
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(num_samples, device)
    
    # Move to CPU and convert to numpy
    samples = samples.cpu().numpy()
    
    # Plot
    rows = 2
    cols = num_samples // rows
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4))
    
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        axes[row, col].imshow(samples[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
    
    plt.suptitle('Generated Samples from Latent Prior', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Generated samples saved to {save_path}")
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
    print(f"\nCreating VAE (latent_dim={args.latent_dim}, beta={args.beta})...")
    model = create_vae(latent_dim=args.latent_dim)
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
    history = {
        'train_total_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_total_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': []
    }
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_total, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, epoch, args.beta
        )
        history['train_total_loss'].append(train_total)
        history['train_recon_loss'].append(train_recon)
        history['train_kl_loss'].append(train_kl)
        
        # Validate
        val_total, val_recon, val_kl = validate(model, val_loader, device, args.beta)
        history['val_total_loss'].append(val_total)
        history['val_recon_loss'].append(val_recon)
        history['val_kl_loss'].append(val_kl)
        
        # Update learning rate
        scheduler.step(val_total)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Total: {train_total:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"  Val   - Total: {val_total:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}")
        print("-" * 80)
        
        # Save best model
        if val_total < best_val_loss:
            best_val_loss = val_total
            save_checkpoint(
                model, optimizer, epoch,
                (train_total, train_recon, train_kl),
                (val_total, val_recon, val_kl),
                os.path.join(args.save_dir, 'best_model.pth')
            )
            print(f"  ✓ New best model! (val_total_loss: {val_total:.4f})")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch,
                (train_total, train_recon, train_kl),
                (val_total, val_recon, val_kl),
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            )
        
        # Save visualizations
        if epoch % 10 == 0 or epoch == 1:
            save_reconstruction_samples(
                model, val_loader, device,
                os.path.join(args.log_dir, f'reconstruction_epoch_{epoch}.png')
            )
            save_generated_samples(
                model, device,
                os.path.join(args.log_dir, f'generated_epoch_{epoch}.png')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs,
        (history['train_total_loss'][-1], history['train_recon_loss'][-1], history['train_kl_loss'][-1]),
        (history['val_total_loss'][-1], history['val_recon_loss'][-1], history['val_kl_loss'][-1]),
        os.path.join(args.save_dir, 'final_model.pth')
    )
    
    # Plot training curves
    plot_training_curves(history, os.path.join(args.log_dir, 'training_curves.png'))
    
    # Save training history
    with open(os.path.join(args.log_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {args.save_dir}")
    print(f"Logs saved to: {args.log_dir}")


if __name__ == '__main__':
    main()
