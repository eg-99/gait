"""
Training script for GEI Denoising Contrastive Variational Autoencoder (D-CVAE)

Trains D-CVAE to learn robust and discriminative probabilistic latent space by:
- Reconstructing clean images from corrupted inputs (denoising)
- Using contrastive learning to pull similar subjects together (discriminative)

Usage:
    python train_dcvae.py --epochs 50 --batch_size 32 --lr 0.001 --beta 1.0 --contrastive-weight 1.0 --noise-type gaussian
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
from dcvae_model import create_dcvae, dcvae_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train GEI Denoising Contrastive VAE')
    
    # Data
    parser.add_argument('--data_root', type=str, 
                       default=r'C:\Users\User\Documents\UNI\CV1\Casia-B-Images\preprocessed',
                       help='Root directory of preprocessed GEI data')
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Dimension of latent space')
    parser.add_argument('--projection_dim', type=int, default=128,
                       help='Dimension of projection head output')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for KL divergence term (β-VAE)')
    
    # Denoising parameters
    parser.add_argument('--noise-type', type=str, default='gaussian',
                       choices=['gaussian', 'masking', 'salt_pepper', 'mixed'],
                       help='Type of noise to apply during training')
    parser.add_argument('--noise-std', type=float, default=0.1,
                       help='Standard deviation for Gaussian noise')
    parser.add_argument('--noise-prob', type=float, default=0.3,
                       help='Probability for masking/salt-pepper noise')
    
    # Contrastive parameters
    parser.add_argument('--contrastive-weight', type=float, default=1.0,
                       help='Weight for contrastive loss term')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature parameter for contrastive loss')
    
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
    parser.add_argument('--save_dir', type=str, default='checkpoints_dcvae',
                       help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs_dcvae',
                       help='Directory to save training logs')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Hardware
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, epoch, beta, contrastive_weight, temperature):
    """
    Train for one epoch.
    
    Returns:
        Average losses: total, recon, kl, contrastive
    """
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    contrastive_loss_sum = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (gei, labels, _) in enumerate(pbar):  # Unpack (data, label, metadata)
        gei = gei.to(device)
        labels = labels.to(device)
        
        # Forward pass with corruption and projection (D-CVAE)
        reconstruction, mu, log_var, corrupted, projection = model(
            gei, corrupt=True, return_projection=True
        )
        
        # Compute loss - combines denoising, KL, and contrastive
        total_loss, recon_loss, kl_loss, contrastive_loss = dcvae_loss(
            reconstruction, gei, mu, log_var,
            corrupted=corrupted,
            projection=projection,
            labels=labels,
            beta=beta,
            contrastive_weight=contrastive_weight,
            temperature=temperature
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        contrastive_loss_sum += contrastive_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'total': f'{total_loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}',
            'contr': f'{contrastive_loss.item():.4f}'
        })
    
    avg_total_loss = total_loss_sum / num_batches
    avg_recon_loss = recon_loss_sum / num_batches
    avg_kl_loss = kl_loss_sum / num_batches
    avg_contrastive_loss = contrastive_loss_sum / num_batches
    
    return avg_total_loss, avg_recon_loss, avg_kl_loss, avg_contrastive_loss


def validate(model, dataloader, device, beta, contrastive_weight, temperature):
    """
    Validate the model.
    
    Returns:
        Average losses: total, recon, kl, contrastive
    """
    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    contrastive_loss_sum = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for gei, labels, _ in tqdm(dataloader, desc='Validation'):
            gei = gei.to(device)
            labels = labels.to(device)
            
            # Forward pass with corruption (to test denoising ability)
            reconstruction, mu, log_var, corrupted, projection = model(
                gei, corrupt=True, return_projection=True
            )
            
            # Compute loss
            total_loss, recon_loss, kl_loss, contrastive_loss = dcvae_loss(
                reconstruction, gei, mu, log_var,
                corrupted=corrupted,
                projection=projection,
                labels=labels,
                beta=beta,
                contrastive_weight=contrastive_weight,
                temperature=temperature
            )
            
            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            contrastive_loss_sum += contrastive_loss.item()
            num_batches += 1
    
    avg_total_loss = total_loss_sum / num_batches
    avg_recon_loss = recon_loss_sum / num_batches
    avg_kl_loss = kl_loss_sum / num_batches
    avg_contrastive_loss = contrastive_loss_sum / num_batches
    
    return avg_total_loss, avg_recon_loss, avg_kl_loss, avg_contrastive_loss


def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, save_path, 
                   noise_type, noise_std, noise_prob, contrastive_weight, temperature):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_total_loss': train_losses[0],
        'train_recon_loss': train_losses[1],
        'train_kl_loss': train_losses[2],
        'train_contrastive_loss': train_losses[3],
        'val_total_loss': val_losses[0],
        'val_recon_loss': val_losses[1],
        'val_kl_loss': val_losses[2],
        'val_contrastive_loss': val_losses[3],
        'noise_type': noise_type,
        'noise_std': noise_std,
        'noise_prob': noise_prob,
        'contrastive_weight': contrastive_weight,
        'temperature': temperature
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def plot_training_curves(history, save_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = range(1, len(history['train_total_loss']) + 1)
    
    # Total loss
    axes[0, 0].plot(epochs, history['train_total_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_total_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Loss', fontsize=12)
    axes[0, 0].set_title('Total D-CVAE Loss', fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Denoising reconstruction loss
    axes[0, 1].plot(epochs, history['train_recon_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_recon_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[0, 1].set_title('Denoising Loss (MSE)', fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Contrastive loss
    axes[0, 2].plot(epochs, history['train_contrastive_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, history['val_contrastive_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Contrastive Loss', fontsize=12)
    axes[0, 2].set_title('Contrastive Loss', fontsize=14)
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)
    
    # KL divergence
    axes[1, 0].plot(epochs, history['train_kl_loss'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_kl_loss'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('KL Divergence', fontsize=12)
    axes[1, 0].set_title('KL Divergence', fontsize=14)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Remove empty subplots
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close()


def save_reconstruction_samples(model, dataloader, device, save_path, num_samples=8):
    """Save sample reconstructions showing corrupted input, clean target, and reconstruction."""
    model.eval()
    
    # Get a batch
    gei, _, _ = next(iter(dataloader))
    gei = gei[:num_samples].to(device)
    
    with torch.no_grad():
        # Forward with corruption
        reconstruction, mu, log_var, corrupted = model(gei, corrupt=True, return_projection=False)
    
    # Move to CPU and convert to numpy
    gei = gei.cpu().numpy()
    reconstruction = reconstruction.cpu().numpy()
    corrupted = corrupted.cpu().numpy()
    
    # Plot: 3 rows (corrupted, clean, reconstructed)
    fig, axes = plt.subplots(3, num_samples, figsize=(16, 6))
    
    for i in range(num_samples):
        # Corrupted input
        axes[0, i].imshow(corrupted[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Corrupted Input', fontsize=10)
        
        # Clean target
        axes[1, i].imshow(gei[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Clean Target', fontsize=10)
        
        # Reconstruction
        axes[2, i].imshow(reconstruction[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Reconstructed', fontsize=10)
    
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
    
    # Create directories (separate from other models)
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
    print(f"Data root: {args.data_root}")
    
    # Validate data directory
    data_root_path = Path(args.data_root)
    if not data_root_path.exists():
        raise ValueError(f"Data root directory does not exist: {args.data_root}")
    
    # Check for data_splits.json or subject directories
    splits_file = data_root_path / 'data_splits.json'
    subject_dirs = [d for d in data_root_path.iterdir() if d.is_dir()]
    
    if not splits_file.exists() and len(subject_dirs) == 0:
        raise ValueError(f"No data found in {args.data_root}. Expected either:\n"
                        f"  1. A 'data_splits.json' file, or\n"
                        f"  2. Subject directories containing '*_gei.npy' files")
    
    # Check for .npy files in first subject directory
    if len(subject_dirs) > 0:
        sample_subject = subject_dirs[0]
        gei_files = list(sample_subject.glob("*_gei.npy"))
        if len(gei_files) == 0:
            print(f"WARNING: No '*_gei.npy' files found in {sample_subject}")
            print(f"  Found files: {list(sample_subject.glob('*'))[:5]}...")
            print(f"  The dataset expects preprocessed .npy files, not raw images.")
            print(f"  Please run preprocessing first or use a directory with preprocessed data.")
            raise ValueError(f"No GEI .npy files found. Dataset expects preprocessed .npy files.")
    
    try:
        train_dataset = GaitDataset(args.data_root, split='train', data_type='gei')
        val_dataset = GaitDataset(args.data_root, split='val', data_type='gei')
    except Exception as e:
        print(f"\nError loading datasets: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check that {args.data_root} contains preprocessed .npy files")
        print(f"  2. Files should be named like: '001_nm-01_090_gei.npy'")
        print(f"  3. If using splits, ensure 'data_splits.json' exists")
        raise
    
    if len(train_dataset) == 0:
        raise ValueError(f"Train dataset is empty! Found 0 samples in {args.data_root}")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty! Found 0 samples in {args.data_root}")
    
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
    print(f"\nCreating Denoising Contrastive VAE:")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Projection dim: {args.projection_dim}")
    print(f"  Beta: {args.beta}")
    print(f"  Contrastive weight: {args.contrastive_weight}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Noise type: {args.noise_type}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Noise prob: {args.noise_prob}")
    
    model = create_dcvae(
        latent_dim=args.latent_dim,
        projection_dim=args.projection_dim,
        noise_type=args.noise_type,
        noise_std=args.noise_std,
        noise_prob=args.noise_prob
    )
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
        'train_contrastive_loss': [],
        'val_total_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'val_contrastive_loss': []
    }
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_total, train_recon, train_kl, train_contr = train_epoch(
            model, train_loader, optimizer, device, epoch,
            args.beta, args.contrastive_weight, args.temperature
        )
        history['train_total_loss'].append(train_total)
        history['train_recon_loss'].append(train_recon)
        history['train_kl_loss'].append(train_kl)
        history['train_contrastive_loss'].append(train_contr)
        
        # Validate
        val_total, val_recon, val_kl, val_contr = validate(
            model, val_loader, device,
            args.beta, args.contrastive_weight, args.temperature
        )
        history['val_total_loss'].append(val_total)
        history['val_recon_loss'].append(val_recon)
        history['val_kl_loss'].append(val_kl)
        history['val_contrastive_loss'].append(val_contr)
        
        # Update learning rate
        scheduler.step(val_total)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Total: {train_total:.4f}, Recon: {train_recon:.4f}, "
              f"KL: {train_kl:.4f}, Contrastive: {train_contr:.4f}")
        print(f"  Val   - Total: {val_total:.4f}, Recon: {val_recon:.4f}, "
              f"KL: {val_kl:.4f}, Contrastive: {val_contr:.4f}")
        print("-" * 80)
        
        # Save best model
        if val_total < best_val_loss:
            best_val_loss = val_total
            save_checkpoint(
                model, optimizer, epoch,
                (train_total, train_recon, train_kl, train_contr),
                (val_total, val_recon, val_kl, val_contr),
                os.path.join(args.save_dir, 'best_model.pth'),
                args.noise_type, args.noise_std, args.noise_prob,
                args.contrastive_weight, args.temperature
            )
            print(f"  ✓ New best model! (val_total_loss: {val_total:.4f})")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch,
                (train_total, train_recon, train_kl, train_contr),
                (val_total, val_recon, val_kl, val_contr),
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'),
                args.noise_type, args.noise_std, args.noise_prob,
                args.contrastive_weight, args.temperature
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
        (history['train_total_loss'][-1], history['train_recon_loss'][-1], 
         history['train_kl_loss'][-1], history['train_contrastive_loss'][-1]),
        (history['val_total_loss'][-1], history['val_recon_loss'][-1], 
         history['val_kl_loss'][-1], history['val_contrastive_loss'][-1]),
        os.path.join(args.save_dir, 'final_model.pth'),
        args.noise_type, args.noise_std, args.noise_prob,
        args.contrastive_weight, args.temperature
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

