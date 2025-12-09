"""
Training script for Contrastive VAE with GEI

Combines VAE reconstruction loss with contrastive learning loss
to learn robust embeddings for gait recognition.

Usage:
    python train_contrastive.py --epochs 50 --batch_size 32 --lr 0.001 --beta 1.0 --contrastive_weight 0.5
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
from contrastive_model import create_contrastive_vae
from contrastive_loss import info_nce_loss, supervised_contrastive_loss
from augmentations import GEIAugmentation, create_contrastive_batch
from model import vae_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train Contrastive GEI VAE')
    
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
    
    # Contrastive Learning
    parser.add_argument('--contrastive_weight', type=float, default=0.5,
                       help='Weight for contrastive loss (vs VAE loss)')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature parameter for contrastive loss')
    parser.add_argument('--use_augmentation', action='store_true',
                       help='Use data augmentation for positive pairs')
    parser.add_argument('--use_modality_pairs', action='store_true',
                       help='Pair different modalities (angles, bag, coat) from same subject as positives')
    parser.add_argument('--pair_by_angle', action='store_true', default=True,
                       help='When using modality pairs, pair samples with different view angles')
    parser.add_argument('--pair_by_condition', action='store_true', default=True,
                       help='When using modality pairs, pair samples with different conditions (nm/bg/cl)')
    parser.add_argument('--contrastive_loss_type', type=str, default='info_nce',
                       choices=['info_nce', 'supervised'],
                       help='Type of contrastive loss to use')
    
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
    parser.add_argument('--save_dir', type=str, default='checkpoints_contrastive',
                       help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs_contrastive',
                       help='Directory to save training logs')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Hardware
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, epoch, beta, 
                contrastive_weight, temperature, use_augmentation, 
                contrastive_loss_type, use_modality_pairs=False,
                pair_by_angle=True, pair_by_condition=True):
    """
    Train for one epoch.
    
    Returns:
        Average losses for the epoch
    """
    model.train()
    total_loss_sum = 0.0
    vae_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    contrastive_loss_sum = 0.0
    num_batches = 0
    
    # Initialize augmentation if needed
    augmentation = GEIAugmentation() if (use_augmentation or use_modality_pairs) else None
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (gei, labels, metadata) in enumerate(pbar):
        gei = gei.to(device)
        labels = labels.to(device)
        
        # Create augmented views for contrastive learning
        if use_augmentation or use_modality_pairs:
            if use_modality_pairs:
                # Convert batched metadata (dict of lists) back to list of dicts
                # DataLoader batches dicts into dict of lists, so we need to convert back
                if isinstance(metadata, dict):
                    batch_size = len(metadata['subject_id'])
                    metadata_list = [
                        {
                            'subject_id': metadata['subject_id'][i],
                            'sequence_id': metadata['sequence_id'][i],
                            'view_angle': metadata['view_angle'][i]
                        }
                        for i in range(batch_size)
                    ]
                else:
                    # If it's already a list, use it directly
                    metadata_list = list(metadata)
                
                gei_aug, labels_aug = create_contrastive_batch(
                    gei, labels, augmentation,
                    metadata_list=metadata_list,
                    use_modality_pairs=True,
                    pair_by_angle=pair_by_angle,
                    pair_by_condition=pair_by_condition
                )
            else:
                gei_aug, labels_aug = create_contrastive_batch(gei, labels, augmentation)
        else:
            # Use original batch twice (no augmentation)
            gei_aug = torch.cat([gei, gei], dim=0)
            labels_aug = torch.cat([labels, labels], dim=0)
        
        # Forward pass with projection
        reconstruction, mu, log_var, projection = model(gei_aug, return_projection=True)
        
        # Split back into two views
        batch_size = gei.size(0)
        projection1 = projection[:batch_size]
        projection2 = projection[batch_size:]
        labels1 = labels_aug[:batch_size]
        labels2 = labels_aug[batch_size:]
        
        # VAE loss (on original batch only)
        recon_target = gei_aug[:batch_size]  # Use first view for reconstruction
        vae_loss_val, recon_loss, kl_loss = vae_loss(
            reconstruction[:batch_size], recon_target, 
            mu[:batch_size], log_var[:batch_size], 
            beta=beta
        )
        
        # Contrastive loss (on both views)
        # Combine projections from both views
        projection_combined = torch.cat([projection1, projection2], dim=0)
        labels_combined = torch.cat([labels1, labels2], dim=0)
        
        if contrastive_loss_type == 'info_nce':
            contrastive_loss_val = info_nce_loss(
                projection_combined, labels_combined, temperature=temperature
            )
        else:  # supervised
            contrastive_loss_val = supervised_contrastive_loss(
                projection_combined, labels_combined, temperature=temperature
            )
        
        # Combined loss
        total_loss = vae_loss_val + contrastive_weight * contrastive_loss_val
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss_sum += total_loss.item()
        vae_loss_sum += vae_loss_val.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        contrastive_loss_sum += contrastive_loss_val.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'total': f'{total_loss.item():.4f}',
            'vae': f'{vae_loss_val.item():.4f}',
            'contrast': f'{contrastive_loss_val.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })
    
    avg_total_loss = total_loss_sum / num_batches
    avg_vae_loss = vae_loss_sum / num_batches
    avg_recon_loss = recon_loss_sum / num_batches
    avg_kl_loss = kl_loss_sum / num_batches
    avg_contrastive_loss = contrastive_loss_sum / num_batches
    
    return (avg_total_loss, avg_vae_loss, avg_recon_loss, 
            avg_kl_loss, avg_contrastive_loss)


def validate(model, dataloader, device, beta, contrastive_weight, 
             temperature, contrastive_loss_type, use_modality_pairs=False,
             pair_by_angle=True, pair_by_condition=True):
    """
    Validate the model.
    
    Returns:
        Average losses
    """
    model.eval()
    total_loss_sum = 0.0
    vae_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    contrastive_loss_sum = 0.0
    num_batches = 0
    
    # Initialize augmentation if using modality pairs
    augmentation = GEIAugmentation() if use_modality_pairs else None
    
    with torch.no_grad():
        for gei, labels, metadata in tqdm(dataloader, desc='Validation'):
            gei = gei.to(device)
            labels = labels.to(device)
            
            # Create two views
            if use_modality_pairs:
                # Convert batched metadata (dict of lists) back to list of dicts
                if isinstance(metadata, dict):
                    batch_size = len(metadata['subject_id'])
                    metadata_list = [
                        {
                            'subject_id': metadata['subject_id'][i],
                            'sequence_id': metadata['sequence_id'][i],
                            'view_angle': metadata['view_angle'][i]
                        }
                        for i in range(batch_size)
                    ]
                else:
                    # If it's already a list, use it directly
                    metadata_list = list(metadata)
                
                gei_aug, labels_aug = create_contrastive_batch(
                    gei, labels, augmentation,
                    metadata_list=metadata_list,
                    use_modality_pairs=True,
                    pair_by_angle=pair_by_angle,
                    pair_by_condition=pair_by_condition
                )
            else:
                # Use original batch twice (no augmentation in validation)
                gei_aug = torch.cat([gei, gei], dim=0)
                labels_aug = torch.cat([labels, labels], dim=0)
            
            # Forward pass
            reconstruction, mu, log_var, projection = model(gei_aug, return_projection=True)
            
            # Split back
            batch_size = gei.size(0)
            projection1 = projection[:batch_size]
            projection2 = projection[batch_size:]
            labels1 = labels_aug[:batch_size]
            labels2 = labels_aug[batch_size:]
            
            # VAE loss
            recon_target = gei_aug[:batch_size]
            vae_loss_val, recon_loss, kl_loss = vae_loss(
                reconstruction[:batch_size], recon_target,
                mu[:batch_size], log_var[:batch_size],
                beta=beta
            )
            
            # Contrastive loss
            projection_combined = torch.cat([projection1, projection2], dim=0)
            labels_combined = torch.cat([labels1, labels2], dim=0)
            
            if contrastive_loss_type == 'info_nce':
                contrastive_loss_val = info_nce_loss(
                    projection_combined, labels_combined, temperature=temperature
                )
            else:
                contrastive_loss_val = supervised_contrastive_loss(
                    projection_combined, labels_combined, temperature=temperature
                )
            
            # Combined loss
            total_loss = vae_loss_val + contrastive_weight * contrastive_loss_val
            
            total_loss_sum += total_loss.item()
            vae_loss_sum += vae_loss_val.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            contrastive_loss_sum += contrastive_loss_val.item()
            num_batches += 1
    
    avg_total_loss = total_loss_sum / num_batches
    avg_vae_loss = vae_loss_sum / num_batches
    avg_recon_loss = recon_loss_sum / num_batches
    avg_kl_loss = kl_loss_sum / num_batches
    avg_contrastive_loss = contrastive_loss_sum / num_batches
    
    return (avg_total_loss, avg_vae_loss, avg_recon_loss,
            avg_kl_loss, avg_contrastive_loss)


def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_total_loss': train_losses[0],
        'train_vae_loss': train_losses[1],
        'train_recon_loss': train_losses[2],
        'train_kl_loss': train_losses[3],
        'train_contrastive_loss': train_losses[4],
        'val_total_loss': val_losses[0],
        'val_vae_loss': val_losses[1],
        'val_recon_loss': val_losses[2],
        'val_kl_loss': val_losses[3],
        'val_contrastive_loss': val_losses[4]
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
    axes[0, 0].set_title('Total Loss', fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # VAE loss
    axes[0, 1].plot(epochs, history['train_vae_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_vae_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('VAE Loss', fontsize=12)
    axes[0, 1].set_title('VAE Loss', fontsize=14)
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
    
    # Reconstruction loss
    axes[1, 0].plot(epochs, history['train_recon_loss'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_recon_loss'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[1, 0].set_title('Reconstruction Loss (MSE)', fontsize=14)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # KL divergence
    axes[1, 1].plot(epochs, history['train_kl_loss'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, history['val_kl_loss'], 'r-', label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('KL Divergence', fontsize=12)
    axes[1, 1].set_title('KL Divergence', fontsize=14)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Remove empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close()


def save_reconstruction_samples(model, dataloader, device, save_path, num_samples=8):
    """Save sample reconstructions for visualization."""
    model.eval()
    
    # Get a batch
    gei, _, _ = next(iter(dataloader))
    gei = gei[:num_samples].to(device)
    
    with torch.no_grad():
        reconstruction, mu, log_var = model(gei, return_projection=False)
    
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


def diagnose_dataset(data_root, split='train'):
    """Diagnose dataset loading issues."""
    from pathlib import Path
    data_path = Path(data_root)
    
    print(f"\nDiagnosing dataset at: {data_root}")
    print(f"  Directory exists: {data_path.exists()}")
    
    if not data_path.exists():
        return
    
    # Check for data_splits.json
    splits_file = data_path / 'data_splits.json'
    print(f"  data_splits.json exists: {splits_file.exists()}")
    
    if splits_file.exists():
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        print(f"  Available splits: {list(splits.keys())}")
        if split in splits:
            print(f"  '{split}' split has {len(splits[split])} subjects")
            if 'metadata' in splits:
                print(f"  Split type: {splits['metadata'].get('split_type', 'subject_based')}")
    
    # Count GEI files
    gei_files = list(data_path.glob("*/*_gei.npy"))
    print(f"  Total GEI files found: {len(gei_files)}")
    
    if len(gei_files) > 0:
        # Show sample filenames
        print(f"  Sample files:")
        for f in gei_files[:3]:
            print(f"    {f.name}")
        if len(gei_files) > 3:
            print(f"    ... and {len(gei_files) - 3} more")


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
    print(f"Data root: {args.data_root}")
    
    # Check if data root exists
    if not os.path.exists(args.data_root):
        raise ValueError(f"Data root directory does not exist: {args.data_root}")
    
    train_dataset = GaitDataset(args.data_root, split='train', data_type='gei')
    val_dataset = GaitDataset(args.data_root, split='val', data_type='gei')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        diagnose_dataset(args.data_root, split='train')
        raise ValueError(
            f"\nNo training samples found in {args.data_root}. "
            "Please check:\n"
            "  1. Data root path is correct\n"
            "  2. data_splits.json exists and has 'train' split\n"
            "  3. GEI files exist in subject directories\n"
            "  4. File naming matches expected pattern: {subject_id}_{sequence_id}_{view_angle}_gei.npy"
        )
    
    if len(val_dataset) == 0:
        print("Warning: No validation samples found. Consider checking data_splits.json")
    
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
    
    # Create model
    print(f"\nCreating Contrastive VAE (latent_dim={args.latent_dim}, "
          f"projection_dim={args.projection_dim}, beta={args.beta}, "
          f"contrastive_weight={args.contrastive_weight})...")
    if args.use_modality_pairs:
        print(f"  Modality pairing: Enabled")
        print(f"    - Pair by angle: {args.pair_by_angle}")
        print(f"    - Pair by condition: {args.pair_by_condition}")
    if args.use_augmentation:
        print(f"  Data augmentation: Enabled")
    model = create_contrastive_vae(
        latent_dim=args.latent_dim,
        projection_dim=args.projection_dim
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
        'train_vae_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'train_contrastive_loss': [],
        'val_total_loss': [],
        'val_vae_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'val_contrastive_loss': []
    }
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, device, epoch,
            args.beta, args.contrastive_weight, args.temperature,
            args.use_augmentation, args.contrastive_loss_type,
            args.use_modality_pairs, args.pair_by_angle, args.pair_by_condition
        )
        history['train_total_loss'].append(train_losses[0])
        history['train_vae_loss'].append(train_losses[1])
        history['train_recon_loss'].append(train_losses[2])
        history['train_kl_loss'].append(train_losses[3])
        history['train_contrastive_loss'].append(train_losses[4])
        
        # Validate
        val_losses = validate(
            model, val_loader, device, args.beta, args.contrastive_weight,
            args.temperature, args.contrastive_loss_type,
            args.use_modality_pairs, args.pair_by_angle, args.pair_by_condition
        )
        history['val_total_loss'].append(val_losses[0])
        history['val_vae_loss'].append(val_losses[1])
        history['val_recon_loss'].append(val_losses[2])
        history['val_kl_loss'].append(val_losses[3])
        history['val_contrastive_loss'].append(val_losses[4])
        
        # Update learning rate
        scheduler.step(val_losses[0])
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Total: {train_losses[0]:.4f}, VAE: {train_losses[1]:.4f}, "
              f"Contrastive: {train_losses[4]:.4f}, Recon: {train_losses[2]:.4f}, "
              f"KL: {train_losses[3]:.4f}")
        print(f"  Val   - Total: {val_losses[0]:.4f}, VAE: {val_losses[1]:.4f}, "
              f"Contrastive: {val_losses[4]:.4f}, Recon: {val_losses[2]:.4f}, "
              f"KL: {val_losses[3]:.4f}")
        print("-" * 80)
        
        # Save best model
        if val_losses[0] < best_val_loss:
            best_val_loss = val_losses[0]
            save_checkpoint(
                model, optimizer, epoch,
                train_losses, val_losses,
                os.path.join(args.save_dir, 'best_model.pth')
            )
            print(f"  ✓ New best model! (val_total_loss: {val_losses[0]:.4f})")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch,
                train_losses, val_losses,
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            )
        
        # Save visualizations
        if epoch % 10 == 0 or epoch == 1:
            save_reconstruction_samples(
                model, val_loader, device,
                os.path.join(args.log_dir, f'reconstruction_epoch_{epoch}.png')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs,
        (history['train_total_loss'][-1], history['train_vae_loss'][-1],
         history['train_recon_loss'][-1], history['train_kl_loss'][-1],
         history['train_contrastive_loss'][-1]),
        (history['val_total_loss'][-1], history['val_vae_loss'][-1],
         history['val_recon_loss'][-1], history['val_kl_loss'][-1],
         history['val_contrastive_loss'][-1]),
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

