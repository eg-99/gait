"""
Extract embeddings frodef parse_args():
    parser = argparse.ArgumentParser(description='Extract embeddings from trained VAE')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Dimension of latent space (must match trained model)')
    
    # Data
    parser.add_argument('--data_root', type=str,
                       default='/Users/AdityaNangia/Desktop/ADITYA/A College/COLUMBIA/Sem 3/CV 1/Project/gait/preprocessing/preprocessed_data',
                       help='Root directory of preprocessed GEI data')r all GEI images.

Processes entire dataset (train + val + test) and saves 128-dim latent embeddings (μ).

Usage:
    python extract_embeddings.py --checkpoint checkpoints/best_model.pth
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import sys
import pickle
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from preprocessing.data_loader import GaitDataset
from model import create_vae


def parse_args():
    parser = argparse.ArgumentParser(description='Extract embeddings from trained VAE')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Dimension of latent space (must match trained model)')
    
    # Data
    parser.add_argument('--data_root', type=str,
                       default='../../preprocessing/preprocessed_data',
                       help='Root directory of preprocessed GEI data')
    parser.add_argument('--splits_file', type=str,
                       default='../../preprocessing/data_splits_by_sequence.json',
                       help='Path to data splits JSON file')
    
    # Processing
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='embeddings',
                       help='Directory to save embeddings')
    parser.add_argument('--save_log_var', action='store_true',
                       help='Also save log variance (σ²) for each embedding')
    
    return parser.parse_args()


def load_model(checkpoint_path, latent_dim, device):
    """Load trained VAE from checkpoint."""
    model = create_vae(latent_dim=latent_dim)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train Total Loss: {checkpoint['train_total_loss']:.4f}")
    print(f"  Train Recon Loss: {checkpoint['train_recon_loss']:.4f}")
    print(f"  Train KL Loss: {checkpoint['train_kl_loss']:.4f}")
    print(f"  Val Total Loss: {checkpoint['val_total_loss']:.4f}")
    print(f"  Val Recon Loss: {checkpoint['val_recon_loss']:.4f}")
    print(f"  Val KL Loss: {checkpoint['val_kl_loss']:.4f}")
    
    return model


def extract_embeddings_for_split(model, dataloader, device, split_name, save_log_var=False):
    """
    Extract embeddings for a dataset split.
    
    Returns:
        mu: numpy array (N, latent_dim) - latent means
        log_var: numpy array (N, latent_dim) - latent log variances (if save_log_var=True)
        labels: numpy array (N,) - subject IDs
        file_paths: list of file paths
    """
    mu_list = []
    log_var_list = []
    labels_list = []
    
    print(f"\nExtracting embeddings for {split_name} split...")
    
    with torch.no_grad():
        for gei, labels, _ in tqdm(dataloader, desc=f'{split_name}'):  # Unpack (data, label, metadata)
            gei = gei.to(device)
            
            # Extract latent parameters
            mu, log_var = model.encode(gei)
            
            # Move to CPU and store
            mu_list.append(mu.cpu().numpy())
            if save_log_var:
                log_var_list.append(log_var.cpu().numpy())
            labels_list.append(labels.numpy())
    
    # Concatenate all batches
    mu = np.concatenate(mu_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Get file paths from dataset
    file_paths = dataloader.dataset.samples
    
    print(f"  Extracted {len(mu)} embeddings (shape: {mu.shape})")
    
    if save_log_var:
        log_var = np.concatenate(log_var_list, axis=0)
        return mu, log_var, labels, file_paths
    else:
        return mu, None, labels, file_paths


def save_embeddings(mu, log_var, labels, file_paths, save_path):
    """Save embeddings to pickle file."""
    data = {
        'embeddings': mu,  # Using 'embeddings' for consistency with autoencoder
        'mu': mu,  # Also save as 'mu' for clarity that it's VAE latent mean
        'labels': labels,
        'file_paths': file_paths,
        'embedding_dim': mu.shape[1],
        'num_samples': len(mu)
    }
    
    if log_var is not None:
        data['log_var'] = log_var
        data['std'] = np.exp(0.5 * log_var)  # σ = exp(0.5 * log_var)
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved embeddings to {save_path}")


def compute_embedding_stats(mu, log_var, labels):
    """Compute and print statistics about embeddings."""
    print("\nEmbedding Statistics (μ):")
    print(f"  Shape: {mu.shape}")
    print(f"  Mean: {mu.mean():.6f}")
    print(f"  Std: {mu.std():.6f}")
    print(f"  Min: {mu.min():.6f}")
    print(f"  Max: {mu.max():.6f}")
    
    # Per-dimension statistics
    print(f"\n  Per-dimension statistics:")
    print(f"    Mean (across dims): {mu.mean(axis=0).mean():.6f}")
    print(f"    Std (across dims): {mu.std(axis=0).mean():.6f}")
    
    if log_var is not None:
        std = np.exp(0.5 * log_var)
        print(f"\n  Uncertainty Statistics (σ):")
        print(f"    Mean σ: {std.mean():.6f}")
        print(f"    Std σ: {std.std():.6f}")
        print(f"    Min σ: {std.min():.6f}")
        print(f"    Max σ: {std.max():.6f}")
    
    # Subject statistics
    unique_subjects = np.unique(labels)
    print(f"\n  Subject statistics:")
    print(f"    Unique subjects: {len(unique_subjects)}")
    print(f"    Samples per subject: {len(labels) / len(unique_subjects):.1f} (avg)")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.latent_dim, device)
    
    # Process each split
    splits = ['train', 'val', 'test']
    all_embeddings = {}
    
    for split in splits:
        # Load dataset
        dataset = GaitDataset(args.data_root, split=split, data_type='gei')
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Extract embeddings
        mu, log_var, labels, file_paths = extract_embeddings_for_split(
            model, dataloader, device, split, args.save_log_var
        )
        
        # Save embeddings
        save_path = os.path.join(args.output_dir, f'{split}_embeddings.pkl')
        save_embeddings(mu, log_var, labels, file_paths, save_path)
        
        # Compute statistics
        compute_embedding_stats(mu, log_var, labels)
        
        # Store for combined file
        all_embeddings[split] = {
            'mu': mu,
            'log_var': log_var,
            'labels': labels,
            'file_paths': file_paths
        }
    
    # Save combined embeddings
    print("\n" + "=" * 80)
    print("Saving combined embeddings...")
    
    combined_mu = np.concatenate([
        all_embeddings['train']['mu'],
        all_embeddings['val']['mu'],
        all_embeddings['test']['mu']
    ], axis=0)
    
    combined_labels = np.concatenate([
        all_embeddings['train']['labels'],
        all_embeddings['val']['labels'],
        all_embeddings['test']['labels']
    ], axis=0)
    
    combined_file_paths = (
        all_embeddings['train']['file_paths'] +
        all_embeddings['val']['file_paths'] +
        all_embeddings['test']['file_paths']
    )
    
    combined_log_var = None
    if args.save_log_var:
        combined_log_var = np.concatenate([
            all_embeddings['train']['log_var'],
            all_embeddings['val']['log_var'],
            all_embeddings['test']['log_var']
        ], axis=0)
    
    combined_save_path = os.path.join(args.output_dir, 'all_embeddings.pkl')
    save_embeddings(combined_mu, combined_log_var, combined_labels, combined_file_paths, combined_save_path)
    
    print("\n" + "=" * 80)
    print("Embedding extraction complete!")
    print(f"\nSummary:")
    print(f"  Train embeddings: {len(all_embeddings['train']['mu'])}")
    print(f"  Val embeddings: {len(all_embeddings['val']['mu'])}")
    print(f"  Test embeddings: {len(all_embeddings['test']['mu'])}")
    print(f"  Total embeddings: {len(combined_mu)}")
    print(f"\nFiles saved to: {args.output_dir}")
    print(f"  - train_embeddings.pkl")
    print(f"  - val_embeddings.pkl")
    print(f"  - test_embeddings.pkl")
    print(f"  - all_embeddings.pkl")
    
    if args.save_log_var:
        print(f"\n  Note: Embeddings include both μ (mean) and log_var (uncertainty)")


if __name__ == '__main__':
    main()
