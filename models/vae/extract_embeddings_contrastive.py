"""
Extract embeddings from trained Contrastive VAE for all GEI images.

Uses the VAE encoder (mean) as the embedding, which is trained with
both reconstruction and contrastive learning.

Usage:
    python extract_embeddings_contrastive.py --checkpoint checkpoints_contrastive/best_model.pth
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
from contrastive_model import create_contrastive_vae


def parse_args():
    parser = argparse.ArgumentParser(description='Extract embeddings from trained Contrastive VAE')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Dimension of latent space (must match trained model)')
    parser.add_argument('--projection_dim', type=int, default=128,
                       help='Dimension of projection head (must match trained model)')
    
    # Data
    parser.add_argument('--data_root', type=str,
                       default='/Users/AdityaNangia/Desktop/ADITYA/A College/COLUMBIA/Sem 3/CV 1/Project/gait/preprocessing/preprocessed_data',
                       help='Root directory of preprocessed GEI data')
    
    # Processing
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='embeddings_contrastive',
                       help='Directory to save embeddings')
    parser.add_argument('--use_projection', action='store_true',
                       help='Use projection head embeddings instead of raw latent mean')
    
    return parser.parse_args()


def load_model(checkpoint_path, latent_dim, projection_dim, device):
    """Load trained Contrastive VAE from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model
    model = create_contrastive_vae(
        latent_dim=latent_dim,
        projection_dim=projection_dim
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded successfully!")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Train Total Loss: {checkpoint.get('train_total_loss', 'unknown'):.6f}")
    print(f"  Val Total Loss: {checkpoint.get('val_total_loss', 'unknown'):.6f}")
    print(f"  Val VAE Loss: {checkpoint.get('val_vae_loss', 'unknown'):.6f}")
    print(f"  Val Contrastive Loss: {checkpoint.get('val_contrastive_loss', 'unknown'):.6f}")
    
    return model


def extract_embeddings_for_split(model, dataloader, device, split_name, use_projection=False):
    """
    Extract embeddings for a dataset split.
    
    Returns:
        embeddings: numpy array (N, embedding_dim)
        labels: numpy array (N,) - subject IDs
        file_paths: list of file paths
    """
    embeddings_list = []
    labels_list = []
    
    print(f"\nExtracting embeddings for {split_name} split...")
    
    with torch.no_grad():
        for gei, labels, _ in tqdm(dataloader, desc=f'{split_name}'):
            gei = gei.to(device)
            
            if use_projection:
                # Use projection head embeddings
                projection, _, _ = model.get_projection(gei)
                embeddings_list.append(projection.cpu().numpy())
            else:
                # Use raw latent mean (VAE embedding)
                embedding = model.get_embedding(gei)
                embeddings_list.append(embedding.cpu().numpy())
            
            labels_list.append(labels.numpy())
    
    # Concatenate all batches
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Get file paths from dataset
    file_paths = [str(sample['path']) for sample in dataloader.dataset.samples]
    
    print(f"  Extracted {len(embeddings)} embeddings (shape: {embeddings.shape})")
    
    return embeddings, labels, file_paths


def save_embeddings(embeddings, labels, file_paths, save_path):
    """Save embeddings to pickle file."""
    data = {
        'embeddings': embeddings,
        'labels': labels,
        'file_paths': file_paths
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"  Saved embeddings to {save_path}")


def compute_embedding_stats(embeddings, labels):
    """Compute and print embedding statistics."""
    print(f"\n  Embedding Statistics:")
    print(f"    Shape: {embeddings.shape}")
    print(f"    Mean: {embeddings.mean():.6f}")
    print(f"    Std: {embeddings.std():.6f}")
    print(f"    Min: {embeddings.min():.6f}")
    print(f"    Max: {embeddings.max():.6f}")
    
    # Compute L2 norms
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"    L2 Norm - Mean: {norms.mean():.6f}, Std: {norms.std():.6f}")
    
    # Subject statistics
    unique_subjects = np.unique(labels)
    print(f"    Unique subjects: {len(unique_subjects)}")
    print(f"    Total samples: {len(labels)}")
    print(f"    Samples per subject: {len(labels) / len(unique_subjects):.1f} (avg)")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, args.latent_dim, args.projection_dim, device)
    
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
        embeddings, labels, file_paths = extract_embeddings_for_split(
            model, dataloader, device, split, args.use_projection
        )
        
        # Save embeddings
        save_path = os.path.join(args.output_dir, f'{split}_embeddings.pkl')
        save_embeddings(embeddings, labels, file_paths, save_path)
        
        # Compute statistics
        compute_embedding_stats(embeddings, labels)
        
        # Store for combined file
        all_embeddings[split] = {
            'embeddings': embeddings,
            'labels': labels,
            'file_paths': file_paths
        }
    
    # Save combined embeddings
    print("\n" + "=" * 80)
    print("Saving combined embeddings...")
    
    combined_embeddings = np.concatenate([
        all_embeddings['train']['embeddings'],
        all_embeddings['val']['embeddings'],
        all_embeddings['test']['embeddings']
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
    
    save_path = os.path.join(args.output_dir, 'all_embeddings.pkl')
    save_embeddings(combined_embeddings, combined_labels, combined_file_paths, save_path)
    
    compute_embedding_stats(combined_embeddings, combined_labels)
    
    print("\n" + "=" * 80)
    print("Embedding extraction complete!")
    print(f"Embeddings saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

