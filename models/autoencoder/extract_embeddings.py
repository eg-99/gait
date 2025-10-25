"""
Extract embeddings from trained autoencoder for all GEI images.

Processes entire dataset (train + val + test) and saves 128-dim embeddings.

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
from model import create_autoencoder


def parse_args():
    parser = argparse.ArgumentParser(description='Extract embeddings from trained autoencoder')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Dimension of embedding (must match trained model)')
    
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
    parser.add_argument('--output_dir', type=str, default='embeddings',
                       help='Directory to save embeddings')
    
    return parser.parse_args()


def load_model(checkpoint_path, embedding_dim, device):
    """Load trained autoencoder from checkpoint."""
    model = create_autoencoder(embedding_dim=embedding_dim)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train Loss: {checkpoint['train_loss']:.6f}")
    print(f"  Val Loss: {checkpoint['val_loss']:.6f}")
    
    return model


def extract_embeddings_for_split(model, dataloader, device, split_name):
    """
    Extract embeddings for a dataset split.
    
    Returns:
        embeddings: numpy array (N, embedding_dim)
        labels: numpy array (N,) - subject IDs
        file_paths: list of file paths
    """
    embeddings_list = []
    labels_list = []
    file_paths_list = []
    
    print(f"\nExtracting embeddings for {split_name} split...")
    
    with torch.no_grad():
        for gei, labels, _ in tqdm(dataloader, desc=f'{split_name}'):  # Unpack (data, label, metadata)
            gei = gei.to(device)
            
            # Extract embeddings
            embedding = model.encode(gei)
            
            # Move to CPU and store
            embeddings_list.append(embedding.cpu().numpy())
            labels_list.append(labels.numpy())
    
    # Concatenate all batches
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Get file paths from dataset
    file_paths = dataloader.dataset.samples
    
    print(f"  Extracted {len(embeddings)} embeddings (shape: {embeddings.shape})")
    
    return embeddings, labels, file_paths


def save_embeddings(embeddings, labels, file_paths, save_path):
    """Save embeddings to pickle file."""
    data = {
        'embeddings': embeddings,
        'labels': labels,
        'file_paths': file_paths,
        'embedding_dim': embeddings.shape[1],
        'num_samples': len(embeddings)
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved embeddings to {save_path}")


def compute_embedding_stats(embeddings, labels):
    """Compute and print statistics about embeddings."""
    print("\nEmbedding Statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")
    
    # Per-dimension statistics
    print(f"\n  Per-dimension statistics:")
    print(f"    Mean (across dims): {embeddings.mean(axis=0).mean():.6f}")
    print(f"    Std (across dims): {embeddings.std(axis=0).mean():.6f}")
    
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
    model = load_model(args.checkpoint, args.embedding_dim, device)
    
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
            model, dataloader, device, split
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
    
    combined_save_path = os.path.join(args.output_dir, 'all_embeddings.pkl')
    save_embeddings(combined_embeddings, combined_labels, combined_file_paths, combined_save_path)
    
    print("\n" + "=" * 80)
    print("Embedding extraction complete!")
    print(f"\nSummary:")
    print(f"  Train embeddings: {len(all_embeddings['train']['embeddings'])}")
    print(f"  Val embeddings: {len(all_embeddings['val']['embeddings'])}")
    print(f"  Test embeddings: {len(all_embeddings['test']['embeddings'])}")
    print(f"  Total embeddings: {len(combined_embeddings)}")
    print(f"\nFiles saved to: {args.output_dir}")
    print(f"  - train_embeddings.pkl")
    print(f"  - val_embeddings.pkl")
    print(f"  - test_embeddings.pkl")
    print(f"  - all_embeddings.pkl")


if __name__ == '__main__':
    main()
