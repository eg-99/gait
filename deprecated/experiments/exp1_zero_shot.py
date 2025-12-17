"""
Experiment 1: Zero-Shot Transfer
Train on CASIA-B (healthy), test on pathology data.
"""

import argparse
import sys
from pathlib import Path
import torch

# Add model paths
sys.path.append(str(Path(__file__).parent.parent / 'models' / 'vae'))
sys.path.append(str(Path(__file__).parent))

from models.vae.model import create_vae
from models.vae.contrastive_model import create_contrastive_vae
from evaluation_utils import (
    extract_embeddings, collect_pathology_samples,
    binary_classification, multiclass_classification,
    plot_confusion_matrix, plot_roc_curve, save_results, compare_models
)


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Zero-Shot Transfer')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                        help='Path to trained VAE checkpoint (trained on CASIA-B)')
    parser.add_argument('--contrastive_checkpoint', type=str, required=True,
                        help='Path to trained Contrastive VAE checkpoint (trained on CASIA-B)')
    parser.add_argument('--pathology_root', type=str, required=True,
                        help='Root directory of pathology dataset')
    parser.add_argument('--output_dir', type=str, default='results/exp1_zero_shot',
                        help='Output directory for results')
    parser.add_argument('--use_projection', action='store_true',
                        help='Use projection head for contrastive model')
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: ZERO-SHOT TRANSFER")
    print("="*70)
    print("Training: CASIA-B (healthy gaits)")
    print("Testing:  Pathology dataset (unseen)")
    print("="*70 + "\n")
    
    # Load pathology samples
    print("Loading pathology dataset...")
    samples = collect_pathology_samples(args.pathology_root)
    
    if len(samples['paths']) == 0:
        print("❌ No samples found. Check pathology_root path.")
        return
    
    image_paths = samples['paths']
    conditions = samples['conditions']
    
    # Load VAE model
    print(f"\nLoading VAE from {args.vae_checkpoint}...")
    vae = create_vae(latent_dim=128)
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    print(f"✅ VAE loaded (Epoch: {vae_ckpt.get('epoch', 'N/A')})")
    
    # Load Contrastive VAE model
    print(f"\nLoading Contrastive VAE from {args.contrastive_checkpoint}...")
    contrastive_vae = create_contrastive_vae(latent_dim=128, projection_dim=128)
    cont_ckpt = torch.load(args.contrastive_checkpoint, map_location=device)
    contrastive_vae.load_state_dict(cont_ckpt['model_state_dict'])
    contrastive_vae = contrastive_vae.to(device)
    contrastive_vae.eval()
    print(f"✅ Contrastive VAE loaded (Epoch: {cont_ckpt.get('epoch', 'N/A')})")
    
    # Extract embeddings
    print("\n" + "="*70)
    print("EXTRACTING EMBEDDINGS")
    print("="*70)
    
    print("\nExtracting VAE embeddings...")
    vae_embeddings = extract_embeddings(
        vae, image_paths, model_type='vae', device=device, batch_size=32
    )
    print(f"✅ VAE embeddings: {vae_embeddings.shape}")
    
    print("\nExtracting Contrastive VAE embeddings...")
    contrastive_embeddings = extract_embeddings(
        contrastive_vae, image_paths, model_type='contrastive', 
        device=device, batch_size=32, use_projection=args.use_projection
    )
    print(f"✅ Contrastive VAE embeddings: {contrastive_embeddings.shape}")
    
    # Evaluate: Binary Classification
    print("\n" + "="*70)
    print("BINARY CLASSIFICATION: Normal vs Pathological")
    print("="*70)
    
    vae_binary_results = binary_classification(
        vae_embeddings, conditions, model_name='VAE (Zero-Shot)'
    )
    
    contrastive_binary_results = binary_classification(
        contrastive_embeddings, conditions, model_name='Contrastive VAE (Zero-Shot)'
    )
    
    # Plot ROC curves
    plot_roc_curve(
        vae_binary_results['y_test'], 
        vae_binary_results['y_proba'],
        'VAE (Zero-Shot)',
        save_path=output_dir / 'vae_binary_roc.png'
    )
    
    plot_roc_curve(
        contrastive_binary_results['y_test'],
        contrastive_binary_results['y_proba'],
        'Contrastive VAE (Zero-Shot)',
        save_path=output_dir / 'contrastive_binary_roc.png'
    )
    
    # Evaluate: Multi-class Classification
    print("\n" + "="*70)
    print("MULTI-CLASS CLASSIFICATION: Specific Conditions")
    print("="*70)
    
    vae_multiclass_results = multiclass_classification(
        vae_embeddings, conditions, model_name='VAE (Zero-Shot)'
    )
    
    contrastive_multiclass_results = multiclass_classification(
        contrastive_embeddings, conditions, model_name='Contrastive VAE (Zero-Shot)'
    )
    
    # Plot confusion matrices
    plot_confusion_matrix(
        vae_multiclass_results['confusion_matrix'],
        vae_multiclass_results['class_names'],
        'VAE (Zero-Shot) - Confusion Matrix',
        save_path=output_dir / 'vae_multiclass_cm.png'
    )
    
    plot_confusion_matrix(
        contrastive_multiclass_results['confusion_matrix'],
        contrastive_multiclass_results['class_names'],
        'Contrastive VAE (Zero-Shot) - Confusion Matrix',
        save_path=output_dir / 'contrastive_multiclass_cm.png'
    )
    
    # Save all results
    all_results = [
        vae_binary_results,
        contrastive_binary_results,
        vae_multiclass_results,
        contrastive_multiclass_results
    ]
    
    save_results(vae_binary_results, output_dir / 'vae_binary.json')
    save_results(contrastive_binary_results, output_dir / 'contrastive_binary.json')
    save_results(vae_multiclass_results, output_dir / 'vae_multiclass.json')
    save_results(contrastive_multiclass_results, output_dir / 'contrastive_multiclass.json')
    
    # Compare models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    compare_models(all_results, save_path=output_dir / 'comparison.json')
    
    print(f"\n✅ Experiment 1 complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
