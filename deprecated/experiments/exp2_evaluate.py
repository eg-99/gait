"""
Experiment 2 Evaluation: Evaluate fine-tuned models.
"""

import argparse
import sys
from pathlib import Path
import torch
import json

# Add model paths
sys.path.append(str(Path(__file__).parent.parent / 'models' / 'vae'))
sys.path.append(str(Path(__file__).parent))

from models.vae.model import create_vae
from models.vae.contrastive_model import create_contrastive_vae
from evaluation_utils import (
    extract_embeddings, binary_classification, multiclass_classification,
    plot_confusion_matrix, plot_roc_curve, save_results, compare_models
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Fine-Tuned Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['vae', 'contrastive'], required=True,
                        help='Model type')
    parser.add_argument('--pathology_root', type=str, required=True,
                        help='Root directory of pathology dataset')
    parser.add_argument('--split_file', type=str, required=True,
                        help='Path to data splits JSON file')
    parser.add_argument('--output_dir', type=str, default='results/exp2_finetune',
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
    output_dir = Path(args.output_dir) / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("EXPERIMENT 2 EVALUATION: FINE-TUNED MODEL")
    print("="*70)
    print(f"Model type: {args.model_type}")
    print("="*70 + "\n")
    
    # Load data splits
    print(f"Loading data splits from {args.split_file}...")
    with open(args.split_file, 'r') as f:
        splits = json.load(f)
    
    # Load pathology samples
    from evaluation_utils import collect_pathology_samples
    samples = collect_pathology_samples(args.pathology_root)
    
    # Get test set
    test_indices = splits['test']
    test_paths = [samples['paths'][i] for i in test_indices]
    test_conditions = [samples['conditions'][i] for i in test_indices]
    
    print(f"Test set: {len(test_paths)} samples")
    print(f"Conditions: {sorted(set(test_conditions))}")
    
    # Load model
    print(f"\nLoading fine-tuned model from {args.checkpoint}...")
    if args.model_type == 'vae':
        model = create_vae(latent_dim=128)
    else:
        model = create_contrastive_vae(latent_dim=128, projection_dim=128)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded (Epoch: {checkpoint.get('epoch', 'N/A')})")
    
    # Extract embeddings
    print("\nExtracting embeddings from test set...")
    embeddings = extract_embeddings(
        model, test_paths, 
        model_type=args.model_type, 
        device=device, 
        batch_size=32,
        use_projection=args.use_projection
    )
    print(f"✅ Embeddings: {embeddings.shape}")
    
    # Binary classification
    print("\n" + "="*70)
    print("BINARY CLASSIFICATION: Normal vs Pathological")
    print("="*70)
    
    binary_results = binary_classification(
        embeddings, test_conditions, 
        model_name=f'{args.model_type.upper()} (Fine-Tuned)'
    )
    
    plot_roc_curve(
        binary_results['y_test'],
        binary_results['y_proba'],
        f'{args.model_type.upper()} (Fine-Tuned)',
        save_path=output_dir / 'binary_roc.png'
    )
    
    save_results(binary_results, output_dir / 'binary_results.json')
    
    # Multi-class classification
    print("\n" + "="*70)
    print("MULTI-CLASS CLASSIFICATION: Specific Conditions")
    print("="*70)
    
    multiclass_results = multiclass_classification(
        embeddings, test_conditions,
        model_name=f'{args.model_type.upper()} (Fine-Tuned)'
    )
    
    plot_confusion_matrix(
        multiclass_results['confusion_matrix'],
        multiclass_results['class_names'],
        f'{args.model_type.upper()} (Fine-Tuned) - Confusion Matrix',
        save_path=output_dir / 'multiclass_cm.png'
    )
    
    save_results(multiclass_results, output_dir / 'multiclass_results.json')
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
