"""
GEI-CNN Difference Detector

Test the trained GEI-CNN model's ability to identify:
1. Same person from different sequences/views
2. Different people

Uses feature extraction + cosine similarity for verification.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# Add parent directory to path for imports
sys.path.append('..')
from gei_cnn.model import create_gei_cnn


def load_gei(gei_path):
    """Load a GEI .npy file and prepare for model input."""
    gei = np.load(gei_path)  # (128, 64)
    gei = torch.from_numpy(gei).float()
    gei = gei.unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, 128, 64), normalized
    return gei


def extract_features_for_all_geis(model, data_root, split='test', device='cpu'):
    """
    Extract features for all GEI files in a split.
    
    Returns:
        features_dict: {file_path: (feature_vector, subject_id)}
    """
    model.eval()
    data_root = Path(data_root)
    
    # Load splits
    splits_path = data_root / 'data_splits.json'
    with open(splits_path) as f:
        splits = json.load(f)
    
    subject_ids = splits[split]
    
    # Get sequence filter if available
    sequence_filter = None
    if 'metadata' in splits and splits['metadata'].get('split_type') == 'sequence_based':
        seq_key = f'{split}_sequences'
        sequence_filter = splits['metadata'].get(seq_key, None)
    
    features_dict = {}
    
    print(f"\nExtracting features for {split} split...")
    
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        subject_dir = data_root / subject_id
        if not subject_dir.exists():
            continue
        
        for gei_path in subject_dir.glob('*_gei.npy'):
            # Parse filename
            filename = gei_path.stem
            parts = filename.replace('_gei', '').split('_')
            
            if len(parts) >= 3:
                sequence_id = parts[1]
                
                # Filter by sequence if needed
                if sequence_filter is not None:
                    if sequence_id not in sequence_filter:
                        continue
                
                # Load and extract features
                gei = load_gei(gei_path).to(device)
                
                with torch.no_grad():
                    features = model.extract_features(gei)
                    features = features.cpu().squeeze(0).numpy()  # (256,)
                
                features_dict[str(gei_path)] = (features, subject_id)
    
    print(f"Extracted features for {len(features_dict)} GEI files")
    return features_dict


def create_verification_pairs(features_dict, num_same_pairs=1000, num_diff_pairs=1000):
    """
    Create pairs for verification task.
    
    Returns:
        pairs: List of (path1, path2, label) where label=1 (same), label=0 (different)
    """
    import random
    
    # Group by subject
    subject_groups = {}
    for path, (features, subject_id) in features_dict.items():
        if subject_id not in subject_groups:
            subject_groups[subject_id] = []
        subject_groups[subject_id].append(path)
    
    pairs = []
    
    # Create SAME person pairs
    print(f"\nCreating {num_same_pairs} same-person pairs...")
    same_count = 0
    subjects_with_multiple = [s for s, paths in subject_groups.items() if len(paths) >= 2]
    
    while same_count < num_same_pairs and subjects_with_multiple:
        subject = random.choice(subjects_with_multiple)
        paths = subject_groups[subject]
        
        if len(paths) >= 2:
            path1, path2 = random.sample(paths, 2)
            pairs.append((path1, path2, 1))  # label=1 for same person
            same_count += 1
    
    # Create DIFFERENT person pairs
    print(f"Creating {num_diff_pairs} different-person pairs...")
    diff_count = 0
    all_subjects = list(subject_groups.keys())
    
    while diff_count < num_diff_pairs and len(all_subjects) >= 2:
        subject1, subject2 = random.sample(all_subjects, 2)
        path1 = random.choice(subject_groups[subject1])
        path2 = random.choice(subject_groups[subject2])
        pairs.append((path1, path2, 0))  # label=0 for different people
        diff_count += 1
    
    print(f"Created {len(pairs)} total pairs ({same_count} same, {diff_count} different)")
    return pairs


def compute_similarities(model, pairs, features_dict, device='cpu'):
    """
    Compute cosine similarities for all pairs.
    
    Returns:
        similarities: List of similarity scores
        labels: List of ground truth labels (1=same, 0=different)
    """
    similarities = []
    labels = []
    
    print("\nComputing similarities...")
    
    for path1, path2, label in tqdm(pairs, desc="Computing"):
        features1, _ = features_dict[path1]
        features2, _ = features_dict[path2]
        
        # Convert to torch tensors
        features1 = torch.from_numpy(features1).unsqueeze(0).to(device)
        features2 = torch.from_numpy(features2).unsqueeze(0).to(device)
        
        # Cosine similarity
        similarity = F.cosine_similarity(features1, features2, dim=1).item()
        
        similarities.append(similarity)
        labels.append(label)
    
    return np.array(similarities), np.array(labels)


def find_optimal_threshold(similarities, labels):
    """Find optimal threshold using ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    
    # Find threshold that maximizes TPR - FPR
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute AUC
    roc_auc = auc(fpr, tpr)
    
    return optimal_threshold, roc_auc, fpr, tpr, thresholds


def evaluate_performance(similarities, labels, threshold):
    """Evaluate performance at given threshold."""
    predictions = (similarities >= threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_same = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_same = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_same = 2 * (precision_same * recall_same) / (precision_same + recall_same) if (precision_same + recall_same) > 0 else 0
    
    precision_diff = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_diff = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_diff = 2 * (precision_diff * recall_diff) / (precision_diff + recall_diff) if (precision_diff + recall_diff) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'same_person_detection': recall_same,  # TPR
        'different_person_detection': recall_diff,  # TNR
        'precision_same': precision_same,
        'precision_diff': precision_diff,
        'f1_same': f1_same,
        'f1_diff': f1_diff,
        'confusion_matrix': cm,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }


def plot_results(similarities, labels, threshold, roc_auc, fpr, tpr, save_dir='../logs/gei_cnn'):
    """Plot ROC curve and similarity distributions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: ROC Curve
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0].axvline(x=fpr[np.argmax(tpr - fpr)], color='red', linestyle=':', label=f'Optimal threshold = {threshold:.3f}')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve - Same/Different Person Detection')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Similarity Distributions
    same_sims = similarities[labels == 1]
    diff_sims = similarities[labels == 0]
    
    axes[1].hist(same_sims, bins=50, alpha=0.6, label='Same Person', color='green', density=True)
    axes[1].hist(diff_sims, bins=50, alpha=0.6, label='Different Person', color='red', density=True)
    axes[1].axvline(x=threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Similarity Score Distributions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'difference_detection_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plots saved to {save_dir / 'difference_detection_results.png'}")
    plt.close()


def main():
    """Main execution function."""
    print("="*80)
    print("GEI-CNN DIFFERENCE DETECTOR")
    print("="*80)
    
    # Configuration
    data_root = '../../preprocessing/preprocessed_data'
    checkpoint_path = '../checkpoints/gei_cnn/gei_cnn_best.pth'
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    print(f"Data root: {data_root}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load model
    print("\nLoading model...")
    model = create_gei_cnn(num_classes=124, dropout=0.4)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    print(f"  Best validation accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}" if 'val_acc' in checkpoint else "")
    
    # Extract features for all test GEIs
    features_dict = extract_features_for_all_geis(model, data_root, split='test', device=device)
    
    if len(features_dict) == 0:
        print("\n❌ No GEI files found! Check data_root path.")
        return
    
    # Create verification pairs
    num_pairs = min(2000, len(features_dict) // 2)  # Create up to 2000 pairs of each type
    pairs = create_verification_pairs(features_dict, num_same_pairs=num_pairs, num_diff_pairs=num_pairs)
    
    # Compute similarities
    similarities, labels = compute_similarities(model, pairs, features_dict, device)
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    optimal_threshold, roc_auc, fpr, tpr, thresholds = find_optimal_threshold(similarities, labels)
    
    print(f"\n✓ Optimal threshold: {optimal_threshold:.4f}")
    print(f"✓ AUC-ROC: {roc_auc:.4f}")
    
    # Evaluate performance
    print("\nEvaluating performance...")
    results = evaluate_performance(similarities, labels, optimal_threshold)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nOverall Accuracy: {results['accuracy']:.2%}")
    print(f"AUC-ROC Score: {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    print(f"\n{'='*40}")
    print("SAME PERSON DETECTION")
    print(f"{'='*40}")
    print(f"Detection Rate (Recall):  {results['same_person_detection']:.2%}")
    print(f"Precision:                {results['precision_same']:.2%}")
    print(f"F1-Score:                 {results['f1_same']:.4f}")
    
    print(f"\n{'='*40}")
    print("DIFFERENT PERSON DETECTION")
    print(f"{'='*40}")
    print(f"Detection Rate (Recall):  {results['different_person_detection']:.2%}")
    print(f"Precision:                {results['precision_diff']:.2%}")
    print(f"F1-Score:                 {results['f1_diff']:.4f}")
    
    print(f"\n{'='*40}")
    print("CONFUSION MATRIX")
    print(f"{'='*40}")
    print(f"                    Predicted")
    print(f"                    Different  Same")
    print(f"Actual  Different   {results['true_negatives']:<10} {results['false_positives']:<10}")
    print(f"        Same        {results['false_negatives']:<10} {results['true_positives']:<10}")
    
    # Similarity statistics
    same_sims = similarities[labels == 1]
    diff_sims = similarities[labels == 0]
    
    print(f"\n{'='*40}")
    print("SIMILARITY STATISTICS")
    print(f"{'='*40}")
    print(f"Same Person:")
    print(f"  Mean: {same_sims.mean():.4f}")
    print(f"  Std:  {same_sims.std():.4f}")
    print(f"  Min:  {same_sims.min():.4f}")
    print(f"  Max:  {same_sims.max():.4f}")
    
    print(f"\nDifferent Person:")
    print(f"  Mean: {diff_sims.mean():.4f}")
    print(f"  Std:  {diff_sims.std():.4f}")
    print(f"  Min:  {diff_sims.min():.4f}")
    print(f"  Max:  {diff_sims.max():.4f}")
    
    print(f"\nSeparation: {same_sims.mean() - diff_sims.mean():.4f}")
    
    # Plot results
    plot_results(similarities, labels, optimal_threshold, roc_auc, fpr, tpr)
    
    # Save results to JSON
    save_dir = Path('../logs/gei_cnn')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_json = {
        'overall_accuracy': float(results['accuracy']),
        'auc_roc': float(roc_auc),
        'optimal_threshold': float(optimal_threshold),
        'same_person_detection': float(results['same_person_detection']),
        'different_person_detection': float(results['different_person_detection']),
        'precision_same': float(results['precision_same']),
        'precision_diff': float(results['precision_diff']),
        'f1_same': float(results['f1_same']),
        'f1_diff': float(results['f1_diff']),
        'num_test_pairs': len(pairs),
        'same_similarity_mean': float(same_sims.mean()),
        'same_similarity_std': float(same_sims.std()),
        'diff_similarity_mean': float(diff_sims.mean()),
        'diff_similarity_std': float(diff_sims.std())
    }
    
    with open(save_dir / 'difference_detection_metrics.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to {save_dir / 'difference_detection_metrics.json'}")
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == '__main__':
    main()
