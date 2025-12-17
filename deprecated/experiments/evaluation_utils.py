"""
Evaluation utilities for pathology detection experiments.
Provides binary and multi-class classification evaluation.
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import json


def load_and_preprocess_image(image_path, target_size=(128, 64)):
    """
    Load GEI image and convert to tensor (1, 1, H, W) in [0,1].
    
    Args:
        image_path: Path to image file
        target_size: (height, width) tuple
    
    Returns:
        torch.Tensor: (1, 1, H, W)
    """
    img = Image.open(image_path).convert('L')  # Grayscale
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def extract_embeddings(model, image_paths, model_type='vae', device='cpu', 
                       batch_size=32, use_projection=False):
    """
    Extract embeddings from images using trained model.
    
    Args:
        model: Trained VAE or Contrastive VAE model
        image_paths: List of image paths
        model_type: 'vae' or 'contrastive'
        device: torch device
        batch_size: Batch size for processing
        use_projection: For contrastive model, use projection head
    
    Returns:
        np.ndarray: (N, embedding_dim)
    """
    embeddings_list = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for img_path in batch_paths:
                tensor = load_and_preprocess_image(img_path)
                batch_tensors.append(tensor)
            
            batch = torch.cat(batch_tensors, dim=0).to(device)
            
            # Extract embeddings based on model type
            if model_type == 'vae':
                mu, _ = model.encode(batch)
                embedding = mu
            elif model_type == 'contrastive':
                if use_projection:
                    projection, _, _ = model.get_projection(batch)
                    embedding = projection
                else:
                    embedding = model.get_embedding(batch)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            embeddings_list.append(embedding.cpu().numpy())
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    return embeddings


def collect_pathology_samples(pathology_root, conditions=None):
    """
    Collect pathology samples from organized directory structure.
    
    Expected structure:
        pathology_root/
            normal/
                subject_001/*.png
                subject_002/*.png
            parkinson/
                subject_001/*.png
            neuropathy/
                subject_001/*.png
    
    Args:
        pathology_root: Root directory containing condition folders
        conditions: List of condition names to include (None = all)
    
    Returns:
        dict: {'paths': [...], 'subjects': [...], 'conditions': [...]}
    """
    samples = {
        'paths': [],
        'subjects': [],
        'conditions': []
    }
    
    pathology_root = Path(pathology_root)
    if not pathology_root.exists():
        print(f"⚠️  Pathology root not found: {pathology_root}")
        return samples
    
    # Iterate through condition directories
    condition_dirs = sorted([d for d in pathology_root.iterdir() if d.is_dir()])
    
    for cond_dir in condition_dirs:
        condition_name = cond_dir.name.lower()
        
        # Filter conditions if specified
        if conditions and condition_name not in conditions:
            continue
        
        # Iterate through subject directories
        subject_dirs = sorted([d for d in cond_dir.iterdir() if d.is_dir()])
        
        # If no subject dirs, treat images directly in condition dir
        if not subject_dirs:
            subject_dirs = [cond_dir]
        
        for subj_dir in subject_dirs:
            subject_id = subj_dir.name if subj_dir != cond_dir else condition_name
            
            # Collect all image files
            for pattern in ['*.png', '*.jpg', '*.PNG', '*.JPG']:
                for img_path in subj_dir.glob(pattern):
                    samples['paths'].append(str(img_path))
                    samples['subjects'].append(subject_id)
                    samples['conditions'].append(condition_name)
    
    print(f"📸 Collected {len(samples['paths'])} samples")
    print(f"   Conditions: {sorted(set(samples['conditions']))}")
    print(f"   Subjects: {len(set(samples['subjects']))}")
    
    return samples


def binary_classification(embeddings, labels, model_name='VAE', random_state=42):
    """
    Binary classification: Normal vs Pathological.
    
    Args:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of condition labels
        model_name: Name for display
        random_state: Random seed
    
    Returns:
        dict: Results with metrics
    """
    # Convert to binary labels (0 = normal, 1 = pathological)
    binary_labels = np.array([0 if label == 'normal' else 1 for label in labels])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, binary_labels, test_size=0.2, random_state=random_state, 
        stratify=binary_labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    clf = LogisticRegression(max_iter=3000, random_state=random_state)
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]  # Probability of pathological
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    auc_roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'model_name': model_name,
        'task': 'binary',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm.tolist(),
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_proba': y_proba.tolist()
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{model_name} - Binary Classification (Normal vs Pathological)")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"               Normal  Pathological")
    print(f"Actual Normal     {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Pathol     {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    return results


def multiclass_classification(embeddings, labels, model_name='VAE', random_state=42):
    """
    Multi-class classification: Specific conditions.
    
    Args:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of condition labels (strings)
        model_name: Name for display
        random_state: Random seed
    
    Returns:
        dict: Results with metrics
    """
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    class_names = le.classes_
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y_encoded, test_size=0.2, random_state=random_state,
        stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    clf = LogisticRegression(max_iter=3000, multi_class='multinomial', random_state=random_state)
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)
    
    # Top-k accuracy
    def topk_accuracy(probs, y_true, k):
        topk_preds = np.argsort(probs, axis=1)[:, ::-1][:, :k]
        hits = np.any(topk_preds == y_true[:, None], axis=1)
        return hits.mean()
    
    top1_acc = accuracy_score(y_test, y_pred)
    top2_acc = topk_accuracy(y_proba, y_test, k=2)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    macro_f1 = np.mean(f1)
    
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'model_name': model_name,
        'task': 'multiclass',
        'num_classes': len(class_names),
        'class_names': class_names.tolist(),
        'accuracy': top1_acc,
        'top2_accuracy': top2_acc,
        'macro_f1': macro_f1,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist(),
        'confusion_matrix': cm.tolist(),
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist()
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{model_name} - Multi-class Classification ({len(class_names)} classes)")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-2 Accuracy: {top2_acc:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"{'-'*65}")
    for i, cls in enumerate(class_names):
        print(f"{cls:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10.0f}")
    
    return results


def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_test, y_proba, model_name, save_path=None):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_test: True labels
        y_proba: Predicted probabilities
        model_name: Model name for display
        save_path: Path to save figure (optional)
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def save_results(results, output_path):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary of results
        output_path: Path to save JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {output_path}")


def compare_models(results_list, save_path=None):
    """
    Compare multiple model results and generate comparison table.
    
    Args:
        results_list: List of result dictionaries
        save_path: Path to save comparison table (optional)
    """
    # Binary classification comparison
    print(f"\n{'='*80}")
    print("Binary Classification Comparison (Normal vs Pathological)")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10}")
    print(f"{'-'*80}")
    
    binary_results = [r for r in results_list if r.get('task') == 'binary']
    for result in binary_results:
        print(f"{result['model_name']:<30} {result['accuracy']:>10.4f} "
              f"{result['f1']:>10.4f} {result['auc_roc']:>10.4f}")
    
    # Multi-class comparison
    print(f"\n{'='*80}")
    print("Multi-class Classification Comparison")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Top-1 Acc':>12} {'Top-2 Acc':>12} {'Macro F1':>12}")
    print(f"{'-'*80}")
    
    multiclass_results = [r for r in results_list if r.get('task') == 'multiclass']
    for result in multiclass_results:
        print(f"{result['model_name']:<30} {result['accuracy']:>12.4f} "
              f"{result.get('top2_accuracy', 0):>12.4f} {result['macro_f1']:>12.4f}")
    
    if save_path:
        comparison = {
            'binary': binary_results,
            'multiclass': multiclass_results
        }
        save_results(comparison, save_path)
