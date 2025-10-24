"""
Metrics and Evaluation Utilities for Gait Recognition

Provides functions for calculating accuracy, confusion matrices, and other metrics.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Model predictions (logits or probabilities), shape (N, num_classes)
        labels: True labels, shape (N,)
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    pred_labels = torch.argmax(predictions, dim=1)
    correct = (pred_labels == labels).sum().item()
    total = labels.size(0)
    return correct / total


def calculate_top_k_accuracy(predictions: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        predictions: Model predictions (logits or probabilities), shape (N, num_classes)
        labels: True labels, shape (N,)
        k: Top k predictions to consider
    
    Returns:
        Top-k accuracy as a float between 0 and 1
    """
    _, top_k_preds = predictions.topk(k, dim=1)
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds)).any(dim=1).sum().item()
    total = labels.size(0)
    return correct / total


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Update statistics with new value.
        
        Args:
            val: New value to add
            n: Number of items (for averaging)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """
    Tracks training and validation metrics across epochs.
    """
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def update(self, epoch_metrics: Dict[str, float]):
        """
        Update history with metrics from an epoch.
        
        Args:
            epoch_metrics: Dictionary of metric name -> value
        """
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best_epoch(self, metric: str = 'val_acc', maximize: bool = True) -> int:
        """
        Get the epoch with the best performance for a metric.
        
        Args:
            metric: Metric to check
            maximize: If True, higher is better; if False, lower is better
        
        Returns:
            Epoch number (0-indexed)
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return 0
        
        values = self.history[metric]
        if maximize:
            return int(np.argmax(values))
        else:
            return int(np.argmin(values))
    
    def plot_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: If provided, save plot to this path
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        if len(self.history['train_loss']) > 0:
            axes[0].plot(self.history['train_loss'], label='Train Loss')
        if len(self.history['val_loss']) > 0:
            axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        if len(self.history['train_acc']) > 0:
            axes[1].plot(self.history['train_acc'], label='Train Accuracy')
        if len(self.history['val_acc']) > 0:
            axes[1].plot(self.history['val_acc'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         class_names: List[str] = None, 
                         save_path: str = None,
                         normalize: bool = True):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: If provided, save plot to this path
        normalize: If True, normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_model(model: torch.nn.Module, 
                   data_loader: torch.utils.data.DataLoader,
                   device: torch.device,
                   print_report: bool = True) -> Dict[str, float]:
    """
    Comprehensive evaluation of a model.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        print_report: If True, print classification report
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels, _ in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    top5_acc = calculate_top_k_accuracy(
        torch.from_numpy(all_probs), 
        torch.from_numpy(all_labels), 
        k=5
    )
    
    metrics = {
        'accuracy': accuracy,
        'top5_accuracy': top5_acc,
        'num_samples': len(all_labels)
    }
    
    if print_report:
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Top-5 Accuracy: {top5_acc:.4f}")
        print(f"Number of samples: {len(all_labels)}")
        print("\nPer-class metrics:")
        print(classification_report(all_labels, all_preds, zero_division=0))
    
    return metrics, all_labels, all_preds
