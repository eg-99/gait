"""
Training Utilities for Gait Recognition Models

Provides checkpoint management, early stopping, and training helpers.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints during training.
    Saves best model and periodic checkpoints.
    """
    def __init__(self, checkpoint_dir: str, model_name: str, keep_last_n: int = 3):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            model_name: Name of the model (used in filename)
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.keep_last_n = keep_last_n
        self.best_metric = None
        self.checkpoints = []
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, metrics: Dict[str, float], 
                       is_best: bool = False, scheduler=None):
        """
        Save a model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch number
            metrics: Dictionary of metrics
            is_best: If True, save as best model
            scheduler: Optional learning rate scheduler
        """
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Remove old checkpoints if we have too many
        if len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        # Save as best model if applicable
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")
            
            # Save metrics separately for easy access
            metrics_path = self.checkpoint_dir / f"{self.model_name}_best_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    def load_checkpoint(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                       checkpoint_path: Optional[str] = None, load_best: bool = True,
                       scheduler=None) -> Dict:
        """
        Load a model checkpoint.
        
        Args:
            model: PyTorch model to load weights into
            optimizer: Optional optimizer to load state into
            checkpoint_path: Specific checkpoint to load (if None, loads best or latest)
            load_best: If True and checkpoint_path is None, load best model
            scheduler: Optional scheduler to load state into
        
        Returns:
            Dictionary with checkpoint information
        """
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            else:
                # Load most recent checkpoint
                if self.checkpoints:
                    checkpoint_path = self.checkpoints[-1]
                else:
                    raise FileNotFoundError("No checkpoints found")
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}, Metrics: {checkpoint['metrics']}")
        
        return checkpoint


class EarlyStopping:
    """
    Early stopping handler to stop training when validation metric stops improving.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'max', verbose: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics where higher is better, 'min' for lower is better
            verbose: If True, print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric: Current metric value
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self.mode == 'max':
            improved = metric > (self.best_score + self.min_delta)
        else:
            improved = metric < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = metric
            self.counter = 0
            if self.verbose:
                print(f"✓ Metric improved to {metric:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement")
                self.early_stop = True
                return True
        
        return False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device for training.
    
    Args:
        prefer_gpu: If True, use GPU if available
    
    Returns:
        torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif prefer_gpu and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_training_config(config: Dict, save_path: str):
    """
    Save training configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training configuration saved to {save_path}")


def load_training_config(config_path: str) -> Dict:
    """
    Load training configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Training configuration loaded from {config_path}")
    return config
