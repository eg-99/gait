"""
Training Script for GEI-based CNN Model

This script handles the complete training pipeline:
1. Load preprocessed GEI data
2. Initialize model, optimizer, and loss function
3. Train with validation monitoring
4. Save checkpoints and best model
5. Evaluate on test set

Usage:
    python train.py --data_root ../preprocessing/preprocessed_data --epochs 100 --batch_size 32

The script automatically uses GPU if available (CUDA or MPS for Apple Silicon).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import json

# Add parent directory to path to import from preprocessing
sys.path.append(str(Path(__file__).parent.parent.parent / 'preprocessing'))
sys.path.append(str(Path(__file__).parent.parent))

from data_loader import GaitDataset, GaitDataLoader
from model import create_gei_cnn
from utils import (
    MetricsTracker, AverageMeter, CheckpointManager, EarlyStopping,
    get_device, count_parameters, save_training_config, evaluate_model,
    plot_confusion_matrix
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Tuple of (average loss, average accuracy)
    """
    model.train()  # Set model to training mode
    
    # Metrics tracking
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (data, labels, metadata) in enumerate(pbar):
        # Move data to device
        data = data.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = correct / labels.size(0)
        
        # Update metrics
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(accuracy, labels.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })
    
    return loss_meter.avg, acc_meter.avg


def validate_epoch(model, val_loader, criterion, device, epoch):
    """
    Validate for one epoch.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
    
    Returns:
        Tuple of (average loss, average accuracy)
    """
    model.eval()  # Set model to evaluation mode
    
    # Metrics tracking
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():  # No gradients needed for validation
        for data, labels, metadata in pbar:
            # Move data to device
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            accuracy = correct / labels.size(0)
            
            # Update metrics
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(accuracy, labels.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })
    
    return loss_meter.avg, acc_meter.avg


def main(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    print("="*80)
    print("GEI-CNN Training for Gait Recognition")
    print("="*80)
    
    # Set device
    device = get_device(prefer_gpu=not args.cpu)
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    config = vars(args)
    config['device'] = str(device)
    save_training_config(config, log_dir / 'training_config.json')
    
    # Load data
    print("\nLoading data...")
    loaders = GaitDataLoader.create_loaders(
        data_root=args.data_root,
        data_type='gei',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    # Get number of classes from dataset
    num_classes = train_loader.dataset.get_num_classes()
    print(f"Number of classes (subjects): {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = create_gei_cnn(num_classes=num_classes, dropout=args.dropout)
    model = model.to(device)
    
    # Print model summary
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    else:
        scheduler = None
    
    # Initialize training utilities
    metrics_tracker = MetricsTracker()
    checkpoint_manager = CheckpointManager(checkpoint_dir, 'gei_cnn', keep_last_n=3)
    early_stopping = EarlyStopping(patience=args.patience, mode='max', verbose=True)
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Update metrics tracker
        metrics_tracker.update({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr
        })
        
        # Check if this is the best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            model, optimizer, epoch,
            {'train_loss': train_loss, 'train_acc': train_acc,
             'val_loss': val_loss, 'val_acc': val_acc},
            is_best=is_best,
            scheduler=scheduler
        )
        
        # Check early stopping
        if early_stopping(val_acc):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Plot training history
    print("\nGenerating training plots...")
    metrics_tracker.plot_history(log_dir / 'training_history.png')
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint_manager.load_checkpoint(model, load_best=True)
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)
    
    test_metrics, test_labels, test_preds = evaluate_model(
        model, test_loader, device, print_report=True
    )
    
    # Save test metrics
    with open(log_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=log_dir / 'confusion_matrix.png',
        normalize=True
    )
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test top-5 accuracy: {test_metrics['top5_accuracy']:.4f}")
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"Logs and plots saved to: {log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GEI-CNN for gait recognition')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, 
                       default='../preprocessing/preprocessed_data',
                       help='Root directory of preprocessed data')
    
    # Model parameters
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout probability (default: 0.5)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer to use (default: adam)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization (default: 1e-4)')
    
    # Learning rate scheduler
    parser.add_argument('--scheduler', type=str, default='step',
                       choices=['step', 'cosine', 'plateau', 'none'],
                       help='Learning rate scheduler (default: step)')
    parser.add_argument('--step_size', type=int, default=30,
                       help='Step size for StepLR scheduler (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for StepLR scheduler (default: 0.1)')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=15,
                       help='Patience for early stopping (default: 15)')
    
    # System parameters
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    
    # Output directories
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='../checkpoints/gei_cnn',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str,
                       default='../logs/gei_cnn',
                       help='Directory to save logs and plots')
    
    args = parser.parse_args()
    
    # Run training
    main(args)
