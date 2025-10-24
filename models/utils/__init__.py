"""
Utilities for Gait Recognition Model Training
"""

from .metrics import (
    calculate_accuracy,
    calculate_top_k_accuracy,
    AverageMeter,
    MetricsTracker,
    plot_confusion_matrix,
    evaluate_model
)

from .training import (
    CheckpointManager,
    EarlyStopping,
    get_device,
    count_parameters,
    save_training_config,
    load_training_config
)

__all__ = [
    'calculate_accuracy',
    'calculate_top_k_accuracy',
    'AverageMeter',
    'MetricsTracker',
    'plot_confusion_matrix',
    'evaluate_model',
    'CheckpointManager',
    'EarlyStopping',
    'get_device',
    'count_parameters',
    'save_training_config',
    'load_training_config'
]
