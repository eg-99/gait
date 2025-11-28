"""
Data Augmentations for Contrastive Learning with GEI

Creates positive pairs through data augmentation for contrastive learning.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class GEIAugmentation:
    """
    Augmentation strategies for GEI images to create positive pairs.
    """
    
    def __init__(self, 
                 noise_std=0.05,
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 rotation_range=(-5, 5),
                 translation_range=(-5, 5),
                 flip_prob=0.5):
        """
        Initialize augmentation parameters.
        
        Args:
            noise_std: Standard deviation of Gaussian noise
            brightness_range: Range for brightness adjustment (min, max)
            contrast_range: Range for contrast adjustment (min, max)
            rotation_range: Range for rotation in degrees (min, max)
            translation_range: Range for translation in pixels (min, max)
            flip_prob: Probability of horizontal flip
        """
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.flip_prob = flip_prob
    
    def add_noise(self, x):
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.noise_std
        return torch.clamp(x + noise, 0, 1)
    
    def adjust_brightness(self, x):
        """Adjust brightness."""
        factor = np.random.uniform(*self.brightness_range)
        return torch.clamp(x * factor, 0, 1)
    
    def adjust_contrast(self, x):
        """Adjust contrast."""
        factor = np.random.uniform(*self.contrast_range)
        mean = x.mean()
        return torch.clamp((x - mean) * factor + mean, 0, 1)
    
    def rotate(self, x, angle):
        """Rotate image by angle degrees."""
        # Convert to radians
        angle_rad = angle * np.pi / 180.0
        
        # Create rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Get center
        h, w = x.shape[-2:]
        center = (w / 2, h / 2)
        
        # Create affine transformation matrix
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)
        
        # Apply rotation
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        rotated = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
        
        return rotated
    
    def translate(self, x, tx, ty):
        """Translate image."""
        h, w = x.shape[-2:]
        
        # Create translation matrix
        theta = torch.tensor([
            [1, 0, tx / w],
            [0, 1, ty / h]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)
        
        # Apply translation
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        translated = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
        
        return translated
    
    def horizontal_flip(self, x):
        """Horizontally flip image."""
        return torch.flip(x, dims=[-1])
    
    def apply_random_augmentation(self, x):
        """
        Apply a random augmentation to create a positive pair.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64) or (1, 128, 64)
        
        Returns:
            Augmented GEI tensor
        """
        # Ensure batch dimension
        was_single = len(x.shape) == 3
        if was_single:
            x = x.unsqueeze(0)
        
        # Randomly select augmentation
        aug_type = np.random.choice([
            'noise',
            'brightness',
            'contrast',
            'rotation',
            'translation',
            'flip',
            'combined'
        ])
        
        if aug_type == 'noise':
            x_aug = self.add_noise(x)
        elif aug_type == 'brightness':
            x_aug = self.adjust_brightness(x)
        elif aug_type == 'contrast':
            x_aug = self.adjust_contrast(x)
        elif aug_type == 'rotation':
            angle = np.random.uniform(*self.rotation_range)
            x_aug = self.rotate(x, angle)
        elif aug_type == 'translation':
            tx = np.random.uniform(*self.translation_range)
            ty = np.random.uniform(*self.translation_range)
            x_aug = self.translate(x, tx, ty)
        elif aug_type == 'flip':
            if np.random.random() < self.flip_prob:
                x_aug = self.horizontal_flip(x)
            else:
                x_aug = x
        else:  # combined
            x_aug = x
            if np.random.random() < 0.5:
                x_aug = self.add_noise(x_aug)
            if np.random.random() < 0.5:
                x_aug = self.adjust_brightness(x_aug)
            if np.random.random() < 0.5:
                x_aug = self.adjust_contrast(x_aug)
        
        if was_single:
            x_aug = x_aug.squeeze(0)
        
        return x_aug
    
    def create_positive_pair(self, x):
        """
        Create a positive pair by applying two different augmentations.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64) or (1, 128, 64)
        
        Returns:
            x1: First augmented view
            x2: Second augmented view
        """
        x1 = self.apply_random_augmentation(x)
        x2 = self.apply_random_augmentation(x)
        return x1, x2


def create_contrastive_batch(gei_batch, labels, augmentation):
    """
    Create augmented batch for contrastive learning.
    
    For each sample, creates two augmented views.
    
    Args:
        gei_batch: Batch of GEI images (batch_size, 1, 128, 64)
        labels: Batch of labels (batch_size,)
        augmentation: GEIAugmentation instance
    
    Returns:
        gei_aug1: First augmented views (batch_size, 1, 128, 64)
        gei_aug2: Second augmented views (batch_size, 1, 128, 64)
        labels_aug: Labels repeated (batch_size * 2,)
    """
    batch_size = gei_batch.size(0)
    device = gei_batch.device
    
    # Create two augmented views for each sample
    gei_aug1 = torch.zeros_like(gei_batch)
    gei_aug2 = torch.zeros_like(gei_batch)
    
    for i in range(batch_size):
        gei_aug1[i] = augmentation.apply_random_augmentation(gei_batch[i])
        gei_aug2[i] = augmentation.apply_random_augmentation(gei_batch[i])
    
    # Concatenate both views
    gei_combined = torch.cat([gei_aug1, gei_aug2], dim=0)
    labels_combined = torch.cat([labels, labels], dim=0)
    
    return gei_combined, labels_combined


if __name__ == "__main__":
    # Test augmentations
    aug = GEIAugmentation()
    
    # Create dummy GEI
    dummy_gei = torch.rand(1, 128, 64)
    
    # Test augmentations
    print("Testing GEI augmentations...")
    
    aug_noise = aug.add_noise(dummy_gei)
    aug_bright = aug.adjust_brightness(dummy_gei)
    aug_contrast = aug.adjust_contrast(dummy_gei)
    aug_rot = aug.rotate(dummy_gei, 5)
    aug_trans = aug.translate(dummy_gei, 3, 3)
    aug_flip = aug.horizontal_flip(dummy_gei)
    
    print(f"Original shape: {dummy_gei.shape}")
    print(f"Augmented shapes: {aug_noise.shape}, {aug_bright.shape}, {aug_contrast.shape}")
    
    # Test positive pair creation
    x1, x2 = aug.create_positive_pair(dummy_gei)
    print(f"Positive pair shapes: {x1.shape}, {x2.shape}")
    
    # Test batch creation
    batch_gei = torch.rand(4, 1, 128, 64)
    batch_labels = torch.tensor([0, 1, 0, 1])
    
    gei_combined, labels_combined = create_contrastive_batch(batch_gei, batch_labels, aug)
    print(f"Batch GEI shape: {batch_gei.shape}")
    print(f"Combined GEI shape: {gei_combined.shape}")
    print(f"Combined labels shape: {labels_combined.shape}")



