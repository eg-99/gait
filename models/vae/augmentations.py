"""
Data Augmentations for Contrastive Learning with GEI

Creates positive pairs through data augmentation for contrastive learning.
Supports pairing different modalities (angles, bag, coat) from the same subject.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import defaultdict


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


def create_modality_pairs(gei_batch, labels, metadata_list, 
                          pair_by_angle=True, pair_by_condition=True,
                          augmentation=None):
    """
    Create positive pairs by pairing different modalities from the same subject.
    
    Pairs samples from the same subject but with:
    - Different view angles (if pair_by_angle=True)
    - Different conditions (bag/coat/normal, if pair_by_condition=True)
    
    Args:
        gei_batch: Batch of GEI images (batch_size, 1, 128, 64)
        labels: Batch of labels (batch_size,) - subject IDs
        metadata_list: List of metadata dicts with 'subject_id', 'sequence_id', 'view_angle'
        pair_by_angle: If True, pair samples with different view angles
        pair_by_condition: If True, pair samples with different conditions (nm/bg/cl)
        augmentation: Optional GEIAugmentation instance to apply to pairs
    
    Returns:
        gei_paired: Paired GEI images (batch_size * 2, 1, 128, 64)
        labels_paired: Labels for paired samples (batch_size * 2,)
    """
    batch_size = gei_batch.size(0)
    device = gei_batch.device
    
    # Group samples by subject_id
    subject_groups = defaultdict(list)
    for i in range(batch_size):
        subject_id = metadata_list[i]['subject_id']
        subject_groups[subject_id].append(i)
    
    # Extract condition type from sequence_id (nm, bg, cl)
    def get_condition_type(sequence_id):
        """Extract condition type from sequence_id (e.g., 'nm-01' -> 'nm')."""
        if '-' in sequence_id:
            return sequence_id.split('-')[0]
        return sequence_id
    
    # Create pairs
    gei_pairs = []
    label_pairs = []
    
    for subject_id, indices in subject_groups.items():
        if len(indices) < 2:
            # If only one sample for this subject, duplicate it
            idx = indices[0]
            gei_pairs.append(gei_batch[idx])
            gei_pairs.append(gei_batch[idx])
            label_pairs.append(labels[idx])
            label_pairs.append(labels[idx])
            continue
        
        # Try to find pairs with different modalities
        paired_indices = set()
        
        for i, idx1 in enumerate(indices):
            if idx1 in paired_indices:
                continue
            
            metadata1 = metadata_list[idx1]
            view1 = metadata1['view_angle']
            cond1 = get_condition_type(metadata1['sequence_id'])
            
            # Find a matching pair
            best_pair_idx = None
            best_score = -1
            
            for idx2 in indices:
                if idx2 == idx1 or idx2 in paired_indices:
                    continue
                
                metadata2 = metadata_list[idx2]
                view2 = metadata2['view_angle']
                cond2 = get_condition_type(metadata2['sequence_id'])
                
                # Score based on modality differences
                score = 0
                if pair_by_angle and view1 != view2:
                    score += 1
                if pair_by_condition and cond1 != cond2:
                    score += 2  # Higher weight for condition difference
                
                if score > best_score:
                    best_score = score
                    best_pair_idx = idx2
            
            # Create pair
            if best_pair_idx is not None and best_score > 0:
                # Found a good modality pair
                gei_pairs.append(gei_batch[idx1])
                gei_pairs.append(gei_batch[best_pair_idx])
                label_pairs.append(labels[idx1])
                label_pairs.append(labels[best_pair_idx])
                paired_indices.add(idx1)
                paired_indices.add(best_pair_idx)
            else:
                # No good pair found, use augmentation or duplicate
                gei_pairs.append(gei_batch[idx1])
                if augmentation is not None:
                    gei_pairs.append(augmentation.apply_random_augmentation(gei_batch[idx1]))
                else:
                    gei_pairs.append(gei_batch[idx1])
                label_pairs.append(labels[idx1])
                label_pairs.append(labels[idx1])
                paired_indices.add(idx1)
        
        # Handle unpaired samples
        for idx in indices:
            if idx not in paired_indices:
                gei_pairs.append(gei_batch[idx])
                if augmentation is not None:
                    gei_pairs.append(augmentation.apply_random_augmentation(gei_batch[idx]))
                else:
                    gei_pairs.append(gei_batch[idx])
                label_pairs.append(labels[idx])
                label_pairs.append(labels[idx])
    
    # Stack into tensors
    gei_paired = torch.stack(gei_pairs, dim=0)
    # Handle labels - they might be tensors or scalars
    if isinstance(label_pairs[0], torch.Tensor):
        labels_paired = torch.stack(label_pairs, dim=0)
    else:
        labels_paired = torch.tensor(label_pairs, device=device, dtype=labels.dtype)
    
    return gei_paired, labels_paired


def create_contrastive_batch(gei_batch, labels, augmentation, 
                            metadata_list=None, 
                            use_modality_pairs=False,
                            pair_by_angle=True,
                            pair_by_condition=True):
    """
    Create augmented batch for contrastive learning.
    
    Can create positive pairs through:
    1. Data augmentation (default)
    2. Different modalities from same subject (if use_modality_pairs=True)
    3. Combination of both
    
    Args:
        gei_batch: Batch of GEI images (batch_size, 1, 128, 64)
        labels: Batch of labels (batch_size,)
        augmentation: GEIAugmentation instance
        metadata_list: Optional list of metadata dicts with 'subject_id', 'sequence_id', 'view_angle'
        use_modality_pairs: If True, pair different modalities instead of just augmenting
        pair_by_angle: If True, pair samples with different view angles (when use_modality_pairs=True)
        pair_by_condition: If True, pair samples with different conditions (when use_modality_pairs=True)
    
    Returns:
        gei_combined: Combined GEI images (batch_size * 2, 1, 128, 64)
        labels_combined: Labels repeated (batch_size * 2,)
    """
    batch_size = gei_batch.size(0)
    
    if use_modality_pairs and metadata_list is not None:
        # Use modality-based pairing
        gei_combined, labels_combined = create_modality_pairs(
            gei_batch, labels, metadata_list,
            pair_by_angle=pair_by_angle,
            pair_by_condition=pair_by_condition,
            augmentation=augmentation  # Still apply augmentation to pairs
        )
    else:
        # Original augmentation-based approach
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



